"""A character-level language model for the decoder's transition.

The route A v2 evaluation put the limit of the fused decoder in one place: the
transition. The trigram reads two characters back, the fill tower's emissions
are independent per position, and between them the beam keeps one good
hypothesis and seven near-copies of it, which on abbreviated input is not the
sentence the user meant. This module trains the model that replaces the
trigram: a recurrent language model over the same characters, conditioned on
the same context the context tower sees, with a state per beam that the Rust
decoder carries forward one character at a time.

Recurrent rather than attention-based on purpose. The decoder advances every
beam by one character per position and merges beams by their last characters;
a state that is a fixed vector per beam costs the same to carry whether the
sentence is three characters or thirty, and a step is one matrix product, which
is what an input method can afford per keystroke.

A training sequence is ``<bos> context <sep> text <eos>`` over the characters
of ``char_pinyin.tsv`` (the Rust lexicon's alphabet) with everything else
mapped to ``<unk>``; the loss is taken at every position after ``<bos>``, so
the context is training data as well as conditioning. At decode time the
decoder feeds the context and ``<sep>`` through the model once, and the state
after ``<sep>`` is the start of every beam.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import IterableDataset

from mlime.logging import log
from mlime.train.lexicon import read_char_readings
from mlime.train.loop import Distributed, MetricLog, agreed, cosine_with_warmup, seed_everything
from mlime.train.run import Slices
from mlime.train.samples import context_tail

#: The reserved ids, in order; every character of the lexicon follows them.
SPECIALS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<sep>", "<unk>")
PAD, BOS, EOS, SEP, UNK = range(len(SPECIALS))

#: Characters of context a sequence keeps, the end nearest the text. Matches the
#: context tower's window (64 tokens with its two sentinels).
DEFAULT_CONTEXT_CHARS = 62


@dataclass(frozen=True)
class CharVocab:
    """The model's alphabet: the specials, then the lexicon's characters sorted."""

    chars: tuple[str, ...]

    @classmethod
    def from_char_table(cls, char_table: Path) -> CharVocab:
        """The alphabet of ``char_pinyin.tsv``, in code point order."""
        return cls(chars=SPECIALS + tuple(sorted(read_char_readings(char_table))))

    def __post_init__(self) -> None:
        if self.chars[: len(SPECIALS)] != SPECIALS:
            raise ValueError("a vocabulary must begin with the reserved ids")

    def __len__(self) -> int:
        return len(self.chars)

    @property
    def index(self) -> dict[str, int]:
        """``{character: id}``; built on demand, cached by the caller that loops."""
        return {ch: i for i, ch in enumerate(self.chars)}

    def encode(self, text: str, index: dict[str, int]) -> list[int]:
        """Ids of *text*'s characters, ``<unk>`` for any outside the alphabet."""
        return [index.get(ch, UNK) for ch in text]


@dataclass(frozen=True)
class CharLmConfig:
    """The model's shape."""

    embedding: int = 384
    hidden: int = 1024
    layers: int = 2
    dropout: float = 0.1
    context_chars: int = DEFAULT_CONTEXT_CHARS

    def __post_init__(self) -> None:
        for name in ("embedding", "hidden", "layers", "context_chars"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class CharLm(nn.Module):
    """Embedding, LSTM stack, projection back to the embedding, tied output."""

    def __init__(self, vocab_size: int, config: CharLmConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_size, config.embedding, padding_idx=PAD)
        self.lstm = nn.LSTM(
            config.embedding,
            config.hidden,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )
        self.project = nn.Linear(config.hidden, config.embedding)
        self.dropout = nn.Dropout(config.dropout)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """Scores over the alphabet from LSTM outputs, through the tied embedding."""
        scores: torch.Tensor = self.project(self.dropout(features)) @ self.embed.weight.T
        return scores

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits for the token after each position of ``tokens`` (``[B, T, V]``)."""
        features, _ = self.lstm(self.dropout(self.embed(tokens)))
        return self.logits(features)

    def step(
        self, token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance by one token: log probabilities of the next one and the new state.

        ``token`` is ``[B]``; the states are ``[layers, B, hidden]``.
        """
        features, (hidden, cell) = self.lstm(self.embed(token).unsqueeze(1), (hidden, cell))
        return torch.log_softmax(self.logits(features.squeeze(1)).float(), dim=-1), hidden, cell

    def initial_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The all-zero state every sequence starts from."""
        shape = (self.config.layers, batch, self.config.hidden)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


class StepModule(nn.Module):
    """``CharLm.step`` as a module of its own, the graph the decoder runs."""

    def __init__(self, model: CharLm):
        super().__init__()
        self.model = model

    def forward(
        self, token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.step(token, hidden, cell)


def sequence(
    text: str, context: str | None, vocab_index: dict[str, int], context_chars: int
) -> list[int]:
    """``<bos> context <sep> text <eos>`` as ids."""
    ids = [BOS]
    if context:
        ids.extend(vocab_index.get(ch, UNK) for ch in context_tail(context, context_chars + 2))
    ids.append(SEP)
    ids.extend(vocab_index.get(ch, UNK) for ch in text)
    ids.append(EOS)
    return ids


def read_sequences(shard: Path, vocab_index: dict[str, int], context_chars: int) -> list[list[int]]:
    """Every row of a samples shard as a training sequence."""
    frame = pl.read_parquet(shard, columns=["text", "context"])
    return [
        sequence(text, context, vocab_index, context_chars)
        for text, context in zip(frame["text"], frame["context"], strict=True)
    ]


@dataclass(frozen=True)
class Batch:
    """Padded token ids and the targets the loss is taken against."""

    tokens: torch.Tensor
    targets: torch.Tensor

    @property
    def target_count(self) -> int:
        """Positions the loss is taken at."""
        return int((self.targets != PAD).sum())


def pad_batch(sequences: Sequence[Sequence[int]]) -> Batch:
    """Stack sequences right-padded with ``<pad>``; targets are the inputs shifted."""
    width = max(len(ids) for ids in sequences)
    tokens = torch.full((len(sequences), width), PAD, dtype=torch.long)
    for row, ids in enumerate(sequences):
        tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return Batch(tokens=tokens[:, :-1], targets=tokens[:, 1:])


class SequenceBatches(IterableDataset[Batch]):
    """Token-budgeted batches over the rank's share of the shards, epoch after epoch.

    Shards are dealt round-robin to ranks and shuffled each epoch; rows are
    shuffled within a shard, then batched in length-sorted buckets so a batch's
    padding is small. The seed makes rank *r*'s epoch *e* the same sequence of
    batches on every machine, which is what a resumed run needs.
    """

    def __init__(
        self,
        shards: Sequence[Path],
        vocab: CharVocab,
        context_chars: int,
        max_tokens: int,
        world: Distributed,
        seed: int,
        bucket_rows: int = 8192,
    ):
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.shards = tuple(shards)
        self.vocab_index = vocab.index
        self.context_chars = context_chars
        self.max_tokens = max_tokens
        self.world = world
        self.seed = seed
        self.bucket_rows = bucket_rows
        self.epoch = 0

    def __iter__(self) -> Iterator[Batch]:
        while True:
            rng = random.Random(self.seed * 1_000_003 + self.epoch)
            order = list(self.shards)
            rng.shuffle(order)
            mine = order[self.world.rank :: self.world.world_size]
            for shard in mine:
                rows = read_sequences(shard, self.vocab_index, self.context_chars)
                rng.shuffle(rows)
                for start in range(0, len(rows), self.bucket_rows):
                    bucket = sorted(rows[start : start + self.bucket_rows], key=len)
                    batches = list(self._batches(bucket))
                    rng.shuffle(batches)
                    yield from batches
            self.epoch += 1

    def _batches(self, bucket: Sequence[list[int]]) -> Iterator[Batch]:
        batch: list[list[int]] = []
        for ids in bucket:
            if batch and (len(batch) + 1) * max(len(batch[-1]), len(ids)) > self.max_tokens:
                yield pad_batch(batch)
                batch = []
            batch.append(ids)
        if batch:
            yield pad_batch(batch)


@dataclass(frozen=True)
class CharLmTraining:
    """Every number the loop needs."""

    max_steps: int
    lr: float = 1e-3
    warmup_fraction: float = 0.02
    weight_decay: float = 0.01
    max_tokens: int = 16384
    gradient_clip: float = 1.0
    seed: int = 0
    log_every: int = 50
    checkpoint_every: int = 2000
    held_out_every: int = 2000
    fp16: bool = True
    #: Stop after this many seconds and write the final checkpoint from wherever
    #: the run is, for a session that would otherwise be killed with nothing
    #: kept; ``None`` runs every step.
    wall_budget_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}")
        if self.wall_budget_seconds is not None and self.wall_budget_seconds <= 0:
            raise ValueError(
                f"wall_budget_seconds must be positive, got {self.wall_budget_seconds}"
            )

    @property
    def warmup_steps(self) -> int:
        """Steps spent ramping the learning rate up from zero."""
        return max(1, int(self.max_steps * self.warmup_fraction))


def loss_of(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the non-padding targets."""
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=PAD
    )


@torch.no_grad()
def held_out_loss(model: CharLm, batches: Sequence[Batch], device: torch.device) -> float:
    """Token-weighted cross-entropy over the held-out batches, in nats per character."""
    model.eval()
    total, count = 0.0, 0
    for batch in batches:
        tokens, targets = batch.tokens.to(device), batch.targets.to(device)
        total += float(loss_of(model(tokens), targets)) * batch.target_count
        count += batch.target_count
    model.train()
    return total / count


def held_out_batches(
    shards: Sequence[Path], vocab: CharVocab, context_chars: int, max_tokens: int, limit: int
) -> list[Batch]:
    """The first *limit* rows of the held-out shards, batched like training data."""
    rows: list[list[int]] = []
    index = vocab.index
    for shard in shards:
        rows.extend(read_sequences(shard, index, context_chars))
        if len(rows) >= limit:
            break
    rows = sorted(rows[:limit], key=len)
    batches: list[Batch] = []
    start = 0
    while start < len(rows):
        end = start
        while end < len(rows) and (end - start + 1) * len(rows[end]) <= max_tokens:
            end += 1
        batches.append(pad_batch(rows[start : max(end, start + 1)]))
        start = max(end, start + 1)
    return batches


def save_checkpoint(
    path: Path, model: CharLm, vocab: CharVocab, step: int, optimizer: torch.optim.Optimizer
) -> None:
    """Weights, shape, alphabet, optimiser and step, in one file."""
    torch.save(
        {
            "step": step,
            "config": asdict(model.config),
            "vocab": list(vocab.chars),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> tuple[CharLm, CharVocab, int]:
    """A model, its alphabet and the step it was saved at, from a checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    vocab = CharVocab(chars=tuple(state["vocab"]))
    model = CharLm(len(vocab), CharLmConfig(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    return model, vocab, int(state["step"])


def train(
    samples: Path,
    slices: Slices,
    char_table: Path,
    out: Path,
    config: CharLmConfig,
    training: CharLmTraining,
    world: Distributed,
) -> Path:
    """Train from scratch and return the path of the final checkpoint."""
    world.start()
    device = world.device
    seed_everything(training.seed, world.rank)
    vocab = CharVocab.from_char_table(char_table)
    train_shards = [samples / name for name in slices.train]
    held_shards = [samples / name for name in slices.held_out]
    log.info(
        "char-lm data",
        train_shards=len(train_shards),
        held_out_shards=len(held_shards),
        alphabet=len(vocab),
        rank=world.rank,
    )
    model = CharLm(len(vocab), config).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    log.info("char-lm model", parameters=parameters, **asdict(config))
    wrapped: nn.Module = model
    if world.world_size > 1:
        wrapped = DistributedDataParallel(model, device_ids=[world.local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.lr, weight_decay=training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, training.warmup_steps, training.max_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=training.fp16 and device.type == "cuda")
    batches = iter(
        SequenceBatches(
            train_shards, vocab, config.context_chars, training.max_tokens, world, training.seed
        )
    )
    held = (
        held_out_batches(
            held_shards,
            vocab,
            config.context_chars,
            training.max_tokens,
            slices.max_held_out_examples,
        )
        if held_shards and world.is_main
        else []
    )
    out.mkdir(parents=True, exist_ok=True)
    metrics = MetricLog(out / "metrics.jsonl") if world.is_main else None
    started = time.time()
    tokens_seen = 0
    last_step = training.max_steps
    wrapped.train()
    for step in range(1, training.max_steps + 1):
        budget = training.wall_budget_seconds
        if budget is not None and agreed(world, time.time() - started > budget, device):
            last_step = step - 1
            if metrics is not None:
                metrics.write(event="stopped", step=last_step, elapsed=time.time() - started)
            log.info("wall budget spent; stopping", step=last_step)
            break
        batch = next(batches)
        tokens, targets = batch.tokens.to(device), batch.targets.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = wrapped(tokens)
        loss = loss_of(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(scaler.scale(loss))
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        tokens_seen += batch.target_count
        if metrics is not None and step % training.log_every == 0:
            elapsed = time.time() - started
            metrics.write(
                event="step",
                step=step,
                loss=float(loss),
                lr=scheduler.get_last_lr()[0],
                tokens=tokens_seen,
                tokens_per_second=tokens_seen / elapsed,
                elapsed=elapsed,
            )
        if metrics is not None and held and step % training.held_out_every == 0:
            nats = held_out_loss(model, held, device)
            metrics.write(
                event="held_out", step=step, nats_per_char=nats, perplexity=math.exp(nats)
            )
            log.info("held out", step=step, nats_per_char=round(nats, 4))
        if world.is_main and step % training.checkpoint_every == 0:
            save_checkpoint(out / "charlm.pt", model, vocab, step, optimizer)
    final = out / "charlm-final.pt"
    if world.is_main:
        save_checkpoint(final, model, vocab, last_step, optimizer)
        if held:
            nats = held_out_loss(model, held, device)
            if metrics is not None:
                metrics.write(
                    event="summary",
                    step=last_step,
                    nats_per_char=nats,
                    perplexity=math.exp(nats),
                    parameters=parameters,
                    tokens=tokens_seen,
                    elapsed=time.time() - started,
                )
        if metrics is not None:
            metrics.close()
    world.stop()
    return final


def export_onnx(checkpoint: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write the step graph and its manifest for the Rust decoder.

    ``charlm.onnx`` takes ``token [B]``, ``hidden [layers, B, hidden]`` and
    ``cell [layers, B, hidden]`` and returns ``log_probs [B, V]`` (float32,
    normalised over the alphabet) and the two new states. ``charlm.json`` holds
    the alphabet in id order and the state shape.
    """
    device = torch.device("cpu")
    model, vocab, step = load_model(checkpoint, device)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = out_dir / "charlm.onnx"
    hidden, cell = model.initial_state(2, device)
    torch.onnx.export(
        StepModule(model),
        (torch.tensor([BOS, SEP]), hidden, cell),
        str(graph),
        input_names=["token", "hidden", "cell"],
        output_names=["log_probs", "next_hidden", "next_cell"],
        dynamic_axes={
            "token": {0: "batch"},
            "hidden": {1: "batch"},
            "cell": {1: "batch"},
            "log_probs": {0: "batch"},
            "next_hidden": {1: "batch"},
            "next_cell": {1: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    manifest = out_dir / "charlm.json"
    manifest.write_text(
        json.dumps(
            {
                "step": step,
                "layers": model.config.layers,
                "hidden": model.config.hidden,
                "context_chars": model.config.context_chars,
                "specials": {"pad": PAD, "bos": BOS, "eos": EOS, "sep": SEP, "unk": UNK},
                "chars": list(vocab.chars),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("exported char-lm", graph=str(graph), manifest=str(manifest), alphabet=len(vocab))
    return graph, manifest
