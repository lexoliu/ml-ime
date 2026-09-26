"""Train and export a character-level language model for the decoder's transition.

The route A v2 evaluation put the limit of the fused decoder in one place: the
transition. The trigram reads two characters back, the fill tower's emissions
are independent per position, and between them the beam keeps one good
hypothesis and seven near-copies of it, which on abbreviated input is not the
sentence the user meant. The models here (:mod:`mlime.train.charlm_model`)
replace the trigram: a language model over the same characters, conditioned
on the same context the context tower sees, with a state per beam that the
Rust decoder carries forward one character at a time.

A training sequence is ``<bos> context <sep> text <eos>`` over the characters
of ``char_pinyin.tsv`` (the Rust lexicon's alphabet) with everything else
mapped to ``<unk>``; the loss is taken at every position after ``<bos>``, so
the context is training data as well as conditioning. At decode time the
decoder feeds the context and ``<sep>`` through the model's prefill graph
once, and what comes out is the start of every beam.

A run is resumable: the checkpoint holds the optimiser, the schedule, the loss
scaler and each rank's place in its stream of batches, so a session that ends
at a wall budget continues in the next one from the batch after the last.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnx.external_data_helper
import polars as pl
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import IterableDataset

from mlime.logging import log
from mlime.train.charlm_model import (
    DEFAULT_CONTEXT_CHARS,
    CharLm,
    CharLmConfig,
    PrefillModule,
    StepModule,
    build,
)
from mlime.train.charlm_vocab import BOS, EOS, PAD, SEP, UNK, CharVocab
from mlime.train.loop import Distributed, MetricLog, agreed, cosine_with_warmup, seed_everything
from mlime.train.run import Slices
from mlime.train.samples import context_tail

__all__ = [
    "BOS",
    "DEFAULT_CONTEXT_CHARS",
    "EOS",
    "PAD",
    "SEP",
    "UNK",
    "CharLmConfig",
    "CharLmTraining",
    "CharVocab",
    "export_onnx",
    "train",
]


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


def read_sequences(
    shard: Path, vocab_index: dict[str, int], context_chars: int, max_length: int
) -> list[list[int]]:
    """Every row of a samples shard as a training sequence, those over *max_length* dropped."""
    frame = pl.read_parquet(shard, columns=["text", "context"])
    sequences = (
        sequence(text, context, vocab_index, context_chars)
        for text, context in zip(frame["text"], frame["context"], strict=True)
    )
    return [ids for ids in sequences if len(ids) <= max_length]


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


@dataclass(frozen=True)
class Position:
    """A rank's place in its stream of batches: the next batch it will yield."""

    epoch: int = 0
    shard: int = 0
    batch: int = 0

    def __post_init__(self) -> None:
        if min(self.epoch, self.shard, self.batch) < 0:
            raise ValueError(f"a stream position is not negative, got {self}")


class SequenceBatches(IterableDataset[Batch]):
    """Token-budgeted batches over the rank's share of the shards, epoch after epoch.

    Shards are dealt round-robin to ranks and shuffled each epoch; rows are
    shuffled within a shard, then batched in length-sorted buckets so a batch's
    padding is small. The seed makes rank *r*'s epoch *e* the same sequence of
    batches on every machine, which is what a resumed run needs: ``skip_to`` a
    position and the stream continues from that batch.
    """

    def __init__(
        self,
        shards: Sequence[Path],
        vocab: CharVocab,
        context_chars: int,
        max_length: int,
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
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.world = world
        self.seed = seed
        self.bucket_rows = bucket_rows
        self.position = Position()

    def skip_to(self, position: Position) -> None:
        """Start the stream at *position* instead of the beginning."""
        self.position = position

    def __iter__(self) -> Iterator[Batch]:
        epoch, first_shard, first_batch = (
            self.position.epoch,
            self.position.shard,
            self.position.batch,
        )
        while True:
            rng = random.Random(self.seed * 1_000_003 + epoch)
            order = list(self.shards)
            rng.shuffle(order)
            mine = order[self.world.rank :: self.world.world_size]
            for shard_index, shard in enumerate(mine):
                # The shard's rows and batches are shuffled with the epoch's
                # generator whether or not they are yielded, so a skipped shard
                # leaves the generator where a yielded one would.
                rows = read_sequences(shard, self.vocab_index, self.context_chars, self.max_length)
                rng.shuffle(rows)
                batches: list[Batch] = []
                for start in range(0, len(rows), self.bucket_rows):
                    bucket = sorted(rows[start : start + self.bucket_rows], key=len)
                    chunk = list(self._batches(bucket))
                    rng.shuffle(chunk)
                    batches.extend(chunk)
                if shard_index < first_shard:
                    continue
                for batch_index, batch in enumerate(batches):
                    if shard_index == first_shard and batch_index < first_batch:
                        continue
                    self.position = Position(epoch, shard_index, batch_index + 1)
                    yield batch
                first_batch = 0
            epoch += 1
            first_shard = 0
            self.position = Position(epoch, 0, 0)

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
    shards: Sequence[Path],
    vocab: CharVocab,
    context_chars: int,
    max_length: int,
    max_tokens: int,
    limit: int,
) -> list[Batch]:
    """The first *limit* rows of the held-out shards, batched like training data."""
    rows: list[list[int]] = []
    index = vocab.index
    for shard in shards:
        rows.extend(read_sequences(shard, index, context_chars, max_length))
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


@dataclass(frozen=True)
class Progress:
    """Where a run is: the step, the tokens seen, and every rank's stream position."""

    step: int
    tokens_seen: int
    positions: tuple[Position, ...]


def save_checkpoint(
    path: Path,
    model: CharLm,
    vocab: CharVocab,
    progress: Progress,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    training: CharLmTraining,
) -> None:
    """Weights, shape, alphabet, optimiser, schedule, scaler and progress, in one file."""
    torch.save(
        {
            "step": progress.step,
            "tokens_seen": progress.tokens_seen,
            "positions": [asdict(position) for position in progress.positions],
            "config": asdict(model.config),
            "training": asdict(training),
            "vocab": list(vocab.chars),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> tuple[CharLm, CharVocab, int]:
    """A model, its alphabet and the step it was saved at, from a checkpoint."""
    state = torch.load(path, map_location=device, weights_only=False)
    vocab = CharVocab(chars=tuple(state["vocab"]))
    model = build(len(vocab), CharLmConfig(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    return model, vocab, int(state["step"])


@dataclass(frozen=True)
class Resumption:
    """A checkpoint read back and checked, ready to be poured into a fresh run."""

    progress: Progress
    model: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]

    @classmethod
    def read(
        cls, path: Path, config: CharLmConfig, training: CharLmTraining, world: Distributed
    ) -> Resumption:
        """Load *path* and refuse it unless it is this run, one session earlier."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        for name, expected in (("config", asdict(config)), ("training", asdict(training))):
            saved = dict(state[name])
            saved.pop("wall_budget_seconds", None)
            expected = dict(expected)
            expected.pop("wall_budget_seconds", None)
            if saved != expected:
                raise ValueError(
                    f"{path} was written for {name} {saved} and this run has {expected}; "
                    "a resumed run keeps every setting but the wall budget"
                )
        positions = state["positions"]
        if len(positions) != world.world_size:
            raise ValueError(
                f"{path} was written by a world of {len(positions)} ranks and this one has "
                f"{world.world_size}; a rank would read shards it has no position for"
            )
        step = int(state["step"])
        if step >= training.max_steps:
            raise ValueError(
                f"{path} is already at step {step} of {training.max_steps}: the run is "
                "complete, and this resume would do nothing"
            )
        return cls(
            progress=Progress(
                step=step,
                tokens_seen=int(state["tokens_seen"]),
                positions=tuple(Position(**position) for position in positions),
            ),
            model=state["model"],
            optimizer=state["optimizer"],
            scheduler=state["scheduler"],
            scaler=state["scaler"],
        )


def gather_positions(
    world: Distributed, position: Position, device: torch.device
) -> tuple[Position, ...]:
    """Every rank's stream position, in rank order, on every rank."""
    if world.world_size == 1:
        return (position,)
    mine = torch.tensor([position.epoch, position.shard, position.batch], device=device)
    gathered = [torch.zeros_like(mine) for _ in range(world.world_size)]
    torch.distributed.all_gather(gathered, mine)
    return tuple(Position(*(int(x) for x in row.tolist())) for row in gathered)


def train(
    samples: Path,
    slices: Slices,
    char_table: Path,
    out: Path,
    config: CharLmConfig,
    training: CharLmTraining,
    world: Distributed,
    resume: Path | None = None,
) -> Path:
    """Train, from scratch or from *resume*, and return the path of the final checkpoint."""
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
    model = build(len(vocab), config).to(device)
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
    stream = SequenceBatches(
        train_shards,
        vocab,
        config.context_chars,
        config.max_positions,
        training.max_tokens,
        world,
        training.seed,
    )
    first_step = 1
    tokens_seen = 0
    if resume is not None:
        resumed = Resumption.read(resume, config, training, world)
        model.load_state_dict(resumed.model)
        optimizer.load_state_dict(resumed.optimizer)
        scheduler.load_state_dict(resumed.scheduler)
        scaler.load_state_dict(resumed.scaler)
        stream.skip_to(resumed.progress.positions[world.rank])
        first_step = resumed.progress.step + 1
        tokens_seen = resumed.progress.tokens_seen
        log.info("resumed", path=str(resume), step=resumed.progress.step, rank=world.rank)
    batches = iter(stream)
    held = (
        held_out_batches(
            held_shards,
            vocab,
            config.context_chars,
            config.max_positions,
            training.max_tokens,
            slices.max_held_out_examples,
        )
        if held_shards and world.is_main
        else []
    )
    out.mkdir(parents=True, exist_ok=True)
    metrics = MetricLog(out / "metrics.jsonl") if world.is_main else None
    started = time.time()
    last_step = training.max_steps
    wrapped.train()

    def checkpoint(path: Path, step: int) -> None:
        progress = Progress(step, tokens_seen, gather_positions(world, stream.position, device))
        if world.is_main:
            save_checkpoint(path, model, vocab, progress, optimizer, scheduler, scaler, training)

    for step in range(first_step, training.max_steps + 1):
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
        if step % training.checkpoint_every == 0:
            checkpoint(out / "charlm.pt", step)
    final = out / "charlm-final.pt"
    checkpoint(final, last_step)
    if world.is_main:
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


def kept_ids(vocab: CharVocab, restrict: Path) -> torch.Tensor:
    """Alphabet ids of the characters listed one per line in *restrict*, plus ``<eos>``."""
    index = vocab.index
    chars = [line.strip() for line in restrict.read_text(encoding="utf-8").splitlines()]
    missing = [ch for ch in chars if ch and ch not in index]
    if missing:
        raise ValueError(
            f"{len(missing)} characters of {restrict} are not in the alphabet: {missing[:5]}"
        )
    ids = sorted({index[ch] for ch in chars if ch} | {EOS})
    return torch.tensor(ids, dtype=torch.long)


def batch_axes(names: Sequence[str], axis: int = 0) -> dict[str, dict[int, str]]:
    """The dynamic ``batch`` axis of every tensor in *names*."""
    return {name: {axis: "batch"} for name in names}


def _external_tensors(model: onnx.ModelProto) -> Iterator[onnx.TensorProto]:
    """Every initializer of *model*, nested subgraphs included, backed by external data."""
    graphs = [model.graph]
    while graphs:
        graph = graphs.pop()
        for tensor in graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                yield tensor
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    graphs.append(attribute.g)
                graphs.extend(attribute.graphs)


def _externalize(graph: Path, location: str) -> onnx.ModelProto:
    """Move every initializer of *graph* into the *location* file beside it.

    *location* is relative to the graph's directory. Returns the saved model
    reloaded without its external data (``load_external_data=False`` keeps the
    ``external_data`` entries; a plain ``onnx.load`` folds the bytes back into
    ``raw_data`` and strips them), so the callers read offsets and lengths of
    what was written, not of the in-memory proto.
    """
    (graph.parent / location).unlink(missing_ok=True)  # appends otherwise
    model = onnx.load(str(graph))
    onnx.external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.save_model(model, str(graph))
    return onnx.load(str(graph), load_external_data=False)


def _weights_table(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    """The manifest's tensor table of a saved externalized model, keyed by name."""
    table = {}
    for tensor in _external_tensors(model):
        entries = {entry.key: entry.value for entry in tensor.external_data}
        table[tensor.name] = {
            "name": tensor.name,
            "dtype": np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).name,
            "shape": list(tensor.dims),
            "offset": int(entries.get("offset", "0")),
            "length": int(entries["length"]),
        }
    return table


def _external_weights(step_graph: Path, prefill_graph: Path) -> dict[str, Any]:
    """Externalize both graphs' weights and return the manifest's ``weights`` value.

    The step and prefill graphs are built from the same parameters, so when
    their initializer names coincide both point into one ``charlm.weights``;
    a tensor of one name whose bytes differ fails the export. When the names
    differ each graph gets its own weights file and the manifest holds one
    table per graph under ``step``/``prefill``.
    """
    step_model = _externalize(step_graph, "charlm.weights")
    prefill_model = _externalize(prefill_graph, "prefill.weights")
    step_table = _weights_table(step_model)
    prefill_table = _weights_table(prefill_model)
    if set(prefill_table) != set(step_table):
        log.info("step and prefill tensors differ by name: one weights file each")
        return {
            "step": {"file": "charlm.weights", "tensors": list(step_table.values())},
            "prefill": {"file": "prefill.weights", "tensors": list(prefill_table.values())},
        }
    dir = step_graph.parent
    step_bytes = (dir / "charlm.weights").read_bytes()
    prefill_bytes = (dir / "prefill.weights").read_bytes()
    for name, entry in prefill_table.items():
        theirs = step_table[name]
        a = prefill_bytes[entry["offset"] : entry["offset"] + entry["length"]]
        b = step_bytes[theirs["offset"] : theirs["offset"] + theirs["length"]]
        if a != b:
            raise ValueError(f"tensor {name!r} differs between the step and prefill graphs")
    # Repoint every prefill tensor at the shared file's extent of the same
    # name. The model is serialized directly rather than through
    # ``onnx.save_model``, which would rewrite the weights file from the
    # (unloaded) ``raw_data`` fields.
    for tensor in _external_tensors(prefill_model):
        entry = step_table[tensor.name]
        for pair in tensor.external_data:
            if pair.key == "location":
                pair.value = "charlm.weights"
            elif pair.key == "offset":
                pair.value = str(entry["offset"])
            elif pair.key == "length":
                pair.value = str(entry["length"])
    prefill_graph.write_bytes(prefill_model.SerializeToString())
    (dir / "prefill.weights").unlink()
    log.info("step and prefill share charlm.weights", tensors=len(step_table))
    return {"file": "charlm.weights", "tensors": list(step_table.values())}


def export_onnx(checkpoint: Path, out_dir: Path, restrict: Path | None = None) -> tuple[Path, Path]:
    """Write the step graph, the prefill graph and their manifest for the Rust decoder.

    ``charlm.onnx`` takes ``token [B]``, the prefix tensors (one row each,
    shared by the batch) and the state tensors (one row per beam) and returns
    ``log_probs [B, V]`` (float32, normalised over the alphabet, or over the
    characters of *restrict* plus ``<eos>`` when given) and the next state
    tensors. ``prefill.onnx`` takes ``tokens [1, T]`` and returns the same
    ``log_probs``, the prefix tensors and the state tensors to start from.
    ``charlm.json`` names the tensors and holds the alphabet in id order.
    The graphs' initializers live in an external weights file beside the
    graphs -- ``charlm.weights``, shared when the step and prefill
    initializers coincide, otherwise one file per graph -- which the manifest's
    ``weights`` table maps name-by-name so the Rust decoder can share one
    mapping across its sessions.
    """
    device = torch.device("cpu")
    model, vocab, step = load_model(checkpoint, device)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = None if restrict is None else kept_ids(vocab, restrict)
    prefix_names, state_names = model.prefix_names, model.state_names
    next_names = [f"next_{name}" for name in state_names]
    prelude = torch.tensor([[BOS, SEP]])
    with torch.no_grad():
        _, prefix, state = model.prefill(prelude)
    # Example inputs with two rows in the batch and two characters in the
    # state, so every dynamic axis is exercised.
    with torch.no_grad():
        two = torch.tensor([SEP, SEP])
        stacked = tuple(tensor.expand(2, *tensor.shape[1:]).contiguous() for tensor in state)
        _, stacked = model.step(two, prefix, stacked)
        _, stacked = model.step(two, prefix, stacked)
    axes: dict[str, dict[int, str]] = {"token": {0: "batch"}, "log_probs": {0: "batch"}}
    for name in (*state_names, *next_names):
        axes[name] = {0: "batch"}
    if model.config.arch == "transformer":
        # Key/value caches are [batch, layers, heads, time, head_dim].
        for name in (*prefix_names, *state_names, *next_names):
            axes.setdefault(name, {})[3] = f"{name}_time"
    step_graph = out_dir / "charlm.onnx"
    torch.onnx.export(
        StepModule(model, keep),
        (two, *prefix, *stacked),
        str(step_graph),
        input_names=["token", *prefix_names, *state_names],
        output_names=["log_probs", *next_names],
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
    )
    prefill_graph = out_dir / "prefill.onnx"
    torch.onnx.export(
        PrefillModule(model, keep),
        (prelude,),
        str(prefill_graph),
        input_names=["tokens"],
        output_names=["log_probs", *prefix_names, *state_names],
        dynamic_axes={
            "tokens": {1: "length"},
            **{name: axes[name] for name in (*prefix_names, *state_names) if name in axes},
        },
        opset_version=17,
        dynamo=False,
    )
    weights = _external_weights(step_graph, prefill_graph)
    manifest = out_dir / "charlm.json"
    manifest.write_text(
        json.dumps(
            {
                "step": step,
                "arch": model.config.arch,
                "context_chars": model.config.context_chars,
                "prefix": list(prefix_names),
                "state": list(state_names),
                "specials": {"pad": PAD, "bos": BOS, "eos": EOS, "sep": SEP, "unk": UNK},
                "restricted_to": None if keep is None else int(keep.numel()),
                "weights": weights,
                "chars": list(vocab.chars),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info(
        "exported char-lm",
        step_graph=str(step_graph),
        prefill_graph=str(prefill_graph),
        manifest=str(manifest),
        alphabet=len(vocab),
        restricted_to=None if keep is None else int(keep.numel()),
    )
    return step_graph, manifest
