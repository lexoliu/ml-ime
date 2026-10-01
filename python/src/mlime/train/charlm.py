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
from typing import Any, Literal

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
    PagedPrefillModule,
    PagedStepModule,
    PrefillModule,
    ResidentStepModule,
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
    # The file is written back-to-back in initializer order, so a tensor's
    # offset stays a multiple of its element size when larger elements go
    # first -- the Rust side hands the runtime raw pointers into the mapped
    # file, which must be aligned for their dtype.
    initializers = sorted(
        model.graph.initializer,
        key=lambda tensor: -onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize,
    )
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)
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


def _quantize_weights_int8(graph: Path) -> None:
    """Rewrite *graph* in place keeping only the MatMul weights int8.

    Each weight folds into a per-output-channel int8 constant plus a
    ``DequantizeLinear`` that rebuilds it float32 ahead of the matmul, so
    activations stay float32 end to end and no row's value depends on its
    batchmates. Full int8 range applies: nothing here reaches a u8 x s8
    kernel, so the seven-bit guard of the dynamic path is unneeded.
    """
    model = onnx.load(str(graph))
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    added: list[onnx.TensorProto] = []
    dropped: set[str] = set()
    dequantized: dict[str, str] = {}  # weight name -> dequantized tensor name
    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        weight_input = -1
        if node.op_type == "MatMul":
            for index in (0, 1):
                if node.input[index] in initializers:
                    weight_input = index
                    break
        if weight_input >= 0:
            wname = node.input[weight_input]
            if wname not in dequantized:
                weight = onnx.numpy_helper.to_array(initializers[wname]).astype(np.float32)
                # Channels are the matmul's output features: the last axis of
                # a [K, N] weight, or the first of [N, K].
                axis = weight.ndim - 1 if weight_input == 1 else 0
                bound = np.abs(weight).max(axis=axis ^ 1)
                scale = (bound / 127.0).astype(np.float32)
                scale[scale == 0.0] = 1.0
                shape = [1] * weight.ndim
                shape[axis] = weight.shape[axis]
                quantized_weight = np.clip(
                    np.round(weight / scale.reshape(shape)), -127, 127
                ).astype(np.int8)
                added += [
                    onnx.numpy_helper.from_array(quantized_weight, f"{wname}_int8"),
                    onnx.numpy_helper.from_array(scale, f"{wname}_scale"),
                ]
                dequantized[wname] = f"{wname}_dequantized"
                dropped.add(wname)
                nodes.append(
                    onnx.helper.make_node(
                        "DequantizeLinear",
                        [f"{wname}_int8", f"{wname}_scale"],
                        [dequantized[wname]],
                        axis=axis,
                    )
                )
            inputs = list(node.input)
            inputs[weight_input] = dequantized[wname]
            nodes.append(onnx.helper.make_node("MatMul", inputs, list(node.output), name=node.name))
        else:
            nodes.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    kept = {name for node in model.graph.node for name in node.input if name in initializers}
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(
        tensor for name, tensor in initializers.items() if name not in dropped or name in kept
    )
    model.graph.initializer.extend(added)
    rewritten = graph.with_suffix(".rewritten.onnx")
    onnx.save_model(model, str(rewritten))
    rewritten.replace(graph)


def export_onnx(
    checkpoint: Path,
    out_dir: Path,
    restrict: Path | None = None,
    quantize: Literal["int8", "fp16"] | None = None,
) -> tuple[Path, Path]:
    """Write the step graph, the prefill graph and their manifest for the Rust decoder.

    The transformer's ``charlm.onnx`` runs the pages layout: ``token
    [rows]`` each beam's next character, ``page_row [rows, T]`` its
    prefix's page ids into the resident ``keys``/``values`` pool
    ``[pages, layers, heads, head_dim]`` (page zero a masked scratch),
    ``mask [rows, T]`` its real positions, and ``candidates [rows, K]``
    the alphabet ids its produced state will be scored on at its next
    position (``<eos>`` included where its path can end there, a
    throwaway id as padding). A state that could not claim pages reads
    the scratch rows ``state_keys``/``state_values`` through
    ``source_row`` (-1 for a paged row). The graph gathers each row's
    history by index, gathers the log-softmax (float32, normalised over
    the alphabet, or over the characters of *restrict* plus ``<eos>``
    when given) at the candidates each row named, and returns
    ``candidate_log_probs [rows, K]`` -- the only tensor that crosses
    the bus -- plus ``next_keys``/``next_values`` ``[rows, layers,
    heads, head_dim]``, the produced rows' own positions the host writes
    into their claimed pages in a single contiguous copy per tensor. A
    step moves indices and one page per produced state; no row of
    history ever crosses the host boundary.
    ``prefill.onnx`` takes ``tokens [1, T]`` and the first position's
    ``candidates [1, K]``, and returns the same ``candidate_log_probs``
    plus the prefix tensors one position per row ``[T, layers, heads,
    head_dim]``: the host writes each position into its own page, so a
    ``start`` claims the prelude's pages and moves no rows. The LSTM has
    no pages: its step graph is ``(token, source_row, candidates,
    hidden, cell)`` over resident rows and its prefill emits
    ``(candidate_log_probs, hidden, cell)``. The manifest's ``rows``
    record what one slot row of each prefix and state buffer is shaped
    as -- the transformer's state rows marking the pages axis with a
    zero -- which is what the resident pool allocates.
    ``charlm.json`` names the tensors and holds the alphabet in id order;
    its ``layout`` records the step graph's layout
    (``"resident-pages"``), which ``ime-lm`` refuses to open without,
    and its ``dtype`` the element type the resident buffers and the
    states exchange.
    The graphs' initializers live in an external weights file beside the
    graphs -- ``charlm.weights``, shared when the step and prefill
    initializers coincide, otherwise one file per graph -- which the manifest's
    ``weights`` table maps name-by-name so the Rust decoder can share one
    mapping across its sessions.

    With ``quantize="int8"`` every MatMul's weight folds into a
    per-channel int8 constant plus a ``DequantizeLinear`` -- activations
    stay float32, so a row's scores cannot move with its batchmates
    (issue #93). With ``quantize="fp16"`` the model is halved before
    export, so the weights, the prefix buffers and the state tensors are
    all ``float16`` while ``candidate_log_probs`` stays ``float32`` through
    the output cast.
    """
    choices = (None, "int8", "fp16")
    if quantize not in choices:
        raise ValueError(f"quantize must be one of {choices}, got {quantize!r}")
    device = torch.device("cpu")
    model, vocab, step = load_model(checkpoint, device)
    model.eval()
    if quantize == "fp16":
        model = model.half()
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = None if restrict is None else kept_ids(vocab, restrict)
    prefix_names, state_names = model.prefix_names, model.state_names
    next_names = [f"next_{name}" for name in state_names]
    prelude = torch.tensor([[BOS, SEP]])
    with torch.no_grad():
        _, prefix, state = model.prefill(prelude)
    # Example inputs of two workers at width two with two characters in the
    # state, so every dynamic axis -- slots, workers, rows, time -- is
    # exercised. Under the pages layout the resident buffers stand in for
    # what the Rust side binds instead: the page pool, the scratch rows a
    # page claim could not cover, and the index tensors naming each beam's
    # prefix pages and source.
    paged = model.config.arch == "transformer"
    if paged:
        module: nn.Module = PagedStepModule(model, keep)
    else:
        module = ResidentStepModule(model, keep)
    with torch.no_grad():
        four = torch.tensor([SEP] * 4)
        # The candidates the produced rows would be scored on; any in-range
        # ids trace the same gather, so a fixed row of specials stands in.
        candidates = torch.tensor([[BOS, SEP, EOS, UNK]] * 4)
        if paged:
            # Example: four beams sharing two context pages plus one page
            # each, one row-sourced state, and a padded page row, so every
            # dynamic axis -- rows, pages, scratch rows, time -- is
            # exercised. Page zero is the pool's masked scratch.
            layers, heads, head_dim = (
                model.config.layers,
                model.config.heads,
                model.config.hidden // model.config.heads,
            )
            pool = tuple(
                torch.zeros(8, layers, heads, head_dim, dtype=state[0].dtype)
                for _ in state_names
            )
            page_row = torch.tensor([[1, 2, 3], [1, 2, 4], [0, 0, 0], [1, 2, 0]])
            mask = torch.tensor(
                [[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=torch.bool
            )
            state_buffers = tuple(
                torch.zeros(2, layers, heads, 3, head_dim, dtype=tensor.dtype)
                for tensor in state
            )
            source_row = torch.tensor([-1, -1, 0, -1])
            step_inputs = (
                four,
                page_row,
                mask,
                candidates,
                *pool,
                *state_buffers,
                source_row,
            )
        else:
            source_row = torch.arange(4, dtype=torch.long)
            state_buffers = tuple(
                tensor.expand(4, *tensor.shape[1:]).contiguous() for tensor in state
            )
            prefix_buffers: tuple[torch.Tensor, ...] = ()
            index_args: tuple[torch.Tensor, ...] = ()
            # Two passes grow the example's state buffers to a real time axis.
            for _ in range(2):
                args = (
                    four,
                    source_row,
                    candidates,
                    *prefix_buffers,
                    *index_args,
                    *state_buffers,
                )
                _, *state_buffers = module(*args)
            step_inputs = (four, source_row, candidates, *state_buffers)
    axes: dict[str, dict[int, str]] = {
        "token": {0: "rows"},
        "source_row": {0: "rows"},
        "candidates": {0: "rows", 1: "candidates"},
        "candidate_log_probs": {0: "rows", 1: "candidates"},
    }
    input_names: list[str]
    if paged:
        # The resident pool is [pages, layers, heads, head_dim]: a page's
        # per-position slices contiguous, the page axis marked by its own
        # symbol. The scratch row buffers keep the old [residents, layers,
        # heads, time, head_dim] shape for states a page claim could not
        # cover.
        for name in state_names:
            axes[name] = {0: "pages"}
        for name in next_names:
            axes[name] = {0: "rows"}
        for name in ("state_keys", "state_values"):
            axes[name] = {0: "residents", 3: f"{name}_time"}
        axes["page_row"] = {0: "rows", 1: "pages_row"}
        axes["mask"] = {0: "rows", 1: "mask_time"}
        input_names = [
            "token",
            "page_row",
            "mask",
            "candidates",
            *state_names,
            "state_keys",
            "state_values",
            "source_row",
        ]
    else:
        for name in (*state_names, *next_names):
            axes[name] = {0: "residents"}
        input_names = ["token", "source_row", "candidates", *state_names]
    step_graph = out_dir / "charlm.onnx"
    torch.onnx.export(
        module,
        step_inputs,
        str(step_graph),
        input_names=input_names,
        output_names=["candidate_log_probs", *next_names],
        dynamic_axes=axes,
        opset_version=17,
        dynamo=False,
    )
    prefill_graph = out_dir / "prefill.onnx"
    if paged:
        prefill_module: nn.Module = PagedPrefillModule(model, keep)
        prefill_outputs = ["candidate_log_probs", *prefix_names]
        prefill_axes = {
            "tokens": {1: "length"},
            "candidates": {1: "candidates"},
            "candidate_log_probs": {1: "candidates"},
            **{name: {0: "prelude"} for name in prefix_names},
        }
    else:
        prefill_module = PrefillModule(model, keep)
        prefill_outputs = ["candidate_log_probs", *state_names]
        prefill_axes = {
            "tokens": {1: "length"},
            "candidates": {1: "candidates"},
            "candidate_log_probs": {1: "candidates"},
            **{name: {**axes[name], 0: "batch"} for name in state_names if name in axes},
        }
    torch.onnx.export(
        prefill_module,
        (prelude, torch.tensor([[BOS, SEP, EOS, UNK]])),
        str(prefill_graph),
        input_names=["tokens", "candidates"],
        output_names=prefill_outputs,
        dynamic_axes=prefill_axes,
        opset_version=17,
        dynamo=False,
    )
    if quantize == "int8":
        _quantize_weights_int8(step_graph)
        _quantize_weights_int8(prefill_graph)
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
                "layout": "resident-pages",
                "dtype": "float16" if quantize == "fp16" else "float32",
                "rows": {
                    # One slot row per buffer: a transformer's prefix page is
                    # one position's [layers, heads, head_dim]; its resident
                    # pool is [pages, layers, heads, head_dim] with the pages
                    # axis unbound. The LSTM keeps its [layers, hidden] row.
                    "prefix": [
                        [*list(tensor.shape[1:3]), tensor.shape[4]] for tensor in prefix
                    ],
                    "state": [
                        [0, tensor.shape[1], tensor.shape[2], tensor.shape[4]]
                        if paged
                        else list(tensor.shape[1:])
                        for tensor in state
                    ],
                },
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
