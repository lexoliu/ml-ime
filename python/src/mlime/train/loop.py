"""The route A training loop: one step shape, whether there is one GPU or two.

Nothing here decides anything about the model. It owns the four things a run has
to get right to be believable: the schedule the plan specified (AdamW, 3e-5 for
the pretrained weights and 1e-4 for the tables we added, cosine after a 4%
warmup), fp16 with loss scaling, a seed that makes the run repeatable, and a
metrics file that records what actually happened rather than what was intended.

Distribution is by shard, not by sampler: :class:`~mlime.train.samples.CorpusStream`
already deals shards out per rank and per dataloader worker, so a rank's job here
is only to know which rank it is. A single-process run is the same code with a
world of one -- there is no separate path to rot.

A run is also longer than the kernel it runs in: Kaggle stops a session at twelve
hours and the plan's budget is several times that, so a checkpoint has to carry
everything the next kernel needs to continue the *same* run -- the optimiser and
its schedule, the loss scaler, where in the corpus each rank had got to, and the
generators the augmentation and the dropout draw from. Anything left out makes
the second half of a run a different experiment from the first.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import random
import shutil
import sys
import threading
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from mlime.logging import log
from mlime.train.model import RouteAConfig, RouteAModel, count_correct
from mlime.train.samples import (
    BaseTokenizer,
    Batch,
    Collator,
    CorpusStream,
    TrainingExample,
    token_budget_batches,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Every number the loop needs, so a run is reproducible from one record."""

    max_steps: int
    base_lr: float = 3e-5
    new_lr: float = 1e-4
    warmup_fraction: float = 0.04
    weight_decay: float = 0.01
    #: Padded positions one step may cost, the fill tower's and the context
    #: tower's rectangles together -- not the fill tower's alone, which is a
    #: bound on the smaller half and was how a run came to use four times it.
    token_budget: int = 8192
    gradient_clip: float = 1.0
    seed: int = 0
    log_every: int = 10
    checkpoint_every: int = 500
    #: Numbered checkpoints kept on disk. One carries the optimiser state as well
    #: as the weights, so it is the size of two models, and a Kaggle session's
    #: output directory is capped at 20 GB -- a run that fills it loses the
    #: checkpoint it was about to write, which is the one that cannot be lost.
    keep_checkpoints: int = 2
    #: How long this *segment* may run before it pauses itself, or ``None`` for
    #: as long as it takes. A Kaggle session is killed at twelve hours and its
    #: output goes with it, so a segment that means to survive stops itself with
    #: time to spare and writes a checkpoint the next kernel resumes from.
    wall_budget_seconds: float | None = None
    fp16: bool = True
    #: Micro-batches whose gradients one optimiser step accumulates. The step's
    #: loss is the mean over every scored position across them -- not the mean
    #: of the per-batch means -- and a recorded stream position counts the
    #: micro-batches consumed.
    accumulate: int = 1
    #: Build the next batches on a producer thread while the step runs.
    prefetch: bool = False
    #: Steps after warm-up to split into per-phase timings. Zero leaves the
    #: loop untouched by the profiler's synchronising marks.
    profile_steps: int = 0
    #: Optimiser steps between the ranks agreeing whether the wall budget is
    #: spent. A world of one checks every step: it pays no collective either way.
    agree_every: int = 64
    #: torch.compile the towers and the decoder stack, per module, before DDP
    #: wraps them. A kernel-side choice: the checkpoint neither records it nor
    #: needs it, and a resumed segment simply compiles again.
    compile: bool = False
    #: The mode torch.compile runs at: "default" first,
    #: "max-autotune-no-cudagraphs" when the default leaves the card underfed.
    compile_mode: str = "default"
    #: Ranks one process runs as, when the run's world outnumbers its devices.
    #: A Kaggle segment is two real ranks and a Colab segment is one process of
    #: two virtual ones; each virtual rank owns the stream deal, the collator
    #: and the generator state of the rank it stands for, so the checkpoints
    #: the two legs write are interchangeable.
    virtual_ranks: int = 1
    #: Minutes between out-of-band checkpoints, beside the step cadence: each
    #: one is a full ``checkpoint-paused.pt`` trio written atomically, for a
    #: leg whose VM can die without warning (Colab) to publish. ``None`` leaves
    #: only the cadence checkpoints and the wall-budget pause.
    checkpoint_minutes: float | None = None

    COMPILE_MODES = ("default", "max-autotune-no-cudagraphs")

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}")
        if self.keep_checkpoints < 1:
            raise ValueError(f"keep_checkpoints must be at least 1, got {self.keep_checkpoints}")
        if self.accumulate < 1:
            raise ValueError(f"accumulate must be at least 1, got {self.accumulate}")
        if self.profile_steps < 0:
            raise ValueError(f"profile_steps must not be negative, got {self.profile_steps}")
        if self.agree_every < 1:
            raise ValueError(f"agree_every must be at least 1, got {self.agree_every}")
        if self.compile_mode not in self.COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {self.COMPILE_MODES}, got {self.compile_mode!r}"
            )
        if self.wall_budget_seconds is not None and self.wall_budget_seconds <= 0:
            raise ValueError(
                f"wall_budget_seconds must be positive, got {self.wall_budget_seconds}"
            )
        if self.virtual_ranks < 1:
            raise ValueError(f"virtual_ranks must be at least 1, got {self.virtual_ranks}")
        if self.checkpoint_minutes is not None and self.checkpoint_minutes <= 0:
            raise ValueError(f"checkpoint_minutes must be positive, got {self.checkpoint_minutes}")

    @property
    def warmup_steps(self) -> int:
        """Steps spent ramping the learning rate up from zero."""
        return max(1, int(self.max_steps * self.warmup_fraction))


@dataclass(frozen=True)
class Distributed:
    """Which process this is, and what it should run on."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @classmethod
    def from_environment(cls) -> Distributed:
        """Read the launcher's variables; a plain ``python`` run is a world of one."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            return cls()
        return cls(
            rank=int(os.environ["RANK"]),
            world_size=world_size,
            local_rank=int(os.environ.get("LOCAL_RANK", os.environ["RANK"])),
        )

    @property
    def is_main(self) -> bool:
        """Whether this process writes the checkpoints and the metrics."""
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        """The device this rank trains on."""
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def start(self) -> None:
        """Join the process group, if there is more than one process."""
        if self.world_size == 1:
            return
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )
        log.info("joined the process group", rank=self.rank, world=self.world_size)

    def stop(self) -> None:
        """Leave the process group."""
        if self.world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed the generators this run draws from.

    Each rank is offset so two ranks do not draw the same dropout masks, and the
    offset is the rank rather than something ambient so the run stays repeatable.
    """
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate multiplier: linear to 1 over the warmup, then cosine to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class MetricLog:
    """One JSON object per line, flushed as it goes so a killed kernel keeps them."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self.path = path

    def write(self, **record: object) -> None:
        """Append one record."""
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the file."""
        self._handle.close()

    def __enter__(self) -> MetricLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class Segment:
    """What one kernel's worth of a run did.

    A run is now a chain of segments, so "the loop returned" no longer means the
    run is over: it can also mean the segment spent its wall budget and left a
    checkpoint. Callers have to tell the two apart -- an unfinished run must not
    be scored or reported as a result -- so the loop says which it was rather
    than leaving it to be inferred from a file's existence.
    """

    metrics: Path
    step: int
    finished: bool


@dataclass(frozen=True)
class Accuracy:
    """How many characters a pass got right."""

    correct: int
    scored: int

    @property
    def rate(self) -> float:
        """Share correct; zero when nothing was scored."""
        return self.correct / self.scored if self.scored else 0.0


def epoch_groups(
    stream: CorpusStream, collator: Collator, budget: int, epoch: int
) -> Iterator[list[TrainingExample]]:
    """The examples of *epoch*, grouped exactly as the steps of that epoch will be.

    Grouping is not collation, and the two are separated here because three
    things need the boundaries and none of them needs the tensors: the loop, on
    its way to collating each group; :meth:`EpochBatches.skip`, walking a resumed
    stream back to a saved position; and counting how many steps an epoch is
    worth before a run fixes its schedule.

    The epoch is an argument rather than a detail because it decides what the
    examples *are*: the augmentation re-samples every epoch, an abbreviated
    sentence is shorter than a typed-out one, and a batch boundary is a function
    of the lengths. Counting epoch 0 twice is not counting two epochs.
    """
    stream.set_epoch(epoch)
    return token_budget_batches(iter(stream), budget, collator.max_context_tokens)


@dataclass(frozen=True)
class Produced:
    """A collated batch, and where the stream stood when it was made.

    ``epoch``/``index`` count the batches produced so far and ``collator_rng``
    is the collator's generator state just after this batch was collated. A
    checkpoint taken on this item resumes exactly after it: the batch a
    re-opened stream produces next is the one an uninterrupted run would have
    consumed. A prefetching producer carries this per item because the queue
    may already hold batches a pause never reaches -- the position a checkpoint
    writes is the last batch *consumed*, not the last one produced.
    """

    batch: Batch
    epoch: int
    index: int
    collator_rng: tuple[Any, ...]


class EpochBatches(Iterator[Batch]):
    """The stream as collated batches, epoch after epoch, and where in it we are.

    The position is what makes a run resumable. A rank's batches are its own --
    it owns its own shards, and how many batches an epoch holds depends on the
    lengths of the sentences in them -- so "step 4000" says nothing about where
    to start reading. The epoch and the count of batches taken from it do, and
    :meth:`skip` walks a fresh stream back to exactly there.
    """

    def __init__(self, stream: CorpusStream, collator: Collator, budget: int):
        self.stream = stream
        self.collator = collator
        self.budget = budget
        #: The epoch the batch last yielded came from.
        self.epoch = stream.epoch
        #: How many batches of that epoch have been yielded.
        self.index = 0
        self._groups = self._open_epoch()

    def __next__(self) -> Batch:
        return self.produce().batch

    def produce(self) -> Produced:
        """Collate the next group, and record the position it leaves behind."""
        batch = self.collator(self._next_group())
        return Produced(
            batch=batch,
            epoch=self.epoch,
            index=self.index,
            collator_rng=self.collator.rng.getstate(),
        )

    def produced_forever(self) -> Iterator[Produced]:
        """This stream as produced items, endlessly -- the no-prefetch path."""
        while True:
            yield self.produce()

    def _open_epoch(self) -> Iterator[list[TrainingExample]]:
        """Start reading :attr:`epoch`, grouped to the budget but not collated."""
        return epoch_groups(self.stream, self.collator, self.budget, self.epoch)

    def _next_group(self) -> list[TrainingExample]:
        """The next group of examples, rolling into the next epoch at the end of one."""
        group = next(self._groups, None)
        if group is not None:
            self.index += 1
            return group
        if self.index == 0:
            raise RuntimeError(f"epoch {self.epoch} of the corpus produced no example")
        self.epoch += 1
        log.info("epoch finished", epoch=self.epoch, **self.stream.builder.counts.as_dict())
        self.index = 0
        self._groups = self._open_epoch()
        return self._next_group()

    def skip(self, epoch: int, index: int) -> None:
        """Read forward to *index* batches into *epoch*, building no tensors.

        The examples still have to be built, because building them is what walks
        the stream, and they are grouped because a group boundary depends on the
        examples before it. They are not collated: collation is where the padding
        and the tensors are, and none of it would be looked at.
        """
        if epoch < 0 or index < 0:
            raise ValueError(f"a stream position is not negative, got epoch {epoch} index {index}")
        started = time.monotonic()
        self.epoch = epoch
        self.index = 0
        self._groups = self._open_epoch()
        for _ in range(index):
            self._next_group()
        if (self.epoch, self.index) != (epoch, index):
            raise RuntimeError(
                f"epoch {epoch} ran out before batch {index}, at {self.epoch}/{self.index}: "
                "this is not the corpus the checkpoint was written from"
            )
        log.info(
            "batch stream skipped",
            epoch=epoch,
            index=index,
            seconds=round(time.monotonic() - started, 1),
        )


#: Batches the producer thread keeps ready ahead of the consumer. Two is
#: enough to hide a collation under a step; deeper would only hold batches a
#: pause never reaches.
PREFETCH_DEPTH = 2


class Prefetcher(Iterator[Produced]):
    """Runs an :class:`EpochBatches` on a daemon thread into a bounded queue.

    Collation is host work -- parquet reads, the tokenizer, the padding -- and
    the tokenizer releases the GIL inside its call, so the producer's batches
    overlap the consumer's device-bound step rather than serialising behind it.
    Exceptions from the producer are ferried to the consumer and raised there,
    so a broken stream fails at the step that needed the batch, not silently on
    a thread nobody joins. Call :meth:`close` when the loop is done; a paused
    run's leftover queue entries are dropped, never trained on.
    """

    def __init__(self, batches: EpochBatches, depth: int = PREFETCH_DEPTH):
        self._queue: queue.Queue[Produced | BaseException] = queue.Queue(maxsize=depth)
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._fill, args=(batches,), daemon=True, name="batch-prefetch"
        )
        self._thread.start()

    def _fill(self, batches: EpochBatches) -> None:
        """Produce until told to stop; a producer failure is queued, not raised here."""
        while not self._stopped.is_set():
            try:
                item: Produced | BaseException = batches.produce()
            except BaseException as error:  # ferried to the consumer thread
                item = error
            while not self._stopped.is_set():
                try:
                    self._queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if isinstance(item, BaseException):
                return

    def __iter__(self) -> Prefetcher:
        return self

    def __next__(self) -> Produced:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        """Stop the producer and wait briefly for it to notice."""
        self._stopped.set()
        self._thread.join(timeout=10)


#: Steps a profiling run leaves unmeasured first, so the phase table describes
#: the loop at cruising speed rather than at cold caches and lazy imports.
PROFILE_WARMUP_STEPS = 10

#: A compiled run's warm-up ends once dynamo has emitted no new graph for this
#: many consecutive steps, and never runs past ``COMPILE_WARMUP_CAP``: a shape
#: that still specialises then keeps doing so, and the profile says which.
COMPILE_STABLE_STEPS = 5
COMPILE_WARMUP_CAP = 30


def _dynamo_graphs() -> int:
    """How many graphs dynamo has compiled in this process so far."""
    from torch._dynamo.utils import counters

    return int(counters["stats"].get("unique_graphs") or 0)


def _dynamo_breaks() -> int:
    """How many graph breaks dynamo has taken in this process so far."""
    from torch._dynamo.utils import counters

    return int(sum(counters["graph_breaks"].values()))


#: The phases one optimiser step is split into, in the order they run.
PHASES = (
    "batch_wait",
    "h2d",
    "forward",
    "backward",
    "unscale_clip",
    "optimizer",
    "agree_log",
)


class Profiler:
    """Per-phase wall timing, collected only while a run asks for it.

    Timing a phase means synchronising the device at its boundary, so a
    profiled step is deliberately serial: the table says where the time went
    and is not itself a faster total. Outside the collected window every mark
    is one branch, so the option costs nothing when it is off. A phase can be
    marked several times in a step -- accumulation's micro-batches each mark
    ``h2d``/``forward``/``backward`` -- and the marks add.

    ``dynamic`` is for a compiled run: its warm-up ends once dynamo has added
    no graph for ``COMPILE_STABLE_STEPS`` steps running (capped at
    ``COMPILE_WARMUP_CAP``), each collected step carries how many graphs and
    breaks happened inside it, and ``done`` lets the loop leave as soon as the
    window is full. ``sync_check`` arms the sync guard -- off for compiled
    runs, whose proof is zero breaks and recompiles, since a legitimate
    mid-window compile would trip a guard written for eager code.
    """

    def __init__(
        self, device: torch.device, steps: int, *, dynamic: bool = False, sync_check: bool = True
    ):
        self.device = device
        self.steps = steps
        self.dynamic = dynamic
        self.sync_check = sync_check
        self.records: list[dict[str, float]] = []
        self._current: dict[str, float] | None = None
        self._t0 = 0.0
        self.warmup_steps = 0
        self._stable = 0
        self._warmed = not dynamic
        self._window: list[float] = []
        self._graphs_at_begin = _dynamo_graphs() if dynamic else 0
        self._breaks_at_begin = _dynamo_breaks() if dynamic else 0

    def counters(self) -> tuple[int, int]:
        """(new graphs, graph breaks) since this step began -- the compile cost."""
        if not self.dynamic:
            return (0, 0)
        return (
            _dynamo_graphs() - self._graphs_at_begin,
            _dynamo_breaks() - self._breaks_at_begin,
        )

    def begin(self, local_step: int) -> None:
        """Start a step: collect it when it lands inside the profiled window."""
        warmed = self._warmed if self.dynamic else local_step > PROFILE_WARMUP_STEPS
        collect = self.steps > 0 and warmed and len(self.records) < self.steps
        if not collect:
            self.warmup_steps += 1
        self._current = {} if collect else None
        if collect and not self._window:
            self._window.append(time.time())
        if self.dynamic:
            self._graphs_at_begin = _dynamo_graphs()
            self._breaks_at_begin = _dynamo_breaks()
        self._t0 = time.perf_counter()

    def mark(self, phase: str) -> None:
        """Attribute everything since the last mark to *phase* and move on."""
        if self._current is None:
            return
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; known: {PHASES}")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        self._current[phase] = self._current.get(phase, 0.0) + (now - self._t0)
        self._t0 = now

    @contextlib.contextmanager
    def guarded(self) -> Iterator[None]:
        """A region a collected step proves needs no device synchronisation.

        Forward and backward are wrapped in it: with ``sync_debug_mode`` at
        ``error``, an ``.item()``, a ``nonzero``, a masked index or a
        ``bool(tensor)`` inside either fails the run loudly instead of stalling
        it quietly. Off CUDA, and off the profiled window, it is a no-op.
        """
        checking = self.sync_check and self._current is not None and self.device.type == "cuda"
        if checking:
            torch.cuda.set_sync_debug_mode("error")
        try:
            yield
        finally:
            if checking:
                torch.cuda.set_sync_debug_mode("default")

    def end(self) -> None:
        """Close the step; keep it if it was being collected.

        In a compiled run the step's compile cost is settled here too: new
        graphs feed the stability counter that ends the warm-up, and both
        deltas are stamped onto a collected record so the report can total
        what happened inside the window.
        """
        recompiles = 0
        graph_breaks = 0
        if self.dynamic:
            recompiles, graph_breaks = self.counters()
            if not self._warmed:
                self._stable = self._stable + 1 if recompiles == 0 else 0
                if self._stable >= COMPILE_STABLE_STEPS or self.warmup_steps >= COMPILE_WARMUP_CAP:
                    self._warmed = True
        if self._current is not None:
            if self.dynamic:
                self._current["recompiles"] = float(recompiles)
                self._current["graph_breaks"] = float(graph_breaks)
            self.records.append(self._current)
            self._current = None
            if len(self.records) == self.steps:
                self._window.append(time.time())

    def done(self) -> bool:
        """The window is full -- a measurement run has nothing left to measure."""
        return self.steps > 0 and len(self.records) >= self.steps

    def report(self) -> dict[str, Any]:
        """Mean and p95 of each phase over the collected steps, as a record.

        ``window`` brackets the collected steps in wall-clock epochs, so the
        kernel can cut its GPU samples to the same stretch. ``recompiles`` and
        ``graph_breaks`` total what dynamo did inside the window; a steady
        compiled run reads zero on both.
        """

        def stats(values: list[float]) -> dict[str, float]:
            ordered = sorted(values)
            p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
            return {"mean": sum(ordered) / len(ordered), "p95": p95}

        return {
            "steps": len(self.records),
            "warmup_steps": self.warmup_steps,
            "window": self._window,
            "recompiles": int(sum(record.get("recompiles", 0.0) for record in self.records)),
            "graph_breaks": int(sum(record.get("graph_breaks", 0.0) for record in self.records)),
            "phases": {
                phase: stats([record.get(phase, 0.0) for record in self.records])
                for phase in PHASES
            },
        }


@dataclass(frozen=True)
class Generators:
    """The generators a rank's *process* draws from: Python's, torch's, CUDA's.

    One process running several virtual ranks keeps a snapshot per rank and
    swaps it in for that rank's accumulation window: from the model's point of
    view the draws then match a world that gave each rank its own process.
    """

    python: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]

    @classmethod
    def capture(cls) -> Generators:
        """The state of the process-level generators as of now."""
        return cls(
            python=random.getstate(),
            torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        )

    def restore(self) -> None:
        """Put the process-level generators back where this snapshot found them."""
        random.setstate(self.python)
        torch.set_rng_state(self.torch_cpu)
        if self.torch_cuda:
            torch.cuda.set_rng_state_all(self.torch_cuda)


@dataclass(frozen=True)
class RandomState:
    """Where each generator one rank draws from had got to.

    Carried in the checkpoint rather than re-seeded on resume: re-seeding
    restarts the sequences, and a step whose dropout masks and context-drop
    decisions differ from the ones the uninterrupted run would have drawn is a
    different step, however small the difference in the loss.
    """

    python: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]
    collator: tuple[Any, ...]

    def as_record(self) -> dict[str, Any]:
        """The plain form that goes in the checkpoint."""
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RandomState:
        """Read back what :meth:`as_record` wrote."""
        return cls(
            python=tuple(record["python"]),
            torch_cpu=record["torch_cpu"],
            torch_cuda=list(record["torch_cuda"]),
            collator=tuple(record["collator"]),
        )


@dataclass(frozen=True)
class RankState:
    """One rank's place in its own stream, and the generators that go with it."""

    epoch: int
    index: int
    random: RandomState

    @classmethod
    def capture(cls, consumed: Produced, generators: Generators) -> RankState:
        """Where this rank is, ready to be gathered onto rank 0.

        The collator's state is carried in from the last *consumed* batch
        rather than read live -- a prefetching producer may have moved the real
        generator ahead of what training has seen -- and the process-level
        generators come from the lane's own snapshot, not whatever rank ran
        most recently in this process.
        """
        return cls(
            epoch=consumed.epoch,
            index=consumed.index,
            random=RandomState(
                python=generators.python,
                torch_cpu=generators.torch_cpu,
                torch_cuda=generators.torch_cuda,
                collator=consumed.collator_rng,
            ),
        )

    def as_record(self) -> dict[str, Any]:
        """The plain form that goes in the checkpoint."""
        return {"epoch": self.epoch, "index": self.index, "random": self.random.as_record()}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RankState:
        """Read back what :meth:`as_record` wrote."""
        return cls(
            epoch=int(record["epoch"]),
            index=int(record["index"]),
            random=RandomState.from_record(record["random"]),
        )


@dataclass
class RankLane:
    """One virtual rank's lane inside a process.

    A lane owns everything a real rank owns: the stream its shard deal comes
    from (built by the caller as rank ``global_rank`` of the whole world), the
    producer feeding it batches, the snapshot of the process-level generators
    restored before its accumulation window and captured after it, and the
    last batch an optimiser step *consumed* -- the position a checkpoint
    writes for it.
    """

    source: Iterator[Produced]
    prefetcher: Prefetcher | None
    generators: Generators
    consumed: Produced | None = None

    def close(self) -> None:
        """Stop the lane's producer, if it has one."""
        if self.prefetcher is not None:
            self.prefetcher.close()


def unwrap(model: nn.Module) -> RouteAModel:
    """The route A model inside, whether or not DDP wrapped it."""
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(inner, RouteAModel):
        raise TypeError(f"expected a RouteAModel, got {type(inner).__name__}")
    return inner


def evaluate(
    model: nn.Module,
    examples: Sequence[TrainingExample],
    tokenizer: BaseTokenizer,
    device: torch.device,
    token_budget: int,
    with_context: bool,
) -> Accuracy:
    """Masked-character accuracy over *examples*, with the context on or off.

    Context is switched by the collator's dropout being 0 or 1 rather than by a
    second code path, so the two numbers differ in exactly one thing.
    """
    route = unwrap(model)
    collator = Collator(
        tokenizer,
        route.candidate_mask,
        context_dropout=0.0 if with_context else 1.0,
    )
    correct = scored = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for group in token_budget_batches(
            iter(examples), token_budget, collator.max_context_tokens
        ):
            batch = collator(group).to(device)
            logits = route.scores(batch)
            hit, total = count_correct(route.predictions(logits, batch), batch.targets)
            correct += hit
            scored += total
    if was_training:
        model.train()
    return Accuracy(correct=correct, scored=scored)


def train(
    model: RouteAModel,
    lanes: Sequence[tuple[CorpusStream, Collator]],
    config: TrainingConfig,
    out_dir: Path,
    distributed: Distributed | None = None,
    resume: Path | None = None,
    run_config: Path | None = None,
) -> Segment:
    """Run towards *config.max_steps* optimiser steps and say what this segment did.

    *lanes* holds one ``(stream, collator)`` pair per virtual rank this process
    runs as -- ``config.virtual_ranks`` of them, built by the caller as
    consecutive ranks of the whole world. An optimiser step runs each lane's
    accumulation window in turn under that lane's own generator state, so a
    one-process Colab segment computes what a two-process Kaggle segment would.

    Writes every step's loss to the metrics file rather than printing it, because
    "the loss went down" is a claim that has to be checkable after the kernel has
    been torn down.

    With *resume*, the run continues the one that wrote that checkpoint: the same
    weights, the same optimiser and schedule, the same generators, and each rank
    reading on from where it was. The metrics file is this segment's own, so the
    losses of a chained run are read by concatenating the segments' files.

    With *config.wall_budget_seconds*, the segment stops itself when the budget
    is spent and writes ``checkpoint-paused``. It writes no ``checkpoint-final``:
    the final checkpoint means the run reached ``max_steps``, and a segment that
    ran out of clock has not. *config.checkpoint_minutes* adds the same trio on
    a wall-clock cadence, atomically, for a leg that can die without warning.
    The trio's run-config is the kernel's own file copied verbatim from
    *run_config*, so the leg that picks it up reads the run's real fields.
    """
    world = distributed or Distributed()
    virtual = config.virtual_ranks
    if len(lanes) != virtual:
        raise ValueError(
            f"train got {len(lanes)} (stream, collator) lanes but virtual_ranks is {virtual}"
        )
    if config.checkpoint_minutes is not None and run_config is None:
        raise ValueError("checkpoint_minutes copies the run's own config; pass run_config")
    world.start()
    total_world = world.world_size * virtual
    device = world.device
    model.to(device)
    resumed = (
        Resumption.read(resume, config, model.config, world, record=model.config_key)
        if resume is not None
        else None
    )
    if resumed is not None:
        model.load_state_dict(resumed.model)
    trained: nn.Module = model
    if world.world_size > 1:
        trained = DistributedDataParallel(
            model,
            device_ids=[world.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    optimiser = torch.optim.AdamW(
        model.parameter_groups(config.base_lr, config.new_lr),
        weight_decay=config.weight_decay,
        # The fused CUDA kernels run the whole step without leaving the device;
        # the flag does not exist on CPU, where the plain implementation stands.
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: cosine_with_warmup(step, config.warmup_steps, config.max_steps),
    )
    amp = config.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricLog(out_dir / "metrics.jsonl")
    if world.is_main:
        metrics.write(
            event="config",
            world_size=world.world_size,
            ranks=total_world,
            amp=amp,
            device=str(device),
            **asdict(config),
        )

    rank_lanes: list[RankLane] = []
    for index, (stream, collator) in enumerate(lanes):
        global_rank = world.rank * virtual + index
        batches = EpochBatches(stream, collator, config.token_budget)
        if resumed is not None:
            position = resumed.positions[index]
            batches.skip(position.epoch, position.index)
            collator.rng.setstate(position.random.collator)
            generators = Generators(
                python=position.random.python,
                torch_cpu=position.random.torch_cpu,
                torch_cuda=position.random.torch_cuda,
            )
        else:
            seed_everything(config.seed, global_rank)
            generators = Generators.capture()
        prefetcher = Prefetcher(batches) if config.prefetch else None
        rank_lanes.append(
            RankLane(
                source=prefetcher if prefetcher is not None else batches.produced_forever(),
                prefetcher=prefetcher,
                generators=generators,
            )
        )

    step = 0
    if resumed is not None:
        optimiser.load_state_dict(resumed.optimiser)
        scheduler.load_state_dict(resumed.scheduler)
        scaler.load_state_dict(resumed.scaler)
        step = resumed.step
        if world.is_main:
            metrics.write(
                event="resume",
                checkpoint=str(resumed.path),
                step=step,
                epoch=resumed.positions[0].epoch,
                index=resumed.positions[0].index,
            )
    checkpoints = Checkpointer(
        model=model,
        optimiser=optimiser,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        out_dir=out_dir,
        world=world,
        run_config=run_config,
    )

    profiler = Profiler(
        device,
        config.profile_steps,
        dynamic=config.compile,
        sync_check=not config.compile,
    )

    trained.train()
    started = time.monotonic()
    # A segment logs its own first step whatever the interval, so the first loss
    # in a metrics file is always the first step that file's segment ran.
    first_step = step + 1
    finished = False
    consumed_any = False
    interval_seconds = (
        config.checkpoint_minutes * 60 if config.checkpoint_minutes is not None else None
    )
    next_interval = interval_seconds if interval_seconds is not None else 0.0
    first_logged_loss: float | None = None
    last_logged_loss: float | None = None

    def states() -> list[RankState]:
        """Each lane's position, in world order, for a checkpoint to write."""
        written = []
        for lane in rank_lanes:
            if lane.consumed is None:
                raise RuntimeError("a lane consumed nothing; it has no position to write")
            written.append(RankState.capture(lane.consumed, lane.generators))
        return written

    def interval_summary(step: int, elapsed: float) -> dict[str, Any]:
        """The run-summary half of an interval trio, shaped like the kernel's own."""
        return {
            "max_steps": config.max_steps,
            "resumed_from": None if resume is None else str(resume),
            "first_step": first_step,
            "last_step": step,
            "checkpoint_step": step,
            "first_loss": first_logged_loss,
            "last_loss": last_logged_loss,
            "train_seconds": round(elapsed, 1),
            "finished": False,
            "world_size": total_world,
            "processes": world.world_size,
            "virtual_ranks": virtual,
        }

    try:
        while True:
            profiler.begin(step - first_step + 2)
            windows = [[next(lane.source) for _ in range(config.accumulate)] for lane in rank_lanes]
            profiler.mark("batch_wait")
            step_loss: torch.Tensor | None = None
            step_extras: dict[str, torch.Tensor] = {}
            for lane_index, (lane, items) in enumerate(zip(rank_lanes, windows, strict=True)):
                lane.generators.restore()
                # One step's gradient is the mean over every scored position
                # across its micro-batches, so each micro-batch's loss --
                # already its own mean -- is weighted by its share of the
                # positions, not by 1/N.
                counts = [item.batch.scored.numel() for item in items]
                scored_total = sum(counts)
                if scored_total == 0 or 0 in counts:
                    raise RuntimeError("a batch reached the loop with no scored position")
                for micro, (item, count) in enumerate(zip(items, counts, strict=True)):
                    batch = item.batch.to(device)
                    profiler.mark("h2d")
                    # Under DDP only the very last micro-batch's backward
                    # allreduces; every earlier one -- of this rank's window
                    # and of every earlier virtual rank's -- accumulates
                    # locally under no_sync, so a step pays one exchange.
                    synchronising = lane_index == len(rank_lanes) - 1 and micro == len(items) - 1
                    sync_context = (
                        contextlib.nullcontext()
                        if synchronising or not isinstance(trained, DistributedDataParallel)
                        else trained.no_sync()
                    )
                    with sync_context:
                        with (
                            profiler.guarded(),
                            torch.autocast(
                                device_type=device.type, dtype=torch.float16, enabled=amp
                            ),
                        ):
                            output = trained(batch)
                        profiler.mark("forward")
                        if output.loss is None:
                            raise RuntimeError("a batch reached the loop with no scored position")
                        weight = count / scored_total
                        # The world's gradient is the mean of the per-rank
                        # accumulations -- what DDP's all-reduce computes when
                        # each rank has its own process. Dividing by the ranks
                        # this process runs makes the same mean come out of one
                        # process.
                        with profiler.guarded():
                            scaler.scale(output.loss * weight / virtual).backward()
                    profiler.mark("backward")
                    # The logged step is lane 0's view: the same numbers a real
                    # rank 0 writes, which is what the segment summaries read.
                    if lane_index == 0:
                        contribution = output.loss.detach() * weight
                        step_loss = contribution if step_loss is None else step_loss + contribution
                        for name, extra in output.extras.items():
                            part = extra.detach() * weight
                            step_extras[name] = step_extras.get(name, 0.0) + part
                lane.generators = Generators.capture()
                lane.consumed = items[-1]
            consumed_any = True
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            profiler.mark("unscale_clip")
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
            scheduler.step()
            profiler.mark("optimizer")
            step += 1
            if world.is_main and (step % config.log_every == 0 or step == first_step):
                if step_loss is None:  # unreachable: every micro-batch raises or contributes
                    raise RuntimeError("a step held no scored position")
                loss_now = float(step_loss)
                record = {
                    "event": "step",
                    "step": step,
                    "loss": loss_now,
                    **{name: float(extra) for name, extra in step_extras.items()},
                    "lr": scheduler.get_last_lr()[0],
                    "new_lr": scheduler.get_last_lr()[1],
                    "examples": sum(item.batch.size for item in windows[0]),
                    "tokens": sum(item.batch.tokens for item in windows[0]),
                    "context_tokens": sum(item.batch.context_tokens for item in windows[0]),
                    "seconds": round(time.monotonic() - started, 1),
                    "gates": model.gates(),
                }
                first_logged_loss = loss_now if first_logged_loss is None else first_logged_loss
                last_logged_loss = loss_now
                if config.compile:
                    # Compiles belong to the step that caused them: all of a
                    # step's dynamo work sits in its forward/backward, ahead
                    # of where this record is written, so the counter delta
                    # since the step began is the step's own compile cost.
                    recompiles, graph_breaks = profiler.counters()
                    record["recompiles"] = recompiles
                    record["graph_breaks"] = graph_breaks
                metrics.write(**record)
                log.info("step", **{k: v for k, v in record.items() if k != "event"})
            if step % config.checkpoint_every == 0:
                checkpoints.numbered(step, states())
            # Ranks cannot leave a collective on different steps, so with more
            # than one the verdict is broadcast on a fixed cadence; a world of
            # one checks every step -- the check costs it no collective.
            elapsed = time.monotonic() - started
            verdict = 0  # 1 = an interval checkpoint is due; 2 = the budget is spent
            if config.wall_budget_seconds is not None and elapsed >= config.wall_budget_seconds:
                verdict = 2
            elif interval_seconds is not None and elapsed >= next_interval:
                verdict = 1
            if world.world_size > 1:
                verdict = agreed(world, verdict, device) if step % config.agree_every == 0 else 0
            profiler.mark("agree_log")
            profiler.end()
            if step >= config.max_steps:
                finished = True
                break
            if verdict == 2:
                checkpoints.paused(step, states())
                if world.is_main:
                    metrics.write(event="paused", step=step, seconds=round(elapsed, 1))
                log.info(
                    "wall budget spent, segment paused",
                    step=step,
                    seconds=round(elapsed, 1),
                )
                break
            if verdict == 1:
                checkpoints.interval(step, states(), interval_summary(step, elapsed))
                if interval_seconds is None:
                    raise RuntimeError("an interval verdict was reached without a cadence")
                next_interval += interval_seconds
                if world.is_main:
                    metrics.write(event="interval", step=step, seconds=round(elapsed, 1))
            # A measurement run ends when its window fills -- a compiled one
            # may warm up late, so its max_steps is a cap, not the target.
            if profiler.done():
                break
    finally:
        for lane in rank_lanes:
            lane.close()
    if not consumed_any:
        raise RuntimeError("the loop ran no step; nothing was consumed")
    if finished:
        checkpoints.final(step, states())
    if profiler.records:
        profile = profiler.report()
        # The kernel reads the phase table out of the log stream, like
        # `peak-memory`; rank 0 also writes it to the metrics file.
        sys.stdout.write(json.dumps({"event": "profile", "rank": world.rank, **profile}) + "\n")
        sys.stdout.flush()
        if world.is_main:
            metrics.write(event="profile", **profile)
    if config.compile and world.is_main:
        # How dynamo did: graphs compiled, breaks taken, and each recompile --
        # the record a compiled run owes the reader of its numbers. The import
        # lives here because torch only loads _dynamo on a compile.
        from torch._dynamo.utils import counters as dynamo_counters

        dynamo = {name: dict(counter) for name, counter in dynamo_counters.items() if counter}
        sys.stdout.write(
            json.dumps({"event": "dynamo", "rank": world.rank, "counters": dynamo}) + "\n"
        )
        sys.stdout.flush()
        metrics.write(event="dynamo", counters=dynamo)
    metrics.close()
    world.stop()
    return Segment(metrics=metrics.path, step=step, finished=finished)


def save_checkpoint(
    model: RouteAModel,
    path: Path,
    step: int,
    config: TrainingConfig,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    positions: Sequence[Mapping[str, Any]],
) -> None:
    """Write the weights, the run's own description, and how to carry on from here.

    The optimiser is AdamW, so half of what a step depends on is in its state and
    not in the weights; the scaler's scale is the same kind of thing, and so is
    the schedule's place on the cosine. *positions* is indexed by rank because
    each rank reads its own shards and no rank can work out another's.
    """
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            model.config_key: asdict(model.config),
            "training": asdict(config),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "positions": list(positions),
        },
        path,
    )
    log.info("checkpoint written", path=str(path), step=step)


def checkpoint_step(path: Path) -> int:
    """The step a numbered checkpoint holds, read off its name."""
    return int(path.stem.rsplit("-", 1)[1])


def rotate_checkpoints(out_dir: Path, keep: int) -> None:
    """Leave only the newest *keep* numbered checkpoints.

    ``checkpoint-final`` and ``checkpoint-paused`` are not numbered and are not
    candidates: the first is the run's result and the second is the only thing
    the next kernel can start from.
    """
    if keep < 1:
        raise ValueError(f"a run must keep at least one checkpoint, got {keep}")
    numbered = sorted(out_dir.glob("checkpoint-[0-9]*.pt"), key=checkpoint_step)
    for path in numbered[:-keep]:
        path.unlink()
        log.info("checkpoint rotated out", path=str(path))


def agreed(world: Distributed, verdict: int, device: torch.device) -> int:
    """Rank 0's *verdict*, made every rank's.

    What a step owes the clock is measured per rank, and ranks do not reach a
    step at the same instant, so two of them can measure it differently on the
    same step. Ranks that disagreed would not disagree politely: the ones that
    stopped would leave the ones that did not waiting in the next gradient
    all-reduce, and a hang is worse than either answer. So one rank decides
    and the rest are told.

    The verdicts: 0 carry on, 1 an interval checkpoint is due, 2 the wall
    budget is spent.
    """
    if world.world_size == 1:
        return verdict
    sent = torch.tensor([verdict], dtype=torch.int64, device=device)
    torch.distributed.broadcast(sent, src=0)
    return int(sent.item())


def gather_positions(world: Distributed, states: Sequence[RankState]) -> list[dict[str, Any]]:
    """Every rank's place in its own stream, gathered so rank 0 can write them all.

    A checkpoint is resumable only if it can put every rank back where it was,
    so it writes a position per rank of the whole world -- real and virtual
    alike, one record per lane in rank order. Collective: every rank calls it,
    including the ones that write nothing, which is why it is not inside an
    ``is_main`` guard.
    """
    mine = [state.as_record() for state in states]
    if world.world_size == 1:
        return mine
    gathered: list[Any] = [None] * world.world_size
    torch.distributed.all_gather_object(gathered, mine)
    positions: list[dict[str, Any]] = []
    for rank, block in enumerate(gathered):
        if not isinstance(block, list):
            raise RuntimeError(f"rank {rank} did not report a stream position")
        positions.extend(dict(record) for record in block)
    return positions


@dataclass(frozen=True)
class Checkpointer:
    """Everything a checkpoint of this run is made of, in one place.

    The optimiser and the scaler belong to the loop, not to the model, so a
    function that writes "the weights" cannot write a resumable checkpoint on its
    own. Holding them together is also what keeps the numbered checkpoint and the
    final one from drifting into two different formats.
    """

    model: RouteAModel
    optimiser: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    config: TrainingConfig
    out_dir: Path
    world: Distributed
    #: The run's own config file, copied verbatim into each interval trio.
    run_config: Path | None

    def numbered(self, step: int, states: Sequence[RankState]) -> None:
        """Write ``checkpoint-<step>`` and rotate the older ones out."""
        self._write(self.out_dir / f"checkpoint-{step:06d}.pt", step, states)
        if self.world.is_main:
            rotate_checkpoints(self.out_dir, self.config.keep_checkpoints)

    def final(self, step: int, states: Sequence[RankState]) -> None:
        """Write ``checkpoint-final``, which is never rotated out."""
        self._write(self.out_dir / "checkpoint-final.pt", step, states)

    def paused(self, step: int, states: Sequence[RankState]) -> None:
        """Write ``checkpoint-paused``, the one the next kernel resumes from.

        Named rather than numbered so the next kernel knows what to mount without
        being told a step number, and so rotation cannot take it: a segment that
        stopped on the clock has no other copy of where it got to.
        """
        self._write(self.out_dir / "checkpoint-paused.pt", step, states)

    def interval(
        self, step: int, states: Sequence[RankState], summary: Mapping[str, Any]
    ) -> Path | None:
        """Write the paused trio under ``interval-<step>``, atomically, and keep the newest.

        A leg that can die without warning cannot wait for the wall-budget
        pause to learn where it got to: this writes the same resume point --
        the checkpoint, the run's config, and a summary in the shape the
        kernel writes -- every so many minutes instead. The directory is
        staged under a hidden name and renamed whole, so a reader never sees
        half of a trio; resuming from one is resuming from a pause.
        """
        positions = gather_positions(self.world, states)
        if not self.world.is_main:
            return None
        target = self.out_dir / f"interval-{step:06d}"
        staging = self.out_dir / f".interval-{step:06d}.tmp"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        save_checkpoint(
            self.model,
            staging / "checkpoint-paused.pt",
            step,
            self.config,
            self.optimiser,
            self.scheduler,
            self.scaler,
            positions,
        )
        if self.run_config is None:
            raise RuntimeError("an interval trio needs the run's own config file")
        shutil.copyfile(self.run_config, staging / "run-config.json")
        (staging / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        staging.replace(target)
        for stale in self.out_dir.glob("interval-*"):
            if stale != target:
                shutil.rmtree(stale)
        return target

    def _write(self, path: Path, step: int, states: Sequence[RankState]) -> None:
        """Gather the positions -- all ranks -- and write the file on rank 0."""
        positions = gather_positions(self.world, states)
        if not self.world.is_main:
            return
        save_checkpoint(
            self.model,
            path,
            step,
            self.config,
            self.optimiser,
            self.scheduler,
            self.scaler,
            positions,
        )


#: The fields of :class:`TrainingConfig` a resume may change. Everything else
#: describes the run, so changing it makes the next segment a different
#: experiment; these describe the *kernel* the segment runs in -- a longer
#: session, a profiling pass, prefetching, how often ranks agree on the clock,
#: whether the towers were compiled, how many ranks a process stands for, how
#: often an interval checkpoint lands -- not the result it produces.
#: ``accumulate`` changes the gradient, so it is deliberately not here.
SEGMENT_FIELDS = frozenset(
    {
        "wall_budget_seconds",
        "prefetch",
        "profile_steps",
        "agree_every",
        "compile",
        "compile_mode",
        "virtual_ranks",
        "checkpoint_minutes",
    }
)


def refuse_mismatch(
    path: Path,
    kind: str,
    saved: Mapping[str, Any],
    wanted: Mapping[str, Any],
    except_for: Collection[str] = (),
) -> None:
    """Refuse to resume when the *kind* record differs, naming every field that does.

    Continuing a run under a different schedule is not a resume: the cosine bends
    somewhere else, the warmup is over a different length, and the two segments'
    metrics do not describe one run. The comparison is over every field but
    *except_for*, so the refusal names what to change rather than saying only
    that something did.
    """
    compared = (set(saved) | set(wanted)) - set(except_for)
    differing = sorted(key for key in compared if saved.get(key) != wanted.get(key))
    if not differing:
        return
    fields = ", ".join(f"{key} {saved.get(key)!r} -> {wanted.get(key)!r}" for key in differing)
    raise ValueError(f"{path} was written under a different {kind} config; these differ: {fields}")


@dataclass(frozen=True)
class Resumption:
    """A checkpoint read back and checked, ready to be poured into a fresh run."""

    path: Path
    step: int
    model: dict[str, Any]
    optimiser: dict[str, Any]
    scheduler: dict[str, Any]
    scaler: dict[str, Any]
    #: This process's lanes' positions, in lane order: the checkpoint's full
    #: list is indexed by rank of the whole world, and a process that stands
    #: for several virtual ranks takes the contiguous block that is its own.
    positions: list[RankState]

    @classmethod
    def read(
        cls,
        path: Path,
        config: TrainingConfig,
        model_config: RouteAConfig,
        world: Distributed,
        record: str = "route_a",
    ) -> Resumption:
        """Load *path* and refuse it unless it is this run, one segment earlier."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        absent = sorted({"optimiser", "scheduler", "scaler", "positions"} - set(state))
        if absent:
            raise ValueError(
                f"{path} predates resumable training and has no {absent}; "
                "it can be trained from, but not resumed"
            )
        if record not in state:
            raise ValueError(
                f"{path} holds no {record} config; it is a checkpoint of a different model"
            )
        refuse_mismatch(path, "training", state["training"], asdict(config), SEGMENT_FIELDS)
        refuse_mismatch(path, record, state[record], asdict(model_config))
        positions = state["positions"]
        expected = world.world_size * config.virtual_ranks
        if len(positions) != expected:
            raise ValueError(
                f"{path} was written by a world of {len(positions)} ranks and this one has "
                f"{expected}; a rank would read shards it has no position for"
            )
        step = int(state["step"])
        if step >= config.max_steps:
            raise ValueError(
                f"{path} is already at step {step} of {config.max_steps}; "
                "raise max_steps to run further, or this resume would do nothing"
            )
        first = world.rank * config.virtual_ranks
        return cls(
            path=path,
            step=step,
            model=state["model"],
            optimiser=state["optimiser"],
            scheduler=state["scheduler"],
            scaler=state["scaler"],
            positions=[
                RankState.from_record(record)
                for record in positions[first : first + config.virtual_ranks]
            ],
        )
