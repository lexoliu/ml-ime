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

import json
import math
import os
import random
import time
from collections.abc import Iterator, Mapping, Sequence
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
    fp16: bool = True

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}")
        if self.keep_checkpoints < 1:
            raise ValueError(f"keep_checkpoints must be at least 1, got {self.keep_checkpoints}")

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
class Accuracy:
    """How many characters a pass got right."""

    correct: int
    scored: int

    @property
    def rate(self) -> float:
        """Share correct; zero when nothing was scored."""
        return self.correct / self.scored if self.scored else 0.0


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
        return self.collator(self._next_group())

    def _open_epoch(self) -> Iterator[list[TrainingExample]]:
        """Start reading :attr:`epoch`, grouped to the budget but not collated."""
        self.stream.set_epoch(self.epoch)
        return token_budget_batches(
            iter(self.stream), self.budget, self.collator.max_context_tokens
        )

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

    @classmethod
    def capture(cls, collator: Collator) -> RandomState:
        """The state of this rank's generators as of now."""
        return cls(
            python=random.getstate(),
            torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            collator=collator.rng.getstate(),
        )

    def restore(self, collator: Collator) -> None:
        """Put this rank's generators back where the checkpoint found them."""
        random.setstate(self.python)
        torch.set_rng_state(self.torch_cpu)
        if self.torch_cuda:
            torch.cuda.set_rng_state_all(self.torch_cuda)
        collator.rng.setstate(self.collator)

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
    def capture(cls, batches: EpochBatches) -> RankState:
        """Where this rank is, ready to be gathered onto rank 0."""
        return cls(
            epoch=batches.epoch,
            index=batches.index,
            random=RandomState.capture(batches.collator),
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
    collator = Collator(tokenizer, context_dropout=0.0 if with_context else 1.0)
    correct = scored = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for group in token_budget_batches(
            iter(examples), token_budget, collator.max_context_tokens
        ):
            batch = collator(group).to(device)
            logits = route(batch).logits
            hit, total = count_correct(route.predictions(logits, batch), batch.targets)
            correct += hit
            scored += total
    if was_training:
        model.train()
    return Accuracy(correct=correct, scored=scored)


def train(
    model: RouteAModel,
    stream: CorpusStream,
    collator: Collator,
    config: TrainingConfig,
    out_dir: Path,
    distributed: Distributed | None = None,
    resume: Path | None = None,
) -> Path:
    """Run to *config.max_steps* optimiser steps and return the metrics file.

    Returns rather than prints, and writes every step's loss to the metrics file,
    because "the loss went down" is a claim that has to be checkable after the
    kernel has been torn down.

    With *resume*, the run continues the one that wrote that checkpoint: the same
    weights, the same optimiser and schedule, the same generators, and each rank
    reading on from where it was. The metrics file is this segment's own, so the
    losses of a chained run are read by concatenating the segments' files.
    """
    world = distributed or Distributed()
    world.start()
    seed_everything(config.seed, world.rank)
    device = world.device
    model.to(device)
    resumed = Resumption.read(resume, config, model.config, world) if resume is not None else None
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
            amp=amp,
            device=str(device),
            **asdict(config),
        )

    batches = EpochBatches(stream, collator, config.token_budget)
    step = 0
    if resumed is not None:
        optimiser.load_state_dict(resumed.optimiser)
        scheduler.load_state_dict(resumed.scheduler)
        scaler.load_state_dict(resumed.scaler)
        step = resumed.step
        batches.skip(resumed.position.epoch, resumed.position.index)
        resumed.position.random.restore(collator)
        if world.is_main:
            metrics.write(
                event="resume",
                checkpoint=str(resumed.path),
                step=step,
                epoch=resumed.position.epoch,
                index=resumed.position.index,
            )
    checkpoints = Checkpointer(
        model=model,
        optimiser=optimiser,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        out_dir=out_dir,
        world=world,
    )

    trained.train()
    started = time.monotonic()
    # A segment logs its own first step whatever the interval, so the first loss
    # in a metrics file is always the first step that file's segment ran.
    first_step = step + 1
    for batch in batches:
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = trained(batch)
        if output.loss is None:
            raise RuntimeError("a batch reached the loop with no scored position")
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimiser)
        scaler.update()
        optimiser.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        if world.is_main and (step % config.log_every == 0 or step == first_step):
            record = {
                "event": "step",
                "step": step,
                "loss": float(output.loss.detach()),
                "lr": scheduler.get_last_lr()[0],
                "new_lr": scheduler.get_last_lr()[1],
                "examples": batch.size,
                "tokens": batch.tokens,
                "context_tokens": batch.context_tokens,
                "seconds": round(time.monotonic() - started, 1),
                "gates": model.gates(),
            }
            metrics.write(**record)
            log.info("step", **{k: v for k, v in record.items() if k != "event"})
        if step % config.checkpoint_every == 0:
            checkpoints.numbered(step, batches)
        if step >= config.max_steps:
            break
    checkpoints.final(step, batches)
    metrics.close()
    world.stop()
    return metrics.path


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
            "route_a": asdict(model.config),
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
    """Leave only the newest *keep* numbered checkpoints; ``checkpoint-final`` stays."""
    if keep < 1:
        raise ValueError(f"a run must keep at least one checkpoint, got {keep}")
    numbered = sorted(out_dir.glob("checkpoint-[0-9]*.pt"), key=checkpoint_step)
    for path in numbered[:-keep]:
        path.unlink()
        log.info("checkpoint rotated out", path=str(path))


def gather_positions(world: Distributed, batches: EpochBatches) -> list[dict[str, Any]]:
    """Every rank's place in its own stream, gathered so rank 0 can write them all.

    Collective: every rank calls it, including the ones that write nothing, which
    is why it is not inside an ``is_main`` guard.
    """
    mine = RankState.capture(batches).as_record()
    if world.world_size == 1:
        return [mine]
    gathered: list[Any] = [None] * world.world_size
    torch.distributed.all_gather_object(gathered, mine)
    missing = [rank for rank, record in enumerate(gathered) if record is None]
    if missing:
        raise RuntimeError(f"these ranks did not report a stream position: {missing}")
    return [dict(record) for record in gathered]


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

    def numbered(self, step: int, batches: EpochBatches) -> None:
        """Write ``checkpoint-<step>`` and rotate the older ones out."""
        self._write(self.out_dir / f"checkpoint-{step:06d}.pt", step, batches)
        if self.world.is_main:
            rotate_checkpoints(self.out_dir, self.config.keep_checkpoints)

    def final(self, step: int, batches: EpochBatches) -> None:
        """Write ``checkpoint-final``, which is never rotated out."""
        self._write(self.out_dir / "checkpoint-final.pt", step, batches)

    def _write(self, path: Path, step: int, batches: EpochBatches) -> None:
        """Gather the positions -- all ranks -- and write the file on rank 0."""
        positions = gather_positions(self.world, batches)
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


def refuse_mismatch(
    path: Path, kind: str, saved: Mapping[str, Any], wanted: Mapping[str, Any]
) -> None:
    """Refuse to resume when the *kind* record differs, naming every field that does.

    Continuing a run under a different schedule is not a resume: the cosine bends
    somewhere else, the warmup is over a different length, and the two segments'
    metrics do not describe one run. The comparison is over every field, so the
    refusal names what to change rather than saying only that something did.
    """
    differing = sorted(key for key in set(saved) | set(wanted) if saved.get(key) != wanted.get(key))
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
    position: RankState

    @classmethod
    def read(
        cls, path: Path, config: TrainingConfig, route: RouteAConfig, world: Distributed
    ) -> Resumption:
        """Load *path* and refuse it unless it is this run, one segment earlier."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        absent = sorted({"optimiser", "scheduler", "scaler", "positions"} - set(state))
        if absent:
            raise ValueError(
                f"{path} predates resumable training and has no {absent}; "
                "it can be trained from, but not resumed"
            )
        refuse_mismatch(path, "training", state["training"], asdict(config))
        refuse_mismatch(path, "route_a", state["route_a"], asdict(route))
        positions = state["positions"]
        if len(positions) != world.world_size:
            raise ValueError(
                f"{path} was written by a world of {len(positions)} ranks and this one has "
                f"{world.world_size}; a rank would read shards it has no position for"
            )
        step = int(state["step"])
        if step >= config.max_steps:
            raise ValueError(
                f"{path} is already at step {step} of {config.max_steps}; "
                "raise max_steps to run further, or this resume would do nothing"
            )
        return cls(
            path=path,
            step=step,
            model=state["model"],
            optimiser=state["optimiser"],
            scheduler=state["scheduler"],
            scaler=state["scaler"],
            position=RankState.from_record(positions[world.rank]),
        )
