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
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from mlime.logging import log
from mlime.train.model import RouteAModel, count_correct
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
    token_budget: int = 8192
    gradient_clip: float = 1.0
    seed: int = 0
    log_every: int = 10
    checkpoint_every: int = 500
    fp16: bool = True

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}")

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


def batches(stream: CorpusStream, collator: Collator, budget: int) -> Iterator[Batch]:
    """Collate the stream into token-budget batches, epoch after epoch."""
    epoch = stream.epoch
    while True:
        stream.set_epoch(epoch)
        for group in token_budget_batches(iter(stream), budget):
            yield collator(group)
        epoch += 1
        log.info("epoch finished", epoch=epoch, **stream.builder.counts.as_dict())


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
        for group in token_budget_batches(iter(examples), token_budget):
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
) -> Path:
    """Run *config.max_steps* optimiser steps and return the metrics file.

    Returns rather than prints, and writes every step's loss to the metrics file,
    because "the loss went down" is a claim that has to be checkable after the
    kernel has been torn down.
    """
    world = distributed or Distributed()
    world.start()
    seed_everything(config.seed, world.rank)
    device = world.device
    model.to(device)
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

    trained.train()
    started = time.monotonic()
    step = 0
    for batch in batches(stream, collator, config.token_budget):
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
        if world.is_main and (step % config.log_every == 0 or step == 1):
            record = {
                "event": "step",
                "step": step,
                "loss": float(output.loss.detach()),
                "lr": scheduler.get_last_lr()[0],
                "new_lr": scheduler.get_last_lr()[1],
                "examples": batch.size,
                "tokens": batch.tokens,
                "seconds": round(time.monotonic() - started, 1),
                "gates": model.gates(),
            }
            metrics.write(**record)
            log.info("step", **{k: v for k, v in record.items() if k != "event"})
        if world.is_main and step % config.checkpoint_every == 0:
            save_checkpoint(model, out_dir / f"checkpoint-{step:06d}.pt", step, config)
        if step >= config.max_steps:
            break
    if world.is_main:
        save_checkpoint(model, out_dir / "checkpoint-final.pt", step, config)
    metrics.close()
    world.stop()
    return metrics.path


def save_checkpoint(model: RouteAModel, path: Path, step: int, config: TrainingConfig) -> None:
    """Write the weights and the run's own description beside them."""
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "route_a": asdict(model.config),
            "training": asdict(config),
        },
        path,
    )
    log.info("checkpoint written", path=str(path), step=step)
