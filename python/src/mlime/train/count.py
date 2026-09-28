"""How many steps an epoch is worth, counted rather than guessed.

``max_steps`` is not a knob that can be approximately right. The cosine reaches
zero exactly at it, so a run given a number a third too large stops while its
learning rate is still high, and one given a number too small spends its last
thousand steps at a rate of nothing. The plan for v2 is "two epochs", and two
epochs is a count of batches, not of examples: a batch holds as many examples as
the token budget leaves room for, which depends on how long the sentences are
and how long the text before them is.

So the number is produced the only way it can be trusted -- by replaying exactly
what the loop will do, over exactly the shards it will read, at exactly the
epochs it will read them, and counting the groups. Nothing is collated and no
model is built: this walks the corpus, and on the full corpus it is minutes.

Every rank is counted rather than measuring one and multiplying: shards are
dealt out whole, they are not the same size, and the run ends when the
*shortest* rank runs out of its epochs -- every other rank stops there too,
because a step is one all-reduce and a rank with nothing left to contribute is a
rank the others wait for. The ranks count concurrently, one process each; a
rank's own walk stays sequential because the batcher's groups reach across
shard boundaries, so a per-shard count would be a different count.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from mlime.logging import log
from mlime.train.loop import epoch_groups
from mlime.train.run import RunPaths, Slices, Vocabularies
from mlime.train.samples import DEFAULT_CONTEXT_TOKENS, Augmentation

#: The format of the census this module writes. The number moves when how the
#: count is taken changes -- version 2 is the first in which each rank counts
#: with its own builder, the draws the training loop's ranks actually make;
#: version 1 walked the ranks off one shared builder. A stored census without
#: the field is read as version 0, and only a census of this version is reused.
CENSUS_VERSION = 2


@dataclass(frozen=True)
class RankBatches:
    """What one rank's share of the corpus comes to, epoch by epoch."""

    rank: int
    #: Batches in each epoch, indexed by epoch.
    epochs: list[int]
    total: int


@dataclass(frozen=True)
class BatchCensus:
    """What the count found, and everything it depended on.

    The settings are recorded beside the numbers because the numbers are only
    true of those settings: a different token budget, a different seed or a
    different set of shards is a different count, and a ``max_steps`` copied out
    of a file that does not say which is a number nobody can check.
    """

    world_size: int
    epochs: int
    token_budget: int
    max_context_tokens: int
    seed: int
    augmentation: dict[str, float]
    shards: list[str]
    ranks: list[RankBatches]
    #: Micro-batches an optimiser step consumes. The rank totals count
    #: micro-batches; the step count is that total divided by this.
    accumulate: int
    #: Steps a run of these epochs takes: the shortest rank's total in
    #: optimiser steps, because that is where every rank stops.
    steps_for_epochs: int
    #: The format this count was taken in; see :data:`CENSUS_VERSION`.
    census_version: int
    build_counts: dict[str, int]

    def write(self, path: Path) -> None:
        """Write the census as JSON, which is how the kernel reads it back."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("census written", path=str(path), steps=self.steps_for_epochs)

    @classmethod
    def read(cls, path: Path) -> BatchCensus:
        """The census a JSON file of :meth:`write` holds.

        A file written before censuses were versioned reads as version 0, so
        the reuse check refuses it rather than taking a count of unknown
        provenance.
        """
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        data["ranks"] = [RankBatches(**rank) for rank in data["ranks"]]
        data.setdefault("census_version", 0)
        return cls(**data)


@dataclass(frozen=True)
class CountSpec:
    """The configuration a stored census has to match before it may be reused.

    Every field here is a field the count depended on: a census taken under
    different settings counts different batches, and a stored one is only ever
    used when all of them agree.
    """

    world_size: int
    epochs: int
    token_budget: int
    max_context_tokens: int
    seed: int
    augmentation: dict[str, float]
    shards: list[str]
    accumulate: int
    census_version: int


def census_mismatches(census: BatchCensus, spec: CountSpec) -> list[str]:
    """The names of the fields on which *census* and *spec* disagree."""
    wanted = asdict(spec)
    actual = asdict(census)
    return [name for name, value in wanted.items() if actual.get(name) != value]


def matching_census(path: Path, spec: CountSpec) -> tuple[BatchCensus | None, list[str]]:
    """The census stored at *path* when it describes *spec*, else the differing fields.

    "Matches" means every field equal, the ``shards`` list included: a census
    counted over a different corpus or different budget is not a shortened count,
    it is a wrong count, and refusing it loudly is what makes the reuse safe.

    A file whose shape is not this dataclass's -- fields missing, fields added,
    or values of a shape the constructor refuses -- is refused the same way,
    with the differing names reported; it is never trusted enough to raise on.
    """
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, ["<unreadable>"]
    declared = {field.name for field in fields(BatchCensus)}
    differed = sorted(set(data) ^ declared)
    if differed:
        return None, differed
    try:
        stored = BatchCensus.read(path)
    except (TypeError, KeyError, ValueError):
        return None, ["ranks"]
    differed = census_mismatches(stored, spec)
    if differed:
        return None, differed
    return stored, []


@dataclass(frozen=True)
class _RankContext:
    """What a rank-counting worker needs, pickled to it as a pool task argument."""

    vocabularies: Vocabularies
    paths: RunPaths
    shards: Sequence[str]
    token_budget: int
    epochs: int
    seed: int
    augmentation: Augmentation
    max_context_tokens: int
    world_size: int


def _count_rank(context: _RankContext, rank: int) -> tuple[RankBatches, dict[str, int]]:
    """One rank's count over its own shards, and the augmentation it drew.

    The builder is built fresh here -- per rank and per process -- exactly as
    the training loop does it in its own rank's process, so the augmentation
    draws the count walks are the same draws the run will make.
    """
    collator = context.vocabularies.collator(0.0, context.max_context_tokens, context.seed)
    builder = context.vocabularies.builder(context.augmentation, context.seed)
    stream = context.vocabularies.stream(
        context.paths, context.shards, builder, rank, context.world_size
    )
    counted = [
        sum(1 for _ in epoch_groups(stream, collator, context.token_budget, epoch))
        for epoch in range(context.epochs)
    ]
    log.info("rank counted", rank=rank, epochs=counted, total=sum(counted))
    return (
        RankBatches(rank=rank, epochs=counted, total=sum(counted)),
        builder.counts.as_dict(),
    )


def _count_ranks(context: _RankContext) -> list[tuple[RankBatches, dict[str, int]]]:
    """Every rank's count, one rank per process where there is more than one.

    Within a rank the count stays sequential over the stream: batches are
    groups of consecutive examples, so a group can hold the end of one shard
    and the start of the next -- and the builder's draws depend on every
    example before them -- which makes per-shard counting a different count.

    The pool is spawned, not forked: a worker forked after polars' thread pool
    has run inherits locks the dead threads held and hangs on its first read.
    """
    if context.world_size < 2:
        return [_count_rank(context, rank) for rank in range(context.world_size)]
    workers = min(context.world_size, os.cpu_count() or 1)
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.starmap(_count_rank, ((context, rank) for rank in range(context.world_size)))


def count_batches(
    vocabularies: Vocabularies,
    paths: RunPaths,
    slices: Slices,
    token_budget: int,
    seed: int = 0,
    augmentation: Augmentation | None = None,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    world_size: int = 1,
    epochs: int = 1,
    accumulate: int = 1,
    reuse_census: Path | None = None,
) -> BatchCensus:
    """Count the batches every rank takes over *epochs*, the way the loop would.

    *vocabularies* is passed in rather than loaded because it is what decides
    which examples survive -- a character the base model cannot emit is dropped,
    and dropping it changes the count -- so the count has to be taken against the
    same tables the run will use, not against tables it built for itself.

    *reuse_census* is a stored count from a previous run: it is used only when
    every configuration field and the shard list equal this call's, and the
    fields that differed are logged when it is refused.
    """
    if world_size < 1:
        raise ValueError(f"a world has at least one rank, got {world_size}")
    if epochs < 1:
        raise ValueError(f"a count covers at least one epoch, got {epochs}")
    if accumulate < 1:
        raise ValueError(f"an optimiser step holds at least one micro-batch, got {accumulate}")
    augmentation = augmentation or Augmentation()
    spec = CountSpec(
        world_size=world_size,
        epochs=epochs,
        token_budget=token_budget,
        max_context_tokens=max_context_tokens,
        seed=seed,
        augmentation=asdict(augmentation),
        shards=list(slices.train),
        accumulate=accumulate,
        census_version=CENSUS_VERSION,
    )
    if reuse_census is not None:
        stored, differed = matching_census(reuse_census, spec)
        if stored is not None:
            log.info("census reused", path=str(reuse_census), steps=stored.steps_for_epochs)
            return stored
        log.info("mounted census refused", path=str(reuse_census), differed=differed)
    context = _RankContext(
        vocabularies=vocabularies,
        paths=paths,
        shards=slices.train,
        token_budget=token_budget,
        epochs=epochs,
        seed=seed,
        augmentation=augmentation,
        max_context_tokens=max_context_tokens,
        world_size=world_size,
    )
    counted = _count_ranks(context)
    ranks = [rank for rank, _ in counted]
    tallies: Counter[str] = Counter()
    for _, counts in counted:
        tallies.update(counts)
    census = BatchCensus(
        world_size=world_size,
        epochs=epochs,
        token_budget=token_budget,
        max_context_tokens=max_context_tokens,
        seed=seed,
        augmentation=asdict(augmentation),
        shards=list(slices.train),
        ranks=ranks,
        accumulate=accumulate,
        steps_for_epochs=-(-min(rank.total for rank in ranks) // accumulate),
        census_version=CENSUS_VERSION,
        build_counts=dict(tallies),
    )
    log.info(
        "batches counted",
        steps_for_epochs=census.steps_for_epochs,
        per_rank=[rank.total for rank in census.ranks],
        epochs=epochs,
        shards=len(census.shards),
        **census.build_counts,
    )
    return census
