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
rank the others wait for.

Almost all of the count is building examples, and an example depends on its
seed, its epoch and its sample id alone, so the shards are built in parallel:
each worker process reads a run of shards once and builds every row of them for
the rank that owns it, at every epoch, keeping only each kept example's widths
and rows. The grouping cannot be split -- a batch reaches across shard
boundaries, and whether an example opens one depends on the batch before it --
so this process replays it over those shapes in reading order, with the loop's
own :class:`~mlime.train.samples.BudgetPacker`. The census is the same at any
number of workers.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
from array import array
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any

from mlime.logging import log
from mlime.train.e2e_model import RankingConfig
from mlime.train.ranking import RankCollator
from mlime.train.run import RunPaths, Slices, Vocabularies
from mlime.train.samples import (
    DEFAULT_CONTEXT_TOKENS,
    Augmentation,
    BatchCollator,
    BudgetPacker,
    example_shape,
    first_row,
    read_pair,
)

#: The format of the census this module writes. The number moves when how the
#: count is taken changes -- version 2 is the first in which each rank counts
#: with its own builder, the draws the training loop's ranks actually make;
#: version 1 walked the ranks off one shared builder; version 3 dealt whole
#: shards to ranks by their text rather than round-robin; version 4 strides the
#: rows of the interleaved shards across readers (`CorpusStream`). A stored
#: census without the field is read as version 0, and only a census of this
#: version is reused.
CENSUS_VERSION = 4


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
    #: The ranking search's shape when the count weighed each example by
    #: every row the ranking step makes of it (see :func:`ranking_shape`);
    #: ``None`` for a plain run.
    ranking: dict[str, Any] | None = None

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
    ranking: dict[str, Any] | None


def ranking_shape(ranking: RankingConfig | None) -> dict[str, Any] | None:
    """The ranking settings a count depends on: the search's shape, not its weight.

    The weight only scales the loss, so runs that differ in nothing else --
    a weight sweep -- share one census.
    """
    if ranking is None:
        return None
    shape = asdict(ranking)
    del shape["weight"]
    return shape


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


#: Runs of shards dealt to each worker. More runs even out shards of unequal
#: size; each run costs one pickled copy of the vocabularies (some 4 MB).
RUNS_PER_WORKER = 8


@dataclass(frozen=True)
class _CountContext:
    """What a counting worker needs, pickled to it with each run of shards."""

    vocabularies: Vocabularies
    labels: Path
    epochs: int
    seed: int
    augmentation: Augmentation
    max_context_tokens: int
    world_size: int
    ranking: RankingConfig | None
    segmenter: Path | None


@dataclass(frozen=True)
class _ShardTask:
    """One sample shard, and the global row it starts at in reading order."""

    path: Path
    offset: int


#: The fill widths, context widths and rows of the examples one rank keeps
#: from one shard, in reading order: all the grouping reads of them.
_Shapes = tuple["array[int]", "array[int]", "array[int]"]


@dataclass(frozen=True)
class _ShardCount:
    """What one shard comes to: ``shapes[epoch][rank]``, and the builder's counters."""

    shapes: list[list[_Shapes]]
    build_counts: dict[str, int]


def _collator(context: _CountContext) -> BatchCollator:
    """The collator whose context budget and row weights the run batches by."""
    plain = context.vocabularies.collator(0.0, context.max_context_tokens, context.seed)
    if context.ranking is None:
        return plain
    if context.segmenter is None:
        raise ValueError("a ranking count reads the segmenter the run will read")
    return RankCollator.launch(
        plain, context.segmenter, context.vocabularies.spans, context.ranking
    )


def _count_shard(context: _CountContext, collator: BatchCollator, task: _ShardTask) -> _ShardCount:
    """Every rank's examples of one shard at every epoch, as shapes.

    The shard is read once and each row built by the rank that owns it -- the
    rows :class:`~mlime.train.samples.CorpusStream` would hand that rank --
    with a builder of the run's seed, so the draws are the run's draws.
    """
    builder = context.vocabularies.builder(context.augmentation, context.seed)
    pairs = list(read_pair(task.path, context.labels))
    world = context.world_size
    shapes: list[list[_Shapes]] = []
    for epoch in range(context.epochs):
        ranks: list[_Shapes] = []
        for rank in range(world):
            fills, contexts, rows = array("H"), array("H"), array("H")
            for sample, syllables in pairs[first_row(task.offset, rank, world) :: world]:
                example = builder.build(sample, syllables, epoch)
                if example is not None:
                    fill, width = example_shape(example, collator.max_context_tokens)
                    fills.append(fill)
                    contexts.append(width)
                    rows.append(collator.rows(example))
            ranks.append((fills, contexts, rows))
        shapes.append(ranks)
    return _ShardCount(shapes=shapes, build_counts=builder.counts.as_dict())


def _count_run(context: _CountContext, run: Sequence[_ShardTask]) -> list[_ShardCount]:
    """A worker's unit: a run of consecutive shards, under one collator."""
    collator = _collator(context)
    return [_count_shard(context, collator, task) for task in run]


def _runs(tasks: Sequence[_ShardTask], workers: int) -> list[Sequence[_ShardTask]]:
    """*tasks* cut into consecutive runs, about :data:`RUNS_PER_WORKER` a worker."""
    size = max(1, math.ceil(len(tasks) / (workers * RUNS_PER_WORKER)))
    return [tasks[start : start + size] for start in range(0, len(tasks), size)]


def _in_order(
    count: Callable[[Sequence[_ShardTask]], list[_ShardCount]],
    runs: Sequence[Sequence[_ShardTask]],
    workers: int,
) -> Iterator[list[_ShardCount]]:
    """Each run's counts, in reading order, from *workers* processes.

    One worker counts in this process. More are spawned, not forked: a worker
    forked after polars' thread pool has run inherits locks the dead threads
    held and hangs on its first read. A worker that fails raises here.
    """
    if workers == 1:
        yield from map(count, runs)
        return
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        yield from pool.map(count, runs)


def _count_ranks(
    context: _CountContext,
    tasks: Sequence[_ShardTask],
    token_budget: int,
    workers: int,
) -> tuple[list[RankBatches], dict[str, int]]:
    """Every rank's batches at every epoch, and the builders' counters summed.

    Each epoch's grouping starts afresh, as the loop's does, and runs over the
    rank's shapes in reading order whatever order the workers finish in.
    """
    packers = [
        [BudgetPacker(token_budget) for _ in range(context.world_size)]
        for _ in range(context.epochs)
    ]
    tallies: Counter[str] = Counter()
    runs = _runs(tasks, workers)
    counted = 0
    for run in _in_order(partial(_count_run, context), runs, workers):
        for shard in run:
            for epoch_packers, epoch_shapes in zip(packers, shard.shapes, strict=True):
                for packer, (fills, contexts, rows) in zip(
                    epoch_packers, epoch_shapes, strict=True
                ):
                    for fill, width, cost in zip(fills, contexts, rows, strict=True):
                        packer.add(fill, width, cost)
            tallies.update(shard.build_counts)
        counted += len(run)
        log.info("shards counted", counted=counted, shards=len(tasks), workers=workers)
    ranks = [
        RankBatches(
            rank=rank,
            epochs=[epoch[rank].batches for epoch in packers],
            total=sum(epoch[rank].batches for epoch in packers),
        )
        for rank in range(context.world_size)
    ]
    for rank in ranks:
        log.info("rank counted", rank=rank.rank, epochs=rank.epochs, total=rank.total)
    return ranks, dict(tallies)


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
    ranking: RankingConfig | None = None,
    segmenter: Path | None = None,
    workers: int = 1,
) -> BatchCensus:
    """Count the batches every rank takes over *epochs*, the way the loop would.

    *vocabularies* is passed in rather than loaded because it is what decides
    which examples survive -- a character the base model cannot emit is dropped,
    and dropping it changes the count -- so the count has to be taken against the
    same tables the run will use, not against tables it built for itself.

    *reuse_census* is a stored count from a previous run: it is used only when
    every configuration field and the shard list equal this call's, and the
    fields that differed are logged when it is refused.

    *ranking* counts a ranking run, whose batches weigh each example by every
    row the ranking step makes of it; *segmenter* is the binary those rows
    come from.

    *workers* is how many processes build the examples; the census does not
    depend on it.
    """
    if workers < 1:
        raise ValueError(f"a count needs at least one worker, got {workers}")
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
        ranking=ranking_shape(ranking),
    )
    if reuse_census is not None:
        stored, differed = matching_census(reuse_census, spec)
        if stored is not None:
            log.info("census reused", path=str(reuse_census), steps=stored.steps_for_epochs)
            return stored
        log.info("mounted census refused", path=str(reuse_census), differed=differed)
    context = _CountContext(
        vocabularies=vocabularies,
        labels=paths.labels,
        epochs=epochs,
        seed=seed,
        augmentation=augmentation,
        max_context_tokens=max_context_tokens,
        world_size=world_size,
        ranking=ranking,
        segmenter=segmenter,
    )
    # The stream decides the reading order and where each shard starts in it,
    # exactly as each rank's stream will in the run.
    stream = vocabularies.stream(
        paths, slices.train, vocabularies.builder(augmentation, seed), 0, world_size
    )
    tasks = [
        _ShardTask(path=path, offset=offset)
        for path, offset in zip(stream.shards, stream.offsets, strict=True)
    ]
    ranks, build_counts = _count_ranks(context, tasks, token_budget, workers)
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
        build_counts=build_counts,
        ranking=ranking_shape(ranking),
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
