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

Ranks are counted one by one rather than measured on one rank and multiplied.
Shards are dealt out whole, they are not the same size, and the run ends when the
*shortest* rank runs out of its epochs -- every other rank stops there too,
because a step is one all-reduce and a rank with nothing left to contribute is a
rank the others wait for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mlime.logging import log
from mlime.train.loop import epoch_groups
from mlime.train.run import RunPaths, Slices, Vocabularies
from mlime.train.samples import DEFAULT_CONTEXT_TOKENS, Augmentation


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
    #: Steps a run of these epochs takes: the shortest rank's total, because that
    #: is where every rank stops.
    steps_for_epochs: int
    build_counts: dict[str, int]

    def write(self, path: Path) -> None:
        """Write the census as JSON, which is how the kernel reads it back."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("census written", path=str(path), steps=self.steps_for_epochs)


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
) -> BatchCensus:
    """Count the batches every rank takes over *epochs*, the way the loop would.

    *vocabularies* is passed in rather than loaded because it is what decides
    which examples survive -- a character the base model cannot emit is dropped,
    and dropping it changes the count -- so the count has to be taken against the
    same tables the run will use, not against tables it built for itself.
    """
    if world_size < 1:
        raise ValueError(f"a world has at least one rank, got {world_size}")
    if epochs < 1:
        raise ValueError(f"a count covers at least one epoch, got {epochs}")
    augmentation = augmentation or Augmentation()
    # The collator is here for its context width, which decides how much of a
    # step's rectangle each example claims. Nothing is collated, so its dropout
    # is never drawn from.
    collator = vocabularies.collator(0.0, max_context_tokens, seed)
    builder = vocabularies.builder(augmentation, seed)
    ranks: list[RankBatches] = []
    for rank in range(world_size):
        stream = vocabularies.stream(paths, slices.train, builder, rank, world_size)
        counted = [
            sum(1 for _ in epoch_groups(stream, collator, token_budget, epoch))
            for epoch in range(epochs)
        ]
        log.info("rank counted", rank=rank, epochs=counted, total=sum(counted))
        ranks.append(RankBatches(rank=rank, epochs=counted, total=sum(counted)))
    census = BatchCensus(
        world_size=world_size,
        epochs=epochs,
        token_budget=token_budget,
        max_context_tokens=max_context_tokens,
        seed=seed,
        augmentation=asdict(augmentation),
        shards=list(slices.train),
        ranks=ranks,
        steps_for_epochs=min(rank.total for rank in ranks),
        build_counts=builder.counts.as_dict(),
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
