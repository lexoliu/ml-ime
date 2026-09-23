"""Annotate target sentences with pinyin twice, and keep only what both agree on.

Every accuracy number this project will ever report is bounded by the quality of
its pinyin labels: a training pair whose reading is wrong teaches the wrong
mapping, and an *evaluation* pair whose reading is wrong makes a correct model
look broken. Polyphones are where that bites -- 得, 朝, 还, 重 -- and they are
exactly the characters a single automatic labeller gets wrong.

So two independent annotators run over every sentence: g2pW, a BERT-based
disambiguator, and an LLM prompted per sentence. Where they agree, the label is
used. Where they disagree, the sentence is set aside as a hard-polyphone case
rather than silently averaged away, and the disagreement rate published by
``mlime g2p report`` is the measured ceiling on everything downstream.

Comparison is on the *toneless* syllable. The input method converts what someone
types, and no pinyin keyboard has a tone key, so a tone disagreement costs the
model nothing; tones are still carried in the output columns because they are
what makes a disagreement interpretable.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mlime.logging import log

from .corpus import Sample
from .shards import ShardWriter, read_shards
from .text import han_characters, toneless

#: Shard prefix for sentences both annotators handled, agreeing or not.
ANNOTATED = "annotated"

#: Shard prefix for the subset they disagreed on -- the hard-polyphone eval set.
HARD = "hard"

#: Shard prefix for sentences an annotator could not label at all.
REFUSED = "refused"

ANNOTATED_SCHEMA = pl.Schema(
    {
        "id": pl.String(),
        "source": pl.String(),
        "text": pl.String(),
        "context": pl.String(),
        "characters": pl.List(pl.String()),
        "g2pw": pl.List(pl.String()),
        "llm": pl.List(pl.String()),
        "agree": pl.List(pl.Boolean()),
        "agree_all": pl.Boolean(),
    }
)

REFUSED_SCHEMA = pl.Schema(
    {
        "id": pl.String(),
        "text": pl.String(),
        "annotator": pl.String(),
        "reason": pl.String(),
    }
)


@dataclass(frozen=True)
class Reading:
    """One annotator's tone-numbered syllable for each Han character of a sentence."""

    syllables: tuple[str, ...]


@dataclass(frozen=True)
class Refusal:
    """Why an annotator produced nothing usable for a sentence.

    A refusal is recorded and counted rather than skipped, so a run that quietly
    lost half its sentences to a broken endpoint cannot look like a clean run.
    """

    reason: str


#: What an annotator returns for one sentence.
Outcome = Reading | Refusal


class Annotator(ABC):
    """A source of per-character pinyin for a batch of sentences."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Column name this annotator's readings are stored under."""

    @abstractmethod
    async def annotate(self, texts: Sequence[str]) -> list[Outcome]:
        """One outcome per input sentence, in order."""


@dataclass(frozen=True)
class Comparison:
    """Two annotators' readings for one sentence, position by position."""

    characters: tuple[str, ...]
    first: tuple[str, ...]
    second: tuple[str, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.characters), len(self.first), len(self.second)}
        if len(lengths) != 1:
            raise ValueError(
                f"cannot compare readings of different lengths: {sorted(lengths)} for "
                f"{''.join(self.characters)!r}"
            )

    @property
    def agree(self) -> tuple[bool, ...]:
        """Per position, whether the two annotators spell the syllable the same way."""
        return tuple(
            toneless(a) == toneless(b) for a, b in zip(self.first, self.second, strict=True)
        )

    @property
    def unanimous(self) -> bool:
        """Whether every position agrees. Sentences that do form the training set."""
        return all(self.agree)


def compare(text: str, first: Reading, second: Reading) -> Comparison:
    """Line up two readings against the Han characters of *text*."""
    return Comparison(tuple(han_characters(text)), first.syllables, second.syllables)


@dataclass
class AnnotationCounts:
    """How an annotation run went, in the three ways it can go."""

    agreed: int = 0
    disagreed: int = 0
    refused: int = 0

    @property
    def agreement_rate(self) -> float:
        """Share of fully-annotated sentences both annotators read identically."""
        total = self.agreed + self.disagreed
        return self.agreed / total if total else 0.0


async def annotate(
    samples: Iterable[Sample],
    first: Annotator,
    second: Annotator,
    out_dir: Path,
    batch_size: int = 32,
    rows_per_shard: int = 50_000,
) -> AnnotationCounts:
    """Run both annotators over *samples* and write the agreed, hard and refused shards."""
    counts = AnnotationCounts()
    with (
        ShardWriter(out_dir, ANNOTATED, ANNOTATED_SCHEMA, rows_per_shard) as annotated,
        ShardWriter(out_dir, HARD, ANNOTATED_SCHEMA, rows_per_shard) as hard,
        ShardWriter(out_dir, REFUSED, REFUSED_SCHEMA, rows_per_shard) as refused,
    ):
        for batch in _batched(samples, batch_size):
            texts = [sample.text for sample in batch]
            outcomes = await asyncio.gather(first.annotate(texts), second.annotate(texts))
            for sample, first_outcome, second_outcome in zip(batch, *outcomes, strict=True):
                readings = []
                for annotator, outcome in ((first, first_outcome), (second, second_outcome)):
                    if isinstance(outcome, Refusal):
                        counts.refused += 1
                        refused.write(
                            {
                                "id": sample.id,
                                "text": sample.text,
                                "annotator": annotator.name,
                                "reason": outcome.reason,
                            }
                        )
                    else:
                        readings.append(outcome)
                if len(readings) != 2:
                    continue
                comparison = compare(sample.text, readings[0], readings[1])
                record: dict[str, object] = {
                    "id": sample.id,
                    "source": sample.source,
                    "text": sample.text,
                    "context": sample.context,
                    "characters": list(comparison.characters),
                    first.name: list(comparison.first),
                    second.name: list(comparison.second),
                    "agree": list(comparison.agree),
                    "agree_all": comparison.unanimous,
                }
                annotated.write(record)
                if comparison.unanimous:
                    counts.agreed += 1
                else:
                    counts.disagreed += 1
                    hard.write(record)
            log.info("annotated", **vars(counts))
    return counts


def _batched(samples: Iterable[Sample], size: int) -> Iterator[list[Sample]]:
    """Group *samples* into lists of at most *size*."""
    batch: list[Sample] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_annotated(directory: Path, prefix: str = ANNOTATED) -> pl.DataFrame:
    """Read the shards written by :func:`annotate`."""
    return read_shards(directory, prefix)
