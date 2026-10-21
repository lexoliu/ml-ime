"""The shape of a decode's top-1 answers, and how rare its substituted characters are.

The ranking stages (#116, #121, #130, #132) move answers between three kinds:
right, wrong at the expected length, and wrong at another length. A loss that
fixes segmentation turns wrong-length answers into the other two; one that
drifts toward rare characters turns right answers into right-length wrong ones
whose characters are uncommon. This module counts the three kinds over a
``--dump`` file's slice and measures that drift.

Rarity is read off the six local run3 sample shards: the characters of their
``text`` column, counted. A character's frequency rank is its index in that
count sorted by frequency, most frequent first (rank 0), with ties kept in the
order the characters first appear when the shards are read in name order.
A character the sample never holds ranks after every one it does, at the
number of distinct characters. The drift is measured over the characters of
right-length wrong answers that differ from the expected text position by
position: the share of them unseen in the sample, and their median rank.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mlime.gui import read_eval_set, slice_indices
from mlime.rescore import DumpedRecord


@dataclass(frozen=True)
class FrequencyRanks:
    """Each sampled character's frequency rank, and the size of the sample."""

    ranks: dict[str, int]
    characters: int

    @classmethod
    def from_shards(cls, shards: Sequence[Path]) -> FrequencyRanks:
        """Count the ``text`` column of *shards*, read in name order."""
        if not shards:
            raise ValueError("no sample shards to count characters over")
        counts: Counter[str] = Counter()
        for shard in sorted(shards, key=lambda path: path.name):
            for text in pl.read_parquet(shard, columns=["text"])["text"]:
                counts.update(text)
        # `most_common` sorts stably, so ties keep first-appearance order.
        ranks = {character: rank for rank, (character, _) in enumerate(counts.most_common())}
        return cls(ranks=ranks, characters=counts.total())

    def rank(self, character: str) -> int:
        """*character*'s rank; an unseen one ranks after every seen one."""
        return self.ranks.get(character, len(self.ranks))


@dataclass(frozen=True)
class Drift:
    """The top-1 answers of one slice, by kind, and the rarity of the substitutions."""

    right: int
    same_length: int
    other_length: int
    substituted: int
    unseen: int
    median_rank: float

    @property
    def unseen_share(self) -> float:
        """The share of substituted characters the sample never holds."""
        return self.unseen / self.substituted

    def render(self) -> str:
        """One table row's worth: the kinds, the unseen share and the median rank."""
        return "\n".join(
            [
                f"right / same length / other length  "
                f"{self.right:,} / {self.same_length:,} / {self.other_length:,}",
                f"substituted characters              {self.substituted:,}",
                f"unseen in the sample                {self.unseen_share:.1%} ({self.unseen:,})",
                f"median frequency rank               {self.median_rank:g}",
            ]
        )


def measure(
    records: Sequence[DumpedRecord],
    eval_set: Path,
    ranks: FrequencyRanks,
    slice_: str,
    dev_share: float,
) -> Drift:
    """The drift over the records of *slice_*; every record of it must be in *records*.

    A record with no hypotheses answers the empty string, which is a
    wrong-length answer, as the evaluation's report counts it unanswered.
    """
    wanted = set(slice_indices(read_eval_set(eval_set), slice_, dev_share))
    found = {record.index: record for record in records}
    missing = wanted - found.keys()
    if missing:
        raise ValueError(
            f"the dump lacks {len(missing)} records of the {slice_} slice, "
            f"the first {min(missing)}; it was decoded over another slice"
        )
    right = same_length = other_length = 0
    substituted: list[int] = []
    for index in sorted(wanted):
        record = found[index]
        top = record.hypotheses[0].text if record.hypotheses else ""
        if top == record.text:
            right += 1
        elif len(top) == len(record.text):
            same_length += 1
            substituted.extend(
                ranks.rank(got) for got, want in zip(top, record.text, strict=True) if got != want
            )
        else:
            other_length += 1
    if not substituted:
        raise ValueError(f"the {slice_} slice has no right-length wrong answers to measure")
    return Drift(
        right=right,
        same_length=same_length,
        other_length=other_length,
        substituted=len(substituted),
        unseen=sum(1 for rank in substituted if rank == len(ranks.ranks)),
        median_rank=statistics.median(substituted),
    )
