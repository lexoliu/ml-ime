"""Summarise a dual-annotation run: the measured ceiling on every later number.

Where g2pW and the LLM disagree, one of them is wrong, and there is no way to
tell which without a human. Those sentences are excluded from training, so the
disagreement rate is a direct loss of data -- and because the same annotation
built the evaluation set, it is also the accuracy no model can be shown to
exceed. Reporting it per frequency band matters because rare characters are
where disagreement concentrates, and rare characters are a small share of tokens
but a large share of the sentences a user notices going wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

from .g2p import ANNOTATED, REFUSED
from .shards import read_shards, shard_paths

#: Rank boundaries between character-frequency bands, most frequent first.
FREQUENCY_BREAKS = (500, 1500, 3500)

BUCKET_LABELS = ("top 500", "501-1500", "1501-3500", "rarer")

#: How many disagreeing characters the report lists.
TOP_DISAGREEMENTS = 30


@dataclass(frozen=True)
class AgreementSummary:
    """The headline numbers of one annotation run."""

    sentences: int
    characters: int
    sentences_agreed: int
    characters_agreed: int
    refusals: int

    @property
    def sentence_rate(self) -> float:
        """Share of sentences both annotators read identically end to end."""
        return self.sentences_agreed / self.sentences if self.sentences else 0.0

    @property
    def character_rate(self) -> float:
        """Share of character positions the two annotators spell the same way."""
        return self.characters_agreed / self.characters if self.characters else 0.0


def per_character(annotated: pl.DataFrame) -> pl.DataFrame:
    """One row per character position, carrying both readings and the verdict."""
    return annotated.select("id", "text", "characters", "g2pw", "llm", "agree").explode(
        ["characters", "g2pw", "llm", "agree"], empty_as_null=False
    )


def summarise(annotated: pl.DataFrame, refusals: int) -> AgreementSummary:
    """Count sentences and character positions, agreed and total."""
    characters = per_character(annotated)
    return AgreementSummary(
        sentences=annotated.height,
        characters=characters.height,
        sentences_agreed=int(annotated["agree_all"].sum()),
        characters_agreed=int(characters["agree"].sum()),
        refusals=refusals,
    )


def by_frequency(characters: pl.DataFrame) -> pl.DataFrame:
    """Agreement rate per character-frequency band, most frequent band first."""
    ranked = (
        characters.group_by("characters")
        .len()
        .sort("len", descending=True)
        .with_row_index("rank", offset=1)
        .with_columns(
            pl.col("rank").cut(list(FREQUENCY_BREAKS), labels=list(BUCKET_LABELS)).alias("band")
        )
        .select("characters", "band")
    )
    return (
        characters.join(ranked, on="characters")
        .group_by("band")
        .agg(
            pl.len().alias("positions"),
            pl.col("characters").n_unique().alias("distinct"),
            pl.col("agree").mean().alias("rate"),
        )
        .sort("band")
    )


def worst_characters(characters: pl.DataFrame, limit: int = TOP_DISAGREEMENTS) -> pl.DataFrame:
    """The characters the annotators fight over most, with one example each."""
    return (
        characters.filter(~pl.col("agree"))
        .group_by("characters")
        .agg(
            pl.len().alias("disagreements"),
            pl.col("g2pw").first().alias("g2pw_example"),
            pl.col("llm").first().alias("llm_example"),
            pl.col("text").first().alias("sentence"),
        )
        .sort("disagreements", descending=True)
        .head(limit)
    )


def render(annotated: pl.DataFrame, refusals: int, console: Console | None = None) -> None:
    """Print the three tables that make up the report."""
    console = console or Console()
    summary = summarise(annotated, refusals)
    characters = per_character(annotated)

    headline = Table(title="g2p dual annotation")
    headline.add_column("Measure")
    headline.add_column("Value", justify="right")
    headline.add_row("Sentences annotated", f"{summary.sentences:,}")
    headline.add_row("Sentences fully agreed", f"{summary.sentences_agreed:,}")
    headline.add_row("Sentence agreement", f"{summary.sentence_rate:.2%}")
    headline.add_row("Character positions", f"{summary.characters:,}")
    headline.add_row("Character agreement", f"{summary.character_rate:.2%}")
    headline.add_row("Refusals recorded", f"{summary.refusals:,}")
    console.print(headline)

    bands = Table(title="Agreement by character frequency")
    for column in ("Band", "Distinct chars", "Positions", "Agreement"):
        bands.add_column(column, justify="right" if column != "Band" else "left")
    for row in by_frequency(characters).iter_rows(named=True):
        bands.add_row(
            str(row["band"]),
            f"{row['distinct']:,}",
            f"{row['positions']:,}",
            f"{row['rate']:.2%}",
        )
    console.print(bands)

    worst = Table(title=f"Top {TOP_DISAGREEMENTS} disagreeing characters")
    for column in ("Char", "Count", "g2pW", "LLM", "Example"):
        worst.add_column(column)
    for row in worst_characters(characters).iter_rows(named=True):
        worst.add_row(
            row["characters"],
            str(row["disagreements"]),
            row["g2pw_example"],
            row["llm_example"],
            row["sentence"],
        )
    console.print(worst)


def report(directory: Path, console: Console | None = None) -> None:
    """Read an annotation directory and print its report."""
    refusals = sum(pl.read_parquet(path).height for path in shard_paths(directory, REFUSED))
    render(read_shards(directory, ANNOTATED), refusals, console)
