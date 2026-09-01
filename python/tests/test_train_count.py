"""Counting the batches a run will take, before it fixes its schedule.

``max_steps`` is where the cosine reaches zero, so it has to be the number of
steps the loop will really take. The counter therefore has to be the loop's own
grouping rather than a second implementation that agrees with it today: these
tests hold it to that by counting the same corpus both ways.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlime.train.count import count_batches
from mlime.train.lexicon import Lexicon
from mlime.train.loop import EpochBatches
from mlime.train.run import RunPaths, Slices, Vocabularies
from mlime.train.samples import BaseTokenizer, Collator, CorpusStream, SampleBuilder
from mlime.train.spans import SpanVocab

#: Small enough that the two-shard fixture is many batches rather than one.
BUDGET = 64
SEED = 1
SHARDS = ("test-00000.parquet", "test-00001.parquet")


def census_paths(corpus: tuple[Path, Path], tmp_path: Path) -> RunPaths:
    """The corpus, and a directory the count writes its document into."""
    samples_dir, labels_dir = corpus
    return RunPaths(
        samples=samples_dir,
        labels=labels_dir,
        char_table=tmp_path / "unused.tsv",
        out=tmp_path / "census",
    )


def batches_in_first_epoch(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    rank: int = 0,
    world_size: int = 1,
) -> int:
    """How many batches the loop's own iterator takes from epoch 0 for *rank*."""
    samples_dir, labels_dir = corpus
    batches = EpochBatches(
        CorpusStream(
            samples_dir,
            labels_dir,
            SampleBuilder(lexicon, spans, seed=SEED),
            rank=rank,
            world_size=world_size,
        ),
        Collator(tokenizer),
        BUDGET,
    )
    taken = 0
    while True:
        next(batches)
        if batches.epoch != 0:
            return taken
        taken += 1


def test_the_count_is_the_number_of_batches_the_loop_would_take(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    census = count_batches(
        Vocabularies(spans=spans, tokenizer=tokenizer, lexicon=lexicon),
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
    )
    taken = batches_in_first_epoch(corpus, lexicon, spans, tokenizer)
    assert taken > 1
    assert census.ranks[0].epochs == [taken]
    assert census.steps_for_epochs == taken
    assert census.build_counts["kept"] == 128


def test_every_rank_is_counted_over_the_shards_it_owns(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    # Shards are dealt out whole, so a rank's count is its own shards' count and
    # not half of everything: the run stops when the shortest rank is done.
    census = count_batches(
        Vocabularies(spans=spans, tokenizer=tokenizer, lexicon=lexicon),
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        world_size=2,
    )
    counted = [
        batches_in_first_epoch(corpus, lexicon, spans, tokenizer, rank=rank, world_size=2)
        for rank in (0, 1)
    ]
    assert [rank.total for rank in census.ranks] == counted
    assert census.steps_for_epochs == min(counted)


def test_a_census_covers_every_epoch_and_is_read_back_from_its_file(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    census = count_batches(
        Vocabularies(spans=spans, tokenizer=tokenizer, lexicon=lexicon),
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        epochs=2,
    )
    assert len(census.ranks[0].epochs) == 2
    assert census.ranks[0].total == sum(census.ranks[0].epochs)
    assert census.build_counts["kept"] == 256  # both epochs were really walked

    out = tmp_path / "census" / "batches.json"
    census.write(out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["steps_for_epochs"] == census.steps_for_epochs
    assert written["ranks"][0]["epochs"] == census.ranks[0].epochs
    assert written["token_budget"] == BUDGET
    assert written["shards"] == list(SHARDS)
    assert written["augmentation"]["full"] == 0.55
