"""Counting the batches a run will take, before it fixes its schedule.

``max_steps`` is where the cosine reaches zero, so it has to be the number of
steps the loop will really take. The counter therefore has to be the loop's own
grouping rather than a second implementation that agrees with it today: these
tests hold it to that by counting the same corpus both ways.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from mlime.train.arbitration import ReadingArbitration
from mlime.train.count import CENSUS_VERSION, CountSpec, count_batches, matching_census
from mlime.train.lexicon import Lexicon
from mlime.train.loop import EpochBatches
from mlime.train.run import RunPaths, Slices, Vocabularies
from mlime.train.samples import (
    DEFAULT_CONTEXT_TOKENS,
    Augmentation,
    BaseTokenizer,
    Collator,
    CorpusStream,
    SampleBuilder,
)
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
    arbitration: ReadingArbitration,
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
            SampleBuilder(lexicon, spans, arbitration, seed=SEED),
            rank=rank,
            world_size=world_size,
        ),
        Collator(tokenizer, lexicon.candidate_mask),
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
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    census = count_batches(
        Vocabularies(spans=spans, tokenizer=tokenizer, lexicon=lexicon, arbitration=arbitration),
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
    )
    taken = batches_in_first_epoch(corpus, lexicon, spans, arbitration, tokenizer)
    assert taken > 1
    assert census.ranks[0].epochs == [taken]
    assert census.steps_for_epochs == taken
    assert census.build_counts["kept"] == 128


def test_every_rank_is_counted_over_the_shards_it_owns(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    # Shards are dealt out whole, so a rank's count is its own shards' count and
    # not half of everything: the run stops when the shortest rank is done.
    census = count_batches(
        Vocabularies(spans=spans, tokenizer=tokenizer, lexicon=lexicon, arbitration=arbitration),
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        world_size=2,
    )
    counted = [
        batches_in_first_epoch(
            corpus, lexicon, spans, arbitration, tokenizer, rank=rank, world_size=2
        )
        for rank in (0, 1)
    ]
    assert [rank.total for rank in census.ranks] == counted
    assert census.steps_for_epochs == min(counted)


def test_a_census_covers_every_epoch_and_is_read_back_from_its_file(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    census = count_batches(
        Vocabularies(spans=spans, tokenizer=tokenizer, lexicon=lexicon, arbitration=arbitration),
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


def test_accumulation_divides_micro_batches_into_steps(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """The rank totals count micro-batches; a step of N of them is a count/N."""
    vocabularies = Vocabularies(
        spans=spans, tokenizer=tokenizer, lexicon=lexicon, arbitration=arbitration
    )
    one = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
    )
    four = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path / "four"),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        accumulate=4,
    )
    assert four.ranks == one.ranks
    assert four.steps_for_epochs == -(-one.steps_for_epochs // 4)
    assert four.accumulate == 4


def spec_for(**overrides) -> CountSpec:
    """The configuration the fixture counts use, with one field swapped out."""
    spec = CountSpec(
        world_size=1,
        epochs=1,
        token_budget=BUDGET,
        max_context_tokens=DEFAULT_CONTEXT_TOKENS,
        seed=SEED,
        augmentation=asdict(Augmentation()),
        shards=list(SHARDS),
        accumulate=1,
        census_version=CENSUS_VERSION,
    )
    return replace(spec, **overrides)


def test_a_stored_census_is_used_when_the_configuration_matches(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    vocabularies = Vocabularies(
        spans=spans, tokenizer=tokenizer, lexicon=lexicon, arbitration=arbitration
    )
    counted = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
    )
    stored = tmp_path / "batch-counts.json"
    counted.write(stored)
    # A field the configuration match ignores is marked, so a recount where a
    # reuse belonged shows up.
    data = json.loads(stored.read_text(encoding="utf-8"))
    data["build_counts"]["kept"] = -1
    stored.write_text(json.dumps(data), encoding="utf-8")
    census, differed = matching_census(stored, spec_for())
    assert census is not None and differed == []
    reused = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        reuse_census=stored,
    )
    assert reused.build_counts["kept"] == -1
    assert reused.steps_for_epochs == counted.steps_for_epochs


def test_a_stored_census_is_refused_when_one_field_differs(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    vocabularies = Vocabularies(
        spans=spans, tokenizer=tokenizer, lexicon=lexicon, arbitration=arbitration
    )
    counted = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
    )
    stored = tmp_path / "batch-counts.json"
    counted.write(stored)
    data = json.loads(stored.read_text(encoding="utf-8"))
    data["build_counts"]["kept"] = -1
    stored.write_text(json.dumps(data), encoding="utf-8")
    for field, changed in (
        ("seed", spec_for(seed=SEED + 1)),
        ("token_budget", spec_for(token_budget=BUDGET * 2)),
        ("accumulate", spec_for(accumulate=4)),
        ("world_size", spec_for(world_size=2)),
        ("shards", spec_for(shards=["other-00000.parquet"])),
    ):
        census, differed = matching_census(stored, changed)
        assert census is None and differed == [field]
    # A refused census is counted again, not used anyway.
    recounted = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        accumulate=4,
        reuse_census=stored,
    )
    assert recounted.build_counts["kept"] == 128
    assert recounted.accumulate == 4


def test_a_census_without_a_version_is_refused(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """A file written before censuses were versioned is not reused.

    The count's method itself is part of what makes it exact: a census whose
    format predates the current counting is a wrong count even when every
    configuration field matches.
    """
    vocabularies = Vocabularies(
        spans=spans, tokenizer=tokenizer, lexicon=lexicon, arbitration=arbitration
    )
    counted = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
    )
    assert counted.census_version == CENSUS_VERSION
    stored = tmp_path / "batch-counts.json"
    counted.write(stored)
    data = json.loads(stored.read_text(encoding="utf-8"))
    del data["census_version"]
    stored.write_text(json.dumps(data), encoding="utf-8")
    census, differed = matching_census(stored, spec_for())
    assert census is None and differed == ["census_version"]
    recounted = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        reuse_census=stored,
    )
    assert recounted.steps_for_epochs == counted.steps_for_epochs


def test_a_v1_shaped_census_is_refused_and_recounted(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """A file shaped like an older census is refused, not trusted enough to raise on."""
    vocabularies = Vocabularies(
        spans=spans, tokenizer=tokenizer, lexicon=lexicon, arbitration=arbitration
    )
    counted = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
    )
    stored = tmp_path / "batch-counts.json"
    counted.write(stored)
    v1 = json.loads(stored.read_text(encoding="utf-8"))
    del v1["accumulate"], v1["census_version"]
    stored.write_text(json.dumps(v1), encoding="utf-8")
    census, differed = matching_census(stored, spec_for())
    assert census is None and differed == ["accumulate", "census_version"]
    # A field this code does not know is also refused, by name.
    data = json.loads(stored.read_text(encoding="utf-8"))
    data["accumulate"] = 1
    data["census_version"] = CENSUS_VERSION
    data["extra_field"] = 1
    stored.write_text(json.dumps(data), encoding="utf-8")
    census, differed = matching_census(stored, spec_for())
    assert census is None and differed == ["extra_field"]
    # And the refusal counts again rather than exiting.
    recounted = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        reuse_census=stored,
    )
    assert recounted.steps_for_epochs == counted.steps_for_epochs
