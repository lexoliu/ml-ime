"""Counting the batches a run will take, before it fixes its schedule.

``max_steps`` is where the cosine reaches zero, so it has to be the number of
steps the loop will really take. The counter therefore has to be the loop's own
grouping rather than a second implementation that agrees with it today: these
tests hold it to that by counting the same corpus both ways.
"""

from __future__ import annotations

import json
import random
import stat
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import polars as pl
from conftest import READINGS

from mlime.data.corpus import SAMPLE_SCHEMA, Sample
from mlime.train.arbitration import ReadingArbitration
from mlime.train.count import (
    CENSUS_VERSION,
    BatchCensus,
    CountSpec,
    RankBatches,
    census_mismatches,
    count_batches,
    matching_census,
    ranking_shape,
)
from mlime.train.e2e_model import RankingConfig
from mlime.train.labels import LABEL_SCHEMA
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.loop import EpochBatches, epoch_groups
from mlime.train.ranking import RankCollator
from mlime.train.run import RunPaths, Slices, Vocabularies
from mlime.train.samples import (
    DEFAULT_CONTEXT_TOKENS,
    Augmentation,
    BaseTokenizer,
    BatchCollator,
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
    segmenter: Path | None = None,
) -> int:
    """How many batches the loop's own iterator takes from epoch 0 for *rank*.

    With a *segmenter*, the loop's collator is the ranking one over it.
    """
    samples_dir, labels_dir = corpus
    plain = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
    collator: BatchCollator = (
        plain if segmenter is None else RankCollator.launch(plain, segmenter, spans, RANKING)
    )
    batches = EpochBatches(
        CorpusStream(
            samples_dir,
            labels_dir,
            SampleBuilder(lexicon, spans, arbitration, seed=SEED),
            rank=rank,
            world_size=world_size,
        ),
        collator,
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
        Vocabularies(
            spans=spans,
            tokenizer=tokenizer,
            lexicon=lexicon,
            arbitration=arbitration,
            syllables=frozenset(),
        ),
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


#: A ranking run small enough that the fixture corpus is still many batches.
RANKING = RankingConfig(segmentations=4, beam=2, negatives=2, weight=1.0)


def two_readings(path: Path, spans: SpanVocab) -> Path:
    """A segmenter binary reading every keystroke string two ways.

    The first is the longest-span reading of the keys, the second the same
    spans backwards: not a spelling of the keys, but rows the collator can
    type, which is all a count and a collation need of them.
    """
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"spans = set({json.dumps([span for span in spans if span])})\n"
        "for line in sys.stdin:\n"
        "    keys, reading, start = line.strip(), [], 0\n"
        "    while start < len(keys):\n"
        "        end = max(e for e in range(start + 1, len(keys) + 1) if keys[start:e] in spans)\n"
        "        reading.append(keys[start:end])\n"
        "        start = end\n"
        "    print(json.dumps([reading, reading[::-1]]), flush=True)\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_a_ranking_count_weighs_each_example_by_every_row_it_costs(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    segmenter = two_readings(tmp_path / "segment", spans)
    vocabularies = Vocabularies(
        spans=spans,
        tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(),
    )
    plain = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
    )
    ranked = count_batches(
        vocabularies,
        census_paths(corpus, tmp_path),
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        ranking=RANKING,
        segmenter=segmenter,
    )
    taken = batches_in_first_epoch(
        corpus, lexicon, spans, arbitration, tokenizer, segmenter=segmenter
    )
    assert ranked.ranks[0].epochs == [taken]
    assert ranked.steps_for_epochs > plain.steps_for_epochs
    assert ranked.ranking == ranking_shape(RANKING)
    assert plain.ranking is None
    assert census_mismatches(ranked, spec_for(ranking=None)) == ["ranking"]
    # The weight only scales the loss: a weight sweep shares one census.
    reweighted = ranking_shape(replace(RANKING, weight=0.1))
    assert census_mismatches(ranked, spec_for(ranking=reweighted)) == []


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
        Vocabularies(
            spans=spans,
            tokenizer=tokenizer,
            lexicon=lexicon,
            arbitration=arbitration,
            syllables=frozenset(),
        ),
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
        Vocabularies(
            spans=spans,
            tokenizer=tokenizer,
            lexicon=lexicon,
            arbitration=arbitration,
            syllables=frozenset(),
        ),
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
        spans=spans,
        tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(),
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
        ranking=None,
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
        spans=spans,
        tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(),
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
        spans=spans,
        tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(),
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
        spans=spans,
        tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(),
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
        spans=spans,
        tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(),
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


#: Sources of a real corpus, each with shards of uneven length: the reading
#: order interleaves them, and the stride's offset into each shard depends on
#: every shard read before it.
REAL_SHAPED = {"news": (37, 52, 11), "wiki-rest": (64, 29), "dialogue": (45,), "moegirl": (8, 70)}


def real_shaped_corpus(root: Path) -> tuple[Path, Path, tuple[str, ...]]:
    """Shards in the corpus's schema whose rows exercise every path a build takes.

    Sentences of one to fourteen characters, contexts from none to past the
    context budget, and rows the builder drops for each of its reasons: no
    label (missing or refused), a non-Han character, a length mismatch, and a reading
    the arbitration rewrites (和 labelled ``han``).
    """
    rng = random.Random(7)
    characters = sorted(READINGS)
    samples_dir, labels_dir = root / "samples", root / "labels"
    samples_dir.mkdir(parents=True)
    labels_dir.mkdir()
    names = []
    for source, lengths in REAL_SHAPED.items():
        for number, rows in enumerate(lengths):
            name = f"{source}-{number:05d}.parquet"
            names.append(name)
            samples, labels = [], []
            for index in range(rows):
                text = "".join(rng.choice(characters) for _ in range(rng.randint(1, 14)))
                readings = [f"{rng.choice(READINGS[c])}{rng.randint(1, 4)}" for c in text]
                draw = rng.random()
                if draw < 0.03:
                    text = text[:-1] + "a"
                elif draw < 0.06:
                    readings = [*readings[:-1], "han2", "he2"]
                elif draw < 0.1 and "和" in text:
                    readings[text.index("和")] = "han4"
                context = (
                    None
                    if rng.random() < 0.3
                    else "".join(rng.choice(characters) for _ in range(rng.randint(0, 90)))
                )
                sample = Sample(
                    id=f"{source}-{number}-{index}", source=source, text=text, context=context
                )
                samples.append(sample.row())
                if draw > 0.97:
                    continue
                refused = draw > 0.95
                labels.append(
                    {
                        "id": sample.id,
                        "syllables": None if refused else readings,
                        "refusal": "refused" if refused else None,
                    }
                )
            pl.DataFrame(samples, schema=SAMPLE_SCHEMA).write_parquet(samples_dir / name)
            pl.DataFrame(labels, schema=LABEL_SCHEMA).write_parquet(labels_dir / name)
    return samples_dir, labels_dir, tuple(sorted(names))


def serial_census(
    vocabularies: Vocabularies,
    paths: RunPaths,
    shards: tuple[str, ...],
    world_size: int,
    epochs: int,
    collator: BatchCollator,
) -> BatchCensus:
    """The census the training loop's own readers make, one rank after another."""
    ranks, tallies = [], Counter[str]()
    for rank in range(world_size):
        builder = vocabularies.builder(None, SEED)
        stream = vocabularies.stream(paths, shards, builder, rank, world_size)
        counted = [
            sum(1 for _ in epoch_groups(stream, collator, BUDGET, epoch)) for epoch in range(epochs)
        ]
        ranks.append(RankBatches(rank=rank, epochs=counted, total=sum(counted)))
        tallies.update(builder.counts.as_dict())
    return BatchCensus(
        world_size=world_size,
        epochs=epochs,
        token_budget=BUDGET,
        max_context_tokens=DEFAULT_CONTEXT_TOKENS,
        seed=SEED,
        augmentation=asdict(Augmentation()),
        shards=list(shards),
        ranks=ranks,
        accumulate=1,
        steps_for_epochs=min(rank.total for rank in ranks),
        census_version=CENSUS_VERSION,
        build_counts=dict(tallies),
    )


def test_the_census_is_the_loops_own_count_at_any_number_of_workers(
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    samples_dir, labels_dir, shards = real_shaped_corpus(tmp_path / "corpus")
    paths = census_paths((samples_dir, labels_dir), tmp_path)
    vocabularies = Vocabularies(
        spans=spans,
        tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(),
    )
    for world_size in (2, 3):
        serial = serial_census(
            vocabularies,
            paths,
            shards,
            world_size,
            epochs=2,
            collator=vocabularies.collator(0.0, DEFAULT_CONTEXT_TOKENS, SEED),
        )
        # Every drop reason and the arbitration occur, so the counters are
        # compared on something.
        assert all(serial.build_counts[name] > 0 for name in DROPS), serial.build_counts
        for workers in (1, 2, 3):
            census = count_batches(
                vocabularies,
                paths,
                Slices(train=shards, held_out=()),
                token_budget=BUDGET,
                seed=SEED,
                world_size=world_size,
                epochs=2,
                workers=workers,
            )
            assert asdict(census) == asdict(serial), (world_size, workers)


#: The builder's counters the real-shaped corpus must exercise.
#: A refused label reaches the builder as no label, so it lands in
#: ``unlabelled``; ``refused`` is never raised there.
DROPS = ("unlabelled", "not_all_han", "length_mismatch", "reading_arbitrated")


def test_a_ranking_census_reads_the_segmenter_in_every_worker(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    paths = census_paths(corpus, tmp_path)
    segmenter = two_readings(tmp_path / "segment", spans)
    vocabularies = Vocabularies(
        spans=spans,
        tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(),
    )
    plain = vocabularies.collator(0.0, DEFAULT_CONTEXT_TOKENS, SEED)
    serial = serial_census(
        vocabularies,
        paths,
        SHARDS,
        2,
        epochs=1,
        collator=RankCollator.launch(plain, segmenter, spans, RANKING),
    )
    census = count_batches(
        vocabularies,
        paths,
        Slices(train=SHARDS, held_out=()),
        token_budget=BUDGET,
        seed=SEED,
        world_size=2,
        ranking=RANKING,
        segmenter=segmenter,
        workers=2,
    )
    assert asdict(census) == asdict(replace(serial, ranking=ranking_shape(RANKING)))
