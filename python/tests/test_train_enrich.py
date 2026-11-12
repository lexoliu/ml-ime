"""The enrichment builder: rare units lifted to a floor by repeating their own sentences.

The builder's promises are the ones a training run leans on: every repeat is a
row of its own so it draws its own typing, a repeat's label is its original's,
no sentence is repeated past the cap, a sentence the training builder would drop
is never repeated, and the output is the same whatever the number of workers.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mlime.data.corpus import SAMPLE_SCHEMA
from mlime.train.enrich import (
    Corpus,
    Match,
    blocked_entries,
    enrich,
    plan_lift,
    repeat_id,
)
from mlime.train.labels import LABEL_SCHEMA
from mlime.train.lexicon import read_emittable
from mlime.train.units import UnitMatcher

#: Every character the sentences below use is emittable, except 龘.
EMITTABLE = frozenset("我爱北京天安门一帆风顺龙马精神今天好吗的")


def write_corpus(
    root: Path, shards: dict[str, list[tuple[str, str, list[str] | None]]]
) -> tuple[Path, Path]:
    """Sample and label shards under *root*: each row is ``(id, text, syllables or None)``."""
    samples, labels = root / "samples", root / "labels"
    samples.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name, rows in shards.items():
        pl.DataFrame(
            [
                {"id": row_id, "source": "test", "text": text, "context": f"before {row_id}"}
                for row_id, text, _ in rows
            ],
            schema=SAMPLE_SCHEMA,
        ).write_parquet(samples / f"{name}.parquet")
        pl.DataFrame(
            [
                {
                    "id": row_id,
                    "syllables": syllables,
                    "refusal": None if syllables is not None else "refused",
                }
                for row_id, _, syllables in rows
            ],
            schema=LABEL_SCHEMA,
        ).write_parquet(labels / f"{name}.parquet")
    return samples.parent, labels


def syllables_of(text: str) -> list[str]:
    """One placeholder syllable per character: the builder copies labels, it never reads them."""
    return [f"s{ord(character) % 97}" for character in text]


def rows_of(directory: Path, prefix: str) -> pl.DataFrame:
    """Every ``prefix`` shard under *directory*, concatenated in name order."""
    return pl.concat(
        [pl.read_parquet(path) for path in sorted(directory.glob(f"{prefix}-*.parquet"))]
    )


def run(
    data: Path, labels: Path, out: Path, entries: list[str], **options: object
) -> dict[str, object]:
    return enrich(
        data / "samples",
        [labels],
        UnitMatcher(entries),
        EMITTABLE,
        out,
        workers=1,
        **options,  # type: ignore[arg-type]
    )


def test_a_rare_unit_is_lifted_to_the_floor_by_repeating_its_own_sentences(
    tmp_path: Path,
) -> None:
    texts = ["我爱北京天安门", "北京今天好吗", "今天北京好"]
    rows = [(f"id{index}", text, syllables_of(text)) for index, text in enumerate(texts)]
    data, labels = write_corpus(tmp_path / "corpus", {"news-00000": rows})
    census = run(data, labels, tmp_path / "out", ["北京"], floor=20, max_repeats=8)
    lift = rows_of(tmp_path / "out" / "samples", "lift")
    # Three occurrences, so seventeen copies; eight per sentence would allow 24.
    assert lift.height == 17
    assert census["lift"]["rows_written"] == 17  # type: ignore[index]
    assert census["entries"]["lifted_to_floor"] == 1  # type: ignore[index]
    assert census["under_floor"]["entries"] == 0  # type: ignore[index]
    assert lift["id"].n_unique() == 17
    originals = {row_id: text for row_id, text, _ in rows}
    for row_id, text in zip(lift["id"], lift["text"], strict=True):
        original, _, repeat = row_id.rpartition("+")
        assert 1 <= int(repeat) <= 8
        assert originals[original] == text
    per_sentence = lift["id"].str.replace(r"\+\d+$", "").value_counts()
    assert per_sentence["count"].max() <= 8
    histogram = census["lift"]["repeats_histogram"]  # type: ignore[index]
    assert sum(histogram.values()) == 3
    assert sum(int(repeats) * count for repeats, count in histogram.items()) == 17


def test_the_labels_of_a_repeat_are_its_originals_copied_not_recomputed(tmp_path: Path) -> None:
    text = "我爱北京天安门"
    syllables = ["wo3", "ai4", "bei3", "jing1", "tian1", "an1", "men2"]
    data, labels = write_corpus(tmp_path / "corpus", {"wiki-00000": [("a", text, syllables)]})
    run(data, labels, tmp_path / "out", ["北京"], floor=4)
    lifted = rows_of(tmp_path / "out" / "labels", "lift")
    assert set(lifted["id"].to_list()) == {repeat_id("a", repeat) for repeat in (1, 2, 3)}
    assert all(row == syllables for row in lifted["syllables"].to_list())
    samples = rows_of(tmp_path / "out" / "samples", "lift")
    assert samples["id"].to_list() == lifted["id"].to_list()
    assert samples.schema == SAMPLE_SCHEMA and lifted.schema == LABEL_SCHEMA


def test_a_sentence_is_never_repeated_past_the_cap_and_the_shortfall_is_reported(
    tmp_path: Path,
) -> None:
    data, labels = write_corpus(
        tmp_path / "corpus", {"wiki-00000": [("a", "我爱北京", syllables_of("我爱北京"))]}
    )
    census = run(data, labels, tmp_path / "out", ["北京"], floor=20, max_repeats=8)
    assert rows_of(tmp_path / "out" / "samples", "lift").height == 8
    assert census["entries"]["lifted_short_of_floor"] == 1  # type: ignore[index]
    # One occurrence and eight copies: nine of twenty, eleven short.
    assert census["under_floor"]["total_shortfall"] == 11  # type: ignore[index]
    assert (tmp_path / "out" / "under-floor.tsv").read_text(encoding="utf-8").splitlines() == [
        "entry\toccurrences\tshortfall",
        "北京\t9\t11",
    ]


def test_a_unit_at_the_floor_or_absent_or_blocked_is_not_lifted(tmp_path: Path) -> None:
    common = [(f"c{index}", "我爱北京", syllables_of("我爱北京")) for index in range(20)]
    rare = [("r", "一帆风顺", syllables_of("一帆风顺"))]
    data, labels = write_corpus(tmp_path / "corpus", {"news-00000": [*common, *rare]})
    census = run(data, labels, tmp_path / "out", ["北京", "一帆风顺", "龙马精神", "龘龘"])
    lift = rows_of(tmp_path / "out" / "samples", "lift")
    assert set(lift["text"]) == {"一帆风顺"}
    entries = census["entries"]
    assert entries["total"] == 4  # type: ignore[index]
    assert entries["blocked"] == 1  # type: ignore[index]
    assert entries["absent"] == 1  # type: ignore[index]
    assert entries["at_or_above_floor"] == 1  # type: ignore[index]


def test_a_sentence_the_training_builder_would_drop_is_not_counted_or_repeated(
    tmp_path: Path,
) -> None:
    rows = [
        ("unlabelled", "我爱北京", None),
        ("short", "我爱北京", ["wo3"]),
        ("outside", "龘北京", syllables_of("龘北京")),
        ("good", "我爱北京", syllables_of("我爱北京")),
    ]
    data, labels = write_corpus(tmp_path / "corpus", {"news-00000": rows})
    census = run(data, labels, tmp_path / "out", ["北京"], floor=5)
    scan = census["scan"]
    assert scan["rows_holding_a_unit"] == 4  # type: ignore[index]
    assert scan["of_which_unlabelled"] == 2  # type: ignore[index]
    assert scan["of_which_not_emittable"] == 1  # type: ignore[index]
    assert scan["trainable_unit_occurrences"] == 1  # type: ignore[index]
    assert set(rows_of(tmp_path / "out" / "samples", "lift")["id"].str.replace(r"\+\d+$", "")) == {
        "good"
    }


def test_a_sentence_shared_by_two_units_serves_both(tmp_path: Path) -> None:
    text = "我爱北京天安门"
    data, labels = write_corpus(
        tmp_path / "corpus", {"news-00000": [("a", text, syllables_of(text))]}
    )
    census = run(data, labels, tmp_path / "out", ["北京", "天安门"], floor=5)
    # Four copies serve both units at once rather than four for each.
    assert rows_of(tmp_path / "out" / "samples", "lift").height == 4
    assert census["entries"]["lifted_to_floor"] == 2  # type: ignore[index]


def test_a_nested_unit_counts_where_it_occurs(tmp_path: Path) -> None:
    text = "一帆风顺"
    data, labels = write_corpus(
        tmp_path / "corpus", {"news-00000": [("a", text, syllables_of(text))]}
    )
    census = run(data, labels, tmp_path / "out", ["一帆", "一帆风顺"], floor=3)
    assert census["scan"]["trainable_unit_occurrences"] == 2  # type: ignore[index]
    assert rows_of(tmp_path / "out" / "samples", "lift").height == 2


def test_the_output_is_the_same_whatever_the_workers_and_shards_order(tmp_path: Path) -> None:
    shards = {
        f"news-{shard:05d}": [
            (f"s{shard}r{row}", text, syllables_of(text))
            for row, text in enumerate(["我爱北京", "今天好吗", "龙马精神的一帆风顺", "北京天安门"])
        ]
        for shard in range(4)
    }
    data, labels = write_corpus(tmp_path / "corpus", shards)
    kwargs: dict[str, object] = {"floor": 30, "rows_per_shard": 7}
    run(data, labels, tmp_path / "one", ["北京", "一帆风顺", "龙马精神"], **kwargs)
    enrich(
        data / "samples",
        [labels],
        UnitMatcher(["北京", "一帆风顺", "龙马精神"]),
        EMITTABLE,
        tmp_path / "two",
        workers=2,
        **kwargs,  # type: ignore[arg-type]
    )
    for kind in ("samples", "labels"):
        first = sorted((tmp_path / "one" / kind).glob("lift-*.parquet"))
        second = sorted((tmp_path / "two" / kind).glob("lift-*.parquet"))
        assert [path.name for path in first] == [path.name for path in second]
        assert len(first) > 1
        for left, right in zip(first, second, strict=True):
            assert pl.read_parquet(left).equals(pl.read_parquet(right))
    assert (tmp_path / "one" / "census.json").read_text(encoding="utf-8") == (
        tmp_path / "two" / "census.json"
    ).read_text(encoding="utf-8")


def test_an_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    data, labels = write_corpus(
        tmp_path / "corpus", {"wiki-00000": [("a", "我爱北京", syllables_of("我爱北京"))]}
    )
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError, match="never overwrites"):
        run(data, labels, out, ["北京"])
    assert list(out.iterdir()) == []


def test_labels_are_found_by_id_whatever_their_shards_are_called(tmp_path: Path) -> None:
    rows = [(f"id{index}", "我爱北京", syllables_of("我爱北京")) for index in range(6)]
    data, labels = write_corpus(tmp_path / "corpus", {"news-00000": rows})
    (labels / "news-00000.parquet").rename(labels / "other-00007.parquet")
    census = run(data, labels, tmp_path / "out", ["北京"], floor=10)
    assert census["scan"]["trainable_unit_occurrences"] == 6  # type: ignore[index]
    # Split across two directories, the same readings are found too.
    second = tmp_path / "second"
    second.mkdir()
    held = pl.read_parquet(labels / "other-00007.parquet")
    held.head(3).write_parquet(labels / "other-00007.parquet")
    held.tail(3).write_parquet(second / "part-00000.parquet")
    census = enrich(
        data / "samples",
        [labels, second],
        UnitMatcher(["北京"]),
        EMITTABLE,
        tmp_path / "split",
        floor=10,
        workers=1,
    )
    assert census["scan"]["trainable_unit_occurrences"] == 6  # type: ignore[index]


def test_a_label_directory_without_shards_is_an_error(tmp_path: Path) -> None:
    data, labels = write_corpus(
        tmp_path / "corpus", {"wiki-00000": [("a", "我爱北京", syllables_of("我爱北京"))]}
    )
    (labels / "wiki-00000.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="no label shards"):
        run(data, labels, tmp_path / "out", ["北京"])


def test_a_repeated_id_is_taken_once(tmp_path: Path) -> None:
    row = ("same", "我爱北京", syllables_of("我爱北京"))
    data, labels = write_corpus(tmp_path / "corpus", {"a-00000": [row], "b-00000": [row]})
    for name in ("a-00000", "b-00000"):
        assert (labels / f"{name}.parquet").exists()
    census = run(data, labels, tmp_path / "out", ["北京"], floor=3)
    assert census["scan"]["of_which_duplicate_ids"] == 1  # type: ignore[index]
    assert rows_of(tmp_path / "out" / "samples", "lift")["id"].n_unique() == 2


def test_extra_shards_pass_through_with_their_labels_or_are_reported_as_needing_them(
    tmp_path: Path,
) -> None:
    data, labels = write_corpus(
        tmp_path / "corpus", {"wiki-00000": [("a", "我爱北京", syllables_of("我爱北京"))]}
    )
    mined, mined_labels = write_corpus(
        tmp_path / "mined",
        {
            "judged": [
                ("m1", "龙马精神", syllables_of("龙马精神")),
                ("m2", "好吗", ["hao3", "ma5"]),
            ],
            "unjudged": [("m3", "今天好吗", syllables_of("今天好吗"))],
        },
    )
    (mined_labels / "unjudged.parquet").unlink()
    census = run(
        data,
        labels,
        tmp_path / "out",
        ["北京"],
        extras=[mined / "samples" / "judged.parquet", mined / "samples" / "unjudged.parquet"],
        extra_labels=mined_labels,
    )
    out = tmp_path / "out"
    assert pl.read_parquet(out / "samples" / "mined-00000.parquet")["id"].to_list() == ["m1", "m2"]
    assert pl.read_parquet(out / "labels" / "mined-00000.parquet")["id"].to_list() == ["m1", "m2"]
    assert pl.read_parquet(out / "needs-labels" / "samples" / "mined-00001.parquet").height == 1
    assert not (out / "labels" / "mined-00001.parquet").exists()
    assert census["extra_rows_needing_labels"] == 1
    extras = census["extras"]
    assert [report["labelled"] for report in extras] == [True, False]  # type: ignore[union-attr]


def test_the_plan_counts_a_units_rows_in_corpus_order() -> None:
    corpus = Corpus(floor=3)
    corpus.add(
        [
            Match("x", "test", "我爱北京", None, (("北京", 1),)),
            Match("y", "test", "北京好", None, (("北京", 1),)),
        ],
        {"x": ("a", "b", "c", "d"), "y": ("a", "b", "c")},
    )
    plan = plan_lift(corpus, frozenset(), max_repeats=8)
    # One more occurrence is needed, and the first sentence in corpus order takes it.
    assert [corpus.rows[index].id for index in plan.copies] == ["x"]
    assert list(plan.copies.values()) == [1]
    assert plan.lifted == {"北京": 3}


def test_blocked_entries_hold_a_character_outside_the_set() -> None:
    assert blocked_entries(["北京", "龘北"], EMITTABLE) == {"龘北"}


def test_the_emittable_file_is_one_character_a_line_in_order(tmp_path: Path) -> None:
    path = tmp_path / "emittable.txt"
    path.write_text("我\n爱\n", encoding="utf-8")
    assert read_emittable(path) == ("我", "爱")
    path.write_text("我爱\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one-character-per-line"):
        read_emittable(path)
    path.write_text("我\n我\n", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        read_emittable(path)
    with pytest.raises(FileNotFoundError, match="no emittable set"):
        read_emittable(tmp_path / "missing.txt")
