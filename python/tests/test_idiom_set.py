"""The idiom selection: only fully kept idioms, one record each, nothing shared with eval."""

import json
from pathlib import Path

import polars as pl
import pytest

from mlime.data.idiom_set import (
    SOURCE,
    Collisions,
    Record,
    SelectionError,
    build,
    kept_units,
    select,
)


def verdicts(record: str, entry: str, *keeps: bool) -> list[dict[str, object]]:
    stages = ("boundary", "jev", "review")
    return [
        {"id": record, "entry": entry, "stage": stage, "keep": keep}
        for stage, keep in zip(stages, keeps, strict=False)
    ]


def test_a_unit_is_kept_only_when_all_three_rules_keep_it() -> None:
    lines = [
        *verdicts("a", "一帆风顺", True, True, True),
        *verdicts("b", "一帆风顺", True, True, False),
        *verdicts("c", "一帆风顺", True, False),
        *verdicts("d", "一帆风顺", False),
        *verdicts("e", "一帆风顺", True, True),
    ]
    assert kept_units(lines) == {("a", "一帆风顺")}


def test_verdicts_out_of_order_or_after_a_rejection_are_errors() -> None:
    swapped = [*verdicts("a", "x", True, True)]
    swapped.reverse()
    with pytest.raises(SelectionError, match="out of order"):
        kept_units(swapped)
    with pytest.raises(SelectionError, match="judged again"):
        kept_units(
            [
                {"id": "a", "entry": "x", "stage": s, "keep": k}
                for s, k in (("boundary", False), ("jev", True))
            ]
        )


def test_collisions_catch_the_record_the_sentence_and_shared_runs() -> None:
    index = Collisions.of(
        [Record("今天天气很好", "我们一起去公园散步吧好不好呀"), Record("你好", None)]
    )
    assert index.reason(Record("今天天气很好", "我们一起去公园散步吧好不好呀")) == "record"
    assert index.reason(Record("你好", "别的")) == "text"
    # Thirteen Han characters across the context and the text, punctuation skipped.
    assert index.reason(Record("今天天气", "，去公园散步吧，好不好呀")) == "window"
    assert index.reason(Record("完全不同", "公园散步吧好不好呀今天天")) is None, (
        "twelve is not a copy"
    )
    assert index.reason(Record("完全不同的话", "前面的话也不同")) is None


ENTRIES = pl.DataFrame(
    {
        "entry": ["一帆风顺", "画蛇添足", "躺平"],
        "category": ["idiom", "idiom", "internet"],
        "occurrences": [0, 0, 0],
    }
)


def rows(*records: tuple[str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [record[0] for record in records],
            "source": [SOURCE] * len(records),
            "text": [record[1] for record in records],
            "context": [record[2] for record in records],
        }
    )


def test_each_idiom_gives_one_record_and_other_categories_none() -> None:
    mined = rows(
        ("b", "祝你一帆风顺", "前文一"),
        ("a", "一路上一帆风顺", "前文二"),
        ("c", "这样写就是画蛇添足", "前文三"),
        ("d", "我想躺平了", "前文四"),
    )
    kept = {("a", "一帆风顺"), ("b", "一帆风顺"), ("c", "画蛇添足"), ("d", "躺平")}
    frame, census = select(mined, kept, ENTRIES, Collisions())
    assert frame.sort("id")["id"].to_list() == ["a", "c"]
    assert census.idioms == 2
    assert census.records == 2
    assert census.records_per_idiom_kept == {2: 1, 1: 1}


def test_a_colliding_record_gives_way_to_the_idioms_next_record() -> None:
    mined = rows(("a", "一路上一帆风顺", "前文"), ("b", "祝你一帆风顺", "前文"))
    collisions = Collisions.of([Record("一路上一帆风顺", "别的")])
    frame, census = select(mined, {("a", "一帆风顺"), ("b", "一帆风顺")}, ENTRIES, collisions)
    assert frame["id"].to_list() == ["b"]
    assert census.dropped == {"collides by text": 1}


def test_a_kept_unit_that_does_not_hold_its_entry_is_an_error() -> None:
    with pytest.raises(SelectionError, match="does not hold"):
        select(rows(("a", "别的句子", "前文")), {("a", "一帆风顺")}, ENTRIES, Collisions())


def test_build_writes_a_samples_shard_the_annotator_reads(tmp_path: Path) -> None:
    mined = tmp_path / "fineweb2-mined"
    (mined / "unjudged").mkdir(parents=True)
    (mined / "judge").mkdir()
    rows(("a", "祝你一帆风顺", "前文")).write_parquet(
        mined / "unjudged" / "fineweb2-mined-0000000.parquet"
    )
    (mined / "judge" / "0000.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in verdicts("a", "一帆风顺", True, True, True)),
        encoding="utf-8",
    )
    table = tmp_path / "entries.tsv"
    ENTRIES.write_csv(table, separator="\t")
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(json.dumps({"pinyin": "nh", "text": "你好", "context": None}) + "\n")
    out = tmp_path / "out"
    census = build(mined, table, [eval_set], out)
    assert census.records == 1
    shard = pl.read_parquet(out / "samples" / f"{SOURCE}-00000.parquet")
    assert shard.columns == ["id", "source", "text", "context"]
    with pytest.raises(SelectionError, match="not empty"):
        build(mined, table, [eval_set], out)
