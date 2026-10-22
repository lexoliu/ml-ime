"""The idiom selection: a draw at eval3's frequencies, then one judged record per idiom.

Every sentence here is written for the test.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from mlime.data import judge
from mlime.data.idiom_set import (
    Collisions,
    Inputs,
    Record,
    SelectionError,
    Strata,
    build,
    draw,
    ks_distance,
    quantile,
    select,
    write_draw,
)

# --- collisions ---------------------------------------------------------------


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


# --- the strata ---------------------------------------------------------------


def test_strata_cut_at_the_quantiles_hold_eval3s_idioms_evenly() -> None:
    strata = Strata.of([5, 8, 40, 50, 300, 900], 3)
    assert strata.lowers == (1, 40, 300)
    assert strata.targets == (2, 2, 2)
    assert [strata.index(value) for value in (1, 39, 40, 299, 300, 10_000)] == [0, 0, 1, 1, 2, 2]
    assert strata.upper(0) == 40 and strata.upper(2) is None
    with pytest.raises(SelectionError, match="no stratum"):
        strata.index(0)


def test_an_eval3_idiom_absent_from_run3_cannot_be_stratified() -> None:
    with pytest.raises(SelectionError, match="must occur in run3"):
        Strata.of([0, 5], 2)


def test_the_size_is_shared_in_eval3s_proportions() -> None:
    strata = Strata(lowers=(1, 40, 300), targets=(1, 2, 1))
    assert strata.allocate(8) == [2, 4, 2]
    # 7 * (1, 2, 1) / 4 = (1.75, 3.5, 1.75): floors (1, 3, 1), then the largest fractions.
    assert strata.allocate(7) == [2, 3, 2]
    assert sum(strata.allocate(9)) == 9


def test_the_largest_size_is_bound_by_the_scarcest_stratum() -> None:
    strata = Strata(lowers=(1, 40), targets=(1, 1))
    assert strata.largest_size([100, 3]) == 7
    assert strata.largest_size([0, 3]) == 0


def test_quantiles_and_the_distribution_gap() -> None:
    assert quantile([1, 2, 3, 4], 0.5) == 3
    assert quantile([1, 2, 3, 4], 0.0) == 1
    assert ks_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert ks_distance([1, 1], [5, 5]) == 1.0


# --- the draw -------------------------------------------------------------------

TABLE = pl.DataFrame(
    {
        "entry": [
            "一帆风顺",
            "画蛇添足",
            "守株待兔",
            "对牛弹琴",
            "亡羊补牢",
            "刻舟求剑",
            "掩耳盗铃",
            "井底之蛙",
            "自相矛盾",
            "叶公好龙",
            "躺平",
        ],
        "category": ["idiom"] * 10 + ["internet"],
        "occurrences": [10, 100, 5, 7, 300, 200, 0, 50, 50, 50, 50],
        "blocked": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    }
)
KNOWLEDGE = frozenset(TABLE["entry"].to_list()) - {"自相矛盾"}
EMITTABLE = frozenset("".join(TABLE["entry"].to_list())) - {"叶"}
EVAL_TEXTS = ["他做事一帆风顺", "这是画蛇添足"]
LEXICON = judge.BoundaryLexicon(["剑客", "一帆风顺"])


def inputs(collisions: Collisions | None = None) -> Inputs:
    return Inputs(
        table=TABLE,
        knowledge_idioms=KNOWLEDGE,
        emittable=EMITTABLE,
        eval_texts=EVAL_TEXTS,
        collisions=collisions or Collisions.of([Record("你这是对牛弹琴啊", None)]),
        lexicon=LEXICON,
    )


def pool(*texts: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [f"p{index}" for index in range(len(texts))],
            "source": ["web" if index % 2 else "forum" for index in range(len(texts))],
            "text": list(texts),
            "context": ["前文"] * len(texts),
        }
    )


POOL = pool(
    "他总是守株待兔不肯努力",
    "不要再守株待兔了",
    "你这是对牛弹琴啊",  # collides with an evaluation sentence
    "现在亡羊补牢还来得及",
    "那个刻舟求剑客很怪",  # crossed at the word boundary by 剑客
    "刻舟求剑的人真不少",
    "掩耳盗铃的井底之蛙在自相矛盾",
    "叶公好龙也是一帆风顺",
)


def test_the_draw_takes_eligible_idioms_at_eval3s_frequencies() -> None:
    frame, census = draw(POOL, inputs(), 2, 2, seed=243)
    assert census.eval3_idioms == 2
    assert census.excluded_idioms == {
        "absent from run3": 1,
        "blocked": 1,
        "not a knowledge-lexicon idiom": 1,
        "a character not emittable": 1,
        "held by eval3": 2,
    }
    assert census.eligible_idioms == 4
    assert census.idioms_without_sentence == 1  # 对牛弹琴's one sentence collides
    assert census.largest_size == 2
    assert [(s["lower"], s["eval3"], s["available"], s["drawn"]) for s in census.strata] == [
        (1, 1, 1, 1),
        (100, 1, 2, 1),
    ]
    entries = frame["entry"].to_list()
    assert len(entries) == 2 and "守株待兔" in entries
    assert set(entries) - {"守株待兔"} <= {"亡羊补牢", "刻舟求剑"}
    for entry, text in zip(entries, frame["text"], strict=True):
        assert entry in text
    assert "那个刻舟求剑客很怪" not in frame["text"].to_list()
    assert census.rejected_sentences["collides by text"] == 1
    assert census.checks["colliding with evaluation"] == 0
    assert frame.columns == ["id", "source", "text", "context", "entry"]


def test_the_draw_is_reproducible_from_its_seed() -> None:
    first, _ = draw(POOL, inputs(), 2, 2, seed=7)
    again, _ = draw(POOL, inputs(), 2, 2, seed=7)
    assert first.equals(again)


def test_a_size_past_what_the_strata_hold_fails_naming_the_largest() -> None:
    with pytest.raises(SelectionError, match="the largest size that fits is 2"):
        draw(POOL, inputs(), 2, 3, seed=243)


def test_a_sentence_two_idioms_share_is_drawn_once() -> None:
    shared = pool("守株待兔和刻舟求剑都是寓言", "现在亡羊补牢还来得及")
    frame, census = draw(shared, inputs(), 2, 2, seed=243)
    assert dict(zip(frame["entry"], frame["id"], strict=True)) == {
        "守株待兔": "p0",
        "亡羊补牢": "p1",
    }
    assert frame["id"].n_unique() == 2
    assert census.rejected_sentences.get("drawn for another idiom", 0) <= 1


# --- the set, from what the judge kept -------------------------------------------


@dataclass
class Keep:
    async def ask(self, unit: judge.Unit) -> judge.JevAnswer:
        return judge.JevAnswer(p=0.95, model="jev-fake")


@dataclass
class Natural:
    async def ask(self, unit: judge.Unit) -> judge.ReviewAnswer:
        return judge.ReviewAnswer(
            context_spun=False, sentence_natural=unit.text != "不要再守株待兔了"
        )


def drawn_and_judged(root: Path) -> judge.CandidateSet:
    frame, census = draw(POOL, inputs(), 2, 2, seed=243)
    candidates = write_draw(root, frame, census)
    asyncio.run(judge.judge_source(candidates, LEXICON, Keep(), Natural(), 1, 1))
    return candidates


def test_build_writes_the_judged_idioms_as_samples_the_annotator_reads(tmp_path: Path) -> None:
    candidates = drawn_and_judged(tmp_path / "pool")
    assert json.loads((tmp_path / "pool" / "draw.json").read_text("utf-8"))["seed"] == 243
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(json.dumps({"pinyin": "nh", "text": "你好", "context": None}) + "\n")
    table = tmp_path / "entries.tsv"
    TABLE.write_csv(table, separator="\t")
    out = tmp_path / "set"
    census = build(candidates, table, [eval_set], out)
    assert census.records == 2
    shards = sorted((out / "samples").iterdir())
    frames = [pl.read_parquet(path) for path in shards]
    assert all(frame.columns == ["id", "source", "text", "context"] for frame in frames)
    assert [path.name for path in shards] == [
        f"{frame['source'][0]}-00000.parquet" for frame in frames
    ]
    with pytest.raises(SelectionError, match="not empty"):
        build(candidates, table, [eval_set], out)


def test_build_waits_for_the_judge_to_finish(tmp_path: Path) -> None:
    frame, census = draw(POOL, inputs(), 2, 2, seed=243)
    candidates = write_draw(tmp_path / "pool", frame, census)
    table = tmp_path / "entries.tsv"
    TABLE.write_csv(table, separator="\t")
    with pytest.raises(SelectionError, match="has not finished"):
        build(candidates, table, [], tmp_path / "set")


ENTRIES = pl.DataFrame(
    {
        "entry": ["一帆风顺", "画蛇添足", "躺平"],
        "category": ["idiom", "idiom", "internet"],
        "occurrences": [3, 4, 0],
    }
)


def units(*records: tuple[str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [record[0] for record in records],
            "entry": [record[2] for record in records],
            "source": ["web"] * len(records),
            "text": [record[1] for record in records],
            "context": ["前文"] * len(records),
        }
    )


def test_each_idiom_gives_one_record_and_other_categories_none() -> None:
    kept = units(
        ("b", "祝你一帆风顺", "一帆风顺"),
        ("a", "一路上一帆风顺", "一帆风顺"),
        ("c", "这样写就是画蛇添足", "画蛇添足"),
        ("d", "我想躺平了", "躺平"),
    )
    frame, census = select(kept, ENTRIES, Collisions())
    assert frame.sort("id")["id"].to_list() == ["a", "c"]
    assert census.idioms == 2
    assert census.records_per_idiom_kept == {2: 1, 1: 1}


def test_a_colliding_record_gives_way_to_the_idioms_next_record() -> None:
    kept = units(("a", "一路上一帆风顺", "一帆风顺"), ("b", "祝你一帆风顺", "一帆风顺"))
    frame, census = select(kept, ENTRIES, Collisions.of([Record("一路上一帆风顺", "别的")]))
    assert frame["id"].to_list() == ["b"]
    assert census.dropped == {"collides by text": 1}


def test_a_kept_unit_that_does_not_hold_its_entry_is_an_error() -> None:
    with pytest.raises(SelectionError, match="does not hold"):
        select(units(("a", "别的句子", "一帆风顺")), ENTRIES, Collisions())
