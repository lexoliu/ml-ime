"""The knowledge lexicon's entry rule and the dumps every system's answers become."""

import json
from pathlib import Path

import pytest

from mlime.data.knowledge import Source, build, counts, entry, write
from mlime.dumps import DumpRow, answer_rows, journal_rows, report_rows, write_dump
from mlime.gui import DEV_SHARE, slice_indices

ORDINARY = {"苏醒": "v", "面包": "n", "刘德华": "nr", "一帆风顺": "i", "北京": "ns"}


def test_an_entry_counts_unless_it_is_short_foreign_or_an_ordinary_word() -> None:
    assert counts("刘德华", ORDINARY)
    assert counts("一帆风顺", ORDINARY), "an idiom is not ordinary"
    assert counts("邓紫棋", ORDINARY), "a name the 2014 vocabulary lacks counts"
    assert not counts("苏醒", ORDINARY), "a name that is also a verb is read as the verb"
    assert not counts("面包", ORDINARY)
    assert not counts("京", ORDINARY), "a single character never counts"
    assert not counts("666", ORDINARY)
    assert not counts("B站", ORDINARY), "a keystroke cannot type the Latin letter"


def test_a_title_loses_its_disambiguator() -> None:
    assert entry(" 吉尔伽美什(作品) ") == "吉尔伽美什"
    assert entry("长城（电影）") == "长城"
    assert entry("北京") == "北京"


def test_the_first_source_listing_an_entry_decides_its_category(tmp_path: Path) -> None:
    lexicon = build(
        [
            Source(name="people", category="person", entries=("刘德华", "苏醒", "北京")),
            Source(name="places", category="place", entries=("北京", "上海")),
        ],
        ORDINARY,
    )
    assert lexicon == {"刘德华": "person", "北京": "person", "上海": "place"}
    out = tmp_path / "knowledge.tsv"
    write(lexicon, out)
    assert out.read_text(encoding="utf-8") == "上海\tplace\n刘德华\tperson\n北京\tperson\n"


def _write(path: Path, lines: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines))
    return path


def _eval_set(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "eval.jsonl",
        [
            {"pinyin": "zhongguo", "text": "中国"},
            {"pinyin": "renmin", "text": "人民"},
            {"pinyin": "yinhang", "text": "银行", "context": "我要去"},
        ],
    )


def test_a_row_is_the_shape_the_rust_side_reads() -> None:
    assert json.loads(DumpRow(record=3, text="中国", hypotheses=("中国",)).as_json()) == {
        "record": 3,
        "text": "中国",
        "hypotheses": [{"text": "中国", "score": 0.0}],
    }
    assert json.loads(DumpRow(record=3, text="中国", hypotheses=()).as_json())["hypotheses"] == []


def test_the_generator_answers_the_records_of_its_dump(tmp_path: Path) -> None:
    eval_set = _eval_set(tmp_path)
    dump = _write(
        tmp_path / "dump.jsonl",
        [
            {"record": 0, "text": "中国", "hypotheses": [{"text": "钟国", "score": -1.0}]},
            {"record": 2, "text": "银行", "hypotheses": []},
        ],
    )
    answers = _write(
        tmp_path / "answers.jsonl",
        [{"record": 0, "answer": "中国"}, {"record": 1, "answer": "人民"}],
    )
    rows = answer_rows(dump, eval_set, answers)
    # Record 1 was not asked about; record 2 was asked and went unanswered.
    assert rows == [
        DumpRow(record=0, text="中国", hypotheses=("中国",)),
        DumpRow(record=2, text="银行", hypotheses=()),
    ]
    out = tmp_path / "out.jsonl"
    assert write_dump(rows, out) == 2
    assert out.read_text(encoding="utf-8").count("\n") == 2


def test_a_journal_keeps_its_slice_and_a_failure_is_unanswered(tmp_path: Path) -> None:
    eval_set = _eval_set(tmp_path)
    journal = _write(
        tmp_path / "journal.jsonl",
        [
            {"engine": "sogou"},
            {"record": 0, "committed": "中国", "wall_ms": 5},
            {"record": 1, "committed": "", "wall_ms": 20000, "failed": True},
            {"record": 2, "committed": "银杏", "wall_ms": 5},
        ],
    )
    rows = journal_rows(journal, eval_set, "all")
    assert rows == [
        DumpRow(record=0, text="中国", hypotheses=("中国",)),
        DumpRow(record=1, text="人民", hypotheses=()),
        DumpRow(record=2, text="银行", hypotheses=("银杏",)),
    ]
    test = set(
        slice_indices(
            [json.loads(line) for line in eval_set.read_text().splitlines()], "test", DEV_SHARE
        )
    )
    assert {row.record for row in journal_rows(journal, eval_set, "test")} == test


def test_a_rime_report_must_come_from_the_same_set(tmp_path: Path) -> None:
    eval_set = _eval_set(tmp_path)
    report = tmp_path / "rime.json"
    report.write_text(
        json.dumps(
            {"typed": [{"record": 1, "expected": "人民", "committed": "人们", "first_page": []}]}
        )
    )
    assert report_rows(report, eval_set) == [DumpRow(record=1, text="人民", hypotheses=("人们",))]
    report.write_text(
        json.dumps({"typed": [{"record": 1, "expected": "人们", "committed": "人们"}]})
    )
    with pytest.raises(ValueError, match="another set"):
        report_rows(report, eval_set)
