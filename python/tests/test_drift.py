"""The answer kinds and the rarity of substituted characters, counted by hand."""

import json
from pathlib import Path

import polars as pl
import pytest

from mlime.drift import FrequencyRanks, measure
from mlime.rescore import DumpedRecord, Hypothesis

RECORDS = [
    ("zhongguo", "中国", ("中国",)),  # right
    ("yinhang", "银行", ("因行", "银行")),  # right length, 因 substituted
    ("tianqi", "天气", ("天",)),  # another length
    ("renmin", "人民", ()),  # unanswered: another length
    ("beijing", "北京", ("北景",)),  # right length, 景 unseen
]


def _eval_set(tmp_path: Path) -> Path:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "".join(
            json.dumps({"pinyin": pinyin, "text": text, "context": None}, ensure_ascii=False) + "\n"
            for pinyin, text, _ in RECORDS
        ),
        encoding="utf-8",
    )
    return path


def _dump() -> list[DumpedRecord]:
    return [
        DumpedRecord(
            index=index,
            text=text,
            pinyin=pinyin,
            context=None,
            hypotheses=tuple(Hypothesis(text=h, score=-float(i)) for i, h in enumerate(found)),
        )
        for index, (pinyin, text, found) in enumerate(RECORDS)
    ]


def _ranks(tmp_path: Path) -> FrequencyRanks:
    # Read in name order: 中 ×3, 国 ×2, then 行 and 因 once each, 行 first.
    pl.DataFrame({"text": ["国行"]}).write_parquet(tmp_path / "b.parquet")
    pl.DataFrame({"text": ["中中中国", "因"]}).write_parquet(tmp_path / "c.parquet")
    return FrequencyRanks.from_shards(sorted(tmp_path.glob("*.parquet"), reverse=True))


def test_ranks_order_by_frequency_then_first_appearance(tmp_path: Path) -> None:
    ranks = _ranks(tmp_path)
    assert ranks.ranks == {"中": 0, "国": 1, "行": 2, "因": 3}
    assert ranks.characters == 7
    assert ranks.rank("景") == 4


def test_measure_counts_kinds_and_substitutions(tmp_path: Path) -> None:
    drift = measure(_dump(), _eval_set(tmp_path), _ranks(tmp_path), "all", 0.0905)
    assert (drift.right, drift.same_length, drift.other_length) == (1, 2, 2)
    assert drift.substituted == 2
    assert drift.unseen == 1
    assert drift.unseen_share == 0.5
    assert drift.median_rank == 3.5


def test_measure_refuses_a_dump_missing_slice_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lacks 1 records"):
        measure(_dump()[:-1], _eval_set(tmp_path), _ranks(tmp_path), "all", 0.0905)
