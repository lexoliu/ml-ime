"""The exported file is the only thing the Rust harness sees, so its shape is a contract.

`pinyin` must be exactly what a user types -- toneless syllables, no separators --
because re-segmenting that string is half of what the harness evaluates. And the
draw must be reproducible from a seed, or two runs of the same evaluation are not
comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from mlime.data.corpus import SAMPLE_SCHEMA
from mlime.data.export import (
    EvalItem,
    allocate,
    eligible,
    export_eval_set,
    export_ngram_corpus,
    read_exclusions,
    sample_rows,
    to_item,
)


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _row(
    identifier: str,
    source: str,
    text: str,
    characters: list[str],
    readings: list[str],
    agree_all: bool = True,
    context: str | None = None,
) -> dict[str, object]:
    return {
        "id": identifier,
        "source": source,
        "text": text,
        "context": context,
        "characters": characters,
        "g2pw": readings,
        "llm": readings,
        "agree": [True] * len(characters),
        "agree_all": agree_all,
    }


def test_disputed_rows_never_reach_the_evaluation_set() -> None:
    """A disputed reading would score a correct conversion as wrong."""
    frame = _frame(
        [
            _row("a", "wiki", "重要", ["重", "要"], ["zhong4", "yao4"]),
            _row("b", "wiki", "还钱", ["还", "钱"], ["hai2", "qian2"], agree_all=False),
        ]
    )
    assert eligible(frame)["id"].to_list() == ["a"]


def test_targets_holding_punctuation_are_left_to_the_training_set() -> None:
    """A comma has no keystrokes behind it, so the syllable count would not match."""
    frame = _frame(
        [
            _row("a", "wiki", "重要，好", ["重", "要", "好"], ["zhong4", "yao4", "hao3"]),
            _row("b", "wiki", "重要好", ["重", "要", "好"], ["zhong4", "yao4", "hao3"]),
        ]
    )
    assert eligible(frame)["id"].to_list() == ["b"]


def test_the_pinyin_field_is_what_the_user_types() -> None:
    item = to_item(
        _row("a", "wiki", "绿色重要", ["绿", "色", "重", "要"], ["lv4", "se4", "zhong4", "yao4"])
    )
    assert item == EvalItem(pinyin="lvsezhongyao", text="绿色重要", context=None)


def test_a_row_whose_readings_do_not_cover_its_target_raises() -> None:
    with pytest.raises(ValueError, match="against 1 syllables"):
        to_item(_row("a", "wiki", "重要", ["重", "要"], ["zhong4"]))


def test_quotas_are_even_when_every_source_can_fill_them() -> None:
    assert allocate({"wiki": 100, "news": 100, "dialogue": 100}, 30) == {
        "wiki": 10,
        "news": 10,
        "dialogue": 10,
    }


def test_a_source_that_runs_short_does_not_shrink_the_export() -> None:
    """A small dialogue slice must not cap the size of the whole evaluation set."""
    assert allocate({"wiki": 100, "news": 100, "dialogue": 2}, 30) == {
        "wiki": 14,
        "news": 14,
        "dialogue": 2,
    }


def test_asking_for_more_than_exists_raises() -> None:
    with pytest.raises(ValueError, match="need 30"):
        allocate({"wiki": 10, "news": 10}, 30)


def _population() -> pl.DataFrame:
    return _frame(
        [
            _row(
                f"{source}{index:02d}",
                source,
                "重要好",
                ["重", "要", "好"],
                ["zhong4", "yao4", "hao3"],
            )
            for source in ("wiki", "news", "dialogue")
            for index in range(20)
        ]
    )


def test_the_same_seed_draws_the_same_rows() -> None:
    first = sample_rows(_population(), 9, seed=7)["id"].to_list()
    second = sample_rows(_population(), 9, seed=7)["id"].to_list()
    assert first == second
    assert sample_rows(_population(), 9, seed=8)["id"].to_list() != first


def test_the_draw_is_stratified_across_sources() -> None:
    drawn = sample_rows(_population(), 9, seed=7)
    assert dict(drawn["source"].value_counts().iter_rows()) == {
        "wiki": 3,
        "news": 3,
        "dialogue": 3,
    }


def test_the_exported_file_is_one_json_object_per_line(tmp_path: Path) -> None:
    out = tmp_path / "eval.jsonl"
    written = export_eval_set(_population(), out, size=6, seed=3)
    assert written == 6
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    for line in lines:
        record = json.loads(line)
        assert set(record) == {"pinyin", "text", "context"}
        assert record["pinyin"] == "zhongyaohao"
        assert record["text"] == "重要好"
        assert record["context"] is None


def test_context_survives_the_export(tmp_path: Path) -> None:
    """The whole product thesis is that this field helps, so it cannot be dropped here."""
    out = tmp_path / "eval.jsonl"
    frame = _frame(
        [
            _row(
                "a",
                "wiki",
                "重要好",
                ["重", "要", "好"],
                ["zhong4", "yao4", "hao3"],
                context="上下文",
            )
        ]
    )
    export_eval_set(frame, out, size=1, seed=0)
    assert json.loads(out.read_text(encoding="utf-8"))["context"] == "上下文"


def _shard(directory: Path, prefix: str, texts: list[str]) -> None:
    """Write one samples shard of the shape ``corpus prepare`` leaves behind."""
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {"id": f"{prefix}{index}", "source": prefix, "text": text, "context": None}
            for index, text in enumerate(texts)
        ],
        schema=SAMPLE_SCHEMA,
    ).write_parquet(directory / f"{prefix}-00000.parquet")


def _jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_the_held_out_sentences_are_absent_from_the_training_text(tmp_path: Path) -> None:
    """A sentence scored at evaluation time must not have been trained on."""
    samples = tmp_path / "samples"
    _shard(samples, "wiki", ["重要好", "绿色东西"])
    _shard(samples, "news", ["他还了钱"])
    out = tmp_path / "corpus.txt"
    held_out = read_exclusions([_jsonl(tmp_path / "eval.jsonl", [{"text": "绿色东西"}])])
    assert export_ngram_corpus(samples, out, held_out) == 2
    assert out.read_text(encoding="utf-8").splitlines() == ["他还了钱", "重要好"]


def test_an_exclusion_that_matches_nothing_raises(tmp_path: Path) -> None:
    """Silently holding nothing out would put the evaluation sentences back into training."""
    samples = tmp_path / "samples"
    _shard(samples, "wiki", ["重要好"])
    held_out = read_exclusions([_jsonl(tmp_path / "eval.jsonl", [{"text": "从未出现过的句子"}])])
    with pytest.raises(ValueError, match="not in"):
        export_ngram_corpus(samples, tmp_path / "corpus.txt", held_out)


def test_the_exported_evaluation_set_is_itself_a_usable_exclusion_file(tmp_path: Path) -> None:
    """`export eval-set` writes `pinyin`/`text`/`context`; only `text` is read back."""
    eval_path = tmp_path / "eval.jsonl"
    export_eval_set(_population(), eval_path, size=3, seed=7)
    assert read_exclusions([eval_path]) == frozenset({"重要好"})


def test_every_source_shard_reaches_the_corpus(tmp_path: Path) -> None:
    """The samples directory holds one shard series per source; dropping one would skew the mix."""
    samples = tmp_path / "samples"
    _shard(samples, "dialogue", ["你好啊"])
    _shard(samples, "news", ["他还了钱"])
    _shard(samples, "wiki", ["重要好"])
    out = tmp_path / "corpus.txt"
    assert export_ngram_corpus(samples, out, frozenset()) == 3
    assert sorted(out.read_text(encoding="utf-8").splitlines()) == sorted(
        ["你好啊", "他还了钱", "重要好"]
    )


def test_a_line_without_a_text_field_raises(tmp_path: Path) -> None:
    path = _jsonl(tmp_path / "eval.jsonl", [{"pinyin": "zhongyaohao"}])
    with pytest.raises(ValueError, match="no usable `text` field"):
        read_exclusions([path])


def test_a_samples_directory_with_no_shards_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no sample shards"):
        export_ngram_corpus(tmp_path / "samples", tmp_path / "corpus.txt", frozenset())
