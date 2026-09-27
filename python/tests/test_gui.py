"""The GUI harness's scoring: keyed slice selection and `evaluate` parity."""

import json
from pathlib import Path

import pytest

from mlime.gui import DEV_SHARE, digest, measure, read_results, slice_indices, write_report


def _eval_rows() -> list[dict[str, object]]:
    return [
        {"pinyin": "zhongguo", "text": "中国", "context": None},
        {"pinyin": "renmin", "text": "人民"},
        {"pinyin": "yinhang", "text": "银行", "context": "我要去"},
        {"pinyin": "beijing", "text": "北京", "context": None},
        {"pinyin": "tianqi", "text": "天气", "context": "今天的"},
    ]


def _write(tmp_path: Path, name: str, lines: list[dict[str, object]]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n")
    return path


def test_digest_matches_the_rust_layout() -> None:
    import hashlib

    expected = hashlib.blake2b(b"zhongguo\x00\xe4\xb8\xad\xe5\x9b\xbd\x00", digest_size=8)
    assert digest("zhongguo", "中国", None) == int.from_bytes(expected.digest(), "big")
    # A missing context is hashed as the empty string.
    assert digest("zhongguo", "中国", None) == digest("zhongguo", "中国", "")


def test_slice_indices_splits_on_the_share() -> None:
    rows = _eval_rows()
    assert slice_indices(rows, "dev", 0.0) == []
    assert slice_indices(rows, "test", 0.0) == list(range(5))
    assert slice_indices(rows, "test", 1.0) == []
    dev = slice_indices(rows, "dev", DEV_SHARE)
    test = slice_indices(rows, "test", DEV_SHARE)
    assert sorted(dev + test) == list(range(5))
    assert not set(dev) & set(test)


def test_measure_scores_like_eval_rime(tmp_path: Path) -> None:
    eval_set = _write(tmp_path, "eval.jsonl", _eval_rows())
    results = _write(
        tmp_path,
        "results.jsonl",
        [
            {"type": "meta", "engine": "apple", "input_source_id": "com.apple.inputmethod.X"},
            {"record": 0, "committed": "中国", "first_page": ["中国", "种过"], "wall_ms": 10},
            {"record": 1, "committed": "人名", "first_page": ["人名"], "wall_ms": 20},
            {"record": 2, "committed": "银行", "first_page": ["银行"], "wall_ms": 30},
            {
                "record": 3,
                "committed": "",
                "first_page": [],
                "wall_ms": 40,
                "candidate_window": False,
            },
            {"record": 4, "committed": "天气", "first_page": ["天气"], "wall_ms": 50},
        ],
    )
    meta, rows = read_results(results)
    assert meta["input_source_id"] == "com.apple.inputmethod.X"
    assert rows[3].candidate_window is False

    report = measure(results, eval_set, "apple", slice_="all")
    r = report.result
    assert r.records == 5
    assert r.top1 == pytest.approx(3 / 5)
    assert r.first_page_exact == pytest.approx(3 / 5)
    assert r.beam_top1 is None
    assert r.characters == 10
    # 人名 keeps one of two characters; the empty commit keeps none.
    assert r.characters_right == pytest.approx((2 + 1 + 2 + 0 + 2) / 10)
    assert r.length_mismatches == 1  # only the empty commit differs in length

    assert report.wall_ms == (10, 20, 30, 40, 50)
    rendered = report.render()
    assert "apple" in rendered and "top-1" in rendered

    out = tmp_path / "report.json"
    write_report(report, out)
    payload = json.loads(out.read_text())
    assert payload["engine"] == "apple"
    assert payload["result"]["records"] == 5
    assert payload["wall_ms"]["total"] == 150
    assert payload["typed"][3]["candidate_window"] is False


def test_measure_restricts_to_the_slice(tmp_path: Path) -> None:
    eval_set = _write(tmp_path, "eval.jsonl", _eval_rows())
    results = _write(
        tmp_path,
        "results.jsonl",
        [{"record": i, "committed": "中国", "first_page": [], "wall_ms": 1} for i in range(5)],
    )
    report = measure(results, eval_set, "apple", slice_="test", dev_share=0.0)
    assert report.result.records == 5


def test_measure_rejects_a_slice_with_no_rows(tmp_path: Path) -> None:
    eval_set = _write(tmp_path, "eval.jsonl", _eval_rows())
    results = _write(
        tmp_path, "results.jsonl", [{"record": 0, "committed": "x", "first_page": [], "wall_ms": 1}]
    )
    # dev_share 0 leaves no dev records, so scoring dev has nothing to say.
    with pytest.raises(ValueError, match="no records"):
        measure(results, eval_set, "apple", slice_="dev", dev_share=0.0)


def test_measure_refuses_a_mismatched_run_slice(tmp_path: Path) -> None:
    eval_set = _write(tmp_path, "eval.jsonl", _eval_rows())
    results = _write(
        tmp_path,
        "results.jsonl",
        [
            {"type": "meta", "engine": "apple", "slice": "dev"},
            {"record": 0, "committed": "中国", "first_page": [], "wall_ms": 1},
        ],
    )
    with pytest.raises(ValueError, match="--slice dev"):
        measure(results, eval_set, "apple", slice_="test")
    # A run typed with --slice all covers every scoring slice.
    results_all = _write(
        tmp_path,
        "results-all.jsonl",
        [
            {"type": "meta", "engine": "apple", "slice": "all"},
            {"record": 0, "committed": "中国", "first_page": [], "wall_ms": 1},
        ],
    )
    report = measure(results_all, eval_set, "apple", slice_="test", dev_share=0.0)
    assert report.result.records == 1
    # Runs written before --slice existed have no meta field and count as all.
    results_plain = _write(
        tmp_path,
        "results-plain.jsonl",
        [{"record": 0, "committed": "中国", "first_page": [], "wall_ms": 1}],
    )
    scored = measure(results_plain, eval_set, "apple", slice_="test", dev_share=0.0)
    assert scored.result.records == 1
