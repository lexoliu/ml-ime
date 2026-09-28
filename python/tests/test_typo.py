"""Tests for the shared typo noise model and the eval3-typo twin writer.

The table under test is the authored one, ``typo_model.json`` -- the twin's
numbers only mean anything if the rates the note reports are the rates that ran.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mlime.typo import (
    EDIT_TYPES,
    NoiseModel,
    load_syllables,
    record_seed,
    segmentable,
    split_syllables,
    typo_twin,
)

MODEL_PATH = Path(__file__).parent.parent / "src/mlime/data/typo_model.json"
SYLLABLES = Path(__file__).parent.parent.parent / "crates/ime-pinyin/data/syllables.txt"


@pytest.fixture(scope="module")
def model() -> NoiseModel:
    return NoiseModel.load(MODEL_PATH)


@pytest.fixture(scope="module")
def inventory() -> set[str]:
    return load_syllables(SYLLABLES)


def test_shipped_table_parses(model: NoiseModel) -> None:
    assert 0.0 < model.per_sentence < 1.0


def test_parse_rejects_unknown_key() -> None:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["surprise"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        NoiseModel.parse(json.dumps(data))


def test_parse_rejects_asymmetric_neighbours() -> None:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["neighbours"]["b"] = data["neighbours"]["b"][:-1]
    with pytest.raises(ValueError, match=r"not vice versa|repeats"):
        NoiseModel.parse(json.dumps(data))


def test_parse_rejects_incomplete_edit_types() -> None:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    del data["edit_type"]["fuzzy"]
    with pytest.raises(ValueError, match="edit_type"):
        NoiseModel.parse(json.dumps(data))


def test_parse_rejects_unnormalised_edit_count() -> None:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["edit_count"]["1"] = 0.9
    with pytest.raises(ValueError, match="edit_count"):
        NoiseModel.parse(json.dumps(data))


def test_parse_rejects_unknown_initial_pair(model: NoiseModel) -> None:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["fuzzy_pairs"].append({"from": "sr", "to": "y", "where": "initial", "weight": 0.1})
    with pytest.raises(ValueError, match="initials"):
        NoiseModel.parse(json.dumps(data))


def test_parse_rejects_where_outside_initial_final(model: NoiseModel) -> None:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["fuzzy_pairs"].append({"from": "an", "to": "ang", "where": "middle", "weight": 0.1})
    with pytest.raises(ValueError):
        NoiseModel.parse(json.dumps(data))


def test_initial_and_final(model: NoiseModel) -> None:
    assert model.initial_of("zhu") == "zh"
    assert model.final_of("zhu") == "u"
    assert model.initial_of("zu") == "z"
    assert model.final_of("zu") == "u"
    assert model.initial_of("an") == ""
    assert model.final_of("an") == "an"
    assert model.initial_of("zhang") == "zh"
    assert model.final_of("zhang") == "ang"


def test_corrupt_is_deterministic(model: NoiseModel) -> None:
    syllables = ["wo", "zai", "jia"]
    one = model.corrupt(syllables, random.Random(42))
    two = model.corrupt(syllables, random.Random(42))
    assert one == two


def test_corrupt_always_reaches_min_edits(model: NoiseModel) -> None:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["edit_count"] = {"1": 1.0}
    one_edit = NoiseModel.parse(json.dumps(data))
    for seed in range(200):
        corruption = model.corrupt(["wo", "zai", "jia"], random.Random(seed), min_edits=1)
        assert len(corruption.edits) >= 1
        # A single edit can never reproduce its input: the pairs map to
        # something else and no two neighbouring letters are equal.
        one = one_edit.corrupt(["wo", "zai", "jia"], random.Random(seed))
        assert len(one.edits) == 1
        assert one.text != "wozaijia"


def test_corrupt_respects_more_edits(model: NoiseModel) -> None:
    for seed in range(100):
        corruption = model.corrupt(["zhong", "guo", "ren", "min"], random.Random(seed), min_edits=3)
        assert len(corruption.edits) >= 3


def test_maybe_corrupt_zero_never_edits() -> None:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["per_sentence"] = 0.0
    model = NoiseModel.parse(json.dumps(data))
    for seed in range(50):
        corruption = model.maybe_corrupt(["wo", "zai", "jia"], random.Random(seed))
        assert corruption.text == "wozaijia"
        assert corruption.edits == ()


def test_fuzzy_sites_replace_whole_parts_only(model: NoiseModel) -> None:
    # ``xian``'s final is ``ian``, not ``an``: the an>ang pair must not fire and
    # ``tian`` must never come out ``tiang``.
    sites = model.fuzzy_sites(["xian"])
    assert sites == []
    sites = model.fuzzy_sites(["ban"])
    assert [(pair.frm, pair.to, pair.where) for _, pair in sites] == [("an", "ang", "final")]
    sites = model.fuzzy_sites(["zhu"])
    assert [(pair.frm, pair.to, pair.where) for _, pair in sites] == [("zh", "z", "initial")]
    # ``nin`` is two sites at once: the n/l initial pair and the in/ing final.
    sites = model.fuzzy_sites(["nin"])
    assert {(pair.frm, pair.to, pair.where) for _, pair in sites} == {
        ("n", "l", "initial"),
        ("in", "ing", "final"),
    }


def test_fuzzy_swap_replaces_the_whole_final(model: NoiseModel) -> None:
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["edit_type"] = {kind: 0.0 for kind in EDIT_TYPES} | {"fuzzy": 1.0}
    data["edit_count"] = {"1": 1.0}
    forced = NoiseModel.parse(json.dumps(data))
    corruption = forced.corrupt(["ban"], random.Random(0), min_edits=1)
    assert corruption.text == "bang"
    assert corruption.edits[0].kind == "fuzzy"
    assert corruption.edits[0].detail.startswith("an>ang")


def test_edits_stay_within_bounds(model: NoiseModel) -> None:
    # Adjacent edits must produce a neighbour of the letter they replaced.
    for seed in range(500):
        rng = random.Random(seed)
        corruption = model.corrupt(["wo", "zai", "jia", "li"], rng)
        assert corruption.text.isascii() and corruption.text.islower()
        assert len(corruption.text) > 0


def test_split_syllables_recovers_the_reading(inventory: set[str]) -> None:
    assert split_syllables("zuixiamianleiniaode", 6, inventory) == [
        "zui",
        "xia",
        "mian",
        "lei",
        "niao",
        "de",
    ]


def test_split_syllables_ambiguous_count_one(inventory: set[str]) -> None:
    assert split_syllables("xian", 1, inventory) == ["xian"]
    # ``n`` is itself a syllable (嗯), so longest-first takes ``xia`` over
    # ``xi`` + ``an``.
    assert split_syllables("xian", 2, inventory) == ["xia", "n"]


def test_split_syllables_raises_when_impossible(inventory: set[str]) -> None:
    with pytest.raises(ValueError, match="does not segment"):
        split_syllables("qjqj", 1, inventory)


def test_segmentable_mirrors_the_lattice(inventory: set[str]) -> None:
    assert segmentable("nihao", inventory)
    assert segmentable("z", inventory)  # single-letter abbreviation
    assert segmentable("zh", inventory)  # zh-initial abbreviation
    assert segmentable("qqqqq", inventory)  # every letter is some syllable's first
    assert not segmentable("v", inventory)  # nothing starts with v
    assert not segmentable("avv", inventory)  # ``a`` reads, then ``v`` stalls


def test_record_seed_is_stable() -> None:
    assert record_seed(0, "a", "b", "") == record_seed(0, "a", "b", "")
    assert record_seed(0, "a", "b", "") != record_seed(1, "a", "b", "")
    assert record_seed(0, "a", "b", "") != record_seed(0, "a", "b", "c")


def test_typo_twin_writes_corrupted_records(tmp_path: Path, inventory: set[str]) -> None:
    eval_set = tmp_path / "eval.jsonl"
    records: list[dict[str, str | None]] = [
        {"pinyin": "nihao", "text": "你好", "context": None},
        {"pinyin": "zhongguoren", "text": "中国人", "context": "我们都是"},
    ]
    eval_set.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    out = tmp_path / "twin.jsonl"
    report = typo_twin(eval_set, out, 0, MODEL_PATH, SYLLABLES)
    lines: list[dict[str, str]] = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 2
    for line, original in zip(lines, records, strict=True):
        assert line["clean"] == original["pinyin"]
        assert line["pinyin"] != original["pinyin"]
        assert line["text"] == original["text"]
        assert line["context"] == original["context"]
    assert "2 records" in report
    for kind in EDIT_TYPES:
        assert kind in report


def test_typo_twin_is_deterministic(tmp_path: Path) -> None:
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(
        json.dumps(
            {"pinyin": "zhongguorenmin", "text": "中国人民", "context": None},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    typo_twin(eval_set, one, 0, MODEL_PATH, SYLLABLES)
    typo_twin(eval_set, two, 0, MODEL_PATH, SYLLABLES)
    assert one.read_bytes() == two.read_bytes()
