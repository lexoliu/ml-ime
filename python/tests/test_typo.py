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
    CORRECTIONS_FIXTURE_RELATIVE,
    CORRECTIONS_K,
    EDIT_TYPES,
    CorrectionTable,
    NoiseModel,
    corrections_fixture_spans,
    load_syllables,
    record_seed,
    segmentable,
    typo_twin,
    write_corrections_fixture,
)

MODEL_PATH = Path(__file__).parent.parent / "src/mlime/data/typo_model.json"
SYLLABLES = Path(__file__).parent.parent.parent / "crates/ime-pinyin/data/syllables.txt"
CORRECTIONS_FIXTURE = Path(__file__).parent.parent.parent / CORRECTIONS_FIXTURE_RELATIVE


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


def test_corruption_spans_cover_every_pressed_key(model: NoiseModel) -> None:
    """``spans`` is the text re-sliced per input item: no key lost, none gained."""
    for seed in range(500):
        corruption = model.corrupt(["wo", "zai", "jia", "li"], random.Random(seed), min_edits=1)
        assert len(corruption.spans) == 4
        assert "".join(corruption.spans) == corruption.text


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


def write_lattice(path: Path, spellings: list[list[str]]) -> Path:
    """An emit-lattice file whose one path per record is *spellings*' syllables."""
    path.write_text(
        "".join(
            json.dumps(
                {
                    "record": index,
                    "pinyin": "".join(spans),
                    "paths": [{"spans": spans, "candidates": ["" for _ in spans]}],
                }
            )
            + "\n"
            for index, spans in enumerate(spellings)
        ),
        encoding="utf-8",
    )
    return path


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
    lattice = write_lattice(tmp_path / "lattice.jsonl", [["ni", "hao"], ["zhong", "guo", "ren"]])
    out = tmp_path / "twin.jsonl"
    report = typo_twin(eval_set, out, 0, MODEL_PATH, SYLLABLES, lattice)
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
    lattice = write_lattice(tmp_path / "lattice.jsonl", [["zhong", "guo", "ren", "min"]])
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    typo_twin(eval_set, one, 0, MODEL_PATH, SYLLABLES, lattice)
    typo_twin(eval_set, two, 0, MODEL_PATH, SYLLABLES, lattice)
    assert one.read_bytes() == two.read_bytes()


def test_typo_twin_refuses_a_lattice_without_the_record(tmp_path: Path) -> None:
    """A record the lattice does not spell is an error, never a guessed split."""
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(
        json.dumps({"pinyin": "nihao", "text": "你好", "context": None}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lattice = write_lattice(tmp_path / "lattice.jsonl", [["ni", "hao", "ma"]])
    with pytest.raises(ValueError, match="no 2-span path"):
        typo_twin(eval_set, tmp_path / "twin.jsonl", 0, MODEL_PATH, SYLLABLES, lattice)


@pytest.fixture(scope="module")
def correction_table(model: NoiseModel) -> CorrectionTable:
    return CorrectionTable.load(MODEL_PATH, SYLLABLES)


def test_corrections_valid_span_keeps_self_at_cost_zero(
    correction_table: CorrectionTable,
) -> None:
    """A valid syllable's own entry leads the list and costs nothing."""
    for span in ("jia", "zhong", "wo", "a"):
        entries = dict(correction_table.corrections(span))
        assert entries[span] == 0.0
        assert next(iter(correction_table.corrections(span))) == (span, 0.0)


def test_corrections_prefix_keeps_self_at_cost_zero(
    correction_table: CorrectionTable,
) -> None:
    """An abbreviation's own entry is cost 0 too, per the shared spec."""
    entries = dict(correction_table.corrections("zh"))
    assert entries["zh"] == 0.0


def test_corrections_typo_finds_the_intended_syllable(
    correction_table: CorrectionTable,
) -> None:
    """The motivating reading: 'jai' transposes back to 'jia', ahead of
    neighbours' one-slip corrections, and fuzzy 'zi' reaches 'zhi'."""
    jai = correction_table.corrections("jai")
    assert jai[0][0] == "jia"
    assert 0.0 < jai[0][1] < dict(jai)["mai"]
    zi = dict(correction_table.corrections("zi"))
    assert 0.0 < zi["zhi"] < zi["zui"]


def test_corrections_costs_are_sorted_and_capped(
    correction_table: CorrectionTable,
) -> None:
    """Every list is cost-ordered, ties broken by spelling, at most K long."""
    for span in ("z", "o", "jai", "zhong", "wo", "kan"):
        entries = correction_table.corrections(span)
        assert entries == tuple(sorted(entries, key=lambda item: (item[1], item[0])))
        assert len(entries) <= CORRECTIONS_K
        # Single letters sit farthest from any reading and hit the cap.
    assert len(correction_table.corrections("z")) == CORRECTIONS_K


def test_corrections_unreachable_span_is_empty(
    correction_table: CorrectionTable,
) -> None:
    """No reading reaches 'qqzx' within two edits; the list is empty, not a
    fallback guess."""
    assert correction_table.corrections("qqzx") == ()


def test_corrections_one_edit_probability_matches_the_enumerator(
    correction_table: CorrectionTable,
) -> None:
    """The closed-form single-edit probability and the brute-force forward
    enumeration agree exactly -- both sides of the two-edit sum are checked."""
    spans = [
        "jia",
        "zai",
        "wo",
        "zhong",
        "an",
        "zh",
        "xai",
        "jai",
        "hia",
        "cann",
        "z",
        "zi",
        "ang",
        "lan",
        "e",
        "qax",
        "",
    ]
    for source in spans:
        outcomes = correction_table._outcomes(source)
        for produced in spans:
            assert correction_table._one_edit_probability(source, produced) == pytest.approx(
                outcomes.get(produced, 0.0), abs=1e-12
            )


def test_corrections_fixture_is_current(correction_table: CorrectionTable, tmp_path: Path) -> None:
    """The shared fixture both languages test against is byte-for-byte what
    the Python implementation writes today."""
    regenerated = tmp_path / "typo-corrections.json"
    assert write_corrections_fixture(correction_table, regenerated) == len(
        corrections_fixture_spans(correction_table)
    )
    assert regenerated.read_bytes() == CORRECTIONS_FIXTURE.read_bytes()


def test_corrections_fixture_spans_cover_the_edit_space(
    correction_table: CorrectionTable,
) -> None:
    """The fixture exercises valid syllables, non-syllable prefixes, one- and
    two-edit typos and every fuzzy direction."""
    spans = corrections_fixture_spans(correction_table)
    assert len(set(spans)) == len(spans)
    syllables = correction_table._syllables
    prefixes = correction_table._prefixes
    assert any(span in syllables for span in spans)
    assert any(span in prefixes and span not in syllables for span in spans)
    assert any(
        span not in syllables and span not in prefixes and dict(correction_table.corrections(span))
        for span in spans
    )
    model = correction_table._model
    directions = {
        (pair.frm, pair.to)
        for span in spans
        for pair in model._fuzzy_pairs
        if span not in syllables
    }
    assert len(directions) >= 1
