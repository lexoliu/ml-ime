"""Tests for the shared key-search fixture the Rust search reproduces (#265).

The fixture is held to its own definitions with an independent SplitMix64, to
the search that wrote it, and to the file on disk, byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mlime.legality import Legality
from mlime.train.key_search_fixture import (
    LEGALITY_RECORDS_RELATIVE,
    SEARCH_FIXTURE_RELATIVE,
    UNBOUNDED_MAX_KEYS,
    HashedTransition,
    draw,
    emissions,
    write_search_fixture,
)

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "crates/ime-pinyin/data"
FIXTURE = ROOT / SEARCH_FIXTURE_RELATIVE
MASK = (1 << 64) - 1


def splitmix(value: int) -> int:
    """SplitMix64's finaliser in plain integers, independent of the numpy one."""
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def reference_draw(seed: int, value: int) -> float:
    return -(splitmix(seed ^ value) >> 44) / 65536.0


@pytest.fixture(scope="module")
def legality() -> Legality:
    return Legality.load(
        None,
        char_table=DATA / "char_pinyin.tsv",
        typo_table=DATA / "typo.json",
        syllables=DATA / "syllables.txt",
    )


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return loaded


def test_the_draws_follow_their_definition() -> None:
    seed = 0x0123456789ABCDEF
    values = np.array([0, 1, 7 << 32, (3 << 32) | 5, MASK], dtype=np.uint64)
    got = draw(seed, values)
    assert got.tolist() == [reference_draw(seed, int(value)) for value in values]
    assert all(-16.0 < each <= 0.0 for each in got.tolist())


def test_the_emissions_are_the_draws_with_the_blank_last() -> None:
    seed, keys, width = 0xABCDEF, 4, 9
    table = emissions(keys, width, seed, flat=False)
    assert table.shape == (keys, width + 1) and table.dtype == np.float32
    for t in range(keys):
        for c in range(width + 1):
            assert float(table[t, c]) == reference_draw(seed, (t << 32) | c)
    flat = emissions(keys, width, seed, flat=True)
    for t in range(keys):
        assert len(set(flat[t, :width].tolist())) == 1
        assert float(flat[t, width]) == reference_draw(seed, (t << 32) | width)


def test_the_transition_rows_follow_the_rolling_hash() -> None:
    width, seed = 6, 0x5EED
    transition = HashedTransition(width, seed, flat=False)
    prefix = (4, 0, 2)
    rolling = 0
    for each in prefix:
        rolling = splitmix(rolling ^ each)
    [row] = transition([0], [prefix])
    assert row.tolist() == [reference_draw(seed, rolling ^ c) for c in range(width)]
    assert transition.calls == [[list(prefix)]]
    [flat] = HashedTransition(width, seed, flat=True)([0], [prefix])
    assert not flat.any()


def test_the_fixture_is_current(legality: Legality, tmp_path: Path) -> None:
    """The committed file is byte for byte what this side writes from its own inputs."""
    regenerated = tmp_path / "key-search.json"
    runs = write_search_fixture(
        legality, DATA / "char_pinyin.tsv", ROOT / LEGALITY_RECORDS_RELATIVE, regenerated
    )
    assert runs > 0
    assert regenerated.read_bytes() == FIXTURE.read_bytes()


def test_the_fixture_covers_the_settings_and_the_ties(document: dict[str, Any]) -> None:
    names = {record["name"] for record in document["records"]}
    assert {"hm", "xian", "shuijiap"} <= names
    assert sum(name.startswith("tie-") for name in names) >= 2
    for record in document["records"]:
        keys = len(record["keys"])
        grid = {(s["combine"], s["beam"], s["gamma"]) for s in record["settings"]}
        beams = {8, None} if keys <= UNBOUNDED_MAX_KEYS else {8}
        assert grid == {
            (combine, beam, gamma)
            for combine in ("sum", "viterbi")
            for beam in beams
            for gamma in (0.0, 0.5)
        }, record["name"]
        for setting in record["settings"]:
            assert setting["hypotheses"], record["name"]
            if setting["beam"] is not None:
                assert len(setting["hypotheses"]) <= setting["beam"]
            assert setting["calls"][0] == [[]]
            for hypothesis in setting["hypotheses"]:
                assert len(hypothesis["ids"]) == len(hypothesis["cuts"])
                assert hypothesis["cuts"][-1] == keys
                assert hypothesis["cuts"] == sorted(set(hypothesis["cuts"]))
    assert FIXTURE.stat().st_size < 1_000_000


def test_a_tie_record_orders_equal_scores_stably(document: dict[str, Any]) -> None:
    """Without a beam every hypothesis of the flat record ties, and ids come out in order."""
    [record] = [r for r in document["records"] if r["name"] == "tie-small"]
    unbounded = [s for s in record["settings"] if s["beam"] is None and s["gamma"] == 0.0]
    for setting in unbounded:
        scores = [hypothesis["score"] for hypothesis in setting["hypotheses"]]
        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) < len(scores)
