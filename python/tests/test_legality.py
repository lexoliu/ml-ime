"""Tests for the legality automaton over keys (#216 §3).

The rule tests each fail when their rule is removed; the fixture test pins this
side to the file the Rust test reads, regenerated from the fixture's own inputs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from mlime.legality import (
    LEGALITY_FIXTURE_RELATIVE,
    LIVE,
    MAX_ARC_KEYS,
    OFFLINE,
    Legality,
    LegalityOptions,
    lattice_digest,
    write_legality_fixture,
)
from mlime.typo import CorrectionTable, corrections_fixture_spans

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "crates/ime-pinyin/data"
FIXTURE = ROOT / LEGALITY_FIXTURE_RELATIVE

#: Offline, with corrections kept off spans that are syllables.
STRICT = LegalityOptions(unfinished_tail=False, correct_syllables=False)


@pytest.fixture(scope="module")
def legality() -> Legality:
    return Legality.load(
        None,
        char_table=DATA / "char_pinyin.tsv",
        typo_table=DATA / "typo.json",
        syllables=DATA / "syllables.txt",
    )


@pytest.fixture(scope="module")
def fixture() -> dict[str, list[dict[str, object]]]:
    document: dict[str, list[dict[str, object]]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return document


def fixture_records(fixture: dict[str, list[dict[str, object]]]) -> list[tuple[str, str]]:
    return [(str(row["keys"]), str(row["sentence"])) for row in fixture["records"]]


def correction_cost(legality: Legality, span: str, syllable: str) -> float:
    return dict(legality.corrections.corrections(span))[syllable]


def test_wmzyq_is_never_women_chi_shuijiao(legality: Legality) -> None:
    for options in (OFFLINE, LIVE):
        assert legality.cost("wmzyq", "我们吃水饺", options) is None
    assert legality.cost("wmcsj", "我们吃水饺", OFFLINE) == 0.0


def test_a_typo_and_a_mixed_reading_are_both_legal(legality: Legality) -> None:
    jiao = correction_cost(legality, "jiap", "jiao")
    assert jiao > 0.0
    assert legality.cost("shuijiap", "水饺", OFFLINE) == jiao
    assert legality.cost("shuijiap", "水家铺", OFFLINE) == 0.0


@pytest.mark.parametrize(
    ("keys", "sentence"),
    [
        ("sj", "水饺"),
        ("sk", "烧烤"),
        ("sl", "沙拉"),
        ("sn", "酸奶"),
        ("zgrm", "中国人民"),
        ("xian", "西安"),
        ("xian", "先"),
        ("hm", "噷"),
        ("hm", "好们"),
    ],
)
def test_the_section_three_table_holds(legality: Legality, keys: str, sentence: str) -> None:
    assert legality.cost(keys, sentence, OFFLINE) == 0.0


def test_a_full_syllable_reads_at_no_cost(legality: Legality) -> None:
    assert legality.cost("nihao", "你好", STRICT) == 0.0
    assert legality.cost("nihao", "你们", STRICT) is None


def test_an_initial_covers_every_syllable_it_begins(legality: Legality) -> None:
    assert legality.cost("zg", "中国", STRICT) == 0.0
    assert legality.cost("zhg", "中国", STRICT) == 0.0
    assert legality.cost("chf", "吃饭", STRICT) == 0.0
    # Three keys standing for one character are no initial: only a correction
    # reads them, at a price.
    corrected = legality.cost("zhg", "中", STRICT)
    assert corrected is not None and corrected > 0.0


def test_an_initial_span_is_never_corrected(legality: Legality) -> None:
    for span in ("z", "zh"):
        assert "a" in dict(legality.corrections.corrections(span))
        for options in (OFFLINE, LIVE):
            assert legality.cost(span, "啊", options) is None


def test_the_unfinished_tail_reads_only_last_and_only_live(legality: Legality) -> None:
    assert legality.cost("guozho", "国中", LIVE) == 0.0
    assert legality.cost("guozho", "国中", OFFLINE) != 0.0
    assert legality.cost("zhoguo", "中国", LIVE) != 0.0


def test_a_typo_off_the_inventory_reads_through_its_correction(legality: Legality) -> None:
    assert legality.cost("wojaijia", "我家家", STRICT) == correction_cost(legality, "jai", "jia")


def test_corrections_on_syllables_follow_the_option(legality: Legality) -> None:
    zhao = correction_cost(legality, "zao", "zhao")
    assert legality.cost("zao", "找", OFFLINE) == zhao
    assert legality.cost("zao", "找", STRICT) is None
    assert legality.cost("zao", "早", STRICT) == 0.0


def test_every_key_is_consumed_and_none_twice(legality: Legality) -> None:
    assert legality.cost("aa", "啊啊", OFFLINE) == 0.0
    assert legality.cost("a", "啊啊", OFFLINE) is None
    assert legality.cost("ab", "啊", OFFLINE) is None


def test_a_polyphone_reads_under_any_of_its_readings(legality: Legality) -> None:
    for keys in ("zhong", "chong"):
        assert legality.cost(keys, "重", STRICT) == 0.0


def test_a_character_outside_the_emittable_set_is_illegal(legality: Legality) -> None:
    readings = {"中": ("zhong",), "国": ("guo",)}
    small = Legality(("国", "中"), readings, legality.corrections)
    assert small.ids("中国") == (1, 0)
    assert small.cost("zg", "中国", STRICT) == 0.0
    assert small.cost("zgr", "中国人", STRICT) is None
    with pytest.raises(ValueError, match="twice"):
        Legality(("中", "中"), readings, legality.corrections)
    with pytest.raises(ValueError, match="no readings"):
        Legality(("人",), readings, legality.corrections)


def test_keys_are_a_non_empty_alphabet_string(legality: Legality) -> None:
    for keys in ("", "ni3", "Ni"):
        with pytest.raises(ValueError, match="keys"):
            legality.lattice(keys, OFFLINE)


def covers(width: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Every way to cut *width* keys into *parts* spans of at most MAX_ARC_KEYS keys."""
    if parts == 0:
        if width == 0:
            yield ()
        return
    for first in range(1, min(width, MAX_ARC_KEYS) + 1):
        for rest in covers(width - first, parts - 1):
            yield (first, *rest)


def test_gold_arcs_are_exactly_the_legal_alignments(
    legality: Legality, fixture: dict[str, list[dict[str, object]]]
) -> None:
    """Brute force over every cover agrees with the enumeration, arc for arc."""
    cases = [
        ("shuijiap", "水饺"),
        ("shuijiap", "水家铺"),
        ("xian", "西安"),
        ("wojaijia", "我家家"),
        ("zhonggu", "中国"),
        ("zgrm", "中国人民"),
        *[(keys, text) for keys, text in fixture_records(fixture) if len(keys) <= 14],
    ]
    for options in (OFFLINE, LIVE):
        for keys, sentence in cases:
            ids = legality.ids(sentence)
            assert ids is not None
            arcs: set[tuple[int, int, int, float]] = set()
            best = math.inf
            for widths in covers(len(keys), len(ids)):
                start, total, path = 0, 0.0, []
                for position, width in enumerate(widths):
                    end = start + width
                    cost = legality.char_cost(
                        keys[start:end], end == len(keys), ids[position], options
                    )
                    if cost == math.inf:
                        break
                    path.append((position, start, end, cost))
                    total += cost
                    start = end
                else:
                    arcs.update(path)
                    best = min(best, total)
            gold = legality.gold(keys, sentence, options)
            if not arcs:
                assert gold is None, (keys, sentence)
                continue
            assert gold is not None, (keys, sentence)
            got = set(
                zip(
                    gold.position.tolist(),
                    gold.start.tolist(),
                    gold.end.tolist(),
                    gold.cost.tolist(),
                    strict=True,
                )
            )
            assert got == arcs, (keys, sentence)
            assert gold.best == pytest.approx(best, rel=1e-12, abs=1e-12)


def test_gold_arcs_are_lattice_arcs_and_densify(
    legality: Legality, fixture: dict[str, list[dict[str, object]]]
) -> None:
    for keys, sentence in [*fixture_records(fixture), ("shuijiap", "水饺")]:
        gold = legality.gold(keys, sentence, OFFLINE)
        assert gold is not None, (keys, sentence)
        ids = legality.ids(sentence)
        assert ids is not None
        lattice = legality.lattice(keys, OFFLINE)
        table = {
            (start, end, character): cost
            for start, end, character, cost in zip(
                lattice.start.tolist(),
                lattice.end.tolist(),
                lattice.character.tolist(),
                lattice.cost.tolist(),
                strict=True,
            )
        }
        for position, start, end, cost in zip(
            gold.position.tolist(),
            gold.start.tolist(),
            gold.end.tolist(),
            gold.cost.tolist(),
            strict=True,
        ):
            assert table[(start, end, ids[position])] == cost
        dense = gold.dense()
        assert dense.shape == (len(ids), len(keys), MAX_ARC_KEYS)
        assert np.isfinite(dense).sum() == gold.cost.size
        assert np.array_equal(
            dense[gold.position, gold.end - 1, gold.end - gold.start - 1], gold.cost
        )


def test_the_lattice_holds_only_arcs_on_complete_covers(legality: Legality) -> None:
    for keys in ("shuijiap", "wojaijia", "zgrm", "xianq"):
        for options in (OFFLINE, LIVE):
            lattice = legality.lattice(keys, options)
            edges = set(zip(lattice.start.tolist(), lattice.end.tolist(), strict=True))
            starts = {start for start, _ in edges} | {len(keys)}
            ends = {end for _, end in edges} | {0}
            for start, end in edges:
                assert start in ends and end in starts, (keys, start, end)
            assert np.all(np.diff(lattice.start) >= 0)


def reference_costs(table: CorrectionTable, span: str) -> dict[str, float]:
    """``_resolve_costs`` as it was before the reach index: every predecessor walked."""
    costs: dict[str, float] = {}
    predecessors = table._predecessors(span)
    origins = predecessors | set().union(
        *(table._predecessors(intermediate) for intermediate in predecessors)
    )
    for syllable in sorted(origins & table.syllables):
        probability = table._one_edit_probability(syllable, span) + sum(
            draw * table._one_edit_probability(intermediate, span)
            for intermediate, draw in table._outcomes(syllable).items()
        )
        if probability > 0.0:
            costs[syllable] = -math.log(probability)
    if span in table.syllables or span in table._prefixes:
        costs[span] = 0.0
    return costs


def test_the_reach_index_changes_no_cost(
    legality: Legality, fixture: dict[str, list[dict[str, object]]]
) -> None:
    """Bit for bit, in order, on the corrections fixture's spans and the records' substrings."""
    table = CorrectionTable(legality.corrections._model, legality.corrections.syllables)
    spans = list(corrections_fixture_spans(table))
    for keys, _ in fixture_records(fixture)[:6]:
        spans.extend(
            keys[start:end]
            for start in range(len(keys))
            for end in range(start + 2, min(len(keys), start + MAX_ARC_KEYS) + 1)
        )
    for span in dict.fromkeys(spans):
        want = reference_costs(table, span)
        got = table.costs(span)
        assert list(got.items()) == list(want.items()), span
        assert table.reaches(span) == bool(want), span


def test_the_fixture_is_current(
    legality: Legality, fixture: dict[str, list[dict[str, object]]], tmp_path: Path
) -> None:
    """The shared fixture is byte for byte what this side writes from its own inputs."""
    regenerated = tmp_path / "legality.json"
    write_legality_fixture(legality, fixture_records(fixture), regenerated)
    assert regenerated.read_bytes() == FIXTURE.read_bytes()


def test_the_digest_is_order_free(legality: Legality) -> None:
    lattice = legality.lattice("shuijiap", OFFLINE)
    order = np.arange(lattice.cost.size)[::-1]
    shuffled = type(lattice)(
        lattice.keys,
        lattice.start[order],
        lattice.end[order],
        lattice.character[order],
        lattice.cost[order],
    )
    assert lattice_digest(shuffled) == lattice_digest(lattice)
