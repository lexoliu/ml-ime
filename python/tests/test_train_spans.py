"""The typed-span inventory is a generated table, so its contents are pinned here."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlime.train.spans import (
    TYPED_SPANS_PATH,
    SpanVocab,
    build,
    default_syllables_path,
    enumerate_spans,
    initial,
    read_syllables,
)

#: Every prefix of every one of the 425 syllables in `ime-pinyin`'s inventory.
#: The training plan estimated "~1.3k forms"; enumeration says 506, and the
#: enumeration is the thing the model is built against.
TYPED_SPAN_COUNT = 506


def test_inventory_size_is_pinned() -> None:
    spans = SpanVocab.load()
    assert len(spans) == TYPED_SPAN_COUNT


@pytest.mark.parametrize("span", ["a", "z", "zh", "zho", "zhon", "zhong", "lv", "nve", "biang"])
def test_typeable_prefixes_are_present(span: str) -> None:
    assert span in SpanVocab.load()


@pytest.mark.parametrize("absent", ["i", "u", "v", "", "zg", "zhongg", "ü"])
def test_untypeable_prefixes_are_absent(absent: str) -> None:
    """No syllable starts with i/u/v, so nothing can be typed towards one."""
    assert absent not in SpanVocab.load()


def test_ids_are_line_numbers_and_round_trip() -> None:
    spans = SpanVocab.load()
    for span in ("a", "zh", "zhong"):
        assert spans.spelling(spans.id(span)) == span
    assert spans.id("a") == 0


def test_an_unknown_span_raises_rather_than_folding() -> None:
    with pytest.raises(KeyError, match="not a typed span"):
        SpanVocab.load().id("qq")


def test_the_shipped_table_matches_the_syllable_inventory() -> None:
    """The generated file is only trustworthy while it is current."""
    source = default_syllables_path()
    if source is None:
        pytest.skip("not running from a checkout; the syllable inventory is absent")
    assert enumerate_spans(read_syllables(source)) == list(SpanVocab.load())


def test_regeneration_reproduces_the_shipped_table(tmp_path: Path) -> None:
    source = default_syllables_path()
    if source is None:
        pytest.skip("not running from a checkout; the syllable inventory is absent")
    regenerated = build(out_path=tmp_path / "typed_spans.txt")
    assert regenerated == list(SpanVocab.load())
    assert (tmp_path / "typed_spans.txt").read_text(encoding="utf-8") == (
        TYPED_SPANS_PATH.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("syllable", "expected"),
    [
        ("zhong", "zh"),
        ("chi", "ch"),
        ("shi", "sh"),
        ("zang", "z"),
        ("ang", "a"),
        ("lv", "l"),
        ("er", "e"),
    ],
)
def test_multi_letter_initials_stay_whole(syllable: str, expected: str) -> None:
    assert initial(syllable) == expected


def test_every_syllable_and_every_initial_is_a_span() -> None:
    """The two forms the augmentation actually produces must both be lookupable."""
    source = default_syllables_path()
    if source is None:
        pytest.skip("not running from a checkout; the syllable inventory is absent")
    spans = SpanVocab.load()
    for syllable in read_syllables(source):
        assert syllable in spans
        assert initial(syllable) in spans


def test_a_malformed_inventory_is_refused() -> None:
    with pytest.raises(ValueError, match="not sorted"):
        SpanVocab(["b", "a"])
    with pytest.raises(ValueError, match="duplicates"):
        SpanVocab(["a", "a"])
    with pytest.raises(ValueError, match="empty"):
        SpanVocab([])
