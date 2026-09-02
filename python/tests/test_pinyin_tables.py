"""The generated tables carry invariants the Rust crate asserts at load time.

`SyllableTable::load` and `Lexicon::load` panic or error when the data violates
sortedness, the `[a-z]` alphabet, the length bound, or cross-table agreement.
Catching a bad generator here means the failure names the cause instead of
surfacing as a panic inside an input method.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mlime.data.pinyin_tables import build, read_overrides

MAX_SYLLABLE_LEN = 6
TYPEABLE = re.compile(r"\A[a-z]+\Z")

#: The syllable inventory, and each character paired with its readings.
Tables = tuple[list[str], list[tuple[str, list[str]]]]


@pytest.fixture(scope="module")
def tables(tmp_path_factory: pytest.TempPathFactory) -> Tables:
    out_dir: Path = tmp_path_factory.mktemp("tables")
    build(out_dir)
    syllables = (out_dir / "syllables.txt").read_text(encoding="utf-8").splitlines()
    rows = []
    for line in (out_dir / "char_pinyin.tsv").read_text(encoding="utf-8").splitlines():
        char, readings = line.split("\t")
        rows.append((char, readings.split(",")))
    return syllables, rows


def test_syllables_are_strictly_sorted(tables: Tables) -> None:
    syllables, _ = tables
    assert syllables == sorted(set(syllables)), "the Rust side binary-searches this file"


def test_syllables_stay_inside_the_keyboard_alphabet(tables: Tables) -> None:
    syllables, _ = tables
    assert syllables, "inventory must not be empty"
    offenders = [s for s in syllables if not TYPEABLE.match(s)]
    assert not offenders, f"untypeable syllables survived generation: {offenders}"
    too_long = [s for s in syllables if len(s) > MAX_SYLLABLE_LEN]
    assert not too_long, f"MAX_SYLLABLE_LEN in the Rust crate would reject: {too_long}"


def test_characters_are_strictly_sorted(tables: Tables) -> None:
    _, rows = tables
    chars = [char for char, _ in rows]
    assert chars == sorted(set(chars)), "the Rust side binary-searches this file"


def test_every_reading_exists_in_the_inventory(tables: Tables) -> None:
    syllables, rows = tables
    inventory = set(syllables)
    for char, readings in rows:
        assert readings, f"{char} has no readings"
        assert len(readings) == len(set(readings)), f"{char} has duplicate readings: {readings}"
        missing = set(readings) - inventory
        assert not missing, f"{char} cites readings absent from the inventory: {missing}"


@pytest.mark.parametrize(
    ("char", "expected"),
    [("重", {"zhong", "chong"}), ("行", {"xing", "hang"}), ("得", {"de", "dei"}), ("吕", {"lv"})],
)
def test_polyphones_and_umlauts_survive(tables: Tables, char: str, expected: set[str]) -> None:
    _, rows = tables
    readings = {r for c, rs in rows if c == char for r in rs}
    assert expected <= readings, f"{char} lost readings: expected {expected}, got {readings}"


@pytest.mark.parametrize(("char", "reading"), sorted(read_overrides().items()))
def test_overrides_land_in_the_table(tables: Tables, char: str, reading: tuple[str, ...]) -> None:
    """Every override reaches the generated file, after pypinyin's own readings.

    Otherwise the fix lives only in whoever last ran the generator: CI
    regenerates and diffs, so a table hand-edited to add ``嗯 en`` fails the
    build and a table that silently dropped the override does not.
    """
    _, rows = tables
    listed = [readings for c, readings in rows if c == char]
    assert listed, f"{char} is not in the generated table at all"
    assert listed[0][-len(reading) :] == list(reading)


def test_an_override_pypinyin_already_lists_is_refused(tmp_path: Path) -> None:
    """A row that has been absorbed upstream is dead weight, and says so."""
    table = tmp_path / "pinyin_overrides.tsv"
    table.write_text("嗯\tn\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already lists"):
        build(tmp_path / "out", table)


def test_an_override_for_a_character_pypinyin_lacks_is_refused(tmp_path: Path) -> None:
    table = tmp_path / "pinyin_overrides.tsv"
    table.write_text("\U0002ebe0\tzhi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no readings for"):
        build(tmp_path / "out", table)


def test_a_malformed_override_row_is_refused(tmp_path: Path) -> None:
    table = tmp_path / "pinyin_overrides.tsv"
    table.write_text("嗯\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a"):
        read_overrides(table)


def test_an_untypeable_override_is_refused(tmp_path: Path) -> None:
    table = tmp_path / "pinyin_overrides.tsv"
    table.write_text("嗯\tên\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no keyboard can produce"):
        read_overrides(table)
