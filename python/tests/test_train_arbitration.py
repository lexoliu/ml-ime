"""The arbitration table decides which sentences a run gets to train on.

A pair that is missing costs the corpus every sentence carrying that character --
和 alone was 3.4% of v1 -- and a pair that is wrong trains the model to produce a
character from keystrokes nobody would press for it. Neither shows up as a crash,
so the file's invariants are asserted here and the substitution is measured on
the builder rather than read off the table.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mlime.train.arbitration import ARBITRATION_PATH, ReadingArbitration
from mlime.train.spans import SpanVocab


def write(path: Path, rows: str) -> Path:
    """A table file holding *rows*, so a refusal can be provoked."""
    table = path / "reading_arbitration.tsv"
    table.write_text(rows, encoding="utf-8")
    return table


def test_the_shipped_table_is_loadable_and_arbitrates_he(arbitration: ReadingArbitration) -> None:
    assert arbitration.reading("和", "han") == "he"
    assert len(arbitration) > 0


def test_the_shipped_table_names_a_digest_of_its_own_bytes(
    arbitration: ReadingArbitration,
) -> None:
    expected = hashlib.blake2b(ARBITRATION_PATH.read_bytes(), digest_size=16).hexdigest()
    assert arbitration.digest == expected


def test_an_unlisted_reading_passes_through(arbitration: ReadingArbitration) -> None:
    """Only the decided pairs move; everything else is the annotator's own label."""
    assert arbitration.reading("和", "he") == "he"
    assert arbitration.reading("我", "wo") == "wo"
    # The pair is keyed by character: 汉 is read `han` and must stay that way.
    assert arbitration.reading("汉", "han") == "han"


def test_a_duplicated_pair_is_refused(spans: SpanVocab, tmp_path: Path) -> None:
    table = write(tmp_path, "和\than\the\n和\than\thuo\n")
    with pytest.raises(ValueError, match="a second time"):
        ReadingArbitration.load(spans, table)


def test_a_reading_that_is_not_a_typed_span_is_refused(spans: SpanVocab, tmp_path: Path) -> None:
    table = write(tmp_path, "和\than\tqqq\n")
    with pytest.raises(ValueError, match="not a typed span"):
        ReadingArbitration.load(spans, table)


def test_a_row_that_substitutes_nothing_is_refused(spans: SpanVocab, tmp_path: Path) -> None:
    table = write(tmp_path, "和\than\than\n")
    with pytest.raises(ValueError, match="to itself"):
        ReadingArbitration.load(spans, table)


def test_a_malformed_row_is_refused(spans: SpanVocab, tmp_path: Path) -> None:
    table = write(tmp_path, "和\than\n")
    with pytest.raises(ValueError, match="is not a"):
        ReadingArbitration.load(spans, table)


def test_a_multi_character_key_is_refused(spans: SpanVocab, tmp_path: Path) -> None:
    table = write(tmp_path, "垃圾\tlese\tlaji\n")
    with pytest.raises(ValueError, match="not one character"):
        ReadingArbitration.load(spans, table)


def test_an_empty_table_is_refused(spans: SpanVocab, tmp_path: Path) -> None:
    table = write(tmp_path, "")
    with pytest.raises(ValueError, match="decides nothing"):
        ReadingArbitration.load(spans, table)
