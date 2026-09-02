"""Reconciling the annotator's readings with the character table's.

g2pW is trained on Taiwan data and emits Taiwan MOE readings; the character
table the lexicon is built from is pypinyin-derived, i.e. mainland. Where the two
standards split, the label is a reading the table does not list for that
character, no typed span admits the target, and :class:`~mlime.train.samples.SampleBuilder`
drops the sentence at its admission gate. On the v1 corpus that was 4.0% of
sentences, and 83% of it was one character: 和 labelled ``han``.

Those are not annotation errors -- ``han`` really is how 和 is read in Taiwan --
so they cannot be fixed by relabelling. They are a disagreement about which
standard the corpus is in, and this is where it is settled: a table of
``(character, Taiwan reading) -> mainland reading`` pairs, applied to the label
before anything looks at it. The pairs are decided by hand and live in a data
file, because each one is a judgement about two standards and not something a
rule can derive.

What this is *not* is a repair of the character table. A reading the mainland
standard genuinely has and pypinyin lacks belongs in the generator's overrides
(:mod:`mlime.data.pinyin_tables`), so the decoder gains it too; arbitration only
rewrites a label that was never going to be typed that way by the users this
model is for.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from mlime.train.spans import SpanVocab

#: The decided pairs, shipped inside the package: ``<char>\t<label>\t<reading>``.
ARBITRATION_PATH = Path(__file__).parent / "data" / "reading_arbitration.tsv"


class ReadingArbitration:
    """Which annotator readings are rewritten, and to what.

    Keyed by ``(character, label)`` rather than by label alone: 和 ``han`` is a
    Taiwan reading of that character and nothing else, and a table that rewrote
    every ``han`` would corrupt 汉 and 韩 to fix 和.
    """

    def __init__(self, substitutions: dict[tuple[str, str], str], digest: str):
        self._substitutions = substitutions
        self._digest = digest

    @classmethod
    def load(cls, spans: SpanVocab, path: Path = ARBITRATION_PATH) -> ReadingArbitration:
        """Read the decided pairs, refusing anything the builder could not act on.

        A duplicated ``(character, label)`` is two decisions about one pair and
        there is no reason to prefer either; a reading outside the span
        inventory would substitute a label that no keystroke can produce, which
        is the failure this exists to prevent rather than a subtler version of
        it; and a row that maps a label to itself substitutes nothing while
        looking as though it does.
        """
        raw = path.read_bytes()
        substitutions: dict[tuple[str, str], str] = {}
        for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{number} is not a `<char>\\t<label>\\t<reading>` row: {line!r}"
                )
            character, label, reading = fields
            if len(character) != 1:
                raise ValueError(f"{path}:{number} names {character!r}, which is not one character")
            if not label or not reading:
                raise ValueError(f"{path}:{number} has an empty reading: {line!r}")
            if reading not in spans:
                raise ValueError(
                    f"{path}:{number} arbitrates {character} {label!r} to {reading!r}, "
                    "which is not a typed span of any syllable"
                )
            if label == reading:
                raise ValueError(
                    f"{path}:{number} arbitrates {character} {label!r} to itself, "
                    "which substitutes nothing"
                )
            if (character, label) in substitutions:
                raise ValueError(f"{path}:{number} arbitrates {character} {label!r} a second time")
            substitutions[(character, label)] = reading
        if not substitutions:
            raise ValueError(f"{path} decides nothing")
        return cls(substitutions, hashlib.blake2b(raw, digest_size=16).hexdigest())

    @property
    def digest(self) -> str:
        """A hash of the file this was read from, for the provenance record.

        Which pairs were arbitrated decides which sentences a run trained on, so
        a metrics file that does not say which table was in force describes a
        corpus nobody can reconstruct.
        """
        return self._digest

    def __len__(self) -> int:
        return len(self._substitutions)

    def __contains__(self, pair: object) -> bool:
        return pair in self._substitutions

    def reading(self, character: str, label: str) -> str:
        """The reading *character* is trained under, given the annotator's *label*.

        Every label that is not an arbitrated pair passes through untouched, so
        this is the only call the builder needs to make.
        """
        return self._substitutions.get((character, label), label)
