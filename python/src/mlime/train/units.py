"""Unit spans: the idioms and internet phrases a typist presses as one word.

A person who abbreviates an idiom types its four initials with the rest of the
sentence in full pinyin, a pattern the three typing styles never produce on
their own: the full style never abbreviates, and the abbreviated style drops each
syllable independently. The typing augmentation therefore needs to know which
stretches of a sentence are units, and :class:`UnitMatcher` finds them.

The matcher indexes the entries by length. A sentence is scanned once for the
positions whose leading two characters begin some entry, and only those
positions are checked against the entry sets of the lengths present. That needs
no automaton library, which keeps it importable wherever the training code runs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

#: The knowledge lexicon's categories that are units: the idioms and the internet
#: phrases people abbreviate whole.
UNIT_CATEGORIES = frozenset({"idiom", "internet"})

#: An entry shorter than this is a single character, which is no unit.
MIN_UNIT_CHARACTERS = 2


def read_unit_entries(path: Path, categories: frozenset[str] = UNIT_CATEGORIES) -> list[str]:
    """The unit entries of a ``word<TAB>category`` lexicon file, in file order.

    Only the rows whose category is in *categories* and whose word is at least
    :data:`MIN_UNIT_CHARACTERS` characters long are units; the rest of the
    knowledge lexicon (people, places, organisations) is not abbreviated whole.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no unit lexicon at {path}")
    entries: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        word, separator, category = line.partition("\t")
        if not separator or not word or not category:
            raise ValueError(f"{path}:{number} is not a `<word>\\t<category>` row: {line!r}")
        if category in categories and len(word) >= MIN_UNIT_CHARACTERS:
            entries.append(word)
    if not entries:
        raise ValueError(f"{path} holds no entry of {sorted(categories)}")
    return entries


class UnitMatcher:
    """Finds the unit entries inside a sentence.

    Built once per process from the entries and shared by every example. A
    matcher is pickled to the workers that count or build examples, so it holds
    plain sets and nothing else.
    """

    def __init__(self, entries: Iterable[str]) -> None:
        unique = frozenset(entries)
        if not unique:
            raise ValueError("a unit matcher needs at least one entry")
        short = sorted(entry for entry in unique if len(entry) < MIN_UNIT_CHARACTERS)
        if short:
            raise ValueError(f"unit entries are at least two characters, got {short[:3]}")
        lengths: dict[int, set[str]] = {}
        for entry in unique:
            lengths.setdefault(len(entry), set()).add(entry)
        self.entries = unique
        #: The entry sets by length, longest first.
        self._by_length = tuple(
            (length, frozenset(lengths[length])) for length in sorted(lengths, reverse=True)
        )
        #: The first two characters of every entry: a position that begins none
        #: of them begins no entry.
        self._heads = frozenset(entry[:2] for entry in unique)

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: Path) -> UnitMatcher:
        """The matcher over the unit entries of the lexicon file at *path*."""
        return cls(read_unit_entries(path))

    @property
    def digest(self) -> str:
        """A fingerprint of the entry set, for a record that must name what it was built with."""
        hasher = hashlib.blake2b(digest_size=16)
        for entry in sorted(self.entries):
            hasher.update(entry.encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def occurrences(self, text: str) -> list[tuple[int, int]]:
        """Every ``(start, end)`` at which an entry occurs in *text*, nested ones included.

        Ordered by start, and at one start by decreasing length: the substring
        occurrences a corpus count reads.
        """
        heads = self._heads
        found: list[tuple[int, int]] = []
        for start in [at for at in range(len(text) - 1) if text[at : at + 2] in heads]:
            for length, entries in self._by_length:
                end = start + length
                if end <= len(text) and text[start:end] in entries:
                    found.append((start, end))
        return found

    def spans(self, text: str) -> list[tuple[int, int]]:
        """The ``(start, end)`` spans of the leftmost-longest, non-overlapping entries of *text*.

        Scanning left to right, the entry that starts first wins, the longest
        among those that start there, and the next is looked for after it.
        """
        heads = self._heads
        found: list[tuple[int, int]] = []
        cursor = 0
        for start in [at for at in range(len(text) - 1) if text[at : at + 2] in heads]:
            if start < cursor:
                continue
            for length, entries in self._by_length:
                end = start + length
                if end <= len(text) and text[start:end] in entries:
                    found.append((start, end))
                    cursor = end
                    break
        return found
