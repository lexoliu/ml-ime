"""Word arcs over a keystroke lattice: a lexicon entry and the chains that emit it (#282).

A *word arc* is a lexicon entry ``c_1..c_L`` (``L >= 2``) together with a
chain of the lattice's own arcs ``(a -> x_1, c_1), (x_1 -> x_2, c_2), ...,
(x_{L-1} -> b, c_L)`` that emits it character by character, at least one of
whose spans is typed as an initial (:func:`mlime.legality._is_initial` of the
typed span: one letter, or zh/ch/sh). The lattice is the only source of
chains, so an arc can never make an illegal output; its correction costs are
whatever its lattice arcs cost. One ``(entry, a, b)`` can have several chains,
different cut points through the same keys; they are covers of the same
sentence and the search combines them as it combines covers.

The lexicon is a ``word<TAB>category`` file (``data/lexicon/knowledge.tsv``),
read for the categories asked for. An entry shorter than two characters, or
holding a character outside the emittable set, is skipped, and the load
counts both. Entries are held in a character trie over emission ids.
:meth:`ArcLexicon.arcs` walks the trie jointly with the lattice's arcs grouped
by start, so a record is enumerated in the work its own lattice and the
lexicon's matching prefixes cost, never by scanning the lexicon.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mlime.legality import KeyLattice, _is_initial
from mlime.logging import log

Ints = npt.NDArray[np.int64]
Entry = tuple[int, ...]

#: An entry shorter than this is a single character: no word, no arc.
MIN_ARC_CHARACTERS = 2
#: The columns of the mined entries table (`ime-corpus`'s frequency audit) an
#: arc lexicon is cut from.
ENTRY_COLUMNS = ("entry", "category", "occurrences")


def frequent_entries(
    table: Path, categories: Collection[str], min_occurrences: int
) -> list[tuple[str, str]]:
    """The ``(word, category)`` rows of *table* a frequency floor keeps, in table order.

    *table* is the entries table the corpus audit writes: a header naming at
    least :data:`ENTRY_COLUMNS`, then one row per lexicon entry with how often
    the corpus holds it. A flat prior over a whole category proposes thousands
    of entries nobody types, and each can land where its initials happen to be
    typed; the floor keeps the entries a sentence plausibly meant (#282).
    """
    if not table.is_file():
        raise FileNotFoundError(f"no entries table at {table}")
    wanted = frozenset(categories)
    if not wanted:
        raise ValueError("name at least one arc category")
    if min_occurrences < 0:
        raise ValueError(f"the occurrence floor must not be negative, got {min_occurrences}")
    lines = table.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{table} is empty")
    header = lines[0].split("\t")
    missing = [column for column in ENTRY_COLUMNS if column not in header]
    if missing:
        raise ValueError(f"{table} has no column {missing}; its header is {header}")
    word_at, category_at, count_at = (header.index(column) for column in ENTRY_COLUMNS)
    kept: list[tuple[str, str]] = []
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(header) or not fields[count_at].isdecimal():
            raise ValueError(f"{table}:{number} is not a row of {header}: {line!r}")
        if fields[category_at] in wanted and int(fields[count_at]) >= min_occurrences:
            kept.append((fields[word_at], fields[category_at]))
    if not kept:
        raise ValueError(
            f"{table} holds no entry of {sorted(wanted)} with {min_occurrences} occurrences"
        )
    return kept


@dataclass(frozen=True)
class WordArc:
    """One lexicon entry over one stretch of a record's keys, with every chain that emits it."""

    #: The entry's characters, as emission indices.
    entry: Entry
    #: The cut it starts at and the cut it ends at.
    start: int
    end: int
    #: ``[chains, len(entry)]``: for each chain, the indices of its lattice arcs in order.
    chains: Ints


class _Node:
    """A trie node over emission ids; ``word`` is set where an entry ends."""

    __slots__ = ("children", "word")

    def __init__(self) -> None:
        self.children: dict[int, _Node] = {}
        self.word: Entry | None = None


class _Follow:
    """The lattice arcs leaving one cut, grouped by character, each group by end."""

    __slots__ = ("ends", "index", "slots")

    def __init__(self, lattice: KeyLattice, bounds: Ints, cut: int) -> None:
        low, high = int(bounds[cut]), int(bounds[cut + 1])
        order = np.argsort(lattice.character[low:high], kind="stable")
        characters = lattice.character[low:high][order]
        heads = np.flatnonzero(np.diff(characters, prepend=-1)).tolist()
        tails = [*heads[1:], characters.size][: len(heads)]
        self.ends: list[int] = lattice.end[low:high][order].tolist()
        self.index: list[int] = (order + low).tolist()
        self.slots: dict[int, tuple[int, int]] = {
            int(characters[head]): (head, tail) for head, tail in zip(heads, tails, strict=True)
        }


class ArcLexicon:
    """A lexicon of entries over emission ids, ready to be walked against lattices."""

    def __init__(
        self, entries: Iterable[Entry], *, skipped_short: int = 0, skipped_outside: int = 0
    ) -> None:
        self._root = _Node()
        self.size = 0
        #: Rows a load skipped: shorter than two characters, or past the emittable set.
        self.skipped_short = skipped_short
        self.skipped_outside = skipped_outside
        for entry in entries:
            if len(entry) < MIN_ARC_CHARACTERS:
                raise ValueError(f"an arc entry has at least {MIN_ARC_CHARACTERS} characters")
            node = self._root
            for character in entry:
                node = node.children.setdefault(character, _Node())
            if node.word is None:
                node.word = tuple(entry)
                self.size += 1
        if not self.size:
            raise ValueError("an arc lexicon holds at least one entry")

    @classmethod
    def load(cls, path: Path, categories: Collection[str], characters: Sequence[str]) -> ArcLexicon:
        """The entries of *path*'s *categories* over *characters*, the emission order.

        Entries shorter than two characters or holding a character outside
        *characters* are skipped and counted in the log line; a row that is not
        ``word<TAB>category`` raises with its line.
        """
        if not path.is_file():
            raise FileNotFoundError(f"no arc lexicon at {path}")
        wanted = frozenset(categories)
        if not wanted:
            raise ValueError("name at least one arc category")
        ids = {character: index for index, character in enumerate(characters)}
        entries: list[Entry] = []
        short = outside = 0
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            word, separator, category = line.partition("\t")
            if not separator or not word or not category:
                raise ValueError(f"{path}:{number} is not a `<word>\\t<category>` row: {line!r}")
            if category not in wanted:
                continue
            if len(word) < MIN_ARC_CHARACTERS:
                short += 1
            elif any(character not in ids for character in word):
                outside += 1
            else:
                entries.append(tuple(ids[character] for character in word))
        lexicon = cls(entries, skipped_short=short, skipped_outside=outside)
        log.info(
            "arc lexicon loaded",
            path=str(path),
            categories=sorted(wanted),
            entries=lexicon.size,
            skipped_short=short,
            skipped_outside_the_emittable_set=outside,
        )
        return lexicon

    def arcs(self, lattice: KeyLattice, keys: str) -> list[WordArc]:
        """Every word arc of *lattice*, over the keys it was built from.

        An arc is offered when some chain of the lattice's arcs from ``a`` to
        ``b`` emits a lexicon entry with at least one initial-typed span. The
        arcs come in a deterministic order: by walk, start cut first.
        """
        if len(keys) != lattice.keys:
            raise ValueError(f"{len(keys)} keys against a lattice of {lattice.keys}")
        if not lattice.character.size:
            return []
        bounds = np.searchsorted(lattice.start, np.arange(lattice.keys + 2)).astype(np.int64)
        follows: dict[int, _Follow] = {}
        initials: dict[tuple[int, int], bool] = {}
        found: dict[tuple[int, Entry, int], list[Entry]] = {}

        def initial(start: int, end: int) -> bool:
            held = initials.get((start, end))
            if held is None:
                held = initials[(start, end)] = _is_initial(keys[start:end])
            return held

        def walk(node: _Node, cut: int, origin: int, chain: list[int], typed: bool) -> None:
            follow = follows.get(cut)
            if follow is None:
                follow = follows[cut] = _Follow(lattice, bounds, cut)
            if len(node.children) <= len(follow.slots):
                matches = [
                    (character, child, follow.slots[character])
                    for character, child in node.children.items()
                    if character in follow.slots
                ]
            else:
                matches = [
                    (character, node.children[character], slot)
                    for character, slot in follow.slots.items()
                    if character in node.children
                ]
            for _, child, (head, tail) in matches:
                for at in range(head, tail):
                    end = follow.ends[at]
                    chain.append(follow.index[at])
                    reaches = typed or initial(cut, end)
                    if child.word is not None and reaches:
                        found.setdefault((origin, child.word, end), []).append(tuple(chain))
                    if child.children:
                        walk(child, end, origin, chain, reaches)
                    chain.pop()

        for cut in np.unique(lattice.start).tolist():
            walk(self._root, int(cut), int(cut), [], False)
        return [
            WordArc(entry, start, end, np.array(chains, dtype=np.int64))
            for (start, entry, end), chains in found.items()
        ]
