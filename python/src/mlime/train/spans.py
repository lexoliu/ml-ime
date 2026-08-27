"""The typed-span inventory: the fill tower's auxiliary vocabulary.

The fill tower reads one slot per typed *syllable span* -- the letters the user
actually pressed for one syllable, which is the whole syllable when they typed it
out, its initial when they abbreviated, and any prefix in between when they are
still mid-syllable. Each slot's input is the base ``[MASK]`` embedding plus an
additive embedding looked up in a table indexed by that span.

The set of spans is closed and small: a span is typed *towards* a syllable, so it
is always a prefix of at least one entry of ``crates/ime-pinyin/data/syllables.txt``.
Enumerating the prefixes gives the exact table. A mean-of-letter-embeddings
encoding was the alternative and is wrong for this: it makes ``na`` and ``an``
the same vector, and those are different syllables.

The three multi-letter initials ``zh``/``ch``/``sh`` need no special case *here*
-- they are prefixes of their syllables like every other span, so enumeration
already contains them. They matter in :func:`initial`, which is what the
abbreviation augmentation calls, and in nothing else.

The enumeration is written out as a generated data file rather than computed at
import time so that a Kaggle kernel, which has the package but not the Rust
crates, still loads the same table the tests pinned.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from mlime.locate import find_upwards

#: The generated inventory, one span per line, shipped inside the package.
TYPED_SPANS_PATH = Path(__file__).parent / "data" / "typed_spans.txt"

#: Where the syllable inventory this is derived from lives in the repository.
SYLLABLES_RELATIVE = Path("crates/ime-pinyin/data/syllables.txt")

#: The initials that are two letters rather than one. Everything else is a single
#: letter, and a bare-vowel syllable's initial is its first vowel.
MULTI_LETTER_INITIALS = ("zh", "ch", "sh")


def initial(syllable: str) -> str:
    """The abbreviation of *syllable*: its initial, kept whole for zh/ch/sh.

    This is the one place the multi-letter initials are special-cased. Typing
    ``z`` for ``zhong`` is legal and the decoder handles it, but nobody who means
    to abbreviate ``zhong`` types ``z`` -- they type ``zh`` -- so the
    augmentation must not manufacture training data that says otherwise.
    """
    if not syllable:
        raise ValueError("the empty string is not a syllable")
    for prefix in MULTI_LETTER_INITIALS:
        if syllable.startswith(prefix):
            return prefix
    return syllable[0]


def enumerate_spans(syllables: Iterable[str]) -> list[str]:
    """Every non-empty prefix of every syllable, sorted and deduplicated."""
    spans: set[str] = set()
    for syllable in syllables:
        if not syllable:
            raise ValueError("the syllable inventory contains an empty line")
        spans.update(syllable[:length] for length in range(1, len(syllable) + 1))
    return sorted(spans)


def read_syllables(path: Path) -> list[str]:
    """Read a syllable inventory file, one spelling per line."""
    syllables = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [syllable for syllable in syllables if syllable]


def default_syllables_path() -> Path | None:
    """Locate the Rust crate's syllable inventory, if this is a checkout."""
    return find_upwards(SYLLABLES_RELATIVE)


def build(out_path: Path = TYPED_SPANS_PATH, syllables_path: Path | None = None) -> list[str]:
    """Regenerate the typed-span table from the syllable inventory."""
    source = syllables_path or default_syllables_path()
    if source is None:
        raise FileNotFoundError(
            f"no {SYLLABLES_RELATIVE} above the working directory; pass syllables_path"
        )
    spans = enumerate_spans(read_syllables(source))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{span}\n" for span in spans), encoding="utf-8")
    return spans


class SpanVocab:
    """The closed set of typed spans, and the id each one has in the table.

    Ids are the file's line numbers, so the table is reproducible from the data
    file alone and a checkpoint stays readable without the repository.
    """

    def __init__(self, spans: list[str]):
        if not spans:
            raise ValueError("the typed-span inventory is empty")
        if spans != sorted(spans):
            raise ValueError("the typed-span inventory is not sorted")
        if len(set(spans)) != len(spans):
            raise ValueError("the typed-span inventory has duplicates")
        self._spans = spans
        self._ids = {span: index for index, span in enumerate(spans)}

    @classmethod
    def load(cls, path: Path = TYPED_SPANS_PATH) -> SpanVocab:
        """Read the generated table."""
        return cls(read_syllables(path))

    def __len__(self) -> int:
        return len(self._spans)

    def __iter__(self) -> Iterator[str]:
        return iter(self._spans)

    def __contains__(self, span: object) -> bool:
        return span in self._ids

    def id(self, span: str) -> int:
        """The table index of *span*.

        Raises rather than folding an unknown span onto a catch-all: the set is
        closed by construction, so a miss means the caller built a span some
        other way -- a stale table, an untyped character, a bug -- and silently
        training on a shared "unknown" row would hide it.
        """
        try:
            return self._ids[span]
        except KeyError:
            raise KeyError(f"{span!r} is not a typed span of any syllable") from None

    def spelling(self, span_id: int) -> str:
        """The span at table index *span_id*."""
        return self._spans[span_id]

    def ids(self, spans: Iterable[str]) -> list[int]:
        """Table indices for a sequence of spans, in order."""
        return [self.id(span) for span in spans]
