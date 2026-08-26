"""Generate the pinyin lexicon tables consumed by the `ime-pinyin` Rust crate.

Two artefacts are produced, both derived from ``pypinyin`` rather than hand-written,
so that the syllable inventory stays in sync with a maintained data source:

``syllables.txt``
    Every toneless syllable that actually occurs in the character dictionary, in
    *typing* spelling -- ``pypinyin`` already renders ``lǜ`` as ``lv``, which is what
    a QWERTY pinyin keyboard produces -- one per line, sorted.

``char_pinyin.tsv``
    ``<char>\t<py1>,<py2>,...`` for every character, readings deduplicated and
    ordered as ``pypinyin`` orders them (most common reading first).

The input alphabet of a pinyin keyboard is exactly ``[a-z]``. Readings outside it
(``ê``, on two rare characters) can never be produced by a keystroke sequence, so they
are dropped -- loudly, with a logged count -- rather than carried into the Rust side
where they would be dead entries in every mask. A character left with no typeable
reading at all is a data error and raises.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pypinyin.constants import PINYIN_DICT
from pypinyin.contrib.tone_convert import to_normal

from mlime.logging import log


TYPEABLE = re.compile(r"\A[a-z]+\Z")


def _readings(raw: str, untypeable: Counter[str]) -> Iterator[str]:
    """Yield deduplicated, keyboard-typeable toneless readings in ``pypinyin`` order."""
    seen: set[str] = set()
    for toned in raw.split(","):
        normal = to_normal(toned)
        if not normal:
            raise ValueError(f"pypinyin produced an empty toneless reading for {toned!r}")
        if not TYPEABLE.match(normal):
            untypeable[normal] += 1
            continue
        if normal not in seen:
            seen.add(normal)
            yield normal


def build(out_dir: Path) -> None:
    """Write ``syllables.txt`` and ``char_pinyin.tsv`` into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    untypeable: Counter[str] = Counter()
    syllables: set[str] = set()
    rows: list[tuple[str, str]] = []
    skipped = 0
    for codepoint, raw in PINYIN_DICT.items():
        readings = list(_readings(raw, untypeable))
        if not readings:
            skipped += 1
            continue
        syllables.update(readings)
        rows.append((chr(codepoint), ",".join(readings)))
    if skipped:
        raise ValueError(f"{skipped} characters have no keyboard-typeable reading")

    rows.sort()
    (out_dir / "syllables.txt").write_text(
        "".join(f"{s}\n" for s in sorted(syllables)), encoding="utf-8"
    )
    (out_dir / "char_pinyin.tsv").write_text(
        "".join(f"{c}\t{p}\n" for c, p in rows), encoding="utf-8"
    )
    log.info(
        "pinyin tables written",
        syllables=len(syllables),
        chars=len(rows),
        longest=max(len(s) for s in syllables),
        dropped_untypeable=dict(untypeable),
        out_dir=str(out_dir),
    )
