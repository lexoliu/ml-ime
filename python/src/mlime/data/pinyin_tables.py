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

``pypinyin`` is not complete, and where a reading it lacks is one people really
type -- 嗯 ``en``, which the dictionary spells only as the syllabic ``n``/``ng``
nobody can press -- it is added from ``pinyin_overrides.tsv`` rather than by
editing the generated file. Editing the output would survive exactly until the
next regeneration, and CI regenerates on every push. Overrides are appended after
the readings ``pypinyin`` gives, because they are additions to that ordering and
not a re-ranking of it, and each one must be new: an override the dictionary
already lists is a row that has done its work and should go.
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

#: Readings the mainland standard has and ``pypinyin`` does not, one
#: ``<char>\t<reading>`` row each, applied after the dictionary's own.
OVERRIDES_PATH = Path(__file__).parent / "pinyin_overrides.tsv"


def read_overrides(path: Path = OVERRIDES_PATH) -> dict[str, tuple[str, ...]]:
    """The extra readings, keyed by character and kept in file order."""
    overrides: dict[str, tuple[str, ...]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{number} is not a `<char>\\t<reading>` row: {line!r}")
        character, reading = fields
        if len(character) != 1:
            raise ValueError(f"{path}:{number} names {character!r}, which is not one character")
        if not TYPEABLE.match(reading):
            raise ValueError(f"{path}:{number} adds {reading!r}, which no keyboard can produce")
        existing = overrides.get(character, ())
        if reading in existing:
            raise ValueError(f"{path}:{number} adds {reading!r} to {character} twice")
        overrides[character] = (*existing, reading)
    return overrides


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


def build(out_dir: Path, overrides_path: Path = OVERRIDES_PATH) -> None:
    """Write ``syllables.txt`` and ``char_pinyin.tsv`` into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    untypeable: Counter[str] = Counter()
    overrides = read_overrides(overrides_path)
    unused = set(overrides)
    syllables: set[str] = set()
    rows: list[tuple[str, str]] = []
    skipped = 0
    for codepoint, raw in PINYIN_DICT.items():
        character = chr(codepoint)
        readings = list(_readings(raw, untypeable))
        if not readings:
            skipped += 1
            continue
        for added in overrides.get(character, ()):
            if added in readings:
                raise ValueError(
                    f"{overrides_path} adds {added!r} to {character}, "
                    "which pypinyin already lists; drop the row"
                )
            readings.append(added)
            unused.discard(character)
        syllables.update(readings)
        rows.append((character, ",".join(readings)))
    if skipped:
        raise ValueError(f"{skipped} characters have no keyboard-typeable reading")
    if unused:
        raise ValueError(
            f"{overrides_path} names characters pypinyin has no readings for: {sorted(unused)}"
        )

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
        overridden=sorted(overrides),
        out_dir=str(out_dir),
    )
