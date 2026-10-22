"""The legality automaton over keys: which sentences are a reading of a keystroke string.

#216 §3 defines it once, and this module and ``ime_pinyin::legality`` implement
it from the same committed tables -- ``syllables.txt``, ``char_pinyin.tsv``,
``typo.json`` -- and the #101 ``corrections`` function; the fixture at
``crates/ime-pinyin/tests/fixtures/legality.json`` holds the two to the same
answers. Rust is the product's checker and the evaluation's verifier; this side
feeds the constrained-CTC loss.

A sentence ``c1..cm`` is legal for keys ``k1..kn`` when the keys cut into ``m``
contiguous spans, span ``j`` matching ``cj`` under one of four rules, each
against some reading ``r`` of the character:

1. **Full syllable** -- the span is ``r``; cost 0.
2. **Initial** -- the span is one letter, or ``zh``/``ch``/``sh``, and a prefix
   of ``r``; cost 0.
3. **Unfinished tail** -- the span is the last one, a prefix of ``r``, and the
   tail allowance is on (live input); cost 0.
4. **Correction** -- the span has two keys or more, is not ``zh``/``ch``/``sh``,
   and ``(r, w)`` is among ``corrections(span)``; cost ``w``. Whether a span that
   is itself a syllable is corrected (``zao`` meant as zhao) is
   :attr:`LegalityOptions.correct_syllables`, chosen on the eval3-typo dev slice.

A span's cost for a character is the cheapest rule over the character's
readings, and a sentence's cost the cheapest cover. Spans are at most
``MAX_ARC_KEYS`` keys long -- a six-letter syllable two edits wide.

Two enumerations come out of it. :meth:`Legality.gold` lists, for one sentence,
every ``(position, start, end, cost)`` arc that lies on a legal alignment --
the constrained-CTC forward sums over exactly these. :meth:`Legality.lattice`
lists every legal ``(start, end, character, cost)`` arc of the keys, pruned to
complete covers -- what the decoder's per-step legal set and the search read.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mlime.typo import (
    CORRECTIONS_MAX_EDITS,
    MAX_SYLLABLE_LEN,
    MULTI_LETTER_INITIALS,
    CorrectionTable,
    NoiseModel,
    load_syllables,
)

#: The longest span an arc covers: a full syllable two edits wide, the same
#: bound as ``ime_pinyin``'s corrected segments.
MAX_ARC_KEYS = MAX_SYLLABLE_LEN + CORRECTIONS_MAX_EDITS

#: The shared fixture both languages test the automaton against.
LEGALITY_FIXTURE_RELATIVE = Path("crates/ime-pinyin/tests/fixtures/legality.json")

_INFINITY = math.inf


@dataclass(frozen=True)
class LegalityOptions:
    """What the automaton admits beyond the rules that always hold."""

    #: Whether the last span may be an unfinished syllable: on for live input,
    #: off in offline evaluation, as ``SegmentOptions`` has it.
    unfinished_tail: bool
    #: Whether corrections also apply to a span that is itself a syllable.
    correct_syllables: bool


#: Offline evaluation and training: complete input, corrections everywhere.
OFFLINE = LegalityOptions(unfinished_tail=False, correct_syllables=True)

#: A live session: the last syllable may still be being typed.
LIVE = LegalityOptions(unfinished_tail=True, correct_syllables=True)


def _keys(keys: str) -> str:
    """*keys*, refused unless it is a non-empty ``[a-z]`` string."""
    if not keys or not keys.isascii() or not keys.isalpha() or not keys.islower():
        raise ValueError(f"keys must be a non-empty [a-z] string, got {keys!r}")
    return keys


def _is_initial(span: str) -> bool:
    """Whether *span* has the initial's shape: one letter, or zh/ch/sh."""
    return len(span) == 1 or span in MULTI_LETTER_INITIALS


@dataclass(frozen=True)
class SpanMatches:
    """The characters one span admits: sorted emission ids, each at its cheapest cost."""

    ids: npt.NDArray[np.int32]
    costs: npt.NDArray[np.float64]


@dataclass(frozen=True)
class GoldArcs:
    """Every arc of one sentence's legal alignments over its keys.

    Arc ``i`` says character ``position[i]`` of the sentence may be emitted at
    key ``end[i] - 1``, consuming keys ``start[i]..end[i]``, at correction cost
    ``cost[i]``. Every arc lies on at least one complete alignment, so a
    forward recursion over them never carries mass that cannot reach the end.
    """

    keys: int
    length: int
    position: npt.NDArray[np.int32]
    start: npt.NDArray[np.int32]
    end: npt.NDArray[np.int32]
    cost: npt.NDArray[np.float64]
    #: The cheapest alignment's summed cost: the sentence's cost.
    best: float

    def dense(self) -> npt.NDArray[np.float64]:
        """``[length, keys, MAX_ARC_KEYS]``: the arc cost by position, end key and span width.

        Entry ``[j, e - 1, e - s - 1]`` is the cost of emitting character ``j``
        at key ``e - 1`` after consuming keys ``s..e``; ``inf`` where no legal
        alignment has that arc. This is the layout a batched constrained-CTC
        forward reads: one gather per position and end key.
        """
        table = np.full((self.length, self.keys, MAX_ARC_KEYS), _INFINITY, dtype=np.float64)
        table[self.position, self.end - 1, self.end - self.start - 1] = self.cost
        return table


@dataclass(frozen=True)
class KeyLattice:
    """Every legal ``(start, end, character, cost)`` arc of a keystroke string.

    Arcs are sorted by ``(start, end, character)`` and pruned to complete
    covers: each arc starts where some cover of the keys before it ends, and
    ends where some cover of the keys after it begins.
    """

    keys: int
    start: npt.NDArray[np.int32]
    end: npt.NDArray[np.int32]
    character: npt.NDArray[np.int32]
    cost: npt.NDArray[np.float64]


class Legality:
    """The automaton, built once from the tables over an emittable character set.

    *characters* fixes the emission order: an arc's character is an index into
    it, which is the CTC head's row. Every character must have readings in
    *readings*, and every reading must be one of the correction table's
    syllables -- the two generated tables come from one run.
    """

    def __init__(
        self,
        characters: Sequence[str],
        readings: Mapping[str, Sequence[str]],
        corrections: CorrectionTable,
    ) -> None:
        syllables = corrections.syllables
        index: dict[str, int] = {}
        char_readings: list[tuple[str, ...]] = []
        homophones: dict[str, list[int]] = {}
        prefixed: dict[str, list[int]] = {}
        for emit, character in enumerate(characters):
            if character in index:
                raise ValueError(f"{character!r} appears twice in the emittable set")
            if character not in readings:
                raise ValueError(f"{character!r} is emittable but has no readings")
            own = tuple(readings[character])
            unknown = [reading for reading in own if reading not in syllables]
            if unknown:
                raise ValueError(f"{character!r} reads as non-syllables {unknown}")
            index[character] = emit
            char_readings.append(own)
            for reading in own:
                homophones.setdefault(reading, []).append(emit)
                # Every prefix, the reading itself included: the initial and
                # tail rules admit a character through any of them.
                for length in range(1, len(reading) + 1):
                    group = prefixed.setdefault(reading[:length], [])
                    if not group or group[-1] != emit:
                        group.append(emit)
        self._characters = tuple(characters)
        self._index = index
        self._readings = tuple(char_readings)
        self._syllables = syllables
        self._corrections = corrections
        self._homophones = {
            syllable: np.array(sorted(set(ids)), dtype=np.int32)
            for syllable, ids in homophones.items()
        }
        self._prefixed = {
            prefix: np.array(sorted(set(ids)), dtype=np.int32) for prefix, ids in prefixed.items()
        }
        self._matches: dict[tuple[str, bool, bool], SpanMatches] = {}
        self._correction_maps: dict[str, dict[str, float]] = {}

    @classmethod
    def load(
        cls,
        characters: Sequence[str] | None,
        *,
        char_table: Path,
        typo_table: Path,
        syllables: Path,
    ) -> Legality:
        """Build the automaton from the generated tables.

        *characters* is the emittable set in emission order; ``None`` takes the
        whole character table in its own order, which is ``ime_pinyin``'s
        lexicon order.
        """
        from mlime.train.lexicon import read_char_readings

        readings = read_char_readings(char_table)
        table = CorrectionTable(NoiseModel.load(typo_table), load_syllables(syllables))
        return cls(tuple(readings) if characters is None else characters, readings, table)

    @property
    def characters(self) -> tuple[str, ...]:
        """The emittable set, in emission order."""
        return self._characters

    @property
    def corrections(self) -> CorrectionTable:
        """The correction table rule 4 reads."""
        return self._corrections

    def ids(self, sentence: str) -> tuple[int, ...] | None:
        """Each character's emission index; ``None`` when one is not emittable."""
        index = self._index
        try:
            return tuple([index[character] for character in sentence])
        except KeyError:
            return None

    def _corrects(self, span: str, options: LegalityOptions) -> bool:
        """Whether rule 4 applies to *span* under *options*."""
        return (
            len(span) >= 2
            and span not in MULTI_LETTER_INITIALS
            and (options.correct_syllables or span not in self._syllables)
            and self._corrections.reaches(span)
        )

    def _correction_map(self, span: str) -> dict[str, float]:
        """``corrections(span)`` as ``{syllable: cost}``, memoised."""
        cached = self._correction_maps.get(span)
        if cached is None:
            cached = dict(self._corrections.corrections(span))
            self._correction_maps[span] = cached
        return cached

    def span_matches(self, span: str, last: bool, options: LegalityOptions) -> SpanMatches:
        """Every character *span* admits, at its cheapest rule.

        *last* says the span ends at the last key, where the tail rule may
        apply. Memoised on what the rules can see: the span, whether the tail
        rule applies, and whether corrections do.
        """
        tail = last and options.unfinished_tail
        corrects = self._corrects(span, options)
        key = (span, tail, corrects)
        cached = self._matches.get(key)
        if cached is not None:
            return cached
        ids: list[npt.NDArray[np.int32]] = []
        costs: list[npt.NDArray[np.float64]] = []
        # The initial and tail rules admit every completion; otherwise only a
        # span that is itself a syllable reads at no cost.
        own = (self._prefixed if _is_initial(span) or tail else self._homophones).get(span)
        if own is not None:
            ids.append(own)
            costs.append(np.zeros(own.shape, dtype=np.float64))
        if corrects:
            for syllable, cost in self._corrections.corrections(span):
                reading = self._homophones.get(syllable)
                if reading is not None:
                    ids.append(reading)
                    costs.append(np.full(reading.shape, cost, dtype=np.float64))
        if not ids:
            matches = SpanMatches(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64))
        elif len(ids) == 1:
            matches = SpanMatches(ids[0], costs[0])
        else:
            joined = np.concatenate(ids)
            priced = np.concatenate(costs)
            order = np.lexsort((priced, joined))
            joined, priced = joined[order], priced[order]
            first = np.ones(joined.shape, dtype=bool)
            first[1:] = joined[1:] != joined[:-1]
            matches = SpanMatches(joined[first], priced[first])
        self._matches[key] = matches
        return matches

    def char_cost(self, span: str, last: bool, emit: int, options: LegalityOptions) -> float:
        """What *span* costs as character *emit*: its cheapest rule, ``inf`` when none.

        The single-character form of :meth:`span_matches`, without building the
        span's whole admitted set: the gold enumeration asks it for one
        character per span, and a one-letter span admits thousands.
        """
        readings = self._readings[emit]
        if span in readings:
            return 0.0
        if (_is_initial(span) or (last and options.unfinished_tail)) and any(
            reading.startswith(span) for reading in readings
        ):
            return 0.0
        if self._corrects(span, options):
            corrected = self._correction_map(span)
            return min(corrected.get(reading, _INFINITY) for reading in readings)
        return _INFINITY

    def gold(self, keys: str, sentence: str, options: LegalityOptions) -> GoldArcs | None:
        """The arcs of *sentence*'s legal alignments over *keys*; ``None`` when it is illegal."""
        ids = self.ids(sentence)
        if ids is None:
            return None
        return self.gold_ids(keys, ids, options)

    def gold_ids(
        self, keys: str, sentence: Sequence[int], options: LegalityOptions
    ) -> GoldArcs | None:
        """:meth:`gold` over emission indices; ``None`` when the sentence is illegal."""
        keys = _keys(keys)
        width, length = len(keys), len(sentence)
        if length == 0 or length > width:
            return None
        # Forward: the arcs each position can take from the offsets the
        # positions before it reach, leaving every later character a key.
        layers: list[list[tuple[int, int, float]]] = []
        frontier = {0}
        for position, emit in enumerate(sentence):
            ceiling = width - (length - position - 1)
            arcs: list[tuple[int, int, float]] = []
            reached: set[int] = set()
            for start in sorted(frontier):
                for end in range(start + 1, min(start + MAX_ARC_KEYS, ceiling) + 1):
                    cost = self.char_cost(keys[start:end], end == width, emit, options)
                    if cost != _INFINITY:
                        arcs.append((start, end, cost))
                        reached.add(end)
            if not reached:
                return None
            layers.append(arcs)
            frontier = reached
        if width not in frontier:
            return None
        # Backward: keep the arcs that reach the end, pricing each offset's
        # cheapest completion on the way.
        after: dict[int, float] = {width: 0.0}
        kept: list[list[tuple[int, int, float]]] = [[] for _ in range(length)]
        for position in range(length - 1, -1, -1):
            before: dict[int, float] = {}
            for start, end, cost in layers[position]:
                rest = after.get(end)
                if rest is None:
                    continue
                kept[position].append((start, end, cost))
                total = cost + rest
                if total < before.get(start, _INFINITY):
                    before[start] = total
            after = before
        rows = [
            (position, start, end, cost)
            for position, arcs in enumerate(kept)
            for start, end, cost in arcs
        ]
        positions, starts, ends, costs = zip(*rows, strict=True)
        return GoldArcs(
            keys=width,
            length=length,
            position=np.array(positions, dtype=np.int32),
            start=np.array(starts, dtype=np.int32),
            end=np.array(ends, dtype=np.int32),
            cost=np.array(costs, dtype=np.float64),
            best=after[0],
        )

    def cost(self, keys: str, sentence: str, options: LegalityOptions) -> float | None:
        """*sentence*'s cost over *keys*: its cheapest legal cover, ``None`` when illegal."""
        arcs = self.gold(keys, sentence, options)
        return None if arcs is None else arcs.best

    def lattice(self, keys: str, options: LegalityOptions) -> KeyLattice:
        """Every legal arc of *keys*, pruned to complete covers."""
        keys = _keys(keys)
        width = len(keys)
        edges: dict[tuple[int, int], SpanMatches] = {}
        for start in range(width):
            for end in range(start + 1, min(start + MAX_ARC_KEYS, width) + 1):
                matches = self.span_matches(keys[start:end], end == width, options)
                if matches.ids.size:
                    edges[(start, end)] = matches
        reachable = [False] * (width + 1)
        reachable[0] = True
        for start, end in sorted(edges):
            if reachable[start]:
                reachable[end] = True
        productive = [False] * (width + 1)
        productive[width] = True
        for start, end in sorted(edges, reverse=True):
            if productive[end]:
                productive[start] = True
        kept = [
            (start, end, matches)
            for (start, end), matches in sorted(edges.items())
            if reachable[start] and productive[end]
        ]
        if not kept:
            empty = np.empty(0, dtype=np.int32)
            return KeyLattice(width, empty, empty, empty, np.empty(0, dtype=np.float64))
        sizes = [matches.ids.size for _, _, matches in kept]
        return KeyLattice(
            keys=width,
            start=np.repeat(np.array([start for start, _, _ in kept], dtype=np.int32), sizes),
            end=np.repeat(np.array([end for _, end, _ in kept], dtype=np.int32), sizes),
            character=np.concatenate([matches.ids for _, _, matches in kept]),
            cost=np.concatenate([matches.costs for _, _, matches in kept]),
        )


def _splitmix(values: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """SplitMix64's finaliser, elementwise, wrapping mod 2^64."""
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def lattice_digest(lattice: KeyLattice) -> str:
    """An order-free 64-bit digest of the lattice's arcs, costs at micro-unit precision.

    Each arc hashes as ``mix(start << 40 | end << 24 | character) ^
    mix(round(cost * 1e6))`` and the digest is the wrapping sum; the Rust
    test computes the same, so the two lattices agree arc for arc without the
    fixture listing them.
    """
    packed: npt.NDArray[np.uint64] = np.bitwise_or(
        np.bitwise_or(
            np.left_shift(lattice.start.astype(np.uint64), np.uint64(40)),
            np.left_shift(lattice.end.astype(np.uint64), np.uint64(24)),
        ),
        lattice.character.astype(np.uint64),
    )
    micro = np.floor(lattice.cost * 1e6 + 0.5).astype(np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        total = np.sum(_splitmix(packed) ^ _splitmix(micro), dtype=np.uint64)
    return f"{int(total):016x}"


#: The hand-written fixture cases: #216 §3's table, then one case per rule
#: and per rule boundary. Each is ``(keys, sentence, unfinished_tail,
#: correct_syllables)``.
RULE_CASES: tuple[tuple[str, str, bool, bool], ...] = (
    # §3's table.
    ("wmzyq", "我们吃水饺", False, True),
    ("shuijiap", "水饺", False, True),
    ("shuijiap", "水家铺", False, True),
    ("sj", "水饺", False, True),
    ("sk", "烧烤", False, True),
    ("sl", "沙拉", False, True),
    ("sn", "酸奶", False, True),
    ("zgrm", "中国人民", False, True),
    ("xian", "西安", False, True),
    ("xian", "先", False, True),
    ("hm", "噷", False, True),
    ("hm", "好们", False, True),
    # Initials are never corrected, live or offline; the right ones read.
    ("wmzyq", "我们吃水饺", True, True),
    ("wmcsj", "我们吃水饺", False, True),
    ("z", "啊", False, True),
    ("zh", "啊", False, True),
    ("zh", "啊", True, True),
    # Full syllables, isolated from the correction self entry.
    ("nihao", "你好", False, False),
    ("nihao", "你们", False, False),
    # Initials: one letter, and zh/ch/sh standing whole.
    ("zg", "中国", False, False),
    ("zhg", "中国", False, False),
    ("chf", "吃饭", False, False),
    ("zhg", "中", False, False),
    # The unfinished tail: only at the end, only when allowed.
    ("guozho", "国中", True, True),
    ("guozho", "国中", False, True),
    ("zhoguo", "中国", True, True),
    # Corrections off the inventory, and on spans that are syllables.
    ("wojaijia", "我家家", False, False),
    ("shuijiap", "水饺", False, False),
    ("zao", "找", False, True),
    ("zao", "找", False, False),
    ("zao", "早", False, False),
    # Every key consumed, none twice.
    ("aa", "啊啊", False, True),
    ("a", "啊啊", False, True),
    ("ab", "啊", False, True),
    # Polyphones read under any reading.
    ("chong", "重", False, False),
    ("zhong", "重", False, False),
)


def training_sample(
    sample_shards: Iterable[Path],
    labels_dir: Path,
    legality: Legality,
    seed: int,
    characters: range,
) -> list[tuple[str, str]]:
    """The first fitting training record of each shard, typed in every style.

    A record fits when it is labelled, all Han, its length in *characters*,
    and each style's keys read it legally offline. Readings take the training
    path -- keyboard form, then the reading arbitration -- and keys come from
    :func:`~mlime.train.samples.type_syllables` with the style forced and the
    record's generator seeded as training seeds it, at epoch 0. The result is
    every record typed full, then abbreviated, then mixed, as ``(keys, text)``.
    """
    from mlime.data.text import HAN
    from mlime.train.arbitration import ReadingArbitration
    from mlime.train.labels import keyboard_form
    from mlime.train.samples import Augmentation, example_rng, read_pair, type_syllables
    from mlime.train.spans import SpanVocab

    styles = (
        Augmentation(full=1.0, abbreviated=0.0, mixed=0.0),
        Augmentation(full=0.0, abbreviated=1.0, mixed=0.0),
        Augmentation(full=0.0, abbreviated=0.0, mixed=1.0),
    )
    arbitration = ReadingArbitration.load(SpanVocab.load())
    typed: list[list[tuple[str, str]]] = [[] for _ in styles]
    for shard in sample_shards:
        for sample, syllables in read_pair(shard, labels_dir):
            text = sample.text
            if (
                syllables is None
                or len(text) not in characters
                or len(text) != len(syllables)
                or not all(HAN.match(character) for character in text)
            ):
                continue
            readings = tuple(
                arbitration.reading(character, keyboard_form(syllable))
                for character, syllable in zip(text, syllables, strict=True)
            )
            keys = [
                "".join(type_syllables(readings, example_rng(seed, 0, sample.id), style))
                for style in styles
            ]
            if all(legality.cost(each, text, OFFLINE) is not None for each in keys):
                for rows, each in zip(typed, keys, strict=True):
                    rows.append((each, text))
                break
        else:
            raise ValueError(f"no record in {shard} fits")
    return [row for rows in typed for row in rows]


def write_legality_fixture(
    legality: Legality, records: Sequence[tuple[str, str]], out: Path
) -> int:
    """Write the shared legality fixture and return how many cases it holds.

    ``cases`` pins each ``(keys, sentence, options)`` to its cost or to
    illegal: the hand-written :data:`RULE_CASES`, then *records* -- training
    key strings and their sentences, from :func:`training_sample` -- under
    offline and live options.
    ``lattices`` pins every case's keys' whole lattice by arc count, cost sum
    and :func:`lattice_digest`. *records* is the fixture's own ``records``
    list when regenerating, so the file is rebuilt from itself.
    """
    import json

    cases: list[dict[str, object]] = []
    lattices: dict[tuple[str, bool, bool], dict[str, object]] = {}
    rows = [*RULE_CASES]
    for keys, text in records:
        rows.append((keys, text, False, True))
        rows.append((keys, text, True, True))
    for keys, text, tail, correct in rows:
        options = LegalityOptions(unfinished_tail=tail, correct_syllables=correct)
        cases.append(
            {
                "keys": keys,
                "sentence": text,
                "unfinished_tail": tail,
                "correct_syllables": correct,
                "cost": legality.cost(keys, text, options),
            }
        )
        if (keys, tail, correct) not in lattices:
            lattice = legality.lattice(keys, options)
            lattices[(keys, tail, correct)] = {
                "keys": keys,
                "unfinished_tail": tail,
                "correct_syllables": correct,
                "arcs": int(lattice.cost.size),
                "cost_sum": float(lattice.cost.sum()),
                "digest": lattice_digest(lattice),
            }
    document = {
        "records": [{"keys": keys, "sentence": text} for keys, text in records],
        "cases": cases,
        "lattices": list(lattices.values()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return len(cases)
