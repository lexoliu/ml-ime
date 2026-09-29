"""The typo noise model: what a QWERTY typist presses, versus what they meant.

The model corrupts the keystroke string -- the run-together toneless pinyin an
evaluation record carries as ``pinyin`` -- never the target text. It is the one
description of typo noise the whole project shares: this module applies it for
evaluation twins (#102) and for training augmentation (#103), and the Rust
decoder (#104) reads the same table to price corrected segmentations.

The table lives in ``crates/ime-pinyin/data/typo.json``, generated from
``python/src/mlime/data/typo_model.json`` by ``mlime gen-pinyin-tables`` next to
the syllable inventory it describes errors over. It holds, with their
probabilities:

- the QWERTY neighbour table: which keys a finger can slip to from each letter;
- the initials a syllable decomposes into, needed to place the fuzzy pairs;
- the fuzzy-pinyin pairs -- z/zh, c/ch, s/sh, n/l as initials and an/ang,
  en/eng, in/ing as finals -- with a weight per direction;
- the rate of each edit operation and the distribution over how many edits a
  corrupted sentence carries, plus the per-sentence probability of any typo.

Five edit operations act on the keystroke string: substitution of a letter by
an adjacent key, transposition of two neighbouring letters, omission of a
letter, doubling of a letter, and a fuzzy-pair swap inside a syllable. A fuzzy
swap replaces a syllable's whole initial or whole final, so ``tian`` cannot
become ``tiang``: the -n/-ng confusion is a property of the rhyme, and ``ian``
is a different one.

Like the typing styles in ``ime-g2p/src/typing.rs``, the draw is seeded per
record from the record's own clean keystrokes, text and context, so a corrupted
sentence is a property of the sentence and survives reordering and regrowth of
the set. Python's generator is a Mersenne Twister and Rust's is a ``StdRng``;
the two cannot agree, and what has to match -- as with the typing styles -- is
the shape of the result, which is what the shared table pins down.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

from mlime.locate import find_upwards
from mlime.logging import log

#: Where the generated table lives inside the repository.
TYPO_TABLE_RELATIVE = Path("crates/ime-pinyin/data/typo.json")

#: Where the inventory of valid syllables lives inside the repository.
SYLLABLES_RELATIVE = Path("crates/ime-pinyin/data/syllables.txt")

#: The longest syllable in the inventory, mirroring `ime_pinyin::MAX_SYLLABLE_LEN`.
MAX_SYLLABLE_LEN = 6

#: The initials spelled with two letters rather than one.
MULTI_LETTER_INITIALS = ("zh", "ch", "sh")

#: The edit operations the model draws between.
EDIT_TYPES = ("adjacent", "transposition", "omission", "doubling", "fuzzy")


def default_typo_table() -> Path | None:
    """The generated typo table, found upwards from the working directory."""
    return find_upwards(TYPO_TABLE_RELATIVE)


def load_syllables(path: Path) -> set[str]:
    """The syllable inventory, one spelling per line."""
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


@dataclass(frozen=True)
class FuzzyPair:
    """One directional fuzzy substitution, and where inside a syllable it sits."""

    frm: str
    to: str
    where: str  # "initial" or "final"
    weight: float


@dataclass(frozen=True)
class Edit:
    """One corruption applied to the keystroke string, for the histogram."""

    kind: str
    detail: str


@dataclass(frozen=True)
class Corruption:
    """The corrupted keystroke string and the edits that produced it.

    ``spans`` is the corrupted text sliced back at the bounds the edits shifted:
    ``spans[i]`` is what was pressed for syllable ``i`` of the input, so training
    keeps its per-character alignment through a corruption that moves letters
    across span boundaries.
    """

    text: str
    edits: tuple[Edit, ...]
    spans: tuple[str, ...] = ()


class NoiseModel:
    """Corrupts keystroke strings with the rates and tables from ``typo.json``."""

    def __init__(
        self,
        per_sentence: float,
        edit_count: dict[int, float],
        edit_type: dict[str, float],
        initials: frozenset[str],
        neighbours: dict[str, str],
        fuzzy_pairs: tuple[FuzzyPair, ...],
    ) -> None:
        self._per_sentence = per_sentence
        self._edit_count = dict(sorted(edit_count.items()))
        self._edit_type = edit_type
        self._initials = initials
        self._neighbours = neighbours
        self._fuzzy_pairs = fuzzy_pairs

    @classmethod
    def parse(cls, raw: str) -> NoiseModel:
        """Read the table, refusing anything malformed.

        Two languages consume this file, so every shape check lives here where
        it can be tested, not in the generator that copies the bytes.
        """
        data = _mapping(json.loads(raw), "typo table")
        unknown = set(data) - {
            "per_sentence",
            "edit_count",
            "edit_type",
            "initials",
            "neighbours",
            "fuzzy_pairs",
        }
        if unknown:
            raise ValueError(f"the typo table holds unknown keys: {sorted(unknown)}")
        per_sentence = _probability(data["per_sentence"], "per_sentence")
        edit_count = {
            _count(k): _probability(v, f"edit_count[{k!r}]")
            for k, v in _mapping(data["edit_count"], "edit_count").items()
        }
        _check_distribution(edit_count.values(), "edit_count")
        edit_type = {
            str(k): _probability(v, f"edit_type[{k!r}]")
            for k, v in _mapping(data["edit_type"], "edit_type").items()
        }
        if set(edit_type) != set(EDIT_TYPES):
            raise ValueError(
                f"edit_type must name exactly {sorted(EDIT_TYPES)}, got {sorted(edit_type)}"
            )
        _check_distribution(edit_type.values(), "edit_type")
        initials = frozenset(
            _word(item, f"initials[{index}]")
            for index, item in enumerate(_list(data["initials"], "initials"))
        )
        if not set(MULTI_LETTER_INITIALS) <= initials:
            raise ValueError(f"initials must include {list(MULTI_LETTER_INITIALS)}")
        neighbours = {
            _key(k): _word(v, f"neighbours[{k!r}]")
            for k, v in _mapping(data["neighbours"], "neighbours").items()
        }
        if set(neighbours) != set("abcdefghijklmnopqrstuvwxyz"):
            raise ValueError("neighbours must cover exactly [a-z]")
        for key, near in neighbours.items():
            if key in near:
                raise ValueError(f"{key!r} is its own neighbour")
            if len(set(near)) != len(near):
                raise ValueError(f"neighbours[{key!r}] repeats a key: {near!r}")
            for other in near:
                if key not in neighbours[other]:
                    raise ValueError(f"{key!r} neighbours {other!r} but not vice versa")
        pairs: list[FuzzyPair] = []
        seen: set[tuple[str, str]] = set()
        for index, entry in enumerate(_list(data["fuzzy_pairs"], "fuzzy_pairs")):
            where = f"fuzzy_pairs[{index}]"
            fields = _mapping(entry, where)
            pair = FuzzyPair(
                frm=_word(fields.get("from"), where),
                to=_word(fields.get("to"), where),
                where=_position(fields.get("where"), where),
                weight=_weight(fields.get("weight"), where),
            )
            if pair.frm == pair.to:
                raise ValueError(f"{where}: the pair {pair.frm!r} maps to itself")
            if pair.where == "initial" and pair.frm not in initials:
                raise ValueError(f"{where}: {pair.frm!r} is not one of the initials")
            if (pair.frm, pair.to) in seen:
                raise ValueError(f"{where}: the pair {pair.frm!r}->{pair.to!r} is duplicated")
            seen.add((pair.frm, pair.to))
            pairs.append(pair)
        return cls(per_sentence, edit_count, edit_type, initials, neighbours, tuple(pairs))

    @classmethod
    def load(cls, path: Path) -> NoiseModel:
        """Read the shared table from *path*."""
        return cls.parse(path.read_text(encoding="utf-8"))

    @property
    def per_sentence(self) -> float:
        """The probability that a typed sentence carries at least one typo."""
        return self._per_sentence

    def initial_of(self, syllable: str) -> str:
        """The syllable's initial: the longest initial prefix, or ``""``."""
        for length in (2, 1):
            if len(syllable) >= length and syllable[:length] in self._initials:
                return syllable[:length]
        return ""

    def final_of(self, syllable: str) -> str:
        """The syllable's final: what its initial leaves."""
        return syllable[len(self.initial_of(syllable)) :]

    def corrupt(
        self, syllables: Sequence[str], rng: random.Random, *, min_edits: int = 1
    ) -> Corruption:
        """Corrupt *syllables* with at least *min_edits* edits.

        The number of edits is drawn from the table's count distribution,
        resampled until it reaches *min_edits*: an evaluation twin that must
        corrupt every record passes one, while augmentation draws zero-or-more
        through the per-sentence coin in [`maybe_corrupt`].
        """
        if not syllables:
            raise ValueError("there is nothing to corrupt in an empty syllable list")
        letters: list[str] = [letter for syllable in syllables for letter in syllable]
        bounds: list[list[int]] = []
        cursor = 0
        for syllable in syllables:
            bounds.append([cursor, cursor + len(syllable)])
            cursor += len(syllable)
        edits = [self._apply(rng, letters, bounds) for _ in range(self._draw_count(rng, min_edits))]
        return Corruption(
            "".join(letters),
            tuple(edits),
            tuple("".join(letters[start:end]) for start, end in bounds),
        )

    def maybe_corrupt(self, syllables: Sequence[str], rng: random.Random) -> Corruption:
        """The augmentation entry point: the per-sentence coin, then [`corrupt`]."""
        if rng.random() >= self._per_sentence:
            return Corruption("".join(syllables), (), tuple(syllables))
        return self.corrupt(syllables, rng, min_edits=1)

    def _draw_count(self, rng: random.Random, at_least: int) -> int:
        counts = list(self._edit_count)
        weights = [self._edit_count[count] for count in counts]
        count = rng.choices(counts, weights)[0]
        while count < at_least:
            count = rng.choices(counts, weights)[0]
        return count

    def _apply(self, rng: random.Random, letters: list[str], bounds: list[list[int]]) -> Edit:
        """Apply one edit of a kind drawn among those the string admits."""
        applicable = [
            kind for kind in EDIT_TYPES if kind != "fuzzy" or self._fuzzy_sites(letters, bounds)
        ]
        if len(letters) < 2:
            applicable.remove("transposition")
            applicable.remove("omission")
        kind = rng.choices(applicable, [self._edit_type[one] for one in applicable])[0]
        if kind == "adjacent":
            at = rng.randrange(len(letters))
            near = self._neighbours[letters[at]]
            replacement = rng.choice(near)
            edit = Edit("adjacent", f"{letters[at]}>{replacement}@{at}")
            letters[at] = replacement
            return edit
        if kind == "transposition":
            at = rng.randrange(len(letters) - 1)
            letters[at], letters[at + 1] = letters[at + 1], letters[at]
            return Edit("transposition", f"swap@{at}")
        if kind == "omission":
            at = rng.randrange(len(letters))
            edit = Edit("omission", f"{letters[at]}@{at}")
            del letters[at]
            _shift(bounds, at, -1)
            return edit
        if kind == "doubling":
            at = rng.randrange(len(letters))
            letters.insert(at + 1, letters[at])
            _shift(bounds, at + 1, 1)
            return Edit("doubling", f"{letters[at]}@{at}")
        return self._apply_fuzzy(rng, letters, bounds)

    def fuzzy_sites(self, syllables: Sequence[str]) -> list[tuple[int, FuzzyPair]]:
        """Every (syllable index, applicable pair) a clean spelling admits."""
        letters = [letter for syllable in syllables for letter in syllable]
        bounds: list[list[int]] = []
        cursor = 0
        for syllable in syllables:
            bounds.append([cursor, cursor + len(syllable)])
            cursor += len(syllable)
        return self._fuzzy_sites(letters, bounds)

    def _fuzzy_sites(
        self, letters: Sequence[str], bounds: Sequence[Sequence[int]]
    ) -> list[tuple[int, FuzzyPair]]:
        """Every (syllable, pair) the current string admits."""
        sites: list[tuple[int, FuzzyPair]] = []
        for index, bound in enumerate(bounds):
            syllable = "".join(letters[bound[0] : bound[1]])
            for pair in self._fuzzy_pairs:
                if (pair.where == "initial" and self.initial_of(syllable) == pair.frm) or (
                    pair.where == "final" and self.final_of(syllable) == pair.frm
                ):
                    sites.append((index, pair))
        return sites

    def _apply_fuzzy(self, rng: random.Random, letters: list[str], bounds: list[list[int]]) -> Edit:
        """Apply one fuzzy swap, chosen among the sites by pair weight."""
        sites = self._fuzzy_sites(letters, bounds)
        if not sites:
            raise ValueError("a fuzzy edit was drawn with no site to apply it to")
        index, pair = rng.choices(sites, [site[1].weight for site in sites])[0]
        start, end = bounds[index]
        syllable = "".join(letters[start:end])
        if pair.where == "initial":
            replacement = pair.to + syllable[len(pair.frm) :]
        else:
            replacement = syllable[: -len(pair.frm)] + pair.to
        letters[start:end] = replacement
        shift = len(replacement) - (end - start)
        bounds[index][1] += shift
        for bound in bounds[index + 1 :]:
            bound[0] += shift
            bound[1] += shift
        return Edit("fuzzy", f"{pair.frm}>{pair.to}@{index}")


def _shift(bounds: list[list[int]], at: int, by: int) -> None:
    """Move bounds after an insertion or deletion at *at*.

    A pressed key always belongs to a bound: an insertion landing exactly on a
    seam joins the bound that ends there -- it repeats that bound's last letter
    -- while the bound that begins there only moves. A deletion touching a
    bound's first letter shrinks that bound, never moves it under the previous
    span's tail.
    """
    for bound in bounds:
        if by > 0:
            if bound[0] >= at:
                bound[0] += by
                bound[1] += by
            elif bound[1] >= at:
                bound[1] += by
        else:
            if bound[0] > at:
                bound[0] += by
                bound[1] += by
            elif bound[1] > at:
                bound[1] += by


#: Corrections kept per typed span, per the shared #101 spec.
CORRECTIONS_K = 8

#: The deepest edit distance a correction can sit at, per the same spec.
CORRECTIONS_MAX_EDITS = 2

#: The generated correction fixture the Rust parity test reads.
CORRECTIONS_FIXTURE_RELATIVE = Path("crates/ime-pinyin/tests/fixtures/typo-corrections.json")


class CorrectionTable:
    """``corrections(span)``: the syllables the noise model can bend into *span*.

    This is the #101 spec's correction function: for a typed span ``s``, every
    syllable ``y`` in the inventory that the noise model can turn into ``s``
    within at most two edits, costed ``-log P(s | y)`` under the model's
    edit-type probabilities. ``P(s | y)`` sums the probability of every one-
    and two-edit path that produces ``s`` -- the edit-count distribution and the
    per-sentence coin do not enter it, only the per-edit draws.

    When ``s`` is itself a valid syllable or a valid prefix (the abbreviation
    case), ``s`` appears in its own list at cost 0: reading the typed letters as
    meant is always an option the decoder may price. Entries sort by cost with
    the syllable breaking ties, and at most ``K`` of them survive -- the cap is
    what keeps a high-entropy typo from exploding the candidate union.
    """

    def __init__(
        self,
        model: NoiseModel,
        syllables: Iterable[str],
        prefixes: Iterable[str] | None = None,
        k: int = CORRECTIONS_K,
    ) -> None:
        if k < 1:
            raise ValueError(f"the correction table keeps at least one entry, got {k}")
        self._model = model
        self._syllables = frozenset(syllables)
        if prefixes is None:
            prefixes = (
                syllable[:length]
                for syllable in self._syllables
                for length in range(1, len(syllable))
            )
        self._prefixes = frozenset(prefixes)
        self._k = k
        self._cache: dict[str, tuple[tuple[str, float], ...]] = {}

    @classmethod
    def load(
        cls, typo_table: Path, syllables_path: Path, k: int = CORRECTIONS_K
    ) -> CorrectionTable:
        """Build the table the generated ``typo.json`` and syllable inventory describe."""
        return cls(NoiseModel.load(typo_table), load_syllables(syllables_path), k=k)

    def corrections(self, span: str) -> tuple[tuple[str, float], ...]:
        """The ``(syllable, cost)`` entries for *span*, sorted then capped at ``K``."""
        cached = self._cache.get(span)
        if cached is None:
            cached = self._resolve(span)
            self._cache[span] = cached
        return cached

    def _resolve(self, span: str) -> tuple[tuple[str, float], ...]:
        costs: dict[str, float] = {}
        predecessors = self._predecessors(span)
        origins = predecessors | set().union(
            *(self._predecessors(intermediate) for intermediate in predecessors)
        )
        for syllable in sorted(origins & self._syllables):
            probability = self._one_edit_probability(syllable, span) + sum(
                draw * self._one_edit_probability(intermediate, span)
                for intermediate, draw in self._outcomes(syllable).items()
            )
            if probability > 0.0:
                costs[syllable] = -math.log(probability)
        if span in self._syllables or span in self._prefixes:
            costs[span] = 0.0
        ranked = sorted(costs.items(), key=lambda entry: (entry[1], entry[0]))
        return tuple(ranked[: self._k])

    def _applicable(self, letters: str) -> dict[str, float]:
        """Each edit kind's probability at *letters*, normalised the way `corrupt` draws it."""
        sites = self._sites(letters)
        kinds = [kind for kind in EDIT_TYPES if kind != "fuzzy" or sites]
        if not letters:
            kinds.clear()
        elif len(letters) < 2:
            kinds.remove("transposition")
            kinds.remove("omission")
        weights = {kind: self._model._edit_type[kind] for kind in kinds}
        total = sum(weights.values())
        return {kind: weight / total for kind, weight in weights.items()}

    def _sites(self, letters: str) -> list[FuzzyPair]:
        """The fuzzy pairs the single-syllable string admits, as `fuzzy_sites` finds them."""
        return [
            pair
            for pair in self._model._fuzzy_pairs
            if (pair.where == "initial" and self._model.initial_of(letters) == pair.frm)
            or (pair.where == "final" and self._model.final_of(letters) == pair.frm)
        ]

    def _apply_pair(self, letters: str, pair: FuzzyPair) -> str:
        """The syllable after one fuzzy swap, as `_apply_fuzzy` writes it."""
        if pair.where == "initial":
            return pair.to + letters[len(pair.frm) :]
        return letters[: -len(pair.frm)] + pair.to

    def _outcomes(self, letters: str) -> dict[str, float]:
        """Every ``(string, probability)`` one edit of *letters* produces, duplicates merged."""
        kinds = self._applicable(letters)
        length = len(letters)
        out: dict[str, float] = {}

        def add(produced: str, probability: float) -> None:
            out[produced] = out.get(produced, 0.0) + probability

        if "adjacent" in kinds:
            each = kinds["adjacent"] / length
            for index, letter in enumerate(letters):
                near = self._model._neighbours[letter]
                for replacement in near:
                    add(
                        letters[:index] + replacement + letters[index + 1 :],
                        each / len(near),
                    )
        if "transposition" in kinds:
            each = kinds["transposition"] / (length - 1)
            for index in range(length - 1):
                add(
                    letters[:index] + letters[index + 1] + letters[index] + letters[index + 2 :],
                    each,
                )
        if "omission" in kinds:
            each = kinds["omission"] / length
            for index in range(length):
                add(letters[:index] + letters[index + 1 :], each)
        if "doubling" in kinds:
            each = kinds["doubling"] / length
            for index in range(length):
                add(letters[: index + 1] + letters[index] + letters[index + 1 :], each)
        if "fuzzy" in kinds:
            sites = self._sites(letters)
            weight = sum(pair.weight for pair in sites)
            for pair in sites:
                add(self._apply_pair(letters, pair), kinds["fuzzy"] * pair.weight / weight)
        return out

    def _one_edit_probability(self, source: str, produced: str) -> float:
        """``P(produced | source)`` for one edit, summed over every way it can happen."""
        kinds = self._applicable(source)
        length = len(source)
        probability = 0.0
        if len(produced) == length:
            diffs = [
                i
                for i, (left, right) in enumerate(zip(source, produced, strict=True))
                if left != right
            ]
            if not diffs and "transposition" in kinds:
                # Transposing a pair of equal letters is a real draw that changes nothing.
                hits = sum(1 for i in range(length - 1) if source[i] == source[i + 1])
                probability += kinds["transposition"] * hits / (length - 1)
            if len(diffs) == 1:
                index = diffs[0]
                near = self._model._neighbours[source[index]]
                if produced[index] in near:
                    probability += kinds["adjacent"] / length / len(near)
            if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
                index = diffs[0]
                if source[index] == produced[index + 1] and source[index + 1] == produced[index]:
                    probability += kinds["transposition"] / (length - 1)
        elif len(produced) == length - 1 and "omission" in kinds:
            hits = sum(
                1 for index in range(length) if source[:index] + source[index + 1 :] == produced
            )
            probability += kinds["omission"] * hits / length
        elif len(produced) == length + 1 and "doubling" in kinds:
            hits = sum(
                1
                for index in range(length)
                if source[: index + 1] + source[index] + source[index + 1 :] == produced
            )
            probability += kinds["doubling"] * hits / length
        if "fuzzy" in kinds:
            sites = self._sites(source)
            weight = sum(pair.weight for pair in sites)
            for pair in sites:
                if self._apply_pair(source, pair) == produced:
                    probability += kinds["fuzzy"] * pair.weight / weight
        return probability

    def _predecessors(self, produced: str) -> set[str]:
        """Every string one edit could have become *produced* from, valid or not.

        The inverse of `_outcomes`: adjacent and transposition are their own
        inverses (the neighbour table is symmetric by construction), omission
        inverts to inserting any letter anywhere, doubling inverts to deleting
        one letter of a repeated pair, and a fuzzy swap inverts to the pair's
        other direction at the same site.
        """
        length = len(produced)
        out: set[str] = set()
        for index, letter in enumerate(produced):
            for original in self._model._neighbours[letter]:
                out.add(produced[:index] + original + produced[index + 1 :])
        for index in range(length - 1):
            out.add(
                produced[:index] + produced[index + 1] + produced[index] + produced[index + 2 :]
            )
        for index in range(length + 1):
            for letter in "abcdefghijklmnopqrstuvwxyz":
                out.add(produced[:index] + letter + produced[index:])
        for index in range(1, length):
            if produced[index] == produced[index - 1]:
                out.add(produced[:index] + produced[index + 1 :])
        for pair in self._model._fuzzy_pairs:
            if pair.where == "initial" and self._model.initial_of(produced) == pair.to:
                out.add(pair.frm + produced[len(pair.to) :])
            if pair.where == "final" and self._model.final_of(produced) == pair.to:
                out.add(produced[: -len(pair.to)] + pair.frm)
        return out


def record_seed(seed: int, *fields: str) -> int:
    """A per-record seed from the set's seed and the record's identity.

    The identity is what ``ime_eval::EvalRecord::digest`` hashes -- the clean
    keystrokes, the target and the context -- so a record's corruption is a
    property of the record and survives reordering and regrowth of the set, the
    same argument the typing styles make for seeding from text.
    """
    digest = blake2b(digest_size=8)
    digest.update(seed.to_bytes(8, "big"))
    for field in fields:
        digest.update(field.encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "big")


def segmentable(joined: str, inventory: set[str]) -> bool:
    """Whether *joined* has any reading under the evaluation's segmentation.

    Mirrors ``ime_pinyin::SegmentLattice`` under fused-eval's options:
    abbreviations allowed, no unfinished trailing syllable. A span counts when
    it is a whole syllable, or an abbreviation -- one letter or zh/ch/sh -- that
    still prefixes some syllable. Used to report how much of a typo twin the
    lattice can no longer read.
    """
    first_letters = {syllable[0] for syllable in inventory}
    two_letter = {syllable[:2] for syllable in inventory}
    length = len(joined)
    reachable = [False] * (length + 1)
    reachable[0] = True
    for i in range(length):
        if not reachable[i]:
            continue
        for j in range(i + 1, min(length, i + MAX_SYLLABLE_LEN) + 1):
            span = joined[i:j]
            if span in inventory or (
                (len(span) == 1 and span in first_letters)
                or (len(span) == 2 and span in two_letter and span in MULTI_LETTER_INITIALS)
            ):
                reachable[j] = True
    return reachable[length]


def typo_twin(
    eval_set: Path,
    out: Path,
    seed: int,
    typo_table: Path,
    syllables_path: Path,
    lattice: Path,
) -> str:
    """Write the typo twin of *eval_set* and return the report for the terminal.

    Every record's ``pinyin`` is corrupted with at least one edit; the clean
    keystrokes stay beside it under ``clean``, so ``EvalRecord::digest`` gives
    the twin the parent set's record identities and its dev/test slices match
    the parent's exactly.

    The fuzzy pairs fire on the syllables the typist meant: *lattice*, the
    set's ``emit-lattice`` output, gives that spelling as its character-aligned
    path -- one syllable per ``text`` character. Re-segmenting the keystrokes
    instead would guess, and a guess gets cases like ``keneng`` (``ke+neng``)
    wrong and puts fuzzy swaps at sites the typist never had.
    """
    model = NoiseModel.load(typo_table)
    inventory = load_syllables(syllables_path)
    gold: dict[int, list[list[str]]] = {}
    with lattice.open(encoding="utf-8") as lines:
        for raw in lines:
            if not raw.strip():
                continue
            row = _mapping(json.loads(raw), f"{lattice} record")
            index = int(str(row["record"]))
            where = f"{lattice} record {index}"
            gold[index] = [
                [str(span) for span in _list(_mapping(path, where)["spans"], where)]
                for path in _list(row["paths"], where)
            ]
    histogram = {kind: 0 for kind in EDIT_TYPES}
    unsegmentable = 0
    records = 0
    edits = 0
    with eval_set.open(encoding="utf-8") as source, out.open("w", encoding="utf-8") as sink:
        for index, raw in enumerate(source):
            if not raw.strip():
                continue
            row = _mapping(json.loads(raw), f"{eval_set} record")
            pinyin = _word(row["pinyin"], "pinyin")
            text = str(row["text"])
            context = row.get("context")
            if index not in gold:
                raise ValueError(f"{lattice} holds no record {index}")
            aligned = [spans for spans in gold[index] if len(spans) == len(text)]
            if not aligned or "".join(aligned[0]) != pinyin:
                raise ValueError(
                    f"{lattice} record {index} has no {len(text)}-span path for {pinyin!r}"
                )
            syllables = aligned[0]
            rng = random.Random(record_seed(seed, pinyin, text, str(context or "")))
            corruption = model.corrupt(syllables, rng, min_edits=1)
            if not segmentable(corruption.text, inventory):
                unsegmentable += 1
            for edit in corruption.edits:
                histogram[edit.kind] += 1
            edits += len(corruption.edits)
            line: dict[str, object] = {
                "pinyin": corruption.text,
                "text": text,
                "context": context,
                "clean": pinyin,
            }
            sink.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
            records += 1
    log.info(
        "typo twin written",
        out=str(out),
        records=records,
        edits=edits,
        unsegmentable=unsegmentable,
    )
    report = [f"{records} records, {edits} edits", "edit types:"]
    report += [f"  {kind:14s}{histogram[kind]}" for kind in EDIT_TYPES]
    report.append(f"unsegmentable under the lattice: {unsegmentable}")
    return "\n".join(report)


def corrections_fixture_spans(table: CorrectionTable) -> tuple[str, ...]:
    """The fixed ~200 spans the shared correction fixture covers.

    Both languages' `corrections` must agree on every entry of this list: a
    slice of valid syllables, the prefixes abbreviations produce, typos one edit
    away, typos two edits away, and the fuzzy swaps the base syllables admit --
    all fourteen pair directions. The list is deterministic -- rebuilt, it is
    identical, so the test can fail on any drift the way the table itself does.
    """
    model = table._model
    spans: list[str] = []
    # Common syllables across the letter range: every initial -- single and
    # multi-letter -- appears, and every fuzzy pair fires somewhere.
    wanted = {
        "a",
        "ai",
        "an",
        "ang",
        "ba",
        "bei",
        "bu",
        "can",
        "cha",
        "chi",
        "chong",
        "ci",
        "de",
        "e",
        "en",
        "eng",
        "er",
        "fa",
        "guo",
        "hao",
        "he",
        "ji",
        "jia",
        "jing",
        "kan",
        "lai",
        "lan",
        "lv",
        "ma",
        "mei",
        "na",
        "ni",
        "o",
        "pa",
        "qin",
        "ren",
        "shi",
        "shui",
        "si",
        "suo",
        "ta",
        "tian",
        "wo",
        "xia",
        "xin",
        "yi",
        "yu",
        "zai",
        "zhe",
        "zhi",
        "zhong",
        "zi",
        "zuo",
    }
    base = [syllable for syllable in sorted(table._syllables) if syllable in wanted]
    if len(base) != len(wanted):
        missing = wanted - set(base)
        raise ValueError(f"the fixture asks about non-syllables: {sorted(missing)}")
    spans.extend(base)
    # The abbreviation layer: every prefix of the common syllables that is not
    # itself a whole syllable, so the cost-0 self entry is exercised both ways.
    spans.extend(
        sorted(
            {
                syllable[:length]
                for syllable in base
                for length in range(1, len(syllable))
                if syllable[:length] not in table._syllables
            }
        )
    )
    # One-edit typos: an adjacent slip at the head of every base syllable, then
    # a second one whose kind rotates so transposition, omission and doubling
    # each appear across the set.
    kinds = ("transposition", "omission", "doubling")
    for index, syllable in enumerate(base):
        spans.append(model._neighbours[syllable[0]][0] + syllable[1:])
        if len(syllable) < 2:
            continue
        kind = kinds[index % 3]
        if kind == "transposition":
            spans.append(syllable[1] + syllable[0] + syllable[2:])
        elif kind == "omission":
            spans.append(syllable[1:])
        else:
            spans.append(syllable + syllable[-1])
    # Two-edit typos: an adjacent slip at the head followed by an omission at
    # the tail, on a rotating third of the longer syllables.
    for index, syllable in enumerate(base):
        if len(syllable) >= 3 and index % 3 == 0:
            slipped = model._neighbours[syllable[0]][0] + syllable[1:]
            spans.append(slipped[:-1])
    # Fuzzy swaps, both directions of every pair a base syllable admits.
    for syllable in base:
        initial, final = model.initial_of(syllable), model.final_of(syllable)
        for pair in model._fuzzy_pairs:
            if pair.where == "initial" and initial == pair.frm:
                spans.append(pair.to + syllable[len(pair.frm) :])
            if pair.where == "final" and final == pair.frm:
                spans.append(syllable[: -len(pair.frm)] + pair.to)
    return tuple(dict.fromkeys(spans))


def write_corrections_fixture(table: CorrectionTable, out: Path) -> int:
    """Write the shared ``corrections`` fixture and return the span count.

    The file is the contract the Rust test (#104) pins its own implementation
    against: one sorted span per line-entry with its ``[(syllable, cost)]``
    list, floats at full precision, so a regenerated file either agrees bit for
    bit or the change was real.
    """
    rows = {
        span: [[syllable, cost] for syllable, cost in table.corrections(span)]
        for span in corrections_fixture_spans(table)
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(rows, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object, got {value!r}")
    return value


def _list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list, got {value!r}")
    return value


def _probability(value: object, where: str) -> float:
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{where} must be a probability, got {value!r}")
    return float(value)


def _weight(value: object, where: str) -> float:
    if not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise ValueError(f"{where} must be a positive weight, got {value!r}")
    return float(value)


def _position(value: object, where: str) -> str:
    if value not in ("initial", "final"):
        raise ValueError(f"{where} must place the pair at 'initial' or 'final', got {value!r}")
    return str(value)


def _count(value: object) -> int:
    if not isinstance(value, str) or not value.isdecimal() or int(value) < 1:
        raise ValueError(f"an edit count must be a positive count, got {value!r}")
    return int(value)


def _word(value: object, where: str) -> str:
    """A non-empty ``[a-z]`` string -- a syllable part, a key, or keystrokes."""
    if not isinstance(value, str) or not value.isascii() or not value.isalpha():
        raise ValueError(f"{where} must be an [a-z] string, got {value!r}")
    if not value or not value.islower():
        raise ValueError(f"{where} must be lowercase letters, got {value!r}")
    return value


def _key(value: object) -> str:
    key = _word(value, "a neighbour key")
    if len(key) != 1:
        raise ValueError(f"a neighbour key must be one letter, got {key!r}")
    return key


def _check_distribution(weights: Iterable[float], where: str) -> None:
    total = sum(weights)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{where} must be a distribution; it sums to {total}")
