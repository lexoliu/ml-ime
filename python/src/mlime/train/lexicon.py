"""What the fill tower is allowed to emit, and which characters each span admits.

Two restrictions stack at every output position.

The first is fixed for a run: the model can only emit a character its base
tokenizer has a single token for. The pinyin lexicon holds 41,923 characters and
MacBERT's vocabulary covers 7,322 of them; the rest are CJK extension rarities
that no one types. Restricting the MLM head to that intersection is what makes
the head useful instead of mostly dead, and it has to be computed against the
*model's* vocabulary rather than assumed, because a different base model
intersects differently.

The second is per position: the typed span. A span is a prefix of what the user
meant to type, so the characters it can stand for are those with a reading that
begins with it -- ``zhong`` admits only zhong-characters, ``zh`` admits every
zh-character, and ``z`` admits both the z- and the zh- ones. That is exactly the
``SyllableTable::prefix_range`` rule the Rust decoder segments with, so the mask
the model trains against and the mask the decoder applies are the same relation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

from mlime.logging import log
from mlime.train.spans import SpanVocab
from mlime.typo import CorrectionTable, corrections_fixture_spans

#: The generated candidate-union fixture the Rust parity test reads.
CANDIDATES_FIXTURE_RELATIVE = Path("crates/ime-decode/tests/fixtures/typo-candidates.json")


def read_char_readings(path: Path) -> dict[str, tuple[str, ...]]:
    """Parse ``char_pinyin.tsv`` into ``{character: (reading, ...)}``.

    ``mlime.data.corpus.load_reference_characters`` reads the same file for the
    characters alone; the masks need the readings that file's second column
    holds, which is why the parse is repeated rather than shared.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"no character table at {path}; generate it with `mlime gen-pinyin-tables`"
        )
    table: dict[str, tuple[str, ...]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        character, separator, readings = line.partition("\t")
        if not separator or not readings or len(character) != 1:
            raise ValueError(f"{path}:{number} is not a `<char>\\t<readings>` row: {line!r}")
        table[character] = tuple(readings.split(","))
    if not table:
        raise ValueError(f"{path} holds no characters")
    return table


@dataclass(frozen=True)
class Lexicon:
    """The emittable character set, its base-vocabulary ids, and the span masks.

    ``characters`` fixes an ordering -- the *emission index* -- that the output
    head, the masks and the loss all share. It is not the base vocabulary order;
    logits are gathered down to it once, so every tensor downstream is
    ``[..., len(characters)]``.
    """

    characters: tuple[str, ...]
    token_ids: torch.Tensor
    candidate_mask: torch.Tensor
    #: Each syllable's exact homophones as emission indices -- the candidate
    #: prior's "the span's own readings" and the per-reading prices the
    #: uncapped correction costs attach to.
    homophones: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_ids.shape != (len(self.characters),):
            raise ValueError(
                f"token_ids has shape {tuple(self.token_ids.shape)} for "
                f"{len(self.characters)} characters"
            )
        if self.candidate_mask.shape[1] != len(self.characters):
            raise ValueError(
                f"candidate_mask covers {self.candidate_mask.shape[1]} characters, "
                f"not {len(self.characters)}"
            )
        if self.candidate_mask.dtype is not torch.bool:
            raise TypeError(f"candidate_mask must be boolean, got {self.candidate_mask.dtype}")

    @property
    def size(self) -> int:
        """How many characters the head emits over."""
        return len(self.characters)

    @property
    def spans(self) -> int:
        """How many typed spans the mask is indexed by."""
        return int(self.candidate_mask.shape[0])

    def index(self, character: str) -> int:
        """The emission index of *character*.

        Built lazily on first use; the reverse table costs nothing to keep and
        the sample builder asks for it once per character of every sentence.
        """
        return self._reverse[character]

    def contains(self, character: str) -> bool:
        """Whether *character* is emittable at all."""
        return character in self._reverse

    def admits(self, span_id: int, character: str) -> bool:
        """Whether the span at *span_id* can stand for *character*."""
        return bool(self.candidate_mask[span_id, self.index(character)])

    @property
    def _reverse(self) -> Mapping[str, int]:
        cached = getattr(self, "_reverse_cache", None)
        if cached is None:
            cached = {character: index for index, character in enumerate(self.characters)}
            object.__setattr__(self, "_reverse_cache", cached)
        return cached


@dataclass
class CandidateSpace:
    """The emission indices a typed span admits at one position.

    The clean space is the lexicon mask, looked up by the span's id -- off-inventory
    spans fold onto the ``<unk>`` row, which admits nothing, and the builder's
    admission check then refuses them the way it always has.

    The typo space widens each position to the union of the homophone lists of
    the span's ``corrections``: a valid span keeps its own row (the cost-0 self
    entry covers it), so the widened set is exactly the old one plus whatever
    the noise model could have meant. The corrections cap keeps the union
    bounded; a span no syllable reaches within two edits resolves to the empty
    list, which the same admission check turns into a counted drop.
    """

    spans: SpanVocab
    mask: torch.Tensor
    table: CorrectionTable | None = None
    #: Each syllable's exact homophones as emission indices -- the lexicon's
    #: ``homophones``. The prior needs them to name the span's own readings
    #: and to price each widened character through the syllables that read it.
    homophones: Mapping[str, tuple[int, ...]] | None = None
    _cache: dict[str, tuple[frozenset[int], tuple[int, ...]]] = field(default_factory=dict)
    _prior_cache: dict[str, dict[int, float]] = field(default_factory=dict)
    _char_readings: dict[int, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self.mask.shape[0] != len(self.spans):
            raise ValueError(
                f"the candidate mask covers {self.mask.shape[0]} spans and the "
                f"inventory holds {len(self.spans)}"
            )
        if self.mask.dtype is not torch.bool:
            raise TypeError(f"the candidate mask must be boolean, got {self.mask.dtype}")
        if self.table is not None and self.homophones is None:
            raise ValueError("the typo space's priors need the lexicon's homophones")

    def resolve(self, span: str) -> tuple[int, ...]:
        """The sorted emission ids *span* admits, cached on first ask."""
        cached = self._cache.get(span)
        if cached is None:
            if self.table is None:
                row = self.mask[self.spans.id_or_unknown(span)]
                ids = tuple(int(index) for index in row.nonzero(as_tuple=True)[0].tolist())
            else:
                admitted: set[int] = set()
                for syllable, _cost in self.table.corrections(span):
                    admitted.update(
                        int(index)
                        for index in self.mask[self.spans.id(syllable)]
                        .nonzero(as_tuple=True)[0]
                        .tolist()
                    )
                ids = tuple(sorted(admitted))
            cached = (frozenset(ids), ids)
            self._cache[span] = cached
        return cached[1]

    def priors(self, span: str) -> dict[int, float]:
        """Each emission id's correction prior, the decoder's per-candidate term.

        A character a span reaches only through a correction *y* pays that
        correction's ``-log P(span | y)``; the span's own readings pay nothing
        under the cost-0 self entry, and a space with no noise model prices
        every admitted id at zero the way the Rust clean path does. The Rust
        decoder charges the same term scaled by one weight, so an e2e decoder
        can price the widened set identically. Like :meth:`resolve`, cached on
        first ask.
        """
        cached = self._prior_cache.get(span)
        if cached is None:
            if self.table is None:
                costs = dict.fromkeys(self.resolve(span), 0.0)
            else:
                costs = {}
                for syllable, cost in self.table.corrections(span):
                    # The self entry builds the union but prices nothing: its
                    # row is the keystrokes themselves, and what the segment
                    # honestly reads is `own`, below -- everything else in
                    # that row reached here through a completion the noise
                    # model still has to pay for.
                    if syllable == span:
                        for index in (
                            self.mask[self.spans.id(syllable)].nonzero(as_tuple=True)[0].tolist()
                        ):
                            costs.setdefault(int(index), math.inf)
                        continue
                    for index in (
                        self.mask[self.spans.id(syllable)].nonzero(as_tuple=True)[0].tolist()
                    ):
                        index = int(index)
                        if index not in costs or costs[index] > cost:
                            costs[index] = cost
                # A widened character also pays the cheapest syllable that
                # reads it and reaches the keystrokes at all -- including
                # completions the K cap dropped from the entry list. A
                # character the model cannot reach within two edits keeps
                # its infinity: it stands in the union but no correction
                # admits it.
                full = self.table.costs(span)
                for index, prior in costs.items():
                    costs[index] = min(
                        (
                            prior,
                            *(
                                cost
                                for syllable in self._readings(index)
                                if (cost := full.get(syllable)) is not None
                            ),
                        )
                    )
                # The span's own readings pay nothing -- they were never
                # widened.
                for index in self._own(span):
                    costs[index] = 0.0
            cached = dict(sorted(costs.items()))
            self._prior_cache[span] = cached
        return cached

    def _own(self, span: str) -> frozenset[int]:
        """The span's own readings, as emission ids.

        A full syllable's own are its exact homophones; a prefix-only span's
        are the row's completions; off the inventory there are none.
        """
        assert self.homophones is not None
        if span in self.homophones:
            return frozenset(self.homophones[span])
        if self.table is not None and span in self.table:
            return frozenset()
        row = self.mask[self.spans.id_or_unknown(span)]
        return frozenset(int(index) for index in row.nonzero(as_tuple=True)[0].tolist())

    def _readings(self, index: int) -> tuple[str, ...]:
        """The syllables the character at emission *index* reads as."""
        assert self.homophones is not None
        if self._char_readings is None:
            readings: dict[int, list[str]] = {}
            for syllable, ids in self.homophones.items():
                for i in ids:
                    readings.setdefault(i, []).append(syllable)
            self._char_readings = {i: tuple(syllables) for i, syllables in readings.items()}
        return self._char_readings.get(index, ())

    def admits(self, span: str, emission_id: int) -> bool:
        """Whether *span* may stand for the character at *emission_id*."""
        return emission_id in self._lookup(span)

    def _lookup(self, span: str) -> frozenset[int]:
        """The membership set behind :meth:`resolve`, cached with it."""
        self.resolve(span)
        return self._cache[span][0]


def _canonical_prior(cost: float) -> str:
    """A prior's digest field, byte-identical to Rust's ``format!("{:.9e}")``.

    Rust's ``{:.9e}`` writes the exponent bare -- no ``+``, no leading zeros --
    while Python's ``{:.9e}`` pads and signs it; parsing it back to an ``int``
    lands both on the same string. ``inf`` stands for the unreachable price
    JSON cannot hold.
    """
    if math.isinf(cost):
        return "inf"
    mantissa, exponent = f"{cost:.9e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _canonical_pairs(pairs: list[tuple[int, float]]) -> str:
    """The canonical serialisation both sides hash: ``id:prior`` joined by ``,``."""
    return ",".join(f"{index}:{_canonical_prior(cost)}" for index, cost in pairs)


#: How many ``(id, prior)`` pairs each fixture row keeps verbatim: enough to
#: point a digest mismatch at a real difference, small enough to keep the file.
FIXTURE_HEAD = 16


def write_candidates_fixture(space: CandidateSpace, out: Path) -> int:
    """Write the shared candidate-union fixture and return the span count.

    The per-span twin of ``mlime.typo.write_corrections_fixture``: each fixture
    span's :meth:`CandidateSpace.priors`, pinned compactly -- the pair count, a
    SHA-256 of the canonical serialisation (``id:prior`` pairs ascending,
    each prior ``{:.9e}``, ``inf`` for the unreachable), and the first
    ``FIXTURE_HEAD`` pairs verbatim so a digest mismatch can still point at a
    real difference. When the space's vocabulary covers the whole lexicon the
    emission index is the character table's own order -- ``char_pinyin.tsv``
    is sorted, so those ids are the ``CharId``s the Rust side numbers by --
    and the file pins both languages to the same rows.
    """
    if space.table is None:
        raise ValueError("the candidates fixture resolves under the typo space")
    rows = {}
    for span in corrections_fixture_spans(space.table):
        pairs = list(space.priors(span).items())
        rows[span] = {
            "n": len(pairs),
            "sha256": hashlib.sha256(_canonical_pairs(pairs).encode()).hexdigest(),
            # A character the noise model cannot reach within two edits
            # prices at infinity -- it stands in the union but no correction
            # admits it. JSON has no infinity, so the pair's cost is null.
            "head": [
                [index, cost if math.isfinite(cost) else None]
                for index, cost in pairs[:FIXTURE_HEAD]
            ],
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    # One span a line -- the file stays diffable where pretty-printing the
    # pairs would put a union on a million lines.
    lines = (
        json.dumps(span, ensure_ascii=False)
        + ": "
        + json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        for span, row in sorted(rows.items())
    )
    out.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    return len(rows)


def write_emittable(path: Path, lexicon: Lexicon) -> int:
    """Write the characters the model can emit, one per line, sorted.

    The decoder needs this set before the model exists: it decides which
    candidates the lattice asks about, and a lattice listing all 41,923 lexicon
    characters would be four fifths padding. Only this side knows the answer --
    it is the character table intersected with the base tokenizer's vocabulary --
    so it is exported as an artefact of the run rather than recomputed in Rust
    from a vocabulary file Rust would then have to parse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{character}\n" for character in lexicon.characters), encoding="utf-8")
    log.info("emittable set written", path=str(path), characters=lexicon.size)
    return lexicon.size


def build_lexicon(
    readings: Mapping[str, tuple[str, ...]],
    vocabulary: Mapping[str, int],
    spans: SpanVocab,
) -> Lexicon:
    """Intersect the lexicon with a tokenizer's vocabulary and build the span masks.

    *vocabulary* is the base model's ``token -> id`` map. Every character that
    appears in it as a whole token and has at least one typeable reading becomes
    emittable; the mask marks a ``(span, character)`` pair when one of that
    character's readings starts with the span.
    """
    characters = tuple(sorted(character for character in readings if character in vocabulary))
    if not characters:
        raise ValueError("the lexicon and the tokenizer vocabulary do not intersect")
    token_ids = torch.tensor([vocabulary[character] for character in characters], dtype=torch.long)
    mask = torch.zeros((len(spans), len(characters)), dtype=torch.bool)
    homophones: dict[str, list[int]] = {}
    for index, character in enumerate(characters):
        for reading in readings[character]:
            homophones.setdefault(reading, []).append(index)
            for length in range(1, len(reading) + 1):
                prefix = reading[:length]
                if prefix in spans:
                    mask[spans.id(prefix), index] = True
    admitted = mask.any(dim=1)
    empty = int((~admitted).sum())
    if empty == len(spans):
        raise ValueError("no span admits any character; the readings and spans disagree")
    log.info(
        "lexicon built",
        lexicon=len(readings),
        vocabulary=len(vocabulary),
        emittable=len(characters),
        spans=len(spans),
        spans_admitting_nothing=empty,
    )
    return Lexicon(
        characters=characters,
        token_ids=token_ids,
        candidate_mask=mask,
        homophones={reading: tuple(ids) for reading, ids in homophones.items()},
    )
