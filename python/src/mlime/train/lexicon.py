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

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from mlime.logging import log
from mlime.train.spans import SpanVocab


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
    for index, character in enumerate(characters):
        for reading in readings[character]:
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
    return Lexicon(characters=characters, token_ids=token_ids, candidate_mask=mask)
