"""The character language model's alphabet: the reserved ids, then the lexicon."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlime.train.lexicon import read_char_readings

#: The reserved ids, in order; every character of the lexicon follows them.
SPECIALS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<sep>", "<unk>")
PAD, BOS, EOS, SEP, UNK = range(len(SPECIALS))


@dataclass(frozen=True)
class CharVocab:
    """The model's alphabet: the specials, then the lexicon's characters sorted."""

    chars: tuple[str, ...]

    @classmethod
    def from_char_table(cls, char_table: Path) -> CharVocab:
        """The alphabet of ``char_pinyin.tsv``, in code point order."""
        return cls(chars=SPECIALS + tuple(sorted(read_char_readings(char_table))))

    def __post_init__(self) -> None:
        if self.chars[: len(SPECIALS)] != SPECIALS:
            raise ValueError("a vocabulary must begin with the reserved ids")

    def __len__(self) -> int:
        return len(self.chars)

    @property
    def index(self) -> dict[str, int]:
        """``{character: id}``; built on demand, cached by the caller that loops."""
        return {ch: i for i, ch in enumerate(self.chars)}

    def encode(self, text: str, index: dict[str, int]) -> list[int]:
        """Ids of *text*'s characters, ``<unk>`` for any outside the alphabet."""
        return [index.get(ch, UNK) for ch in text]
