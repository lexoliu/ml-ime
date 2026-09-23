"""Chinese text primitives shared by the corpus, g2p and export stages.

Everything here answers one question: what would a person actually have typed to
produce this string? That is why normalisation converts traditional characters to
simplified (Chinese Wikipedia stores whatever variant its editors used, mixed
within a single article), de-tokenises the spaces LCCC inserts between
characters, and folds full-width Latin letters and digits while leaving
full-width *punctuation* alone -- ``，``, ``！``, ``？`` and ``；`` live in the
same Unicode block as the letters but are what a Chinese keyboard emits, and the
sentence splitter needs them.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator

import opencc
import regex

#: Any Han character, in the Unicode-property sense rather than a hand-drawn range.
HAN = regex.compile(r"\p{Han}")

#: Characters that terminate a sentence, full-width and ASCII alike.
SENTENCE_DELIMITERS = "。！？；!?;"

_SENTENCE = regex.compile(rf"[^{SENTENCE_DELIMITERS}]+[{SENTENCE_DELIMITERS}]?")

#: Full-width Latin letters, digits and space folded to their ASCII form.
_FULLWIDTH_FOLD = {
    **{codepoint: codepoint - 0xFEE0 for codepoint in range(0xFF10, 0xFF1A)},
    **{codepoint: codepoint - 0xFEE0 for codepoint in range(0xFF21, 0xFF3B)},
    **{codepoint: codepoint - 0xFEE0 for codepoint in range(0xFF41, 0xFF5B)},
    0x3000: 0x20,
}

_HORIZONTAL_SPACE = regex.compile(r"[^\S\n]+")
_BLANK_LINES = regex.compile(r"\n+")
#: Punctuation that occupies a full-width cell, so no space is ever typed beside it.
CJK_PUNCTUATION = "。，、；：！？“”‘’（）《》〈〉【】「」『』—…·"

#: LCCC ships its turns word-segmented ("自贡 哪里 有 好吃 的 鱼"), and the same spaces
#: appear before its punctuation. Nobody types a space between two full-width
#: characters, so those are removed -- while "使用 Python" keeps its space, because
#: one side is not full-width.
_WIDE = rf"[\p{{Han}}{regex.escape(CJK_PUNCTUATION)}]"
_SPACE_BETWEEN_WIDE = regex.compile(rf"(?<={_WIDE}) +(?={_WIDE})")

#: Bracket pairs left empty once Wikipedia's inline templates are stripped, as in
#: ``文學（），在狭义上`` -- the language annotation is gone but the parentheses stay.
_EMPTY_BRACKETS = regex.compile(r"[（(【\[「『《]\s*[）)】\]」』》]")

_TONE_MARK = regex.compile(r"[0-9]$")


class Normalizer:
    """Canonicalises raw corpus text into the form a simplified-Chinese IME types.

    Holds the OpenCC converter, which is expensive to construct and cheap to
    reuse, so build one per run and pass it down.
    """

    def __init__(self) -> None:
        self._to_simplified = opencc.OpenCC("t2s")

    def __call__(self, raw: str) -> str:
        """Normalise *raw*, preserving newlines as sentence boundaries."""
        text = unicodedata.normalize("NFC", raw).translate(_FULLWIDTH_FOLD)
        text = _HORIZONTAL_SPACE.sub(" ", text)
        text = _BLANK_LINES.sub("\n", text)
        text = _SPACE_BETWEEN_WIDE.sub("", text)
        text = _EMPTY_BRACKETS.sub("", text)
        converted = str(self._to_simplified.convert(text))
        if len(converted) != len(text):
            raise ValueError(
                "OpenCC changed the character count, so per-character alignment is lost: "
                f"{text!r} -> {converted!r}"
            )
        return converted.strip()


def split_sentences(text: str) -> Iterator[str]:
    """Yield the sentences of *text*, each keeping its own terminal delimiter.

    Newlines separate sentences and are dropped; the delimiters are kept so that
    a run of preceding sentences reassembles into readable context.
    """
    for line in text.split("\n"):
        for match in _SENTENCE.finditer(line):
            sentence = match.group().strip()
            if sentence:
                yield sentence


def strip_terminal_delimiter(sentence: str) -> str:
    """Drop a sentence's final ``。``-like character.

    A target sentence is what the input method must emit for a run of
    keystrokes, and the terminal punctuation is typed as its own key rather than
    decoded from pinyin, so carrying it into the target would leave a character
    with no syllable behind it.
    """
    return sentence[:-1] if sentence and sentence[-1] in SENTENCE_DELIMITERS else sentence


def han_ratio(text: str) -> float:
    """Fraction of *text*'s non-whitespace characters that are Han. Empty text scores 0."""
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    return sum(1 for character in visible if HAN.match(character)) / len(visible)


def han_characters(text: str) -> list[str]:
    """The Han characters of *text*, in order, with duplicates kept."""
    return [character for character in text if HAN.match(character)]


def toneless(syllable: str) -> str:
    """Strip the tone digit off *syllable* and spell ``ü`` the way a keyboard does.

    The input method's masks are keyed on what the user types, and no pinyin
    keyboard has a ``ü`` key or a tone key, so ``lǜ4``-style output from either
    annotator has to collapse onto ``lv`` before the two can be compared.
    """
    stripped = _TONE_MARK.sub("", syllable.strip().lower())
    return stripped.replace("ü", "v").replace("u:", "v")
