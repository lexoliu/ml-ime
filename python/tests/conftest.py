"""Fixtures shared by the training tests.

The training tests need a lexicon and a tokenizer, and the real ones are a 42k
character table and a 400MB download. Both are replaced here by the smallest
thing that keeps the properties under test real: a handful of characters whose
readings include a homophone pair, a zh-initial, a bare vowel and a ü syllable,
and a tokenizer that is nothing but the four sentinel ids and a deterministic
encoder. The *span* inventory is the real one -- it ships in the package and is
what the model is indexed by.
"""

from __future__ import annotations

import pytest
import torch

from mlime.train.lexicon import Lexicon, build_lexicon
from mlime.train.spans import SpanVocab

#: 中/钟 are homophones, 重 is a polyphone, 绿 carries the ü spelling, and the
#: rest spell 我爱北京 so that a test sentence reads like one.
READINGS = {
    "中": ("zhong",),
    "钟": ("zhong",),
    "重": ("zhong", "chong"),
    "我": ("wo",),
    "爱": ("ai",),
    "绿": ("lv", "lu"),
    "北": ("bei",),
    "京": ("jing",),
}


class StubTokenizer:
    """The four sentinel ids and a deterministic encoder, so the collator is testable."""

    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0
    mask_token_id = 3

    def __call__(
        self,
        text: list[str],
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Encode each string as ``[CLS] <one id per character> [SEP]``, padded."""
        encoded = [
            [self.cls_token_id, *(ord(character) % 90 + 10 for character in item[:max_length])]
            for item in text
        ]
        width = max(len(row) + 1 for row in encoded)
        ids = torch.zeros((len(encoded), width), dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, tokens in enumerate(encoded):
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            ids[row, len(tokens)] = self.sep_token_id
            mask[row, : len(tokens) + 1] = 1
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture(name="spans")
def spans_fixture() -> SpanVocab:
    """The real typed-span inventory that ships in the package."""
    return SpanVocab.load()


@pytest.fixture(name="lexicon")
def lexicon_fixture(spans: SpanVocab) -> Lexicon:
    """A lexicon over :data:`READINGS`, as if the base vocabulary held just those."""
    vocabulary = {character: index + 100 for index, character in enumerate(sorted(READINGS))}
    return build_lexicon(READINGS, vocabulary, spans)


@pytest.fixture(name="tokenizer")
def tokenizer_fixture() -> StubTokenizer:
    """A stand-in for the base model's tokenizer."""
    return StubTokenizer()
