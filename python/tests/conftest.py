"""Fixtures shared by the training tests.

The training tests need a lexicon and a tokenizer, and the real ones are a 42k
character table and a 400MB download. Both are replaced here by the smallest
thing that keeps the properties under test real: a handful of characters whose
readings include a homophone pair, a zh-initial, a bare vowel and a ü syllable,
and a tokenizer that is nothing but the four sentinel ids and a deterministic
encoder. The *span* inventory is the real one -- it ships in the package and is
what the model is indexed by.

The two miniature corpora live here as well, because the loop, the resume and
the batch count all have to read the same one: a count that agreed with the loop
on a corpus of its own would prove nothing about the loop.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch

from mlime.data.corpus import SAMPLE_SCHEMA, Sample
from mlime.train.arbitration import ReadingArbitration
from mlime.train.labels import LABEL_SCHEMA
from mlime.train.lexicon import Lexicon, build_lexicon
from mlime.train.spans import SpanVocab

#: Four sentences whose readings exercise a homophone choice and a polyphone.
CORPUS = [
    ("我爱北京", ["wo3", "ai4", "bei3", "jing1"], "北京"),
    ("中重我绿", ["zhong1", "chong2", "wo3", "lv4"], None),
    ("钟爱北京", ["zhong1", "ai4", "bei3", "jing1"], "钟"),
    ("我爱绿钟", ["wo3", "ai4", "lv4", "zhong1"], "绿"),
]

#: 中/钟 are homophones, 重 is a polyphone, 绿 carries the ü spelling, 和 carries
#: the readings the mainland table really lists for it -- so the arbitration of
#: its Taiwan label is tested against the real disagreement -- and the rest spell
#: 我爱北京 so that a test sentence reads like one.
READINGS = {
    "中": ("zhong",),
    "钟": ("zhong",),
    "重": ("zhong", "chong"),
    "我": ("wo",),
    "爱": ("ai",),
    "绿": ("lv", "lu"),
    "北": ("bei",),
    "京": ("jing",),
    "和": ("he", "hu", "huo"),
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


@pytest.fixture(name="arbitration")
def arbitration_fixture(spans: SpanVocab) -> ReadingArbitration:
    """The real table of decided readings that ships in the package."""
    return ReadingArbitration.load(spans)


@pytest.fixture(name="tokenizer")
def tokenizer_fixture() -> StubTokenizer:
    """A stand-in for the base model's tokenizer."""
    return StubTokenizer()


@pytest.fixture(name="corpus")
def corpus_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A two-shard corpus with its labels, repeated enough to train on."""
    samples_dir, labels_dir = tmp_path / "samples", tmp_path / "labels"
    samples_dir.mkdir()
    labels_dir.mkdir()
    for shard in range(2):
        rows, labels = [], []
        for index in range(64):
            text, readings, context = CORPUS[index % len(CORPUS)]
            sample = Sample(id=f"{shard}-{index}", source="test", text=text, context=context)
            rows.append(sample.row())
            labels.append({"id": sample.id, "syllables": readings, "refusal": None})
        name = f"test-{shard:05d}.parquet"
        pl.DataFrame(rows, schema=SAMPLE_SCHEMA).write_parquet(samples_dir / name)
        pl.DataFrame(labels, schema=LABEL_SCHEMA).write_parquet(labels_dir / name)
    return samples_dir, labels_dir


@pytest.fixture(name="short_corpus")
def short_corpus_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """One shard of eight sentences, so six steps cross an epoch boundary.

    A resume that only ever restarts mid-epoch never exercises the epoch in the
    position it saved, and the epoch is what re-augments the corpus: land on the
    wrong one and every example after the resume is typed differently.
    """
    samples_dir, labels_dir = tmp_path / "short-samples", tmp_path / "short-labels"
    samples_dir.mkdir()
    labels_dir.mkdir()
    rows, labels = [], []
    for index in range(8):
        text, readings, context = CORPUS[index % len(CORPUS)]
        sample = Sample(id=f"0-{index}", source="test", text=text, context=context)
        rows.append(sample.row())
        labels.append({"id": sample.id, "syllables": readings, "refusal": None})
    name = "test-00000.parquet"
    pl.DataFrame(rows, schema=SAMPLE_SCHEMA).write_parquet(samples_dir / name)
    pl.DataFrame(labels, schema=LABEL_SCHEMA).write_parquet(labels_dir / name)
    return samples_dir, labels_dir
