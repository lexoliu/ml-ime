"""The loop, end to end on a tiny model: the schedule, the metrics, the checkpoint.

This is the one test that runs the whole path -- shards on disk, stream, collator,
forward, backward, checkpoint -- because the pieces are individually correct in
ways that still do not compose. It is a plumbing test, not an accuracy test: a
four-layer model on eight characters proves that the loss can move, not that
route A works.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch
from transformers import BertConfig

from mlime.data.corpus import SAMPLE_SCHEMA, Sample
from mlime.train.labels import LABEL_SCHEMA
from mlime.train.lexicon import Lexicon
from mlime.train.loop import (
    Accuracy,
    Distributed,
    MetricLog,
    TrainingConfig,
    cosine_with_warmup,
    evaluate,
    train,
)
from mlime.train.model import RouteAConfig, RouteAModel
from mlime.train.samples import BaseTokenizer, Collator, CorpusStream, SampleBuilder
from mlime.train.spans import SpanVocab

TINY = BertConfig(
    vocab_size=256,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
)

#: Four sentences whose readings exercise a homophone choice and a polyphone.
CORPUS = [
    ("我爱北京", ["wo3", "ai4", "bei3", "jing1"], "北京"),
    ("中重我绿", ["zhong1", "chong2", "wo3", "lv4"], None),
    ("钟爱北京", ["zhong1", "ai4", "bei3", "jing1"], "钟"),
    ("我爱绿钟", ["wo3", "ai4", "lv4", "zhong1"], "绿"),
]


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


def test_the_warmup_ramps_then_the_cosine_decays() -> None:
    assert cosine_with_warmup(0, 4, 100) == pytest.approx(0.25)
    assert cosine_with_warmup(3, 4, 100) == pytest.approx(1.0)
    assert cosine_with_warmup(4, 4, 100) == pytest.approx(1.0)
    assert cosine_with_warmup(52, 4, 100) == pytest.approx(0.5, abs=0.02)
    assert cosine_with_warmup(100, 4, 100) == pytest.approx(0.0, abs=1e-6)


def test_the_warmup_is_four_percent_of_the_run() -> None:
    assert TrainingConfig(max_steps=1000).warmup_steps == 40


def test_a_metric_log_is_readable_as_it_is_written(tmp_path: Path) -> None:
    with MetricLog(tmp_path / "metrics.jsonl") as metrics:
        metrics.write(event="step", step=1, loss=2.5)
        assert json.loads((tmp_path / "metrics.jsonl").read_text())["loss"] == 2.5


def test_a_single_process_run_is_a_world_of_one() -> None:
    world = Distributed.from_environment()
    assert world.world_size == 1
    assert world.is_main


def test_the_loss_falls_and_the_run_leaves_its_evidence(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    samples_dir, labels_dir = corpus
    torch.manual_seed(0)
    model = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    stream = CorpusStream(samples_dir, labels_dir, SampleBuilder(lexicon, spans, seed=1))
    config = TrainingConfig(
        max_steps=40,
        base_lr=1e-3,
        new_lr=3e-3,
        token_budget=64,
        log_every=1,
        checkpoint_every=20,
        fp16=False,
        seed=3,
    )
    out_dir = tmp_path / "run"
    metrics_path = train(model, stream, Collator(tokenizer), config, out_dir)

    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    steps = [record for record in records if record["event"] == "step"]
    assert records[0]["event"] == "config"
    assert len(steps) == config.max_steps
    assert steps[-1]["loss"] < steps[0]["loss"]
    assert (out_dir / "checkpoint-000020.pt").is_file()
    assert (out_dir / "checkpoint-final.pt").is_file()
    saved = torch.load(out_dir / "checkpoint-final.pt", weights_only=False)
    assert saved["step"] == config.max_steps
    assert saved["training"]["base_lr"] == 1e-3


def test_the_schedule_reaches_both_learning_rates(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    samples_dir, labels_dir = corpus
    model = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    stream = CorpusStream(samples_dir, labels_dir, SampleBuilder(lexicon, spans))
    config = TrainingConfig(max_steps=25, token_budget=64, log_every=1, fp16=False)
    metrics_path = train(model, stream, Collator(tokenizer), config, tmp_path / "run")
    steps = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if json.loads(line)["event"] == "step"
    ]
    peak = max(step["lr"] for step in steps)
    assert peak == pytest.approx(config.base_lr, rel=1e-6)
    assert max(step["new_lr"] for step in steps) == pytest.approx(config.new_lr, rel=1e-6)
    assert steps[-1]["lr"] < peak


def test_accuracy_is_reported_with_and_without_context(
    corpus: tuple[Path, Path], lexicon: Lexicon, spans: SpanVocab, tokenizer: BaseTokenizer
) -> None:
    samples_dir, labels_dir = corpus
    model = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    stream = CorpusStream(samples_dir, labels_dir, SampleBuilder(lexicon, spans))
    examples = list(stream)
    assert examples
    for with_context in (True, False):
        accuracy = evaluate(
            model, examples, tokenizer, torch.device("cpu"), 512, with_context=with_context
        )
        assert accuracy.scored == sum(len(example) for example in examples)
        assert 0.0 <= accuracy.rate <= 1.0


def test_an_empty_accuracy_is_zero_not_an_error() -> None:
    assert Accuracy(correct=0, scored=0).rate == 0.0
