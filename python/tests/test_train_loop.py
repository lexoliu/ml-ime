"""The loop, end to end on a tiny model: the schedule, the metrics, the checkpoint.

This is the one test that runs the whole path -- shards on disk, stream, collator,
forward, backward, checkpoint -- because the pieces are individually correct in
ways that still do not compose. It is a plumbing test, not an accuracy test: a
four-layer model on eight characters proves that the loss can move, not that
route A works.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import BertConfig

from mlime.train.lexicon import Lexicon
from mlime.train.loop import (
    Accuracy,
    Distributed,
    EpochBatches,
    MetricLog,
    TrainingConfig,
    cosine_with_warmup,
    evaluate,
    train,
)
from mlime.train.model import RouteAConfig, RouteAModel
from mlime.train.samples import BaseTokenizer, Batch, Collator, CorpusStream, SampleBuilder
from mlime.train.spans import SpanVocab

TINY = BertConfig(
    vocab_size=256,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
)


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
    metrics_path = train(model, stream, Collator(tokenizer), config, out_dir).metrics

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
    metrics_path = train(model, stream, Collator(tokenizer), config, tmp_path / "run").metrics
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


#: Six steps over :func:`short_corpus_fixture`: four batches an epoch, so the
#: checkpoint at step 3 is mid-epoch and the steps after it cross into the next.
RESUMABLE = TrainingConfig(
    max_steps=6,
    base_lr=1e-3,
    new_lr=3e-3,
    token_budget=24,
    log_every=1,
    checkpoint_every=3,
    fp16=False,
    seed=3,
)


def tiny_model(lexicon: Lexicon) -> RouteAModel:
    """The same randomly initialised model every time it is called."""
    torch.manual_seed(0)
    return RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))


def stream_and_collator(
    corpus: tuple[Path, Path], lexicon: Lexicon, spans: SpanVocab, tokenizer: BaseTokenizer
) -> tuple[CorpusStream, Collator]:
    """A reader and a collator seeded the way every segment of one run seeds them."""
    samples_dir, labels_dir = corpus
    return (
        CorpusStream(samples_dir, labels_dir, SampleBuilder(lexicon, spans, seed=1)),
        Collator(tokenizer),
    )


def records(metrics: Path, event: str) -> list[dict[str, object]]:
    """Every *event* record in a metrics file, in order."""
    written = [json.loads(line) for line in metrics.read_text().splitlines()]
    return [record for record in written if record["event"] == event]


def step_losses(metrics: Path) -> list[float]:
    """The loss of every logged step, in order."""
    return [float(record["loss"]) for record in records(metrics, "step")]


def typed(batch: Batch, spans: SpanVocab) -> list[str]:
    """The spans the batch says were pressed, as the strings they spell."""
    return [spans.spelling(int(span)) for span in batch.span_ids[batch.span_positions]]


def test_a_resumed_run_is_the_run_that_was_not_interrupted(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    whole_dir = tmp_path / "whole"
    uninterrupted = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    whole = train(uninterrupted, stream, collator, RESUMABLE, whole_dir).metrics
    assert len(step_losses(whole)) == RESUMABLE.max_steps

    resumed_model = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    resumed = train(
        resumed_model,
        stream,
        collator,
        RESUMABLE,
        tmp_path / "resumed",
        resume=whole_dir / "checkpoint-000003.pt",
    ).metrics

    # Exactly the losses the uninterrupted run recorded for those steps: the
    # batches, the augmentation, the dropout, the schedule and the optimiser
    # state all have to have carried over for these to be the same numbers.
    assert step_losses(resumed) == step_losses(whole)[3:]
    assert records(resumed, "resume") == [
        {
            "event": "resume",
            "checkpoint": str(whole_dir / "checkpoint-000003.pt"),
            "step": 3,
            "epoch": 0,
            "index": 3,
        }
    ]
    finished = torch.load(whole_dir / "checkpoint-final.pt", weights_only=False)
    assert finished["positions"][0]["epoch"] == 1
    weights = uninterrupted.state_dict()
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, weights[name]), name


def test_skipping_lands_where_reading_would_have(
    short_corpus: tuple[Path, Path], lexicon: Lexicon, spans: SpanVocab, tokenizer: BaseTokenizer
) -> None:
    read = EpochBatches(*stream_and_collator(short_corpus, lexicon, spans, tokenizer), 24)
    for _ in range(5):
        next(read)
    epoch, index = read.epoch, read.index
    assert (epoch, index) == (1, 1)  # the fifth batch is into the second epoch
    wanted = next(read)

    skipped = EpochBatches(*stream_and_collator(short_corpus, lexicon, spans, tokenizer), 24)
    skipped.skip(epoch, index)
    assert (skipped.epoch, skipped.index) == (epoch, index)
    landed = next(skipped)
    assert landed.ids == wanted.ids
    assert typed(landed, spans) == typed(wanted, spans)
    assert (skipped.epoch, skipped.index) == (read.epoch, read.index)


def test_a_resume_under_a_different_schedule_is_refused(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "whole"
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    train(tiny_model(lexicon), stream, collator, RESUMABLE, out_dir)

    elsewhere = replace(RESUMABLE, base_lr=RESUMABLE.base_lr * 2, weight_decay=0.5)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    with pytest.raises(ValueError, match=r"base_lr .* weight_decay "):
        train(
            tiny_model(lexicon),
            stream,
            collator,
            elsewhere,
            tmp_path / "elsewhere",
            resume=out_dir / "checkpoint-000003.pt",
        )


def test_resuming_a_run_that_is_already_finished_is_refused(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "whole"
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    train(tiny_model(lexicon), stream, collator, RESUMABLE, out_dir)

    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    with pytest.raises(ValueError, match="already at step 6 of 6"):
        train(
            tiny_model(lexicon),
            stream,
            collator,
            RESUMABLE,
            tmp_path / "again",
            resume=out_dir / "checkpoint-final.pt",
        )


def test_only_the_newest_checkpoints_are_kept(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "rotated"
    config = replace(RESUMABLE, checkpoint_every=1, keep_checkpoints=2)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    train(tiny_model(lexicon), stream, collator, config, out_dir)

    numbered = sorted(path.name for path in out_dir.glob("checkpoint-[0-9]*.pt"))
    assert numbered == ["checkpoint-000005.pt", "checkpoint-000006.pt"]
    assert (out_dir / "checkpoint-final.pt").is_file()


def test_a_run_that_keeps_no_checkpoint_is_refused() -> None:
    with pytest.raises(ValueError, match="keep_checkpoints"):
        TrainingConfig(max_steps=1, keep_checkpoints=0)


#: A budget so small the first step always overruns it, which is what a kernel
#: that is about to be killed looks like from inside the loop.
SPENT = replace(RESUMABLE, wall_budget_seconds=1e-6)


def test_a_segment_out_of_clock_pauses_and_the_next_one_finishes_the_run(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    whole_dir = tmp_path / "whole"
    uninterrupted = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    whole = train(uninterrupted, stream, collator, RESUMABLE, whole_dir)
    assert whole.finished and whole.step == RESUMABLE.max_steps

    paused_dir = tmp_path / "paused"
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    paused = train(tiny_model(lexicon), stream, collator, SPENT, paused_dir)
    assert (paused.step, paused.finished) == (1, False)
    assert (paused_dir / "checkpoint-paused.pt").is_file()
    # No final checkpoint: that file means the run reached max_steps, and a
    # kernel that ran out of clock did not.
    assert not (paused_dir / "checkpoint-final.pt").exists()
    assert [record["step"] for record in records(paused.metrics, "paused")] == [1]

    # The budget belongs to the kernel, not to the run, so the segment that
    # finishes it is allowed to have no budget at all.
    resumed_model = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    resumed = train(
        resumed_model,
        stream,
        collator,
        RESUMABLE,
        tmp_path / "resumed",
        resume=paused_dir / "checkpoint-paused.pt",
    )
    assert resumed.finished and resumed.step == RESUMABLE.max_steps
    assert step_losses(resumed.metrics) == step_losses(whole.metrics)[1:]
    weights = uninterrupted.state_dict()
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, weights[name]), name


def test_rotation_leaves_the_paused_checkpoint_alone(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "chained"
    every_step = replace(RESUMABLE, checkpoint_every=1, keep_checkpoints=1)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    spent = replace(every_step, wall_budget_seconds=SPENT.wall_budget_seconds)
    train(tiny_model(lexicon), stream, collator, spent, out_dir)
    assert (out_dir / "checkpoint-paused.pt").is_file()

    # The next segment writes into the same directory and rotates hard, and the
    # file it is resuming from is the one thing it must not delete.
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    train(
        tiny_model(lexicon),
        stream,
        collator,
        every_step,
        out_dir,
        resume=out_dir / "checkpoint-paused.pt",
    )
    assert (out_dir / "checkpoint-paused.pt").is_file()
    assert sorted(path.name for path in out_dir.glob("checkpoint-[0-9]*.pt")) == [
        "checkpoint-000006.pt"
    ]
    assert (out_dir / "checkpoint-final.pt").is_file()


def test_a_budget_may_change_between_segments_but_nothing_else_may(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "paused"
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    train(tiny_model(lexicon), stream, collator, SPENT, out_dir)

    longer = replace(SPENT, wall_budget_seconds=3600.0, new_lr=SPENT.new_lr / 2)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, tokenizer)
    with pytest.raises(ValueError, match="new_lr") as refusal:
        train(
            tiny_model(lexicon),
            stream,
            collator,
            longer,
            tmp_path / "elsewhere",
            resume=out_dir / "checkpoint-paused.pt",
        )
    assert "wall_budget_seconds" not in str(refusal.value)


def test_a_wall_budget_that_is_not_time_is_refused() -> None:
    with pytest.raises(ValueError, match="wall_budget_seconds must be positive"):
        TrainingConfig(max_steps=1, wall_budget_seconds=0.0)
