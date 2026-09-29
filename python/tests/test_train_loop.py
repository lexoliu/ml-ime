"""The loop, end to end on a tiny model: the schedule, the metrics, the checkpoint.

This is the one test that runs the whole path -- shards on disk, stream, collator,
forward, backward, checkpoint -- because the pieces are individually correct in
ways that still do not compose. It is a plumbing test, not an accuracy test: a
four-layer model on eight characters proves that the loss can move, not that
route A works.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import BertConfig

from mlime.train.arbitration import ReadingArbitration
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.loop import (
    PHASES,
    Accuracy,
    Distributed,
    EpochBatches,
    MetricLog,
    Prefetcher,
    TrainingConfig,
    cosine_with_warmup,
    evaluate,
    train,
)
from mlime.train.model import RouteAConfig, RouteAModel
from mlime.train.samples import (
    IGNORE_INDEX,
    BaseTokenizer,
    Batch,
    Collator,
    CorpusStream,
    SampleBuilder,
)
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
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    samples_dir, labels_dir = corpus
    torch.manual_seed(0)
    model = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    stream = CorpusStream(
        samples_dir, labels_dir, SampleBuilder(lexicon, spans, arbitration, seed=1)
    )
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
    collator = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
    metrics_path = train(model, [(stream, collator)], config, out_dir).metrics

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
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    samples_dir, labels_dir = corpus
    model = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    stream = CorpusStream(samples_dir, labels_dir, SampleBuilder(lexicon, spans, arbitration))
    config = TrainingConfig(max_steps=25, token_budget=64, log_every=1, fp16=False)
    collator = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
    metrics_path = train(model, [(stream, collator)], config, tmp_path / "run").metrics
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
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
) -> None:
    samples_dir, labels_dir = corpus
    model = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    stream = CorpusStream(samples_dir, labels_dir, SampleBuilder(lexicon, spans, arbitration))
    examples = list(stream)
    assert examples
    for with_context in (True, False):
        accuracy = evaluate(
            model,
            examples,
            tokenizer,
            torch.device("cpu"),
            512,
            with_context=with_context,
            candidates=CandidateSpace(spans, lexicon.candidate_mask),
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
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
) -> tuple[CorpusStream, Collator]:
    """A reader and a collator seeded the way every segment of one run seeds them."""
    samples_dir, labels_dir = corpus
    return (
        CorpusStream(samples_dir, labels_dir, SampleBuilder(lexicon, spans, arbitration, seed=1)),
        Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask)),
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
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    whole_dir = tmp_path / "whole"
    uninterrupted = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    whole = train(uninterrupted, [(stream, collator)], RESUMABLE, whole_dir).metrics
    assert len(step_losses(whole)) == RESUMABLE.max_steps

    resumed_model = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    resumed = train(
        resumed_model,
        [(stream, collator)],
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
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
) -> None:
    read = EpochBatches(
        *stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer), 24
    )
    for _ in range(5):
        next(read)
    epoch, index = read.epoch, read.index
    assert (epoch, index) == (1, 1)  # the fifth batch is into the second epoch
    wanted = next(read)

    skipped = EpochBatches(
        *stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer), 24
    )
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
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "whole"
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    train(tiny_model(lexicon), [(stream, collator)], RESUMABLE, out_dir)

    elsewhere = replace(RESUMABLE, base_lr=RESUMABLE.base_lr * 2, weight_decay=0.5)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    with pytest.raises(ValueError, match=r"base_lr .* weight_decay "):
        train(
            tiny_model(lexicon),
            [(stream, collator)],
            elsewhere,
            tmp_path / "elsewhere",
            resume=out_dir / "checkpoint-000003.pt",
        )


def test_resuming_a_run_that_is_already_finished_is_refused(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "whole"
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    train(tiny_model(lexicon), [(stream, collator)], RESUMABLE, out_dir)

    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    with pytest.raises(ValueError, match="already at step 6 of 6"):
        train(
            tiny_model(lexicon),
            [(stream, collator)],
            RESUMABLE,
            tmp_path / "again",
            resume=out_dir / "checkpoint-final.pt",
        )


def test_only_the_newest_checkpoints_are_kept(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "rotated"
    config = replace(RESUMABLE, checkpoint_every=1, keep_checkpoints=2)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    train(tiny_model(lexicon), [(stream, collator)], config, out_dir)

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
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    whole_dir = tmp_path / "whole"
    uninterrupted = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    whole = train(uninterrupted, [(stream, collator)], RESUMABLE, whole_dir)
    assert whole.finished and whole.step == RESUMABLE.max_steps

    paused_dir = tmp_path / "paused"
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    paused = train(tiny_model(lexicon), [(stream, collator)], SPENT, paused_dir)
    assert (paused.step, paused.finished) == (1, False)
    assert (paused_dir / "checkpoint-paused.pt").is_file()
    # No final checkpoint: that file means the run reached max_steps, and a
    # kernel that ran out of clock did not.
    assert not (paused_dir / "checkpoint-final.pt").exists()
    assert [record["step"] for record in records(paused.metrics, "paused")] == [1]

    # The budget belongs to the kernel, not to the run, so the segment that
    # finishes it is allowed to have no budget at all.
    resumed_model = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    resumed = train(
        resumed_model,
        [(stream, collator)],
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
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "chained"
    every_step = replace(RESUMABLE, checkpoint_every=1, keep_checkpoints=1)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    spent = replace(every_step, wall_budget_seconds=SPENT.wall_budget_seconds)
    train(tiny_model(lexicon), [(stream, collator)], spent, out_dir)
    assert (out_dir / "checkpoint-paused.pt").is_file()

    # The next segment writes into the same directory and rotates hard, and the
    # file it is resuming from is the one thing it must not delete.
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    train(
        tiny_model(lexicon),
        [(stream, collator)],
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
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "paused"
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    train(tiny_model(lexicon), [(stream, collator)], SPENT, out_dir)

    longer = replace(SPENT, wall_budget_seconds=3600.0, new_lr=SPENT.new_lr / 2)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    with pytest.raises(ValueError, match="new_lr") as refusal:
        train(
            tiny_model(lexicon),
            [(stream, collator)],
            longer,
            tmp_path / "elsewhere",
            resume=out_dir / "checkpoint-paused.pt",
        )
    assert "wall_budget_seconds" not in str(refusal.value)


def test_a_wall_budget_that_is_not_time_is_refused() -> None:
    with pytest.raises(ValueError, match="wall_budget_seconds must be positive"):
        TrainingConfig(max_steps=1, wall_budget_seconds=0.0)


def test_a_produced_item_carries_where_to_reopen_the_stream(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
) -> None:
    """Each produced batch holds the position *after* it, so a checkpoint taken
    on the last consumed item -- not the last one a prefetcher happened to
    build -- resumes with exactly the batch that would have come next."""
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    source = Prefetcher(EpochBatches(stream, collator, RESUMABLE.token_budget))
    try:
        items = [next(source) for _ in range(3)]
    finally:
        source.close()
    assert [(item.epoch, item.index) for item in items] == [(0, 1), (0, 2), (0, 3)]

    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    replay = EpochBatches(stream, collator, RESUMABLE.token_budget)
    replay.skip(items[0].epoch, items[0].index)
    replay.collator.rng.setstate(items[0].collator_rng)
    continued = replay.produce()
    assert continued.batch.ids == items[1].batch.ids
    assert typed(continued.batch, spans) == typed(items[1].batch, spans)


def test_a_prefetched_segment_pauses_and_resumes_on_the_consumed_batch(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """With the producer thread running ahead, the paused checkpoint still
    records the batch the optimiser last consumed, and the resumed segment
    continues the uninterrupted run's sequence."""
    whole_dir = tmp_path / "whole"
    uninterrupted = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    whole = train(uninterrupted, [(stream, collator)], RESUMABLE, whole_dir)

    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    paused = train(
        tiny_model(lexicon),
        [(stream, collator)],
        replace(SPENT, prefetch=True),
        tmp_path / "paused",
    )
    assert (paused.step, paused.finished) == (1, False)

    resumed_model = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    resumed = train(
        resumed_model,
        [(stream, collator)],
        replace(RESUMABLE, prefetch=True),
        tmp_path / "resumed",
        resume=tmp_path / "paused" / "checkpoint-paused.pt",
    )
    # One consumed batch was recorded although the producer had run ahead.
    assert records(resumed.metrics, "resume")[0]["index"] == 1
    assert step_losses(resumed.metrics) == step_losses(whole.metrics)[1:]
    weights = uninterrupted.state_dict()
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, weights[name]), name


class UnevenTargets:
    """The real collator, blanking one example's targets in every other batch.

    The fixture rows all carry four scored positions, so two consecutive
    micro-batches would always hold the same count and could not tell a
    position-weighted mean from a plain one; clearing a row makes the counts
    differ while the stream, the grouping and the loss stay real.
    """

    def __init__(self, inner: Collator):
        self.inner = inner
        self.rng = inner.rng
        self._calls = 0

    @property
    def max_context_tokens(self) -> int:
        return self.inner.max_context_tokens

    def __call__(self, group: list) -> Batch:
        batch = self.inner(group)
        self._calls += 1
        if self._calls % 2 == 0:
            batch.targets[0, :] = IGNORE_INDEX
            # The collator ships `scored` over the targets it made; a wrapper
            # that blanks them afterwards must rebuild it.
            batch = replace(
                batch,
                scored=(batch.targets != IGNORE_INDEX).reshape(-1).nonzero(as_tuple=True)[0],
            )
        return batch


def test_an_accumulated_step_is_the_mean_over_every_scored_position(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """A step of two micro-batches weights each one's loss by its scored
    positions -- the mean over positions, not the mean of the two means."""
    stream, collator = stream_and_collator(corpus, lexicon, spans, arbitration, tokenizer)
    batches = EpochBatches(stream, UnevenTargets(collator), RESUMABLE.token_budget)
    items = [batches.produce() for _ in range(2)]
    counts = [int((item.batch.targets != IGNORE_INDEX).sum()) for item in items]
    # Same-width micro-batches could not tell the weighting from a plain mean.
    assert counts[0] != counts[1]

    # The loop's forwards sit right after seed_everything's manual_seed, so a
    # replica seeded the same way draws the same dropout masks.
    replica = tiny_model(lexicon)
    random.seed(RESUMABLE.seed)
    torch.manual_seed(RESUMABLE.seed)
    replica.train()
    losses = []
    with torch.no_grad():
        for item in items:
            output = replica(item.batch)
            assert output.loss is not None
            losses.append(float(output.loss))
    expected = sum(loss * count for loss, count in zip(losses, counts, strict=True)) / sum(counts)

    doubled = replace(RESUMABLE, max_steps=1, accumulate=2)
    stream, collator = stream_and_collator(corpus, lexicon, spans, arbitration, tokenizer)
    segment = train(
        tiny_model(lexicon), [(stream, UnevenTargets(collator))], doubled, tmp_path / "run"
    )
    assert step_losses(segment.metrics) == [pytest.approx(expected, rel=1e-5)]


def test_an_accumulating_run_pauses_and_resumes_in_micro_batches(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """Positions count the micro-batches consumed: a pause after one optimiser
    step of two is recorded as index 2, and the resumed segment goes on from
    the third micro-batch exactly as the uninterrupted run did."""
    doubled = replace(RESUMABLE, max_steps=3, accumulate=2)
    whole_dir = tmp_path / "whole"
    uninterrupted = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    whole = train(uninterrupted, [(stream, collator)], doubled, whole_dir)

    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    paused = train(
        tiny_model(lexicon),
        [(stream, collator)],
        replace(doubled, wall_budget_seconds=SPENT.wall_budget_seconds),
        tmp_path / "paused",
    )
    assert (paused.step, paused.finished) == (1, False)

    resumed_model = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    resumed = train(
        resumed_model,
        [(stream, collator)],
        doubled,
        tmp_path / "resumed",
        resume=tmp_path / "paused" / "checkpoint-paused.pt",
    )
    # Two micro-batches made the one step; the resume continues at the third.
    assert records(resumed.metrics, "resume")[0]["index"] == 2
    assert step_losses(resumed.metrics) == step_losses(whole.metrics)[1:]
    weights = uninterrupted.state_dict()
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, weights[name]), name


def test_a_profiled_run_writes_the_phase_table(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """A run asked to profile writes its timed steps' phases to the metrics."""
    config = replace(RESUMABLE, max_steps=12, profile_steps=2)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    segment = train(tiny_model(lexicon), [(stream, collator)], config, tmp_path / "run")
    profiled = records(segment.metrics, "profile")
    assert len(profiled) == 1
    assert profiled[0]["steps"] == 2
    assert set(profiled[0]["phases"]) == set(PHASES)
    for phase in profiled[0]["phases"].values():
        assert phase["p95"] >= phase["mean"] >= 0.0


# ---------------------------------------------------------------------------
# One process running as a world: the virtual-rank path. A Kaggle segment is
# two real ranks; a Colab segment is one process standing for both, and the
# checkpoints the two legs write must be interchangeable.
# ---------------------------------------------------------------------------


def _rank_lanes(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    first_rank: int,
    count: int,
) -> list[tuple[CorpusStream, Collator]]:
    """The ``(stream, collator)`` lanes of *count* consecutive ranks of a world of two."""
    samples_dir, labels_dir = corpus
    return [
        (
            CorpusStream(
                samples_dir,
                labels_dir,
                SampleBuilder(lexicon, spans, arbitration, seed=1),
                rank=first_rank + index,
                world_size=2,
            ),
            Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask)),
        )
        for index in range(count)
    ]


def _gloo_rank(
    rank: int,
    corpus: tuple[str, str],
    resume: str | None,
    out_dir: str,
    port: int,
    config: TrainingConfig,
) -> None:
    """One process of the two-rank gloo world the virtual-rank tests compare against.

    Runs under ``torch.multiprocessing``'s spawn, which starts a fresh
    interpreter: the fixtures are rebuilt inside it rather than pickled over.
    """
    import os
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"

    from conftest import READINGS, StubTokenizer

    from mlime.train.lexicon import build_lexicon
    from mlime.train.spans import SpanVocab

    spans = SpanVocab.load()
    vocabulary = {character: index + 100 for index, character in enumerate(sorted(READINGS))}
    lexicon = build_lexicon(READINGS, vocabulary, spans)
    arbitration = ReadingArbitration.load(spans)
    lanes = _rank_lanes(
        (Path(corpus[0]), Path(corpus[1])), lexicon, spans, arbitration, StubTokenizer(), rank, 1
    )
    train(
        tiny_model(lexicon),
        lanes,
        config,
        Path(out_dir),
        Distributed(rank=rank, world_size=2, local_rank=0),
        resume=None if resume is None else Path(resume),
    )


def _spawn_world(
    corpus: tuple[Path, Path],
    tmp_path: Path,
    config: TrainingConfig,
    out_dir: str,
    resume: Path | None = None,
) -> Path:
    """Run *config*'s segment as a two-process gloo world under *out_dir*."""
    import socket

    import torch.multiprocessing as mp

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    paths = (str(corpus[0]), str(corpus[1]))
    target = tmp_path / out_dir
    mp.spawn(
        _gloo_rank,
        args=(paths, None if resume is None else str(resume), str(target), port, config),
        nprocs=2,
        join=True,
    )
    return target


def _positions(path: Path) -> list[dict[str, object]]:
    """Every rank's (epoch, index) in a checkpoint, in rank order."""
    state = torch.load(path, weights_only=False)
    return [{"epoch": p["epoch"], "index": p["index"]} for p in state["positions"]]


def test_two_virtual_ranks_match_a_two_process_world(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """A single process as two ranks computes what two gloo processes compute.

    Same data, same steps: the virtual run's losses, its final weights and the
    positions it checkpoints all equal the real world's -- within the small
    float difference a different summation order can leave behind.
    """
    config = replace(RESUMABLE, accumulate=2, prefetch=True)
    ddp_dir = _spawn_world(corpus, tmp_path, config, "ddp")

    lanes = _rank_lanes(corpus, lexicon, spans, arbitration, tokenizer, 0, 2)
    model = tiny_model(lexicon)
    virtual = train(
        model,
        lanes,
        replace(config, virtual_ranks=2),
        tmp_path / "virtual",
    )
    assert virtual.finished and virtual.step == config.max_steps
    assert step_losses(virtual.metrics) == pytest.approx(
        step_losses(ddp_dir / "metrics.jsonl"), rel=1e-6
    )
    assert _positions(ddp_dir / "checkpoint-final.pt") == _positions(
        tmp_path / "virtual" / "checkpoint-final.pt"
    )
    reference = torch.load(ddp_dir / "checkpoint-final.pt", weights_only=False)["model"]
    for name, tensor in model.state_dict().items():
        assert torch.allclose(tensor, reference[name], atol=1e-5), name


def test_the_two_worlds_resume_each_other(
    corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """A gloo world's checkpoint resumes in a virtual process, and back again."""
    config = replace(RESUMABLE, accumulate=2, prefetch=True)
    ddp_dir = _spawn_world(corpus, tmp_path, config, "ddp")

    lanes = _rank_lanes(corpus, lexicon, spans, arbitration, tokenizer, 0, 2)
    virtual_model = tiny_model(lexicon)
    virtual = train(
        virtual_model,
        lanes,
        replace(config, virtual_ranks=2),
        tmp_path / "virtual",
    )

    # Direction one: the real world's mid-run checkpoint resumes in a process
    # standing for both ranks, and the same three steps come out.
    lanes = _rank_lanes(corpus, lexicon, spans, arbitration, tokenizer, 0, 2)
    resumed_model = tiny_model(lexicon)
    resumed = train(
        resumed_model,
        lanes,
        replace(config, virtual_ranks=2),
        tmp_path / "from-ddp",
        resume=ddp_dir / "checkpoint-000003.pt",
    )
    assert resumed.finished
    assert step_losses(resumed.metrics) == pytest.approx(
        step_losses(ddp_dir / "metrics.jsonl")[3:], rel=1e-6
    )
    reference = torch.load(ddp_dir / "checkpoint-final.pt", weights_only=False)["model"]
    for name, tensor in resumed_model.state_dict().items():
        assert torch.allclose(tensor, reference[name], atol=1e-5), name

    # Direction two: the virtual world's mid-run checkpoint resumes in the
    # gloo world, each process taking its own rank's position.
    ddp_resumed = _spawn_world(
        corpus,
        tmp_path,
        config,
        "from-virtual",
        resume=tmp_path / "virtual" / "checkpoint-000003.pt",
    )
    assert step_losses(ddp_resumed / "metrics.jsonl") == pytest.approx(
        step_losses(virtual.metrics)[3:], rel=1e-6
    )
    assert _positions(ddp_resumed / "checkpoint-final.pt") == _positions(
        tmp_path / "virtual" / "checkpoint-final.pt"
    )


def test_an_interval_checkpoint_resumes_like_a_pause(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """The trio an interval checkpoint writes continues the run at its step.

    A leg that can die without warning publishes these; resuming from one has
    to be resuming from a pause, so the steps after it equal an uninterrupted
    run's.
    """
    longer = replace(RESUMABLE, max_steps=20, checkpoint_every=1000)
    whole_dir = tmp_path / "whole"
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    whole = train(tiny_model(lexicon), [(stream, collator)], longer, whole_dir)

    # A leg that dies mid-run: the measurement window fills at step 12 and
    # the loop stops there, leaving the interval trios it wrote along the way
    # -- the cadence here is fast enough to fire every step, and only the
    # newest is kept. The trio's run-config is the kernel's own file, verbatim.
    died_dir = tmp_path / "died"
    run_config = tmp_path / "kernel-run-config.json"
    run_config.write_text(json.dumps({"max_steps": 20, "epochs": 1, "segment": 3}) + "\n")
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    died = train(
        tiny_model(lexicon),
        [(stream, collator)],
        replace(longer, checkpoint_minutes=1e-9, profile_steps=2),
        died_dir,
        run_config=run_config,
    )
    assert not died.finished and died.step == 12
    intervals = sorted(died_dir.glob("interval-*"))
    assert [directory.name for directory in intervals] == ["interval-000012"]
    trio = intervals[0]
    for marker in ("checkpoint-paused.pt", "run-config.json", "run-summary.json"):
        assert (trio / marker).is_file(), marker
    summary = json.loads((trio / "run-summary.json").read_text())
    assert summary["checkpoint_step"] == 12 and not summary["finished"]
    assert summary["processes"] == 1 and summary["virtual_ranks"] == 1
    # Verbatim bytes, so the next leg reads segment and epochs too.
    assert (trio / "run-config.json").read_bytes() == run_config.read_bytes()

    resumed_model = tiny_model(lexicon)
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    resumed = train(
        resumed_model,
        [(stream, collator)],
        longer,
        tmp_path / "resumed",
        resume=trio / "checkpoint-paused.pt",
    )
    assert resumed.finished and resumed.step == longer.max_steps
    assert step_losses(resumed.metrics) == step_losses(whole.metrics)[12:]
    uninterrupted = torch.load(whole_dir / "checkpoint-final.pt", weights_only=False)["model"]
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, uninterrupted[name]), name


def test_an_interval_cadence_needs_the_run_config(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """--checkpoint-minutes without --run-config refuses before any step."""
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    with pytest.raises(ValueError, match="run_config"):
        train(
            tiny_model(lexicon),
            [(stream, collator)],
            replace(RESUMABLE, checkpoint_minutes=1),
            tmp_path / "refused",
        )


def test_a_lane_count_mismatch_refuses(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """A process told to stand for two ranks but handed one lane raises."""
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    with pytest.raises(ValueError, match="virtual_ranks"):
        train(
            tiny_model(lexicon),
            [(stream, collator)],
            replace(RESUMABLE, virtual_ranks=2),
            tmp_path,
        )
