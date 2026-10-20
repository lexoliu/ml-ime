"""The per-lane producer process: the same batches as the trainer's own, and no orphans.

A prefetching run collates each lane in a spawned process so the trainer's
GIL stays with the step (#156). That is only a speed-up if nothing else
changes, so the first tests hold the process's items -- positions, the
collator's generator state, every tensor -- to the in-process producer's, from
a fresh start and from a resume. The rest hold the process to its contract
with the trainer: a failure in the child is raised at the trainer's next
batch, and neither a stop, a pause, a failure nor the trainer's own death
leaves a child running.
"""

from __future__ import annotations

import os
import pickle
import random
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import BuiltLane, SleepingLane
from test_train_loop import RESUMABLE, SPENT, stream_and_collator, tiny_model

from mlime.train import loop, producer
from mlime.train.arbitration import ReadingArbitration
from mlime.train.e2e_model import RankingConfig
from mlime.train.lexicon import Lexicon
from mlime.train.producer import (
    LaneSource,
    LocalProducer,
    ProcessProducer,
    Produced,
    StreamPosition,
)
from mlime.train.run import CorpusLane, LaneRanking, RunPaths
from mlime.train.samples import (
    Augmentation,
    BaseTokenizer,
    Collator,
    StepBatch,
    TrainingExample,
)
from mlime.train.spans import SpanVocab

#: Batches of the short corpus at the resumable budget: four an epoch, so ten
#: cross two epoch boundaries.
TAKEN = 10


class BrokenCollator:
    """The real collator until its *works*-th batch, then a collation that raises.

    Module-level so that a producer process can unpickle it.
    """

    def __init__(self, inner: Collator, works: int):
        self.inner = inner
        self.works = works
        self.calls = 0

    @property
    def rng(self) -> random.Random:
        return self.inner.rng

    @property
    def max_context_tokens(self) -> int:
        return self.inner.max_context_tokens

    def rows(self, example: TrainingExample) -> int:
        return self.inner.rows(example)

    def __call__(self, examples: Sequence[TrainingExample]) -> StepBatch:
        self.calls += 1
        if self.calls > self.works:
            raise ValueError(f"collation {self.calls} broke on purpose")
        return self.inner(examples)


def assert_same_items(got: Sequence[Produced], wanted: Sequence[Produced]) -> None:
    """Item for item: the position, the collator's generator, the ids and every tensor."""
    assert len(got) == len(wanted)
    for index, (item, reference) in enumerate(zip(got, wanted, strict=True)):
        assert (item.epoch, item.index) == (reference.epoch, reference.index), index
        assert item.collator_rng == reference.collator_rng, index
        assert item.batch.ids == reference.batch.ids, index
        tensors = [name for name in vars(reference.batch) if name != "ids"]
        assert len(tensors) == 13
        for name in tensors:
            ours, theirs = getattr(item.batch, name), getattr(reference.batch, name)
            assert ours.dtype == theirs.dtype, (index, name)
            assert torch.equal(ours, theirs), (index, name)


@pytest.fixture(name="lane")
def lane_fixture(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
) -> BuiltLane:
    """An unread short-corpus lane."""
    return BuiltLane(*stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer))


def local_items(
    lane: LaneSource, count: int, start: StreamPosition | None = None
) -> list[Produced]:
    """*count* items of the in-process producer -- the thread producer's sequence."""
    source = LocalProducer(lane, RESUMABLE.token_budget, start)
    return [next(source) for _ in range(count)]


def process_items(
    lane: LaneSource, count: int, start: StreamPosition | None = None
) -> tuple[list[Produced], ProcessProducer]:
    """*count* items of a producer process, closed, and the producer itself."""
    source = ProcessProducer(lane, RESUMABLE.token_budget, start, depth=2, pin=False)
    try:
        items = [next(source) for _ in range(count)]
    finally:
        source.close()
    return items, source


def test_a_producer_process_makes_the_in_process_batches(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
) -> None:
    """From a fresh start, across two epoch boundaries, item for item."""
    wanted = local_items(
        BuiltLane(*stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)),
        TAKEN,
    )
    assert {item.epoch for item in wanted} == {0, 1, 2}
    got, source = process_items(
        BuiltLane(*stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)),
        TAKEN,
    )
    assert_same_items(got, wanted)
    assert not source.is_alive()


@pytest.mark.parametrize("after", [3, 5], ids=["at-an-epoch-end", "mid-epoch"])
def test_a_resumed_producer_process_continues_the_in_process_batches(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    after: int,
) -> None:
    """Walked to a consumed item's position, the child makes what came next.

    The fourth item is the last of epoch 0, so resuming after it must roll the
    child into epoch 1 exactly where an uninterrupted stream did.
    """
    wanted = local_items(
        BuiltLane(*stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)),
        TAKEN,
    )
    consumed = wanted[after]
    start = StreamPosition(consumed.epoch, consumed.index, consumed.collator_rng)
    got, _ = process_items(
        BuiltLane(*stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)),
        TAKEN - after - 1,
        start,
    )
    assert_same_items(got, wanted[after + 1 :])
    # And the in-process producer resumes the same way, so the two agree on a resume too.
    resumed = local_items(
        BuiltLane(*stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)),
        TAKEN - after - 1,
        start,
    )
    assert_same_items(resumed, wanted[after + 1 :])


def test_a_failure_in_the_child_is_raised_at_the_next_batch(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
) -> None:
    """The batches made before the failure still arrive, then the child's traceback."""
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    source = ProcessProducer(
        BuiltLane(stream, BrokenCollator(collator, works=2)),
        RESUMABLE.token_budget,
        None,
        depth=2,
        pin=False,
    )
    try:
        assert [(item.epoch, item.index) for item in (next(source), next(source))] == [
            (0, 1),
            (0, 2),
        ]
        with pytest.raises(RuntimeError, match="collation 3 broke on purpose") as raised:
            next(source)
        assert "ValueError" in str(raised.value)
    finally:
        source.close()
    assert not source.is_alive()
    with pytest.raises(RuntimeError, match="reported no build counts"):
        source.counts()


def test_close_stops_a_child_blocked_on_a_full_queue(lane: BuiltLane) -> None:
    """The child has filled the queue and waits to put more; close ends it cleanly."""
    source = ProcessProducer(lane, RESUMABLE.token_budget, None, depth=1, pin=False)
    next(source)
    source.close()
    assert not source.is_alive()
    assert source._process.exitcode == 0
    counts = source.counts()
    assert counts["kept"] > 0


def test_close_ends_a_child_that_never_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child stuck before its first batch is terminated within close's bound."""
    monkeypatch.setattr(producer, "CLOSE_SECONDS", 1.0)
    source = ProcessProducer(SleepingLane(), RESUMABLE.token_budget, None, depth=1, pin=False)
    started = time.monotonic()
    source.close()
    assert time.monotonic() - started < 10
    assert not source.is_alive()
    assert source._process.exitcode == -signal.SIGTERM


def test_a_child_that_dies_is_raised_at_the_next_batch(lane: BuiltLane) -> None:
    source = ProcessProducer(lane, RESUMABLE.token_budget, None, depth=2, pin=False)
    try:
        next(source)
        pid = source.pid
        assert pid is not None
        os.kill(pid, signal.SIGKILL)
        # What was already queued may still arrive; the death comes after it.
        with pytest.raises(RuntimeError, match=f"exited with code {-signal.SIGKILL}"):
            for _ in range(8):
                next(source)
    finally:
        source.close()
    assert not source.is_alive()


def test_a_child_exits_when_its_trainer_is_killed(tmp_path: Path) -> None:
    """SIGKILL gives the trainer no cleanup; the child notices and leaves on its own."""
    opened = tmp_path / "opened"
    script = "\n".join(
        [
            "import sys",
            "from conftest import SleepingLane",
            "from mlime.train.producer import ProcessProducer",
            f"lane = SleepingLane(marker={str(opened)!r})",
            "source = ProcessProducer(lane, 64, None, depth=1, pin=False)",
            "print(source.pid, flush=True)",
            "sys.stdin.read()",
        ]
    )
    tests = Path(__file__).resolve().parent
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tests), *sys.path])}
    trainer = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=environment,
        text=True,
    )
    try:
        assert trainer.stdout is not None
        child = int(trainer.stdout.readline())
        # The child is killed inside its lane, running code of its own, rather
        # than while it is still being handed its arguments.
        deadline = time.monotonic() + 60
        while not opened.exists():
            assert time.monotonic() < deadline, "the producer never reached its lane"
            time.sleep(0.1)
        trainer.kill()
        trainer.wait(timeout=10)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        os.kill(child, signal.SIGKILL)
        pytest.fail(f"producer {child} outlived its killed trainer by 20 s")
    finally:
        if trainer.poll() is None:
            trainer.kill()
            trainer.wait(timeout=10)


@pytest.fixture(name="recorded")
def recorded_fixture(monkeypatch: pytest.MonkeyPatch) -> list[ProcessProducer]:
    """Every producer process the loop starts during the test, so it can check each is gone."""
    made: list[ProcessProducer] = []

    class Recorded(ProcessProducer):
        def __init__(
            self,
            source: LaneSource,
            budget: int,
            start: StreamPosition | None,
            *,
            depth: int,
            pin: bool,
        ) -> None:
            super().__init__(source, budget, start, depth=depth, pin=pin)
            made.append(self)

    monkeypatch.setattr(loop, "ProcessProducer", Recorded)
    return made


def test_a_paused_prefetched_segment_leaves_no_child_and_reports_its_counts(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
    recorded: list[ProcessProducer],
) -> None:
    """The counts come back from the child at close; it built at least what was trained on."""
    local = loop.train(
        tiny_model(lexicon),
        [BuiltLane(*stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer))],
        SPENT,
        tmp_path / "local",
    )
    paused = loop.train(
        tiny_model(lexicon),
        [BuiltLane(*stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer))],
        replace(SPENT, prefetch=True),
        tmp_path / "prefetched",
    )
    assert (paused.step, paused.finished) == (1, False)
    assert len(recorded) == 1
    assert not recorded[0].is_alive()
    assert paused.build_counts.keys() == local.build_counts.keys()
    assert local.build_counts["kept"] > 0
    assert paused.build_counts["kept"] >= local.build_counts["kept"]


def test_a_failing_prefetched_lane_fails_the_run_and_leaves_no_child(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
    recorded: list[ProcessProducer],
) -> None:
    """The trainer raises the child's error at the step that needed the batch."""
    stream, collator = stream_and_collator(short_corpus, lexicon, spans, arbitration, tokenizer)
    with pytest.raises(RuntimeError, match="collation 2 broke on purpose"):
        loop.train(
            tiny_model(lexicon),
            [BuiltLane(stream, BrokenCollator(collator, works=1))],
            replace(RESUMABLE, prefetch=True),
            tmp_path,
        )
    assert len(recorded) == 1
    assert not recorded[0].is_alive()


def test_a_corpus_lane_travels_to_a_producer_intact(tmp_path: Path) -> None:
    """The production lane is the recipe a child rebuilds from, so it pickles to itself."""
    lane = CorpusLane(
        paths=RunPaths(
            samples=tmp_path / "samples",
            labels=tmp_path / "labels",
            char_table=tmp_path / "chars.tsv",
            out=tmp_path / "out",
        ),
        base_model="hfl/chinese-macbert-base",
        shards=("a-00000.parquet", "b-00001.parquet"),
        augmentation=Augmentation(typos=False),
        seed=7,
        rank=3,
        world_size=4,
        context_dropout=0.3,
        max_context_tokens=128,
        ranking=LaneRanking(RankingConfig(segmentations=4), tmp_path / "ime-segment"),
    )
    assert pickle.loads(pickle.dumps(lane)) == lane
