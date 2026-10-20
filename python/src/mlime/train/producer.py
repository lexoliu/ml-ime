"""Where a training lane's batches are made: in the trainer's process, or in one of their own.

A lane is one virtual rank's stream and collator. Collating is Python, and in
the trainer's process every millisecond of it holds the GIL the training step
needs to launch kernels and to free the gradients backward hands to
``AccumulateGrad``: the L4 profile of #156 showed the step's first micro-batch
idling the card 470 ms while two producer threads refilled their queues. So a
prefetching run makes each lane's batches in a process of its own, which
shares nothing with the trainer but the queue the batches come back on.

The child cannot be handed the lane's objects -- a ranking collator owns a
running segmenter -- so it is handed a :class:`LaneSource`, a picklable
description it opens itself, and the position to read on from. It then runs
exactly the :class:`EpochBatches` the trainer's process would have, so its
items are the in-process producer's, position for position.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import multiprocessing.connection
import os
import queue
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch
import torch.multiprocessing

from mlime.logging import configure, log
from mlime.train.samples import (
    BatchCollator,
    CorpusStream,
    StepBatch,
    TrainingExample,
    token_budget_batches,
)

#: How long a blocked queue operation waits before it looks again at whether
#: it should still be waiting: the stop flag, the other side's liveness.
POLL_SECONDS = 0.1

#: How long :meth:`ProcessProducer.close` lets the child finish the batch in
#: hand and report its counts before it is terminated. A batch is milliseconds
#: and a shard read a second or two; a child still busy after this is stuck.
CLOSE_SECONDS = 30.0


def epoch_groups(
    stream: CorpusStream, collator: BatchCollator, budget: int, epoch: int
) -> Iterator[list[TrainingExample]]:
    """The examples of *epoch*, grouped exactly as the steps of that epoch will be.

    Grouping is not collation, and the two are separated here because three
    things need the boundaries and none of them needs the tensors: the loop, on
    its way to collating each group; :meth:`EpochBatches.skip`, walking a resumed
    stream back to a saved position; and counting how many steps an epoch is
    worth before a run fixes its schedule.

    The epoch is an argument rather than a detail because it decides what the
    examples *are*: the augmentation re-samples every epoch, an abbreviated
    sentence is shorter than a typed-out one, and a batch boundary is a function
    of the lengths. Counting epoch 0 twice is not counting two epochs.
    """
    stream.set_epoch(epoch)
    return token_budget_batches(iter(stream), budget, collator.max_context_tokens, collator.rows)


@dataclass(frozen=True)
class Produced:
    """A collated batch, and where the stream stood when it was made.

    ``epoch``/``index`` count the batches produced so far and ``collator_rng``
    is the collator's generator state just after this batch was collated. A
    checkpoint taken on this item resumes exactly after it: the batch a
    re-opened stream produces next is the one an uninterrupted run would have
    consumed. A prefetching producer carries this per item because the queue
    may already hold batches a pause never reaches -- the position a checkpoint
    writes is the last batch *consumed*, not the last one produced.
    """

    batch: StepBatch
    epoch: int
    index: int
    collator_rng: tuple[Any, ...]


class EpochBatches(Iterator[StepBatch]):
    """The stream as collated batches, epoch after epoch, and where in it we are.

    The position is what makes a run resumable. A rank's batches are its own --
    it owns its own shards, and how many batches an epoch holds depends on the
    lengths of the sentences in them -- so "step 4000" says nothing about where
    to start reading. The epoch and the count of batches taken from it do, and
    :meth:`skip` walks a fresh stream back to exactly there.
    """

    def __init__(self, stream: CorpusStream, collator: BatchCollator, budget: int):
        self.stream = stream
        self.collator = collator
        self.budget = budget
        #: The epoch the batch last yielded came from.
        self.epoch = stream.epoch
        #: How many batches of that epoch have been yielded.
        self.index = 0
        self._groups = self._open_epoch()

    def __next__(self) -> StepBatch:
        return self.produce().batch

    def produce(self) -> Produced:
        """Collate the next group, and record the position it leaves behind."""
        batch = self.collator(self._next_group())
        return Produced(
            batch=batch,
            epoch=self.epoch,
            index=self.index,
            collator_rng=self.collator.rng.getstate(),
        )

    def _open_epoch(self) -> Iterator[list[TrainingExample]]:
        """Start reading :attr:`epoch`, grouped to the budget but not collated."""
        return epoch_groups(self.stream, self.collator, self.budget, self.epoch)

    def _next_group(self) -> list[TrainingExample]:
        """The next group of examples, rolling into the next epoch at the end of one."""
        group = next(self._groups, None)
        if group is not None:
            self.index += 1
            return group
        if self.index == 0:
            raise RuntimeError(f"epoch {self.epoch} of the corpus produced no example")
        self.epoch += 1
        log.info("epoch finished", epoch=self.epoch, **self.stream.builder.counts.as_dict())
        self.index = 0
        self._groups = self._open_epoch()
        return self._next_group()

    def skip(self, epoch: int, index: int) -> None:
        """Read forward to *index* batches into *epoch*, building no tensors.

        The examples still have to be built, because building them is what walks
        the stream, and they are grouped because a group boundary depends on the
        examples before it. They are not collated: collation is where the padding
        and the tensors are, and none of it would be looked at.
        """
        if epoch < 0 or index < 0:
            raise ValueError(f"a stream position is not negative, got epoch {epoch} index {index}")
        started = time.monotonic()
        self.epoch = epoch
        self.index = 0
        self._groups = self._open_epoch()
        for _ in range(index):
            self._next_group()
        if (self.epoch, self.index) != (epoch, index):
            raise RuntimeError(
                f"epoch {epoch} ran out before batch {index}, at {self.epoch}/{self.index}: "
                "this is not the corpus the checkpoint was written from"
            )
        log.info(
            "batch stream skipped",
            epoch=epoch,
            index=index,
            seconds=round(time.monotonic() - started, 1),
        )


class LaneSource(Protocol):
    """A picklable description of one lane, opened where its batches are made."""

    def open(self) -> tuple[CorpusStream, BatchCollator]:
        """Build the lane's stream and collator."""
        ...


@dataclass(frozen=True)
class StreamPosition:
    """Where a resumed lane reads on from: a batch count into an epoch, and the coin's state."""

    epoch: int
    index: int
    collator_rng: tuple[Any, ...]


def open_batches(source: LaneSource, budget: int, start: StreamPosition | None) -> EpochBatches:
    """Open *source* as batches, walked forward to *start* when the lane is resumed."""
    stream, collator = source.open()
    batches = EpochBatches(stream, collator, budget)
    if start is not None:
        batches.skip(start.epoch, start.index)
        collator.rng.setstate(start.collator_rng)
    return batches


class LocalProducer(Iterator[Produced]):
    """The lane's batches made on demand, in the trainer's own process and thread."""

    def __init__(self, source: LaneSource, budget: int, start: StreamPosition | None):
        self._batches = open_batches(source, budget, start)

    def __next__(self) -> Produced:
        return self._batches.produce()

    def close(self) -> None:
        """Nothing runs beside the trainer, so there is nothing to stop."""

    def counts(self) -> dict[str, int]:
        """What the lane's builder kept and dropped so far."""
        return self._batches.stream.builder.counts.as_dict()


@dataclass(frozen=True)
class ProducerFailure:
    """What a producer sends instead of a batch when it cannot make one."""

    message: str


class ProcessProducer(Iterator[Produced]):
    """The lane's batches made ahead in a spawned process, pinned on arrival.

    The child opens *source*, walks it to *start*, and puts each produced item
    on a ``torch.multiprocessing`` queue *depth* deep, so its tensors cross in
    shared memory. A thread here takes them off, pins them into page-locked
    memory when *pin* is set -- so the step's ``non_blocking`` copy is
    asynchronous -- and holds up to *depth* ready for the step.

    A failure in the child arrives in order, after the batches it did make,
    and is raised at the trainer's next batch; so is the child's death.
    :meth:`close` stops the child, takes the build counts it reports on the
    way out, and joins it with a bound; the child also exits on its own when
    the trainer's process goes away.
    """

    def __init__(
        self,
        source: LaneSource,
        budget: int,
        start: StreamPosition | None,
        *,
        depth: int,
        pin: bool,
    ):
        context = torch.multiprocessing.get_context("spawn")
        self._items: Any = context.Queue(maxsize=depth)
        self._stop = context.Event()
        self._reports, sender = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_produce,
            args=(source, budget, start, self._items, self._stop, sender),
            daemon=True,
            name="batch-producer",
        )
        self._process.start()
        sender.close()
        self._ready: queue.Queue[Produced | ProducerFailure] = queue.Queue(maxsize=depth)
        self._halt = threading.Event()
        self._counts: dict[str, int] | None = None
        self._pinner = threading.Thread(
            target=self._receive, args=(pin,), daemon=True, name="batch-pin"
        )
        self._pinner.start()

    @property
    def pid(self) -> int | None:
        """The child's process id."""
        return self._process.pid

    def is_alive(self) -> bool:
        """Whether the child is still running."""
        return self._process.is_alive()

    def _receive(self, pin: bool) -> None:
        """Move items from the child's queue to the step's, pinning them on the way."""
        try:
            while not self._halt.is_set():
                try:
                    got: Produced | ProducerFailure = self._items.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    if not self._process.is_alive() and not self._halt.is_set():
                        self._hand(self._death())
                        return
                    continue
                except (EOFError, OSError):
                    # Where tensors cross as file descriptors (Linux), the child
                    # hands each queued batch's storage over as it is unpickled;
                    # a child that died cannot, and its death is what failed.
                    self._process.join(POLL_SECONDS)
                    if self._process.is_alive():
                        raise
                    self._hand(self._death())
                    return
                if pin and not isinstance(got, ProducerFailure):
                    got = replace(got, batch=got.batch.pin_memory())
                self._hand(got)
                if isinstance(got, ProducerFailure):
                    return
        except BaseException:  # ferried to the trainer, which owns the run
            self._hand(ProducerFailure(traceback.format_exc()))

    def _death(self) -> ProducerFailure:
        """The failure a child that exited without reporting one leaves behind."""
        return ProducerFailure(
            f"the batch producer exited with code {self._process.exitcode} without reporting why"
        )

    def _hand(self, item: Produced | ProducerFailure) -> None:
        """Queue *item* for the step, giving up only when the producer is closing."""
        while not self._halt.is_set():
            try:
                self._ready.put(item, timeout=POLL_SECONDS)
                return
            except queue.Full:
                continue

    def __next__(self) -> Produced:
        item = self._ready.get()
        if isinstance(item, ProducerFailure):
            raise RuntimeError(f"the batch producer failed:\n{item.message}")
        return item

    def close(self) -> None:
        """Stop the child and the pinning thread, keep the counts, and join both, bounded.

        The pinning thread goes first: it is the only other caller that reaps
        the child, and two reapers racing can each see the other's exit as
        "still running". A child that does not finish within
        :data:`CLOSE_SECONDS` of being told to stop is terminated, then killed;
        none outlives this call.
        """
        self._stop.set()
        self._halt.set()
        self._pinner.join(CLOSE_SECONDS)
        # A child that failed or died closes its end with no counts to give.
        if self._reports.poll(CLOSE_SECONDS):
            with contextlib.suppress(EOFError):
                self._counts = self._reports.recv()
        self._process.join(CLOSE_SECONDS)
        if self._process.is_alive():
            log.warning("batch producer ignored its stop; terminating", pid=self._process.pid)
            self._process.terminate()
            self._process.join(CLOSE_SECONDS)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()
        self._reports.close()
        self._items.close()
        if self._pinner.is_alive():
            raise RuntimeError(f"the batch pinning thread did not stop in {CLOSE_SECONDS} s")

    def counts(self) -> dict[str, int]:
        """The child's build counts, as it reported them when :meth:`close` stopped it."""
        if self._counts is None:
            raise RuntimeError(
                "the batch producer reported no build counts: it was not closed yet, "
                f"or it failed (exit code {self._process.exitcode})"
            )
        return self._counts


def _exit_with_parent() -> None:
    """End this process the moment the one that spawned it is gone."""
    parent = multiprocessing.parent_process()
    if parent is None:
        raise RuntimeError("the batch producer runs only as a spawned child")
    multiprocessing.connection.wait([parent.sentinel])
    os._exit(1)


def _produce(
    source: LaneSource,
    budget: int,
    start: StreamPosition | None,
    items: Any,
    stop: Any,
    reports: multiprocessing.connection.Connection,
) -> None:
    """The child: produce into *items* until *stop*, then report the build counts."""
    threading.Thread(target=_exit_with_parent, daemon=True, name="parent-watch").start()
    configure()
    # Collation's tensor ops are small integer work; a pool of intra-op
    # threads per child would only take cores from the trainer's threads.
    torch.set_num_threads(1)
    # Whatever is still queued when the trainer stops is never read; the
    # child must not wait on the pipe to flush it before it may exit.
    items.cancel_join_thread()
    try:
        batches = open_batches(source, budget, start)
        while not stop.is_set():
            item = batches.produce()
            while not stop.is_set():
                try:
                    items.put(item, timeout=POLL_SECONDS)
                    break
                except queue.Full:
                    continue
    except BaseException:  # sent to the trainer, which raises it at its next batch
        failure = ProducerFailure(traceback.format_exc())
        while not stop.is_set():
            try:
                items.put(failure, timeout=POLL_SECONDS)
                break
            except queue.Full:
                continue
        # The batches already queued may still be unpickled by the trainer,
        # which can need this process alive to hand their storage over; it
        # waits for the trainer's close, or its exit, before it leaves.
        stop.wait()
        reports.close()
        return
    reports.send(batches.stream.builder.counts.as_dict())
    reports.close()
