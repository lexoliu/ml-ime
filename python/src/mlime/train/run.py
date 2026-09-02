"""Wiring: the one place that turns paths and a config into a finished route A run.

The kernel that runs on Kaggle and the command that runs on a laptop must be the
same run, or the numbers the kernel reports say nothing about the code in the
repository. So neither owns any assembly: both call :func:`route_a`, which builds
the lexicon against the base model's own vocabulary, streams the shards it was
given, trains, and then measures held-out accuracy twice -- once with the context
and once without -- because the whole hypothesis of milestone 3 is the difference
between those two numbers.

Every run records what it was made of: which shards, which labels, which seed,
and which commit, so a metrics file found six months later is still evidence.

A run can also be longer than one kernel. Then :func:`route_a` is called once per
segment, each continuing the last one's checkpoint, and every segment but the
last stops on its wall budget with nothing scored: the held-out numbers describe
a finished model, and a segment that paused halfway through the schedule has not
got one to describe.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from mlime.data.shards import shard_paths
from mlime.logging import log
from mlime.train.arbitration import ReadingArbitration
from mlime.train.lexicon import Lexicon, build_lexicon, read_char_readings
from mlime.train.loop import (
    Accuracy,
    Distributed,
    MetricLog,
    Segment,
    TrainingConfig,
    evaluate,
    train,
)
from mlime.train.model import RouteAConfig, RouteAModel
from mlime.train.samples import (
    DEFAULT_CONTEXT_TOKENS,
    Augmentation,
    BaseTokenizer,
    Collator,
    CorpusStream,
    SampleBuilder,
    TrainingExample,
)
from mlime.train.spans import SpanVocab


@dataclass(frozen=True)
class RunPaths:
    """Where a run reads and writes."""

    samples: Path
    labels: Path
    char_table: Path
    out: Path


@dataclass(frozen=True)
class Slices:
    """Which shards train and which are held out.

    Named shards rather than a fraction, because a held-out *fraction* of a
    stream is only reproducible if the stream is, and the point of holding data
    out is that it stays out no matter how the reader is sharded.
    """

    train: tuple[str, ...]
    held_out: tuple[str, ...]
    max_held_out_examples: int = 4096

    @classmethod
    def all_but(
        cls, samples: Path, held_out: Sequence[str], max_held_out_examples: int = 4096
    ) -> Slices:
        """Train on every shard under *samples* except the held-out ones.

        A full epoch names four hundred shards, and naming them on a command line
        is both unreadable and a way to leave one out by accident. Naming what is
        *withheld* is short, is the part a reader has to check, and cannot
        silently shrink the training set.
        """
        available = [path.name for path in shard_paths(samples, "*")]
        if not available:
            raise FileNotFoundError(f"no sample shards under {samples}")
        missing = set(held_out) - set(available)
        if missing:
            raise FileNotFoundError(f"no such shards under {samples}: {sorted(missing)}")
        withheld = set(held_out)
        return cls(
            train=tuple(name for name in available if name not in withheld),
            held_out=tuple(held_out),
            max_held_out_examples=max_held_out_examples,
        )

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.held_out)
        if overlap:
            raise ValueError(f"these shards are both trained on and held out: {sorted(overlap)}")
        if not self.train:
            raise ValueError("a run needs at least one training shard")


@dataclass(frozen=True)
class RunResult:
    """What a run produced, in the form the report quotes.

    ``first_loss`` and ``last_loss`` are the first and last losses this
    *segment*'s metrics file recorded. A run resumed into a new kernel writes a
    new metrics file, so its ``first_loss`` is the loss at the step after the
    checkpoint, not the loss the run started from: the run's own first loss is in
    the first segment's file, and the segments are read together.

    A segment that paused on its wall budget has no accuracies, and they are
    ``None`` rather than zeroes or last-known values: a placeholder here would be
    quoted as a result, and "0.0 with context" is a number somebody would act on.
    :attr:`scored` is how a caller asks for them, and it refuses rather than
    inventing.
    """

    metrics: Path
    first_loss: float
    last_loss: float
    steps: int
    step: int
    finished: bool
    with_context: Accuracy | None
    without_context: Accuracy | None
    gates: list[float]

    def __post_init__(self) -> None:
        measured = self.with_context is not None and self.without_context is not None
        if measured != self.finished:
            raise ValueError(
                f"a run that {'finished' if self.finished else 'paused'} at step {self.step} "
                f"{'has no' if measured else 'has'} held-out accuracy"
            )

    @property
    def scored(self) -> tuple[Accuracy, Accuracy]:
        """The accuracies with and without context, or a refusal if it paused."""
        if self.with_context is None or self.without_context is None:
            raise ValueError(
                f"the run paused at step {self.step} and was never scored; "
                "resume it to the end before quoting an accuracy"
            )
        return self.with_context, self.without_context


def corpus_digest(paths: list[Path]) -> str:
    """A hash of the shard names and sizes -- enough to catch a swapped corpus."""
    digest = hashlib.blake2b(digest_size=16)
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
    return digest.hexdigest()


def git_commit() -> str:
    """The checkout's commit, or a marker when there is no checkout to ask.

    A Kaggle kernel has the package but not the repository, so this is allowed to
    come back unknown -- what must never happen is a *wrong* commit being
    recorded, which is why nothing is guessed.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def load_tokenizer(base_model: str) -> BaseTokenizer:
    """The base model's tokenizer, which is also the lexicon's vocabulary."""
    from transformers import AutoTokenizer

    tokenizer: BaseTokenizer = AutoTokenizer.from_pretrained(base_model)
    return tokenizer


def vocabulary_of(tokenizer: BaseTokenizer) -> dict[str, int]:
    """The tokenizer's ``token -> id`` map."""
    vocabulary: dict[str, int] = tokenizer.get_vocab()  # type: ignore[attr-defined]
    return vocabulary


def build_lexicon_for(char_table: Path, tokenizer: BaseTokenizer, spans: SpanVocab) -> Lexicon:
    """Intersect the pinyin lexicon with the base model's vocabulary."""
    return build_lexicon(read_char_readings(char_table), vocabulary_of(tokenizer), spans)


@dataclass(frozen=True)
class Vocabularies:
    """The four tables everything a run does is indexed by.

    They belong together because they are built from one another: the lexicon is
    the pinyin table intersected with the *base model's* vocabulary, so it cannot
    be built without the tokenizer, the spans index both the augmentation and
    the model's extra embedding table, and the arbitration table is checked
    against those same spans. Anything that reads the corpus the way a run reads
    it -- training, and counting what training will do -- needs all four, built
    the same way, which is why they are assembled once here rather than gathered
    again at each call site.
    """

    spans: SpanVocab
    tokenizer: BaseTokenizer
    lexicon: Lexicon
    arbitration: ReadingArbitration

    @classmethod
    def load(cls, char_table: Path, base_model: str) -> Vocabularies:
        """Load the span table, the tokenizer, the lexicon over both, and the arbitration."""
        spans = SpanVocab.load()
        tokenizer = load_tokenizer(base_model)
        return cls(
            spans=spans,
            tokenizer=tokenizer,
            lexicon=build_lexicon_for(char_table, tokenizer, spans),
            arbitration=ReadingArbitration.load(spans),
        )

    @property
    def vocabulary(self) -> dict[str, int]:
        """The base model's ``token -> id`` map."""
        return vocabulary_of(self.tokenizer)

    def builder(self, augmentation: Augmentation | None, seed: int) -> SampleBuilder:
        """A builder that types sentences the way *seed* and *augmentation* say."""
        return SampleBuilder(self.lexicon, self.spans, self.arbitration, augmentation, seed=seed)

    def stream(
        self,
        paths: RunPaths,
        shards: Sequence[str],
        builder: SampleBuilder,
        rank: int = 0,
        world_size: int = 1,
    ) -> CorpusStream:
        """The shards *rank* owns, as the examples one reader of them produces."""
        return CorpusStream(
            paths.samples,
            paths.labels,
            builder,
            shards=shards,
            rank=rank,
            world_size=world_size,
        )

    def collator(
        self,
        context_dropout: float = 0.3,
        max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        seed: int = 0,
    ) -> Collator:
        """The collator a run of these settings pads its batches with."""
        return Collator(
            self.tokenizer,
            context_dropout=context_dropout,
            max_context_tokens=max_context_tokens,
            seed=seed,
        )


def held_out_examples(
    paths: RunPaths,
    builder: SampleBuilder,
    slices: Slices,
) -> list[TrainingExample]:
    """Build the evaluation slice once, so both passes score the same examples.

    Drawn evenly across the held-out shards rather than in file order. The shards
    are chosen one per source so that the number this produces speaks for the
    corpus, and any one of them holds several times the whole quota -- read in
    order, the "held-out accuracy" would be one source's accuracy with six
    sources' names on it, and whichever source sorted first would silently decide
    what the run reports.
    """
    if not slices.held_out:
        return []
    quota = slices.max_held_out_examples
    per_shard = max(1, quota // len(slices.held_out))
    examples: list[TrainingExample] = []
    for shard in slices.held_out:
        stream = CorpusStream(paths.samples, paths.labels, builder, shards=[shard])
        for taken, example in enumerate(stream, start=1):
            examples.append(example)
            if taken >= per_shard or len(examples) >= quota:
                break
        if len(examples) >= quota:
            break
    return examples


def route_a(
    paths: RunPaths,
    slices: Slices,
    training: TrainingConfig,
    route: RouteAConfig | None = None,
    augmentation: Augmentation | None = None,
    context_dropout: float = 0.3,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    resume: Path | None = None,
) -> RunResult:
    """Train route A over *slices.train* and score *slices.held_out* both ways.

    With *resume*, this segment continues the run that wrote that checkpoint
    rather than starting one: same weights, same optimiser, same place in the
    corpus. The checkpoint carries the config it was written under and refuses
    anything else, so the chain of kernels is one run or it is an error.
    """
    route = route or RouteAConfig()
    world = Distributed.from_environment()
    vocabularies = Vocabularies.load(paths.char_table, route.base_model)
    spans, tokenizer, lexicon = vocabularies.spans, vocabularies.tokenizer, vocabularies.lexicon
    model = RouteAModel.from_pretrained(route, lexicon, spans, vocabularies.vocabulary)

    builder = vocabularies.builder(augmentation, training.seed)
    stream = vocabularies.stream(paths, slices.train, builder, world.rank, world.world_size)
    collator = vocabularies.collator(context_dropout, max_context_tokens, training.seed)

    paths.out.mkdir(parents=True, exist_ok=True)
    with MetricLog(paths.out / "metrics.jsonl") as provenance:
        if world.is_main:
            provenance.write(
                event="provenance",
                commit=git_commit(),
                corpus=corpus_digest(shard_paths(paths.samples, "*")),
                labels=corpus_digest(shard_paths(paths.labels, "*")),
                label_source="g2pw",
                reading_arbitration=vocabularies.arbitration.digest,
                train_shards=list(slices.train),
                held_out_shards=list(slices.held_out),
                augmentation=asdict(augmentation or Augmentation()),
                route_a=asdict(route),
                emittable_characters=lexicon.size,
                typed_spans=lexicon.spans,
                resume=str(resume) if resume is not None else None,
            )

    segment = train(model, stream, collator, training, paths.out, world, resume)
    losses = _losses(segment.metrics)
    if not segment.finished:
        return paused_run(segment, losses, model.gates(), builder.counts.as_dict(), world)
    evaluation = held_out_examples(
        paths, vocabularies.builder(augmentation, training.seed + 1), slices
    )
    device = world.device
    with_context = evaluate(model, evaluation, tokenizer, device, training.token_budget, True)
    without_context = evaluate(model, evaluation, tokenizer, device, training.token_budget, False)
    result = RunResult(
        metrics=segment.metrics,
        first_loss=losses[0],
        last_loss=losses[-1],
        steps=len(losses),
        step=segment.step,
        finished=True,
        with_context=with_context,
        without_context=without_context,
        gates=model.gates(),
    )
    if world.is_main:
        with MetricLog(segment.metrics) as summary:
            summary.write(
                event="summary",
                finished=True,
                step=result.step,
                first_loss=result.first_loss,
                last_loss=result.last_loss,
                logged_steps=result.steps,
                held_out_examples=len(evaluation),
                accuracy_with_context=with_context.rate,
                accuracy_without_context=without_context.rate,
                scored_characters=with_context.scored,
                gates=result.gates,
                build_counts=builder.counts.as_dict(),
            )
    log.info(
        "route A run finished",
        first_loss=round(result.first_loss, 4),
        last_loss=round(result.last_loss, 4),
        accuracy_with_context=round(with_context.rate, 4),
        accuracy_without_context=round(without_context.rate, 4),
        held_out=len(evaluation),
        gates=[round(gate, 5) for gate in result.gates],
    )
    return result


def paused_run(
    segment: Segment,
    losses: list[float],
    gates: list[float],
    build_counts: dict[str, int],
    world: Distributed,
) -> RunResult:
    """The result of a segment that spent its wall budget before the last step.

    The held-out slice is not scored: scoring is minutes of forward passes, and
    the number it would produce describes a model that is halfway through its
    schedule, at a learning rate the cosine has not brought down yet. It is not a
    cheaper version of the run's accuracy, it is a different quantity, and
    reporting it beside finished runs is how the two get compared.
    """
    if world.is_main:
        with MetricLog(segment.metrics) as summary:
            summary.write(
                event="summary",
                finished=False,
                step=segment.step,
                first_loss=losses[0],
                last_loss=losses[-1],
                logged_steps=len(losses),
                gates=gates,
                build_counts=build_counts,
            )
    log.info(
        "route A segment paused on its wall budget",
        step=segment.step,
        first_loss=round(losses[0], 4),
        last_loss=round(losses[-1], 4),
        gates=[round(gate, 5) for gate in gates],
    )
    return RunResult(
        metrics=segment.metrics,
        first_loss=losses[0],
        last_loss=losses[-1],
        steps=len(losses),
        step=segment.step,
        finished=False,
        with_context=None,
        without_context=None,
        gates=gates,
    )


def _losses(metrics: Path) -> list[float]:
    """Every step loss the metrics file recorded, in order."""
    import json

    losses = [
        float(record["loss"])
        for record in (json.loads(line) for line in metrics.read_text().splitlines())
        if record.get("event") == "step"
    ]
    if not losses:
        raise ValueError(f"{metrics} recorded no training step")
    return losses


def describe_device() -> str:
    """What the run is about to happen on, for the log's first line."""
    if not torch.cuda.is_available():
        return "cpu"
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return ", ".join(names)
