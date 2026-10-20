"""Turning the prepared corpus into (typed, target, context) training examples.

A prepared sample holds a sentence and what was on screen before it; a label
shard holds that sentence's per-character readings. What the model trains on is
neither: it is what somebody would have *pressed* to produce the sentence, and
people do not press the same keys twice.

So the typing style is sampled per example rather than fixed. 55% of the time the
whole sentence is typed out; 25% of the time it is an abbreviation pass, each
syllable independently dropping to its initial with probability 0.7 -- which is
why an "abbreviated" example still carries some full syllables, the way a real
one does; 20% of the time it is mixed, full at the start and abbreviated from a
random point on, which is the shape of a person who starts careful and gets lazy.
The typed string never carries tone, because no pinyin keyboard has a tone key.

The style is re-sampled every epoch, so one sentence is a different training
example each time it comes round. That is done by seeding a per-example
generator from ``(seed, epoch, sample id)`` rather than by drawing from a shared
stream: the stream's state would depend on how many workers were reading and in
what order, and the whole point of logging an augmentation seed is that the run
can be reproduced.

Examples that cannot be built are counted, never patched. A sentence with a
non-Han character has no syllable behind that character; a character outside the
model's vocabulary cannot be emitted; a reading that no typed span admits would
train the model towards a candidate it is masked out of. Each of those is a fact
about the data, and :class:`BuildCounts` is where it shows up.

The one thing that *is* rewritten happens before any of that: the annotator and
the character table follow different standards on a few dozen characters, and
:class:`~mlime.train.arbitration.ReadingArbitration` settles each of them by a
decision recorded in a data file. It is applied to the label as it arrives, so
every gate downstream tests the reading the run actually trains under.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from fractions import Fraction
from itertools import accumulate
from pathlib import Path
from typing import Protocol

import polars as pl
import torch
from torch.utils.data import IterableDataset, get_worker_info

from mlime.data.corpus import Sample
from mlime.data.shards import shard_paths
from mlime.data.text import all_han
from mlime.logging import log
from mlime.train.arbitration import ReadingArbitration
from mlime.train.labels import keyboard_form
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.spans import SpanVocab, initial
from mlime.typo import NoiseModel

#: Loss is not taken at padding or at the two sentinel positions.
IGNORE_INDEX = -100

#: The letter id of a padding slot in ``span_letters``: ``a``..``z`` are 0..25,
#: so 26 marks a column no letter occupies.
LETTER_PAD = 26

#: The most letters a span's encoding reads. The longest syllable is six and a
#: corruption can stretch a bound by at most the four edits the model can draw,
#: so twelve never truncates real input.
MAX_SPAN_LETTERS = 12

#: The three typing styles, in the order their probabilities are given.
STYLES = ("full", "abbreviated", "mixed")

#: How wide the context tower's input may be, sentinels included. Sixty-four
#: covers five sixths of the corpus's contexts whole and bounds the cost of the
#: rest; the tower is quadratic in this number, and at 128 it was three times the
#: fill tower's entire cost.
DEFAULT_CONTEXT_TOKENS = 64


def context_tail(context: str, max_tokens: int) -> str:
    """The end of *context*, which is the part the cursor is next to.

    A tokenizer truncates from the right, and the context runs *up to* the
    sentence being typed -- so right truncation throws away the words that
    determine it and keeps the ones furthest from it. Cutting from the left here
    means the model always sees the characters immediately before the target.
    Two positions are left for the sentinels the tokenizer adds.
    """
    if max_tokens < 3:
        raise ValueError(
            f"a context needs room for two sentinels and a character, got {max_tokens}"
        )
    return context[-(max_tokens - 2) :]


@dataclass(frozen=True)
class Augmentation:
    """How a sentence's readings become keystrokes.

    The defaults are milestone 3's plan verbatim. They are a dataclass rather
    than constants so an ablation is a different value, not a different branch.
    ``typos`` is a switch, not a rate: on, the keystrokes the styles produce
    pass through the shared noise model, whose own per-sentence coin decides
    each example.
    """

    full: float = 0.55
    abbreviated: float = 0.25
    mixed: float = 0.20
    abbreviate_syllable: float = 0.7
    typos: bool = False

    def __post_init__(self) -> None:
        total = self.full + self.abbreviated + self.mixed
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"the style probabilities must sum to 1, got {total}")
        for name, value in vars(self).items():
            if isinstance(value, bool):
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability, got {value}")

    def style(self, rng: random.Random) -> str:
        """Draw one typing style."""
        draw = rng.random()
        if draw < self.full:
            return "full"
        if draw < self.full + self.abbreviated:
            return "abbreviated"
        return "mixed"


def type_syllables(
    syllables: Sequence[str], rng: random.Random, augmentation: Augmentation
) -> tuple[str, ...]:
    """The spans somebody would have pressed for *syllables*, one style sampled.

    A mixed example needs both halves to exist, so the cut falls strictly inside
    the sentence; a one-syllable sentence has nowhere to put a cut and is
    abbreviated whole, which is what "get lazy after the first syllable" degrades
    to when there is only one.
    """
    if not syllables:
        raise ValueError("a sentence with no syllables has nothing to type")
    style = augmentation.style(rng)
    if style == "full":
        return tuple(syllables)
    if style == "abbreviated":
        return tuple(
            initial(syllable) if rng.random() < augmentation.abbreviate_syllable else syllable
            for syllable in syllables
        )
    if len(syllables) == 1:
        return (initial(syllables[0]),)
    cut = rng.randrange(1, len(syllables))
    return (*syllables[:cut], *(initial(syllable) for syllable in syllables[cut:]))


def example_rng(seed: int, epoch: int, sample_id: str) -> random.Random:
    """A generator determined by *(seed, epoch, sample_id)* and nothing else.

    Python's ``hash`` is salted per process, so the digest is taken explicitly:
    two workers, two machines and two runs must draw the same keystrokes for the
    same example.
    """
    material = f"{seed}:{epoch}:{sample_id}".encode()
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


@dataclass
class BuildCounts:
    """Why examples were kept or dropped, over one pass of the corpus."""

    kept: int = 0
    unlabelled: int = 0
    refused: int = 0
    not_all_han: int = 0
    length_mismatch: int = 0
    unknown_span: int = 0
    unemittable_character: int = 0
    target_not_admitted: int = 0
    #: Label positions the arbitration table rewrote. Positions, not sentences:
    #: one sentence can carry several, and an arbitrated sentence still lands
    #: under whichever outcome it reaches. So this one is not an outcome and is
    #: kept out of :attr:`seen`, which counts samples.
    reading_arbitrated: int = field(default=0, metadata={"outcome": False})

    @property
    def seen(self) -> int:
        """Every sample the builder was offered."""
        return sum(
            getattr(self, counter.name)
            for counter in fields(self)
            if counter.metadata.get("outcome", True)
        )

    def as_dict(self) -> dict[str, int]:
        """The counters, plus the total, ready for a metrics record."""
        return {**vars(self), "seen": self.seen}


@dataclass(frozen=True)
class TrainingExample:
    """One built example: what was pressed, what it should decode to, what preceded it."""

    id: str
    spans: tuple[str, ...]
    span_ids: tuple[int, ...]
    targets: tuple[int, ...]
    context: str | None

    def __post_init__(self) -> None:
        lengths = {len(self.spans), len(self.span_ids), len(self.targets)}
        if len(lengths) != 1:
            raise ValueError(f"example {self.id} is not aligned: lengths {sorted(lengths)}")
        if not self.spans:
            raise ValueError(f"example {self.id} is empty")

    def __len__(self) -> int:
        return len(self.spans)


class SampleBuilder:
    """Builds one training example from a sample and its readings.

    The arbitration table is passed in rather than loaded here, because it is one
    of the things a run is made of -- like the lexicon and the span inventory --
    and a builder that loaded its own could differ from the one the provenance
    record describes. *noise* is the shared typo model the ``typos`` switch
    expects: a builder told to corrupt without it is a wiring bug, not a quiet
    clean pass, and so is a noise model handed in beside the switch off.
    """

    def __init__(
        self,
        lexicon: Lexicon,
        spans: SpanVocab,
        arbitration: ReadingArbitration,
        augmentation: Augmentation | None = None,
        seed: int = 0,
        noise: NoiseModel | None = None,
        candidates: CandidateSpace | None = None,
    ):
        self.lexicon = lexicon
        self.spans = spans
        self.arbitration = arbitration
        self.augmentation = augmentation or Augmentation()
        has_noise = noise is not None
        has_table = candidates is not None and candidates.table is not None
        if has_noise != self.augmentation.typos or has_table != self.augmentation.typos:
            raise ValueError(
                "the typo noise model, the widened candidate space and "
                "augmentation.typos must come together; "
                f"got noise={has_noise}, table={has_table}, typos={self.augmentation.typos}"
            )
        self.noise = noise
        self.candidates = candidates or CandidateSpace(spans, lexicon.candidate_mask)
        self.seed = seed
        self.counts = BuildCounts()
        #: The reading each (character, annotator syllable) pair settles to,
        #: and whether the arbitration rewrote it. The pairs number in the tens
        #: of thousands; folding one afresh -- a regex, three string copies and
        #: the arbitration lookup -- was the largest part of a sentence's cost.
        self._settled: dict[tuple[str, str], tuple[str, bool]] = {}

    def build(
        self, sample: Sample, syllables: Sequence[str] | None, epoch: int
    ) -> TrainingExample | None:
        """The example for *sample* at *epoch*, or ``None`` with a counter raised."""
        if syllables is None:
            self.counts.unlabelled += 1
            return None
        text = sample.text
        if not all_han(text):
            self.counts.not_all_han += 1
            return None
        if len(text) != len(syllables):
            self.counts.length_mismatch += 1
            return None
        known = self._settled
        readings = []
        arbitrated = 0
        for pair in zip(text, syllables, strict=True):
            settled = known.get(pair)
            if settled is None:
                label = keyboard_form(pair[1])
                reading = self.arbitration.reading(pair[0], label)
                settled = known[pair] = (reading, reading != label)
            readings.append(settled[0])
            arbitrated += settled[1]
        self.counts.reading_arbitrated += arbitrated
        if not self.spans.covers(readings):
            self.counts.unknown_span += 1
            return None
        targets = self.lexicon.indices(text)
        if targets is None:
            self.counts.unemittable_character += 1
            return None

        rng = example_rng(self.seed, epoch, sample.id)
        spans = type_syllables(tuple(readings), rng, self.augmentation)
        if self.noise is not None:
            # The styles choose *which* keys the typist meant to press and the
            # noise model then slips them -- corrupting before the abbreviation
            # would let the initial of a typoed syllable erase the slip.
            spans = self.noise.maybe_corrupt(spans, rng).spans
        if not self.candidates.admits_all(spans, targets):
            self.counts.target_not_admitted += 1
            return None
        self.counts.kept += 1
        return TrainingExample(
            id=sample.id,
            spans=spans,
            span_ids=tuple(self.spans.id_or_unknown(span) for span in spans),
            targets=targets,
            context=sample.context,
        )


def read_pair(
    sample_shard: Path, labels_dir: Path, first: int = 0, step: int = 1
) -> Iterator[tuple[Sample, list[str] | None]]:
    """Stream rows ``first, first + step, ...`` of one sample shard beside its labels.

    The two are joined on id rather than on row order, because a label shard is
    written from its sample shard but nothing in the format promises the order
    survived; the join is a dictionary over one shard, not over the corpus.
    """
    label_shard = labels_dir / sample_shard.name
    if not label_shard.is_file():
        raise FileNotFoundError(
            f"no label shard {label_shard} for {sample_shard}; run `mlime train labels` first"
        )
    labels = pl.read_parquet(label_shard)
    readings: dict[str, list[str] | None] = dict(
        zip(labels["id"].to_list(), labels["syllables"].to_list(), strict=True)
    )
    for row in pl.read_parquet(sample_shard).gather_every(step, offset=first).iter_rows(named=True):
        sample = Sample.from_row(row)
        yield sample, readings.get(sample.id)


def shard_rows(shard: Path) -> int:
    """The rows *shard* holds, from its metadata."""
    return int(pl.scan_parquet(shard).select(pl.len()).collect().item())


def interleave(shards: Sequence[Path]) -> list[Path]:
    """*shards* in an order that spreads every source evenly over the whole list.

    A source is a shard's name without its trailing number (``news-rest-00042``
    belongs to ``news-rest``). The *i*-th of a source's *n* shards, in name
    order, sits at ``(2i + 1) / 2n`` of the way through, so no stretch of an
    epoch reads one source alone. v1 read its shards in name order, and its last
    near-zero-rate steps saw only the corpus's hardest tail (#133).
    """
    sources: dict[str, list[Path]] = {}
    for shard in sorted(shards, key=lambda shard: shard.name):
        sources.setdefault(shard.stem.rsplit("-", 1)[0], []).append(shard)
    keyed = [
        (Fraction(2 * index + 1, 2 * len(members)), source, shard.name, shard)
        for source, members in sources.items()
        for index, shard in enumerate(members)
    ]
    return [shard for *_, shard in sorted(keyed)]


def first_row(offset: int, reader: int, readers: int) -> int:
    """The first row of a shard starting at global row *offset* that *reader* takes.

    The readers stride the shards' concatenation: reader *i* takes the global
    rows ``i, i + readers, ...``, so within a shard it takes every
    *readers*-th row from this one on. A result past the shard's end means the
    reader takes nothing there.
    """
    return (reader - offset) % readers


class CorpusStream(IterableDataset[TrainingExample]):
    """The corpus as an epoch-reaugmented stream of examples.

    Every reader -- a DDP rank's dataloader worker -- walks the shards in
    :func:`interleave` order and takes every *readers*-th row of their
    concatenation, starting at its own index. So each row is read by exactly
    one reader, the readers' example counts differ by at most one, and each
    reader's stream mixes every source throughout the epoch. Dealing whole
    shards left #130's two ranks 4.5% apart over 22 shards, and DDP ends the
    epoch at the shorter rank. Which reader gets which row does not change what
    an example looks like -- that depends only on the seed, the epoch and the
    sample id.
    """

    def __init__(
        self,
        samples_dir: Path,
        labels_dir: Path,
        builder: SampleBuilder,
        shards: Sequence[str] | None = None,
        epoch: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not a member of a world of {world_size}")
        self.builder = builder
        self.epoch = epoch
        self.rank = rank
        self.world_size = world_size
        self.labels_dir = labels_dir
        available = shard_paths(samples_dir, "*")
        if not available:
            raise FileNotFoundError(f"no sample shards under {samples_dir}")
        if shards is None:
            self.shards = available
        else:
            wanted = set(shards)
            self.shards = [path for path in available if path.name in wanted]
            missing = wanted - {path.name for path in self.shards}
            if missing:
                raise FileNotFoundError(f"no such shards under {samples_dir}: {sorted(missing)}")
        self.shards = interleave(self.shards)
        self.rows = [shard_rows(shard) for shard in self.shards]

    @property
    def offsets(self) -> list[int]:
        """The global row each shard starts at, in reading order."""
        return list(accumulate(self.rows[:-1], initial=0))

    def set_epoch(self, epoch: int) -> None:
        """Re-augment from the next pass on."""
        self.epoch = epoch

    def _reader(self) -> tuple[int, int]:
        """This reader's index and the number of readers, from rank and dataloader worker."""
        info = get_worker_info()
        workers = info.num_workers if info is not None else 1
        worker = info.id if info is not None else 0
        return self.rank * workers + worker, self.world_size * workers

    def __iter__(self) -> Iterator[TrainingExample]:
        index, readers = self._reader()
        for shard, rows, offset in zip(self.shards, self.rows, self.offsets, strict=True):
            first = first_row(offset, index, readers)
            if first >= rows:
                continue
            for sample, syllables in read_pair(shard, self.labels_dir, first, readers):
                example = self.builder.build(sample, syllables, self.epoch)
                if example is not None:
                    yield example
            log.debug("shard streamed", shard=shard.name, **self.builder.counts.as_dict())


class BaseTokenizer(Protocol):
    """What the collator needs of the base model's tokenizer."""

    cls_token_id: int
    sep_token_id: int
    pad_token_id: int
    mask_token_id: int

    def __call__(
        self,
        text: list[str],
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> Mapping[str, list[list[int]]]:
        """Encode a batch of strings into padded ``input_ids`` and ``attention_mask`` rows.

        Rows rather than tensors: transformers' own tensor conversion walks
        every token in Python before it builds the tensor, and the collator's
        one ``torch.tensor`` over the rows is the same int64 tensor at half
        the batch's tokenizer time.
        """
        ...


@dataclass(frozen=True)
class Batch:
    """One collated step.

    ``input_ids`` carries the sentinels and one ``[MASK]`` per typed span;
    ``span_ids`` carries the span at those same positions and is meaningless
    elsewhere, which ``span_positions`` marks. ``targets`` is in emission-index
    space, not vocabulary space, and is ``IGNORE_INDEX`` wherever no loss is due.
    ``scored`` is ``targets != IGNORE_INDEX`` as flat indices into the ``[B, W]``
    layout -- the ``nonzero`` a step would otherwise run on the device, done here
    on the host where finding positions costs no synchronisation.

    ``candidate_ids [B, W, K]`` is each position's own candidate list in
    emission space -- the lexicon row, or the corrections' union for a typoed
    span -- padded by ``candidate_counts``; carrying it is what lets a
    correction-widened set vary per position without a new mask row. ``span_letters
    [B, W, L]`` is the raw keys pressed, ``a``-``z`` as 0-25 and ``LETTER_PAD``
    past each span's length, which is all the letter encoder sees of a typo.

    The context fields hold one row per *context*, not per fill row:
    ``context_rows [B]`` names the context row each fill row reads. Every
    example owns its own context in a plain batch, so there it is ``arange``;
    a ranking batch types one example several ways and runs the context tower
    once for all of them.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    span_ids: torch.Tensor
    span_positions: torch.Tensor
    targets: torch.Tensor
    scored: torch.Tensor
    context_ids: torch.Tensor
    context_mask: torch.Tensor
    has_context: torch.Tensor
    context_rows: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_counts: torch.Tensor
    span_letters: torch.Tensor
    ids: tuple[str, ...] = field(default=())

    @property
    def size(self) -> int:
        """Fill rows in the batch: examples, unless a ranking batch types them several ways."""
        return int(self.input_ids.shape[0])

    @property
    def tokens(self) -> int:
        """Padded fill-tower positions the step costs."""
        return int(self.input_ids.shape[0] * self.input_ids.shape[1])

    @property
    def context_tokens(self) -> int:
        """Padded context-tower positions the step costs.

        Counted separately and reported separately because it is usually the
        larger of the two: a target is nine characters on average and the text
        before it is thirty-two, so the tower nobody thinks about is the one
        filling the card.
        """
        return int(self.context_ids.shape[0] * self.context_ids.shape[1])

    def to(self, device: torch.device) -> Batch:
        """The same batch on *device*; the ids stay where they are."""
        return Batch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            span_ids=self.span_ids.to(device, non_blocking=True),
            span_positions=self.span_positions.to(device, non_blocking=True),
            targets=self.targets.to(device, non_blocking=True),
            scored=self.scored.to(device, non_blocking=True),
            context_ids=self.context_ids.to(device, non_blocking=True),
            context_mask=self.context_mask.to(device, non_blocking=True),
            has_context=self.has_context.to(device, non_blocking=True),
            context_rows=self.context_rows.to(device, non_blocking=True),
            candidate_ids=self.candidate_ids.to(device, non_blocking=True),
            candidate_counts=self.candidate_counts.to(device, non_blocking=True),
            span_letters=self.span_letters.to(device, non_blocking=True),
            ids=self.ids,
        )

    def pin_memory(self) -> Batch:
        """The same batch in page-locked host memory, so :meth:`to` copies asynchronously.

        From pageable memory a ``non_blocking`` copy is not: the driver stages
        it through a bounce buffer whose DMA queues behind the stream's work,
        so the host waits for the device to drain before it can launch the
        next forward -- a synchronisation per micro-batch, hidden in ``h2d``.
        """
        return Batch(
            **{
                field.name: getattr(self, field.name).pin_memory()
                for field in fields(self)
                if field.name != "ids"
            },
            ids=self.ids,
        )


class StepBatch(Protocol):
    """What the training loop reads of a collated step, whatever the model consumes."""

    @property
    def scored(self) -> torch.Tensor:
        """The flat indices of the positions the step's per-position loss scores."""
        ...

    @property
    def size(self) -> int:
        """Examples in the step."""
        ...

    @property
    def tokens(self) -> int:
        """Padded fill-tower positions."""
        ...

    @property
    def context_tokens(self) -> int:
        """Padded context-tower positions."""
        ...

    def to(self, device: torch.device) -> StepBatch:
        """The same step on *device*."""
        ...

    def pin_memory(self) -> StepBatch:
        """The same step in page-locked host memory."""
        ...


class BatchCollator(Protocol):
    """What the training loop needs of a collator: a batch, its generator, its budget."""

    @property
    def rng(self) -> random.Random:
        """The generator whose state a checkpoint carries."""
        ...

    @property
    def max_context_tokens(self) -> int:
        """The context budget batching is grouped under."""
        ...

    def rows(self, example: TrainingExample) -> int:
        """Rows at the fill width *example* costs a step, which batching weighs it by."""
        ...

    def __call__(self, examples: Sequence[TrainingExample]) -> StepBatch:
        """Collate one group of examples."""
        ...


class Collator:
    """Pads a list of examples into a :class:`Batch`, dropping context as it goes.

    Context is dropped per example rather than per batch, because the model has
    to work both ways at inference and a whole-batch drop would correlate the
    decision with everything else in the step.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        candidates: CandidateSpace,
        context_dropout: float = 0.3,
        max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        seed: int = 0,
    ):
        if not 0.0 <= context_dropout <= 1.0:
            raise ValueError(f"context_dropout must be a probability, got {context_dropout}")
        self.tokenizer = tokenizer
        #: The per-position candidate resolver -- the lexicon mask under clean
        #: input, the corrections' union under the typo model. The batch is
        #: where the examples become tensors, so it is also where an example
        #: whose target its span does not admit fails, on the host, instead of
        #: inside a training step on the device.
        self.candidates = candidates
        self.context_dropout = context_dropout
        self.max_context_tokens = max_context_tokens
        self.rng = random.Random(seed)
        #: Each span's candidate row and padded letters, built on first sight.
        #: The batch gathers copies out of them; nothing it holds aliases them.
        self._candidates: dict[str, torch.Tensor] = {}
        self._letters: dict[str, torch.Tensor] = {}

    def rows(self, example: TrainingExample) -> int:
        """One: an example is one row of the fill tower."""
        return 1

    def _candidate_row(self, span: str) -> torch.Tensor:
        """The emission ids *span* admits, as the row a batch gathers."""
        row = self._candidates.get(span)
        if row is None:
            row = self._candidates[span] = torch.tensor(
                self.candidates.resolve(span), dtype=torch.long
            )
        return row

    def _letter_row(self, span: str, example_id: str) -> None:
        """Check *span*'s keys once and keep them padded to the letter width."""
        if span in self._letters:
            return
        if len(span) > MAX_SPAN_LETTERS:
            raise ValueError(
                f"example {example_id} holds a {len(span)}-letter span {span!r}; "
                f"the letter encoding reads at most {MAX_SPAN_LETTERS}"
            )
        keys = [letter_id(letter) for letter in span]
        self._letters[span] = torch.tensor(
            keys + [LETTER_PAD] * (MAX_SPAN_LETTERS - len(keys)), dtype=torch.long
        )

    def __call__(self, examples: Sequence[TrainingExample]) -> Batch:
        """Collate *examples*, keeping each one's context with probability 1-p."""
        return self.collate(
            examples, [example.context for example in examples], range(len(examples))
        )

    def collate(
        self,
        examples: Sequence[TrainingExample],
        contexts: Sequence[str | None],
        context_rows: Sequence[int],
    ) -> Batch:
        """Collate fill rows *examples* over *contexts*, row *i* reading ``context_rows[i]``.

        The dropout coin is thrown once per context, so the rows that share one
        are all typed with it or all without it.
        """
        if not examples:
            raise ValueError("cannot collate an empty batch")
        if len(context_rows) != len(examples):
            raise ValueError(
                f"{len(examples)} fill rows but {len(context_rows)} context assignments"
            )
        lengths = [len(example) for example in examples]
        width = max(lengths) + 2
        size = len(examples)
        # Each distinct span of the batch is looked up once -- its candidate
        # row and its letters are properties of the span, not of the position
        # -- and the positions then gather them in one indexing op apiece. A
        # batch is some ninety rows and eight hundred positions; building it
        # a position at a time was ~1,700 small tensors, all of it Python
        # holding the GIL the training step needs to launch its kernels.
        slots: dict[str, int] = {}
        slot_of_position = [
            slots.setdefault(span, len(slots)) for example in examples for span in example.spans
        ]
        listed = [self._candidate_row(span) for span in slots]
        for example in examples:
            for span in example.spans:
                self._letter_row(span, example.id)
            for span, target in zip(example.spans, example.targets, strict=True):
                if target != IGNORE_INDEX and not self.candidates.admits(span, int(target)):
                    raise ValueError(f"example {example.id} has a target its span does not admit")

        # Row-major over the typed columns ``1..length``: a boolean index walks
        # the positions in exactly the order the examples list them.
        span_lengths = torch.tensor(lengths, dtype=torch.long)
        columns = torch.arange(width)
        span_positions = (columns >= 1) & (columns <= span_lengths[:, None])
        input_ids = torch.full((size, width), self.tokenizer.pad_token_id, dtype=torch.long)
        input_ids[span_positions] = self.tokenizer.mask_token_id
        input_ids[:, 0] = self.tokenizer.cls_token_id
        input_ids[torch.arange(size), span_lengths + 1] = self.tokenizer.sep_token_id
        attention_mask = (columns < span_lengths[:, None] + 2).long()
        span_ids = torch.zeros((size, width), dtype=torch.long)
        span_ids[span_positions] = torch.tensor(
            [span_id for example in examples for span_id in example.span_ids], dtype=torch.long
        )
        targets = torch.full((size, width), IGNORE_INDEX, dtype=torch.long)
        targets[span_positions] = torch.tensor(
            [target for example in examples for target in example.targets], dtype=torch.long
        )

        slot_index = torch.tensor(slot_of_position, dtype=torch.long)
        table = torch.nn.utils.rnn.pad_sequence(listed, batch_first=True, padding_value=0)
        candidate_ids = torch.zeros((size, width, table.shape[1]), dtype=torch.long)
        candidate_ids[span_positions] = table[slot_index]
        candidate_counts = torch.zeros((size, width), dtype=torch.long)
        candidate_counts[span_positions] = torch.tensor(
            [row.numel() for row in listed], dtype=torch.long
        )[slot_index]
        span_letters = torch.full((size, width, MAX_SPAN_LETTERS), LETTER_PAD, dtype=torch.long)
        span_letters[span_positions] = torch.stack([self._letters[span] for span in slots])[
            slot_index
        ]

        kept: list[str] = []
        has_context = torch.zeros(len(contexts), dtype=torch.float)
        for index, context in enumerate(contexts):
            keep = context is not None and self.rng.random() >= self.context_dropout
            kept.append(context_tail(context, self.max_context_tokens) if keep and context else "")
            has_context[index] = 1.0 if keep else 0.0
        encoded = self.tokenizer(
            kept,
            padding=True,
            truncation=True,
            max_length=self.max_context_tokens,
        )
        # The device code indexes the scored positions rather than finding
        # them: the nonzero and the check that they sit inside the span columns
        # run here, on the host, so the training step stays asynchronous.
        scored_positions = (targets != IGNORE_INDEX).reshape(-1).nonzero(as_tuple=True)[0]
        columns = scored_positions % width
        if bool(((columns == 0) | (columns == width - 1)).any()):
            raise ValueError("a scored position sits outside the span columns")
        return Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_ids=span_ids,
            span_positions=span_positions,
            targets=targets,
            scored=scored_positions,
            context_ids=torch.tensor(encoded["input_ids"], dtype=torch.long),
            context_mask=torch.tensor(encoded["attention_mask"], dtype=torch.long),
            has_context=has_context,
            context_rows=torch.tensor(context_rows, dtype=torch.long),
            candidate_ids=candidate_ids,
            candidate_counts=candidate_counts,
            span_letters=span_letters,
            ids=tuple(example.id for example in examples),
        )


def letter_id(letter: str) -> int:
    """The letter-encoder id of one pressed key; anything but a-z is a bug."""
    value = ord(letter) - ord("a")
    if not 0 <= value < LETTER_PAD:
        raise ValueError(f"{letter!r} is not a keystroke letter")
    return value


def token_budget_batches(
    examples: Iterator[TrainingExample],
    budget: int,
    max_context_tokens: int = 0,
    rows: Callable[[TrainingExample], int] | None = None,
) -> Iterator[list[TrainingExample]]:
    """Group *examples* so one step's padded cost stays under *budget*.

    The cost is two rectangles and the budget covers their sum. Both are padded,
    not summed: a step pays for the rectangle each tower occupies, so one long
    sentence shrinks the batch around it and one long context shrinks it again.

    Bounding only the fill tower -- which is what this did first -- bounds the
    smaller of the two. Targets average nine characters and the text before them
    averages thirty-two, so the context tower is most of the step; and a batch of
    unusually short sentences is an unusually *large* batch, whose context
    rectangle is then large in both dimensions. That is not a hypothetical: it is
    what took a two-rank run out of memory ten steps in, on the same budget a
    one-rank run had held for two hundred.

    A *max_context_tokens* of zero means the contexts are not encoded at all,
    which is the case wherever only the fill tower is under test.

    *rows* is how many rows at the fill width an example costs; one each when
    it is not given. The ranking loss (#116) types an example every way its
    keystrokes read and decodes several sentences on them, so one example
    there is a dozen rows, and how many varies with the keystrokes: a budget
    over its examples alone would let a batch of long full-pinyin sentences run
    to twice the rows of the batch beside it.
    """
    packer = BudgetPacker(budget)
    batch: list[TrainingExample] = []
    for example in examples:
        fill, context = example_shape(example, max_context_tokens)
        if packer.add(fill, context, 1 if rows is None else rows(example)):
            yield batch
            batch = []
        batch.append(example)
    if batch:
        yield batch


class BudgetPacker:
    """:func:`token_budget_batches`' grouping, one example's shape at a time.

    The grouping is greedy and sequential -- whether an example opens a new
    batch depends on every example before it in the batch -- but it reads
    nothing of an example except its two widths and its rows. Kept apart from
    the examples, it is what the batch census replays over shapes measured in
    other processes, so the census and the loop group with one implementation.
    """

    def __init__(self, budget: int) -> None:
        if budget <= 0:
            raise ValueError(f"the token budget must be positive, got {budget}")
        self.budget = budget
        #: Batches closed so far; :attr:`batches` adds the open one.
        self.closed = 0
        self._fill = self._context = self._rows = self._examples = 0

    def add(self, fill: int, context: int, rows: int) -> bool:
        """Place one example; true when it opens a new batch, closing the one before.

        *fill* and *context* are its padded widths (:func:`example_shape`),
        *rows* how many rows at the fill width it costs.
        """
        wide = max(self._fill, fill)
        deep = max(self._context, context)
        opens = (
            self._examples > 0
            and wide * (self._rows + rows) + deep * (self._examples + 1) > self.budget
        )
        if opens:
            self.closed += 1
            self._rows = self._examples = 0
            wide, deep = fill, context
        self._examples += 1
        self._fill, self._context = wide, deep
        self._rows += rows
        return opens

    @property
    def batches(self) -> int:
        """Every batch so far, the open one included."""
        return self.closed + (self._examples > 0)


def example_shape(example: TrainingExample, max_context_tokens: int) -> tuple[int, int]:
    """The fill and context widths *example* pads a batch to, sentinels included."""
    return len(example) + 2, _context_width(example, max_context_tokens)


def _context_width(example: TrainingExample, max_context_tokens: int) -> int:
    """How many context-tower positions *example* would occupy on its own."""
    if max_context_tokens <= 0:
        return 0
    return min(len(example.context or ""), max_context_tokens - 2) + 2
