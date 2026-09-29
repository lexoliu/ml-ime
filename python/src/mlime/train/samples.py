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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol

import polars as pl
import torch
from torch.utils.data import IterableDataset, get_worker_info

from mlime.data.corpus import Sample
from mlime.data.shards import shard_paths
from mlime.data.text import HAN
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

    def build(
        self, sample: Sample, syllables: Sequence[str] | None, epoch: int
    ) -> TrainingExample | None:
        """The example for *sample* at *epoch*, or ``None`` with a counter raised."""
        if syllables is None:
            self.counts.unlabelled += 1
            return None
        characters = tuple(sample.text)
        if not all(HAN.match(character) for character in characters):
            self.counts.not_all_han += 1
            return None
        if len(characters) != len(syllables):
            self.counts.length_mismatch += 1
            return None
        labelled = tuple(keyboard_form(syllable) for syllable in syllables)
        readings = tuple(
            self.arbitration.reading(character, label)
            for character, label in zip(characters, labelled, strict=True)
        )
        self.counts.reading_arbitrated += sum(
            arbitrated != label for arbitrated, label in zip(readings, labelled, strict=True)
        )
        if any(reading not in self.spans for reading in readings):
            self.counts.unknown_span += 1
            return None
        if not all(self.lexicon.contains(character) for character in characters):
            self.counts.unemittable_character += 1
            return None

        rng = example_rng(self.seed, epoch, sample.id)
        spans = type_syllables(readings, rng, self.augmentation)
        if self.noise is not None:
            # The styles choose *which* keys the typist meant to press and the
            # noise model then slips them -- corrupting before the abbreviation
            # would let the initial of a typoed syllable erase the slip.
            spans = self.noise.maybe_corrupt(spans, rng).spans
        span_ids = tuple(self.spans.id_or_unknown(span) for span in spans)
        targets = tuple(self.lexicon.index(character) for character in characters)
        if not all(
            self.candidates.admits(span, target)
            for span, target in zip(spans, targets, strict=True)
        ):
            self.counts.target_not_admitted += 1
            return None
        self.counts.kept += 1
        return TrainingExample(
            id=sample.id,
            spans=spans,
            span_ids=span_ids,
            targets=targets,
            context=sample.context,
        )


def read_pair(sample_shard: Path, labels_dir: Path) -> Iterator[tuple[Sample, list[str] | None]]:
    """Stream one sample shard beside the label shard of the same name.

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
    for row in pl.read_parquet(sample_shard).iter_rows(named=True):
        sample = Sample.from_row(row)
        yield sample, readings.get(sample.id)


class CorpusStream(IterableDataset[TrainingExample]):
    """The corpus as an epoch-reaugmented stream of examples.

    Shards are dealt out across DDP ranks and dataloader workers, so a shard is
    read by exactly one reader and nothing is trained on twice per epoch. Which
    reader gets which shard does not change what an example looks like -- that
    depends only on the seed, the epoch and the sample id.
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

    def set_epoch(self, epoch: int) -> None:
        """Re-augment from the next pass on."""
        self.epoch = epoch

    def _mine(self) -> list[Path]:
        """The shards this reader owns, given its rank and dataloader worker."""
        info = get_worker_info()
        workers = info.num_workers if info is not None else 1
        worker = info.id if info is not None else 0
        readers = self.world_size * workers
        index = self.rank * workers + worker
        return self.shards[index::readers]

    def __iter__(self) -> Iterator[TrainingExample]:
        for shard in self._mine():
            for sample, syllables in read_pair(shard, self.labels_dir):
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
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]:
        """Encode a batch of strings into ``input_ids`` and ``attention_mask``."""
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
    candidate_ids: torch.Tensor
    candidate_counts: torch.Tensor
    span_letters: torch.Tensor
    ids: tuple[str, ...] = field(default=())

    @property
    def size(self) -> int:
        """Examples in the batch."""
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
            candidate_ids=self.candidate_ids.to(device, non_blocking=True),
            candidate_counts=self.candidate_counts.to(device, non_blocking=True),
            span_letters=self.span_letters.to(device, non_blocking=True),
            ids=self.ids,
        )


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

    def __call__(self, examples: Sequence[TrainingExample]) -> Batch:
        """Collate *examples*, keeping each one's context with probability 1-p."""
        if not examples:
            raise ValueError("cannot collate an empty batch")
        width = max(len(example) for example in examples) + 2
        size = len(examples)
        input_ids = torch.full((size, width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((size, width), dtype=torch.long)
        span_ids = torch.zeros((size, width), dtype=torch.long)
        span_positions = torch.zeros((size, width), dtype=torch.bool)
        targets = torch.full((size, width), IGNORE_INDEX, dtype=torch.long)

        candidate_lists = [
            [self.candidates.resolve(span) for span in example.spans] for example in examples
        ]
        candidate_width = max(
            (len(row) for example_lists in candidate_lists for row in example_lists),
            default=1,
        )
        candidate_ids = torch.zeros((size, width, candidate_width), dtype=torch.long)
        candidate_counts = torch.zeros((size, width), dtype=torch.long)
        span_letters = torch.full((size, width, MAX_SPAN_LETTERS), LETTER_PAD, dtype=torch.long)

        contexts: list[str] = []
        has_context = torch.zeros(size, dtype=torch.float)
        for row, example in enumerate(examples):
            length = len(example)
            input_ids[row, 0] = self.tokenizer.cls_token_id
            input_ids[row, 1 : length + 1] = self.tokenizer.mask_token_id
            input_ids[row, length + 1] = self.tokenizer.sep_token_id
            attention_mask[row, : length + 2] = 1
            row_spans = torch.tensor(example.span_ids, dtype=torch.long)
            row_targets = torch.tensor(example.targets, dtype=torch.long)
            for column, (span, listed) in enumerate(
                zip(example.spans, candidate_lists[row], strict=True)
            ):
                if len(span) > MAX_SPAN_LETTERS:
                    raise ValueError(
                        f"example {example.id} holds a {len(span)}-letter span {span!r}; "
                        f"the letter encoding reads at most {MAX_SPAN_LETTERS}"
                    )
                candidate_ids[row, column + 1, : len(listed)] = torch.tensor(
                    listed, dtype=torch.long
                )
                candidate_counts[row, column + 1] = len(listed)
                span_letters[row, column + 1, : len(span)] = torch.tensor(
                    [letter_id(letter) for letter in span], dtype=torch.long
                )
            scored = row_targets != IGNORE_INDEX
            admitted = torch.tensor(
                [
                    self.candidates.admits(span, int(target))
                    for span, target in zip(example.spans, example.targets, strict=True)
                ]
            )
            if not bool(admitted[scored].all()):
                raise ValueError(f"example {example.id} has a target its span does not admit")
            span_ids[row, 1 : length + 1] = row_spans
            span_positions[row, 1 : length + 1] = True
            targets[row, 1 : length + 1] = row_targets
            keep = example.context is not None and self.rng.random() >= self.context_dropout
            contexts.append(
                context_tail(example.context, self.max_context_tokens)
                if keep and example.context
                else ""
            )
            has_context[row] = 1.0 if keep else 0.0

        encoded = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_context_tokens,
            return_tensors="pt",
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
            context_ids=encoded["input_ids"],
            context_mask=encoded["attention_mask"],
            has_context=has_context,
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
    """
    if budget <= 0:
        raise ValueError(f"the token budget must be positive, got {budget}")
    batch: list[TrainingExample] = []
    fill = context = 0
    for example in examples:
        wide = max(fill, len(example) + 2)
        deep = max(context, _context_width(example, max_context_tokens))
        if batch and (wide + deep) * (len(batch) + 1) > budget:
            yield batch
            batch = []
            wide = len(example) + 2
            deep = _context_width(example, max_context_tokens)
        batch.append(example)
        fill, context = wide, deep
    if batch:
        yield batch


def _context_width(example: TrainingExample, max_context_tokens: int) -> int:
    """How many context-tower positions *example* would occupy on its own."""
    if max_context_tokens <= 0:
        return 0
    return min(len(example.context or ""), max_context_tokens - 2) + 2
