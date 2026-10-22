"""Key-level training data: the keystrokes, their gold alignments, and the batches (#216).

The span model reads one slot per typed span of one lattice reading; the key
model reads one position per keystroke, and a per-key head emits a character
or a blank. What the model may emit for a record is the legality automaton's
(:mod:`mlime.legality`): this module builds, per example, the arcs of every
legal alignment of the gold sentence over its keys, so the constrained-CTC loss
(:mod:`mlime.train.ctc`) sums over exactly those.

An example is typed as the span model's is -- the reading arbitration, then a
typing style per example and epoch -- and the keys are the typed spans run
together. With typos on, the noise model then slips some of the spans typed in
full: never one typed as an initial, and only into keys the corrections read
back as the intended syllable (#216 §4), so the typo never makes the gold
illegal. The spans as the typist meant them stay on the example: the S1
warm-up's teacher, the span model, reads those.

A gold the automaton rejects is counted (``illegal_gold``) and dropped, never
trained on: on clean data it is a defect in the data, and the count is zero.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from mlime.data.corpus import Sample
from mlime.legality import MAX_ARC_KEYS, GoldArcs, Legality, LegalityOptions
from mlime.train.arbitration import ReadingArbitration
from mlime.train.lexicon import Lexicon
from mlime.train.samples import (
    LETTER_PAD,
    Augmentation,
    Batch,
    Collator,
    SampleBuilder,
    TrainingExample,
    example_rng,
    letter_id,
    type_syllables,
)
from mlime.train.spans import SpanVocab
from mlime.typo import MULTI_LETTER_INITIALS, NoiseModel

#: The QWERTY letter rows, top to bottom, and how far each row's first key
#: sits right of the top row's, in key widths: the stagger of the standard
#: layout (Tab 1.5, Caps Lock 1.75, Shift 2.25 key widths wide).
QWERTY_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
QWERTY_STAGGER = (0.0, 0.25, 0.75)

#: How many times one edit is redrawn when the corrections cannot read the
#: span it produced back. A span typed in full is at least two keys, and every
#: edit kind but a doubling or a fuzzy swap leaves most two-to-six-letter
#: spans readable, so a draw that fails this often has no readable edit left.
EDIT_DRAWS = 16


def key_coordinates() -> torch.Tensor:
    """``[26, 2]``: each letter's (row, column) on QWERTY, in key widths, ``a`` first."""
    table = torch.zeros((LETTER_PAD, 2), dtype=torch.float32)
    for row, (letters, stagger) in enumerate(zip(QWERTY_ROWS, QWERTY_STAGGER, strict=True)):
        for column, letter in enumerate(letters):
            table[letter_id(letter)] = torch.tensor([float(row), column + stagger])
    return table


@dataclass(frozen=True)
class KeyFrontConfig:
    """The key-level front and its losses (#216 §1-§2, §7).

    ``layers`` is the thin causal stack between the key embeddings and the
    fill tower, at the tower's width. ``distill_weight`` weighs the S1
    warm-up's distillation term against the constrained CTC.
    ``correct_syllables`` is the legality automaton's: whether a span that is
    itself a syllable may be read through a correction. ``max_keys`` is the
    longest key string an example may have; the fill tower's positions are
    the bound, two of them taken by the sentinels.
    """

    layers: int = 2
    distill_weight: float = 1.0
    correct_syllables: bool = True
    max_keys: int = 510

    def __post_init__(self) -> None:
        if self.layers < 1:
            raise ValueError(f"the key front needs at least one layer, got {self.layers}")
        if self.distill_weight < 0.0:
            raise ValueError(f"distill_weight must not be negative, got {self.distill_weight}")
        if self.max_keys < 1:
            raise ValueError(f"max_keys must be positive, got {self.max_keys}")

    @property
    def options(self) -> LegalityOptions:
        """The automaton's options: complete offline input, as a training record is."""
        return LegalityOptions(unfinished_tail=False, correct_syllables=self.correct_syllables)

    def sampling(self) -> dict[str, Any]:
        """The fields that decide which examples a run keeps and how wide they are."""
        return {"correct_syllables": self.correct_syllables, "max_keys": self.max_keys}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> KeyFrontConfig:
        """Rebuild the config a checkpoint's ``e2e`` record holds."""
        return cls(**record)


@dataclass(frozen=True)
class FillTowerTraining:
    """What of the parent's fill path S1 trains beside the key front and the head (#241).

    The fill tower always, at the head's rate; ``cross_attention`` adds the
    tower's gated context blocks. Each part that trains is the student's
    alone: the model keeps a frozen copy of the parent's for the teacher.
    """

    cross_attention: bool = False

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FillTowerTraining:
        """Rebuild the switch a checkpoint's ``e2e`` record holds."""
        return cls(**record)


@dataclass(frozen=True)
class KeyExample(TrainingExample):
    """A training example as keystrokes, with the gold's legal alignments over them.

    ``spans`` are the spans the typist meant, which the span model reads;
    ``keys`` are what was pressed, the spans after the noise model, run
    together. ``ends[j]`` is the key that closes span ``j`` in ``keys`` and
    ``corrupted[j]`` whether the noise model slipped it. ``gold`` holds every
    arc of the gold sentence's legal alignments over ``keys``.
    """

    keys: str = ""
    ends: tuple[int, ...] = ()
    corrupted: tuple[bool, ...] = ()
    gold: GoldArcs | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.ends) != len(self.spans) or len(self.corrupted) != len(self.spans):
            raise ValueError(f"example {self.id}'s ends or typo marks do not cover its spans")
        if not self.keys or self.ends[-1] != len(self.keys) - 1:
            raise ValueError(f"example {self.id}'s last span does not end at its last key")
        if self.gold is None or self.gold.keys != len(self.keys):
            raise ValueError(f"example {self.id} carries no alignments over its keys")
        if self.gold.length != len(self.targets):
            raise ValueError(f"example {self.id}'s alignments are not over its sentence")

    def fill_width(self) -> int:
        """Both fill passes' positions: the keys' and the spans', each with sentinels.

        The warm-up runs the span model over the spans as its teacher beside
        the key model over the keys, and a later stage's decoder runs one row
        per character, so a key example costs its keys and its spans together.
        """
        return len(self.keys) + len(self.spans) + 4

    def arcs(self) -> GoldArcs:
        """The gold's alignments, present by construction."""
        if self.gold is None:
            raise ValueError(f"example {self.id} carries no alignments")
        return self.gold


def typed_in_full(span: str, reading: str) -> bool:
    """Whether *span* is its reading typed out, and long enough for a correction to read.

    A one-key span, and ``zh``/``ch``/``sh`` standing whole, are initials to
    the automaton, which never corrects one (#216 §3): a slip there is another
    legal initial, not a typo.
    """
    return span == reading and len(span) >= 2 and span not in MULTI_LETTER_INITIALS


class KeySampleBuilder(SampleBuilder):
    """Builds key-level examples: the span model's readings and typing, then keys.

    The span builder's own corruption stays off -- it slips letters across
    span seams, which a key-level example must not -- and this builder's noise
    model corrupts within the spans typed in full instead
    (:meth:`corrupt`). *noise* comes exactly when ``augmentation.typos`` is on.
    """

    def __init__(
        self,
        lexicon: Lexicon,
        spans: SpanVocab,
        arbitration: ReadingArbitration,
        legality: Legality,
        config: KeyFrontConfig,
        augmentation: Augmentation | None = None,
        seed: int = 0,
        noise: NoiseModel | None = None,
    ):
        augmented = augmentation or Augmentation()
        if (noise is not None) != augmented.typos:
            raise ValueError(
                f"the noise model and augmentation.typos come together; "
                f"got noise={noise is not None}, typos={augmented.typos}"
            )
        if legality.characters != lexicon.characters:
            raise ValueError("the legality automaton emits over a different set than the lexicon")
        super().__init__(lexicon, spans, arbitration, replace(augmented, typos=False), seed)
        self.augmentation = augmented
        self.legality = legality
        self.config = config
        self.key_noise = noise

    def build(
        self, sample: Sample, syllables: Sequence[str] | None, epoch: int
    ) -> KeyExample | None:
        """The key-level example for *sample* at *epoch*, or ``None`` with a counter raised."""
        settled = self.settle(sample, syllables)
        if settled is None:
            return None
        readings, targets = settled
        rng = example_rng(self.seed, epoch, sample.id)
        meant = type_syllables(readings, rng, self.augmentation)
        if not self.candidates.admits_all(meant, targets):
            self.counts.target_not_admitted += 1
            return None
        pressed, corrupted = self.corrupt(meant, readings, targets, rng)
        keys = "".join(pressed)
        if len(keys) > self.config.max_keys:
            self.counts.keys_too_long += 1
            return None
        gold = self.legality.gold_ids(keys, targets, self.config.options)
        if gold is None:
            self.counts.illegal_gold += 1
            return None
        ends: list[int] = []
        cursor = 0
        for span in pressed:
            cursor += len(span)
            ends.append(cursor - 1)
        self.counts.kept += 1
        return KeyExample(
            id=sample.id,
            spans=meant,
            span_ids=tuple(self.spans.id_or_unknown(span) for span in meant),
            targets=targets,
            context=sample.context,
            keys=keys,
            ends=tuple(ends),
            corrupted=corrupted,
            gold=gold,
        )

    def corrupt(
        self,
        spans: tuple[str, ...],
        readings: tuple[str, ...],
        targets: tuple[int, ...],
        rng: random.Random,
    ) -> tuple[tuple[str, ...], tuple[bool, ...]]:
        """*spans* as pressed, and which of them the noise model slipped.

        The per-sentence coin and the edit count are the noise model's own
        draws. Each edit lands on a span typed in full, chosen with weight its
        current length -- the model's own draw is uniform over letters -- and
        is kept only when the corrections read the slipped span back as the
        span's character; otherwise it is redrawn, up to :data:`EDIT_DRAWS`
        times, and given up after that. A sentence with no span typed in full
        stays clean however the coin fell.
        """
        clean = (False,) * len(spans)
        noise = self.key_noise
        if noise is None or rng.random() >= noise.per_sentence:
            return spans, clean
        eligible = [
            index
            for index, (span, reading) in enumerate(zip(spans, readings, strict=True))
            if typed_in_full(span, reading)
        ]
        if not eligible:
            return spans, clean
        self.counts.typo_sentences += 1
        pressed = list(spans)
        corrupted = [False] * len(spans)
        options = self.config.options
        for _ in range(noise.draw_count(rng, 1)):
            for _ in range(EDIT_DRAWS):
                index = rng.choices(eligible, [len(pressed[at]) for at in eligible])[0]
                slipped, _edit = noise.edit_span(pressed[index], rng)
                if (
                    len(slipped) >= 2
                    and slipped not in MULTI_LETTER_INITIALS
                    and math.isfinite(
                        self.legality.char_cost(slipped, False, targets[index], options)
                    )
                ):
                    pressed[index] = slipped
                    corrupted[index] = True
                    self.counts.typo_edits += 1
                    break
                self.counts.typo_edits_redrawn += 1
            else:
                self.counts.typo_edits_dropped += 1
        return tuple(pressed), tuple(corrupted)


@dataclass(frozen=True)
class KeyBatch:
    """One collated key-level step: the span model's batch, and the keys beside it.

    ``spans`` is the batch the span model reads -- the teacher's input in the
    warm-up, and the rows the decoder is fed in evaluation. The keys' own fill
    row is ``[CLS] k1 .. kn [SEP]``: ``key_input_ids`` holds the base tokens
    there (``[MASK]`` at every key), ``key_letters`` the letter ids the key
    front reads at the same columns (``LETTER_PAD`` elsewhere), and
    ``key_attention`` the real positions.

    ``key_positions [K]`` are the batch's keys as flat indices into the
    ``[B, Wk]`` layout, row-major, and ``key_index [B, T]`` names key ``t``
    of row ``b`` among them. ``gold [B, M]`` is each row's sentence,
    ``lengths`` and ``keys`` its length and key count, and ``cost [B, M, T,
    W]`` the arc table of its legal alignments (:func:`constrained_ctc`).
    ``span_end [B, M]`` is each typed span's last key, ``end_positions [P]``
    the same keys as flat indices into ``[B, Wk]`` in the order of
    ``spans.scored``, ``ends [B, T]`` the typed cover's ends as a mask, and
    ``corrupted``, ``position_mask`` and ``key_mask`` what is typoed and what
    is real.
    """

    spans: Batch
    key_input_ids: torch.Tensor
    key_attention: torch.Tensor
    key_letters: torch.Tensor
    key_positions: torch.Tensor
    key_index: torch.Tensor
    gold: torch.Tensor
    lengths: torch.Tensor
    keys: torch.Tensor
    cost: torch.Tensor
    span_end: torch.Tensor
    end_positions: torch.Tensor
    ends: torch.Tensor
    corrupted: torch.Tensor
    position_mask: torch.Tensor
    key_mask: torch.Tensor

    @property
    def scored(self) -> torch.Tensor:
        """The gold characters, as the span batch's scored positions: what the loss averages."""
        return self.spans.scored

    @property
    def size(self) -> int:
        """Examples in the step."""
        return self.spans.size

    @property
    def tokens(self) -> int:
        """Padded fill-tower positions: the keys' rows and the spans' rows."""
        return self.spans.tokens + int(self.key_input_ids.numel())

    @property
    def context_tokens(self) -> int:
        """Padded context-tower positions."""
        return self.spans.context_tokens

    def _map(self, spans: Batch, apply: Callable[[torch.Tensor], torch.Tensor]) -> KeyBatch:
        return KeyBatch(
            spans=spans,
            **{
                item.name: apply(getattr(self, item.name))
                for item in fields(self)
                if item.name != "spans"
            },
        )

    def to(self, device: torch.device) -> KeyBatch:
        """The same batch on *device*."""
        return self._map(self.spans.to(device), lambda value: value.to(device, non_blocking=True))

    def pin_memory(self) -> KeyBatch:
        """The same batch in page-locked host memory."""
        return self._map(self.spans.pin_memory(), lambda value: value.pin_memory())


class KeyCollator:
    """Pads key-level examples into a :class:`KeyBatch`.

    The span half is the span model's own :class:`~mlime.train.samples.Collator`
    over the spans the typist meant, which also throws each example's context
    dropout coin; the keys share it.
    """

    def __init__(self, spans: Collator):
        self.spans = spans

    @property
    def rng(self) -> random.Random:
        """The span collator's generator: the context dropout's draws."""
        return self.spans.rng

    @property
    def max_context_tokens(self) -> int:
        """The context budget batching is grouped under."""
        return self.spans.max_context_tokens

    def rows(self, example: TrainingExample) -> int:
        """One: an example is one row of each fill pass, which its width already counts."""
        return 1

    def __call__(self, examples: Sequence[TrainingExample]) -> KeyBatch:
        """Collate *examples*, which must be key-level ones."""
        keyed: list[KeyExample] = []
        for example in examples:
            if not isinstance(example, KeyExample):
                raise TypeError(f"the key collator got a {type(example).__name__}")
            keyed.append(example)
        span_batch = self.spans(keyed)
        tokenizer = self.spans.tokenizer
        size = len(keyed)
        counts = torch.tensor([len(example.keys) for example in keyed], dtype=torch.long)
        lengths = torch.tensor([len(example.spans) for example in keyed], dtype=torch.long)
        width = int(counts.max())
        positions = int(lengths.max())
        columns = torch.arange(width + 2)
        at_key = (columns >= 1) & (columns <= counts[:, None])
        key_input_ids = torch.full((size, width + 2), tokenizer.pad_token_id, dtype=torch.long)
        key_input_ids[at_key] = tokenizer.mask_token_id
        key_input_ids[:, 0] = tokenizer.cls_token_id
        key_input_ids[torch.arange(size), counts + 1] = tokenizer.sep_token_id
        key_attention = (columns < counts[:, None] + 2).long()
        key_letters = torch.full((size, width + 2), LETTER_PAD, dtype=torch.long)
        key_letters[at_key] = torch.tensor(
            [letter_id(letter) for example in keyed for letter in example.keys], dtype=torch.long
        )
        key_positions = at_key.reshape(-1).nonzero(as_tuple=True)[0]
        within = torch.arange(width)
        key_mask = within < counts[:, None]
        starts = torch.cumsum(counts, 0) - counts
        key_index = torch.where(key_mask, starts[:, None] + within, starts[:, None])

        gold = torch.zeros((size, positions), dtype=torch.long)
        span_end = torch.zeros((size, positions), dtype=torch.long)
        corrupted = torch.zeros((size, positions), dtype=torch.bool)
        cost = torch.full((size, positions, width, MAX_ARC_KEYS), math.inf, dtype=torch.float32)
        ends = torch.zeros((size, width), dtype=torch.bool)
        end_positions: list[int] = []
        for row, example in enumerate(keyed):
            length = len(example.spans)
            gold[row, :length] = torch.tensor(example.targets, dtype=torch.long)
            span_end[row, :length] = torch.tensor(example.ends, dtype=torch.long)
            corrupted[row, :length] = torch.tensor(example.corrupted, dtype=torch.bool)
            cost[row, :length, : len(example.keys)] = torch.from_numpy(example.arcs().dense())
            ends[row, list(example.ends)] = True
            end_positions.extend(row * (width + 2) + end + 1 for end in example.ends)
        position_mask = torch.arange(positions) < lengths[:, None]
        if len(end_positions) != span_batch.scored.numel():
            raise ValueError("the keys' span ends and the span batch's scored positions disagree")
        return KeyBatch(
            spans=span_batch,
            key_input_ids=key_input_ids,
            key_attention=key_attention,
            key_letters=key_letters,
            key_positions=key_positions,
            key_index=key_index,
            gold=gold,
            lengths=lengths,
            keys=counts,
            cost=cost,
            span_end=span_end,
            end_positions=torch.tensor(end_positions, dtype=torch.long),
            ends=ends,
            corrupted=corrupted,
            position_mask=position_mask,
            key_mask=key_mask,
        )
