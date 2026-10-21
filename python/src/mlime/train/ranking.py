"""The sequence-level ranking loss of #116: rank the gold sentence first.

#88's model trained each position against its homophones, teacher-forced on
the gold reading. The evaluation then compares whole paths: each step's
log-probability over the emittable set, summed, across every reading of the
keystrokes. Nothing trained that sum to put the gold sentence first, and the
mass a step puts outside its own candidates -- never seen by the per-position
loss -- moves it at every position. Most of v1's top-1 misses keep the gold
length and get one or two characters wrong.

This loss trains the sum itself. A batch types every example each way the
segmenter reads its keystrokes; a beam over every reading finds the sentences
the model would offer; the gold sentence and the best of the others are scored
on the decode's own path score, teacher-forced with gradient; and the softmax
over those scores is the loss, beside v1's per-position and auxiliary terms.

That softmax moves only the differences between the paths the beam found.
Trained beside v1's per-position term, which normalised over each span's
homophones, it pushed down the common characters the beam offers and nothing
held down the ones it never offers: #116's run ranked rarer characters first
and lost three points of top-1 on abbreviated input, where a span admits
hundreds. ``E2EConfig.whole_alphabet`` normalises the per-position term like
the path score, over the whole emittable set (#121).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.utils.checkpoint

from mlime.train.charlm_vocab import BOS
from mlime.train.e2e_model import E2EModel, RankingConfig
from mlime.train.e2e_search import SearchGraphs, padded_inputs, search
from mlime.train.model import (
    RouteAOutput,
    mark_dynamic,
    restricted_cross_entropy,
)
from mlime.train.samples import IGNORE_INDEX, Batch, Collator, TrainingExample
from mlime.train.segmenter import Segmentations, Segmenter
from mlime.train.spans import SpanVocab


@dataclass(frozen=True)
class RankLayout:
    """Where each row, beam hypothesis and ranked path of a step sits, laid out on the host.

    The step selects what it ranks on the device, into tensors of fixed
    shapes; every shape is a host number here, so nothing about it is read
    back. ``items`` is how many paths the step scores: each example's gold
    path first (item ``b`` for example ``b``), then a block per example --
    the gold sentence on another reading, then one slot per negative it can
    rank, ``min(negatives, beam * readings)`` -- which is what
    :meth:`RankCollator.rows` budgets the example for. ``steps`` bounds the
    typed positions of those paths: a gold path and the gold sentence on
    another reading are the gold length, a negative at most the example's
    longest reading. ``hypotheses`` is the widest example's beam hypotheses
    (its readings times the beam) and ``slots`` its ranked paths.

    The tensors say the same per row and per item: ``row_owner [R]`` and
    ``row_length [R]`` each row's example and typed positions, ``gold_row
    [B]`` each example's gold reading, ``block [B]`` the first item of each
    example's block and ``item_owner [I]`` each item's example.
    """

    items: int
    steps: int
    hypotheses: int
    slots: int
    row_owner: torch.Tensor
    row_length: torch.Tensor
    gold_row: torch.Tensor
    block: torch.Tensor
    item_owner: torch.Tensor

    def to(self, device: torch.device) -> RankLayout:
        """The same layout with its tensors on *device*."""
        return replace(
            self,
            **{name: getattr(self, name).to(device, non_blocking=True) for name in _LAYOUT_TENSORS},
        )

    def pin_memory(self) -> RankLayout:
        """The same layout with its tensors in page-locked host memory."""
        return replace(self, **{name: getattr(self, name).pin_memory() for name in _LAYOUT_TENSORS})


_LAYOUT_TENSORS = ("row_owner", "row_length", "gold_row", "block", "item_owner")


@dataclass(frozen=True)
class RankBatch:
    """Every reading of every example in one step, and which example each row is.

    ``rows`` is a plain :class:`Batch` over the readings. Each example's gold
    reading comes first among its rows and is the only one carrying targets,
    so ``rows.scored`` is exactly the gold positions -- v1's per-position and
    auxiliary losses read it unchanged, and the gold sentences are the gold
    rows' targets. The context tower runs once per example
    (``rows.context_rows``). The host-side tuples are what the search's
    shapes are decided from without reading anything back from the device:
    ``owners`` the example of each row, ``gold_rows`` the row of each
    example's gold reading and ``lengths`` each row's typed positions;
    ``layout`` places what the step ranks.
    """

    rows: Batch
    owners: tuple[int, ...]
    gold_rows: tuple[int, ...]
    lengths: tuple[int, ...]
    layout: RankLayout

    @property
    def scored(self) -> torch.Tensor:
        """The gold rows' scored positions, as the loop weighs micro-batches by."""
        return self.rows.scored

    @property
    def size(self) -> int:
        """Examples in the batch."""
        return len(self.gold_rows)

    @property
    def tokens(self) -> int:
        """Padded fill-tower positions over every reading."""
        return self.rows.tokens

    @property
    def context_tokens(self) -> int:
        """Padded context-tower positions, one context per example."""
        return self.rows.context_tokens

    def to(self, device: torch.device) -> RankBatch:
        """The same batch with its tensors on *device*."""
        return replace(self, rows=self.rows.to(device), layout=self.layout.to(device))

    def pin_memory(self) -> RankBatch:
        """The same batch with its tensors in page-locked host memory."""
        return replace(self, rows=self.rows.pin_memory(), layout=self.layout.pin_memory())


class RankCollator:
    """Types each example every way its keystrokes read, then collates the rows.

    The segmenter reads the keystrokes the example's spans spell, joined; the
    gold reading leads each example's rows whether or not the segmenter's
    k-best holds it, and a reading identical to it is not repeated. The
    context-dropout coin is the wrapped collator's, thrown once per example.
    """

    def __init__(
        self,
        collator: Collator,
        segment: Segmentations,
        spans: SpanVocab,
        ranking: RankingConfig,
    ):
        self.collator = collator
        self.segment = segment
        self.spans = spans
        self.ranking = ranking

    @classmethod
    def launch(
        cls, collator: Collator, segmenter: Path, spans: SpanVocab, ranking: RankingConfig
    ) -> RankCollator:
        """A collator reading keystrokes through a running *segmenter* binary."""
        return cls(collator, Segmenter(segmenter, ranking.segmentations), spans, ranking)

    @property
    def rng(self) -> random.Random:
        """The wrapped collator's generator: the state a checkpoint carries."""
        return self.collator.rng

    @property
    def max_context_tokens(self) -> int:
        """The wrapped collator's context budget, which batching reads."""
        return self.collator.max_context_tokens

    def readings(self, example: TrainingExample) -> list[tuple[int, tuple[str, ...]]]:
        """The segmenter's readings of *example* other than its gold one, by rank."""
        return [
            (index, reading)
            for index, reading in enumerate(self.segment("".join(example.spans)))
            if reading != example.spans
        ]

    def negatives(self, readings: int) -> int:
        """The negatives an example of *readings* readings ranks at most.

        No more than ``negatives``, and no more than the beam can have found
        on the readings.
        """
        return min(self.ranking.negatives, self.ranking.beam * readings)

    def rows(self, example: TrainingExample) -> int:
        """Every row *example* costs: its readings, and the paths it ranks.

        A step runs the fill tower over each reading and the decoder over each
        ranked path, at the same width; the paths are the gold one, the gold
        sentence on another reading, and :meth:`negatives` others.
        """
        readings = 1 + len(self.readings(example))
        return readings + 2 + self.negatives(readings)

    def __call__(self, examples: Sequence[TrainingExample]) -> RankBatch:
        """Collate *examples* with every reading of each."""
        rows: list[TrainingExample] = []
        owners: list[int] = []
        gold_rows: list[int] = []
        for owner, example in enumerate(examples):
            if IGNORE_INDEX in example.targets:
                raise ValueError(
                    f"example {example.id} leaves a span untargeted; its gold sentence is "
                    "every position's target"
                )
            gold_rows.append(len(rows))
            rows.append(example)
            for index, reading in self.readings(example):
                rows.append(
                    TrainingExample(
                        id=f"{example.id}:{index}",
                        spans=reading,
                        span_ids=tuple(self.spans.id(span) for span in reading),
                        targets=tuple(IGNORE_INDEX for _ in reading),
                        context=example.context,
                    )
                )
            owners.extend(owner for _ in range(len(rows) - len(owners)))
        lengths = tuple(len(row) for row in rows)
        return RankBatch(
            rows=self.collator.collate(rows, [example.context for example in examples], owners),
            owners=tuple(owners),
            gold_rows=tuple(gold_rows),
            lengths=lengths,
            layout=self.layout(gold_rows, lengths),
        )

    def layout(self, gold_rows: Sequence[int], lengths: Sequence[int]) -> RankLayout:
        """Place what a step over rows of *lengths*, led by *gold_rows*, ranks."""
        examples = len(gold_rows)
        bounds = [*gold_rows, len(lengths)]
        readings = [bounds[index + 1] - bounds[index] for index in range(examples)]
        negatives = [self.negatives(count) for count in readings]
        block: list[int] = []
        item_owner = list(range(examples))
        steps = 0
        for example, ranked in enumerate(negatives):
            block.append(len(item_owner))
            item_owner.extend(example for _ in range(1 + ranked))
            longest = max(lengths[bounds[example] : bounds[example + 1]])
            steps += 2 * lengths[gold_rows[example]] + ranked * longest
        return RankLayout(
            items=len(item_owner),
            steps=steps,
            hypotheses=self.ranking.beam * max(readings),
            slots=2 + max(negatives),
            row_owner=torch.tensor(
                [owner for owner, count in enumerate(readings) for _ in range(count)],
                dtype=torch.long,
            ),
            row_length=torch.tensor(lengths, dtype=torch.long),
            gold_row=torch.tensor(gold_rows, dtype=torch.long),
            block=torch.tensor(block, dtype=torch.long),
            item_owner=torch.tensor(item_owner, dtype=torch.long),
        )


@dataclass(frozen=True)
class Items:
    """The paths one step scores, on the device: which row each reads, what it says, how it ranks.

    ``rows [I]`` and ``sequences [I, S]`` (emission indices, ``-1`` past
    each path's length, which ``steps [I, S]`` marks) are laid out as
    :class:`RankLayout` says: item ``b < B`` is example ``b``'s gold path on
    its gold row, so the gold rows' per-position loss reads the first ``B``
    items' features, and an unused slot of a block holds its example's gold
    path again, scored and ignored. ``slots [B, M]`` lists each example's
    items in ranking order -- the gold path, the gold sentence on another
    reading when the beam found it there, the negatives best first -- with
    ``-1`` past the end, and ``positive [B, M]`` marks the ones that say the
    gold sentence.
    """

    rows: torch.Tensor
    sequences: torch.Tensor
    steps: torch.Tensor
    slots: torch.Tensor
    positive: torch.Tensor


def select_items(
    batch: RankBatch, sequences: torch.Tensor, scores: torch.Tensor, negatives: int
) -> Items:
    """Merge each example's beam hypotheses over its readings, and pick what to rank, on the device.

    *sequences* ``[L, W, T]`` and *scores* ``[L, W]`` are the search's, one
    lane per row (lanes past the rows are a bucket's padding); a score of
    minus infinity is a slot the lattice never filled. A sentence two
    readings both produce is one answer, scored by the better of them -- the
    evaluation's own merge -- and found first, by row then beam, on a tie.
    Beside the gold path, an example ranks its ``negatives`` best sentences
    that are not the gold one, best first by score, then row, then sentence,
    each from the reading that scored it best, and the gold sentence's best
    reading when the beam found it on a row other than the gold one: either
    reading is the right answer. Sentences compare whole -- every position
    and the length -- and every shape is the layout's, so nothing is read
    back.
    """
    layout = batch.layout
    device = scores.device
    count = len(batch.lengths)
    examples = batch.size
    width = scores.shape[1]
    span = batch.rows.targets.shape[1] - 2
    hypotheses = layout.hypotheses
    floor = float("-inf")
    position = torch.arange(span, device=device)
    found = sequences[:count, :, :span].masked_fill(
        (position >= layout.row_length.unsqueeze(1)).unsqueeze(1), -1
    )

    # Example-major: beam w of an example's j-th row is column j * width + w,
    # so column order is the order the evaluation's merge meets them in.
    local = torch.arange(count, device=device) - layout.gold_row.index_select(0, layout.row_owner)
    column = local.unsqueeze(1) * width + torch.arange(width, device=device)
    owner = layout.row_owner.unsqueeze(1).expand(count, width)
    at = (owner, column)
    score = scores.new_full((examples, hypotheses), floor).index_put(at, scores[:count])
    row = (
        layout.gold_row.unsqueeze(1)
        .repeat(1, hypotheses)
        .index_put(at, torch.arange(count, device=device).unsqueeze(1).expand(count, width))
    )
    said = found.new_full((examples, hypotheses, span), -1).index_put(at, found)
    valid = score != floor

    # [B, i, j]: what hypothesis j is to hypothesis i.
    differ = said.unsqueeze(2) != said.unsqueeze(1)
    same = ~differ.any(3) & valid.unsqueeze(2) & valid.unsqueeze(1)
    mine, theirs = score.unsqueeze(2), score.unsqueeze(1)
    columns = torch.arange(hypotheses, device=device)
    earlier = columns.unsqueeze(0) < columns.unsqueeze(1)
    stands_over = (theirs > mine) | ((theirs == mine) & earlier)
    kept = valid & ~(same & stands_over).any(2)

    gold = batch.rows.targets[:, 1 : span + 1].index_select(0, layout.gold_row)
    gold_length = layout.row_length.index_select(0, layout.gold_row)
    gold = gold.masked_fill(position >= gold_length.unsqueeze(1), -1)
    says_gold = (said == gold.unsqueeze(1)).all(2)
    other = kept & says_gold & (row != layout.gold_row.unsqueeze(1))
    candidate = kept & ~says_gold

    # Rank by (score, row, sentence), best first; sentences of one row share
    # its length, so the first position they differ at orders them.
    first = differ.int().argmax(3, keepdim=True)
    shape = differ.shape
    later = (
        said.unsqueeze(1).expand(shape).gather(3, first)
        > said.unsqueeze(2).expand(shape).gather(3, first)
    ).squeeze(3)
    mine_row, their_row = row.unsqueeze(2), row.unsqueeze(1)
    ahead = (theirs > mine) | (
        (theirs == mine)
        & ((their_row > mine_row) | ((their_row == mine_row) & differ.any(3) & later))
    )
    rank = (ahead & candidate.unsqueeze(1)).sum(2)
    chosen = candidate & (rank < negatives)

    # Each kept hypothesis to its item: the gold sentence on another row to
    # its example's block head, the chosen negatives after it by rank, and
    # everything else to a spare item past the end.
    items = layout.items
    block = layout.block.unsqueeze(1)
    spare = torch.full_like(block, items)
    target = torch.where(chosen, block + 1 + rank, torch.where(other, block, spare)).reshape(-1)
    item_rows = torch.cat(
        [layout.gold_row.index_select(0, layout.item_owner), layout.gold_row[:1]]
    ).scatter(0, target, row.reshape(-1))[:items]
    item_sequences = torch.cat([gold.index_select(0, layout.item_owner), gold[:1]]).scatter(
        0, target.unsqueeze(1).expand(-1, span), said.reshape(-1, span)
    )[:items]
    steps = position < layout.row_length.index_select(0, item_rows).unsqueeze(1)

    # Each example's ranked list, compact as the loss reads it: the gold
    # path, the gold sentence on another row if found, the negatives.
    width_slots = layout.slots
    has_other = other.any(1, keepdim=True).long()
    place = torch.where(
        chosen,
        1 + has_other + rank,
        torch.where(other, torch.ones_like(rank), torch.full_like(rank, width_slots)),
    )
    filled = torch.full((examples, width_slots + 1), -1, dtype=torch.long, device=device).scatter(
        1, place, target.view(examples, hypotheses)
    )
    marked = torch.zeros((examples, width_slots + 1), dtype=torch.bool, device=device).scatter(
        1, place, other
    )
    heads = torch.arange(examples, device=device).unsqueeze(1)
    return Items(
        rows=item_rows,
        sequences=item_sequences,
        steps=steps,
        slots=torch.cat([heads, filled[:, 1:width_slots]], dim=1),
        positive=torch.cat([torch.ones_like(heads, dtype=torch.bool), marked[:, 1:width_slots]], 1),
    )


class RankingE2EModel(E2EModel):
    """The end-to-end model trained to rank whole sentences, not only positions.

    Everything but :meth:`forward` is :class:`E2EModel`'s: the same module
    tree, the same checkpoint record (with its ``ranking`` field set), the
    same evaluation. The forward takes a :class:`RankBatch` instead of a
    plain one.

    ``search_graphs``, set by the trainer on a CUDA run that asks for it,
    replays the forward's search from captured CUDA graphs (#137).
    """

    search_graphs: SearchGraphs | None = None

    def ranking(self) -> RankingConfig:
        """The ranking settings, which this model cannot be built without."""
        ranking = self.e2e_config.ranking
        if ranking is None:
            raise ValueError("a ranking model needs a config with ranking settings")
        return ranking

    # The training forward of this model reads every reading of each example,
    # which a plain batch does not carry; evaluation goes through ``scores``,
    # inherited unchanged.
    def forward(self, batch: RankBatch) -> RouteAOutput:  # type: ignore[override]
        """Per-position, auxiliary and ranking losses over every reading of the batch.

        Nothing here reads the device back: the search, the walk back, the
        merge and the pick of what to rank all run on the device into the
        layout's fixed shapes, so the host queues the whole step.
        """
        ranking = self.ranking()
        rows = batch.rows
        layout = batch.layout
        mark_dynamic(rows)
        hidden = self.encode(rows)
        width = rows.targets.shape[1]
        spans = hidden[:, 1 : width - 1]
        admitted = self._admitted(rows)

        decoder = self.decoder
        training = decoder.training
        decoder.eval()
        with torch.no_grad(), torch.profiler.record_function("rank.beam"):
            # Zero past each row's own length, as a lane padded alone is.
            outside = torch.arange(spans.shape[1], device=spans.device) >= (
                layout.row_length.unsqueeze(1)
            )
            vectors = spans.detach().masked_fill(outside.unsqueeze(2), 0)
            sequences, scores = search(
                decoder,
                padded_inputs(
                    vectors,
                    admitted,
                    layout.row_length,
                    batch.lengths,
                    ranking.beam,
                    bucketed=self.search_graphs is not None,
                ),
                self.search_graphs,
            )
        decoder.train(training)
        # Autocast caches each weight's low-precision cast for the rest of its
        # region, and the casts the search just made carry no graph: reused
        # below, they would cut the head out of the loss (DDP then reports
        # `decoder.reader.project` as unused). Dropping them recasts with grad.
        torch.clear_autocast_cache()
        with torch.profiler.record_function("rank.select_items"):
            items = select_items(batch, sequences, scores, ranking.negatives)
        with torch.profiler.record_function("rank.path_scores"):
            paths, projected = self.path_scores(
                spans, items.rows, items.sequences, items.steps, layout.steps
            )
        length = projected.shape[1]

        listed = items.slots >= 0
        floor = float("-inf")
        scores = paths[items.slots.clamp(min=0)].masked_fill(~listed, floor)
        rank_loss = (
            scores.logsumexp(dim=1) - scores.masked_fill(~items.positive, floor).logsumexp(dim=1)
        ).mean()
        first_negative = scores.masked_fill(~listed | items.positive, floor).amax(dim=1)
        rank_accuracy = (scores[:, 0] > first_negative).float().mean()

        # The gold items are the first B and the gold rows alone carry targets:
        # a scored position of row r, column c is step c - 1 of r's example's
        # gold item.
        positions = rows.scored
        region = (
            layout.row_owner.index_select(0, positions // width) * length + positions % width - 1
        )
        decoder_logits = (
            projected.reshape(-1, projected.shape[-1]).index_select(0, region) @ self._emitted().T
        )
        candidates = self.candidates_at(rows, positions)
        targets = rows.targets.reshape(-1).index_select(0, positions)
        encoder_logits = self.head.decoder(
            self.head.transform(hidden.reshape(-1, hidden.shape[-1]).index_select(0, positions))
        )
        smoothing = self.config.label_smoothing
        decoder_loss = self.position_loss(decoder_logits, targets, candidates)
        encoder_loss = restricted_cross_entropy(
            encoder_logits, targets, candidates, smoothing, check=False
        )
        return RouteAOutput(
            loss=decoder_loss
            + self.e2e_config.aux_weight * encoder_loss
            + ranking.weight * rank_loss,
            extras={
                "decoder_loss": decoder_loss.detach(),
                "encoder_loss": encoder_loss.detach(),
                "rank_loss": rank_loss.detach(),
                "rank_accuracy": rank_accuracy.detach(),
            },
        )

    def path_scores(
        self,
        spans: torch.Tensor,
        rows: torch.Tensor,
        sequences: torch.Tensor,
        steps: torch.Tensor,
        bound: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each path's score on its row, teacher-forced; and the step features.

        ``spans [R, S, H]`` is the encoder's output at every row's typed
        positions; path *i* reads row ``rows[i]`` and says ``sequences[i]``
        (``[I, S]``) in emission indices over the positions ``steps[i]``
        marks, any value past them. Its score is what the search sums for it:
        each step's log-probability over the whole emittable set. The
        projected features come back ``[I, S, D]`` for the per-position loss,
        which takes its own logits at the gold positions.

        The whole-alphabet logits exist only a chunk of steps at a time and
        are recomputed in backward rather than kept: a step needs its own
        logit and the normaliser, not the seven thousand others, and keeping
        them would hold more than the decoder's own activations do. They are
        taken over the real steps alone, packed in order into *bound* slots
        -- a count the host knows, so nothing is read back -- and the slots
        past the last real step score a step nobody reads.
        """
        decoder = self.decoder
        device = spans.device
        count, length = sequences.shape
        said = sequences.clamp(min=0)
        mask = steps
        real = mask.reshape(-1)
        # Each real step's place in the packed order, and a spare place past
        # the bound for every other one (whose index then points nowhere read).
        place = torch.where(real, real.cumsum(0) - 1, bound)
        flat = torch.arange(real.numel(), device=device)
        taken = torch.zeros(bound + 1, dtype=torch.long, device=device).scatter(0, place, flat)[
            :bound
        ]
        owner = torch.full((bound + 1,), count, dtype=torch.long, device=device).scatter(
            0, place, flat // length
        )[:bound]

        reader_ids = decoder.emit_ids[said]
        previous = torch.cat([torch.full_like(reader_ids[:, :1], BOS), reader_ids[:, :-1]], dim=1)
        read_spans = spans.index_select(0, rows)
        # The compiled decoder sees a new item count and length every step;
        # marked here, its first compile is the dynamic one (as in `_scored`).
        for tensor in (previous, read_spans, mask):
            torch._dynamo.maybe_mark_dynamic(tensor, (0, 1))
        features = decoder(previous, read_spans, mask)
        reader = decoder.reader
        projected: torch.Tensor = reader.project(reader.dropout(features))
        picked = projected.reshape(-1, projected.shape[-1]).index_select(0, taken)
        targets = said.reshape(-1).index_select(0, taken)
        emitted = self._emitted()
        stepped = torch.cat(
            [
                torch.as_tensor(
                    torch.utils.checkpoint.checkpoint(
                        _step_log_probabilities,
                        picked[start : start + STEP_CHUNK],
                        emitted,
                        targets[start : start + STEP_CHUNK],
                        use_reentrant=False,
                    )
                )
                for start in range(0, bound, STEP_CHUNK)
            ]
        )
        paths = torch.zeros(count + 1, dtype=stepped.dtype, device=device).index_add(
            0, owner, stepped
        )[:count]
        return paths, projected

    def _emitted(self) -> torch.Tensor:
        """``[E, D]``: the reader's embedding row of each emittable character."""
        decoder = self.decoder
        return decoder.reader.embed.weight.index_select(0, decoder.emit_ids)

    def _admitted(self, rows: Batch) -> torch.Tensor:
        """``[R, S, E]``: what each typed position of each row admits.

        Built from the rows' own candidate lists, padded entries sent to a
        spare column that is cut off afterwards.
        """
        width = rows.targets.shape[1]
        ids = rows.candidate_ids[:, 1 : width - 1]
        counts = rows.candidate_counts[:, 1 : width - 1]
        listed = torch.arange(ids.shape[-1], device=ids.device) < counts.unsqueeze(-1)
        emittable = self.decoder.emit_ids.numel()
        spare = ids.masked_fill(~listed, emittable)
        admitted = torch.zeros((*ids.shape[:2], emittable + 1), dtype=torch.bool, device=ids.device)
        return admitted.scatter_(2, spare, True)[..., :emittable]


#: Steps whose whole-alphabet logits exist at once in :meth:`RankingE2EModel.path_scores`:
#: 2048 of them over 7,322 characters is 90 MiB in fp32, beside activations of
#: gigabytes.
STEP_CHUNK = 2048


def _step_log_probabilities(
    features: torch.Tensor, emitted: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Each step's log-probability of its own character over the emittable set."""
    logits = (features @ emitted.T).float()
    said = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    return said - logits.logsumexp(dim=1)
