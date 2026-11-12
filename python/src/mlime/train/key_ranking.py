"""Sequence-level training of the key-level model over its own constrained search (S4, #278).

S3's model finds the gold sentence far more often than it ranks it first. The
search ranks whole sentences by a path score -- the constrained CTC of the
sentence over the keys, plus ``beta`` times the decoder's transition summed
over its characters -- and nothing trained that sum to put the gold first.
This stage trains it, as #132 did for the span model: S3's two terms
unchanged, and beside them ``lambda`` times a term over the scores of the gold
sentence and the sentences the model itself offers instead.

**The score.** With the correction costs scaled by ``gamma``,

    S(y) = -constrained_ctc(y; gamma * cost) + beta * sum_j log T(y_j | y_<j, keys)

is what :func:`~mlime.train.key_search.search` returns without a beam and with
the sum combination. ``beta`` and ``gamma`` are constants of the stage. The
only difference between the training path and the search is the dropout of the
scoring pass.

**The candidates** come from a beam on the device that reads no gradient. A
sentence is a character per span of a cover of the keys; the beam runs over
*cover lanes*, each a fixed cover. Lane 0 is the gold's own typed cover --
for a typo example, through the corrected spans, so the lane carries their
cost -- and each other lane is the model's best cover of some other length,
found by a length-indexed Viterbi over the automaton's legal spans
(:func:`cover_dp`) that scores a span by its best character. The batch ships
the legal spans sparsely, and the device densifies them (:func:`dense_span_table`),
so nothing in the step reads a tensor back: every shape is a host number the
collator computed (:class:`KeyRankBatch`).

**The term** is ``mean[logsumexp_{y in C + gold} S(y) - S(gold)]`` -- N-best
MMI with the reference forced in (Chow 1990; Edunov et al. 2018) -- or, with
``objective="risk"``, the expectation over the same softmax of the sentences'
normalised edit distance to the gold, minus its mean.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional

from mlime.legality import MAX_ARC_KEYS, Legality, LegalityOptions
from mlime.train.charlm_vocab import BOS
from mlime.train.ctc import alignment_scores, constrained_ctc
from mlime.train.e2e_model import KeyRankingConfig
from mlime.train.e2e_search import (
    KeySearchInputs,
    SearchGraphs,
    padded_key_inputs,
    search_cover_lanes,
)
from mlime.train.key_model import KeyE2EModel, KeyStates
from mlime.train.keys import KeyBatch, KeyCollator, KeyExample
from mlime.train.model import RouteAOutput, mark_dynamic
from mlime.train.ranking import packed_path_scores
from mlime.train.samples import TrainingExample

_NEGATIVE = float("-inf")


@dataclass(frozen=True)
class KeyRankBatch:
    """A key batch, and the legal spans of its keys: what the sequence-level term reads.

    ``keys`` is the joint stage's own batch. The legal spans are shipped
    sparsely, as ``(row, character, cost)`` triples over the batch's
    ``unique_spans`` distinct key strings (``span_rows``, ``span_ids``,
    ``span_costs``, all ``[N]``), and ``span_index [B, T, D]`` names the row of
    the span that ends at key ``t`` after ``d + 1`` keys, ``-1`` where no
    character matches it or the span runs off the record -- the arc table's
    layout (:meth:`~mlime.legality.GoldArcs.dense`). Everything the device
    needs to size a tensor is a host number computed with the batch:
    ``cover_cap``, the most characters a candidate may have (the longest gold
    plus two), and ``steps``, the most real decoder steps the scored sentences
    can have.
    """

    keys: KeyBatch
    span_rows: torch.Tensor
    span_ids: torch.Tensor
    span_costs: torch.Tensor
    span_index: torch.Tensor
    unique_spans: int
    cover_cap: int
    steps: int

    @property
    def scored(self) -> torch.Tensor:
        """The gold characters, as the loop weighs micro-batches by."""
        return self.keys.scored

    @property
    def size(self) -> int:
        """Examples in the step."""
        return self.keys.size

    @property
    def tokens(self) -> int:
        """Padded fill-tower positions."""
        return self.keys.tokens

    @property
    def context_tokens(self) -> int:
        """Padded context-tower positions."""
        return self.keys.context_tokens

    def to(self, device: torch.device) -> KeyRankBatch:
        """The same batch on *device*."""
        return replace(
            self,
            keys=self.keys.to(device),
            **{name: getattr(self, name).to(device, non_blocking=True) for name in _SPAN_TENSORS},
        )

    def pin_memory(self) -> KeyRankBatch:
        """The same batch in page-locked host memory."""
        return replace(
            self,
            keys=self.keys.pin_memory(),
            **{name: getattr(self, name).pin_memory() for name in _SPAN_TENSORS},
        )


_SPAN_TENSORS = tuple(item.name for item in fields(KeyRankBatch) if item.name.startswith("span_"))


class KeyRankCollator:
    """Collates key examples and ships each batch's legal spans (:class:`KeyRankBatch`).

    The legal spans of a batch are the characters every substring of every
    record's keys admits, from the automaton's memoised
    :meth:`~mlime.legality.Legality.span_matches`. Offline input has no
    unfinished tail, so what a span admits does not depend on where it ends
    (:class:`~mlime.legality.LegalityOptions`). The context-dropout coin is the
    wrapped collator's.
    """

    def __init__(
        self,
        collator: KeyCollator,
        legality: Legality,
        options: LegalityOptions,
        ranking: KeyRankingConfig,
    ):
        if options.unfinished_tail:
            raise ValueError("the key ranking trains on complete input; the tail rule is off")
        self.collator = collator
        self.legality = legality
        self.options = options
        self.ranking = ranking

    @property
    def rng(self) -> random.Random:
        """The wrapped collator's generator: the state a checkpoint carries."""
        return self.collator.rng

    @property
    def max_context_tokens(self) -> int:
        """The wrapped collator's context budget, which batching reads."""
        return self.collator.max_context_tokens

    def rows(self, example: TrainingExample) -> int:
        """Rows *example* costs a step: its key row and the sentences scored with it."""
        return self.ranking.rows()

    def __call__(self, examples: Sequence[TrainingExample]) -> KeyRankBatch:
        """Collate *examples* with the legal spans of their keys."""
        batch = self.collator(examples)
        keyed = [example for example in examples if isinstance(example, KeyExample)]
        size, width = len(keyed), int(batch.key_mask.shape[1])
        index = np.full((size, width, MAX_ARC_KEYS), -1, dtype=np.int64)
        rows: dict[str, int] = {}
        pieces: list[tuple[int, npt.NDArray[np.int32], npt.NDArray[np.float64]]] = []
        for owner, example in enumerate(keyed):
            typed = example.keys
            for end in range(1, len(typed) + 1):
                for span_width in range(1, min(end, MAX_ARC_KEYS) + 1):
                    span = typed[end - span_width : end]
                    row = rows.get(span)
                    if row is None:
                        matches = self.legality.span_matches(span, False, self.options)
                        row = -1
                        if matches.ids.size:
                            row = len(pieces)
                            pieces.append((row, matches.ids, matches.costs))
                        rows[span] = row
                    index[owner, end - 1, span_width - 1] = row
        sizes = [ids.size for _, ids, _ in pieces]
        lengths = [len(example.targets) for example in keyed]
        negatives = self.ranking.negatives
        return KeyRankBatch(
            keys=batch,
            span_rows=torch.from_numpy(np.repeat(np.arange(len(pieces), dtype=np.int64), sizes)),
            span_ids=torch.from_numpy(
                np.concatenate([ids for _, ids, _ in pieces]).astype(np.int64)
            ),
            span_costs=torch.from_numpy(
                np.concatenate([costs for _, _, costs in pieces]).astype(np.float32)
            ),
            span_index=torch.from_numpy(index),
            unique_spans=len(pieces),
            cover_cap=max(lengths) + 2,
            steps=sum(length + negatives * (length + 2) for length in lengths),
        )


def dense_span_table(batch: KeyRankBatch, emittable: int) -> torch.Tensor:
    """``[U + 1, E]``: the correction cost of every legal span-character pair, infinity elsewhere.

    Row ``u`` is the batch's ``u``-th distinct span; the extra last row is
    all infinity and stands for a span nothing matches. Built on the device
    from the sparse triples, in float32: the cost enters the score, which the
    search computes in doubles.
    """
    device = batch.span_rows.device
    table = torch.full((batch.unique_spans + 1, emittable), float("inf"), device=device)
    table[batch.span_rows, batch.span_ids] = batch.span_costs
    return table


def span_rows_of(batch: KeyRankBatch) -> torch.Tensor:
    """``span_index`` with the spans nothing matches sent to the table's all-infinity row."""
    return torch.where(batch.span_index < 0, batch.unique_spans, batch.span_index)


def priced(cost: torch.Tensor, gamma: float) -> torch.Tensor:
    """*cost* scaled by *gamma*, with an absent arc's infinity kept at ``gamma`` of 0 too."""
    return torch.where(torch.isinf(cost), cost, cost * gamma)


def span_gains(
    log_probs: torch.Tensor,
    key_index: torch.Tensor,
    table: torch.Tensor,
    spans: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """``[B, T, D]``: the best character's gain for every span, minus infinity where none.

    Span ``(b, t, d)`` ends at key ``t`` after ``d + 1`` keys. Its gain is
    the search's: the head's log-probability of the character at key ``t``
    less the blank's, less ``gamma`` times the span's correction cost,
    maximised over the characters the span admits (the transition's share is
    left out). ``log_probs [K, E + 1]`` holds every real key of the batch,
    ``key_index [B, T]`` names each row's keys among them, and ``table`` and
    ``spans`` are :func:`dense_span_table` and :func:`span_rows_of`.
    """
    rows, width = key_index.shape
    emittable = table.shape[1]
    keyed = log_probs.index_select(0, key_index.reshape(-1)).view(rows, width, -1)
    emit, blank = keyed[..., :emittable], keyed[..., emittable]
    best = []
    for span in range(spans.shape[2]):
        cost = table.index_select(0, spans[:, :, span].reshape(-1)).view(rows, width, emittable)
        best.append((emit - priced(cost, gamma)).amax(dim=-1) - blank)
    return torch.stack(best, dim=2)


def cover_dp(gain: torch.Tensor, keys: torch.Tensor, cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The best cover of the keys by exactly ``j`` spans, for every ``j`` up to *cap*.

    *gain* ``[B, T, D]`` is :func:`span_gains` and *keys* ``[B]`` each
    record's key count. Returns ``value [B, cap + 1]``, the best total gain
    of a cover of each record's keys by ``j`` spans (minus infinity where
    there is none), and ``back [B, T + 1, cap + 1]``, the width of the last
    span of the best cover of the first ``e`` keys by ``j`` spans. A
    Viterbi over (end, length) in ``T`` steps of one batched operation each.
    """
    rows, width, spans = gain.shape
    device = gain.device
    # Offsets are shifted by the widest span, so that a span reaching before
    # the first key reads minus infinity instead of wrapping.
    best = gain.new_full((rows, width + spans + 1, cap + 1), _NEGATIVE)
    best[:, spans, 0] = 0.0
    back = torch.zeros((rows, width + 1, cap + 1), dtype=torch.long, device=device)
    for end in range(1, width + 1):
        # Position p of the flipped window is the cover of the first end - 1 - p keys.
        before = best[:, end : end + spans, :cap].flip(1)
        top, which = (before + gain[:, end - 1].unsqueeze(2)).max(dim=1)
        best[:, spans + end, 1:] = top
        back[:, end, 1:] = which + 1
    value = best[torch.arange(rows, device=device), spans + keys]
    return value, back


@dataclass(frozen=True)
class Lanes:
    """The covers the candidate search runs over, lane-major: lane ``l * B + b`` is example ``b``'s.

    ``ends [L, M]`` and ``widths [L, M]`` are each lane's spans in order, as
    the key offset one past the span's last key and its key count (1 past the
    lane's ``lengths [L]``); ``owner [L]`` its example; ``usable [L]`` whether
    the lane is a cover at all -- an other-length lane of an example that has
    no cover of any other length is not.
    """

    owner: torch.Tensor
    lengths: torch.Tensor
    ends: torch.Tensor
    widths: torch.Tensor
    usable: torch.Tensor


def cover_lanes(
    value: torch.Tensor,
    back: torch.Tensor,
    keys: torch.Tensor,
    gold_end: torch.Tensor,
    gold_length: torch.Tensor,
    other_lengths: int,
    cap: int,
) -> Lanes:
    """Lane 0 the gold's typed cover, then the best cover of each of the best other lengths.

    *value* and *back* are :func:`cover_dp`'s, *gold_end* ``[B, Mg]`` is each
    typed span's last key (:attr:`~mlime.train.keys.KeyBatch.span_end`) and
    *gold_length* ``[B]`` the sentence lengths. An other length is within two
    of the gold's, at most *cap*, and not the gold's own; the lanes take the
    *other_lengths* best by the cover's gain. Backtracking runs on the
    device, a step per position.
    """
    rows, golds = gold_end.shape
    device = value.device
    length = torch.arange(cap + 1, device=device)
    allowed = (
        (length >= 1)
        & (length <= (gold_length + 2).unsqueeze(1))
        & (length != gold_length.unsqueeze(1))
    )
    chosen = value.masked_fill(~allowed, _NEGATIVE).topk(other_lengths, dim=1)
    lengths = torch.cat([gold_length, chosen.indices.t().reshape(-1)])
    owner = torch.arange(rows, device=device).repeat(1 + other_lengths)
    usable = torch.cat(
        [
            torch.ones(rows, dtype=torch.bool, device=device),
            (chosen.values > _NEGATIVE).t().reshape(-1),
        ]
    )

    ends = torch.ones((owner.numel(), cap), dtype=torch.long, device=device)
    widths = torch.ones_like(ends)
    cursor, count = keys[owner], lengths
    for step in range(cap - 1, -1, -1):
        live = step < lengths
        width = back[owner, cursor, count].clamp(min=1)
        ends[:, step] = torch.where(live, cursor, 1)
        widths[:, step] = torch.where(live, width, 1)
        cursor = torch.where(live, (cursor - width).clamp(min=0), cursor)
        count = torch.where(live, (count - 1).clamp(min=0), count)

    # The gold's lane is its typed cover, not a search result.
    typed = functional.pad(gold_end + 1, (0, cap - golds), value=1)
    position = torch.arange(cap, device=device)
    real = position < gold_length.unsqueeze(1)
    spans = typed - functional.pad(typed[:, :-1], (1, 0))
    ends[:rows] = torch.where(real, typed, 1)
    widths[:rows] = torch.where(real, spans, 1)
    return Lanes(owner=owner, lengths=lengths, ends=ends, widths=widths, usable=usable)


def lane_gains(
    log_probs: torch.Tensor,
    key_index: torch.Tensor,
    table: torch.Tensor,
    spans: torch.Tensor,
    lanes: Lanes,
    gamma: float,
) -> torch.Tensor:
    """``[L, M, E]``: each lane position's gain for every character, minus infinity if not admitted.

    The search's gain of emitting a character at the span the lane puts at that
    position: its emission log-probability at the span's last key less the
    blank's, less ``gamma`` times the correction cost. Positions past a lane's
    length hold whatever the span there says (the search's free steps ignore
    them), and an unusable lane admits nothing.
    """
    emittable = table.shape[1]
    owner = lanes.owner.unsqueeze(1)
    last = lanes.ends - 1
    keyed = log_probs.index_select(0, key_index[owner, last].reshape(-1))
    span = spans[owner, last, lanes.widths - 1]
    cost = table.index_select(0, span.reshape(-1))
    gain = keyed[:, :emittable] - keyed[:, emittable].unsqueeze(1) - priced(cost, gamma)
    gain = gain.view(*lanes.ends.shape, emittable)
    return gain.masked_fill(~lanes.usable.view(-1, 1, 1), _NEGATIVE)


@dataclass(frozen=True)
class KeyItems:
    """The sentences one step scores, on the device: the gold first, then each example's negatives.

    Item ``b < B`` is example ``b``'s gold sentence; item ``B + b * K + k``
    its ``k``-th best negative, or the gold again where it has fewer than
    ``K`` (``valid [B, K]`` is false there, and the term ignores the slot).
    ``sequences [I, M]`` are emission indices, zero past each item's
    ``lengths [I]``, and ``owner [I]`` the example each belongs to.
    """

    owner: torch.Tensor
    sequences: torch.Tensor
    lengths: torch.Tensor
    valid: torch.Tensor


def select_key_items(
    sequences: torch.Tensor,
    scores: torch.Tensor,
    lanes: Lanes,
    gold: torch.Tensor,
    gold_length: torch.Tensor,
    negatives: int,
) -> KeyItems:
    """Merge each example's lane beams and pick the sentences to rank, on the device.

    *sequences* ``[L, W, M]`` and *scores* ``[L, W]`` are the search's, one
    lane per row of *lanes*; a score of minus infinity is a slot the lane
    never filled. A sentence two lanes both found is one answer, scored by the
    better of them and found first, by lane then beam, on a tie; the gold
    sentence is never a negative. Each example keeps its *negatives* best
    sentences, best first by score then by where it was found, so the choice
    is fully determined. Every shape is a host number, so nothing is read
    back.
    """
    examples, golds = gold.shape
    device = scores.device
    group, width, span = sequences.shape
    per = group // examples
    hypotheses = per * width
    position = torch.arange(span, device=device)

    said = sequences.reshape(per, examples, width, span).transpose(0, 1).reshape(examples, -1, span)
    score = scores.reshape(per, examples, width).transpose(0, 1).reshape(examples, hypotheses)
    length = (
        lanes.lengths.view(per, examples)
        .t()
        .unsqueeze(2)
        .expand(examples, per, width)
        .reshape(examples, hypotheses)
    )
    said = said.masked_fill(position >= length.unsqueeze(2), -1)
    gold = functional.pad(gold, (0, span - golds)).masked_fill(
        position >= gold_length.unsqueeze(1), -1
    )

    valid = score > _NEGATIVE
    same = (said.unsqueeze(2) == said.unsqueeze(1)).all(3) & valid.unsqueeze(2) & valid.unsqueeze(1)
    mine, theirs = score.unsqueeze(2), score.unsqueeze(1)
    columns = torch.arange(hypotheses, device=device)
    ahead = (theirs > mine) | ((theirs == mine) & (columns.unsqueeze(0) < columns.unsqueeze(1)))
    kept = valid & ~(same & ahead).any(2)
    candidate = kept & ~(said == gold.unsqueeze(1)).all(2)
    rank = (ahead & candidate.unsqueeze(1)).sum(2)
    chosen = candidate & (rank < negatives)

    items = examples * (1 + negatives)
    owner = torch.arange(examples, device=device)
    target = torch.where(
        chosen,
        examples + owner.unsqueeze(1) * negatives + rank,
        torch.full_like(rank, items),
    ).reshape(-1)
    base = torch.cat([gold, gold.repeat_interleave(negatives, dim=0)])
    base_length = torch.cat([gold_length, gold_length.repeat_interleave(negatives)])
    picked = torch.cat([base, base[:1]]).scatter(
        0, target.unsqueeze(1).expand(-1, span), said.reshape(-1, span)
    )[:items]
    picked_length = torch.cat([base_length, base_length[:1]]).scatter(
        0, target, length.reshape(-1)
    )[:items]
    flagged = torch.zeros(items + 1, dtype=torch.bool, device=device).scatter(
        0, target, chosen.reshape(-1)
    )[:items]
    return KeyItems(
        owner=torch.cat([owner, owner.repeat_interleave(negatives)]),
        sequences=picked.clamp(min=0),
        lengths=picked_length,
        valid=flagged[examples:].view(examples, negatives),
    )


def item_arc_costs(
    table: torch.Tensor, spans: torch.Tensor, owner: torch.Tensor, sequences: torch.Tensor
) -> torch.Tensor:
    """``[I, M, T, D]``: the arc table of each item's sentence over its record's keys.

    Entry ``[i, j, t, d]`` is the correction cost of emitting character ``j``
    of item ``i`` at key ``t`` after ``d + 1`` keys, infinity where the span
    does not admit it: :meth:`~mlime.legality.GoldArcs.dense`'s layout, but
    over every legal arc rather than the ones on a complete alignment, which
    the forward recursion cannot tell apart, gathered on the device.
    """
    return table[spans.index_select(0, owner).unsqueeze(1), sequences[:, :, None, None]]


@dataclass(frozen=True)
class SentenceScores:
    """The two parts of ``S(y)`` of each sentence: the CTC term and the summed transition."""

    nll: torch.Tensor
    transition: torch.Tensor

    def score(self, beta: float) -> torch.Tensor:
        """``S(y) = -nll + beta * transition`` ``[I]``."""
        return beta * self.transition - self.nll


def sentence_scores(
    model: KeyE2EModel,
    states: KeyStates,
    log_probs: torch.Tensor,
    key_index: torch.Tensor,
    keys: torch.Tensor,
    owner: torch.Tensor,
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    cost: torch.Tensor,
    gamma: float,
    bound: int,
) -> SentenceScores:
    """The score parts of sentences over their records' keys, with gradient.

    ``sequences [I, M]`` are emission indices (any valid one past
    ``lengths [I]``), item ``i`` reading record ``owner[i]``. ``log_probs [K,
    E + 1]``, ``key_index [B, T]`` and ``keys [B]`` are the batch's CTC
    log-probabilities, their layout and the key counts; ``cost [I, M, T, D]``
    the items' arc tables (:func:`item_arc_costs`), scaled by *gamma* here.
    The transition is the decoder teacher-forced over each sentence, reading
    every key, its whole-alphabet log-probabilities taken a chunk of steps at
    a time and recomputed in backward (:func:`packed_path_scores`); *bound*
    is the host's count of the real steps.
    """
    blank, emit = alignment_scores(log_probs, key_index.index_select(0, owner), sequences)
    nll = constrained_ctc(blank, emit, priced(cost, gamma), keys.index_select(0, owner), lengths)
    decoder = model.decoder
    ids = decoder.emit_ids[sequences]
    previous = torch.cat([torch.full_like(ids[:, :1], BOS), ids[:, :-1]], dim=1)
    steps = torch.arange(sequences.shape[1], device=sequences.device) < lengths.unsqueeze(1)
    for tensor in (previous, steps):
        torch._dynamo.maybe_mark_dynamic(tensor, (0, 1))
    features = decoder.over_keys(previous, states.states, states.mask, owner)
    reader = decoder.reader
    projected = reader.project(reader.dropout(features))
    emitted = reader.embed.weight.index_select(0, decoder.emit_ids)
    return SentenceScores(
        nll=nll, transition=packed_path_scores(projected, emitted, sequences, steps, bound)
    )


def edit_distance(
    said: torch.Tensor, lengths: torch.Tensor, gold: torch.Tensor, gold_lengths: torch.Tensor
) -> torch.Tensor:
    """``[I]``: the Levenshtein distance of each sentence to its gold, over the longer one's length.

    ``said`` and ``gold`` are ``[I, M]`` emission indices, ``lengths`` and
    ``gold_lengths`` ``[I]``. A row of the distance table is a minimum over
    its predecessor and, along the row, a running minimum of the
    candidates plus the distance walked, so the whole table is ``M``
    batched steps.
    """
    items, span = said.shape
    device = said.device
    walked = torch.arange(span + 1, device=device)
    row = walked.expand(items, span + 1)
    result = torch.zeros(items, dtype=torch.long, device=device)
    for step in range(1, span + 1):
        replace_ = row[:, :-1] + (said[:, step - 1 : step] != gold).long()
        candidate = torch.minimum(replace_, row[:, 1:] + 1)
        head = torch.full((items, 1), step, dtype=torch.long, device=device)
        along = torch.cat([head, candidate], dim=1) - walked
        row = torch.cummin(along, dim=1).values + walked
        result = torch.where(
            lengths == step, row.gather(1, gold_lengths.unsqueeze(1)).squeeze(1), result
        )
    return result.float() / torch.maximum(lengths, gold_lengths).float()


class KeyRankingModel(KeyE2EModel):
    """The joint-stage key model trained to rank whole sentences, not only keys and characters.

    Everything but :meth:`forward` is :class:`~mlime.train.key_model.KeyE2EModel`'s:
    the same module tree, the same checkpoint record (with its
    ``key_ranking`` field set), the same evaluation. The forward takes a
    :class:`KeyRankBatch`.

    ``search_graphs``, set by the trainer on a CUDA run that asks for it,
    replays the candidate search from captured CUDA graphs (#137).
    """

    search_graphs: SearchGraphs | None = None

    @property
    def ranking(self) -> KeyRankingConfig:
        """The ranking settings, which this model cannot be built without."""
        ranking = self.e2e_config.key_ranking
        if ranking is None:
            raise ValueError("a key ranking model needs a config with key ranking settings")
        return ranking

    # The training forward reads the batch's spans beside its keys; evaluation
    # goes through ``key_states`` and the search, inherited unchanged.
    def forward(self, batch: KeyRankBatch) -> RouteAOutput:  # type: ignore[override]
        """S3's two terms, and ``weight`` times the term over the scored sentences.

        Nothing here reads the device back: the candidates are found, merged
        and chosen on the device into shapes the collator computed.
        """
        if not isinstance(batch, KeyRankBatch):
            raise TypeError(f"the key ranking model trains on its own batches, got {type(batch)}")
        ranking = self.ranking
        if ranking.weight == 0.0:
            return self.joint_loss(batch.keys)
        keyed = batch.keys
        mark_dynamic(keyed.spans)
        encoded = self.encode_keys(keyed, self.encode_context(keyed.spans))
        hidden = encoded.shape[-1]
        log_probs = self.ctc_log_probs(
            encoded.reshape(-1, hidden).index_select(0, keyed.key_positions)
        )
        gold_nll, blank, emit = self.gold_alignment(keyed, log_probs)
        states = self.states_at_keys(keyed.key_mask, encoded)

        table = dense_span_table(batch, self.decoder.emit_ids.numel())
        spans = span_rows_of(batch)
        with torch.profiler.record_function("key_rank.candidates"):
            items = self.candidates(batch, states, log_probs.detach(), table, spans)
        with torch.profiler.record_function("key_rank.scores"):
            parts = sentence_scores(
                self,
                states,
                log_probs,
                keyed.key_index,
                keyed.keys,
                items.owner,
                items.sequences,
                items.lengths,
                item_arc_costs(table, spans, items.owner, items.sequences),
                ranking.gamma,
                batch.steps,
            )

        scored = keyed.scored.numel()
        examples = keyed.size
        ctc = gold_nll.sum() / scored
        transition = -parts.transition[:examples].sum() / scored
        term, extras = self.sequence_term(items, parts, ranking)
        loss = ctc + self.keys_config.transition_weight * transition + ranking.weight * term
        return RouteAOutput(
            loss=loss,
            extras={
                "ctc_loss": ctc.detach(),
                "transition_loss": transition.detach(),
                f"{ranking.objective}_loss": term.detach(),
                **extras,
                **self.components(keyed, blank, emit),
            },
        )

    def candidates(
        self,
        batch: KeyRankBatch,
        states: KeyStates,
        log_probs: torch.Tensor,
        table: torch.Tensor,
        spans: torch.Tensor,
    ) -> KeyItems:
        """The gold and each example's negatives: the search over cover lanes, no gradient.

        The decoder runs in evaluation mode, as the search it stands for does.
        """
        ranking = self.ranking
        keyed = batch.keys
        cap = batch.cover_cap
        if ranking.other_lengths > cap:
            raise ValueError(
                f"{ranking.other_lengths} other-length lanes of covers of at most {cap} spans"
            )
        decoder = self.decoder
        training = decoder.training
        decoder.eval()
        with torch.no_grad():
            gain = span_gains(log_probs, keyed.key_index, table, spans, ranking.gamma)
            value, back = cover_dp(gain, keyed.keys, cap)
            lanes = cover_lanes(
                value, back, keyed.keys, keyed.span_end, keyed.lengths, ranking.other_lengths, cap
            )
            gains = lane_gains(log_probs, keyed.key_index, table, spans, lanes, ranking.gamma)
            inputs = padded_key_inputs(
                KeySearchInputs(
                    states=states.states.detach(),
                    key_mask=states.mask,
                    lane_owner=lanes.owner,
                    gains=gains,
                    lane_lengths=lanes.lengths,
                    width=ranking.beam,
                    beta=ranking.beta,
                ),
                bucketed=self.search_graphs is not None,
            )
            sequences, scores = search_cover_lanes(decoder, inputs, self.search_graphs)
            count = lanes.owner.numel()
            items = select_key_items(
                sequences[:count, :, :cap],
                scores[:count],
                lanes,
                keyed.gold,
                keyed.lengths,
                ranking.negatives,
            )
        decoder.train(training)
        # Autocast caches each weight's low-precision cast for the rest of its
        # region, and the casts the search just made carry no graph: reused
        # by the scoring pass, they would cut the head out of the loss.
        torch.clear_autocast_cache()
        return items

    def sequence_term(
        self, items: KeyItems, parts: SentenceScores, ranking: KeyRankingConfig
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The ranking or the risk term over the scored sentences, and the metrics behind it.

        ``rank_accuracy`` is the share of examples whose gold outscores their
        best negative, ``negatives_found`` the mean count of negatives, and
        ``beta_probe`` the term's gradient with respect to ``beta`` -- the
        expected transition under the softmax less the gold's -- which says
        which way the stage would move ``beta`` were it a parameter.
        """
        examples, negatives = items.valid.shape
        score = parts.score(ranking.beta)
        slots = torch.cat(
            [score[:examples].unsqueeze(1), score[examples:].view(examples, negatives)], dim=1
        )
        listed = torch.cat([torch.ones_like(items.valid[:, :1]), items.valid], dim=1)
        slots = slots.masked_fill(~listed, _NEGATIVE)
        if ranking.objective == "rank":
            term = (slots.logsumexp(dim=1) - slots[:, 0]).mean()
        else:
            gold = items.sequences.index_select(0, items.owner)
            gold_lengths = items.lengths.index_select(0, items.owner)
            distance = edit_distance(items.sequences, items.lengths, gold, gold_lengths)
            risk = torch.cat(
                [distance[:examples].unsqueeze(1), distance[examples:].view(examples, negatives)],
                dim=1,
            )
            mean = (risk * listed).sum(dim=1, keepdim=True) / listed.sum(dim=1, keepdim=True)
            term = (slots.softmax(dim=1) * (risk - mean)).sum(dim=1).mean()
        with torch.no_grad():
            summed = torch.cat(
                [
                    parts.transition[:examples].unsqueeze(1),
                    parts.transition[examples:].view(examples, negatives),
                ],
                dim=1,
            ).masked_fill(~listed, 0.0)
            chosen = slots.detach().softmax(dim=1)
            chosen[:, 0] -= 1.0
            beaten = slots[:, 1:].amax(dim=1)
            extras = {
                "rank_accuracy": (slots[:, 0] > beaten).float().mean(),
                "negatives_found": items.valid.float().sum(dim=1).mean(),
                "beta_probe": (chosen * summed).sum(dim=1).mean(),
            }
        return term, extras
