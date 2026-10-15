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
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mlime.train.charlm_vocab import BOS
from mlime.train.e2e_model import E2EModel, RankingConfig
from mlime.train.e2e_search import Lane, decode_lanes
from mlime.train.model import RouteAOutput, mark_dynamic, restricted_cross_entropy
from mlime.train.samples import IGNORE_INDEX, Batch, Collator, TrainingExample
from mlime.train.segmenter import Segmentations
from mlime.train.spans import SpanVocab


@dataclass(frozen=True)
class RankBatch:
    """Every reading of every example in one step, and which example each row is.

    ``rows`` is a plain :class:`Batch` over the readings. Each example's gold
    reading comes first among its rows and is the only one carrying targets,
    so ``rows.scored`` is exactly the gold positions -- v1's per-position and
    auxiliary losses read it unchanged. The context tower runs once per
    example (``rows.context_rows``). The host-side tuples are what the search
    and the item layout need without reading anything back from the device:
    ``owners`` the example of each row, ``gold_rows`` the row of each
    example's gold reading, ``lengths`` each row's typed positions and
    ``gold`` each example's gold sentence as emission indices.
    """

    rows: Batch
    owners: tuple[int, ...]
    gold_rows: tuple[int, ...]
    lengths: tuple[int, ...]
    gold: tuple[tuple[int, ...], ...]

    @property
    def scored(self) -> torch.Tensor:
        """The gold rows' scored positions, as the loop weighs micro-batches by."""
        return self.rows.scored

    @property
    def size(self) -> int:
        """Examples in the batch."""
        return len(self.gold)

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
        return RankBatch(
            rows=self.rows.to(device),
            owners=self.owners,
            gold_rows=self.gold_rows,
            lengths=self.lengths,
            gold=self.gold,
        )


class RankCollator:
    """Types each example every way its keystrokes read, then collates the rows.

    The segmenter reads the keystrokes the example's spans spell, joined; the
    gold reading leads each example's rows whether or not the segmenter's
    k-best holds it, and a reading identical to it is not repeated. The
    context-dropout coin is the wrapped collator's, thrown once per example.
    """

    def __init__(self, collator: Collator, segment: Segmentations, spans: SpanVocab):
        self.collator = collator
        self.segment = segment
        self.spans = spans

    @property
    def rng(self) -> random.Random:
        """The wrapped collator's generator: the state a checkpoint carries."""
        return self.collator.rng

    @property
    def max_context_tokens(self) -> int:
        """The wrapped collator's context budget, which batching reads."""
        return self.collator.max_context_tokens

    def __call__(self, examples: Sequence[TrainingExample]) -> RankBatch:
        """Collate *examples* with every reading of each."""
        rows: list[TrainingExample] = []
        owners: list[int] = []
        gold_rows: list[int] = []
        for owner, example in enumerate(examples):
            gold_rows.append(len(rows))
            rows.append(example)
            for index, reading in enumerate(self.segment("".join(example.spans))):
                if reading == example.spans:
                    continue
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
        return RankBatch(
            rows=self.collator.collate(rows, [example.context for example in examples], owners),
            owners=tuple(owners),
            gold_rows=tuple(gold_rows),
            lengths=tuple(len(row) for row in rows),
            gold=tuple(example.targets for example in examples),
        )


@dataclass(frozen=True)
class Items:
    """The paths one step scores: which row each reads, what it says, how it ranks.

    Item ``b`` for ``b < B`` is example ``b``'s gold path on its gold row, so
    the gold rows' per-position loss reads the first ``B`` items' logits.
    ``slots [B, M]`` lists each example's items (``-1`` past the end) and
    ``positive [B, M]`` marks the ones that say the gold sentence.
    """

    rows: list[int]
    sequences: list[tuple[int, ...]]
    slots: list[list[int]]
    positive: list[list[bool]]


def select_items(
    batch: RankBatch, found: Sequence[Sequence[tuple[Sequence[int], float]]], negatives: int
) -> Items:
    """Merge each example's beam hypotheses over its readings, and pick what to rank.

    A sentence two readings both produce is one answer, scored by the better
    of them -- the evaluation's own merge. Beside the gold path, an example
    ranks its ``negatives`` best sentences that are not the gold one, and the
    gold sentence's best reading when the beam found it on a row other than
    the gold one: either reading is the right answer.
    """
    best: list[dict[tuple[int, ...], tuple[float, int]]] = [{} for _ in batch.gold]
    for row, hypotheses in enumerate(found):
        merged = best[batch.owners[row]]
        for sequence, score in hypotheses:
            key = tuple(sequence)
            if score > merged.get(key, (float("-inf"), -1))[0]:
                merged[key] = (score, row)

    rows = list(batch.gold_rows)
    sequences = list(batch.gold)
    slots: list[list[int]] = [[example] for example in range(len(batch.gold))]
    positive: list[list[bool]] = [[True] for _ in batch.gold]
    for example, merged in enumerate(best):
        gold = batch.gold[example]
        if gold in merged and merged[gold][1] != batch.gold_rows[example]:
            slots[example].append(len(rows))
            positive[example].append(True)
            rows.append(merged[gold][1])
            sequences.append(gold)
        ranked = sorted(
            (
                (score, row, sequence)
                for sequence, (score, row) in merged.items()
                if sequence != gold
            ),
            reverse=True,
        )
        for _, row, sequence in ranked[:negatives]:
            slots[example].append(len(rows))
            positive[example].append(False)
            rows.append(row)
            sequences.append(sequence)
    width = max(len(listed) for listed in slots)
    return Items(
        rows=rows,
        sequences=sequences,
        slots=[listed + [-1] * (width - len(listed)) for listed in slots],
        positive=[marks + [False] * (width - len(marks)) for marks in positive],
    )


class RankingE2EModel(E2EModel):
    """The end-to-end model trained to rank whole sentences, not only positions.

    Everything but :meth:`forward` is :class:`E2EModel`'s: the same module
    tree, the same checkpoint record (with its ``ranking`` field set), the
    same evaluation. The forward takes a :class:`RankBatch` instead of a
    plain one.
    """

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
        """Per-position, auxiliary and ranking losses over every reading of the batch."""
        ranking = self.ranking()
        rows = batch.rows
        mark_dynamic(rows)
        hidden = self.encode(rows)
        width = rows.targets.shape[1]
        spans = hidden[:, 1 : width - 1]
        admitted = self._admitted(rows)

        decoder = self.decoder
        training = decoder.training
        decoder.eval()
        with torch.no_grad():
            detached = spans.detach()
            found = decode_lanes(
                self,
                [
                    Lane(row, detached[row, :length], admitted[row, :length])
                    for row, length in enumerate(batch.lengths)
                ],
                ranking.beam,
            )
        decoder.train(training)
        items = select_items(batch, found, ranking.negatives)
        paths, logits = self.path_scores(spans, items.rows, items.sequences)
        length = logits.shape[1]

        device = spans.device
        slots = torch.tensor(items.slots, dtype=torch.long).to(device, non_blocking=True)
        positive = torch.tensor(items.positive, dtype=torch.bool).to(device, non_blocking=True)
        listed = slots >= 0
        floor = torch.tensor(float("-inf"), device=device)
        scores = torch.where(listed, paths[slots.clamp(min=0)], floor)
        rank_loss = (
            scores.logsumexp(dim=1) - torch.where(positive, scores, floor).logsumexp(dim=1)
        ).mean()
        first_negative = torch.where(listed & ~positive, scores, floor).amax(dim=1)
        rank_accuracy = (scores[:, 0] > first_negative).float().mean()

        # The gold items are the first B and the gold rows alone carry targets:
        # a scored position of row r, column c is step c - 1 of r's example's
        # gold item.
        positions = rows.scored
        owners = torch.tensor(batch.owners, dtype=torch.long).to(device, non_blocking=True)
        region = owners[positions // width] * length + positions % width - 1
        decoder_logits = logits.reshape(-1, logits.shape[-1]).index_select(0, region)
        candidates = self.candidates_at(rows, positions)
        targets = rows.targets.reshape(-1).index_select(0, positions)
        encoder_logits = self.head.decoder(
            self.head.transform(hidden.reshape(-1, hidden.shape[-1]).index_select(0, positions))
        )
        smoothing = self.config.label_smoothing
        decoder_loss = restricted_cross_entropy(
            decoder_logits, targets, candidates, smoothing, check=False
        )
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
        self, spans: torch.Tensor, rows: Sequence[int], sequences: Sequence[tuple[int, ...]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each sentence's path score on its row, teacher-forced; and the step logits.

        ``spans [R, S, H]`` is the encoder's output at every row's typed
        positions; sentence *i* reads row ``rows[i]`` and says ``sequences[i]``
        in emission indices, one per typed position. Its score is what the
        search sums for it: each step's log-probability over the whole
        emittable set. The logits come back ``[I, T, E]`` for the per-position
        loss.
        """
        decoder = self.decoder
        device = spans.device
        length = max(len(sequence) for sequence in sequences)
        padded = torch.zeros((len(sequences), length), dtype=torch.long)
        steps = torch.zeros((len(sequences), length), dtype=torch.bool)
        for item, sequence in enumerate(sequences):
            padded[item, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            steps[item, : len(sequence)] = True
        said = padded.to(device, non_blocking=True)
        steps = steps.to(device, non_blocking=True)
        read = torch.tensor(rows, dtype=torch.long).to(device, non_blocking=True)

        reader_ids = decoder.emit_ids[said]
        previous = torch.cat([torch.full_like(reader_ids[:, :1], BOS), reader_ids[:, :-1]], dim=1)
        features = decoder(previous, spans[read, :length], steps)
        reader = decoder.reader
        emitted = reader.embed.weight.index_select(0, decoder.emit_ids)
        logits: torch.Tensor = reader.project(reader.dropout(features)) @ emitted.T
        stepped = logits.float().log_softmax(dim=-1).gather(-1, said.unsqueeze(-1)).squeeze(-1)
        return stepped.masked_fill(~steps, 0.0).sum(dim=1), logits

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
