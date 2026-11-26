"""Unified offline + on-policy distillation into the key-level student (S6, #191, #350).

Consumes the per-shard Zstd Parquet files written by ``mlime train label-teacher``
(S5, :func:`mlime.train.teacher.label_corpus_shards`):

- **Gold anchor + T1 (per-position top-k KL)**: keeps the student's constrained
  CTC and gold transition CE anchored while aligning the decoder's teacher-forced
  transition distribution ``log P_S(. | y_<j, keys, prelude)`` on gold positions
  (including ``<eos>`` when ``score_eos`` is on) to the 27B teacher's stored
  top-k distribution ``(t1_ids, t1_log_probs)``.
- **T2 + T3 (residual listwise KL + optional hard-negative MMI)**: scores the
  gold sentence plus the 27B teacher's lattice-constrained beam-8 negatives (T3)
  under the student's exact sentence score ``S_student(y)`` in a single vectorized
  pass (:func:`~mlime.train.key_ranking.sentence_scores`) and minimises residual
  KL ``KL(softmax(sg(S_student) + t2_scale * s_T) || softmax(S_student))`` via
  :func:`~mlime.train.key_teacher.distillation`, preserving the student's CTC and
  typo-cost dynamic range.
- **On-policy student beam MMI + optional frozen Char-LM calibration**: runs the
  student's own constrained cover-lane beam search (reusing
  :mod:`~mlime.train.key_ranking` and :class:`~mlime.train.e2e_search.SearchGraphs`)
  to penalise the student's own beam negatives and calibrate them against a frozen
  character LM (:class:`~mlime.train.key_teacher.Teacher`), preventing transition
  score inflation on typo and abbreviation slices.
"""

from __future__ import annotations

import json
import math
import queue
import random
import threading
import time
import traceback
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
import torch.multiprocessing as mp
import torch.nn.functional as functional

from mlime.legality import MAX_ARC_KEYS, OFFLINE, Legality, LegalityOptions
from mlime.logging import log
from mlime.train.ctc import alignment_scores, constrained_ctc
from mlime.train.e2e_search import (
    KeySearchInputs,
    SearchGraphs,
    padded_key_inputs,
    search_cover_lanes,
)
from mlime.train.key_model import KeyE2EModel, KeyStates
from mlime.train.key_ranking import (
    by_example,
    cover_dp,
    cover_lanes,
    item_arc_costs,
    lane_gains,
    priced,
    select_key_items,
    sentence_scores,
    span_gains,
)
from mlime.train.key_teacher import (
    Teacher,
    TeacherContext,
    TeacherVocab,
    distillation,
    teacher_context,
)
from mlime.train.keys import KeyCollator, KeyInputs
from mlime.train.qwen_tower import QwenContextConfig


@dataclass(frozen=True)
class DistillConfig:
    """Loss weights and search settings for Phase 2 S6 unified distillation (#191, #350)."""

    #: Weight on gold constrained CTC anchor.
    ctc_weight: float = 1.0
    #: Weight on gold transition cross-entropy beside T1 KL.
    ce_weight: float = 0.5
    #: Weight on per-position top-k KL (T1) from the offline 27B teacher.
    t1_weight: float = 0.5
    #: Softmax temperature applied to the teacher's T1 log-probs over top-k support.
    t1_temperature: float = 1.9
    #: Weight on residual listwise KL (T2) over gold + 27B beam-8 negatives (T3).
    t2_weight: float = 1.0
    #: Residual scale ``w`` in ``q = softmax(sg(S) + w * s_T)`` for T2 KL.
    t2_scale: float = 0.6
    #: Weight on hard-negative MMI over ``{gold} U T3``.
    rank_t3_weight: float = 0.0
    #: Weight on on-policy student beam MMI over ``{gold} U student_beam``.
    rank_online_weight: float = 0.05
    #: Weight on frozen Char-LM residual KL over the student's on-policy beam.
    lm_weight: float = 1.0
    #: Residual scale ``w`` for the frozen Char-LM teacher over the on-policy beam.
    lm_scale: float = 0.6
    #: Transition weight ``beta`` in the student's sentence score ``S(y)``.
    beta: float = 2.0
    #: Correction-cost weight ``gamma`` in the student's sentence score ``S(y)``.
    gamma: float = 1.0
    #: Online beam width for student negatives.
    beam: int = 8
    #: Number of online student negatives retained per example.
    negatives: int = 8
    #: Extra cover-length lanes explored beside the gold's typed cover.
    other_lengths: int = 1

    def __post_init__(self) -> None:
        if self.t1_temperature <= 0.0:
            raise ValueError("t1_temperature must be positive")
        for name in (
            "ctc_weight",
            "ce_weight",
            "t1_weight",
            "t2_weight",
            "t2_scale",
            "rank_t3_weight",
            "rank_online_weight",
            "lm_weight",
            "lm_scale",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.beam < 1 or self.negatives < 1 or self.other_lengths < 0:
            raise ValueError("beam and negatives must be >= 1 and other_lengths >= 0")

    @property
    def uses_online_beam(self) -> bool:
        """Whether a step needs the student's on-policy constrained beam search."""
        return self.rank_online_weight > 0.0 or self.lm_weight > 0.0

    @property
    def uses_t3_candidates(self) -> bool:
        """Whether a step scores the offline T3 candidate list."""
        return self.t2_weight > 0.0 or self.rank_t3_weight > 0.0


@dataclass(frozen=True)
class DistillTrainConfig:
    """Training loop hyperparameters for Phase 2 S6 student distillation (#191, #350)."""

    distill: DistillConfig = DistillConfig()
    base_lr: float = 1e-5
    new_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.04
    batch_size: int = 32
    accumulate: int = 4
    max_steps: int = 1562
    gradient_clip: float = 1.0
    seed: int = 0
    log_every: int = 20
    checkpoint_every: int = 500
    pool_batches: int = 32
    prefetch_batches: int = 8
    search_graph_limit: int = 192


@dataclass(frozen=True)
class TeacherLabelRow:
    """One record read from an S5 offline teacher label Parquet shard."""

    id: str
    keys: str
    text: str
    t1_ids: tuple[tuple[int, ...], ...]
    t1_log_probs: tuple[tuple[float, ...], ...]
    candidates: tuple[str, ...]
    candidate_scores: tuple[float, ...]
    context: str | None = None


def read_teacher_label_shard(path: Path) -> list[TeacherLabelRow]:
    """Read one S5 Parquet shard into :class:`TeacherLabelRow` objects."""
    table = pq.read_table(path)
    rows: list[TeacherLabelRow] = []
    for raw in table.to_pylist():
        ctx_raw = raw.get("context")
        rows.append(
            TeacherLabelRow(
                id=str(raw["id"]),
                context=None if ctx_raw is None else str(ctx_raw),
                keys=str(raw["keys"]),
                text=str(raw["text"]),
                t1_ids=tuple(tuple(int(x) for x in pos) for pos in raw["t1_ids"]),
                t1_log_probs=tuple(tuple(float(x) for x in pos) for pos in raw["t1_log_probs"]),
                candidates=tuple(str(c) for c in raw["candidates"]),
                candidate_scores=tuple(float(s) for s in raw["candidate_scores"]),
            )
        )
    return rows


def iter_teacher_label_shards(shards_dir: Path) -> Iterator[TeacherLabelRow]:
    """Yield every :class:`TeacherLabelRow` across sorted ``*.parquet`` shards in *shards_dir*."""
    paths = sorted(p for p in shards_dir.glob("*.parquet") if not p.name.startswith("."))
    if not paths:
        raise FileNotFoundError(f"no teacher label parquet shards found under {shards_dir}")
    for path in paths:
        yield from read_teacher_label_shard(path)


class TeacherLabelShards:
    """Re-iterable view over S5 Parquet shards for multi-epoch and resumed S6 runs (#342)."""

    def __init__(self, shards_dir: Path, *, seed: int = 0) -> None:
        self.shards_dir = shards_dir
        self.seed = seed
        self._paths = sorted(p for p in shards_dir.glob("*.parquet") if not p.name.startswith("."))
        if not self._paths:
            raise FileNotFoundError(f"no teacher label parquet shards found under {shards_dir}")
        self._epoch = 0

    def __iter__(self) -> Iterator[TeacherLabelRow]:
        order = list(self._paths)
        if len(order) > 1:
            random.Random(self.seed + self._epoch).shuffle(order)
        self._epoch += 1
        for path in order:
            yield from read_teacher_label_shard(path)


def min_cost_span_ends(dense_cost: npt.NDArray[np.floating[Any]]) -> list[int]:
    """Minimum-cost legal key-span end indices for a gold sentence from ``GoldArcs.dense()``."""
    m, k, max_w = dense_cost.shape
    padded_prev = np.full(k + max_w, math.inf, dtype=np.float64)
    padded_prev[max_w - 1] = 0.0
    bp = np.zeros((m + 1, k + 1), dtype=np.int64)
    offsets = np.arange(k)[:, None] - np.arange(max_w)[None, :] + (max_w - 1)
    rows_idx = np.arange(k)
    for j in range(1, m + 1):
        cand = padded_prev[offsets] + dense_cost[j - 1]
        best_idx = np.argmin(cand, axis=1)
        bp[j, 1:] = best_idx + 1
        padded_prev[max_w - 1] = math.inf
        padded_prev[max_w:] = cand[rows_idx, best_idx]
    if not math.isfinite(float(padded_prev[max_w + k - 1])):
        raise ValueError("gold arc table has no complete legal cover over keys")
    ends = [0] * m
    cur = k
    for j in range(m, 0, -1):
        ends[j - 1] = cur - 1
        cur -= int(bp[j, cur])
    return ends


def t1_kl_loss(
    student_log_probs: torch.Tensor,
    t1_ids: torch.Tensor,
    t1_log_probs: torch.Tensor,
    *,
    temperature: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k per-position KL ``KL(P_T1 || P_S)`` over gold positions + ``<eos>``."""
    if student_log_probs.ndim != 2:
        raise ValueError(f"expected [N, V] student_log_probs, got {tuple(student_log_probs.shape)}")
    if t1_ids.shape != t1_log_probs.shape or t1_ids.ndim != 2:
        raise ValueError(
            f"t1_ids {tuple(t1_ids.shape)} and t1_log_probs {tuple(t1_log_probs.shape)} must match"
        )
    if student_log_probs.shape[0] != t1_ids.shape[0]:
        raise ValueError("position count mismatch between student_log_probs and t1_ids")

    scaled = t1_log_probs.float() / temperature
    teacher_log_p = functional.log_softmax(scaled, dim=-1)
    teacher_p = teacher_log_p.exp()
    student_at_k = student_log_probs.float().gather(1, t1_ids.long())
    per_pos = (teacher_p * (teacher_log_p - student_at_k)).sum(dim=-1)
    if mask is not None:
        active = mask.bool()
        denom = active.sum().clamp_min(1)
        return (per_pos * active.float()).sum() / denom
    return per_pos.mean()


def t2_residual_kl_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    scale: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual listwise KL ``KL(softmax(sg(S) + scale * T) || softmax(S))`` (#350).

    Preserves the student's CTC and typo-cost dynamic range by shifting the
    detached student scores by ``scale * teacher_scores`` via
    :func:`~mlime.train.key_teacher.distillation`. Rows with fewer than 2 valid
    candidates contribute zero loss. Returns ``(kl_mean, teacher_flip)``.
    """
    if student_scores.shape != teacher_scores.shape or student_scores.shape != valid_mask.shape:
        raise ValueError("student_scores, teacher_scores, and valid_mask must share shape [B, C]")
    if student_scores.ndim != 2:
        raise ValueError(f"expected [B, C] scores, got {tuple(student_scores.shape)}")

    mask = valid_mask.bool()
    has_pair = mask.sum(dim=-1) >= 2
    if not bool(has_pair.any()):
        zero = student_scores.sum() * 0.0
        return zero, zero.detach()

    slots = student_scores[has_pair].float().masked_fill(~mask[has_pair], float("-inf"))
    kl, extras = distillation(slots, mask[has_pair], teacher_scores[has_pair].float(), scale)
    return kl.mean(), extras["teacher_flip"]


def _flat_ctc_layout(
    ctc_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``[B, Wk, E + 1]`` CTC log-probs into ``([B * Wk, E + 1], key_index [B, Wk])``."""
    batch, width, classes = ctc_log_probs.shape
    device = ctc_log_probs.device
    flat_lp = ctc_log_probs.reshape(batch * width, classes)
    key_index = torch.arange(batch * width, dtype=torch.long, device=device).view(batch, width)
    return flat_lp, key_index


def score_candidate_sentences(
    model: KeyE2EModel,
    states: KeyStates,
    ctc_log_probs: torch.Tensor,
    keys_list: Sequence[str],
    candidates_per_row: Sequence[Sequence[Sequence[int]]],
    legality: Legality,
    *,
    beta: float = 2.0,
    gamma: float = 1.0,
    options: LegalityOptions = OFFLINE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable student sentence scores ``S(y)`` for each row's candidates."""
    device = states.states.device
    batch_size = len(keys_list)
    width = int(states.states.shape[1])
    max_c = max((len(cands) for cands in candidates_per_row), default=0)
    scores = torch.zeros((batch_size, max_c), dtype=torch.float32, device=device)
    valid = torch.zeros((batch_size, max_c), dtype=torch.bool, device=device)
    if max_c == 0:
        return scores, valid

    owners: list[int] = []
    slot_b: list[int] = []
    slot_c: list[int] = []
    seqs: list[list[int]] = []
    dense_costs: list[torch.Tensor] = []
    for b, (keys, cands) in enumerate(zip(keys_list, candidates_per_row, strict=True)):
        for c, emit_ids in enumerate(cands):
            if not emit_ids:
                continue
            arcs = legality.gold_ids(keys, emit_ids, options)
            if arcs is None:
                continue
            owners.append(b)
            slot_b.append(b)
            slot_c.append(c)
            seqs.append(list(emit_ids))
            dense_costs.append(torch.from_numpy(arcs.dense()).to(torch.float32))

    if not owners:
        return scores, valid

    n_items = len(owners)
    max_m = max(len(s) for s in seqs)
    cand_pad = torch.zeros((n_items, max_m), dtype=torch.long)
    cand_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    cost_grid = torch.full((n_items, max_m, width, MAX_ARC_KEYS), math.inf, dtype=torch.float32)
    for i, (seq, dense) in enumerate(zip(seqs, dense_costs, strict=True)):
        ln, k_len = dense.shape[0], dense.shape[1]
        cand_pad[i, :ln] = torch.tensor(seq, dtype=torch.long)
        cost_grid[i, :ln, :k_len] = dense

    flat_lp, key_index = _flat_ctc_layout(ctc_log_probs)
    key_counts = torch.tensor([len(k) for k in keys_list], dtype=torch.long, device=device)
    owner_t = torch.tensor(owners, dtype=torch.long, device=device)
    parts = sentence_scores(
        model,
        states,
        flat_lp,
        key_index,
        key_counts,
        owner_t,
        cand_pad.to(device),
        cand_lengths.to(device),
        cost_grid.to(device),
        gamma,
        sum(len(s) for s in seqs),
    )
    item_scores = parts.score(beta)
    b_idx = torch.tensor(slot_b, dtype=torch.long, device=device)
    c_idx = torch.tensor(slot_c, dtype=torch.long, device=device)
    scores = scores.index_put((b_idx, c_idx), item_scores)
    valid = valid.index_put((b_idx, c_idx), torch.ones(n_items, dtype=torch.bool, device=device))
    return scores, valid


@dataclass(frozen=True)
class OnlineSpanLayout:
    """Sparse legal spans and optional Char-LM context for on-policy student beam search (#350)."""

    span_end: torch.Tensor
    span_rows: torch.Tensor
    span_ids: torch.Tensor
    span_costs: torch.Tensor
    span_index: torch.Tensor
    unique_spans: int
    cover_cap: int
    online_steps: int
    teacher: TeacherContext | None = None

    def to(self, device: torch.device) -> OnlineSpanLayout:
        """Move every tensor of the online span layout to *device*."""
        return OnlineSpanLayout(
            span_end=self.span_end.to(device, non_blocking=True),
            span_rows=self.span_rows.to(device, non_blocking=True),
            span_ids=self.span_ids.to(device, non_blocking=True),
            span_costs=self.span_costs.to(device, non_blocking=True),
            span_index=self.span_index.to(device, non_blocking=True),
            unique_spans=self.unique_spans,
            cover_cap=self.cover_cap,
            online_steps=self.online_steps,
            teacher=None if self.teacher is None else self.teacher.to(device),
        )


def _dense_span_table(online: OnlineSpanLayout, emittable: int) -> torch.Tensor:
    device = online.span_rows.device
    table = torch.full((online.unique_spans + 1, emittable), float("inf"), device=device)
    table[online.span_rows, online.span_ids] = online.span_costs
    return table


def _span_rows_of(online: OnlineSpanLayout) -> torch.Tensor:
    return torch.where(online.span_index < 0, online.unique_spans, online.span_index)


@dataclass(frozen=True)
class DistillBatch:
    """Pre-collated S6 distillation batch ready for a single GPU step (#340, #350)."""

    inputs: KeyInputs
    gold: torch.Tensor
    gold_lengths: torch.Tensor
    gold_keys: torch.Tensor
    gold_cost: torch.Tensor
    gold_at: torch.Tensor
    gold_targets: torch.Tensor
    scored_chars: int
    t1_ids: torch.Tensor
    t1_log_probs: torch.Tensor
    t1_mask: torch.Tensor | None
    cand_owner: torch.Tensor
    cand_slot_b: torch.Tensor
    cand_slot_c: torch.Tensor
    cand_sequences: torch.Tensor
    cand_lengths: torch.Tensor
    cand_cost: torch.Tensor
    cand_step_bound: int
    teacher_scores: torch.Tensor
    cand_valid: torch.Tensor
    online: OnlineSpanLayout | None = None

    @property
    def size(self) -> int:
        """Number of examples in the batch."""
        return self.inputs.size

    def to(self, device: torch.device) -> DistillBatch:
        """Move every tensor of the batch to *device*."""
        return DistillBatch(
            inputs=self.inputs.to(device),
            gold=self.gold.to(device, non_blocking=True),
            gold_lengths=self.gold_lengths.to(device, non_blocking=True),
            gold_keys=self.gold_keys.to(device, non_blocking=True),
            gold_cost=self.gold_cost.to(device, non_blocking=True),
            gold_at=self.gold_at.to(device, non_blocking=True),
            gold_targets=self.gold_targets.to(device, non_blocking=True),
            scored_chars=self.scored_chars,
            t1_ids=self.t1_ids.to(device, non_blocking=True),
            t1_log_probs=self.t1_log_probs.to(device, non_blocking=True),
            t1_mask=None if self.t1_mask is None else self.t1_mask.to(device, non_blocking=True),
            cand_owner=self.cand_owner.to(device, non_blocking=True),
            cand_slot_b=self.cand_slot_b.to(device, non_blocking=True),
            cand_slot_c=self.cand_slot_c.to(device, non_blocking=True),
            cand_sequences=self.cand_sequences.to(device, non_blocking=True),
            cand_lengths=self.cand_lengths.to(device, non_blocking=True),
            cand_cost=self.cand_cost.to(device, non_blocking=True),
            cand_step_bound=self.cand_step_bound,
            teacher_scores=self.teacher_scores.to(device, non_blocking=True),
            cand_valid=self.cand_valid.to(device, non_blocking=True),
            online=None if self.online is None else self.online.to(device),
        )


def collate_distill_batch(
    rows: Sequence[TeacherLabelRow],
    collator: KeyCollator,
    legality: Legality,
    *,
    score_eos: bool = True,
    options: LegalityOptions = OFFLINE,
    include_online: bool = False,
    negatives: int = 8,
    teacher_vocab: TeacherVocab | None = None,
) -> DistillBatch | None:
    """Collate S5 :class:`TeacherLabelRow` entries into a vectorized :class:`DistillBatch`.

    Rows whose gold sentence is illegal under *legality* or whose stored T1
    positions do not cover the sentence are skipped; returns ``None`` if no
    valid rows remain. When *include_online* is true, also builds the sparse
    legal span table, gold minimum-cost cover ends, and (when *teacher_vocab*
    is given) the Char-LM :class:`~mlime.train.key_teacher.TeacherContext`.
    """
    emittable = len(legality.characters)
    valid_rows: list[tuple[TeacherLabelRow, tuple[int, ...], npt.NDArray[np.float64]]] = []
    for row in rows:
        gold_ids = legality.ids(row.text)
        if gold_ids is None or not gold_ids:
            continue
        expected_t1 = len(gold_ids) + 1 if score_eos else len(gold_ids)
        if len(row.t1_ids) < expected_t1 or len(row.t1_log_probs) < expected_t1:
            continue
        arcs = legality.gold_ids(row.keys, gold_ids, options)
        if arcs is None:
            continue
        valid_rows.append((row, gold_ids, arcs.dense()))

    if not valid_rows:
        return None

    batch_size = len(valid_rows)
    keys_list = [r.keys for r, _, _ in valid_rows]
    contexts = [r.context for r, _, _ in valid_rows]
    inputs = collator.inputs(keys_list, contexts)
    width = int(inputs.key_mask.shape[1])

    max_m = max(len(g) for _, g, _ in valid_rows)
    gold = torch.zeros((batch_size, max_m), dtype=torch.long)
    gold_lengths = torch.tensor([len(g) for _, g, _ in valid_rows], dtype=torch.long)
    gold_keys = torch.tensor([len(k) for k in keys_list], dtype=torch.long)
    gold_cost = torch.full((batch_size, max_m, width, MAX_ARC_KEYS), math.inf, dtype=torch.float32)

    gold_at_list: list[int] = []
    gold_char_targets: list[int] = []
    t1_char_ids: list[list[int]] = []
    t1_char_lps: list[list[float]] = []
    t1_eos_ids: list[list[int]] = []
    t1_eos_lps: list[list[float]] = []

    k_t1 = min(len(pos) for r, g, _ in valid_rows for pos in r.t1_ids[: len(g) + int(score_eos)])
    for b, (row, gold_ids, dense_np) in enumerate(valid_rows):
        ln = len(gold_ids)
        gold[b, :ln] = torch.tensor(gold_ids, dtype=torch.long)
        gold_cost[b, :ln, : dense_np.shape[1]] = torch.from_numpy(dense_np).to(torch.float32)
        for j in range(ln):
            gold_at_list.append(b * max_m + j)
            gold_char_targets.append(gold_ids[j])
            t1_char_ids.append(list(row.t1_ids[j][:k_t1]))
            t1_char_lps.append(list(row.t1_log_probs[j][:k_t1]))
        if score_eos:
            t1_eos_ids.append(list(row.t1_ids[ln][:k_t1]))
            t1_eos_lps.append(list(row.t1_log_probs[ln][:k_t1]))

    scored_chars = len(gold_at_list)
    if score_eos:
        gold_targets = torch.tensor(
            [*gold_char_targets, *([emittable] * batch_size)], dtype=torch.long
        )
        t1_ids_t = torch.tensor([*t1_char_ids, *t1_eos_ids], dtype=torch.long)
        t1_lps_t = torch.tensor([*t1_char_lps, *t1_eos_lps], dtype=torch.float32)
        t1_mask: torch.Tensor | None = None
    else:
        gold_targets = torch.tensor(gold_char_targets, dtype=torch.long)
        raw_ids = torch.tensor(t1_char_ids, dtype=torch.long)
        raw_lps = torch.tensor(t1_char_lps, dtype=torch.float32)
        in_vocab = raw_ids < emittable
        t1_ids_t = raw_ids.clamp(max=emittable - 1)
        t1_lps_t = torch.where(in_vocab, raw_lps, torch.full_like(raw_lps, -1e9))
        t1_mask = in_vocab.any(dim=-1)

    max_c = max((len(r.candidates) for r, _, _ in valid_rows), default=0)
    teacher_scores = torch.zeros((batch_size, max_c), dtype=torch.float32)
    cand_valid = torch.zeros((batch_size, max_c), dtype=torch.bool)
    cand_owners: list[int] = []
    cand_slot_b: list[int] = []
    cand_slot_c: list[int] = []
    cand_seqs: list[tuple[int, ...]] = []
    cand_dense: list[torch.Tensor] = []

    for b, (row, _, _) in enumerate(valid_rows):
        for c, (cand_text, t_score) in enumerate(
            zip(row.candidates, row.candidate_scores, strict=True)
        ):
            cand_ids = legality.ids(cand_text)
            if cand_ids is None or not cand_ids:
                continue
            c_arcs = legality.gold_ids(row.keys, cand_ids, options)
            if c_arcs is None:
                continue
            cand_owners.append(b)
            cand_slot_b.append(b)
            cand_slot_c.append(c)
            cand_seqs.append(cand_ids)
            cand_dense.append(torch.from_numpy(c_arcs.dense()).to(torch.float32))
            teacher_scores[b, c] = float(t_score)
            cand_valid[b, c] = True

    if cand_owners:
        n_cands = len(cand_owners)
        max_cm = max(len(s) for s in cand_seqs)
        cand_sequences = torch.zeros((n_cands, max_cm), dtype=torch.long)
        cand_lengths = torch.tensor([len(s) for s in cand_seqs], dtype=torch.long)
        cand_cost = torch.full(
            (n_cands, max_cm, width, MAX_ARC_KEYS), math.inf, dtype=torch.float32
        )
        for i, (seq, dense) in enumerate(zip(cand_seqs, cand_dense, strict=True)):
            ln, k_len = dense.shape[0], dense.shape[1]
            cand_sequences[i, :ln] = torch.tensor(seq, dtype=torch.long)
            cand_cost[i, :ln, :k_len] = dense
        cand_step_bound = sum(len(s) for s in cand_seqs)
    else:
        cand_sequences = torch.zeros((0, 1), dtype=torch.long)
        cand_lengths = torch.zeros((0,), dtype=torch.long)
        cand_cost = torch.zeros((0, 1, width, MAX_ARC_KEYS), dtype=torch.float32)
        cand_step_bound = 0

    online: OnlineSpanLayout | None = None
    if include_online:
        span_end = torch.zeros((batch_size, max_m), dtype=torch.long)
        lengths: list[int] = []
        for b, (_, gold_ids, dense_np) in enumerate(valid_rows):
            ends = min_cost_span_ends(dense_np)
            span_end[b, : len(ends)] = torch.tensor(ends, dtype=torch.long)
            lengths.append(len(gold_ids))

        index = np.full((batch_size, width, MAX_ARC_KEYS), -1, dtype=np.int64)
        span_map: dict[str, int] = {}
        pieces: list[tuple[int, npt.NDArray[np.int32], npt.NDArray[np.float64]]] = []
        for owner, (row, _, _) in enumerate(valid_rows):
            typed = row.keys
            for end in range(1, len(typed) + 1):
                for sw in range(1, min(end, MAX_ARC_KEYS) + 1):
                    sp = typed[end - sw : end]
                    r_idx = span_map.get(sp)
                    if r_idx is None:
                        matches = legality.span_matches(sp, False, options)
                        r_idx = -1
                        if matches.ids.size:
                            r_idx = len(pieces)
                            pieces.append((r_idx, matches.ids, matches.costs))
                        span_map[sp] = r_idx
                    index[owner, end - 1, sw - 1] = r_idx

        sizes = [ids.size for _, ids, _ in pieces]
        cover_cap = max(lengths) + 2
        online_steps = sum(ln + negatives * (ln + 2) for ln in lengths)
        t_ctx: TeacherContext | None = None
        if teacher_vocab is not None:
            owners = [*range(batch_size), *(o for o in range(batch_size) for _ in range(negatives))]
            t_ctx = teacher_context(contexts, teacher_vocab, owners, cover_cap)
        online = OnlineSpanLayout(
            span_end=span_end,
            span_rows=torch.from_numpy(np.repeat(np.arange(len(pieces), dtype=np.int64), sizes)),
            span_ids=torch.from_numpy(
                np.concatenate([ids for _, ids, _ in pieces]).astype(np.int64)
            ),
            span_costs=torch.from_numpy(
                np.concatenate([costs for _, _, costs in pieces]).astype(np.float32)
            ),
            span_index=torch.from_numpy(index),
            unique_spans=len(pieces),
            cover_cap=cover_cap,
            online_steps=online_steps,
            teacher=t_ctx,
        )

    return DistillBatch(
        inputs=inputs,
        gold=gold,
        gold_lengths=gold_lengths,
        gold_keys=gold_keys,
        gold_cost=gold_cost,
        gold_at=torch.tensor(gold_at_list, dtype=torch.long),
        gold_targets=gold_targets,
        scored_chars=scored_chars,
        t1_ids=t1_ids_t,
        t1_log_probs=t1_lps_t,
        t1_mask=t1_mask,
        cand_owner=torch.tensor(cand_owners, dtype=torch.long),
        cand_slot_b=torch.tensor(cand_slot_b, dtype=torch.long),
        cand_slot_c=torch.tensor(cand_slot_c, dtype=torch.long),
        cand_sequences=cand_sequences,
        cand_lengths=cand_lengths,
        cand_cost=cand_cost,
        cand_step_bound=cand_step_bound,
        teacher_scores=teacher_scores,
        cand_valid=cand_valid,
        online=online,
    )


@dataclass(frozen=True)
class DistillStepLosses:
    """Loss terms and diagnostics from one unified S6 micro-batch forward pass (#350)."""

    total: torch.Tensor
    ctc: torch.Tensor
    ce: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor
    t2_flip: torch.Tensor
    rank_t3: torch.Tensor
    rank_online: torch.Tensor
    rank_acc: torch.Tensor
    lm: torch.Tensor
    lm_flip: torch.Tensor


def distill_batch_loss(
    model: KeyE2EModel,
    batch: DistillBatch,
    config: DistillConfig,
    *,
    charlm_teacher: Teacher | None = None,
    search_graphs: SearchGraphs | None = None,
) -> DistillStepLosses:
    """Run one vectorized unified S6 distillation forward pass (#191, #350)."""
    states = model.key_states(batch.inputs)
    ctc_lp = model.ctc_log_probs(states.states)
    flat_lp, key_index = _flat_ctc_layout(ctc_lp)

    blank, emit = alignment_scores(flat_lp, key_index, batch.gold)
    gold_nll = constrained_ctc(
        blank, emit, priced(batch.gold_cost, config.gamma), batch.gold_keys, batch.gold_lengths
    )
    gold_ctc = gold_nll.sum() / max(1, batch.scored_chars)
    zero = gold_ctc.new_zeros(())

    trans_logits = model.transition_logits(states, batch.gold, batch.gold_at, batch.gold_lengths)
    trans_lp = trans_logits.float().log_softmax(dim=-1)
    gold_ce = -trans_lp.gather(1, batch.gold_targets.unsqueeze(1)).mean()

    if config.t1_weight > 0.0:
        t1_loss = t1_kl_loss(
            trans_lp,
            batch.t1_ids,
            batch.t1_log_probs,
            temperature=config.t1_temperature,
            mask=batch.t1_mask,
        )
    else:
        t1_loss = zero

    rank_online = zero
    rank_acc = zero.detach()
    lm_loss = zero
    lm_flip = zero.detach()
    if config.uses_online_beam:
        online = batch.online
        if online is None:
            raise ValueError(
                "online beam ranking or Char-LM calibration requires DistillBatch.online"
            )
        if config.lm_weight > 0.0 and (charlm_teacher is None or online.teacher is None):
            raise ValueError("lm_weight > 0 requires both charlm_teacher and online.teacher")

        emittable = int(model.decoder.emit_ids.numel())
        table_dense = _dense_span_table(online, emittable)
        spans_dense = _span_rows_of(online)
        has_prelude = model.keys_config.decoder_prelude
        prelude_width = model.decoder.reader.config.context_chars if has_prelude else None

        decoder = model.decoder
        was_training = decoder.training
        decoder.eval()
        with torch.no_grad():
            gain = span_gains(flat_lp.detach(), key_index, table_dense, spans_dense, config.gamma)
            value, back = cover_dp(gain, batch.gold_keys, online.cover_cap)
            lanes = cover_lanes(
                value,
                back,
                batch.gold_keys,
                online.span_end,
                batch.gold_lengths,
                config.other_lengths,
                online.cover_cap,
            )
            gains = lane_gains(
                flat_lp.detach(), key_index, table_dense, spans_dense, lanes, config.gamma
            )
            s_in = padded_key_inputs(
                KeySearchInputs(
                    states=states.states.detach(),
                    key_mask=states.mask,
                    lane_owner=lanes.owner,
                    gains=gains,
                    lane_lengths=lanes.lengths,
                    width=config.beam,
                    beta=config.beta,
                    prelude_chars=batch.inputs.prelude_chars if has_prelude else None,
                    prelude_lengths=batch.inputs.prelude_lengths if has_prelude else None,
                ),
                bucketed=search_graphs is not None,
                prelude_width=prelude_width,
            )
            sequences, scores = search_cover_lanes(decoder, s_in, search_graphs)
            count = lanes.owner.numel()
            items = select_key_items(
                sequences[:count, :, : online.cover_cap],
                scores[:count],
                lanes,
                batch.gold,
                batch.gold_lengths,
                config.negatives,
            )
        decoder.train(was_training)
        if states.states.device.type == "cuda":
            torch.clear_autocast_cache()

        online_parts = sentence_scores(
            model,
            states,
            flat_lp,
            key_index,
            batch.gold_keys,
            items.owner,
            items.sequences,
            items.lengths,
            item_arc_costs(table_dense, spans_dense, items.owner, items.sequences),
            config.gamma,
            online.online_steps,
        )
        examples = batch.size
        o_slots = by_example(online_parts.score(config.beta), examples, config.negatives)
        o_listed = torch.cat([torch.ones_like(items.valid[:, :1]), items.valid], dim=1)
        o_slots = o_slots.masked_fill(~o_listed, float("-inf"))
        rank_online = (o_slots.logsumexp(dim=1) - o_slots[:, 0]).mean()
        with torch.no_grad():
            rank_acc = (o_slots[:, 0] > o_slots[:, 1:].amax(dim=1)).float().mean()

        if config.lm_weight > 0.0:
            assert charlm_teacher is not None and online.teacher is not None
            taught = charlm_teacher.log_probs(
                online.teacher, items.owner, items.sequences, items.lengths
            )
            lm_kl_vec, lm_ex = distillation(
                o_slots,
                o_listed,
                by_example(taught, examples, config.negatives),
                config.lm_scale,
            )
            lm_loss = lm_kl_vec.mean()
            lm_flip = lm_ex["teacher_flip"]

    t2_loss = zero
    t2_flip = zero.detach()
    rank_t3 = zero
    if config.uses_t3_candidates and batch.cand_owner.numel() > 0:
        parts = sentence_scores(
            model,
            states,
            flat_lp,
            key_index,
            batch.gold_keys,
            batch.cand_owner,
            batch.cand_sequences,
            batch.cand_lengths,
            batch.cand_cost,
            config.gamma,
            batch.cand_step_bound,
        )
        item_scores = parts.score(config.beta)
        student_scores = torch.zeros_like(batch.teacher_scores).index_put(
            (batch.cand_slot_b, batch.cand_slot_c), item_scores
        )
        if config.t2_weight > 0.0:
            t2_loss, t2_flip = t2_residual_kl_loss(
                student_scores,
                batch.teacher_scores,
                batch.cand_valid,
                scale=config.t2_scale,
            )
        if config.rank_t3_weight > 0.0:
            valid = batch.cand_valid
            has_gold_pair = valid[:, 0] & (valid.sum(dim=-1) >= 2)
            if bool(has_gold_pair.any()):
                t_slots = (
                    student_scores[has_gold_pair]
                    .float()
                    .masked_fill(~valid[has_gold_pair], float("-inf"))
                )
                rank_t3 = (t_slots.logsumexp(dim=1) - t_slots[:, 0]).mean()

    total = (
        config.ctc_weight * gold_ctc
        + config.ce_weight * gold_ce
        + config.t1_weight * t1_loss
        + config.t2_weight * t2_loss
        + config.rank_t3_weight * rank_t3
        + config.rank_online_weight * rank_online
        + config.lm_weight * lm_loss
    )
    return DistillStepLosses(
        total=total,
        ctc=gold_ctc,
        ce=gold_ce,
        t1=t1_loss,
        t2=t2_loss,
        t2_flip=t2_flip,
        rank_t3=rank_t3,
        rank_online=rank_online,
        rank_acc=rank_acc,
        lm=lm_loss,
        lm_flip=lm_flip,
    )


def bucketed_label_batches(
    stream: Iterable[TeacherLabelRow],
    *,
    batch_size: int,
    pool_batches: int = 16,
    seed: int = 0,
) -> Iterator[list[TeacherLabelRow]]:
    """Group :class:`TeacherLabelRow` items into length-bucketed microbatches by key length."""
    pool_size = max(1, pool_batches) * max(1, batch_size)
    rng = random.Random(seed + 5_000_003)
    iterator = iter(stream)
    while True:
        pool: list[TeacherLabelRow] = []
        while len(pool) < pool_size:
            try:
                pool.append(next(iterator))
            except StopIteration:
                break
        if not pool:
            break
        pool.sort(key=lambda row: len(row.keys))
        batches = [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
        rng.shuffle(batches)
        yield from batches


def save_distill_checkpoint(
    out_dir: Path,
    model: KeyE2EModel,
    optimiser: torch.optim.Optimizer,
    step: int,
    config: DistillTrainConfig,
    *,
    filename: str = "checkpoint.pt",
) -> Path:
    """Atomically write a resumable S6 student checkpoint readable by :func:`load_key_model`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            model.config_key: asdict(model.e2e_config),
            "optimiser": optimiser.state_dict(),
            "distill": asdict(config),
        },
        tmp,
    )
    tmp.replace(path)
    return path


class DistillStreamSource(Protocol):
    """Picklable factory opening a worker's stream, collator, and legality off-GIL (#353)."""

    def open(
        self,
    ) -> tuple[Iterable[TeacherLabelRow], KeyCollator, Legality, TeacherVocab | None]:
        """Build the worker-local stream, key collator, legality, and optional LM vocab."""
        ...


@dataclass(frozen=True)
class ParquetDistillSource:
    """Picklable S5 Parquet source opened inside a spawned prefetch worker (#353)."""

    shards_dir: Path
    char_table: Path
    typo_table: Path
    syllables: Path
    base_model: str
    context_model: QwenContextConfig | None = None
    seed: int = 0
    charlm_path: Path | None = None

    def open(
        self,
    ) -> tuple[Iterable[TeacherLabelRow], KeyCollator, Legality, TeacherVocab | None]:
        """Open shards, vocabularies, legality, and optional Char-LM vocab in the worker."""
        from mlime.train.run import Vocabularies

        vocabularies = Vocabularies.load(self.char_table, self.base_model, self.context_model)
        legality = Legality.load(
            vocabularies.lexicon.characters,
            char_table=self.char_table,
            typo_table=self.typo_table,
            syllables=self.syllables,
        )
        collator = vocabularies.key_collator(context_dropout=0.0, seed=self.seed)
        charlm_vocab = None if self.charlm_path is None else TeacherVocab.read(self.charlm_path)
        return TeacherLabelShards(self.shards_dir, seed=self.seed), collator, legality, charlm_vocab


def _cyclic_label_rows(
    stream: Iterable[TeacherLabelRow],
    skip_target: int,
) -> Iterator[TeacherLabelRow]:
    skipped = 0
    while True:
        yielded_any = False
        for row in stream:
            yielded_any = True
            if skipped < skip_target:
                skipped += 1
                continue
            yield row
        if not yielded_any:
            break


def _distill_process_worker(
    source: DistillStreamSource,
    skip_target: int,
    batch_size: int,
    pool_batches: int,
    seed: int,
    score_eos: bool,
    options: LegalityOptions,
    include_online: bool,
    negatives: int,
    out_q: Any,
    stop_event: Any,
) -> None:
    torch.set_num_threads(1)

    def _put(item: Any) -> None:
        while not stop_event.is_set():
            try:
                out_q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    try:
        stream, collator, legality, charlm_vocab = source.open()
        micro_iter = bucketed_label_batches(
            _cyclic_label_rows(stream, skip_target),
            batch_size=batch_size,
            pool_batches=pool_batches,
            seed=seed,
        )
        while not stop_event.is_set():
            try:
                raw_rows = next(micro_iter)
            except StopIteration:
                _put(("eof", None))
                return
            built = collate_distill_batch(
                raw_rows,
                collator,
                legality,
                score_eos=score_eos,
                options=options,
                include_online=include_online,
                negatives=negatives,
                teacher_vocab=charlm_vocab,
            )
            if built is not None:
                _put(("batch", built))
    except BaseException:
        _put(("error", traceback.format_exc()))


class _ProcessDistillPrefetcher:
    """Spawned process prefetcher isolating S6 collation from the trainer's GIL (#353)."""

    def __init__(
        self,
        source: DistillStreamSource,
        *,
        skip_target: int,
        batch_size: int,
        pool_batches: int,
        seed: int,
        score_eos: bool,
        options: LegalityOptions,
        include_online: bool,
        negatives: int,
        maxsize: int,
    ) -> None:
        ctx = mp.get_context("spawn")
        self._q: Any = ctx.Queue(maxsize=maxsize)
        self._stop: Any = ctx.Event()
        self._proc: Any = ctx.Process(
            target=_distill_process_worker,
            args=(
                source,
                skip_target,
                batch_size,
                pool_batches,
                seed,
                score_eos,
                options,
                include_online,
                negatives,
                self._q,
                self._stop,
            ),
            daemon=True,
        )
        self._proc.start()

    def get(self) -> DistillBatch | None:
        """Return the next collated :class:`DistillBatch`, or ``None`` on exhaustion."""
        while True:
            try:
                tag, payload = self._q.get(timeout=0.5)
            except queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(
                        f"S6 prefetch worker exited unexpectedly (code={self._proc.exitcode})"
                    ) from None
                continue
            if tag == "batch":
                assert isinstance(payload, DistillBatch)
                return payload
            if tag == "eof":
                return None
            if tag == "error":
                raise RuntimeError(f"S6 prefetch worker failed:\n{payload}")
            raise RuntimeError(f"unexpected prefetch message tag: {tag!r}")

    def close(self) -> None:
        """Stop the worker cleanly without blocking on a full queue."""
        self._stop.set()
        while True:
            try:
                self._q.get_nowait()
            except (queue.Empty, ValueError, OSError):
                break
        self._proc.join(timeout=10.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5.0)
        self._q.close()
        self._q.join_thread()


def train_distill_loop(
    model: KeyE2EModel,
    collator: KeyCollator,
    legality: Legality,
    stream: Iterable[TeacherLabelRow],
    out_dir: Path,
    config: DistillTrainConfig,
    *,
    resume_path: Path | None = None,
    charlm_path: Path | None = None,
    stream_source: DistillStreamSource | None = None,
    options: LegalityOptions = OFFLINE,
) -> dict[str, Any]:
    """Resumable unified S6 distillation loop for the key-level student (#191, #350, #353)."""
    if isinstance(stream, Iterator):
        raise TypeError(
            "train_distill_loop requires a re-iterable Iterable, not a single-use Iterator"
        )
    device = next(model.parameters()).device
    dcfg = config.distill

    charlm_vocab: TeacherVocab | None = None
    charlm_teacher: Teacher | None = None
    if dcfg.lm_weight > 0.0:
        if charlm_path is None:
            raise ValueError("lm_weight > 0 requires --charlm-teacher / charlm_path")
        charlm_vocab = TeacherVocab.read(charlm_path)
        charlm_teacher = Teacher.load(charlm_path, legality.characters, device)

    model.freeze_for_joint()
    groups = model.parameter_groups(config.base_lr, config.new_lr)
    optimiser = torch.optim.AdamW(
        groups,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    trainable = [p for group in groups for p in group["params"]]  # type: ignore[attr-defined]

    start_step = 0
    if resume_path is not None and resume_path.is_file():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_resumed(ckpt["model"], resume_path)
        if "optimiser" in ckpt:
            optimiser.load_state_dict(ckpt["optimiser"])
        start_step = int(ckpt["step"])
        log.info("distill resumed", path=str(resume_path), step=start_step)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    model.train()

    skip_target = start_step * config.batch_size * config.accumulate
    prefetch = config.prefetch_batches > 0
    proc_prefetcher: _ProcessDistillPrefetcher | None = None
    batch_q: queue.Queue[DistillBatch | BaseException | None] | None = None
    stop_prefetch = threading.Event()
    prod_thread: threading.Thread | None = None
    micro_iter: Iterator[list[TeacherLabelRow]] | None = None

    def _next_collated() -> DistillBatch | None:
        assert micro_iter is not None
        while True:
            try:
                raw_rows = next(micro_iter)
            except StopIteration:
                return None
            built = collate_distill_batch(
                raw_rows,
                collator,
                legality,
                score_eos=model.decoder.score_eos,
                options=options,
                include_online=dcfg.uses_online_beam,
                negatives=dcfg.negatives,
                teacher_vocab=charlm_vocab,
            )
            if built is not None:
                return built

    if prefetch and stream_source is not None:
        proc_prefetcher = _ProcessDistillPrefetcher(
            stream_source,
            skip_target=skip_target,
            batch_size=config.batch_size,
            pool_batches=config.pool_batches,
            seed=config.seed,
            score_eos=model.decoder.score_eos,
            options=options,
            include_online=dcfg.uses_online_beam,
            negatives=dcfg.negatives,
            maxsize=config.prefetch_batches,
        )
    else:
        micro_iter = bucketed_label_batches(
            _cyclic_label_rows(stream, skip_target),
            batch_size=config.batch_size,
            pool_batches=config.pool_batches,
            seed=config.seed,
        )
        if prefetch:
            batch_q = queue.Queue(maxsize=config.prefetch_batches)

            def _put(item: DistillBatch | BaseException | None) -> None:
                assert batch_q is not None
                while not stop_prefetch.is_set():
                    try:
                        batch_q.put(item, timeout=0.1)
                        return
                    except queue.Full:
                        continue

            def _producer() -> None:
                try:
                    while not stop_prefetch.is_set():
                        built = _next_collated()
                        _put(built)
                        if built is None:
                            return
                except BaseException as exc:
                    _put(exc)

            prod_thread = threading.Thread(target=_producer, daemon=True)
            prod_thread.start()

    def _pull_batch() -> DistillBatch | None:
        if proc_prefetcher is not None:
            return proc_prefetcher.get()
        if batch_q is None:
            return _next_collated()
        item = batch_q.get()
        if isinstance(item, BaseException):
            raise item
        return item

    search_graphs = (
        SearchGraphs(limit=config.search_graph_limit)
        if device.type == "cuda" and dcfg.uses_online_beam and config.search_graph_limit > 0
        else None
    )
    use_bf16 = device.type == "cuda"
    t0 = time.monotonic()
    step = start_step
    last_loss = 0.0

    try:
        while step < config.max_steps:
            step += 1
            scale = cosine_lr(step, config.max_steps, 1.0, config.warmup_fraction)
            optimiser.param_groups[0]["lr"] = config.base_lr * scale
            optimiser.param_groups[1]["lr"] = config.new_lr * scale
            optimiser.zero_grad(set_to_none=True)

            acc_total = torch.zeros((), device=device)
            acc_ctc = torch.zeros((), device=device)
            acc_ce = torch.zeros((), device=device)
            acc_t1 = torch.zeros((), device=device)
            acc_t2 = torch.zeros((), device=device)
            acc_t2_flip = torch.zeros((), device=device)
            acc_rank_t3 = torch.zeros((), device=device)
            acc_rank_online = torch.zeros((), device=device)
            acc_rank_acc = torch.zeros((), device=device)
            acc_lm = torch.zeros((), device=device)
            acc_lm_flip = torch.zeros((), device=device)
            ran_micros = 0

            for _ in range(config.accumulate):
                batch = _pull_batch()
                if batch is None:
                    break
                dev_batch = batch.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    losses = distill_batch_loss(
                        model,
                        dev_batch,
                        dcfg,
                        charlm_teacher=charlm_teacher,
                        search_graphs=search_graphs,
                    )
                torch.autograd.backward(losses.total / config.accumulate)
                inv = 1.0 / config.accumulate
                acc_total = acc_total + losses.total.detach() * inv
                acc_ctc = acc_ctc + losses.ctc.detach() * inv
                acc_ce = acc_ce + losses.ce.detach() * inv
                acc_t1 = acc_t1 + losses.t1.detach() * inv
                acc_t2 = acc_t2 + losses.t2.detach() * inv
                acc_t2_flip = acc_t2_flip + losses.t2_flip.detach() * inv
                acc_rank_t3 = acc_rank_t3 + losses.rank_t3.detach() * inv
                acc_rank_online = acc_rank_online + losses.rank_online.detach() * inv
                acc_rank_acc = acc_rank_acc + losses.rank_acc.detach() * inv
                acc_lm = acc_lm + losses.lm.detach() * inv
                acc_lm_flip = acc_lm_flip + losses.lm_flip.detach() * inv
                ran_micros += 1

            if ran_micros == 0:
                step -= 1
                break

            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimiser.step()
            last_loss = float(acc_total.item())

            if step == start_step + 1 or step % config.log_every == 0 or step == config.max_steps:
                record = distill_step_metrics(
                    DistillStepLosses(
                        total=acc_total,
                        ctc=acc_ctc,
                        ce=acc_ce,
                        t1=acc_t1,
                        t2=acc_t2,
                        t2_flip=acc_t2_flip,
                        rank_t3=acc_rank_t3,
                        rank_online=acc_rank_online,
                        rank_acc=acc_rank_acc,
                        lm=acc_lm,
                        lm_flip=acc_lm_flip,
                    ),
                    step=step,
                    lr=config.base_lr * scale,
                    elapsed=time.monotonic() - t0,
                )
                log_distill_step(metrics_path, record)

            if step % config.checkpoint_every == 0 and step < config.max_steps:
                save_distill_checkpoint(out_dir, model, optimiser, step, config)
    finally:
        stop_prefetch.set()
        if prod_thread is not None:
            prod_thread.join(timeout=5.0)
        if proc_prefetcher is not None:
            proc_prefetcher.close()

    final_path = save_distill_checkpoint(
        out_dir, model, optimiser, step, config, filename="checkpoint-final.pt"
    )
    return {
        "steps": step,
        "last_loss": last_loss,
        "seconds": round(time.monotonic() - t0, 2),
        "checkpoint": str(final_path),
    }


def distill_step_metrics(
    losses: DistillStepLosses,
    step: int,
    lr: float,
    elapsed: float,
) -> dict[str, Any]:
    """Format one unified S6 distillation metrics record."""
    return {
        "event": "distill_step",
        "step": step,
        "loss": round(float(losses.total.detach().item()), 5),
        "ctc_loss": round(float(losses.ctc.detach().item()), 5),
        "ce_loss": round(float(losses.ce.detach().item()), 5),
        "t1_loss": round(float(losses.t1.detach().item()), 5),
        "t2_loss": round(float(losses.t2.detach().item()), 5),
        "t2_flip": round(float(losses.t2_flip.detach().item()), 4),
        "rank_t3_loss": round(float(losses.rank_t3.detach().item()), 5),
        "rank_online_loss": round(float(losses.rank_online.detach().item()), 5),
        "rank_acc": round(float(losses.rank_acc.detach().item()), 4),
        "lm_loss": round(float(losses.lm.detach().item()), 5),
        "lm_flip": round(float(losses.lm_flip.detach().item()), 4),
        "lr": lr,
        "seconds": round(elapsed, 2),
    }


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_fraction: float = 0.04) -> float:
    """Linear warmup followed by cosine decay to zero."""
    warmup = max(1, int(max_steps * warmup_fraction))
    if step <= warmup:
        return base_lr * (step / warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON record to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def log_distill_step(metrics_path: Path, record: Mapping[str, Any]) -> None:
    """Write *record* to *metrics_path* and emit a structured log line."""
    append_jsonl(metrics_path, record)
    log.info(
        "distill step",
        **{k: v for k, v in record.items() if k != "event"},
    )
    _ = time.monotonic()
