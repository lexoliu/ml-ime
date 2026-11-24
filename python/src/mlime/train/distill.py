"""Offline T1/T2/T3 distillation from the Qwen teacher into the key-level student (#191, #340).

Consumes the per-shard Zstd Parquet files written by ``mlime train label-teacher``
(S5, :func:`mlime.train.teacher.label_corpus_shards`):

- **T1 (per-position top-k KL)**: aligns the student decoder's teacher-forced
  transition distribution ``log P_S(· | y_<j, keys, prelude)`` on gold positions
  (including ``<eos>`` when ``score_eos`` is on) to the teacher's stored top-k
  distribution ``(t1_ids, t1_log_probs)``.
- **T2 + T3 (listwise KL over teacher hard negatives)**: scores the gold
  sentence plus the teacher's lattice-constrained beam-8 negatives (T3) under
  the student's exact sentence score ``S_student(y)`` in a single vectorized
  pass (:func:`~mlime.train.key_ranking.sentence_scores`) and minimises
  ``KL(softmax(s_T / tau_t2) || softmax(S_student))`` (T2).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
import torch.nn.functional as functional

from mlime.legality import MAX_ARC_KEYS, OFFLINE, Legality, LegalityOptions
from mlime.logging import log
from mlime.train.ctc import alignment_scores, constrained_ctc
from mlime.train.key_model import KeyE2EModel, KeyStates
from mlime.train.key_ranking import priced, sentence_scores
from mlime.train.keys import KeyCollator, KeyInputs


@dataclass(frozen=True)
class DistillConfig:
    """Loss weights and temperatures for Phase 2 S6 offline distillation (#191, #330)."""

    #: Weight on per-position top-k KL (T1) beside the student's gold CTC + CE.
    t1_weight: float = 1.0
    #: Softmax temperature applied to the teacher's T1 log-probs over top-k support.
    t1_temperature: float = 1.0
    #: Weight on listwise KL (T2) over gold + teacher beam-8 hard negatives (T3).
    t2_weight: float = 0.5
    #: Softmax temperature applied to the teacher's T2 candidate sentence scores.
    t2_temperature: float = 1.0
    #: Transition weight ``beta`` in the student's sentence score ``S(y)``.
    beta: float = 2.0
    #: Correction-cost weight ``gamma`` in the student's sentence score ``S(y)``.
    gamma: float = 1.0
    #: Weight on the gold CTC + transition CE anchor term.
    gold_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.t1_temperature <= 0.0 or self.t2_temperature <= 0.0:
            raise ValueError("distillation temperatures must be positive")
        if self.t1_weight < 0.0 or self.t2_weight < 0.0 or self.gold_weight < 0.0:
            raise ValueError("distillation weights must be non-negative")


@dataclass(frozen=True)
class DistillTrainConfig:
    """Training loop hyperparameters for Phase 2 S6 student distillation (#191, #340)."""

    distill: DistillConfig = DistillConfig()
    base_lr: float = 2e-5
    new_lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.04
    batch_size: int = 64
    accumulate: int = 2
    max_steps: int = 5000
    gradient_clip: float = 1.0
    seed: int = 0
    log_every: int = 20
    checkpoint_every: int = 500
    pool_batches: int = 16


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


def t1_kl_loss(
    student_log_probs: torch.Tensor,
    t1_ids: torch.Tensor,
    t1_log_probs: torch.Tensor,
    *,
    temperature: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top-k per-position KL ``KL(P_T1 || P_S)`` over gold positions + ``<eos>``.

    Parameters
    ----------
    student_log_probs:
        ``[N, V]`` log-probabilities from the student decoder's transition head.
    t1_ids:
        ``[N, K]`` integer class indices of the teacher's top-K emissions.
    t1_log_probs:
        ``[N, K]`` teacher log-probabilities at *t1_ids*.
    temperature:
        Softmax temperature applied to *t1_log_probs* over the top-K support.
    mask:
        Optional ``[N]`` boolean mask selecting active positions.
    """
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


def t2_listwise_kl_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Listwise KL ``KL(softmax(s_T / tau) || softmax(s_S))`` over variable-length candidate lists.

    Parameters
    ----------
    student_scores:
        ``[B, C]`` student sentence scores ``S_student(cand)``.
    teacher_scores:
        ``[B, C]`` teacher log-probabilities ``log P_T(cand + <eos> | context, keys)``.
    valid_mask:
        ``[B, C]`` boolean mask marking real candidates (True) vs padding (False).
        Rows with fewer than 2 valid candidates contribute zero loss.
    temperature:
        Softmax temperature on *teacher_scores*.
    """
    if student_scores.shape != teacher_scores.shape or student_scores.shape != valid_mask.shape:
        raise ValueError("student_scores, teacher_scores, and valid_mask must share shape [B, C]")
    if student_scores.ndim != 2:
        raise ValueError(f"expected [B, C] scores, got {tuple(student_scores.shape)}")

    neg_inf = torch.tensor(float("-inf"), device=student_scores.device, dtype=torch.float32)
    mask = valid_mask.bool()
    has_pair = mask.sum(dim=-1) >= 2
    if not bool(has_pair.any()):
        return student_scores.sum() * 0.0

    t_masked = torch.where(mask, teacher_scores.float() / temperature, neg_inf)
    s_masked = torch.where(mask, student_scores.float(), neg_inf)
    t_log_p = functional.log_softmax(t_masked[has_pair], dim=-1)
    s_log_p = functional.log_softmax(s_masked[has_pair], dim=-1)
    active_m = mask[has_pair]
    t_p = torch.where(active_m, t_log_p.exp(), torch.zeros_like(t_log_p))
    diff = torch.where(active_m, t_log_p - s_log_p, torch.zeros_like(t_log_p))
    return (t_p * diff).sum(dim=-1).mean()


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
    """Compute differentiable student sentence scores ``S(y)`` for each row's candidates.

    Batches constrained CTC and decoder transition scoring across all valid legal
    candidates in one device pass via :func:`~mlime.train.key_ranking.sentence_scores`.
    Returns ``(scores [B, C], valid_mask [B, C])``.
    """
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
class DistillBatch:
    """Pre-collated offline distillation batch ready for a single GPU step (#340)."""

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
        )


def collate_distill_batch(
    rows: Sequence[TeacherLabelRow],
    collator: KeyCollator,
    legality: Legality,
    *,
    score_eos: bool = True,
    options: LegalityOptions = OFFLINE,
) -> DistillBatch | None:
    """Collate S5 :class:`TeacherLabelRow` entries into a vectorized :class:`DistillBatch`.

    Rows whose gold sentence is illegal under *legality* or whose stored T1
    positions do not cover the sentence are skipped; returns ``None`` if no
    valid rows remain.
    """
    emittable = len(legality.characters)
    valid_rows: list[tuple[TeacherLabelRow, tuple[int, ...], torch.Tensor]] = []
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
        valid_rows.append((row, gold_ids, torch.from_numpy(arcs.dense()).to(torch.float32)))

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
    for b, (row, gold_ids, dense) in enumerate(valid_rows):
        ln = len(gold_ids)
        gold[b, :ln] = torch.tensor(gold_ids, dtype=torch.long)
        gold_cost[b, :ln, : dense.shape[1]] = dense
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
        # Drop any teacher <eos> entries when the student decoder does not score <eos>.
        in_vocab = raw_ids < emittable
        t1_ids_t = raw_ids.clamp(max=emittable - 1)
        t1_lps_t = torch.where(in_vocab, raw_lps, torch.full_like(raw_lps, -1e9))
        t1_mask = in_vocab.any(dim=-1)

    # Collate T2/T3 candidate lists across the batch.
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
    )


def distill_batch_loss(
    model: KeyE2EModel,
    batch: DistillBatch,
    config: DistillConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one vectorized S6 distillation forward pass and return ``(loss, gold, t1, t2)``."""
    states = model.key_states(batch.inputs)
    ctc_lp = model.ctc_log_probs(states.states)
    flat_lp, key_index = _flat_ctc_layout(ctc_lp)

    blank, emit = alignment_scores(flat_lp, key_index, batch.gold)
    gold_nll = constrained_ctc(
        blank, emit, priced(batch.gold_cost, config.gamma), batch.gold_keys, batch.gold_lengths
    )
    gold_ctc = gold_nll.sum() / max(1, batch.scored_chars)

    trans_logits = model.transition_logits(states, batch.gold, batch.gold_at, batch.gold_lengths)
    trans_lp = trans_logits.float().log_softmax(dim=-1)
    gold_ce = -trans_lp.gather(1, batch.gold_targets.unsqueeze(1)).mean()
    gold_loss = gold_ctc + model.keys_config.transition_weight * gold_ce

    if config.t1_weight > 0.0:
        t1_loss = t1_kl_loss(
            trans_lp,
            batch.t1_ids,
            batch.t1_log_probs,
            temperature=config.t1_temperature,
            mask=batch.t1_mask,
        )
    else:
        t1_loss = gold_loss.new_zeros(())

    if config.t2_weight > 0.0 and batch.cand_owner.numel() > 0:
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
        student_scores = torch.zeros_like(batch.teacher_scores)
        student_scores = student_scores.index_put(
            (batch.cand_slot_b, batch.cand_slot_c), item_scores
        )
        t2_loss = t2_listwise_kl_loss(
            student_scores,
            batch.teacher_scores,
            batch.cand_valid,
            temperature=config.t2_temperature,
        )
    else:
        t2_loss = gold_loss.new_zeros(())

    total = config.gold_weight * gold_loss + config.t1_weight * t1_loss + config.t2_weight * t2_loss
    return total, gold_loss, t1_loss, t2_loss


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


def train_distill_loop(
    model: KeyE2EModel,
    collator: KeyCollator,
    legality: Legality,
    stream: Iterable[TeacherLabelRow],
    out_dir: Path,
    config: DistillTrainConfig,
    *,
    resume_path: Path | None = None,
    options: LegalityOptions = OFFLINE,
) -> dict[str, Any]:
    """Resumable offline T1/T2/T3 distillation loop for the key-level student (S6, #191)."""
    device = next(model.parameters()).device
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

    def _cyclic_stream() -> Iterator[TeacherLabelRow]:
        skipped = 0
        skip_target = start_step * config.batch_size * config.accumulate
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

    micro_iter = bucketed_label_batches(
        _cyclic_stream(),
        batch_size=config.batch_size,
        pool_batches=config.pool_batches,
        seed=config.seed,
    )
    use_bf16 = device.type == "cuda"
    t0 = time.monotonic()
    step = start_step
    last_loss = 0.0

    while step < config.max_steps:
        step += 1
        scale = cosine_lr(step, config.max_steps, 1.0, config.warmup_fraction)
        optimiser.param_groups[0]["lr"] = config.base_lr * scale
        optimiser.param_groups[1]["lr"] = config.new_lr * scale
        optimiser.zero_grad(set_to_none=True)

        acc_total = torch.zeros((), device=device)
        acc_gold = torch.zeros((), device=device)
        acc_t1 = torch.zeros((), device=device)
        acc_t2 = torch.zeros((), device=device)
        ran_micros = 0

        for _ in range(config.accumulate):
            batch: DistillBatch | None = None
            while batch is None:
                try:
                    raw_rows = next(micro_iter)
                except StopIteration:
                    break
                batch = collate_distill_batch(
                    raw_rows,
                    collator,
                    legality,
                    score_eos=model.decoder.score_eos,
                    options=options,
                )
            if batch is None:
                break
            dev_batch = batch.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss, g_loss, t1_loss, t2_loss = distill_batch_loss(
                    model, dev_batch, config.distill
                )
            torch.autograd.backward(loss / config.accumulate)
            acc_total = acc_total + loss.detach() / config.accumulate
            acc_gold = acc_gold + g_loss.detach() / config.accumulate
            acc_t1 = acc_t1 + t1_loss.detach() / config.accumulate
            acc_t2 = acc_t2 + t2_loss.detach() / config.accumulate
            ran_micros += 1

        if ran_micros == 0:
            step -= 1
            break

        torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
        optimiser.step()
        last_loss = float(acc_total.item())

        if step == start_step + 1 or step % config.log_every == 0 or step == config.max_steps:
            record = distill_step_metrics(
                acc_total,
                acc_gold,
                acc_t1,
                acc_t2,
                step=step,
                lr=config.base_lr * scale,
                elapsed=time.monotonic() - t0,
            )
            log_distill_step(metrics_path, record)

        if step % config.checkpoint_every == 0 and step < config.max_steps:
            save_distill_checkpoint(out_dir, model, optimiser, step, config)

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
    loss: torch.Tensor,
    gold_loss: torch.Tensor,
    t1_loss: torch.Tensor,
    t2_loss: torch.Tensor,
    step: int,
    lr: float,
    elapsed: float,
) -> dict[str, Any]:
    """Format one S6 distillation metrics record."""
    return {
        "event": "distill_step",
        "step": step,
        "loss": round(float(loss.detach().item()), 5),
        "gold_loss": round(float(gold_loss.detach().item()), 5),
        "t1_loss": round(float(t1_loss.detach().item()), 5),
        "t2_loss": round(float(t2_loss.detach().item()), 5),
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
