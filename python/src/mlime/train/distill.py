"""Offline T1/T2/T3 distillation from the Qwen teacher into the key-level student (#191, #330).

Consumes the per-shard Zstd Parquet files written by ``mlime train label-teacher``
(S5, :func:`mlime.train.teacher.label_corpus_shards`):

- **T1 (per-position top-k KL)**: aligns the student decoder's teacher-forced
  transition distribution ``log P_S(· | y_<j, keys, prelude)`` on gold positions
  (including ``<eos>``) to the teacher's stored top-k distribution
  ``(t1_ids, t1_log_probs)``.
- **T2 + T3 (listwise KL over teacher hard negatives)**: scores the gold
  sentence plus the teacher's lattice-constrained beam-8 negatives (T3) under
  the student's exact sentence score ``S_student(y)`` and minimises
  ``KL(softmax(s_T / tau_t2) || softmax(S_student))`` (T2).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
import torch.nn.functional as functional

from mlime.legality import OFFLINE, Legality, LegalityOptions
from mlime.logging import log
from mlime.train.ctc import alignment_scores, constrained_ctc
from mlime.train.key_model import KeyE2EModel, KeyStates


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
class TeacherLabelRow:
    """One record read from an S5 offline teacher label Parquet shard."""

    id: str
    keys: str
    text: str
    t1_ids: tuple[tuple[int, ...], ...]
    t1_log_probs: tuple[tuple[float, ...], ...]
    candidates: tuple[str, ...]
    candidate_scores: tuple[float, ...]


def read_teacher_label_shard(path: Path) -> list[TeacherLabelRow]:
    """Read one S5 Parquet shard into :class:`TeacherLabelRow` objects."""
    table = pq.read_table(path)
    rows: list[TeacherLabelRow] = []
    for raw in table.to_pylist():
        rows.append(
            TeacherLabelRow(
                id=str(raw["id"]),
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
    paths = sorted(shards_dir.glob("*.parquet"))
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

    Returns ``(scores [B, C], valid_mask [B, C])`` where ``valid_mask[b, c]`` is
    True when candidate ``c`` is present and legal under *legality* for
    ``keys_list[b]``.
    """
    device = states.states.device
    batch_size = len(keys_list)
    max_c = max((len(cands) for cands in candidates_per_row), default=0)
    scores = torch.zeros((batch_size, max_c), dtype=torch.float32, device=device)
    valid = torch.zeros((batch_size, max_c), dtype=torch.bool, device=device)
    if max_c == 0:
        return scores, valid

    flat_owners: list[int] = []
    flat_cands: list[Sequence[int]] = []
    flat_coords: list[tuple[int, int]] = []
    for b, (keys, cands) in enumerate(zip(keys_list, candidates_per_row, strict=True)):
        for c, emit_ids in enumerate(cands):
            if not emit_ids:
                continue
            cand_text = "".join(legality.characters[e] for e in emit_ids)
            gold_arcs = legality.gold(keys, cand_text, options)
            if gold_arcs is None:
                continue
            flat_owners.append(b)
            flat_cands.append(emit_ids)
            flat_coords.append((b, c))
            # Constrained CTC term on this candidate's legal alignment arcs.
            arc_key_idx = torch.as_tensor(gold_arcs.end - 1, dtype=torch.long, device=device)
            arc_pos = torch.as_tensor(gold_arcs.position, dtype=torch.long, device=device)
            arc_start = torch.as_tensor(gold_arcs.start, dtype=torch.long, device=device)
            arc_cost = torch.as_tensor(gold_arcs.cost, dtype=torch.float32, device=device)
            row_lp = ctc_log_probs[b]  # [Wk, E + 1]
            cand_t = torch.tensor(emit_ids, dtype=torch.long, device=device)
            # Single-item constrained CTC via alignment_scores + constrained_ctc.
            key_len = len(keys)
            m = len(emit_ids)
            blank, emit = alignment_scores(
                row_lp[:key_len],
                torch.arange(key_len, dtype=torch.long, device=device).unsqueeze(0),
                cand_t.unsqueeze(0),
            )
            cost_grid = torch.full((1, m, key_len, key_len), float("inf"), device=device)
            for p_i, s_i, e_i, w_i in zip(
                arc_pos.tolist(),
                arc_start.tolist(),
                arc_key_idx.tolist(),
                arc_cost.tolist(),
                strict=True,
            ):
                cost_grid[0, p_i, s_i, e_i] = gamma * w_i
            nll = constrained_ctc(
                blank,
                emit,
                cost_grid,
                torch.tensor([key_len], dtype=torch.long, device=device),
                torch.tensor([m], dtype=torch.long, device=device),
            )[0]
            scores[b, c] = -nll
            valid[b, c] = True

    if not flat_owners:
        return scores, valid

    # Batched transition log-prob pass over all valid candidates.
    owner_idx = torch.tensor(flat_owners, dtype=torch.long, device=device)
    cand_states = KeyStates(
        states=states.states.index_select(0, owner_idx),
        mask=states.mask.index_select(0, owner_idx),
        prelude=None if states.prelude is None else states.prelude.select(owner_idx),
    )
    max_len = max(len(c) for c in flat_cands)
    n_items = len(flat_cands)
    cand_pad = torch.zeros((n_items, max_len), dtype=torch.long, device=device)
    lengths = torch.tensor([len(c) for c in flat_cands], dtype=torch.long, device=device)
    at_list: list[int] = []
    for i, cand_seq in enumerate(flat_cands):
        cand_pad[i, : len(cand_seq)] = torch.tensor(cand_seq, dtype=torch.long, device=device)
        for j in range(len(cand_seq)):
            at_list.append(i * max_len + j)
    at = torch.tensor(at_list, dtype=torch.long, device=device)
    trans_logits = model.transition_logits(cand_states, cand_pad, at, lengths)
    trans_lp = trans_logits.float().log_softmax(dim=-1)

    char_targets = cand_pad.reshape(-1).index_select(0, at)
    char_lp = trans_lp[: at.numel()].gather(1, char_targets.unsqueeze(1)).squeeze(1)
    offset = 0
    eos_id = int(model.decoder.emit_ids.numel())
    for i, (b, c) in enumerate(flat_coords):
        ln = len(flat_cands[i])
        seq_lp = char_lp[offset : offset + ln].sum()
        offset += ln
        if model.decoder.score_eos:
            seq_lp = seq_lp + trans_lp[at.numel() + i, eos_id]
        scores[b, c] = scores[b, c] + beta * seq_lp

    return scores, valid


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
