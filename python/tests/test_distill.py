"""Unit tests for Phase 2 S6 offline T1/T2/T3 distillation (#191, #330)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from mlime.train.distill import (
    DistillConfig,
    cosine_lr,
    iter_teacher_label_shards,
    read_teacher_label_shard,
    t1_kl_loss,
    t2_listwise_kl_loss,
)
from mlime.train.teacher import write_labels_parquet


def test_t1_kl_loss_zero_when_matching_and_decreases_under_grad() -> None:
    torch.manual_seed(0)
    vocab = 16
    n_pos = 5
    k = 4
    t1_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [1, 3, 5, 7],
            [2, 4, 6, 8],
            [15, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    t1_lps = functional.log_softmax(torch.randn(n_pos, k), dim=-1)

    # Exact match over top-k support with zero probability outside yields 0 KL.
    student_logits = torch.full((n_pos, vocab), -1e9, dtype=torch.float32)
    student_logits.scatter_(1, t1_ids, t1_lps)
    student_lp = functional.log_softmax(student_logits, dim=-1)
    assert float(t1_kl_loss(student_lp, t1_ids, t1_lps)) == pytest.approx(0.0, abs=1e-5)

    # Gradient step lowers KL.
    trainable = torch.randn(n_pos, vocab, requires_grad=True)
    loss0 = t1_kl_loss(functional.log_softmax(trainable, dim=-1), t1_ids, t1_lps)
    loss0.backward()
    assert trainable.grad is not None
    with torch.no_grad():
        updated = trainable - 0.5 * trainable.grad
    loss1 = t1_kl_loss(functional.log_softmax(updated, dim=-1), t1_ids, t1_lps)
    assert float(loss1) < float(loss0)

    # Masking ignores masked rows.
    mask = torch.tensor([True, True, False, False, False])
    masked_loss = t1_kl_loss(
        functional.log_softmax(trainable.detach(), dim=-1), t1_ids, t1_lps, mask=mask
    )
    subset_loss = t1_kl_loss(
        functional.log_softmax(trainable.detach()[:2], dim=-1), t1_ids[:2], t1_lps[:2]
    )
    assert float(masked_loss) == pytest.approx(float(subset_loss), abs=1e-6)


def test_t2_listwise_kl_loss_handles_variable_candidates_and_singleton_rows() -> None:
    teacher_scores = torch.tensor(
        [
            [-1.0, -2.5, -4.0, 0.0],
            [-0.5, -1.2, 0.0, 0.0],
            [-0.3, 0.0, 0.0, 0.0],  # only 1 valid candidate -> ignored
        ],
        dtype=torch.float32,
    )
    valid = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
        ],
        dtype=torch.bool,
    )
    # Identical scores give 0 KL.
    assert float(t2_listwise_kl_loss(teacher_scores, teacher_scores, valid)) == pytest.approx(
        0.0, abs=1e-6
    )

    # Perturbed student scores produce positive KL and valid gradients; padded entries have 0 grad.
    student = (
        teacher_scores
        + torch.tensor([[0.0, 1.0, -0.5, 99.0], [1.0, -1.0, 50.0, 50.0], [0.0, 0.0, 0.0, 0.0]])
    ).requires_grad_(True)
    loss = t2_listwise_kl_loss(student, teacher_scores, valid)
    assert float(loss) > 0.0
    loss.backward()
    assert student.grad is not None
    assert float(student.grad[0, 3]) == 0.0
    assert float(student.grad[1, 2]) == 0.0
    assert float(student.grad[2].abs().sum()) == 0.0


def test_read_and_iter_teacher_label_shards(tmp_path: Path) -> None:
    write_labels_parquet(
        [
            {
                "id": "r0",
                "keys": "woaibeijing",
                "text": "我爱北京",
                "t1_ids": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                "t1_log_probs": [
                    [-0.1, -2.3],
                    [-0.2, -1.7],
                    [-0.3, -1.3],
                    [-0.1, -2.3],
                    [-0.05, -3.0],
                ],
                "candidates": ["我爱北京", "我爱背景"],
                "candidate_scores": [-0.8, -4.2],
            }
        ],
        tmp_path / "shard-00000.parquet",
    )
    rows = read_teacher_label_shard(tmp_path / "shard-00000.parquet")
    assert len(rows) == 1
    assert rows[0].text == "我爱北京"
    assert len(rows[0].t1_ids) == 5
    assert rows[0].candidates == ("我爱北京", "我爱背景")
    assert list(iter_teacher_label_shards(tmp_path)) == rows

    cfg = DistillConfig()
    assert cfg.t1_weight == 1.0
    assert cosine_lr(1, 100, 1e-4) > 0.0
    assert cosine_lr(100, 100, 1e-4) == pytest.approx(0.0, abs=1e-9)
