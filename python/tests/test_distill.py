"""Unit tests for Phase 2 S6 offline T1/T2/T3 distillation (#191, #330, #340)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional
from test_e2e_model import READER_TINY, TINY
from test_keys import (
    CPU,
    KEYS,
    legality_fixture,  # noqa: F401
    reader_vocab_fixture,  # noqa: F401
    vocabularies_fixture,  # noqa: F401
)

from mlime.legality import OFFLINE, Legality
from mlime.train.charlm_vocab import CharVocab
from mlime.train.distill import (
    DistillConfig,
    DistillTrainConfig,
    TeacherLabelRow,
    cosine_lr,
    iter_teacher_label_shards,
    read_teacher_label_shard,
    score_candidate_sentences,
    t1_kl_loss,
    t2_listwise_kl_loss,
    train_distill_loop,
)
from mlime.train.e2e import load_key_model
from mlime.train.e2e_model import E2EConfig, emittable_ids
from mlime.train.key_eval import DecodeTimes, ModelTransition
from mlime.train.key_model import KeyE2EModel
from mlime.train.key_search import search
from mlime.train.lexicon import Lexicon
from mlime.train.run import Vocabularies
from mlime.train.teacher import write_labels_parquet


def _tiny_prelude_key_model(lexicon: Lexicon, reader_vocab: CharVocab) -> KeyE2EModel:
    keys_cfg = replace(KEYS, decoder_prelude=True, score_eos=True)
    cfg = E2EConfig(
        cross_attention_layers=1,
        decoder=READER_TINY,
        keys=keys_cfg,
        key_warm_up=False,
    )
    torch.manual_seed(0)
    return KeyE2EModel.compose(
        TINY, lexicon, cfg, len(reader_vocab), emittable_ids(lexicon, reader_vocab)
    )


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
    assert float(loss1.detach()) < float(loss0.detach())

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
    assert float(loss.detach()) > 0.0
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
    assert rows[0].context is None
    assert rows[0].text == "我爱北京"
    assert len(rows[0].t1_ids) == 5
    assert rows[0].candidates == ("我爱北京", "我爱背景")
    assert list(iter_teacher_label_shards(tmp_path)) == rows

    cfg = DistillConfig()
    assert cfg.t1_weight == 1.0
    assert cosine_lr(1, 100, 1e-4) > 0.0
    assert cosine_lr(100, 100, 1e-4) == pytest.approx(0.0, abs=1e-9)


def test_score_candidate_sentences_matches_unbounded_search(
    vocabularies: Vocabularies,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    legality: Legality,
) -> None:
    model = _tiny_prelude_key_model(lexicon, reader_vocab).eval()
    collator = vocabularies.key_collator(context_dropout=0.0)
    keys_list = ["woaibeijing", "beijing"]
    contexts: list[str | None] = ["我和北京", None]
    inputs = collator.inputs(keys_list, contexts)

    with torch.no_grad():
        states = model.key_states(inputs)
        ctc_lp = model.ctc_log_probs(states.states)
        lattices = [legality.lattice(k, OFFLINE) for k in keys_list]
        found = search(
            lattices,
            ctc_lp,
            ModelTransition(model, states, DecodeTimes()),
            beta=2.0,
            gamma=1.0,
            beam=None,
            viterbi=False,
            score_eos=True,
        )
        cands = [[list(h.ids) for h in hyps[:4]] for hyps in found]
        scores, valid = score_candidate_sentences(
            model, states, ctc_lp, keys_list, cands, legality, beta=2.0, gamma=1.0, options=OFFLINE
        )

    for row_i, hyps in enumerate(found):
        for cand_i, hyp in enumerate(hyps[:4]):
            assert bool(valid[row_i, cand_i])
            assert float(scores[row_i, cand_i]) == pytest.approx(hyp.score, abs=1e-4)


def test_train_distill_loop_lowers_loss_resumes_and_reloads(
    tmp_path: Path,
    vocabularies: Vocabularies,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    legality: Legality,
) -> None:
    model = _tiny_prelude_key_model(lexicon, reader_vocab)
    collator = vocabularies.key_collator(context_dropout=0.0)
    eos_id = lexicon.size

    # Build synthetic TeacherLabelRows over legal sentences from the test lexicon.
    pairs = [
        ("r0", "我和北京", "woaibeijing", "我爱北京"),
        ("r1", None, "beijing", "北京"),
    ]
    label_rows: list[TeacherLabelRow] = []
    for rid, ctx, keys, text in pairs:
        ids = legality.ids(text)
        assert ids is not None
        with torch.no_grad():
            alt_hyps = [
                "".join(legality.characters[c] for c in h.ids)
                for h in search(
                    [legality.lattice(keys, OFFLINE)],
                    torch.zeros((1, len(keys), lexicon.size + 1)),
                    ModelTransition(
                        model.eval(),
                        model.key_states(collator.inputs([keys], [ctx])),
                        DecodeTimes(),
                    ),
                    beta=0.0,
                    gamma=1.0,
                    beam=4,
                    viterbi=False,
                    score_eos=True,
                )[0]
            ]
        cands = [text, *[h for h in alt_hyps if h != text][:2]]
        t1_ids = (
            *((eid, (eid + 1) % eos_id, eos_id) for eid in ids),
            (eos_id, ids[0], (ids[0] + 1) % eos_id),
        )
        t1_lps = tuple((-0.05, -3.5, -4.5) for _ in range(len(ids) + 1))
        scores = tuple(-0.5 - 2.0 * i for i in range(len(cands)))
        label_rows.append(
            TeacherLabelRow(
                id=rid,
                context=ctx,
                keys=keys,
                text=text,
                t1_ids=t1_ids,
                t1_log_probs=t1_lps,
                candidates=tuple(cands),
                candidate_scores=scores,
            )
        )

    run_dir = tmp_path / "s6-run"
    cfg = DistillTrainConfig(
        distill=DistillConfig(t1_weight=1.0, t2_weight=0.5, gold_weight=1.0),
        base_lr=5e-3,
        new_lr=5e-3,
        batch_size=2,
        accumulate=1,
        max_steps=6,
        log_every=1,
        checkpoint_every=3,
        pool_batches=2,
    )
    # Run first 3 steps, then resume to step 6.
    first_cfg = replace(cfg, max_steps=3)
    s1 = train_distill_loop(model, collator, legality, label_rows * 8, run_dir, first_cfg)
    assert s1["steps"] == 3
    mid_ckpt = Path(s1["checkpoint"])
    assert mid_ckpt.is_file()

    s2 = train_distill_loop(
        model, collator, legality, label_rows * 8, run_dir, cfg, resume_path=mid_ckpt
    )
    assert s2["steps"] == 6
    final_ckpt = Path(s2["checkpoint"])
    assert final_ckpt.is_file()

    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    losses = [json.loads(line)["loss"] for line in lines]
    assert losses[-1] < losses[0]

    # Checkpoint reloads cleanly through load_key_model (used by eval keys & Core ML export).
    reloaded, step = load_key_model(final_ckpt, lexicon, CPU, bert_config=TINY)
    assert step == 6
    model.eval()
    probe = collator.inputs(["woaibeijing", "beijing"], ["我和北京", None])
    with torch.no_grad():
        a = model.ctc_log_probs(model.key_states(probe).states)
        b = reloaded.ctc_log_probs(reloaded.key_states(probe).states)
    assert torch.allclose(a, b, atol=1e-5)
