"""Unit tests for the Phase 2 Qwen3.5 LoRA teacher (#191, #326)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from mlime.legality import OFFLINE, Legality
from mlime.train.qwen_tower import LoraConfig
from mlime.train.teacher import (
    TEACHER_LORA_TARGETS,
    EmittableVocab,
    QwenTeacher,
    TeacherTrainConfig,
    build_training_batch,
    constrained_beam_decode,
    score_candidates_and_t1,
    train_teacher_loop,
    write_labels_parquet,
)


class _FakeTokenizer:
    """Minimal byte/char tokenizer: common chars are single tokens, rare chars split to 2."""

    def __init__(self) -> None:
        self.eos_token_id = 0
        self.pad_token_id = 0
        self._single = {
            "我": 10,
            "爱": 11,
            "北": 12,
            "京": 13,
            "中": 14,
            "国": 15,
            "天": 16,
            "安": 17,
            "门": 18,
            "人": 19,
        }
        self._vocab_size = 256

    def __len__(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        out: list[int] = []
        for ch in text:
            if ch in self._single:
                out.append(self._single[ch])
            elif ch.isascii():
                out.append(32 + (ord(ch) % 100))
            else:
                # Multi-byte residue character: split into 2 deterministic byte pieces.
                code = ord(ch)
                out.append(140 + (code % 50))
                out.append(190 + ((code // 50) % 50))
        return out


def _tiny_qwen35() -> Qwen3_5ForCausalLM:
    cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        bos_token_id=0,
        eos_token_id=0,
    )
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(cfg)


def _tiny_legality(characters: tuple[str, ...]) -> Legality:
    repo = Path(__file__).resolve().parents[2]
    data = repo / "crates/ime-pinyin/data"
    return Legality.load(
        characters,
        char_table=data / "char_pinyin.tsv",
        typo_table=data / "typo.json",
        syllables=data / "syllables.txt",
    )


def test_emittable_vocab_and_lora_teacher_training_and_resume(tmp_path: Path) -> None:
    tok = _FakeTokenizer()
    chars = ("我", "爱", "北", "京", "中", "国", "弑", "玑")
    vocab = EmittableVocab.build(chars, tok)
    assert vocab.size == 8
    assert vocab.num_classes == 9
    assert vocab.num_residue == 2  # 弑, 玑
    assert vocab.emit_to_token[0] == 10
    assert vocab.emit_to_token[6] == 256
    assert vocab.emit_to_token[7] == 257
    assert vocab.emit_to_token[8] == tok.eos_token_id

    lora = LoraConfig(rank=4, alpha=8.0, targets=TEACHER_LORA_TARGETS)
    teacher = QwenTeacher(_tiny_qwen35(), vocab, lora, gradient_checkpointing=False)

    examples = [
        ("今天", "woaibeijing", "我爱北京"),
        (None, "zhongguo", "中国"),
        ("历史", "shiji", "弑玑"),
    ] * 20

    cfg = TeacherTrainConfig(
        lora_rank=4,
        lora_alpha=8.0,
        lr=5e-3,
        batch_size=3,
        accumulate=1,
        max_steps=8,
        log_every=1,
        checkpoint_every=4,
        gradient_checkpointing=False,
    )
    summary = train_teacher_loop(teacher, tok, examples, tmp_path / "run", cfg)
    assert summary["steps"] == 8
    assert Path(summary["checkpoint"]).is_file()

    # Check loss fell from step 1 to step 8.
    lines = (tmp_path / "run/metrics.jsonl").read_text(encoding="utf-8").splitlines()
    import json

    losses = [json.loads(line)["loss"] for line in lines]
    assert losses[-1] < losses[0]

    # Verify compact adapter reload reproduces exact logits.
    teacher.eval()
    c2e = vocab.char_to_emit()
    inp, mask, lbl = build_training_batch(
        tok, vocab, examples[:3], char_to_emit=c2e, pad_token_id=0, device=torch.device("cpu")
    )
    with torch.no_grad():
        loss_a, _ = teacher(inp, mask, lbl)

    reloaded = QwenTeacher(_tiny_qwen35(), vocab, lora, gradient_checkpointing=False)
    ckpt = torch.load(summary["checkpoint"], map_location="cpu", weights_only=False)
    reloaded.load_adapter_state_dict(ckpt["adapter"])
    reloaded.eval()
    with torch.no_grad():
        loss_b, _ = reloaded(inp, mask, lbl)
    assert float(loss_a) == pytest.approx(float(loss_b), abs=1e-5)


def test_constrained_beam_decode_and_offline_t1_t2_labels(tmp_path: Path) -> None:
    tok = _FakeTokenizer()
    chars = ("我", "爱", "北", "京", "中", "国", "天", "安", "门", "人", "弑", "玑")
    vocab = EmittableVocab.build(chars, tok)
    lora = LoraConfig(rank=4, alpha=8.0, targets=TEACHER_LORA_TARGETS)
    teacher = QwenTeacher(_tiny_qwen35(), vocab, lora, gradient_checkpointing=False).eval()
    legality = _tiny_legality(chars)

    hyps = constrained_beam_decode(
        teacher, tok, "今天", "woaibeijing", legality, OFFLINE, beam=4, gamma=1.0
    )
    assert len(hyps) >= 1
    for h in hyps:
        assert legality.cost("woaibeijing", h.text, OFFLINE) is not None

    c2e = vocab.char_to_emit()
    gold_ids = [c2e[ch] for ch in "我爱北京"]
    cand_ids = [gold_ids, *[list(h.emit_ids) for h in hyps if list(h.emit_ids) != gold_ids]]
    t1_ids, t1_lps, scores = score_candidates_and_t1(
        teacher, tok, "今天", "woaibeijing", gold_ids, cand_ids, top_k_t1=5
    )
    assert len(t1_ids) == len(gold_ids) + 1  # gold positions + <eos>
    assert all(len(row) == 5 for row in t1_ids)
    assert len(scores) == len(cand_ids)
    # Score of a clean hypothesis from score_candidates_and_t1 matches constrained_beam_decode.
    for h in hyps:
        if list(h.emit_ids) in cand_ids and legality.cost("woaibeijing", h.text, OFFLINE) == 0.0:
            idx = cand_ids.index(list(h.emit_ids))
            assert scores[idx] == pytest.approx(h.score, abs=1e-4)

    out_pq = tmp_path / "labels-00000.parquet"
    write_labels_parquet(
        [
            {
                "id": "rec-0",
                "keys": "woaibeijing",
                "text": "我爱北京",
                "t1_ids": t1_ids,
                "t1_log_probs": t1_lps,
                "candidates": ["我爱北京", *[h.text for h in hyps if h.text != "我爱北京"]],
                "candidate_scores": scores,
            }
        ],
        out_pq,
    )
    table = pq.read_table(out_pq)
    assert table.num_rows == 1
    assert table.column("text")[0].as_py() == "我爱北京"
