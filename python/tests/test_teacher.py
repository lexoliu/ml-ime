"""Unit tests for the Phase 2 Qwen3.5 LoRA teacher (#191, #326)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from mlime.legality import OFFLINE, Legality
from mlime.train.qwen_tower import LoraConfig
from mlime.train.teacher import (
    TEACHER_LORA_TARGETS,
    BeamHypothesis,
    EmittableVocab,
    QwenTeacher,
    TeacherTrainConfig,
    _decode_from_prefill,
    _expand_record_beams,
    _merge_label_parts,
    _prefill_prompts,
    _select_cache_rows,
    _SpanLattice,
    bucketed_microbatches,
    build_training_batch,
    constrained_beam_decode,
    constrained_beam_decode_batch,
    encode_prompt,
    label_records_batch,
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


def test_encode_prompt_keeps_one_keystroke_per_token() -> None:
    class _MergingTokenizer(_FakeTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if text == "fake":
                return [250]
            if "fake" in text:
                return [250]
            return super().encode(text, add_special_tokens=add_special_tokens)

    tok = _MergingTokenizer()
    ids = encode_prompt(tok, "今天", "fakeidea")
    # Every letter in "fakeidea" is encoded individually, never merged to token 250.
    assert 250 not in ids
    expected_keys = [tok.encode(ch, add_special_tokens=False)[0] for ch in "fakeidea"]
    sep = tok.encode("\t", add_special_tokens=False)
    assert ids[-(len(expected_keys) + len(sep)) : -len(sep)] == expected_keys


def test_bucketed_microbatches_reduces_padding_and_keeps_all_examples() -> None:
    tok = _FakeTokenizer()
    chars = ("我", "爱", "北", "京")
    vocab = EmittableVocab.build(chars, tok)
    c2e = vocab.char_to_emit()
    short = [(None, "woai", "我爱")] * 8
    long = [("我爱北京" * 8, "woaibeijing" * 4, "我爱北京")] * 8
    mixed = [x for pair in zip(short, long, strict=True) for x in pair]
    batches = list(bucketed_microbatches(mixed, tok, c2e, batch_size=4, pool_batches=4, seed=0))
    assert len(batches) == 4
    assert sum(len(b) for b in batches) == 16
    # Every bucketed batch is homogeneous (either all short or all long -> 100% pad efficiency).
    for b in batches:
        lengths = {len(encode_prompt(tok, ctx, k)) + len(t) for ctx, k, t in b}
        assert len(lengths) == 1


def test_batched_decode_and_prefix_reusing_labeller_match_single_record(tmp_path: Path) -> None:
    tok = _FakeTokenizer()
    chars = ("我", "爱", "北", "京", "中", "国", "天", "安", "门", "人", "弑", "玑")
    vocab = EmittableVocab.build(chars, tok)
    lora = LoraConfig(rank=4, alpha=8.0, targets=TEACHER_LORA_TARGETS)
    teacher = QwenTeacher(_tiny_qwen35(), vocab, lora, gradient_checkpointing=False).eval()
    legality = _tiny_legality(chars)
    c2e = vocab.char_to_emit()

    # Mix very different prompt and key lengths to stress left-padding and multi-step alignment.
    cases = [
        ("r0", None, "zg", "中国"),
        ("r1", "我爱北京天安门" * 3, "woaibeijing", "我爱北京"),
        ("r2", "历史", "shiji", "弑玑"),
        ("r3", "今天去天安门", "tiananmen", "天安门"),
    ]

    batched_hyps = constrained_beam_decode_batch(
        teacher,
        tok,
        [(ctx, keys) for _, ctx, keys, _ in cases],
        legality,
        OFFLINE,
        beam=4,
        gamma=1.0,
    )
    for (_rid, ctx, keys, _text), b_hyps in zip(cases, batched_hyps, strict=True):
        s_hyps = constrained_beam_decode(
            teacher, tok, ctx, keys, legality, OFFLINE, beam=4, gamma=1.0
        )
        assert [h.text for h in b_hyps] == [h.text for h in s_hyps]
        for bh, sh in zip(b_hyps, s_hyps, strict=True):
            assert bh.score == pytest.approx(sh.score, abs=1e-4)

    batch_records = [
        (rid, ctx, keys, text, [c2e[ch] for ch in text]) for rid, ctx, keys, text in cases
    ]
    labelled = label_records_batch(
        teacher,
        tok,
        batch_records,
        legality,
        OFFLINE,
        beam=4,
        gamma=1.0,
        top_k_t1=5,
        score_batch_size=3,
    )
    assert len(labelled) == len(cases)
    for (rid, ctx, keys, text, gold_ids), row, b_hyps in zip(
        batch_records, labelled, batched_hyps, strict=True
    ):
        assert row["id"] == rid
        assert row["context"] == ctx
        assert row["keys"] == keys
        assert row["text"] == text
        cand_ids = [gold_ids, *[list(h.emit_ids) for h in b_hyps if h.text != text]]
        ref_ids, ref_lps, ref_scores = score_candidates_and_t1(
            teacher, tok, ctx, keys, gold_ids, cand_ids, top_k_t1=5
        )
        assert row["t1_ids"] == ref_ids
        for got_p, ref_p in zip(row["t1_log_probs"], ref_lps, strict=True):
            assert got_p == pytest.approx(ref_p, abs=1e-4)
        assert row["candidate_scores"] == pytest.approx(ref_scores, abs=1e-4)

    # Verify incremental part merge preserves schema including nullable context and cleans parts.
    p0 = tmp_path / ".shard.part-00000.parquet"
    p1 = tmp_path / ".shard.part-00001.parquet"
    write_labels_parquet(labelled[:2], p0)
    write_labels_parquet(labelled[2:3], p1)
    merged_out = tmp_path / "shard.parquet"
    _merge_label_parts([p0, p1], labelled[3:], merged_out)
    assert not p0.exists() and not p1.exists()
    table = pq.read_table(merged_out)
    assert table.num_rows == 4
    assert table.column("context").to_pylist() == [
        None,
        "我爱北京天安门" * 3,
        "历史",
        "今天去天安门",
    ]

    # Verify consume=True frees source cache tensors while _decode_from_prefill preserves prefill.
    prefill = _prefill_prompts(teacher, tok, [("今天", "woaibeijing"), (None, "zg")])
    snap = [
        layer.recurrent_states[0].clone()
        for layer in prefill.prefix_cache.layers
        if hasattr(layer, "recurrent_states")
    ]
    _ = _decode_from_prefill(teacher, prefill, ["woaibeijing", "zg"], legality, OFFLINE, beam=4)
    after = [
        layer.recurrent_states[0]
        for layer in prefill.prefix_cache.layers
        if hasattr(layer, "recurrent_states")
    ]
    for before_t, after_t in zip(snap, after, strict=True):
        assert torch.equal(before_t, after_t)

    consumed = _select_cache_rows(prefill.prefix_cache, [0, 1], consume=False)
    _ = _select_cache_rows(consumed, [0], consume=True)
    for layer in consumed.layers:
        if hasattr(layer, "recurrent_states"):
            assert layer.recurrent_states == {}
        if hasattr(layer, "keys"):
            assert layer.keys is None


def test_expand_record_beams_charges_eos_residual_cost() -> None:
    """Terminating on a costlier final offset must subtract its residual gamma cost (#346)."""
    lattice = _SpanLattice(key_len=5, by_start={}, merged_by_start={})
    lp = np.array([-2.0, -3.0, -0.4], dtype=np.float32)  # eos_emit = 2
    active: list[tuple[float, tuple[int, ...], dict[int, float], np.ndarray, int]] = [
        (-1.2, (0,), {3: 0.0, 5: 2.0}, lp, 0)
    ]
    finished: list[BeamHypothesis] = []
    specs = _expand_record_beams(
        lattice,
        active,
        finished,
        beam=4,
        gamma=1.5,
        eos_emit=2,
        characters=("我", "爱"),
    )
    assert specs == []
    assert len(finished) == 1
    # score (-1.2) + lp[eos] (-0.4) - gamma (1.5) * (offsets[5] (2.0) - min_cost (0.0)) = -4.6
    assert finished[0].text == "我"
    assert finished[0].score == pytest.approx(-4.6, abs=1e-5)
