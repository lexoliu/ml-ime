"""Large LLM teacher for Phase 2 offline distillation (#191, #326).

Fine-tunes ``Qwen/Qwen3.8-27B`` (``Qwen3_5ForCausalLM``, 48 linear-attention +
16 full-attention layers, bf16) with LoRA to predict the target sentence one
hanzi per output step from ``(context <= 62 chars, raw keystrokes)``.

Why one hanzi per output token (#208):
- 5,330 of the 7,322 emittable characters are already single tokens in Qwen
  3.8's vocabulary (99.85% of eval3 occurrences).
- The remaining 1,992 rare emittable characters receive dedicated input
  embedding rows initialised from the mean of their Qwen byte-piece embeddings
  and trained alongside the LoRA adapter.
- Output logits are projected through a dedicated ``(7,323, hidden_size)``
  ``emit_head`` (7,322 emittable characters + ``<eos>``) initialised from Qwen
  3.8's ``lm_head.weight``. This avoids a 248,320-way softmax (34x less output
  compute and memory) and aligns emission indices 1-to-1 with the student's
  ``Lexicon``.
- Resumable checkpoints store only the LoRA adapters, 1,992 residue embedding
  rows, and ``emit_head`` (~0.6 GB instead of 54 GB).
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
import torch.nn.functional as functional
from torch import nn
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from mlime.legality import OFFLINE, Legality, LegalityOptions
from mlime.logging import log
from mlime.train.qwen_tower import CONSTANT_HEAD, LoraConfig, LoraLinear

#: Pinned Qwen 3.8 27B release (#191, #208).
TEACHER_MODEL = "Qwen/Qwen3.8-27B"
TEACHER_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"

#: Every linear projection across Qwen3.5's 48 linear-attention layers,
#: 16 full-attention layers, and 64 SwiGLU MLPs.
TEACHER_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Separator between context, raw keystrokes, and the one-hanzi-per-token output.
FIELD_SEP = "\t"


@dataclass(frozen=True)
class EmittableVocab:
    """Mapping between the student's 7,322 emittable characters (+ ``<eos>``) and Qwen ids.

    Single-token characters keep their Qwen token id; multi-byte residue
    characters receive ids ``base_vocab_size .. base_vocab_size + len(residue) - 1``
    so every output step is one token.
    """

    characters: tuple[str, ...]
    base_vocab_size: int
    eos_token_id: int
    #: ``emit_id -> extended_token_id`` for ``0..len(characters)`` (last is ``<eos>``).
    emit_to_token: tuple[int, ...]
    #: ``(emit_id, byte_piece_ids)`` for each multi-byte residue character, in order.
    residue: tuple[tuple[int, tuple[int, ...]], ...]

    @property
    def size(self) -> int:
        """Number of emittable characters (excluding ``<eos>``)."""
        return len(self.characters)

    @property
    def eos_emit_id(self) -> int:
        """Emission index of ``<eos>``."""
        return len(self.characters)

    @property
    def num_classes(self) -> int:
        """Total output classes: emittable characters plus ``<eos>``."""
        return len(self.characters) + 1

    @property
    def num_residue(self) -> int:
        """How many emittable characters needed new single-token rows."""
        return len(self.residue)

    def char_to_emit(self) -> dict[str, int]:
        """Lookup from character to emission index ``0..size - 1``."""
        return {ch: i for i, ch in enumerate(self.characters)}

    @classmethod
    def build(cls, characters: Sequence[str], tokenizer: Any) -> EmittableVocab:
        """Construct the mapping from *characters* and a Qwen *tokenizer*."""
        base_vocab_size = len(tokenizer)
        eos_id = int(tokenizer.eos_token_id)
        emit_to_token: list[int] = []
        residue: list[tuple[int, tuple[int, ...]]] = []
        for emit_id, ch in enumerate(characters):
            pieces: list[int] = tokenizer.encode(ch, add_special_tokens=False)
            if len(pieces) == 1:
                emit_to_token.append(int(pieces[0]))
            else:
                if not pieces:
                    raise ValueError(f"tokenizer produced no ids for emittable character {ch!r}")
                new_id = base_vocab_size + len(residue)
                emit_to_token.append(new_id)
                residue.append((emit_id, tuple(int(p) for p in pieces)))
        emit_to_token.append(eos_id)
        return cls(
            characters=tuple(characters),
            base_vocab_size=base_vocab_size,
            eos_token_id=eos_id,
            emit_to_token=tuple(emit_to_token),
            residue=tuple(residue),
        )


def format_prompt(context: str | None, keys: str, max_context_chars: int = 62) -> str:
    """Format the canonical-BPE prompt read before the one-hanzi-per-token output."""
    ctx = "" if not context else context[max(0, len(context) - max_context_chars) :]
    return f"{CONSTANT_HEAD}{ctx}{FIELD_SEP}{keys}{FIELD_SEP}"


def encode_prompt(tokenizer: Any, context: str | None, keys: str) -> list[int]:
    """Encode the prompt prefix with Qwen's canonical BPE (no added special tokens)."""
    ids: list[int] = tokenizer.encode(format_prompt(context, keys), add_special_tokens=False)
    return ids


def _apply_lora(module: nn.Module, config: LoraConfig) -> int:
    """Replace every target ``nn.Linear`` in *module* with :class:`LoraLinear`."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, LoraLinear):
            if child_name in config.targets:
                setattr(module, child_name, LoraLinear(child, config.rank, config.alpha))
                replaced += 1
        else:
            replaced += _apply_lora(child, config)
    return replaced


class QwenTeacher(nn.Module):
    """Frozen bf16 Qwen3.5 text backbone + LoRA + residue embeddings + 7,323-way emit head."""

    def __init__(
        self,
        causal_lm: Qwen3_5ForCausalLM,
        vocab: EmittableVocab,
        lora: LoraConfig,
        *,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.lora_config = lora
        self.backbone = causal_lm.model
        hidden_size = int(causal_lm.config.hidden_size)
        base_embed = self.backbone.embed_tokens.weight.detach()
        lm_head_weight = causal_lm.lm_head.weight.detach()
        device = base_embed.device

        # Freeze entire base backbone first.
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        # Initialise trainable fp32 residue input embeddings from byte-piece means.
        self.residue_embed = nn.Embedding(
            max(1, vocab.num_residue), hidden_size, device=device, dtype=torch.float32
        )
        with torch.no_grad():
            self.residue_embed.weight.zero_()
            for idx, (_, pieces) in enumerate(vocab.residue):
                piece_ids = torch.tensor(pieces, dtype=torch.long, device=device)
                self.residue_embed.weight[idx] = (
                    base_embed.index_select(0, piece_ids).float().mean(dim=0)
                )
        if vocab.num_residue == 0:
            self.residue_embed.weight.requires_grad_(False)

        # Initialise trainable fp32 7,323-way output head from Qwen's lm_head.
        self.emit_head = nn.Linear(
            hidden_size, vocab.num_classes, bias=False, device=device, dtype=torch.float32
        )
        with torch.no_grad():
            for emit_id, tok_id in enumerate(vocab.emit_to_token):
                if tok_id < vocab.base_vocab_size:
                    self.emit_head.weight[emit_id] = lm_head_weight[tok_id].float()
            for _, (emit_id, pieces) in enumerate(vocab.residue):
                piece_ids = torch.tensor(pieces, dtype=torch.long, device=device)
                self.emit_head.weight[emit_id] = (
                    lm_head_weight.index_select(0, piece_ids).float().mean(dim=0)
                )

        # Attach LoRA adapters to every linear-attention, full-attention, and MLP projection.
        replaced = _apply_lora(self.backbone, lora)
        if replaced == 0:
            raise ValueError(f"no modules matched LoRA targets {lora.targets}")

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        log.info(
            "teacher initialised",
            lora_layers=replaced,
            rank=lora.rank,
            residue=vocab.num_residue,
            classes=vocab.num_classes,
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    @property
    def backbone_dtype(self) -> torch.dtype:
        """Precision of the frozen base weights (bf16 on G4, fp32 in CPU tests)."""
        dtype: torch.dtype = self.backbone.embed_tokens.weight.dtype
        return dtype

    def embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed extended token ids (base Qwen vocab + residue ids) in ``backbone_dtype``."""
        base_limit = self.vocab.base_vocab_size
        is_residue = input_ids >= base_limit
        safe_base = torch.where(is_residue, torch.zeros_like(input_ids), input_ids)
        embeds: torch.Tensor = self.backbone.embed_tokens(safe_base)
        if self.vocab.num_residue > 0 and bool(is_residue.any()):
            res_idx = torch.where(is_residue, input_ids - base_limit, torch.zeros_like(input_ids))
            res_emb = self.residue_embed(res_idx).to(embeds.dtype)
            embeds = torch.where(is_residue.unsqueeze(-1), res_emb, embeds)
        return embeds

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states ``[..., H]`` to fp32 logits ``[..., 7323]``."""
        return functional.linear(hidden.float(), self.emit_head.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Teacher-forced CE loss over output hanzi + ``<eos>`` positions (``labels >= 0``).

        *input_ids* ``[B, T]`` holds ``prompt + gold_tokens``; *labels* ``[B, T]``
        holds ``-100`` on prompt/padding positions and ``emit_id`` in ``0..7322``
        at each position whose hidden state predicts the next output token.
        Returns ``(loss, scored_tokens)``.
        """
        embeds = self.embed_ids(input_ids)
        out = self.backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden: torch.Tensor = out.last_hidden_state
        mask = labels >= 0
        scored = int(mask.sum().item())
        if scored == 0:
            return hidden.sum() * 0.0, 0
        picked = hidden[mask]
        targets = labels[mask]
        logits = self.project_logits(picked)
        loss = functional.cross_entropy(logits, targets)
        return loss, scored

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Compact state dict containing only trainable weights (~0.6 GB for 27B)."""
        return {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load a saved :meth:`adapter_state_dict` strictly into trainable parameters."""
        trainable = {name: param for name, param in self.named_parameters() if param.requires_grad}
        missing = set(trainable) - set(state)
        unexpected = set(state) - set(trainable)
        if missing or unexpected:
            raise ValueError(
                f"adapter state mismatch: missing={len(missing)} unexpected={len(unexpected)}"
            )
        with torch.no_grad():
            for name, param in trainable.items():
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def build_training_batch(
    tokenizer: Any,
    vocab: EmittableVocab,
    examples: Sequence[tuple[str | None, str, str]],
    *,
    char_to_emit: Mapping[str, int],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate ``(context, keys, target_text)`` triples into training tensors.

    Returns ``(input_ids, attention_mask, labels)``. Every target character
    must exist in *char_to_emit*; callers filter OOV targets before batching.
    Position ``p`` predicts token ``p + 1``, so the last prompt token predicts
    ``target[0]`` and ``target[-1]`` predicts ``<eos>``.
    """
    rows_in: list[list[int]] = []
    rows_lbl: list[list[int]] = []
    for context, keys, text in examples:
        prompt = encode_prompt(tokenizer, context, keys)
        emit_ids = [char_to_emit[ch] for ch in text]
        target_toks = [vocab.emit_to_token[e] for e in emit_ids]
        # Input: prompt + target_toks (len = P + L)
        # Predict at indices P-1 .. P+L-1 -> targets emit_ids + [eos_emit_id]
        inp = prompt + target_toks
        lbl = [-100] * (len(prompt) - 1) + emit_ids + [vocab.eos_emit_id]
        rows_in.append(inp)
        rows_lbl.append(lbl)

    max_len = max(len(r) for r in rows_in)
    batch = len(rows_in)
    input_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    labels = torch.full((batch, max_len), -100, dtype=torch.long)
    for i, (inp, lbl) in enumerate(zip(rows_in, rows_lbl, strict=True)):
        n = len(inp)
        input_ids[i, :n] = torch.tensor(inp, dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(lbl, dtype=torch.long)
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


@dataclass(frozen=True)
class BeamHypothesis:
    """One finished hypothesis from lattice-constrained beam search."""

    text: str
    emit_ids: tuple[int, ...]
    score: float


def _lattice_adjacency(
    legality: Legality, keys: str, options: LegalityOptions
) -> tuple[int, dict[int, list[tuple[int, int, float]]]]:
    """Build ``start_offset -> [(end_offset, emit_id, cost), ...]`` from the pruned lattice."""
    lattice = legality.lattice(keys, options)
    adj: dict[int, list[tuple[int, int, float]]] = {}
    starts = lattice.start.tolist()
    ends = lattice.end.tolist()
    chars = lattice.character.tolist()
    costs = lattice.cost.tolist()
    for s, e, c, cost in zip(starts, ends, chars, costs, strict=True):
        adj.setdefault(int(s), []).append((int(e), int(c), float(cost)))
    return len(keys), adj


def _select_cache_rows(cache: Any, indices: Sequence[int]) -> Any:
    """Index batch rows of a Qwen3.5 hybrid ``DynamicCache`` (linear + full attention)."""
    new = copy.copy(cache)
    idx = torch.as_tensor(indices, dtype=torch.long)
    new_layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        if hasattr(layer, "keys") and layer.keys is not None:
            dev_idx = idx.to(layer.keys.device)
            nl.keys = layer.keys.index_select(0, dev_idx)
            nl.values = layer.values.index_select(0, dev_idx)
        if hasattr(layer, "conv_states"):
            nl.conv_states = {
                k: v.index_select(0, idx.to(v.device)) for k, v in layer.conv_states.items()
            }
            nl.recurrent_states = {
                k: v.index_select(0, idx.to(v.device)) for k, v in layer.recurrent_states.items()
            }
            nl.is_conv_states_initialized = dict(layer.is_conv_states_initialized)
            nl.is_recurrent_states_initialized = dict(layer.is_recurrent_states_initialized)
            nl.has_previous_state = dict(layer.has_previous_state)
            nl.conv_kernel_size = dict(layer.conv_kernel_size)
        new_layers.append(nl)
    new.layers = new_layers
    return new


@torch.no_grad()
def constrained_beam_decode(
    teacher: QwenTeacher,
    tokenizer: Any,
    context: str | None,
    keys: str,
    legality: Legality,
    options: LegalityOptions = OFFLINE,
    *,
    beam: int = 8,
    gamma: float = 1.0,
) -> list[BeamHypothesis]:
    """Batched lattice-constrained beam search over one record, sharing the prefix cache.

    Every output sentence is guaranteed legal under *legality* and *options*.
    Each beam state tracks ``offsets: dict[int, float]`` mapping reachable
    keystroke offsets to their minimum accumulated correction cost. Active beams
    are advanced in a single batched ``[K, 1]`` forward pass per step.
    """
    key_len, adj = _lattice_adjacency(legality, keys, options)
    if not adj or 0 not in adj:
        return []

    device = teacher.emit_head.weight.device
    vocab = teacher.vocab
    eos_emit = vocab.eos_emit_id

    prompt_ids = encode_prompt(tokenizer, context, keys)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = teacher.backbone(
        inputs_embeds=teacher.embed_ids(prompt_tensor),
        use_cache=True,
    )
    first_logits = teacher.project_logits(out.last_hidden_state[0, -1])
    first_log_probs = functional.log_softmax(first_logits, dim=-1).cpu().numpy()
    batch_cache = out.past_key_values

    # Active beam entries: (score, emit_ids, offsets_costs, row_lp)
    active: list[tuple[float, tuple[int, ...], dict[int, float], np.ndarray]] = [
        (0.0, (), {0: 0.0}, first_log_probs)
    ]
    finished: list[BeamHypothesis] = []
    max_steps = key_len

    for _step in range(max_steps + 1):
        if not active:
            break
        candidates: list[tuple[float, bool, tuple[int, ...], dict[int, float], int]] = []
        for parent_idx, (score, emitted, offsets, lp) in enumerate(active):
            # 1. Can we close the sentence here?
            if key_len in offsets:
                end_score = score + float(lp[eos_emit])
                candidates.append((end_score, True, emitted, offsets, parent_idx))
            # 2. Group reachable (emit_id -> next_offsets_costs) from current live offsets.
            next_by_char: dict[int, dict[int, float]] = {}
            for off, cur_cost in offsets.items():
                for end_off, eid, arc_cost in adj.get(off, ()):
                    total_cost = cur_cost + arc_cost
                    bucket = next_by_char.setdefault(eid, {})
                    prev = bucket.get(end_off)
                    if prev is None or total_cost < prev:
                        bucket[end_off] = total_cost
            cur_min_cost = min(offsets.values())
            for eid, next_offs in next_by_char.items():
                next_min_cost = min(next_offs.values())
                delta_cost = next_min_cost - cur_min_cost
                cand_score = score + float(lp[eid]) - gamma * delta_cost
                candidates.append((cand_score, False, (*emitted, eid), next_offs, parent_idx))

        if not candidates:
            break
        candidates.sort(key=lambda item: -item[0])

        next_active_specs: list[tuple[float, tuple[int, ...], dict[int, float], int]] = []
        seen_emitted: set[tuple[int, ...]] = set()
        for cand_score, is_eos, emitted, next_offs, parent_idx in candidates:
            if is_eos:
                text = "".join(vocab.characters[e] for e in emitted)
                finished.append(BeamHypothesis(text=text, emit_ids=emitted, score=cand_score))
                continue
            if emitted in seen_emitted:
                continue
            seen_emitted.add(emitted)
            next_active_specs.append((cand_score, emitted, next_offs, parent_idx))
            if len(next_active_specs) >= beam:
                break

        if not next_active_specs:
            break
        # Stop expanding if we already have `beam` finished hypotheses better than the best active.
        finished.sort(key=lambda h: -h.score)
        if len(finished) >= beam and finished[beam - 1].score >= next_active_specs[0][0]:
            break

        # Advance all active beams together in one batched [K, 1] forward pass.
        parent_indices = [spec[3] for spec in next_active_specs]
        step_toks = [[vocab.emit_to_token[spec[1][-1]]] for spec in next_active_specs]
        step_tensor = torch.tensor(step_toks, dtype=torch.long, device=device)
        child_cache = _select_cache_rows(batch_cache, parent_indices)
        step_out = teacher.backbone(
            inputs_embeds=teacher.embed_ids(step_tensor),
            past_key_values=child_cache,
            use_cache=True,
        )
        step_logits = teacher.project_logits(step_out.last_hidden_state[:, -1])
        step_lps = functional.log_softmax(step_logits, dim=-1).cpu().numpy()
        batch_cache = step_out.past_key_values
        active = [
            (cand_score, emitted, next_offs, step_lps[i])
            for i, (cand_score, emitted, next_offs, _) in enumerate(next_active_specs)
        ]

    # Deduplicate finished hypotheses by text, keeping highest score.
    best_by_text: dict[str, BeamHypothesis] = {}
    for hyp in finished:
        prev_hyp = best_by_text.get(hyp.text)
        if prev_hyp is None or hyp.score > prev_hyp.score:
            best_by_text[hyp.text] = hyp
    ranked = sorted(best_by_text.values(), key=lambda h: -h.score)
    return ranked[:beam]


@torch.no_grad()
def score_candidates_and_t1(
    teacher: QwenTeacher,
    tokenizer: Any,
    context: str | None,
    keys: str,
    gold_emit_ids: Sequence[int],
    candidate_emit_ids: Sequence[Sequence[int]],
    *,
    top_k_t1: int = 32,
) -> tuple[list[list[int]], list[list[float]], list[float]]:
    """Compute T1 (top-k per-position distribution on gold) and T2 (sentence log-probs).

    Shares the prompt prefill and scores gold plus all non-empty candidates in
    a single batched forward pass.
    Returns ``(t1_ids, t1_log_probs, candidate_scores)`` where:
    - ``t1_ids[j]`` / ``t1_log_probs[j]`` are the top-*top_k_t1* emission ids and
      log-probabilities at gold position ``j`` (including ``<eos>`` at ``len(gold)``).
    - ``candidate_scores[c]`` is ``log P_T(candidate_c + <eos> | context, keys)``.
    """
    device = teacher.emit_head.weight.device
    vocab = teacher.vocab
    eos_emit = vocab.eos_emit_id
    k_t1 = min(top_k_t1, vocab.num_classes)

    prompt_ids = encode_prompt(tokenizer, context, keys)
    prompt_len = len(prompt_ids)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prefix_out = teacher.backbone(
        inputs_embeds=teacher.embed_ids(prompt_tensor),
        use_cache=True,
    )
    first_lp = functional.log_softmax(
        teacher.project_logits(prefix_out.last_hidden_state[0, -1]), dim=-1
    )
    prefix_cache = prefix_out.past_key_values

    t1_ids: list[list[int]] = []
    t1_log_probs: list[list[float]] = []
    vals0, idxs0 = first_lp.topk(k_t1)
    t1_ids.append([int(x) for x in idxs0.tolist()])
    t1_log_probs.append([float(x) for x in vals0.tolist()])

    # Row 0 is gold (if non-empty); subsequent rows are non-empty candidates.
    all_seqs: list[Sequence[int]] = [gold_emit_ids, *candidate_emit_ids]
    non_empty_indices = [i for i, seq in enumerate(all_seqs) if len(seq) > 0]
    row_scores: dict[int, float] = {
        i: float(first_lp[eos_emit].item()) for i, seq in enumerate(all_seqs) if len(seq) == 0
    }

    if non_empty_indices:
        max_len = max(len(all_seqs[i]) for i in non_empty_indices)
        batch_n = len(non_empty_indices)
        pad_tok = int(
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        seq_tensor = torch.full((batch_n, max_len), pad_tok, dtype=torch.long, device=device)
        attn_mask = torch.ones((batch_n, prompt_len + max_len), dtype=torch.long, device=device)
        for row_i, seq_idx in enumerate(non_empty_indices):
            seq = all_seqs[seq_idx]
            toks = [vocab.emit_to_token[e] for e in seq]
            seq_tensor[row_i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
            if len(toks) < max_len:
                attn_mask[row_i, prompt_len + len(toks) :] = 0

        expanded_cache = _select_cache_rows(prefix_cache, [0] * batch_n)
        out = teacher.backbone(
            inputs_embeds=teacher.embed_ids(seq_tensor),
            attention_mask=attn_mask,
            past_key_values=expanded_cache,
            use_cache=False,
        )
        logits = teacher.project_logits(out.last_hidden_state)
        lps = functional.log_softmax(logits, dim=-1)

        for row_i, seq_idx in enumerate(non_empty_indices):
            seq = all_seqs[seq_idx]
            total = float(first_lp[seq[0]].item())
            targets = [*seq[1:], eos_emit]
            for pos, tgt in enumerate(targets):
                total += float(lps[row_i, pos, tgt].item())
                if seq_idx == 0:
                    vals, idxs = lps[row_i, pos].topk(k_t1)
                    t1_ids.append([int(x) for x in idxs.tolist()])
                    t1_log_probs.append([float(x) for x in vals.tolist()])
            row_scores[seq_idx] = total

    scores = [row_scores[i + 1] for i in range(len(candidate_emit_ids))]
    return t1_ids, t1_log_probs, scores


@dataclass(frozen=True)
class TeacherTrainConfig:
    """Hyperparameters for LoRA fine-tuning of the Qwen3.8-27B teacher (#191)."""

    model: str = TEACHER_MODEL
    revision: str | None = TEACHER_REVISION
    lora_rank: int = 64
    lora_alpha: float = 128.0
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.04
    batch_size: int = 16
    accumulate: int = 4
    max_steps: int = 3125  # ~200k examples at effective batch 64
    gradient_clip: float = 1.0
    seed: int = 0
    log_every: int = 10
    checkpoint_every: int = 250
    gradient_checkpointing: bool = True


def write_labels_parquet(rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    """Write offline T1/T2/T3 teacher labels for one shard atomically to Parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("keys", pa.string()),
            ("text", pa.string()),
            ("t1_ids", pa.list_(pa.list_(pa.int32()))),
            ("t1_log_probs", pa.list_(pa.list_(pa.float32()))),
            ("candidates", pa.list_(pa.string())),
            ("candidate_scores", pa.list_(pa.float32())),
        ]
    )
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out_path)


def load_teacher(
    characters: Sequence[str],
    *,
    model_name: str = TEACHER_MODEL,
    revision: str | None = TEACHER_REVISION,
    lora_rank: int = 64,
    lora_alpha: float = 128.0,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
    gradient_checkpointing: bool = True,
    adapter_path: Path | None = None,
) -> tuple[QwenTeacher, Any]:
    """Load Qwen3.8-27B (or compatible Qwen3_5ForCausalLM), wrap with LoRA + emit head."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    vocab = EmittableVocab.build(characters, tokenizer)
    causal_lm = Qwen3_5ForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
    ).to(device)
    lora = LoraConfig(rank=lora_rank, alpha=lora_alpha, targets=TEACHER_LORA_TARGETS)
    teacher = QwenTeacher(causal_lm, vocab, lora, gradient_checkpointing=gradient_checkpointing)
    if adapter_path is not None:
        ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
        state = ckpt.get("adapter", ckpt)
        teacher.load_adapter_state_dict(state)
        log.info("teacher adapter loaded", path=str(adapter_path))
    return teacher, tokenizer


def save_teacher_checkpoint(
    out_dir: Path,
    teacher: QwenTeacher,
    optimiser: torch.optim.Optimizer,
    step: int,
    config: TeacherTrainConfig,
    *,
    filename: str = "teacher-checkpoint.pt",
) -> Path:
    """Atomically write a compact resumable teacher checkpoint (~0.6 GB)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "step": step,
            "config": asdict(config),
            "adapter": teacher.adapter_state_dict(),
            "optimiser": optimiser.state_dict(),
            "characters": teacher.vocab.characters,
        },
        tmp,
    )
    tmp.replace(path)
    return path


def train_teacher_loop(
    teacher: QwenTeacher,
    tokenizer: Any,
    stream: Iterable[tuple[str | None, str, str]],
    out_dir: Path,
    config: TeacherTrainConfig,
    *,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Resumable LoRA training loop over ``(context, keys, target_text)`` triples."""
    device = teacher.emit_head.weight.device
    char_to_emit = teacher.vocab.char_to_emit()
    pad_id = int(
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    trainable = [p for p in teacher.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)

    start_step = 0
    if resume_path is not None and resume_path.is_file():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        teacher.load_adapter_state_dict(ckpt["adapter"])
        optimiser.load_state_dict(ckpt["optimiser"])
        start_step = int(ckpt["step"])
        log.info("teacher resumed", path=str(resume_path), step=start_step)

    warmup = max(1, int(config.max_steps * config.warmup_fraction))

    def lr_for(step: int) -> float:
        if step <= warmup:
            return config.lr * (step / warmup)
        progress = (step - warmup) / max(1, config.max_steps - warmup)
        return config.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    teacher.train()
    iterator = iter(stream)
    t0 = time.monotonic()
    last_loss = 0.0
    step = start_step

    while step < config.max_steps:
        step += 1
        cur_lr = lr_for(step)
        for group in optimiser.param_groups:
            group["lr"] = cur_lr
        optimiser.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_tokens = 0
        step_examples = 0
        for _micro in range(config.accumulate):
            batch_items: list[tuple[str | None, str, str]] = []
            while len(batch_items) < config.batch_size:
                try:
                    ctx, keys, text = next(iterator)
                except StopIteration:
                    break
                if all(ch in char_to_emit for ch in text):
                    batch_items.append((ctx, keys, text))
            if not batch_items:
                break
            input_ids, attn_mask, labels = build_training_batch(
                tokenizer,
                teacher.vocab,
                batch_items,
                char_to_emit=char_to_emit,
                pad_token_id=pad_id,
                device=device,
            )
            loss, scored = teacher(input_ids, attn_mask, labels)
            if scored > 0:
                (loss / config.accumulate).backward()
                step_loss += float(loss.detach().item()) / config.accumulate
                step_tokens += scored
                step_examples += len(batch_items)

        if step_examples == 0:
            step -= 1
            break

        torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
        optimiser.step()
        last_loss = step_loss

        if step == start_step + 1 or step % config.log_every == 0 or step == config.max_steps:
            elapsed = time.monotonic() - t0
            record = {
                "event": "step",
                "step": step,
                "loss": step_loss,
                "lr": cur_lr,
                "examples": step_examples,
                "tokens": step_tokens,
                "seconds": round(elapsed, 2),
            }
            with metrics_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(record) + "\n")
            log.info(
                "teacher step",
                step=step,
                loss=round(step_loss, 4),
                lr=cur_lr,
                examples=step_examples,
                tokens=step_tokens,
                seconds=round(elapsed, 2),
            )

        if step % config.checkpoint_every == 0 and step < config.max_steps:
            save_teacher_checkpoint(out_dir, teacher, optimiser, step, config)

    final_path = save_teacher_checkpoint(
        out_dir, teacher, optimiser, step, config, filename="teacher-final.pt"
    )
    return {
        "steps": step,
        "last_loss": last_loss,
        "seconds": round(time.monotonic() - t0, 2),
        "checkpoint": str(final_path),
    }


def deterministic_shuffle(items: list[Any], seed: int) -> list[Any]:
    """Return a deterministically shuffled copy of *items*."""
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def load_student_lexicon(char_table: Path) -> Any:
    """Build the 7,322-character student :class:`~mlime.train.lexicon.Lexicon`."""
    from mlime.train.model import RouteAConfig
    from mlime.train.run import build_lexicon_for, load_tokenizer
    from mlime.train.spans import SpanVocab

    spans = SpanVocab.load()
    return build_lexicon_for(char_table, load_tokenizer(RouteAConfig().base_model), spans)


def corpus_teacher_stream(
    samples_dir: Path,
    labels_dir: Path,
    char_table: Path,
    *,
    shards: Sequence[str] | None = None,
    typos: bool = True,
    typo_table: Path | None = None,
    syllables: Path | None = None,
    unit_initials: float = 0.0,
    unit_lexicon: Path | None = None,
    context_dropout: float = 0.30,
    seed: int = 0,
    skip_examples: int = 0,
) -> Iterable[tuple[str | None, str, str]]:
    """Stream ``(context, raw_keys, target_text)`` triples from corpus shards.

    Reuses :class:`~mlime.train.samples.CorpusStream` and
    :class:`~mlime.train.samples.SampleBuilder` so typing styles (55/25/20),
    typos, unit-initials, and interleaved shard order match the student's
    training distribution. *skip_examples* fast-forwards past examples already
    consumed by a resumed run.
    """
    from mlime.locate import find_upwards
    from mlime.train.arbitration import ReadingArbitration
    from mlime.train.lexicon import CandidateSpace
    from mlime.train.samples import Augmentation, CorpusStream, SampleBuilder
    from mlime.train.spans import SpanVocab
    from mlime.train.units import UnitMatcher
    from mlime.typo import (
        SYLLABLES_RELATIVE,
        CorrectionTable,
        NoiseModel,
        default_typo_table,
        load_syllables,
    )

    spans = SpanVocab.load()
    lexicon = load_student_lexicon(char_table)
    arbitration = ReadingArbitration.load(spans)
    noise: NoiseModel | None = None
    candidates: CandidateSpace | None = None
    if typos:
        table_path = typo_table or default_typo_table()
        if table_path is None:
            raise FileNotFoundError("no typo.json found; pass --typo-table")
        syl_path = syllables or find_upwards(SYLLABLES_RELATIVE)
        if syl_path is None:
            raise FileNotFoundError("no syllables.txt found; pass --syllables")
        noise = NoiseModel.load(table_path)
        table = CorrectionTable(noise, load_syllables(syl_path))
        candidates = CandidateSpace(spans, lexicon.candidate_mask, table)
    units: UnitMatcher | None = None
    if unit_initials > 0.0:
        if unit_lexicon is None:
            raise ValueError("unit_initials > 0 requires unit_lexicon")
        units = UnitMatcher.load(unit_lexicon)
    aug = Augmentation(typos=typos, unit_initials=unit_initials)
    builder = SampleBuilder(
        lexicon,
        spans,
        arbitration,
        augmentation=aug,
        seed=seed,
        noise=noise,
        candidates=candidates,
        units=units,
    )
    stream = CorpusStream(samples_dir, labels_dir, builder, shards=shards, epoch=0)
    rng = random.Random(seed + 1_000_003)
    seen = 0
    epoch = 0
    while True:
        stream.set_epoch(epoch)
        for ex in stream:
            drop_ctx = ex.context is None or rng.random() < context_dropout
            seen += 1
            if seen <= skip_examples:
                continue
            ctx = None if drop_ctx else ex.context
            keys = "".join(ex.spans)
            text = "".join(lexicon.characters[t] for t in ex.targets)
            yield ctx, keys, text
        epoch += 1


def evaluate_teacher(
    teacher: QwenTeacher,
    tokenizer: Any,
    eval_set: Path,
    legality: Legality,
    *,
    slice_: str = "dev",
    dev_share: float = 0.0905,
    with_context: bool = True,
    beam: int = 8,
    gamma: float = 1.0,
    options: LegalityOptions = OFFLINE,
    dump_path: Path | None = None,
    limit: int | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Evaluate *teacher* with lattice-constrained beam search on *eval_set*.

    Resumable when *dump_path* is given: previously written rows are reloaded
    and skipped so an interrupted evaluation continues where it left off.
    """
    from mlime.gui import read_eval_set, slice_indices
    from mlime.train.e2e_eval import Report, read_eval_records

    rows = read_eval_records(eval_set)
    indices = slice_indices(read_eval_set(eval_set), slice_, dev_share)
    if limit is not None:
        indices = indices[:limit]

    existing: dict[int, dict[str, Any]] = {}
    if dump_path is not None and dump_path.is_file():
        for line in dump_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            existing[int(rec["index"])] = rec
        if existing:
            log.info("teacher eval resumed", path=str(dump_path), done=len(existing))

    teacher.eval()
    report = Report(top_k=beam)
    results: list[dict[str, Any]] = []
    t0 = time.monotonic()

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)

    for pos, idx in enumerate(indices):
        row = rows[idx]
        if idx in existing:
            rec = existing[idx]
        else:
            ctx = row.context if with_context else None
            hyps = constrained_beam_decode(
                teacher,
                tokenizer,
                ctx,
                row.pinyin,
                legality,
                options,
                beam=beam,
                gamma=gamma,
            )
            rec = {
                "index": idx,
                "pinyin": row.pinyin,
                "expected": row.text,
                "hypotheses": [h.text for h in hyps],
                "scores": [h.score for h in hyps],
            }
            if dump_path is not None:
                with dump_path.open("a", encoding="utf-8") as df:
                    df.write(json.dumps(rec, ensure_ascii=False) + "\n")
        report.observe(row.text, rec["hypotheses"])
        results.append(rec)
        if (pos + 1) % 50 == 0 or (pos + 1) == len(indices):
            elapsed = time.monotonic() - t0
            log.info(
                "teacher eval progress",
                done=pos + 1,
                total=len(indices),
                top1=round(report.top1_hits / max(1, report.records), 4),
                seconds=round(elapsed, 1),
            )
    return report, results


def label_corpus_shards(
    teacher: QwenTeacher,
    tokenizer: Any,
    samples_dir: Path,
    labels_dir: Path,
    char_table: Path,
    legality: Legality,
    out_dir: Path,
    *,
    shards: Sequence[str] | None = None,
    typos: bool = True,
    typo_table: Path | None = None,
    syllables: Path | None = None,
    unit_initials: float = 0.0,
    unit_lexicon: Path | None = None,
    seed: int = 0,
    beam: int = 8,
    gamma: float = 1.0,
    top_k_t1: int = 32,
    max_per_shard: int | None = None,
) -> dict[str, Any]:
    """Generate resumable offline T1/T2/T3 Parquet labels shard by shard (S5)."""
    from mlime.data.shards import shard_paths
    from mlime.locate import find_upwards
    from mlime.train.arbitration import ReadingArbitration
    from mlime.train.lexicon import CandidateSpace
    from mlime.train.samples import Augmentation, SampleBuilder, interleave, read_pair
    from mlime.train.spans import SpanVocab
    from mlime.train.units import UnitMatcher
    from mlime.typo import (
        SYLLABLES_RELATIVE,
        CorrectionTable,
        NoiseModel,
        default_typo_table,
        load_syllables,
    )

    spans = SpanVocab.load()
    lexicon = load_student_lexicon(char_table)
    arbitration = ReadingArbitration.load(spans)
    noise: NoiseModel | None = None
    candidates: CandidateSpace | None = None
    if typos:
        table_path = typo_table or default_typo_table()
        if table_path is None:
            raise FileNotFoundError("no typo.json found; pass --typo-table")
        syl_path = syllables or find_upwards(SYLLABLES_RELATIVE)
        if syl_path is None:
            raise FileNotFoundError("no syllables.txt found; pass --syllables")
        noise = NoiseModel.load(table_path)
        table = CorrectionTable(noise, load_syllables(syl_path))
        candidates = CandidateSpace(spans, lexicon.candidate_mask, table)
    units: UnitMatcher | None = None
    if unit_initials > 0.0:
        if unit_lexicon is None:
            raise ValueError("unit_initials > 0 requires unit_lexicon")
        units = UnitMatcher.load(unit_lexicon)

    builder = SampleBuilder(
        lexicon,
        spans,
        arbitration,
        augmentation=Augmentation(typos=typos, unit_initials=unit_initials),
        seed=seed,
        noise=noise,
        candidates=candidates,
        units=units,
    )
    available = shard_paths(samples_dir, "*")
    if shards is not None:
        wanted = set(shards)
        selected = [p for p in available if p.name in wanted]
    else:
        selected = available
    selected = interleave(selected)

    char_to_emit = teacher.vocab.char_to_emit()
    teacher.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    written_shards = 0
    skipped_shards = 0
    total_rows = 0

    for shard_path in selected:
        out_pq = out_dir / shard_path.name
        if out_pq.is_file():
            skipped_shards += 1
            log.info("teacher label shard skipped (exists)", shard=shard_path.name)
            continue
        rows_out: list[dict[str, Any]] = []
        for sample, syls in read_pair(shard_path, labels_dir):
            ex = builder.build(sample, syls, epoch=0)
            if ex is None:
                continue
            keys = "".join(ex.spans)
            text = "".join(lexicon.characters[t] for t in ex.targets)
            if not all(ch in char_to_emit for ch in text):
                continue
            gold_ids = [char_to_emit[ch] for ch in text]
            hyps = constrained_beam_decode(
                teacher, tokenizer, ex.context, keys, legality, OFFLINE, beam=beam, gamma=gamma
            )
            cand_texts: list[str] = [text]
            cand_ids: list[list[int]] = [gold_ids]
            for h in hyps:
                if h.text != text:
                    cand_texts.append(h.text)
                    cand_ids.append(list(h.emit_ids))
            t1_ids, t1_lps, cand_scores = score_candidates_and_t1(
                teacher, tokenizer, ex.context, keys, gold_ids, cand_ids, top_k_t1=top_k_t1
            )
            rows_out.append(
                {
                    "id": ex.id,
                    "keys": keys,
                    "text": text,
                    "t1_ids": t1_ids,
                    "t1_log_probs": t1_lps,
                    "candidates": cand_texts,
                    "candidate_scores": cand_scores,
                }
            )
            if max_per_shard is not None and len(rows_out) >= max_per_shard:
                break
        write_labels_parquet(rows_out, out_pq)
        written_shards += 1
        total_rows += len(rows_out)
        log.info("teacher label shard written", shard=shard_path.name, rows=len(rows_out))

    return {
        "written_shards": written_shards,
        "skipped_shards": skipped_shards,
        "total_rows": total_rows,
    }
