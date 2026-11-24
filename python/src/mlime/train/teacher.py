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
    """Format the readable prompt prefix before the one-hanzi-per-token output."""
    ctx = "" if not context else context[max(0, len(context) - max_context_chars) :]
    return f"{CONSTANT_HEAD}{ctx}{FIELD_SEP}{keys}{FIELD_SEP}"


def encode_prompt(
    tokenizer: Any, context: str | None, keys: str, max_context_chars: int = 62
) -> list[int]:
    """Encode *context* with canonical BPE and *keys* one ASCII letter per token.

    Encoding raw pinyin through BPE splits long keystrokes into arbitrary
    English/multilingual subwords (e.g. ``mageqiujiganfakeyidea`` ->
    ``['mage', 'qi', 'uj', 'igan', 'fake', 'y', 'idea']``), collapsing accuracy
    on long full-pinyin sentences (#328). Every ``a..z`` keystroke is a single
    token in Qwen's vocabulary, so encoding keys one character at a time keeps
    every letter aligned cleanly while preserving canonical BPE for Chinese
    context.
    """
    ctx = "" if not context else context[max(0, len(context) - max_context_chars) :]
    prefix_ids: list[int] = tokenizer.encode(
        f"{CONSTANT_HEAD}{ctx}{FIELD_SEP}", add_special_tokens=False
    )
    key_ids: list[int] = []
    for ch in keys:
        ch_ids: list[int] = tokenizer.encode(ch, add_special_tokens=False)
        key_ids.extend(ch_ids)
    sep_ids: list[int] = tokenizer.encode(FIELD_SEP, add_special_tokens=False)
    return [*prefix_ids, *key_ids, *sep_ids]


class TeacherLoraLinear(LoraLinear):
    """BF16 Tensor Core LoRA matmul with fp32 master weights for AdamW (#328).

    Casts the small ``[rank, in]`` / ``[out, rank]`` factors to ``x.dtype``
    instead of casting the full ``[B, T, H]`` activation tensor to fp32 across
    400 adapted projections (forward and gradient-checkpointed recompute).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base: torch.Tensor = nn.Linear.forward(self, x)
        a = self.lora_a.to(x.dtype)
        b = self.lora_b.to(x.dtype)
        update = (x @ a.T) @ b.T
        result: torch.Tensor = base + self.scale * update
        return result


def _apply_lora(module: nn.Module, config: LoraConfig) -> int:
    """Replace every target ``nn.Linear`` in *module* with :class:`TeacherLoraLinear`."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, LoraLinear):
            if child_name in config.targets:
                setattr(module, child_name, TeacherLoraLinear(child, config.rank, config.alpha))
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


@dataclass(frozen=True)
class _SpanLattice:
    """Pruned legality spans indexed by start keystroke offset.

    Stores the memoised NumPy ``(ids, costs)`` arrays from
    :meth:`~mlime.legality.Legality.span_matches` directly and pre-merges
    per-start-offset ``(uniq_ids, min_gamma_costs)`` in float32 so beam
    expansion runs one vectorized score op per live offset instead of looping
    over thousands of flattened character arcs.
    """

    key_len: int
    by_start: dict[int, list[tuple[int, np.ndarray, np.ndarray]]]
    merged_by_start: dict[int, tuple[np.ndarray, np.ndarray]]


def _span_lattice(
    legality: Legality, keys: str, options: LegalityOptions, *, gamma: float = 1.0
) -> _SpanLattice:
    """Build a pruned :class:`_SpanLattice` for *keys* with pre-merged per-offset costs."""
    from mlime.legality import MAX_ARC_KEYS

    if not keys or not keys.isascii() or not keys.isalpha() or not keys.islower():
        raise ValueError(f"keys must be a non-empty [a-z] string, got {keys!r}")
    width = len(keys)
    edges: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for start in range(width):
        for end in range(start + 1, min(start + MAX_ARC_KEYS, width) + 1):
            matches = legality.span_matches(keys[start:end], end == width, options)
            if matches.ids.size:
                edges[(start, end)] = (matches.ids, matches.costs)
    reachable = [False] * (width + 1)
    reachable[0] = True
    for start, end in sorted(edges):
        if reachable[start]:
            reachable[end] = True
    productive = [False] * (width + 1)
    productive[width] = True
    for start, end in sorted(edges, reverse=True):
        if productive[end]:
            productive[start] = True
    by_start: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for (start, end), (ids, costs) in sorted(edges.items()):
        if reachable[start] and productive[end]:
            by_start.setdefault(start, []).append((end, ids, costs))

    gamma_f32 = np.float32(gamma)
    merged_by_start: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for start, spans in by_start.items():
        if len(spans) == 1:
            _, ids, costs = spans[0]
            merged_by_start[start] = (ids, (costs * gamma).astype(np.float32))
            continue
        all_ids = np.concatenate([ids for _, ids, _ in spans])
        all_costs = np.concatenate([costs for _, _, costs in spans])
        order = np.lexsort((all_costs, all_ids))
        s_ids = all_ids[order]
        s_costs = all_costs[order]
        first = np.ones(s_ids.shape, dtype=bool)
        first[1:] = s_ids[1:] != s_ids[:-1]
        merged_by_start[start] = (s_ids[first], s_costs[first].astype(np.float32) * gamma_f32)
    return _SpanLattice(width, by_start, merged_by_start)


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


@dataclass
class _PrefillBatch:
    """Left-padded prompt prefill shared across batched beam search and T1/T2 scoring."""

    prompt_lens: list[int]
    max_prompt_len: int
    attn_mask: torch.Tensor
    first_log_probs: torch.Tensor
    prefix_cache: Any


@torch.no_grad()
def _prefill_prompts(
    teacher: QwenTeacher,
    tokenizer: Any,
    records: Sequence[tuple[str | None, str]],
) -> _PrefillBatch:
    """Prefill a batch of ``(context, keys)`` prompts with left-padding and exact RoPE positions.

    Left-padding keeps ``Qwen3_5GatedDeltaNet`` conv/recurrent states and full-attention KV
    outputs numerically identical to unpadded single-record prefills while saturating the GPU.
    """
    device = teacher.emit_head.weight.device
    pad_id = int(
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    prompts = [encode_prompt(tokenizer, ctx, keys) for ctx, keys in records]
    prompt_lens = [len(p) for p in prompts]
    max_prompt_len = max(prompt_lens)
    batch = len(prompts)

    input_ids = torch.full((batch, max_prompt_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((batch, max_prompt_len), dtype=torch.long)
    for i, prompt in enumerate(prompts):
        n = len(prompt)
        input_ids[i, max_prompt_len - n :] = torch.tensor(prompt, dtype=torch.long)
        attn_mask[i, max_prompt_len - n :] = 1
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    position_ids = (attn_mask.cumsum(dim=-1) - 1).clamp_min(0)
    cache_position = torch.arange(max_prompt_len, dtype=torch.long, device=device)

    out = teacher.backbone(
        inputs_embeds=teacher.embed_ids(input_ids),
        attention_mask=attn_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
    )
    first_logits = teacher.project_logits(out.last_hidden_state[:, -1])
    first_log_probs = functional.log_softmax(first_logits, dim=-1)
    return _PrefillBatch(
        prompt_lens=prompt_lens,
        max_prompt_len=max_prompt_len,
        attn_mask=attn_mask,
        first_log_probs=first_log_probs,
        prefix_cache=out.past_key_values,
    )


def _expand_record_beams(
    lattice: _SpanLattice,
    active: list[tuple[float, tuple[int, ...], dict[int, float], np.ndarray, int]],
    finished: list[BeamHypothesis],
    *,
    beam: int,
    gamma: float,
    eos_emit: int,
    characters: tuple[str, ...],
) -> list[tuple[float, tuple[int, ...], dict[int, float], int]]:
    """Expand one record's active beams using span-vectorized NumPy top-K selection.

    Returns the surviving ``(score, new_emitted, next_offsets, parent_cache_row)``
    specs (up to *beam*), or an empty list when the record has finished decoding.
    """
    per_offset_k = beam + 4
    candidates: list[tuple[float, bool, tuple[int, ...], int, int]] = []

    for parent_idx, (score, emitted, offsets, lp, _cache_row) in enumerate(active):
        if lattice.key_len in offsets:
            end_score = score + float(lp[eos_emit])
            candidates.append((end_score, True, emitted, -1, parent_idx))

        cur_min_cost = min(offsets.values())
        best_by_eid: dict[int, float] = {}
        for off, cur_cost in offsets.items():
            merged = lattice.merged_by_start.get(off)
            if merged is None:
                continue
            uniq_ids, min_g_costs = merged
            shift = np.float32(gamma * (cur_cost - cur_min_cost))
            scores_arr = (
                lp[uniq_ids] - min_g_costs if shift == 0.0 else lp[uniq_ids] - (min_g_costs + shift)
            )
            n = uniq_ids.size
            if n > per_offset_k:
                top_idx = np.argpartition(scores_arr, n - per_offset_k)[n - per_offset_k :]
                top_ids = uniq_ids[top_idx].tolist()
                top_sc = scores_arr[top_idx].tolist()
            else:
                top_ids = uniq_ids.tolist()
                top_sc = scores_arr.tolist()
            for eid, rel_s in zip(top_ids, top_sc, strict=True):
                cand_s = score + rel_s
                prev_s = best_by_eid.get(eid)
                if prev_s is None or cand_s > prev_s:
                    best_by_eid[eid] = cand_s

        for eid, cand_s in best_by_eid.items():
            candidates.append((cand_s, False, (*emitted, eid), eid, parent_idx))

    if not candidates:
        return []
    candidates.sort(key=lambda item: -item[0])

    next_specs: list[tuple[float, tuple[int, ...], dict[int, float], int]] = []
    seen_emitted: set[tuple[int, ...]] = set()
    for cand_score, is_eos, emitted, eid, parent_idx in candidates:
        if is_eos:
            text = "".join(characters[e] for e in emitted)
            finished.append(BeamHypothesis(text=text, emit_ids=emitted, score=cand_score))
            continue
        if emitted in seen_emitted:
            continue
        seen_emitted.add(emitted)
        next_offs: dict[int, float] = {}
        parent_offsets = active[parent_idx][2]
        for off, cur_cost in parent_offsets.items():
            for end_off, ids, arc_costs in lattice.by_start.get(off, ()):
                pos = int(np.searchsorted(ids, eid))
                if pos < ids.size and int(ids[pos]) == eid:
                    cost = cur_cost + float(arc_costs[pos])
                    prev_cost = next_offs.get(end_off)
                    if prev_cost is None or cost < prev_cost:
                        next_offs[end_off] = cost
        next_specs.append((cand_score, emitted, next_offs, active[parent_idx][4]))
        if len(next_specs) >= beam:
            break

    if not next_specs:
        return []
    finished.sort(key=lambda h: -h.score)
    if len(finished) >= beam and finished[beam - 1].score >= next_specs[0][0]:
        return []
    return next_specs


def _finalise_hypotheses(finished: Sequence[BeamHypothesis], beam: int) -> list[BeamHypothesis]:
    """Deduplicate finished hypotheses by text, keeping the highest score per text."""
    best_by_text: dict[str, BeamHypothesis] = {}
    for hyp in finished:
        prev_hyp = best_by_text.get(hyp.text)
        if prev_hyp is None or hyp.score > prev_hyp.score:
            best_by_text[hyp.text] = hyp
    ranked = sorted(best_by_text.values(), key=lambda h: -h.score)
    return ranked[:beam]


@torch.no_grad()
def _decode_from_prefill(
    teacher: QwenTeacher,
    prefill: _PrefillBatch,
    keys_list: Sequence[str],
    legality: Legality,
    options: LegalityOptions = OFFLINE,
    *,
    beam: int = 8,
    gamma: float = 1.0,
) -> list[list[BeamHypothesis]]:
    """Run batched multi-record lattice-constrained beam search from a shared *prefill*.

    Does not mutate ``prefill.prefix_cache`` (``_select_cache_rows`` always allocates fresh
    tensors before the backbone updates conv/recurrent states), so callers can immediately reuse
    *prefill* for T1/T2 scoring.
    """
    n_records = len(keys_list)
    if n_records == 0:
        return []
    device = teacher.emit_head.weight.device
    vocab = teacher.vocab
    eos_emit = vocab.eos_emit_id

    lattices = [_span_lattice(legality, k, options, gamma=gamma) for k in keys_list]
    first_lps_np = prefill.first_log_probs.cpu().numpy()

    # Per-record active beams: (score, emitted, offsets, row_lp, cache_row_idx)
    per_rec_active: list[list[tuple[float, tuple[int, ...], dict[int, float], np.ndarray, int]]] = [
        [] for _ in range(n_records)
    ]
    per_rec_finished: list[list[BeamHypothesis]] = [[] for _ in range(n_records)]
    for r, lat in enumerate(lattices):
        if lat.by_start and 0 in lat.by_start:
            per_rec_active[r] = [(0.0, (), {0: 0.0}, first_lps_np[r], r)]

    batch_cache = prefill.prefix_cache
    cur_attn_mask = prefill.attn_mask
    max_steps = max((lat.key_len for lat in lattices), default=0)

    for step_t in range(max_steps + 1):
        parent_cache_rows: list[int] = []
        step_toks: list[list[int]] = []
        step_pos_ids: list[list[int]] = []
        flat_meta: list[tuple[int, float, tuple[int, ...], dict[int, float]]] = []

        for r in range(n_records):
            if not per_rec_active[r]:
                continue
            next_specs = _expand_record_beams(
                lattices[r],
                per_rec_active[r],
                per_rec_finished[r],
                beam=beam,
                gamma=gamma,
                eos_emit=eos_emit,
                characters=vocab.characters,
            )
            per_rec_active[r] = []
            for cand_score, emitted, next_offs, parent_cache_row in next_specs:
                parent_cache_rows.append(parent_cache_row)
                step_toks.append([vocab.emit_to_token[emitted[-1]]])
                step_pos_ids.append([prefill.prompt_lens[r] + step_t])
                flat_meta.append((r, cand_score, emitted, next_offs))

        if not flat_meta:
            break

        parent_idx_t = torch.tensor(parent_cache_rows, dtype=torch.long, device=device)
        child_cache = _select_cache_rows(batch_cache, parent_cache_rows)
        step_mask = torch.cat(
            [
                cur_attn_mask.index_select(0, parent_idx_t),
                torch.ones((len(flat_meta), 1), dtype=torch.long, device=device),
            ],
            dim=1,
        )
        step_tensor = torch.tensor(step_toks, dtype=torch.long, device=device)
        pos_tensor = torch.tensor(step_pos_ids, dtype=torch.long, device=device)
        cache_pos = torch.tensor([prefill.max_prompt_len + step_t], dtype=torch.long, device=device)
        step_out = teacher.backbone(
            inputs_embeds=teacher.embed_ids(step_tensor),
            attention_mask=step_mask,
            position_ids=pos_tensor,
            cache_position=cache_pos,
            past_key_values=child_cache,
            use_cache=True,
        )
        step_logits = teacher.project_logits(step_out.last_hidden_state[:, -1])
        step_lps = functional.log_softmax(step_logits, dim=-1).cpu().numpy()
        batch_cache = step_out.past_key_values
        cur_attn_mask = step_mask

        for i, (r, cand_score, emitted, next_offs) in enumerate(flat_meta):
            per_rec_active[r].append((cand_score, emitted, next_offs, step_lps[i], i))

    return [_finalise_hypotheses(fin, beam) for fin in per_rec_finished]


@torch.no_grad()
def constrained_beam_decode_batch(
    teacher: QwenTeacher,
    tokenizer: Any,
    records: Sequence[tuple[str | None, str]],
    legality: Legality,
    options: LegalityOptions = OFFLINE,
    *,
    beam: int = 8,
    gamma: float = 1.0,
) -> list[list[BeamHypothesis]]:
    """Batched lattice-constrained beam search across multiple ``(context, keys)`` records."""
    if not records:
        return []
    prefill = _prefill_prompts(teacher, tokenizer, records)
    return _decode_from_prefill(
        teacher,
        prefill,
        [keys for _, keys in records],
        legality,
        options,
        beam=beam,
        gamma=gamma,
    )


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
    """Batched lattice-constrained beam search over one record, sharing the prefix cache."""
    return constrained_beam_decode_batch(
        teacher,
        tokenizer,
        [(context, keys)],
        legality,
        options,
        beam=beam,
        gamma=gamma,
    )[0]


@torch.no_grad()
def _score_candidates_and_t1_from_prefill(
    teacher: QwenTeacher,
    tokenizer: Any,
    prefill: _PrefillBatch,
    gold_emit_ids_list: Sequence[Sequence[int]],
    candidate_emit_ids_list: Sequence[Sequence[Sequence[int]]],
    *,
    top_k_t1: int = 32,
    score_batch_size: int = 64,
) -> list[tuple[list[list[int]], list[list[float]], list[float]]]:
    """Compute T1 and T2 across a batch of records reusing *prefill* without re-encoding prompts.

    Deduplicates identical ``(record, candidate)`` sequences (so gold appearing inside the candidate
    list is scored once) and groups unique continuations by length into chunks of
    *score_batch_size* with vectorized score extraction.
    """
    n_records = len(gold_emit_ids_list)
    if n_records == 0:
        return []
    device = teacher.emit_head.weight.device
    vocab = teacher.vocab
    eos_emit = vocab.eos_emit_id
    k_t1 = min(top_k_t1, vocab.num_classes)
    pad_tok = int(
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    # Position 0 top-k T1 for every record directly from the shared prefill.
    vals0, idxs0 = prefill.first_log_probs.topk(k_t1, dim=-1)
    vals0_list = vals0.cpu().tolist()
    idxs0_list = idxs0.cpu().tolist()
    first_lps_cpu = prefill.first_log_probs.cpu()

    t1_ids_out: list[list[list[int]]] = [
        [[int(x) for x in idxs0_list[r]]] for r in range(n_records)
    ]
    t1_lps_out: list[list[list[float]]] = [
        [[float(x) for x in vals0_list[r]]] for r in range(n_records)
    ]

    # Deduplicate (rec_idx, seq_tuple) across gold + candidates while flagging which entry is gold.
    unique_entries: dict[tuple[int, tuple[int, ...]], bool] = {}
    empty_scores: dict[int, float] = {
        r: float(first_lps_cpu[r, eos_emit].item()) for r in range(n_records)
    }
    for r in range(n_records):
        gold_t = tuple(int(e) for e in gold_emit_ids_list[r])
        if gold_t:
            unique_entries[(r, gold_t)] = True
        for cand in candidate_emit_ids_list[r]:
            cand_t = tuple(int(e) for e in cand)
            if cand_t and (r, cand_t) not in unique_entries:
                unique_entries[(r, cand_t)] = False

    # Sort unique continuations by length so right-padding inside each chunk is minimal.
    work = sorted(unique_entries.items(), key=lambda item: len(item[0][1]))
    seq_scores: dict[tuple[int, tuple[int, ...]], float] = {}
    gold_t1_rest: dict[int, tuple[list[list[int]], list[list[float]]]] = {}

    chunk_size = max(1, score_batch_size)
    for start in range(0, len(work), chunk_size):
        chunk = work[start : start + chunk_size]
        batch_n = len(chunk)
        max_len = max(len(seq) for (_, seq), _ in chunk)
        owners = [rec_idx for (rec_idx, _), _ in chunk]
        owner_t = torch.tensor(owners, dtype=torch.long, device=device)

        seq_tensor = torch.full((batch_n, max_len), pad_tok, dtype=torch.long)
        cand_mask = torch.zeros((batch_n, max_len), dtype=torch.long)
        pos_ids = torch.zeros((batch_n, max_len), dtype=torch.long)
        for row_i, ((rec_idx, seq), _) in enumerate(chunk):
            ln = len(seq)
            seq_tensor[row_i, :ln] = torch.tensor(
                [vocab.emit_to_token[e] for e in seq], dtype=torch.long
            )
            cand_mask[row_i, :ln] = 1
            base_pos = prefill.prompt_lens[rec_idx]
            pos_ids[row_i] = torch.arange(base_pos, base_pos + max_len, dtype=torch.long)

        seq_tensor = seq_tensor.to(device)
        cand_mask = cand_mask.to(device)
        pos_ids = pos_ids.to(device)
        attn_mask = torch.cat([prefill.attn_mask.index_select(0, owner_t), cand_mask], dim=1)
        cache_pos = torch.arange(
            prefill.max_prompt_len,
            prefill.max_prompt_len + max_len,
            dtype=torch.long,
            device=device,
        )
        expanded_cache = _select_cache_rows(prefill.prefix_cache, owners)
        out = teacher.backbone(
            inputs_embeds=teacher.embed_ids(seq_tensor),
            attention_mask=attn_mask,
            position_ids=pos_ids,
            cache_position=cache_pos,
            past_key_values=expanded_cache,
            use_cache=False,
        )
        logits = teacher.project_logits(out.last_hidden_state)
        lps = functional.log_softmax(logits, dim=-1)

        # Gather all target token log-probs in one vectorized op instead of per-token GPU syncs.
        row_indices: list[int] = []
        pos_indices: list[int] = []
        tgt_indices: list[int] = []
        lengths: list[int] = []
        gold_rows: list[tuple[int, int, int]] = []
        for row_i, ((rec_idx, seq), is_gold) in enumerate(chunk):
            targets = [*seq[1:], eos_emit]
            ln = len(targets)
            lengths.append(ln)
            for pos, tgt in enumerate(targets):
                row_indices.append(row_i)
                pos_indices.append(pos)
                tgt_indices.append(tgt)
            if is_gold:
                gold_rows.append((row_i, rec_idx, ln))

        gathered = (
            lps[
                torch.tensor(row_indices, dtype=torch.long, device=device),
                torch.tensor(pos_indices, dtype=torch.long, device=device),
                torch.tensor(tgt_indices, dtype=torch.long, device=device),
            ]
            .cpu()
            .tolist()
        )
        cursor = 0
        for ((rec_idx, seq), _), ln in zip(chunk, lengths, strict=True):
            total = float(first_lps_cpu[rec_idx, seq[0]].item()) + sum(
                gathered[cursor : cursor + ln]
            )
            cursor += ln
            seq_scores[(rec_idx, seq)] = total

        if gold_rows:
            g_row_t = torch.tensor([r for r, _, _ in gold_rows], dtype=torch.long, device=device)
            g_vals, g_idxs = lps.index_select(0, g_row_t).topk(k_t1, dim=-1)
            g_vals_list = g_vals.cpu().tolist()
            g_idxs_list = g_idxs.cpu().tolist()
            for idx_in_g, (_row_i, rec_idx, ln) in enumerate(gold_rows):
                gold_t1_rest[rec_idx] = (
                    [[int(x) for x in g_idxs_list[idx_in_g][p]] for p in range(ln)],
                    [[float(x) for x in g_vals_list[idx_in_g][p]] for p in range(ln)],
                )

    results: list[tuple[list[list[int]], list[list[float]], list[float]]] = []
    for r in range(n_records):
        if r in gold_t1_rest:
            rest_ids, rest_lps = gold_t1_rest[r]
            t1_ids_out[r].extend(rest_ids)
            t1_lps_out[r].extend(rest_lps)
        scores = [
            empty_scores[r] if len(cand) == 0 else seq_scores[(r, tuple(int(e) for e in cand))]
            for cand in candidate_emit_ids_list[r]
        ]
        results.append((t1_ids_out[r], t1_lps_out[r], scores))
    return results


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
    prefill = _prefill_prompts(teacher, tokenizer, [(context, keys)])
    return _score_candidates_and_t1_from_prefill(
        teacher,
        tokenizer,
        prefill,
        [gold_emit_ids],
        [candidate_emit_ids],
        top_k_t1=top_k_t1,
        score_batch_size=max(1, len(candidate_emit_ids) + 1),
    )[0]


@torch.no_grad()
def label_records_batch(
    teacher: QwenTeacher,
    tokenizer: Any,
    records: Sequence[tuple[str, str | None, str, str, list[int]]],
    legality: Legality,
    options: LegalityOptions = OFFLINE,
    *,
    beam: int = 8,
    gamma: float = 1.0,
    top_k_t1: int = 32,
    score_batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Prefill once, beam-decode, and compute T1/T2/T3 labels for a batch of records.

    Each item of *records* is ``(record_id, context, keys, gold_text, gold_emit_ids)``.
    """
    if not records:
        return []
    prefill = _prefill_prompts(teacher, tokenizer, [(ctx, keys) for _, ctx, keys, _, _ in records])
    hyps_batch = _decode_from_prefill(
        teacher,
        prefill,
        [keys for _, _, keys, _, _ in records],
        legality,
        options,
        beam=beam,
        gamma=gamma,
    )
    cand_texts_batch: list[list[str]] = []
    cand_ids_batch: list[list[list[int]]] = []
    for (_rec_id, _ctx, _keys, text, gold_ids), hyps in zip(records, hyps_batch, strict=True):
        cand_texts: list[str] = [text]
        cand_ids: list[list[int]] = [gold_ids]
        for h in hyps:
            if h.text != text:
                cand_texts.append(h.text)
                cand_ids.append(list(h.emit_ids))
        cand_texts_batch.append(cand_texts)
        cand_ids_batch.append(cand_ids)

    scored_batch = _score_candidates_and_t1_from_prefill(
        teacher,
        tokenizer,
        prefill,
        [gold_ids for _, _, _, _, gold_ids in records],
        cand_ids_batch,
        top_k_t1=top_k_t1,
        score_batch_size=score_batch_size,
    )
    rows_out: list[dict[str, Any]] = []
    for (rec_id, ctx, keys, text, _gold_ids), cand_texts, (t1_ids, t1_lps, cand_scores) in zip(
        records, cand_texts_batch, scored_batch, strict=True
    ):
        rows_out.append(
            {
                "id": rec_id,
                "context": ctx,
                "keys": keys,
                "text": text,
                "t1_ids": t1_ids,
                "t1_log_probs": t1_lps,
                "candidates": cand_texts,
                "candidate_scores": cand_scores,
            }
        )
    return rows_out


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


def _labels_parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            ("id", pa.string()),
            ("context", pa.string()),
            ("keys", pa.string()),
            ("text", pa.string()),
            ("t1_ids", pa.list_(pa.list_(pa.int32()))),
            ("t1_log_probs", pa.list_(pa.list_(pa.float32()))),
            ("candidates", pa.list_(pa.string())),
            ("candidate_scores", pa.list_(pa.float32())),
        ]
    )


def write_labels_parquet(rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    """Write offline T1/T2/T3 teacher labels for one shard atomically to Parquet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    normalised = [{**row, "context": row.get("context")} for row in rows]
    table = pa.Table.from_pylist(normalised, schema=_labels_parquet_schema())
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


def bucketed_microbatches(
    stream: Iterable[tuple[str | None, str, str]],
    tokenizer: Any,
    char_to_emit: Mapping[str, int],
    *,
    batch_size: int,
    pool_batches: int = 16,
    seed: int = 0,
) -> Iterable[list[tuple[str | None, str, str]]]:
    """Group *stream* into length-bucketed microbatches to eliminate padding waste (#336).

    Collects ``pool_batches * batch_size`` valid examples, sorts them by exact
    encoded length ``len(prompt) + len(text)``, slices into tight microbatches
    of *batch_size*, and yields those microbatches in shuffled order so gradient
    accumulation mixes short and long sentences unbiasedly.
    """
    pool_size = max(1, pool_batches) * batch_size
    rng = random.Random(seed + 3_000_003)
    iterator = iter(stream)
    while True:
        pool: list[tuple[int, tuple[str | None, str, str]]] = []
        while len(pool) < pool_size:
            try:
                ctx, keys, text = next(iterator)
            except StopIteration:
                break
            if all(ch in char_to_emit for ch in text):
                seq_len = len(encode_prompt(tokenizer, ctx, keys)) + len(text)
                pool.append((seq_len, (ctx, keys, text)))
        if not pool:
            break
        pool.sort(key=lambda pair: pair[0])
        batches = [
            [item for _, item in pool[i : i + batch_size]] for i in range(0, len(pool), batch_size)
        ]
        rng.shuffle(batches)
        yield from batches


def train_teacher_loop(
    teacher: QwenTeacher,
    tokenizer: Any,
    stream: Iterable[tuple[str | None, str, str]],
    out_dir: Path,
    config: TeacherTrainConfig,
    *,
    resume_path: Path | None = None,
    init_from: Path | None = None,
) -> dict[str, Any]:
    """Resumable LoRA training loop over ``(context, keys, target_text)`` triples.

    *resume_path* restores adapter weights, AdamW state, and step counter to
    continue an interrupted run. *init_from* loads only adapter weights from a
    finished parent stage (such as S3 -> S4) and starts a fresh LR schedule at
    step 0.
    """
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
    elif init_from is not None and init_from.is_file():
        ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        teacher.load_adapter_state_dict(ckpt.get("adapter", ckpt))
        log.info("teacher initialised from parent", path=str(init_from))

    warmup = max(1, int(config.max_steps * config.warmup_fraction))

    def lr_for(step: int) -> float:
        if step <= warmup:
            return config.lr * (step / warmup)
        progress = (step - warmup) / max(1, config.max_steps - warmup)
        return config.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    teacher.train()
    micro_iter = iter(
        bucketed_microbatches(
            stream,
            tokenizer,
            char_to_emit,
            batch_size=config.batch_size,
            seed=config.seed,
        )
    )
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
        step_real_in = 0
        step_padded_in = 0
        for _micro in range(config.accumulate):
            try:
                batch_items = next(micro_iter)
            except StopIteration:
                break
            input_ids, attn_mask, labels = build_training_batch(
                tokenizer,
                teacher.vocab,
                batch_items,
                char_to_emit=char_to_emit,
                pad_token_id=pad_id,
                device=device,
            )
            step_real_in += int(attn_mask.sum().item())
            step_padded_in += int(attn_mask.numel())
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
            pad_eff = round(step_real_in / max(1, step_padded_in), 3)
            record = {
                "event": "step",
                "step": step,
                "loss": step_loss,
                "lr": cur_lr,
                "examples": step_examples,
                "tokens": step_tokens,
                "pad_eff": pad_eff,
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
                pad_eff=pad_eff,
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
    shuffle_buffer: int = 4096,
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
        candidates = CandidateSpace(spans, lexicon.candidate_mask, table, lexicon.homophones)
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
    shuf_rng = random.Random(seed + 2_000_003)
    buf: list[tuple[str | None, str, str]] = []
    seen = 0
    epoch = 0
    while True:
        stream.set_epoch(epoch)
        for ex in stream:
            drop_ctx = ex.context is None or rng.random() < context_dropout
            ctx = None if drop_ctx else ex.context
            keys = "".join(ex.spans)
            text = "".join(lexicon.characters[t] for t in ex.targets)
            item = (ctx, keys, text)
            if shuffle_buffer <= 1:
                seen += 1
                if seen > skip_examples:
                    yield item
            elif len(buf) < shuffle_buffer:
                buf.append(item)
            else:
                pick = shuf_rng.randrange(len(buf))
                out_item = buf[pick]
                buf[pick] = item
                seen += 1
                if seen > skip_examples:
                    yield out_item
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
    batch_records: int = 16,
) -> tuple[Any, list[dict[str, Any]]]:
    """Evaluate *teacher* with batched lattice-constrained beam search on *eval_set*.

    Pending records are length-bucketed into batches of *batch_records* so left-padding
    and decode step stragglers stay minimal. Resumable when *dump_path* is given:
    previously written rows are reloaded and skipped.
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
    t0 = time.monotonic()

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)

    pending = [idx for idx in indices if idx not in existing]
    pending.sort(
        key=lambda idx: (
            len(
                encode_prompt(
                    tokenizer, rows[idx].context if with_context else None, rows[idx].pinyin
                )
            )
            + len(rows[idx].pinyin)
        )
    )

    step_size = max(1, batch_records)
    decoded_new = 0
    for start in range(0, len(pending), step_size):
        batch_indices = pending[start : start + step_size]
        batch_inputs = [
            (rows[idx].context if with_context else None, rows[idx].pinyin) for idx in batch_indices
        ]
        hyps_batch = constrained_beam_decode_batch(
            teacher,
            tokenizer,
            batch_inputs,
            legality,
            options,
            beam=beam,
            gamma=gamma,
        )
        lines_to_append: list[str] = []
        for idx, hyps in zip(batch_indices, hyps_batch, strict=True):
            row = rows[idx]
            rec = {
                "index": idx,
                "pinyin": row.pinyin,
                "expected": row.text,
                "hypotheses": [h.text for h in hyps],
                "scores": [h.score for h in hyps],
            }
            existing[idx] = rec
            if dump_path is not None:
                lines_to_append.append(json.dumps(rec, ensure_ascii=False) + "\n")
        if dump_path is not None and lines_to_append:
            with dump_path.open("a", encoding="utf-8") as df:
                df.writelines(lines_to_append)
        decoded_new += len(batch_indices)
        done_total = len(indices) - (len(pending) - decoded_new)
        if decoded_new == len(pending) or (start // step_size + 1) % max(1, 50 // step_size) == 0:
            elapsed = time.monotonic() - t0
            hits = sum(
                1
                for i in indices
                if i in existing
                and existing[i]["hypotheses"]
                and existing[i]["hypotheses"][0] == rows[i].text
            )
            log.info(
                "teacher eval progress",
                done=done_total,
                total=len(indices),
                top1=round(hits / max(1, done_total), 4),
                sec_per_rec=round(elapsed / max(1, decoded_new), 3),
                seconds=round(elapsed, 1),
            )

    report = Report(top_k=beam)
    results: list[dict[str, Any]] = []
    for idx in indices:
        rec = existing[idx]
        report.observe(rows[idx].text, rec["hypotheses"])
        results.append(rec)
    return report, results


def _merge_label_parts(
    part_paths: Sequence[Path],
    tail_rows: Sequence[Mapping[str, Any]],
    out_path: Path,
) -> None:
    """Atomically combine partial shard Parquet files and *tail_rows* into *out_path*."""
    tables = [pq.read_table(p) for p in part_paths]
    if tail_rows:
        normalised = [{**row, "context": row.get("context")} for row in tail_rows]
        tables.append(pa.Table.from_pylist(normalised, schema=_labels_parquet_schema()))
    merged = (
        pa.concat_tables(tables)
        if tables
        else pa.Table.from_pylist([], schema=_labels_parquet_schema())
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(merged, tmp, compression="zstd")
    tmp.replace(out_path)
    for part in part_paths:
        part.unlink(missing_ok=True)


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
    context_dropout: float = 0.25,
    seed: int = 0,
    beam: int = 8,
    gamma: float = 1.0,
    top_k_t1: int = 32,
    max_per_shard: int | None = None,
    batch_records: int = 16,
    score_batch_size: int = 64,
    checkpoint_every_rows: int = 1000,
) -> dict[str, Any]:
    """Generate resumable offline T1/T2/T3 Parquet labels shard by shard (S5).

    Supports both shard-level and mid-shard resumability via incremental
    ``.part-*.parquet`` checkpoints flushed every *checkpoint_every_rows* rows.
    """
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
        candidates = CandidateSpace(spans, lexicon.candidate_mask, table, lexicon.homophones)
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
    step_size = max(1, batch_records)
    flush_rows = max(1, checkpoint_every_rows)

    for shard_idx, shard_path in enumerate(selected):
        out_pq = out_dir / shard_path.name
        if out_pq.is_file():
            skipped_shards += 1
            log.info("teacher label shard skipped (exists)", shard=shard_path.name)
            continue

        part_prefix = f".{shard_path.stem}.part-"
        existing_parts = sorted(out_dir.glob(f"{part_prefix}*.parquet"))
        done_ids: set[str] = set()
        for part in existing_parts:
            col = pq.read_table(part, columns=["id"]).column("id").to_pylist()
            done_ids.update(str(x) for x in col)
        if done_ids:
            log.info(
                "teacher label shard resumed",
                shard=shard_path.name,
                parts=len(existing_parts),
                done=len(done_ids),
            )

        # Deterministic per-shard context dropout so mid-shard resume reproduces exact inputs.
        ctx_rng = random.Random(seed + 4_000_003 + shard_idx * 9973)
        work_items: list[tuple[int, str, str | None, str, str, list[int]]] = []
        target_cap = max_per_shard
        eligible_seen = 0
        for sample, syls in read_pair(shard_path, labels_dir):
            ex = builder.build(sample, syls, epoch=0)
            if ex is None:
                continue
            keys = "".join(ex.spans)
            text = "".join(lexicon.characters[t] for t in ex.targets)
            if not all(ch in char_to_emit for ch in text):
                continue
            drop_ctx = ex.context is None or ctx_rng.random() < context_dropout
            ctx = None if drop_ctx else ex.context
            eligible_seen += 1
            if ex.id not in done_ids:
                gold_ids = [char_to_emit[ch] for ch in text]
                sort_len = len(encode_prompt(tokenizer, ctx, keys)) + len(keys)
                work_items.append((sort_len, ex.id, ctx, keys, text, gold_ids))
            if target_cap is not None and eligible_seen >= target_cap:
                break

        work_items.sort(key=lambda item: item[0])
        part_paths = list(existing_parts)
        next_part_idx = len(part_paths)
        part_rows: list[dict[str, Any]] = []
        labelled_new = 0
        t0 = time.monotonic()

        for start in range(0, len(work_items), step_size):
            batch = [
                (rec_id, ctx, keys, text, gold_ids)
                for _, rec_id, ctx, keys, text, gold_ids in work_items[start : start + step_size]
            ]
            rows = label_records_batch(
                teacher,
                tokenizer,
                batch,
                legality,
                OFFLINE,
                beam=beam,
                gamma=gamma,
                top_k_t1=top_k_t1,
                score_batch_size=score_batch_size,
            )
            part_rows.extend(rows)
            labelled_new += len(rows)
            if len(part_rows) >= flush_rows:
                part_path = out_dir / f"{part_prefix}{next_part_idx:05d}.parquet"
                write_labels_parquet(part_rows, part_path)
                part_paths.append(part_path)
                next_part_idx += 1
                part_rows.clear()
                elapsed = time.monotonic() - t0
                log.info(
                    "teacher label shard checkpoint",
                    shard=shard_path.name,
                    done=len(done_ids) + labelled_new,
                    total=len(done_ids) + len(work_items),
                    sec_per_rec=round(elapsed / max(1, labelled_new), 3),
                )

        if part_paths:
            _merge_label_parts(part_paths, part_rows, out_pq)
        else:
            write_labels_parquet(part_rows, out_pq)
        shard_total = len(done_ids) + labelled_new
        written_shards += 1
        total_rows += shard_total
        elapsed = time.monotonic() - t0
        log.info(
            "teacher label shard written",
            shard=shard_path.name,
            rows=shard_total,
            sec_per_rec=round(elapsed / max(1, labelled_new), 3),
        )

    return {
        "written_shards": written_shards,
        "skipped_shards": skipped_shards,
        "total_rows": total_rows,
    }
