"""Route A as ONNX for the Rust side: the context tower once, the fill tower per record.

Two graphs, because their cadences differ. `context.onnx` encodes what was on
screen before the keystrokes -- it takes the context's ids and mask and returns
the tower's last hidden states. `fill.onnx` scores one record: every reading of
the keystrokes at once, each row a `[CLS] [MASK]* [SEP]` sentence carrying the
span that was typed at each mask position, plus the context tower's output (or
zeros) and a per-row `has_context` switch for the gated cross-attention. It
returns `log_probs` -- the restricted head's logits masked per position to the
characters the typed span admits and normalised by log-softmax, so the number
the score file holds is read straight off the graph's output.

Both forwards are spelled out in primitives -- matmul, reshape, softmax --
rather than traced through the transformers modules. The spellings are the same
operations the modules perform, written so that every sequence axis and the
batch axis stay dynamic and `has_context` stays a runtime tensor.

One weights file per graph: the towers share no parameters, so
`context.weights` and `fill.weights` are independent. The manifest names both
graphs, both weight tables, the candidate mask inside the fill weights, the
span and character vocabularies the indices refer to, and the sentinels the
inputs are built from. Its `layout` is the pin the loader checks before opening
either graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer

from mlime.logging import log
from mlime.train.charlm import _externalize, _quantize_dynamic_int8, _weights_table
from mlime.train.lexicon import Lexicon
from mlime.train.model import GatedCrossAttention, RouteAModel
from mlime.train.samples import DEFAULT_CONTEXT_TOKENS, BaseTokenizer
from mlime.train.spans import SpanVocab

#: The layout string `route-a.json` carries and `ime-neural` requires.
LAYOUT = "towers"

#: The lowest finite fp32, standing in for -inf where a probability is wanted.
FLOOR = torch.finfo(torch.float32).min


def _additive_mask(mask: torch.Tensor) -> torch.Tensor:
    """An additive ``[rows, 1, 1, length]`` attention mask from a padding mask."""
    masked: torch.Tensor = (1.0 - mask.to(torch.float32)) * FLOOR
    return masked[:, None, None, :]


def _encoder_layer(layer: BertLayer, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """One BERT encoder layer over *hidden* under the additive *mask*."""
    attention = layer.attention.self
    heads, head_dim = attention.num_attention_heads, attention.attention_head_size
    width = hidden.shape[1]
    query: torch.Tensor = (
        attention.query(hidden).reshape(-1, width, heads, head_dim).transpose(1, 2)
    )
    key: torch.Tensor = attention.key(hidden).reshape(-1, width, heads, head_dim).transpose(1, 2)
    value: torch.Tensor = (
        attention.value(hidden).reshape(-1, width, heads, head_dim).transpose(1, 2)
    )
    scores: torch.Tensor = (query / head_dim**0.5) @ key.transpose(-1, -2) + mask
    attended: torch.Tensor = (
        (scores.softmax(dim=-1) @ value).transpose(1, 2).reshape(-1, width, heads * head_dim)
    )
    hidden = layer.attention.output.LayerNorm(layer.attention.output.dense(attended) + hidden)
    feedforward: torch.Tensor = layer.output.dense(layer.intermediate(hidden))
    output: torch.Tensor = layer.output.LayerNorm(feedforward + hidden)
    return output


def _embeddings(embeddings: BertEmbeddings, embeds: torch.Tensor) -> torch.Tensor:
    """Position and token-type embeddings plus the input's own, layer-normed."""
    position = embeddings.position_embeddings.weight[: embeds.shape[1]]
    token_type = embeddings.token_type_embeddings.weight[0]
    normed: torch.Tensor = embeddings.LayerNorm(embeds + position[None] + token_type[None, None])
    return normed


def _cross_attention(
    layer: GatedCrossAttention,
    hidden: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    has_context: torch.Tensor,
) -> torch.Tensor:
    """The gated multi-head attention of a fill row into the context tower's states."""
    attention = layer.attention
    heads, head_dim = attention.num_heads, attention.head_dim
    channels = attention.embed_dim
    query: torch.Tensor = nn.functional.linear(
        layer.norm(hidden), attention.in_proj_weight[:channels], attention.in_proj_bias[:channels]
    )
    key: torch.Tensor = nn.functional.linear(
        context,
        attention.in_proj_weight[channels : 2 * channels],
        attention.in_proj_bias[channels : 2 * channels],
    )
    value: torch.Tensor = nn.functional.linear(
        context, attention.in_proj_weight[2 * channels :], attention.in_proj_bias[2 * channels :]
    )
    length = context.shape[1]
    query = query.reshape(-1, query.shape[1], heads, head_dim).transpose(1, 2)
    key = key.reshape(-1, length, heads, head_dim).transpose(1, 2)
    value = value.reshape(-1, length, heads, head_dim).transpose(1, 2)
    scores: torch.Tensor = (query / head_dim**0.5) @ key.transpose(-1, -2)
    scores = scores.masked_fill((context_mask == 0)[:, None, None, :], FLOOR)
    attended: torch.Tensor = (
        (scores.softmax(dim=-1) @ value)
        .transpose(1, 2)
        .reshape(query.shape[0], query.shape[2], channels)
    )
    projected: torch.Tensor = attention.out_proj(attended)
    gated: torch.Tensor = hidden + layer.gate * has_context[:, None, None] * projected
    return gated


class _ContextGraph(nn.Module):
    """The context tower alone: ``(context_ids, context_mask) -> last_hidden_state``."""

    def __init__(self, model: RouteAModel):
        super().__init__()
        self.model = model

    def forward(self, context_ids: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        embeds: torch.Tensor = self.model.context.embeddings.word_embeddings(context_ids)
        hidden = _embeddings(self.model.context.embeddings, embeds)
        mask = _additive_mask(context_mask)
        for layer in self.model.context.encoder.layer:
            hidden = _encoder_layer(layer, hidden, mask)
        return hidden


class _FillGraph(nn.Module):
    """The fill tower's masked, normalised log probabilities for one record's readings."""

    def __init__(self, model: RouteAModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_ids: torch.Tensor,
        span_positions: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        has_context: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        words: torch.Tensor = model.fill.embeddings.word_embeddings(input_ids)
        embeds = words + model.span_embeddings(span_ids) * span_positions.to(words.dtype)[..., None]
        hidden = _embeddings(model.fill.embeddings, embeds)
        mask = _additive_mask(attention_mask)
        gated = model.gated_layers()
        gated_from = len(model.fill.encoder.layer) - len(gated)
        for depth, layer in enumerate(model.fill.encoder.layer):
            hidden = _encoder_layer(layer, hidden, mask)
            if depth >= gated_from:
                hidden = _cross_attention(
                    gated[depth - gated_from], hidden, context, context_mask, has_context
                )
        logits: torch.Tensor = model.head(hidden)
        masks: torch.Tensor = model.candidate_mask.index_select(0, span_ids.reshape(-1)).reshape(
            *span_ids.shape, -1
        )
        log_probs: torch.Tensor = logits.masked_fill(~masks, FLOOR).log_softmax(dim=-1)
        return log_probs


def export_onnx(
    model: RouteAModel,
    step: int,
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    out_dir: Path,
    quantize: Literal["int8"] | None = None,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> tuple[Path, Path]:
    """Write the two graphs, their weights files and `route-a.json` in *out_dir*.

    ``context.onnx`` takes ``context_ids [rows, length]`` and ``context_mask``
    and returns ``context [rows, length, hidden]``. ``fill.onnx`` takes one
    record's readings as ``input_ids``, ``attention_mask``, ``span_ids`` and
    ``span_positions`` -- all ``[paths, width]`` -- plus ``context`` and
    ``context_mask`` ``[paths, context_length]`` (the context tower's output
    repeated per path, or zeros) and ``has_context [paths]``, and returns
    ``log_probs [paths, width, emissions]``. *quantize* is the same dynamic
    per-channel int8 ``export char-lm`` offers; *context_tokens* is the bound
    the collator tokenises contexts to, recorded for the loader.

    The manifest's ``specials`` and its `tokenizer.json` come from *tokenizer*,
    which must be the base model's own.
    """
    if quantize not in (None, "int8"):
        raise ValueError(f"quantize must be 'int8' or None, got {quantize!r}")
    was_training = model.training
    model = model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    # torch.onnx.export leaves the traced module back in training mode; the
    # graphs are unaffected (these forwards call no dropout) but the caller's
    # model is restored to however it arrived.
    try:
        _export_graphs(model, tokenizer, out_dir, context_tokens)
    finally:
        model.train(was_training)
    context_graph = out_dir / "context.onnx"
    fill_graph = out_dir / "fill.onnx"
    if quantize == "int8":
        _quantize_dynamic_int8(context_graph)
        _quantize_dynamic_int8(fill_graph)
    context_model = _externalize(context_graph, "context.weights")
    fill_model = _externalize(fill_graph, "fill.weights")
    context_table = _weights_table(context_model)
    fill_table = _weights_table(fill_model)

    emissions = int(model.candidate_mask.shape[1])
    mask_names = [
        name
        for name, tensor in fill_table.items()
        if tensor["shape"] == [model.candidate_mask.shape[0], emissions]
        and tensor["dtype"] == "bool"
    ]
    if len(mask_names) != 1:
        raise ValueError(
            f"the fill graph's weights hold {len(mask_names)} candidate masks, not one"
        )
    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    save = getattr(tokenizer, "save", None)
    if save_pretrained is not None:
        save_pretrained(str(out_dir))
    elif save is not None:
        save(str(out_dir / "tokenizer.json"))
    else:
        raise ValueError("the tokenizer has neither save_pretrained nor save")
    if not (out_dir / "tokenizer.json").is_file():
        raise ValueError("the tokenizer did not write tokenizer.json into the export directory")
    hidden = int(model.fill.embeddings.word_embeddings.weight.shape[1])
    manifest = out_dir / "route-a.json"
    manifest.write_text(
        json.dumps(
            {
                "step": step,
                "base_model": model.config.base_model,
                "layout": LAYOUT,
                "dtype": "float32",
                "quantize": quantize,
                "hidden": hidden,
                "context_tokens": context_tokens,
                "specials": {
                    "pad": int(tokenizer.pad_token_id),
                    "cls": int(tokenizer.cls_token_id),
                    "sep": int(tokenizer.sep_token_id),
                    "mask": int(tokenizer.mask_token_id),
                },
                "spans": list(spans),
                "characters": list(lexicon.characters),
                "graphs": {"context": "context.onnx", "fill": "fill.onnx"},
                "candidate_mask": mask_names[0],
                "weights": {
                    "context": {"file": "context.weights", "tensors": list(context_table.values())},
                    "fill": {"file": "fill.weights", "tensors": list(fill_table.values())},
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info(
        "exported route A",
        context_graph=str(context_graph),
        fill_graph=str(fill_graph),
        manifest=str(manifest),
        spans=model.candidate_mask.shape[0],
        emissions=emissions,
    )
    return fill_graph, manifest


def _export_graphs(
    model: RouteAModel, tokenizer: BaseTokenizer, out_dir: Path, context_tokens: int
) -> None:
    """Trace `context.onnx` and `fill.onnx` into *out_dir*."""
    context_ids = torch.tensor(
        [[tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id]],
        dtype=torch.long,
    )
    context_mask = torch.ones(1, context_ids.shape[1], dtype=torch.long)
    torch.onnx.export(
        _ContextGraph(model),
        (context_ids, context_mask),
        str(out_dir / "context.onnx"),
        input_names=["context_ids", "context_mask"],
        output_names=["context"],
        dynamic_axes={
            "context_ids": {0: "batch", 1: "length"},
            "context_mask": {0: "batch", 1: "length"},
            "context": {0: "batch", 1: "length"},
        },
        opset_version=17,
        dynamo=False,
    )

    paths, width = 2, 5
    input_ids = torch.zeros(paths, width, dtype=torch.long)
    attention_mask = torch.ones(paths, width, dtype=torch.long)
    span_ids = torch.zeros(paths, width, dtype=torch.long)
    span_positions = torch.zeros(paths, width, dtype=torch.bool)
    hidden = int(model.fill.embeddings.word_embeddings.weight.shape[1])
    context = torch.zeros(paths, context_ids.shape[1], hidden)
    fill_context_mask = torch.ones(paths, context_ids.shape[1], dtype=torch.long)
    has_context = torch.ones(paths)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            _FillGraph(model),
            (
                input_ids,
                attention_mask,
                span_ids,
                span_positions,
                context,
                fill_context_mask,
                has_context,
            ),
            str(out_dir / "fill.onnx"),
            input_names=[
                "input_ids",
                "attention_mask",
                "span_ids",
                "span_positions",
                "context",
                "context_mask",
                "has_context",
            ],
            output_names=["log_probs"],
            dynamic_axes={
                "input_ids": {0: "paths", 1: "width"},
                "attention_mask": {0: "paths", 1: "width"},
                "span_ids": {0: "paths", 1: "width"},
                "span_positions": {0: "paths", 1: "width"},
                "context": {0: "paths", 1: "context"},
                "context_mask": {0: "paths", 1: "context"},
                "has_context": {0: "paths"},
                "log_probs": {0: "paths", 1: "width"},
            },
            opset_version=17,
            dynamo=False,
        )
