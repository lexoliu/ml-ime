"""The key-level model as ONNX for the Rust side: context, keys and decoder-step graphs (#266).

A keystroke costs three graphs, all on the CPU, because their cadences differ.

``context.onnx`` encodes the text on screen once per context: route A's context
graph (:func:`~mlime.train.routea.export_context_graph`), unchanged.

``keys.onnx`` runs once per keystroke over one row. It is
:meth:`~mlime.train.key_model.KeyE2EModel.encode_inputs`, the CTC head and the
decoder's cross-attention key and value projections folded into one graph: the
fill tower reads ``[CLS] [MASK]*n [SEP]`` plus the key front, its top layers
reading the context, and the graph returns each key's CTC log-probabilities
``[n, E + 1]`` (blank last) and the decoder's per-layer cross keys and values of
the key states ``[layers, heads, n, head_dim]``. The decoder step then never
re-projects them. The key front's causal attention is an explicit mask.

``decoder.onnx`` is the char LM's paged step (:class:`~mlime.train.charlm_model.
PagedStepModule`) with the two changes the key model makes: one gated
cross-attention after each block, reading the keys graph's cross keys and
values, broadcast over the rows; and an emission head over the whole emittable
set, log-softmaxed as training normalises it. Its pages are the char LM's --
``pool_keys`` and ``pool_values`` ``[pages, layers, heads, head_dim]``, page 0 a
masked scratch, ``page_row [rows, T]`` int64 zero-padded on the right,
``page_mask [rows, T]`` bool -- with these differences, all of them forced by
the model:

* **No prelude.** The decoder reads ``<bos>`` then the characters; the char LM
  reads ``<bos> context <sep>`` then them. A row's position is therefore its
  prefix length, passed as ``position [rows]``; the char LM derives it from the
  mask because its history is not the whole of the positions before it.
* **A tree mask.** ``tree_mask [rows, rows]`` bool says which rows of the call
  a row reads the fresh key and value of (the diagonal is the row itself and is
  always set), so one call can carry a chain of prefixes whose ancestors have
  no page yet. The char LM's graph reads only its pages and itself.
* **Tokens are emission indices.** ``token [rows]`` is the emission index, with
  ``E`` meaning ``<bos>``, mapped in the graph through a constant table to the
  reader's ids; the char LM's takes reader ids.
* **No candidate gather and no scratch rows.** The graph returns the whole
  ``log_probs [rows, E]``; the char LM gathers ``candidates`` and has
  ``state_keys``/``state_values``/``source_row`` for rows whose page claim
  failed. A page claim that fails has no counterpart here: the pool holds one
  keystroke.
* **Cross inputs.** ``cross_keys`` and ``cross_values`` are inputs of the step.

Quantisation follows the graphs' kinds: the context and keys graphs are dynamic
per-channel int8 as route A's towers are; the decoder is weight-only int8 as
the char LM's is (``decoder_quantize="weights"``), or dynamic int8
(``"dynamic"``) so the two can be measured against each other.

The export checks itself at float32 before it quantises: the graphs are run with
onnxruntime over three probes, and the keys graph's CTC log-probabilities and
cross keys and values, and the decoder's log-probabilities over chains and
pooled prefixes, must match the PyTorch model's within :data:`CLOSE`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from torch import nn

from mlime.legality import Legality
from mlime.logging import log
from mlime.train.charlm_vocab import BOS
from mlime.train.e2e_model import Decoder
from mlime.train.key_eval import build_key_collator
from mlime.train.key_model import KeyBlock, KeyE2EModel
from mlime.train.key_onnx import (
    CONTEXT_GRAPH,
    DECODER_GRAPH,
    KEYS_GRAPH,
    LAYOUT,
    MANIFEST,
    TOKENIZER,
    OnnxKeyModel,
    OnnxTransition,
    legality_probes,
)
from mlime.train.key_search import Prefix
from mlime.train.keys import KeyCollator
from mlime.train.lexicon import Lexicon
from mlime.train.model import BertContextTower
from mlime.train.onnx_tools import (
    externalize,
    quantize_dynamic_int8,
    quantize_weights_int8,
    weights_table,
    write_tokenizer,
)
from mlime.train.routea import FLOOR, export_context_graph, fill_tower
from mlime.train.samples import DEFAULT_CONTEXT_TOKENS, LETTER_PAD, BaseTokenizer
from mlime.train.spans import SpanVocab

#: How far the exported graphs may stray from the PyTorch model at float32.
CLOSE = 1e-4

#: The key strings (and contexts) the self-check runs: a full sentence with
#: its context, an initials string with none, and a mixed one with a context.
VERIFY_PROBES: tuple[tuple[str, str | None], ...] = (
    ("wmzyq", "我们吃水饺"),
    ("zhongguo", None),
    ("shuijiap", "水饺好吃"),
)

#: What the manifest says of a graph's weights: its quantisation, or None for float32.
DYNAMIC_INT8 = "dynamic-int8"
WEIGHTS_INT8 = "weights-int8"

DecoderQuantize = Literal["weights", "dynamic"]

_GRAPHS = {"context": CONTEXT_GRAPH, "keys": KEYS_GRAPH, "decoder": DECODER_GRAPH}


def _key_block(block: KeyBlock, x: torch.Tensor) -> torch.Tensor:
    """One key-front block with its causal attention spelled out under an explicit mask.

    The arithmetic of :meth:`~mlime.train.key_model.KeyBlock.forward`, whose
    fused ``is_causal`` attention would not export with a dynamic length.
    """
    width = x.shape[1]
    heads = block.heads
    head_dim = x.shape[2] // heads
    q, k, v = (
        block.qkv(block.attention_norm(x))
        .reshape(-1, width, 3, heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    scores = (q / head_dim**0.5) @ k.transpose(-1, -2)
    ahead = torch.triu(torch.ones(width, width, dtype=torch.bool, device=x.device), diagonal=1)
    attended = scores.masked_fill(ahead, FLOOR).softmax(dim=-1) @ v
    x = x + block.out(attended.transpose(1, 2).reshape(-1, width, heads * head_dim))
    result: torch.Tensor = x + block.down(nn.functional.gelu(block.up(block.feedforward_norm(x))))
    return result


class _KeysGraph(nn.Module):
    """One keystroke: the fill tower over the keys, the CTC head, the cross keys and values.

    ``(input_ids, key_letters, attention_mask, context, context_mask,
    has_context) -> (ctc_log_probs, cross_keys, cross_values)`` over one row,
    ``[1, n + 2]`` wide: ``[CLS]``, ``[MASK]`` per key, ``[SEP]``.
    """

    letter_table: torch.Tensor

    def __init__(self, model: KeyE2EModel):
        super().__init__()
        self.model = model
        front = model.key_front
        # Each letter's embedding plus its QWERTY coordinates' projection is a
        # function of the letter alone: one row of a table, the padding row
        # included, so the graph gathers it instead of multiplying a
        # two-wide input.
        with torch.no_grad():
            table = front.letters.weight + front.place(front.coordinates)
        self.register_buffer("letter_table", table)

    def forward(
        self,
        input_ids: torch.Tensor,
        key_letters: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        has_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.model
        front = model.key_front
        words: torch.Tensor = model.fill.embeddings.word_embeddings(input_ids)
        at_key = (key_letters != LETTER_PAD).unsqueeze(-1)
        keyed = (
            self.letter_table[key_letters] + front.positions.weight[: key_letters.shape[1]][None]
        )
        for block in front.blocks:
            keyed = _key_block(cast(KeyBlock, block), keyed)
        hidden = fill_tower(
            model, words + keyed * at_key, attention_mask, context, context_mask, has_context
        )
        states = hidden[:, 1:-1]
        transformed = model.head.transform(states)
        logits = torch.cat([model.head.decoder(transformed), model.blank(transformed)], dim=-1)
        pairs = [layer.kv(states) for layer in model.decoder.gated_layers()]
        return (
            logits.float().log_softmax(dim=-1)[0],
            torch.stack([keys[0] for keys, _ in pairs]),
            torch.stack([values[0] for _, values in pairs]),
        )


class _DecoderStep(nn.Module):
    """One step of the decoder over rows of prefixes, their histories paged.

    ``(token, position, page_row, page_mask, tree_mask, pool_keys,
    pool_values, cross_keys, cross_values) -> (log_probs, next_keys,
    next_values)``. Row ``r`` is a prefix: the decoder's token at its
    ``position``, attending its history's pages (``page_row[r]`` where
    ``page_mask[r]`` is set), the fresh positions of the rows ``tree_mask[r]``
    names -- itself always -- and the key states' cross keys and values.
    """

    reader_ids: torch.Tensor
    emitted: torch.Tensor

    def __init__(self, model: KeyE2EModel):
        super().__init__()
        decoder = model.decoder
        if decoder.span_input is not None:
            raise ValueError("the decoder has a span input; only a joint-stage model exports")
        self.decoder = decoder
        emit = decoder.emit_ids
        self.register_buffer("reader_ids", torch.cat([emit, torch.tensor([BOS], dtype=emit.dtype)]))
        # The tied head's rows at the emittable characters, laid out [embedding, E]
        # so the graph's projection is a MatMul against a constant.
        with torch.no_grad():
            head = decoder.reader.embed.weight.index_select(0, emit).T.contiguous()
        self.register_buffer("emitted", head)

    def forward(
        self,
        token: torch.Tensor,
        position: torch.Tensor,
        page_row: torch.Tensor,
        page_mask: torch.Tensor,
        tree_mask: torch.Tensor,
        pool_keys: torch.Tensor,
        pool_values: torch.Tensor,
        cross_keys: torch.Tensor,
        cross_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.decoder
        reader = decoder.reader
        rows, depth = page_row.shape[0], page_row.shape[1]
        ids = self.reader_ids[token]
        x = reader.input(reader.embed(ids.unsqueeze(1))) + reader.positions(position).unsqueeze(1)
        # One gather per tensor over the page axis, then the layer axis permuted
        # out: [rows, layers, heads, T, head_dim].
        pages = page_row.reshape(-1)
        paged_keys = (
            pool_keys.index_select(0, pages)
            .reshape(rows, depth, *pool_keys.shape[1:])
            .permute(0, 2, 3, 1, 4)
        )
        paged_values = (
            pool_values.index_select(0, pages)
            .reshape(rows, depth, *pool_values.shape[1:])
            .permute(0, 2, 3, 1, 4)
        )
        readable = torch.cat([page_mask, tree_mask], dim=-1).logical_not().view(rows, 1, 1, -1)
        new_keys, new_values = [], []
        for layer, (block, cross) in enumerate(
            zip(reader.layers(), decoder.gated_layers(), strict=True)
        ):
            q, k, v = block.split(x)
            # The history's scores per row, and the call's fresh positions' --
            # every row's key, shared by all rows -- side by side under one softmax.
            scores = torch.cat(
                [q @ paged_keys[:, layer].transpose(-1, -2), q @ k.permute(2, 1, 3, 0)], dim=-1
            )
            scores = scores.masked_fill(readable, -math.inf) / math.sqrt(block.head_dim)
            weights = scores.softmax(dim=-1)
            attended = weights[..., :depth] @ paged_values[:, layer] + (
                weights[..., depth:] @ v.permute(2, 1, 0, 3)
            )
            x = block.merge(x, attended)
            x = cross.attend_beams(
                x, cross_keys[layer].unsqueeze(0), cross_values[layer].unsqueeze(0), None
            )
            new_keys.append(k.squeeze(2))
            new_values.append(v.squeeze(2))
        features = reader.norm(x)
        logits = reader.project(features) @ self.emitted
        return (
            logits.float().log_softmax(dim=-1).squeeze(1),
            torch.stack(new_keys, dim=1),
            torch.stack(new_values, dim=1),
        )


def _example_context(model: KeyE2EModel) -> tuple[torch.Tensor, torch.Tensor]:
    """A three-token context of zeros, all real, for tracing."""
    hidden = int(model.fill.config.hidden_size)
    return torch.zeros(1, 3, hidden), torch.ones(1, 3, dtype=torch.long)


def _export_keys_graph(model: KeyE2EModel, tokenizer: BaseTokenizer, path: Path) -> None:
    """Trace `keys.onnx` into *path*; the key count and the context length are dynamic."""
    width = 5
    input_ids = torch.full((1, width), tokenizer.mask_token_id, dtype=torch.long)
    key_letters = torch.zeros(1, width, dtype=torch.long)
    key_letters[:, 0] = key_letters[:, -1] = LETTER_PAD
    context, context_mask = _example_context(model)
    with torch.no_grad():
        torch.onnx.export(
            # In evaluation mode itself: the exporter puts the wrapper back to
            # the mode it found it in, and a wrapper is born in training mode.
            _KeysGraph(model).eval(),
            (
                input_ids,
                key_letters,
                torch.ones(1, width, dtype=torch.long),
                context,
                context_mask,
                torch.ones(1),
            ),
            str(path),
            input_names=[
                "input_ids",
                "key_letters",
                "attention_mask",
                "context",
                "context_mask",
                "has_context",
            ],
            output_names=["ctc_log_probs", "cross_keys", "cross_values"],
            dynamic_axes={
                "input_ids": {1: "width"},
                "key_letters": {1: "width"},
                "attention_mask": {1: "width"},
                "context": {1: "context"},
                "context_mask": {1: "context"},
                "ctc_log_probs": {0: "keys"},
                "cross_keys": {2: "keys"},
                "cross_values": {2: "keys"},
            },
            opset_version=17,
            dynamo=False,
        )


def _export_decoder_graph(model: KeyE2EModel, path: Path) -> None:
    """Trace `decoder.onnx` into *path*; rows, history, pages and keys are all dynamic."""
    decoder: Decoder = model.decoder
    config = decoder.reader.config
    layers, heads = config.layers, config.heads
    head_dim = config.hidden // heads
    emittable = int(decoder.emit_ids.numel())
    # Distinct sizes -- three rows, two history pages, five pool pages, four
    # keys -- so no axis is mistaken for another when it is traced.
    rows, pool, keys = 3, 5, 4
    token = torch.tensor([emittable, 0, 1])
    position = torch.tensor([0, 1, 2])
    page_row = torch.tensor([[0, 0], [1, 0], [1, 2]])
    page_mask = torch.tensor([[False, False], [True, False], [True, True]])
    tree_mask = torch.eye(rows, dtype=torch.bool)
    with torch.no_grad():
        torch.onnx.export(
            _DecoderStep(model).eval(),
            (
                token,
                position,
                page_row,
                page_mask,
                tree_mask,
                torch.zeros(pool, layers, heads, head_dim),
                torch.zeros(pool, layers, heads, head_dim),
                torch.zeros(layers, heads, keys, head_dim),
                torch.zeros(layers, heads, keys, head_dim),
            ),
            str(path),
            input_names=[
                "token",
                "position",
                "page_row",
                "page_mask",
                "tree_mask",
                "pool_keys",
                "pool_values",
                "cross_keys",
                "cross_values",
            ],
            output_names=["log_probs", "next_keys", "next_values"],
            dynamic_axes={
                "token": {0: "rows"},
                "position": {0: "rows"},
                "page_row": {0: "rows", 1: "history"},
                "page_mask": {0: "rows", 1: "history"},
                "tree_mask": {0: "rows", 1: "rows"},
                "pool_keys": {0: "pages"},
                "pool_values": {0: "pages"},
                "cross_keys": {2: "keys"},
                "cross_values": {2: "keys"},
                "log_probs": {0: "rows"},
                "next_keys": {0: "rows"},
                "next_values": {0: "rows"},
            },
            opset_version=17,
            dynamo=False,
        )


def _gap(name: str, got: np.ndarray, expected: np.ndarray) -> float:
    """The largest absolute difference; a shape mismatch or a non-finite gap raises."""
    if got.shape != expected.shape:
        raise ValueError(f"{name}: the graph gives {got.shape}, the model {expected.shape}")
    gap = float(np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64))))
    if not gap <= CLOSE:
        raise ValueError(f"{name}: the exported graph strays {gap:.3g} from the model (> {CLOSE})")
    return gap


def _verify(model: KeyE2EModel, directory: Path, collator: KeyCollator) -> float:
    """Run the float32 export over :data:`VERIFY_PROBES` against the model; the worst gap."""
    onnx_model = OnnxKeyModel(directory)
    emittable = onnx_model.emittable
    # Prefixes in four calls: roots and a chain with no pages yet (all through
    # the tree mask), prefixes whose history is paged, and one deep prefix
    # whose ancestors are all missing again.
    call_prefixes: list[list[Prefix]] = [
        [(), (0,), (1 % emittable,)],
        [(0, 1 % emittable), (0, 2 % emittable)],
        [(0, 1 % emittable, 3 % emittable)],
        [(2 % emittable, 0, 1 % emittable, 3 % emittable, 0)],
    ]
    worst = 0.0
    with torch.no_grad():
        for keys, context in VERIFY_PROBES:
            inputs = collator.inputs([keys], [context])
            states = model.key_states(inputs)
            encoding = onnx_model.encode(inputs)
            worst = max(
                worst,
                _gap(
                    "ctc_log_probs",
                    encoding.log_probs,
                    model.ctc_log_probs(states.states)[0].numpy(),
                ),
            )
            for layer, cross in enumerate(model.decoder.gated_layers()):
                expected_keys, expected_values = cross.kv(states.states)
                worst = max(
                    worst,
                    _gap(
                        f"cross_keys[{layer}]", encoding.cross_keys[layer], expected_keys[0].numpy()
                    ),
                    _gap(
                        f"cross_values[{layer}]",
                        encoding.cross_values[layer],
                        expected_values[0].numpy(),
                    ),
                )
            transition = OnnxTransition(onnx_model, [encoding])
            for prefixes in call_prefixes:
                got = transition([0] * len(prefixes), prefixes)
                longest = max((len(prefix) for prefix in prefixes), default=0)
                padded = torch.zeros(len(prefixes), longest, dtype=torch.long)
                for row, prefix in enumerate(prefixes):
                    padded[row, : len(prefix)] = torch.tensor(prefix, dtype=torch.long)
                expected = model.transition_log_probs(
                    states,
                    torch.zeros(len(prefixes), dtype=torch.long),
                    padded,
                    torch.tensor([len(prefix) for prefix in prefixes]),
                )
                worst = max(worst, _gap(f"transition after {prefixes}", got, expected.numpy()))
    return worst


def _write_manifest(
    model: KeyE2EModel,
    step: int,
    lexicon: Lexicon,
    legality: Legality,
    tokenizer: BaseTokenizer,
    out_dir: Path,
    context_tokens: int,
    quantisation: dict[str, str | None],
) -> Path:
    """Externalize each graph's weights into its own file and write `key-model.json`."""
    tables = {}
    for name, graph in _GRAPHS.items():
        tables[name] = weights_table(externalize(out_dir / graph, f"{name}.weights"))
    config = model.decoder.reader.config
    keys = model.keys_config
    manifest = out_dir / MANIFEST
    manifest.write_text(
        json.dumps(
            {
                "step": step,
                "base_model": model.config.base_model,
                "layout": LAYOUT,
                "precision": {
                    name: {"dtype": "float32", "quantize": quantisation[name]} for name in _GRAPHS
                },
                "hidden": int(model.fill.config.hidden_size),
                "context_tokens": context_tokens,
                "max_keys": keys.max_keys,
                "specials": {
                    "pad": int(tokenizer.pad_token_id),
                    "cls": int(tokenizer.cls_token_id),
                    "sep": int(tokenizer.sep_token_id),
                    "mask": int(tokenizer.mask_token_id),
                },
                "letter_pad": LETTER_PAD,
                "characters": list(lexicon.characters),
                "bos": len(lexicon.characters),
                "decoder": {
                    "layers": config.layers,
                    "heads": config.heads,
                    "head_dim": config.hidden // config.heads,
                },
                "legality": {
                    "correct_syllables": keys.correct_syllables,
                    "probes": [
                        probe.as_record()
                        for probe in legality_probes(legality, keys.correct_syllables)
                    ],
                },
                "graphs": dict(_GRAPHS),
                "weights": {
                    name: {"file": f"{name}.weights", "tensors": list(tables[name].values())}
                    for name in _GRAPHS
                },
                "tokenizer": TOKENIZER,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def export_key_model(
    model: KeyE2EModel,
    step: int,
    lexicon: Lexicon,
    legality: Legality,
    tokenizer: BaseTokenizer,
    out_dir: Path,
    quantize: Literal["int8"] | None = None,
    decoder_quantize: DecoderQuantize = "weights",
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> Path:
    """Write the three graphs, a weights file per graph, `key-model.json` and `tokenizer.json`.

    *model* is a joint-stage key-level model over the base model's own context
    tower; a Qwen context, or a warm-up whose decoder reads span ends, is
    refused. *legality* is the automaton the model was trained against over
    *lexicon*: its probe lattices are pinned in the manifest, so a consumer can
    check that its own automaton builds the same ones.

    The float32 export is checked against *model* before anything is
    quantised (:func:`_verify`, within :data:`CLOSE`). With *quantize*
    ``"int8"`` the context and keys graphs become dynamic per-channel int8, as
    route A's towers do, and the decoder weight-only int8 (*decoder_quantize*
    ``"weights"``, as the char LM's) or dynamic int8 (``"dynamic"``);
    *decoder_quantize* means nothing without *quantize*. Returns the manifest's
    path.
    """
    if quantize not in (None, "int8"):
        raise ValueError(f"quantize must be 'int8' or None, got {quantize!r}")
    if decoder_quantize not in ("weights", "dynamic"):
        raise ValueError(
            f"decoder_quantize must be 'weights' or 'dynamic', got {decoder_quantize!r}"
        )
    if not isinstance(model.context, BertContextTower):
        raise ValueError(
            f"the key export reads context through route A's context graph, which is MacBERT's "
            f"tower; this model's context tower is a {type(model.context).__name__}"
        )
    if model.e2e_config.key_warm_up:
        raise ValueError("a key warm-up (S1) cannot be exported: its decoder reads span ends")
    if tuple(lexicon.characters) != tuple(legality.characters):
        raise ValueError("the legality automaton emits over another set than the lexicon")
    emitted = int(model.decoder.emit_ids.numel())
    if emitted != lexicon.size:
        raise ValueError(f"the model emits {emitted} characters, the lexicon {lexicon.size}")
    spans = SpanVocab.load()
    collator = build_key_collator(
        spans, lexicon, tokenizer, tokenizer, with_context=True, max_context_tokens=context_tokens
    )
    was_training = model.training
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        export_context_graph(model, tokenizer, out_dir)
        # Route A's wrapper leaves the model in training mode.
        model.eval()
        _export_keys_graph(model, tokenizer, out_dir / KEYS_GRAPH)
        _export_decoder_graph(model, out_dir / DECODER_GRAPH)
        write_tokenizer(tokenizer, out_dir)
        quantisation: dict[str, str | None] = dict.fromkeys(_GRAPHS)
        manifest = _write_manifest(
            model, step, lexicon, legality, tokenizer, out_dir, context_tokens, quantisation
        )
        worst = _verify(model, out_dir, collator)
        log.info("the float32 export matches the model", worst_gap=worst, probes=len(VERIFY_PROBES))
        if quantize == "int8":
            quantize_dynamic_int8(out_dir / CONTEXT_GRAPH)
            quantize_dynamic_int8(out_dir / KEYS_GRAPH)
            if decoder_quantize == "weights":
                quantize_weights_int8(out_dir / DECODER_GRAPH)
            else:
                quantize_dynamic_int8(out_dir / DECODER_GRAPH)
            quantisation = {
                "context": DYNAMIC_INT8,
                "keys": DYNAMIC_INT8,
                "decoder": WEIGHTS_INT8 if decoder_quantize == "weights" else DYNAMIC_INT8,
            }
            manifest = _write_manifest(
                model, step, lexicon, legality, tokenizer, out_dir, context_tokens, quantisation
            )
    finally:
        model.train(was_training)
    log.info(
        "exported the key-level model",
        directory=str(out_dir),
        quantize=quantize,
        decoder_quantize=decoder_quantize if quantize else None,
        emissions=lexicon.size,
    )
    return manifest
