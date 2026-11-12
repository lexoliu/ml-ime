"""A tiny random bundle, with its reference computed by an independent oracle.

The oracle is the key-level model written the way its training framework writes it, with stock
torch parts: ``scaled_dot_product_attention`` with boolean masks and ``is_causal``,
``nn.MultiheadAttention``, and the decoder run causally over each whole prefix (so the tree
function's packed, biased attention is checked against plain per-prefix decoding). None of it
shares a line with :mod:`mlime_coreml.modules`, so a bundle written here checks the port the
way a bundle written by the training side does, on any host and without mlime.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

from mlime_coreml import spec
from mlime_coreml.bundle import (
    MANIFEST,
    REFERENCE,
    TOKENIZER,
    WEIGHTS,
    DecoderDims,
    KeyFrontDims,
    Manifest,
    ModelDims,
    ReferenceProbe,
    Specials,
    TensorEntry,
    TowerDims,
    expected_families,
)
from mlime_coreml.spec import Family

TOWER = TowerDims(
    layers=2,
    heads=4,
    hidden=32,
    intermediate=64,
    positions=130,
    vocabulary=256,
    layer_norm_eps=1e-12,
)
KEY_FRONT = KeyFrontDims(layers=2, heads=4, feedforward=64, positions=128)
DECODER = DecoderDims(
    layers=2,
    heads=4,
    head_dim=4,
    hidden=16,
    embedding=8,
    feedforward=32,
    positions=300,
    vocabulary=14,
    layer_norm_eps=1e-5,
)
EMITTED = 9
GATED = 1
LETTERS = 26
SPECIALS = Specials(pad=0, cls=1, sep=2, mask=3)
READER_BOS = 1
READER_IDS = (4, 5, 6, 7, 8, 9, 10, 11, 12)
EPS = spec.DEFAULT_LAYER_NORM_EPS

#: The probes the reference holds: a name, its keys (as letter ids) and its context's token ids.
PROBES: tuple[tuple[str, list[int], list[int] | None], ...] = (
    ("sentence", [22, 12, 25, 24, 16], [1, 40, 41, 42, 43, 2]),
    ("initials", list(range(10)), None),
    ("mixed", [18, 7, 20, 8, 9, 0, 15], [1, 90, 91, 92, 2]),
)

Weights = dict[str, torch.Tensor]


def dims() -> ModelDims:
    """The tiny model's dimensions."""
    return ModelDims(
        context=TOWER,
        fill=TOWER,
        gated=GATED,
        key_front=KEY_FRONT,
        decoder=DECODER,
        emitted=EMITTED,
    )


# --- weights ----------------------------------------------------------------------------------


def _bert_shapes(prefix: str, tower: TowerDims) -> dict[str, tuple[int, ...]]:
    h, i = tower.hidden, tower.intermediate
    shapes: dict[str, tuple[int, ...]] = {
        f"{prefix}.embeddings.word_embeddings.weight": (tower.vocabulary, h),
        f"{prefix}.embeddings.position_embeddings.weight": (tower.positions, h),
        f"{prefix}.embeddings.token_type_embeddings.weight": (2, h),
        f"{prefix}.embeddings.LayerNorm.weight": (h,),
        f"{prefix}.embeddings.LayerNorm.bias": (h,),
    }
    for layer in range(tower.layers):
        base = f"{prefix}.encoder.layer.{layer}"
        for part in ("query", "key", "value"):
            shapes[f"{base}.attention.self.{part}.weight"] = (h, h)
            shapes[f"{base}.attention.self.{part}.bias"] = (h,)
        shapes[f"{base}.attention.output.dense.weight"] = (h, h)
        shapes[f"{base}.attention.output.dense.bias"] = (h,)
        for norm in ("attention.output", "output"):
            shapes[f"{base}.{norm}.LayerNorm.weight"] = (h,)
            shapes[f"{base}.{norm}.LayerNorm.bias"] = (h,)
        shapes[f"{base}.intermediate.dense.weight"] = (i, h)
        shapes[f"{base}.intermediate.dense.bias"] = (i,)
        shapes[f"{base}.output.dense.weight"] = (h, i)
        shapes[f"{base}.output.dense.bias"] = (h,)
    return shapes


def shapes() -> dict[str, tuple[int, ...]]:
    """Every tensor of the bundle by name: the layout the training side writes."""
    h, d = TOWER.hidden, DECODER
    found = _bert_shapes("context", TOWER) | _bert_shapes("fill", TOWER)
    for layer in range(GATED):
        base = f"cross_attention.{layer}"
        found |= {
            f"{base}.norm.weight": (h,),
            f"{base}.norm.bias": (h,),
            f"{base}.attention.in_proj_weight": (3 * h, h),
            f"{base}.attention.in_proj_bias": (3 * h,),
            f"{base}.attention.out_proj.weight": (h, h),
            f"{base}.attention.out_proj.bias": (h,),
            f"{base}.gate": (),
        }
    kf = KEY_FRONT
    found |= {
        "key_front.letter_table": (LETTERS + 1, h),
        "key_front.positions.weight": (kf.positions, h),
    }
    for layer in range(kf.layers):
        base = f"key_front.blocks.{layer}"
        found |= {
            f"{base}.attention_norm.weight": (h,),
            f"{base}.attention_norm.bias": (h,),
            f"{base}.qkv.weight": (3 * h, h),
            f"{base}.qkv.bias": (3 * h,),
            f"{base}.out.weight": (h, h),
            f"{base}.out.bias": (h,),
            f"{base}.feedforward_norm.weight": (h,),
            f"{base}.feedforward_norm.bias": (h,),
            f"{base}.up.weight": (kf.feedforward, h),
            f"{base}.up.bias": (kf.feedforward,),
            f"{base}.down.weight": (h, kf.feedforward),
            f"{base}.down.bias": (h,),
        }
    found |= {
        "head.transform.dense.weight": (h, h),
        "head.transform.dense.bias": (h,),
        "head.transform.LayerNorm.weight": (h,),
        "head.transform.LayerNorm.bias": (h,),
        "head.decoder.weight": (EMITTED, h),
        "head.decoder.bias": (EMITTED,),
        "blank.weight": (1, h),
        "blank.bias": (1,),
        "decoder.emit_ids": (EMITTED,),
        "decoder.reader.embed.weight": (d.vocabulary, d.embedding),
        "decoder.reader.input.weight": (d.hidden, d.embedding),
        "decoder.reader.input.bias": (d.hidden,),
        "decoder.reader.positions.weight": (d.positions, d.hidden),
        "decoder.reader.norm.weight": (d.hidden,),
        "decoder.reader.norm.bias": (d.hidden,),
        "decoder.reader.project.weight": (d.embedding, d.hidden),
        "decoder.reader.project.bias": (d.embedding,),
    }
    for layer in range(d.layers):
        base = f"decoder.reader.blocks.{layer}"
        found |= {
            f"{base}.norm1.weight": (d.hidden,),
            f"{base}.norm1.bias": (d.hidden,),
            f"{base}.qkv.weight": (3 * d.hidden, d.hidden),
            f"{base}.qkv.bias": (3 * d.hidden,),
            f"{base}.out.weight": (d.hidden, d.hidden),
            f"{base}.out.bias": (d.hidden,),
            f"{base}.norm2.weight": (d.hidden,),
            f"{base}.norm2.bias": (d.hidden,),
            f"{base}.ff.0.weight": (d.feedforward, d.hidden),
            f"{base}.ff.0.bias": (d.feedforward,),
            f"{base}.ff.2.weight": (d.hidden, d.feedforward),
            f"{base}.ff.2.bias": (d.hidden,),
        }
        cross = f"decoder.cross_attention.{layer}"
        found |= {
            f"{cross}.norm.weight": (d.hidden,),
            f"{cross}.norm.bias": (d.hidden,),
            f"{cross}.query.weight": (d.hidden, d.hidden),
            f"{cross}.query.bias": (d.hidden,),
            f"{cross}.key.weight": (d.hidden, h),
            f"{cross}.key.bias": (d.hidden,),
            f"{cross}.value.weight": (d.hidden, h),
            f"{cross}.value.bias": (d.hidden,),
            f"{cross}.out.weight": (d.hidden, d.hidden),
            f"{cross}.out.bias": (d.hidden,),
            f"{cross}.gate": (),
        }
    return dict(sorted(found.items()))


def random_weights(seed: int = 5) -> Weights:
    """Random weights shaped like a trained model's: small, norms near one, gates open."""
    generator = torch.Generator().manual_seed(seed)
    weights: Weights = {}
    for name, shape in shapes().items():
        if name == "decoder.emit_ids":
            weights[name] = torch.tensor(READER_IDS, dtype=torch.long)
        elif name.endswith("gate"):
            weights[name] = torch.tensor(0.5)
        elif "LayerNorm.weight" in name or ("norm" in name and name.endswith("weight")):
            weights[name] = 1.0 + 0.1 * torch.randn(shape, generator=generator)
        elif name.endswith("bias"):
            weights[name] = 0.02 * torch.randn(shape, generator=generator)
        else:
            weights[name] = 0.1 * torch.randn(shape, generator=generator)
    return weights


def _families(name: str) -> tuple[Family, ...]:
    """The families that read a tensor, as the training side's table names them."""
    if name.startswith("context."):
        return ("context",)
    if name.startswith("decoder.cross_attention.") and name.split(".")[-2] in ("key", "value"):
        return ("keys",)
    if name.startswith("decoder."):
        return ("tree",)
    return ("keys",)


# --- the oracle -------------------------------------------------------------------------------


def _linear(w: Weights, name: str, x: torch.Tensor) -> torch.Tensor:
    return nn.functional.linear(x, w[f"{name}.weight"], w[f"{name}.bias"])


def _norm(w: Weights, name: str, x: torch.Tensor, eps: float) -> torch.Tensor:
    return nn.functional.layer_norm(x, x.shape[-1:], w[f"{name}.weight"], w[f"{name}.bias"], eps)


def _heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    return x.view(x.shape[0], x.shape[1], heads, -1).transpose(1, 2)


def _merge(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)


def _bert(
    w: Weights,
    prefix: str,
    embeds: torch.Tensor,
    keep: torch.Tensor,
    gates: list[tuple[torch.nn.MultiheadAttention, torch.Tensor, str]] | None = None,
    context: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """BERT over ``[1, L, H]`` embeddings, ``keep`` the bool ``[L]`` of real positions."""
    tower = TOWER
    length = embeds.shape[1]
    x = embeds + w[f"{prefix}.embeddings.position_embeddings.weight"][:length]
    x = x + w[f"{prefix}.embeddings.token_type_embeddings.weight"][0]
    x = _norm(w, f"{prefix}.embeddings.LayerNorm", x, tower.layer_norm_eps)
    first = tower.layers - (len(gates) if gates else 0)
    for layer in range(tower.layers):
        base = f"{prefix}.encoder.layer.{layer}"
        q, k, v = (
            _heads(_linear(w, f"{base}.attention.self.{part}", x), tower.heads)
            for part in ("query", "key", "value")
        )
        attended = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=keep[None, None, None, :]
        )
        x = _norm(
            w,
            f"{base}.attention.output.LayerNorm",
            _linear(w, f"{base}.attention.output.dense", _merge(attended)) + x,
            tower.layer_norm_eps,
        )
        lifted = nn.functional.gelu(_linear(w, f"{base}.intermediate.dense", x))
        x = _norm(
            w,
            f"{base}.output.LayerNorm",
            _linear(w, f"{base}.output.dense", lifted) + x,
            tower.layer_norm_eps,
        )
        if gates and layer >= first:
            module, gate, name = gates[layer - first]
            assert context is not None
            rows, real = context
            normed = _norm(w, f"{name}.norm", x, EPS)
            attended_rows, _ = module(
                normed, rows[None], rows[None], key_padding_mask=~real[None], need_weights=False
            )
            x = x + gate * attended_rows
    return x


def _mha(w: Weights, name: str) -> torch.nn.MultiheadAttention:
    """``nn.MultiheadAttention`` over a bundle's tensors, batch first."""
    module = torch.nn.MultiheadAttention(TOWER.hidden, TOWER.heads, batch_first=True)
    module.load_state_dict(
        {
            "in_proj_weight": w[f"{name}.attention.in_proj_weight"],
            "in_proj_bias": w[f"{name}.attention.in_proj_bias"],
            "out_proj.weight": w[f"{name}.attention.out_proj.weight"],
            "out_proj.bias": w[f"{name}.attention.out_proj.bias"],
        }
    )
    return module.eval()


def oracle_context(w: Weights, ids: list[int]) -> torch.Tensor:
    """The context tower over *ids* at their exact length: ``[len, H]``."""
    tokens = torch.tensor([ids])
    embeds = nn.functional.embedding(tokens, w["context.embeddings.word_embeddings.weight"])
    return _bert(w, "context", embeds, torch.ones(len(ids), dtype=torch.bool))[0]


def oracle_keys(
    w: Weights, letters: list[int], context: tuple[torch.Tensor, torch.Tensor] | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CTC logits ``[n, E+1]`` and cross keys and values ``[layers, heads, N_MAX, d]``."""
    n = len(letters)
    ids = torch.tensor([[SPECIALS.cls, *([SPECIALS.mask] * n), SPECIALS.sep]])
    key_letters = torch.tensor([[LETTERS, *letters, LETTERS]])
    front = nn.functional.embedding(key_letters, w["key_front.letter_table"])
    front = front + w["key_front.positions.weight"][: n + 2]
    for layer in range(KEY_FRONT.layers):
        base = f"key_front.blocks.{layer}"
        q, k, v = _linear(w, f"{base}.qkv", _norm(w, f"{base}.attention_norm", front, EPS)).chunk(
            3, -1
        )
        attended = nn.functional.scaled_dot_product_attention(
            _heads(q, KEY_FRONT.heads),
            _heads(k, KEY_FRONT.heads),
            _heads(v, KEY_FRONT.heads),
            is_causal=True,
        )
        front = front + _linear(w, f"{base}.out", _merge(attended))
        lifted = nn.functional.gelu(
            _linear(w, f"{base}.up", _norm(w, f"{base}.feedforward_norm", front, EPS))
        )
        front = front + _linear(w, f"{base}.down", lifted)
    at_key = (key_letters != LETTERS).unsqueeze(-1)
    embeds = (
        nn.functional.embedding(ids, w["fill.embeddings.word_embeddings.weight"]) + front * at_key
    )
    gates = [
        (_mha(w, f"cross_attention.{i}"), w[f"cross_attention.{i}.gate"], f"cross_attention.{i}")
        for i in range(GATED)
    ]
    states = _bert(
        w,
        "fill",
        embeds,
        torch.ones(n + 2, dtype=torch.bool),
        gates if context is not None else None,
        context,
    )[0]
    transformed = _norm(
        w,
        "head.transform.LayerNorm",
        nn.functional.gelu(_linear(w, "head.transform.dense", states)),
        TOWER.layer_norm_eps,
    )
    logits = torch.cat(
        [_linear(w, "head.decoder", transformed), _linear(w, "blank", transformed)], dim=-1
    )[1 : n + 1]
    keys = torch.zeros(DECODER.layers, DECODER.heads, spec.N_MAX, DECODER.head_dim)
    values = torch.zeros_like(keys)
    real = states[1 : n + 1]
    for layer in range(DECODER.layers):
        base = f"decoder.cross_attention.{layer}"
        for tensor, part in ((keys, "key"), (values, "value")):
            projected = _linear(w, f"{base}.{part}", real).view(n, DECODER.heads, -1)
            tensor[layer, :, :n] = projected.transpose(0, 1)
    return logits, keys, values


def oracle_row(
    w: Weights, prefix: tuple[int, ...], keys: torch.Tensor, values: torch.Tensor, real: int
) -> torch.Tensor:
    """The decoder's next-character logits ``[E]`` after *prefix*, run causally over all of it."""
    d = DECODER
    tokens = torch.tensor([[READER_BOS, *(READER_IDS[index] for index in prefix)]])
    x = nn.functional.linear(
        nn.functional.embedding(tokens, w["decoder.reader.embed.weight"]),
        w["decoder.reader.input.weight"],
        w["decoder.reader.input.bias"],
    )
    x = x + w["decoder.reader.positions.weight"][: tokens.shape[1]]
    for layer in range(d.layers):
        base = f"decoder.reader.blocks.{layer}"
        q, k, v = _linear(w, f"{base}.qkv", _norm(w, f"{base}.norm1", x, d.layer_norm_eps)).chunk(
            3, -1
        )
        attended = nn.functional.scaled_dot_product_attention(
            _heads(q, d.heads), _heads(k, d.heads), _heads(v, d.heads), is_causal=True
        )
        x = x + _linear(w, f"{base}.out", _merge(attended))
        lifted = nn.functional.gelu(
            _linear(w, f"{base}.ff.0", _norm(w, f"{base}.norm2", x, d.layer_norm_eps))
        )
        x = x + nn.functional.linear(lifted, w[f"{base}.ff.2.weight"], w[f"{base}.ff.2.bias"])
        cross = f"decoder.cross_attention.{layer}"
        query = _heads(_linear(w, f"{cross}.query", _norm(w, f"{cross}.norm", x, EPS)), d.heads)
        attended = nn.functional.scaled_dot_product_attention(
            query, keys[layer][None, :, :real], values[layer][None, :, :real]
        )
        x = x + w[f"{cross}.gate"] * _linear(w, f"{cross}.out", _merge(attended))
    features = _norm(w, "decoder.reader.norm", x, d.layer_norm_eps)[0, -1]
    projected = _linear(w, "decoder.reader.project", features)
    emitted = w["decoder.reader.embed.weight"][torch.tensor(READER_IDS)]
    return emitted @ projected


# --- the bundle -------------------------------------------------------------------------------


def _call(
    w: Weights,
    requests: list[tuple[int, ...]],
    keys: torch.Tensor,
    values: torch.Tensor,
    real: int,
) -> dict[str, np.ndarray]:
    """One packed call over *requests*: the policy's single call, with the oracle's rows."""
    nodes = sorted({p[:i] for p in requests for i in range(len(p) + 1)}, key=lambda p: (len(p), p))
    bucket = next(size for size in spec.TREE_BUCKETS if size >= len(nodes))
    index = {node: row for row, node in enumerate(nodes)}
    reader_ids = np.full(bucket, READER_BOS, dtype=np.int32)
    positions = np.zeros(bucket, dtype=np.int32)
    bias = np.full((bucket, bucket), spec.NEG, dtype=np.float32)
    logits = np.zeros((bucket, EMITTED), dtype=np.float32)
    root = oracle_row(w, (), keys, values, real).numpy()
    for row in range(bucket):
        bias[row, row] = 0.0
        logits[row] = root
    for row, node in enumerate(nodes):
        positions[row] = len(node)
        if node:
            reader_ids[row] = READER_IDS[node[-1]]
            for ancestor in (node[:i] for i in range(len(node))):
                bias[row, index[ancestor]] = 0.0
        logits[row] = oracle_row(w, node, keys, values, real).numpy()
    answers = np.array([(i, index[p]) for i, p in enumerate(requests)], dtype=np.int32).reshape(
        -1, 2
    )
    log_probs = torch.from_numpy(logits[answers[:, 1]]).log_softmax(-1).numpy()
    return {
        "reader_ids": reader_ids,
        "positions": positions,
        "tree_bias": bias,
        "logits": logits,
        "answers": answers,
        "log_probs": log_probs,
    }


def reference_sets(seed: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """A scripted request set and a seeded one that spans two buckets' worth of depth."""
    rng = random.Random(seed)
    scripted = ((), (0,), (1,), (0, 1), (0, 1, 2))
    seeded = tuple(
        tuple(rng.randrange(EMITTED) for _ in range(rng.randint(0, 6))) for _ in range(16)
    )
    return scripted, seeded


def write_bundle(directory: Path, seed: int = 5) -> Manifest:
    """Write the tiny bundle: weights, manifest, tokenizer stand-in and an oracle-made reference."""
    directory.mkdir(parents=True, exist_ok=True)
    w = random_weights(seed)
    arrays: dict[str, np.ndarray] = {}
    probes = []
    with torch.no_grad():
        for index, (name, letters, context_ids) in enumerate(PROBES):
            ids = np.full(spec.CONTEXT_BUCKETS[0], SPECIALS.pad, dtype=np.int32)
            mask = np.zeros(spec.CONTEXT_BUCKETS[0], dtype=np.float32)
            rows = np.zeros((spec.CONTEXT_BUCKETS[0], TOWER.hidden), dtype=np.float32)
            context = None
            if context_ids is not None:
                real = len(context_ids)
                ids[:real], mask[:real] = context_ids, 1.0
                encoded = oracle_context(w, context_ids)
                rows[:real] = encoded.numpy()
                context = (encoded, torch.ones(real, dtype=torch.bool))
            arrays |= {
                f"{name}.context_ids": ids,
                f"{name}.context_mask": mask,
                f"{name}.context": rows,
            }
            n = len(letters)
            row_ids = np.array([SPECIALS.cls, *([SPECIALS.mask] * n), SPECIALS.sep], dtype=np.int32)
            arrays[f"{name}.input_ids"] = row_ids
            arrays[f"{name}.key_letters"] = np.array([LETTERS, *letters, LETTERS], dtype=np.int32)
            arrays[f"{name}.attention_mask"] = np.ones(n + 2, dtype=np.float32)
            logits, keys, values = oracle_keys(w, letters, context)
            arrays[f"{name}.ctc_logits"] = logits.numpy()
            arrays[f"{name}.cross_keys"] = keys.numpy()
            arrays[f"{name}.cross_values"] = values.numpy()
            bias = np.full((1, 1, 1, spec.N_MAX), spec.NEG, dtype=np.float32)
            bias[..., :n] = 0.0
            arrays[f"{name}.cross_bias"] = bias
            sets = reference_sets(index)
            for number, requests in enumerate(sets):
                call = _call(w, list(requests), keys, values, n)
                for key, value in call.items():
                    arrays[f"{name}.tree{number}.call0.{key}"] = value
            probes.append(ReferenceProbe(name, "".join(chr(97 + c) for c in letters), None, sets))
    save_file(w, str(directory / WEIGHTS))
    np.savez_compressed(directory / REFERENCE, **arrays)  # type: ignore[arg-type]
    (directory / TOKENIZER).write_text("{}\n")
    manifest = Manifest(
        layout="key-bundle",
        step=1,
        base_model="tiny",
        policy_version=spec.POLICY_VERSION,
        n_max=spec.N_MAX,
        neg=spec.NEG,
        context_tokens=spec.CONTEXT_BUCKETS[0],
        specials=SPECIALS,
        letter_pad=LETTERS,
        characters=tuple("abcdefghi"),
        reader_ids=READER_IDS,
        reader_bos=READER_BOS,
        dims=dims(),
        correct_syllables=True,
        probes=(),
        families=expected_families(),
        tensors=tuple(
            TensorEntry(
                name, tuple(tensor.shape), str(tensor.dtype).removeprefix("torch."), _families(name)
            )
            for name, tensor in w.items()
        ),
        reference=tuple(probes),
    )
    (directory / MANIFEST).write_text(json.dumps(dataclasses.asdict(manifest), indent=2) + "\n")
    return manifest
