"""The key-level model as a bundle for the Core ML conversion (#266).

The key model runs on the device through native Core ML, converted from a
bundle by a project of its own (``python/coreml/``) that does not import
``mlime``: its torch is older than ``mlime``'s. The bundle is the interchange.
:func:`write_key_bundle` writes four files into one directory.

``weights.safetensors`` holds the float32 weights the three function families
read (and ``decoder.emit_ids``, the int64 map from emission index to reader id
the tied head is gathered by), under the model's own parameter names, and
nothing else: the span table,
the letter encoder and the candidate mask, which only the span model reads, are
left out. Two tensors are derived so no consumer repeats the arithmetic:
``key_front.letter_table`` ``[27, hidden]`` is each letter's embedding plus its
QWERTY coordinates' projection, the padding row included, replacing
``key_front.letters.weight`` and ``key_front.place.weight``; the rest are the
parameters as trained. :mod:`mlime.train.key_manifest` lists each tensor with
its shape and the families that read it.

``key-model.json`` is the manifest (:class:`~mlime.train.key_manifest.KeyManifest`),
``tokenizer.json`` the base model's tokenizer.

``reference.npz`` holds the float32 PyTorch model's answers over fixed probes,
which the conversion verifies against. The manifest's ``reference`` lists the
probes (a key string, its context, the prefix sets the tree was asked); arrays
are named ``<probe>.<what>``:

* ``context_ids`` int32 ``[64]`` and ``context_mask`` float32 ``[64]``, the
  context function's inputs, padded to its length; the mask is all zeros for a
  probe with no context. ``context`` ``[64, hidden]`` is the context function's
  output where the mask has a real position, and zeros where it has none, which
  is what the host feeds the keys function then.
* ``input_ids``, ``key_letters`` int32 and ``attention_mask`` float32, each
  ``[n + 2]``: the keys function's inputs at exact width. ``ctc_logits``
  ``[n, E + 1]`` is its CTC logits, blank last, and ``cross_keys`` and
  ``cross_values`` ``[layers, heads, 128, head_dim]`` the cross keys and
  values padded to ``N_MAX``, with ``cross_bias`` ``[1, 1, 1, 128]``.
* ``tree<k>.call<c>.reader_ids`` and ``.positions`` int32 ``[S]``,
  ``.tree_bias`` float32 ``[S, S]`` and ``.logits`` float32 ``[S, E]``: tree
  call ``c`` of request set ``k`` (the calls :func:`~mlime.train.key_tree.plan`
  makes), with ``.answers`` int32 ``[A, 2]`` pairing a request index with the
  row that answers it and ``.log_probs`` float32 ``[A, E]`` those requests'
  rows by :meth:`~mlime.train.key_model.KeyE2EModel.transition_log_probs`:
  the search's own transition, which the tree's rows must reproduce.

The bundle checks itself while it is written: every tree row it records is
compared with the transition's, within :data:`CLOSE`.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from safetensors.torch import save_file
from transformers import BertConfig

from mlime.legality import Legality
from mlime.logging import log
from mlime.train.charlm_vocab import BOS
from mlime.train.key_eval import DecodeTimes, ModelTransition, build_key_collator
from mlime.train.key_manifest import (
    CONTEXT_BUCKETS,
    LAYOUT,
    N_MAX,
    NEG,
    REFERENCE,
    WEIGHTS,
    DecoderDims,
    KeyFrontDims,
    KeyManifest,
    ModelDims,
    ReferenceProbe,
    Specials,
    TensorEntry,
    TowerDims,
    families,
    legality_probes,
)
from mlime.train.key_model import KeyBlock, KeyE2EModel
from mlime.train.key_tree import POLICY_VERSION, Prefix, plan
from mlime.train.keys import KeyCollator
from mlime.train.lexicon import Lexicon
from mlime.train.model import BertContextTower
from mlime.train.onnx_tools import write_tokenizer
from mlime.train.samples import DEFAULT_CONTEXT_TOKENS, LETTER_PAD, BaseTokenizer
from mlime.train.spans import SpanVocab

#: How far a tree row of the bundle may stray from the transition's at float32.
CLOSE = 1e-4

#: The probes the reference is computed over, as ``(name, keys, context)``: a
#: full sentence with its context, an initials string with none, and a mixed one.
REFERENCE_PROBES: tuple[tuple[str, str, str | None], ...] = (
    ("sentence", "wmzyq", "我们吃水饺"),
    ("initials", "zhongguo", None),
    ("mixed", "shuijiap", "水饺好吃"),
)

#: How many prefixes the seeded request set of each probe asks, and how deep they may go.
RANDOM_REQUESTS = 16
RANDOM_DEPTH = 6

Family = Literal["context", "keys", "tree"]

#: Which parameters each family reads, by name; a parameter no pattern names is
#: not in the bundle. The derived ``key_front.letter_table`` stands for the
#: letters and the coordinates' projection (:func:`bundle_weights`).
_READS: dict[Family, re.Pattern[str]] = {
    "context": re.compile(r"context\.(embeddings|encoder)\."),
    "keys": re.compile(
        r"(fill\.(embeddings|encoder)\.|cross_attention\.\d+\.|key_front\.(positions|blocks|"
        r"letter_table)|head\.(transform|decoder)\.|blank\.|"
        r"decoder\.cross_attention\.\d+\.(key|value)\.)"
    ),
    "tree": re.compile(
        r"(decoder\.reader\.|decoder\.cross_attention\.\d+\.(gate|norm|query|out)\.?|decoder\.emit_ids)"
    ),
}

#: What the derived letter table replaces.
_REPLACED = ("key_front.letters.", "key_front.place.")


@dataclass(frozen=True)
class ReferenceArrays:
    """The arrays of ``reference.npz`` by name, and the probes that describe them."""

    arrays: dict[str, npt.NDArray[np.generic]]
    probes: tuple[ReferenceProbe, ...]


def check_model(model: KeyE2EModel, lexicon: Lexicon, legality: Legality) -> None:
    """Refuse a model, or tables, the bundle does not describe.

    The bundle reads context through MacBERT's tower, from a joint-stage model
    over the lexicon's emission order, and a legality automaton over the same
    set.
    """
    if not isinstance(model.context, BertContextTower):
        raise ValueError(
            f"the key bundle reads context through MacBERT's tower; this model's context tower "
            f"is a {type(model.context).__name__}"
        )
    if model.e2e_config.key_warm_up:
        raise ValueError("a key warm-up (S1) cannot be bundled: its decoder reads span ends")
    if model.decoder.span_input is not None:
        raise ValueError("the decoder has a span input; only a joint-stage model bundles")
    if tuple(lexicon.characters) != tuple(legality.characters):
        raise ValueError("the legality automaton emits over another set than the lexicon")
    emitted = int(model.decoder.emit_ids.numel())
    if emitted != lexicon.size:
        raise ValueError(f"the model emits {emitted} characters, the lexicon {lexicon.size}")


def bundle_weights(model: KeyE2EModel) -> dict[str, tuple[torch.Tensor, tuple[Family, ...]]]:
    """The tensors of ``weights.safetensors`` by name, each with the families that read it.

    Sorted by name. Every family's pattern must match something, so a renamed
    module fails here and not in the conversion.
    """
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    front = model.key_front
    with torch.no_grad():
        state["key_front.letter_table"] = (
            (front.letters.weight + front.place(front.coordinates)).detach().cpu()
        )
    chosen: dict[str, tuple[torch.Tensor, tuple[Family, ...]]] = {}
    for name in sorted(state):
        if name.startswith(_REPLACED):
            continue
        reading = tuple(family for family, pattern in _READS.items() if pattern.match(name))
        if reading:
            chosen[name] = (state[name].clone().contiguous(), reading)
    for family, pattern in _READS.items():
        if not any(pattern.match(name) for name in chosen):
            raise ValueError(f"no parameter is read by the {family} family; the names have moved")
    return chosen


def _tower_dims(config: BertConfig) -> TowerDims:
    return TowerDims(
        layers=int(config.num_hidden_layers),
        heads=int(config.num_attention_heads),
        hidden=int(config.hidden_size),
        intermediate=int(config.intermediate_size),
        positions=int(config.max_position_embeddings),
        vocabulary=int(config.vocab_size),
        layer_norm_eps=float(config.layer_norm_eps),
    )


def model_dims(model: KeyE2EModel) -> ModelDims:
    """Every dimension the families' modules are built from, read off *model*."""
    reader = model.decoder.reader
    config = reader.config
    front = model.key_front
    first = cast(KeyBlock, front.blocks[0])
    return ModelDims(
        context=_tower_dims(model.context.config),
        fill=_tower_dims(model.fill.config),
        gated=len(model.gated_layers()),
        key_front=KeyFrontDims(
            layers=len(front.blocks),
            heads=int(first.heads),
            feedforward=int(first.up.out_features),
            positions=int(front.positions.num_embeddings),
        ),
        decoder=DecoderDims(
            layers=config.layers,
            heads=config.heads,
            head_dim=config.hidden // config.heads,
            hidden=config.hidden,
            embedding=config.embedding,
            feedforward=config.feedforward,
            positions=config.max_positions,
            vocabulary=int(reader.embed.num_embeddings),
            layer_norm_eps=float(reader.norm.eps),
        ),
        emitted=int(model.decoder.emit_ids.numel()),
    )


def reference_requests(emitted: int, seed: int) -> tuple[tuple[Prefix, ...], ...]:
    """The prefix sets one probe's tree reference asks, from the probe's seed.

    A scripted set (the root, two roots' children, a chain) and a seeded one
    that mixes depths, so the packing sees siblings, chains and a root.
    """
    scripted = ((), (0,), (1 % emitted,), (0, 1 % emitted), (0, 1 % emitted, 2 % emitted))
    rng = random.Random(seed)
    seeded = tuple(
        tuple(rng.randrange(emitted) for _ in range(rng.randint(0, RANDOM_DEPTH)))
        for _ in range(RANDOM_REQUESTS)
    )
    return scripted, seeded


def _int32(values: torch.Tensor) -> npt.NDArray[np.int32]:
    return values.detach().cpu().numpy().astype(np.int32)


def _float32(values: torch.Tensor) -> npt.NDArray[np.float32]:
    return values.detach().cpu().float().numpy()


def reference_arrays(
    model: KeyE2EModel, collator: KeyCollator, tokenizer: BaseTokenizer, context_tokens: int
) -> ReferenceArrays:
    """The reference's arrays over :data:`REFERENCE_PROBES`, checked against the transition.

    Run under ``eval()``; a tree row that strays more than :data:`CLOSE` from
    :meth:`~mlime.train.key_model.KeyE2EModel.transition_log_probs` raises.
    """
    emitted = int(model.decoder.emit_ids.numel())
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    described = []
    with torch.no_grad():
        for index, (name, keys, context) in enumerate(REFERENCE_PROBES):
            inputs = collator.inputs([keys], [context])
            real = int(inputs.context_ids.shape[1])
            ids = torch.full((1, context_tokens), int(tokenizer.pad_token_id), dtype=torch.long)
            mask = torch.zeros((1, context_tokens), dtype=torch.long)
            ids[:, :real] = inputs.context_ids
            mask[:, :real] = inputs.context_mask
            inputs = replace(inputs, context_ids=ids, context_mask=mask)
            has_context = bool(inputs.has_context[0])
            arrays[f"{name}.context_ids"] = _int32(ids[0])
            arrays[f"{name}.context_mask"] = _float32((mask * int(has_context))[0])
            hidden = int(model.fill.config.hidden_size)
            arrays[f"{name}.context"] = (
                _float32(model.context.memory(ids, mask).states[0])
                if has_context
                else np.zeros((context_tokens, hidden), dtype=np.float32)
            )
            arrays[f"{name}.input_ids"] = _int32(inputs.key_input_ids[0])
            arrays[f"{name}.key_letters"] = _int32(inputs.key_letters[0])
            arrays[f"{name}.attention_mask"] = _float32(inputs.key_attention[0])
            states = model.key_states(inputs)
            arrays[f"{name}.ctc_logits"] = _float32(model.ctc_logits(states.states[0]))
            cross = model.cross_memory(states)
            arrays[f"{name}.cross_keys"] = _float32(cross.keys)
            arrays[f"{name}.cross_values"] = _float32(cross.values)
            arrays[f"{name}.cross_bias"] = _float32(cross.bias)
            transition = ModelTransition(model, states, DecodeTimes())
            sets = reference_requests(emitted, index)
            for number, requests in enumerate(sets):
                for position, call in enumerate(plan(requests)):
                    prefix = f"{name}.tree{number}.call{position}"
                    feeds = model.tree_feeds(call)
                    logits = model.tree_logits(
                        feeds.reader_ids, feeds.positions, feeds.tree_bias, cross
                    )[0]
                    answers = list(call.answers)
                    expected = transition([0] * len(answers), [requests[a] for a, _ in answers])
                    got = logits[[row for _, row in answers]].float().log_softmax(dim=-1)
                    gap = float((got - expected).abs().max())
                    if not gap <= CLOSE:
                        raise ValueError(
                            f"{prefix}: the tree's rows stray {gap:.3g} from the transition's"
                        )
                    arrays[f"{prefix}.reader_ids"] = _int32(feeds.reader_ids[0])
                    arrays[f"{prefix}.positions"] = _int32(feeds.positions[0])
                    arrays[f"{prefix}.tree_bias"] = _float32(feeds.tree_bias[0, 0])
                    arrays[f"{prefix}.logits"] = _float32(logits)
                    arrays[f"{prefix}.answers"] = np.array(answers, dtype=np.int32).reshape(-1, 2)
                    arrays[f"{prefix}.log_probs"] = _float32(expected)
            described.append(ReferenceProbe(name, keys, context, sets))
    return ReferenceArrays(arrays, tuple(described))


def write_key_bundle(
    model: KeyE2EModel,
    step: int,
    lexicon: Lexicon,
    legality: Legality,
    tokenizer: BaseTokenizer,
    out_dir: Path,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
) -> Path:
    """Write the bundle into *out_dir* and return the manifest's path.

    The bundle is the weights, the manifest, the tokenizer and the reference.

    *model* is a joint-stage key-level model over the base model's own context
    tower; a Qwen context, or a warm-up whose decoder reads span ends, is
    refused. *legality* is the automaton the model was trained against over
    *lexicon*: its probe lattices are pinned in the manifest, so a consumer can
    check that its own automaton builds the same ones. *context_tokens* is the
    context function's length, which the bundle's buckets fix.
    """
    if context_tokens not in CONTEXT_BUCKETS:
        raise ValueError(
            f"the context function holds {CONTEXT_BUCKETS} tokens, not {context_tokens}"
        )
    check_model(model, lexicon, legality)
    spans = SpanVocab.load()
    collator = build_key_collator(
        spans, lexicon, tokenizer, tokenizer, with_context=True, max_context_tokens=context_tokens
    )
    was_training = model.training
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        weights = bundle_weights(model)
        reference = reference_arrays(model, collator, tokenizer, context_tokens)
    finally:
        model.train(was_training)
    save_file({name: tensor for name, (tensor, _) in weights.items()}, str(out_dir / WEIGHTS))
    np.savez_compressed(out_dir / REFERENCE, **reference.arrays)  # type: ignore[arg-type]
    write_tokenizer(tokenizer, out_dir)
    keys = model.keys_config
    manifest = KeyManifest(
        layout=LAYOUT,
        step=step,
        base_model=model.config.base_model,
        policy_version=POLICY_VERSION,
        n_max=N_MAX,
        neg=NEG,
        context_tokens=context_tokens,
        specials=Specials(
            pad=int(tokenizer.pad_token_id),
            cls=int(tokenizer.cls_token_id),
            sep=int(tokenizer.sep_token_id),
            mask=int(tokenizer.mask_token_id),
        ),
        letter_pad=LETTER_PAD,
        characters=tuple(lexicon.characters),
        reader_ids=tuple(int(index) for index in model.decoder.emit_ids),
        reader_bos=BOS,
        dims=model_dims(model),
        correct_syllables=keys.correct_syllables,
        probes=tuple(legality_probes(legality, keys.correct_syllables)),
        families=families(),
        tensors=tuple(
            TensorEntry(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).removeprefix("torch."),
                families=reading,
            )
            for name, (tensor, reading) in weights.items()
        ),
        reference=reference.probes,
    )
    path = manifest.write(out_dir)
    log.info(
        "wrote the key-level bundle",
        directory=str(out_dir),
        tensors=len(weights),
        probes=len(REFERENCE_PROBES),
        emissions=lexicon.size,
    )
    return path


def read_reference(directory: Path) -> dict[str, npt.NDArray[np.generic]]:
    """``reference.npz`` of *directory* as a dict of arrays: what a consumer verifies against."""
    with np.load(directory / REFERENCE) as archive:
        return {name: archive[name] for name in archive.files}
