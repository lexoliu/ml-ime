"""The exported route A graphs reproduce the PyTorch forward, manifest included.

What `ime-neural` computes is what `mlime train emit` computed -- the masked,
normalised log probability of every candidate at every position of every
reading -- so the checks here are against `RouteAModel`'s own forward on a
fixture model small enough to build without a download, with the gate set
nonzero so the context path is exercised rather than shorted out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from transformers import BertConfig

from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.model import RouteAConfig, RouteAModel
from mlime.train.routea import LAYOUT, export_onnx
from mlime.train.samples import BaseTokenizer, Collator, TrainingExample
from mlime.train.spans import SpanVocab

TINY = BertConfig(
    vocab_size=256,
    hidden_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
)


@pytest.fixture(name="model")
def model_fixture(lexicon: Lexicon) -> RouteAModel:
    """A tiny route A model with a nonzero gate, so the context path is real."""
    torch.manual_seed(0)
    model = RouteAModel.from_config(
        TINY, lexicon, RouteAConfig(cross_attention_layers=2, cross_attention_dropout=0.0)
    )
    with torch.no_grad():
        for layer in model.gated_layers():
            layer.gate.fill_(0.4)
    return model.eval()


def _examples(spans: SpanVocab, context: str | None) -> list[TrainingExample]:
    """Two readings over the fixture lexicon's own spans and characters."""
    return [
        TrainingExample(
            id="a:0",
            spans=("zhong", "wo"),
            span_ids=(spans.id("zhong"), spans.id("wo")),
            targets=(-100, -100),
            context=context,
        ),
        TrainingExample(
            id="a:1",
            spans=("zh", "o"),
            span_ids=(spans.id("zh"), spans.id("o")),
            targets=(-100, -100),
            context=context,
        ),
    ]


def _restricted_log_probs(model: RouteAModel, batch) -> torch.Tensor:
    """The masked log-softmax the score file rounds: logits at ``candidate_mask``."""
    with torch.no_grad():
        logits = model.scores(batch)
    masks = model.candidate_mask.index_select(0, batch.span_ids.reshape(-1)).reshape(
        *batch.span_ids.shape, -1
    )
    return logits.masked_fill(~masks, torch.finfo(logits.dtype).min).log_softmax(dim=-1)


def _run_fill(export: Path, batch) -> np.ndarray:
    """Run `fill.onnx` over *batch*'s tensors."""
    fill = ort.InferenceSession(str(export / "fill.onnx"))
    context = ort.InferenceSession(str(export / "context.onnx")).run(
        None,
        {
            "context_ids": batch.context_ids.numpy(),
            "context_mask": batch.context_mask.numpy(),
        },
    )[0]
    has_context = batch.has_context.numpy()
    # The graph expects one context row per path; a dropped context binds
    # zeros, exactly as the gate's zero switch makes them irrelevant.
    paths = batch.input_ids.shape[0]
    context = np.broadcast_to(context, (paths, *context.shape[1:])).copy()
    context[~has_context.astype(bool)] = 0.0
    return fill.run(
        None,
        {
            "input_ids": batch.input_ids.numpy(),
            "attention_mask": batch.attention_mask.numpy(),
            "span_ids": batch.span_ids.numpy(),
            "span_positions": batch.span_positions.numpy(),
            "context": context,
            "context_mask": np.broadcast_to(
                batch.context_mask.numpy(), (paths, *batch.context_mask.shape[1:])
            ).copy(),
            "has_context": has_context,
        },
    )[0]


def test_exported_graphs_reproduce_torch(
    model: RouteAModel,
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """Context on and off, the graphs emit the tower's masked log probabilities."""
    export_onnx(model, 7, lexicon, spans, tokenizer, tmp_path)

    for dropout, context in ((0.0, "北京大学"), (1.0, "北京大学"), (0.0, None)):
        batch = Collator(
            tokenizer, CandidateSpace(spans, lexicon.candidate_mask), context_dropout=dropout
        )(_examples(spans, context))
        got = _run_fill(tmp_path, batch)
        expected = _restricted_log_probs(model, batch).numpy()
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)

    # A different batch width and a longer context exercise the dynamic axes.
    batch = Collator(
        tokenizer,
        CandidateSpace(spans, lexicon.candidate_mask),
        context_dropout=0.0,
        max_context_tokens=16,
    )(_examples(spans, "北京" * 10)[:1])
    got = _run_fill(tmp_path, batch)
    expected = _restricted_log_probs(model, batch).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_the_manifest_describes_the_files(
    model: RouteAModel,
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """The manifest is the loader's contract: layout pin, vocabularies, tensor table."""
    export_onnx(model, 7, lexicon, spans, tokenizer, tmp_path)
    manifest = json.loads((tmp_path / "route-a.json").read_text())
    assert manifest["layout"] == LAYOUT
    assert manifest["step"] == 7
    assert manifest["dtype"] == "float32"
    assert manifest["quantize"] is None
    assert manifest["hidden"] == TINY.hidden_size
    assert manifest["specials"] == {"pad": 0, "cls": 1, "sep": 2, "mask": 3}
    assert manifest["spans"] == list(spans)
    assert manifest["characters"] == list(lexicon.characters)
    for name in ("context", "fill"):
        table = manifest["weights"][name]
        blob = (tmp_path / table["file"]).read_bytes()
        proto = onnx.load(tmp_path / manifest["graphs"][name], load_external_data=False)
        external = {
            t.name for t in proto.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL
        }
        assert external == {t["name"] for t in table["tensors"]}
        for tensor in table["tensors"]:
            size = np.dtype(tensor["dtype"]).itemsize
            assert np.prod(tensor["shape"]) * size == tensor["length"]
            assert (
                len(blob[tensor["offset"] : tensor["offset"] + tensor["length"]])
                == tensor["length"]
            )
    # The candidate mask the loader checks admissions against is one named
    # tensor in the fill weights, shaped [spans, emissions].
    mask = next(
        t for t in manifest["weights"]["fill"]["tensors"] if t["name"] == manifest["candidate_mask"]
    )
    assert mask["dtype"] == "bool"
    assert mask["shape"] == [len(spans), len(lexicon.characters)]


def test_int8_export_tracks_fp32(
    model: RouteAModel,
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: BaseTokenizer,
    tmp_path: Path,
) -> None:
    """The quantized graphs load in onnxruntime and stay within the int8 bound."""
    export_onnx(model, 7, lexicon, spans, tokenizer, tmp_path / "fp32")
    export_onnx(model, 7, lexicon, spans, tokenizer, tmp_path / "int8", quantize="int8")
    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask), context_dropout=0.0)(
        _examples(spans, "北京大学")
    )
    np.testing.assert_allclose(
        _run_fill(tmp_path / "int8", batch), _run_fill(tmp_path / "fp32", batch), atol=0.05
    )
