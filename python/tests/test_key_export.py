"""The key-level export: three graphs that reproduce the model, and the search over them (#266).

A tiny randomly initialised joint-stage model is exported at float32 and at
int8. At float32 the graphs match the PyTorch model's CTC log-probabilities,
cross keys and values, and transition rows -- through the page pool, a chain of
prefixes in one call and a deep prefix whose ancestors have no page -- and the
very same search (:func:`mlime.train.key_search.search`) over the exported graphs
returns the PyTorch search's top-k. The manifest carries what a loader needs,
and the refusals fire: a Qwen context, a warm-up, an automaton over another
set, a legality digest that moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnx
import pytest
import torch
from test_key_eval import (
    decoder,
    eval_rows,
    key_run,
    verifier_fixture,  # noqa: F401 -- the fixture, by name
)
from test_keys import (
    KeyTokenizer,
    key_collator,
    legality_fixture,  # noqa: F401
    reader_vocab_fixture,  # noqa: F401
    tiny_joint_model,
    tiny_key_model,
    vocabularies_fixture,  # noqa: F401
)

from mlime.legality import Legality
from mlime.train.charlm_vocab import CharVocab
from mlime.train.key_eval import KeyRecord, KeyWeights, report_keys
from mlime.train.key_export import CLOSE, VERIFY_PROBES, export_key_model
from mlime.train.key_model import KeyE2EModel
from mlime.train.key_onnx import (
    LAYOUT,
    LEGALITY_PROBES,
    KeyManifest,
    OnnxKeyDecoder,
    OnnxKeyModel,
    OnnxTransition,
)
from mlime.train.key_search import LegalityVerifier
from mlime.train.lexicon import Lexicon
from mlime.train.spans import SpanVocab

#: The records the searches are compared on: every eval row, typed three ways.
RECORDS = [
    KeyRecord(record=index, keys=keys, text=text, context=context)
    for index, (keys, text, context) in enumerate(eval_rows())
]

GRID = [KeyWeights(1.0, 1.0, False), KeyWeights(0.5, 2.0, True)]


@dataclass(frozen=True)
class Exported:
    """A tiny model and where it was exported."""

    model: KeyE2EModel
    directory: Path
    manifest: Path


def export(
    model: KeyE2EModel,
    legality: Legality,
    lexicon: Lexicon,
    directory: Path,
    **options: Any,
) -> Exported:
    manifest = export_key_model(model, 11, lexicon, legality, KeyTokenizer(), directory, **options)
    return Exported(model, directory, manifest)


@pytest.fixture(name="fp32")
def fp32_fixture(
    lexicon: Lexicon, legality: Legality, reader_vocab: CharVocab, tmp_path: Path
) -> Exported:
    return export(tiny_joint_model(lexicon, reader_vocab), legality, lexicon, tmp_path / "fp32")


def graph_ops(directory: Path, name: str) -> dict[str, int]:
    """Operator counts of one exported graph."""
    proto = onnx.load(directory / name, load_external_data=False)
    counts: dict[str, int] = {}
    for node in proto.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def onnx_decoder(
    exported: Exported, lexicon: Lexicon, spans: SpanVocab, legality: Legality
) -> OnnxKeyDecoder:
    keys = exported.model.e2e_config.keys
    assert keys is not None
    return OnnxKeyDecoder(
        model=OnnxKeyModel(exported.directory),
        lexicon=lexicon,
        collator=key_collator(spans, lexicon),
        legality=legality,
        options=keys.options,
    )


def test_the_float32_graphs_reproduce_the_model(
    fp32: Exported, lexicon: Lexicon, spans: SpanVocab
) -> None:
    """The export checked itself; check it again, and on a key string it did not run."""
    model = fp32.model.eval()
    onnx_model = OnnxKeyModel(fp32.directory)
    collator = key_collator(spans, lexicon)
    for keys, context in (*VERIFY_PROBES, ("zhongchongwolv", "钟"), ("a", None)):
        inputs = collator.inputs([keys], [context])
        with torch.no_grad():
            states = model.key_states(inputs)
            expected = model.ctc_log_probs(states.states)[0].numpy()
        encoding = onnx_model.encode(inputs)
        assert encoding.log_probs.shape == (len(keys), lexicon.size + 1)
        np.testing.assert_allclose(encoding.log_probs, expected, atol=CLOSE)
        config = model.decoder.reader.config
        assert encoding.cross_keys.shape == (
            config.layers,
            config.heads,
            len(keys),
            config.hidden // config.heads,
        )
        np.testing.assert_allclose(np.exp(encoding.log_probs).sum(-1), 1.0, atol=1e-5)


def test_the_pool_chains_and_cold_prefixes_agree(
    fp32: Exported, spans: SpanVocab, lexicon: Lexicon
) -> None:
    """A prefix's row is the same whether its ancestors were paged, chained or recomputed."""
    model = fp32.model.eval()
    onnx_model = OnnxKeyModel(fp32.directory)
    inputs = key_collator(spans, lexicon).inputs(["woaibeijing"], ["北京"])
    encoding = onnx_model.encode(inputs)
    deep = (1, 2, 3, 4)
    cold = OnnxTransition(onnx_model, [encoding])
    [chained] = cold([0], [deep])
    # Ancestors and the prefix in one call: four ancestors and itself.
    assert cold.calls == [5]
    warm = OnnxTransition(onnx_model, [encoding])
    for length in range(len(deep)):
        warm([0], [deep[:length]])
    [paged] = warm([0], [deep])
    assert warm.calls == [1, 1, 1, 1, 1]
    np.testing.assert_allclose(chained, paged, atol=CLOSE)
    # A second call for a prefix already paged still answers, and claims no page.
    used = cold.pool.used
    [again] = cold([0], [deep])
    assert cold.pool.used == used
    np.testing.assert_allclose(again, chained, atol=1e-6)
    with torch.no_grad():
        states = model.key_states(inputs)
        expected = model.transition_log_probs(
            states, torch.zeros(1, dtype=torch.long), torch.tensor([deep]), torch.tensor([4])
        )
    np.testing.assert_allclose(chained, expected[0].numpy(), atol=CLOSE)
    # Two records in one call are answered by their own encodings.
    other = onnx_model.encode(key_collator(spans, lexicon).inputs(["wabj"], [None]))
    pair = OnnxTransition(onnx_model, [encoding, other])
    first, second = pair([0, 1], [(1,), (1,)])
    assert not np.allclose(first, second)
    np.testing.assert_allclose(first, OnnxTransition(onnx_model, [encoding])([0], [(1,)])[0])


def test_the_search_over_the_graphs_is_the_pytorch_search(
    fp32: Exported, lexicon: Lexicon, spans: SpanVocab, legality: Legality
) -> None:
    model = fp32.model.eval()
    torch_side = decoder(model, lexicon, spans, legality)
    onnx_side = onnx_decoder(fp32, lexicon, spans, legality)
    expected = torch_side.decode(RECORDS, GRID)
    got = onnx_side.decode(RECORDS, GRID)
    assert len(got) == len(expected) == len(GRID)
    for point_got, point_expected in zip(got, expected, strict=True):
        for record, mine, theirs in zip(RECORDS, point_got, point_expected, strict=True):
            assert [h.ids for h in mine] == [h.ids for h in theirs], record.keys
            assert [h.cuts for h in mine] == [h.cuts for h in theirs], record.keys
            for a, b in zip(mine, theirs, strict=True):
                assert a.score == pytest.approx(b.score, abs=1e-3)
                assert a.transition == pytest.approx(b.transition, abs=1e-3)
    assert onnx_side.times.records == len(RECORDS)
    assert onnx_side.times.transition > 0.0


def test_the_report_over_the_graphs_is_the_pytorch_report(
    fp32: Exported,
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    verifier: LegalityVerifier,
    tmp_path: Path,
) -> None:
    """`eval keys --onnx` is :func:`report_keys` over the same run with the export's decoder."""
    model = fp32.model.eval()
    torch_run = key_run(model, lexicon, spans, legality, verifier)
    onnx_run = replace(torch_run, decoder=onnx_decoder(fp32, lexicon, spans, legality))
    weights = KeyWeights(0.7, 1.3, False)
    expected = report_keys(torch_run, tmp_path / "torch", [weights], slice_="all")
    got = report_keys(onnx_run, tmp_path / "onnx", [weights], slice_="all")
    assert got == expected
    dumps = [
        [json.loads(line) for line in (tmp_path / name / "all-beta0.7-gamma1.3-sum.jsonl").open()]
        for name in ("torch", "onnx")
    ]
    for a, b in zip(*dumps, strict=True):
        assert [h["text"] for h in a["hypotheses"]] == [h["text"] for h in b["hypotheses"]]


def test_records_cuts_each_slice_to_its_first_records(
    fp32: Exported,
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    verifier: LegalityVerifier,
    tmp_path: Path,
) -> None:
    run = replace(
        key_run(fp32.model.eval(), lexicon, spans, legality, verifier),
        decoder=onnx_decoder(fp32, lexicon, spans, legality),
        limit=2,
    )
    for name in ("dev", "test"):
        held = run.indices(name)
        assert len(held) == 2 and held == sorted(held)
    report_keys(run, tmp_path, [GRID[0]], slice_="all")
    written = (tmp_path / "all-beta1-gamma1-sum.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(written) == 2
    assert run.indices("all") == [0, 1]
    assert run.decoder.times.records == 2


def test_the_manifest_describes_the_export(
    fp32: Exported, lexicon: Lexicon, legality: Legality
) -> None:
    record = json.loads(fp32.manifest.read_text(encoding="utf-8"))
    model = fp32.model
    config = model.decoder.reader.config
    assert record["layout"] == LAYOUT == "keys-pages"
    assert record["step"] == 11
    assert record["base_model"] == model.config.base_model
    assert record["hidden"] == model.fill.config.hidden_size
    assert record["specials"] == {"pad": 0, "cls": 1, "sep": 2, "mask": 3}
    assert record["letter_pad"] == 26
    assert record["characters"] == list(lexicon.characters)
    assert record["bos"] == lexicon.size
    assert record["decoder"] == {
        "layers": config.layers,
        "heads": config.heads,
        "head_dim": config.hidden // config.heads,
    }
    assert record["graphs"] == {
        "context": "context.onnx",
        "keys": "keys.onnx",
        "decoder": "decoder.onnx",
    }
    assert record["tokenizer"] == "tokenizer.json" and (fp32.directory / "tokenizer.json").is_file()
    assert record["precision"] == {
        name: {"dtype": "float32", "quantize": None} for name in ("context", "keys", "decoder")
    }
    keys = model.keys_config
    assert record["legality"]["correct_syllables"] == keys.correct_syllables
    probes = record["legality"]["probes"]
    assert [(p["keys"], p["unfinished_tail"]) for p in probes] == list(LEGALITY_PROBES)
    assert all(set(p) == {"keys", "unfinished_tail", "arcs", "digest"} for p in probes)
    assert any(p["arcs"] > 0 for p in probes)
    for name in ("context", "keys", "decoder"):
        table = record["weights"][name]
        assert table["file"] == f"{name}.weights"
        blob = (fp32.directory / table["file"]).read_bytes()
        proto = onnx.load(fp32.directory / record["graphs"][name], load_external_data=False)
        external = {
            t.name for t in proto.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL
        }
        assert external == {t["name"] for t in table["tensors"]}
        for tensor in table["tensors"]:
            size = np.dtype(tensor["dtype"]).itemsize
            assert int(np.prod(tensor["shape"])) * size == tensor["length"]
            assert tensor["offset"] + tensor["length"] <= len(blob)
            assert tensor["offset"] % size == 0
    # What a loader checks against its own tables.
    manifest = KeyManifest.load(fp32.directory)
    manifest.check_lexicon(lexicon)
    manifest.check_legality(legality)


def test_the_graphs_export_as_matmuls_and_take_their_axes_dynamic(fp32: Exported) -> None:
    for name in ("context.onnx", "keys.onnx", "decoder.onnx"):
        assert "Gemm" not in graph_ops(fp32.directory, name), name
    proto = onnx.load(fp32.directory / "decoder.onnx", load_external_data=False)
    inputs = {i.name: i for i in proto.graph.input}
    assert list(inputs) == [
        "token",
        "position",
        "page_row",
        "page_mask",
        "tree_mask",
        "pool_keys",
        "pool_values",
        "cross_keys",
        "cross_values",
    ]
    dims = {
        name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
        for name, i in inputs.items()
    }
    assert dims["token"] == ["rows"]
    assert dims["page_row"] == ["rows", "history"]
    assert dims["tree_mask"] == ["rows", "rows"]
    assert dims["pool_keys"][0] == "pages" and len(dims["pool_keys"]) == 4
    assert dims["cross_keys"][2] == "keys" and len(dims["cross_keys"]) == 4
    assert [o.name for o in proto.graph.output] == ["log_probs", "next_keys", "next_values"]
    keys = onnx.load(fp32.directory / "keys.onnx", load_external_data=False)
    assert [o.name for o in keys.graph.output] == ["ctc_log_probs", "cross_keys", "cross_values"]
    assert [i.name for i in keys.graph.input] == [
        "input_ids",
        "key_letters",
        "attention_mask",
        "context",
        "context_mask",
        "has_context",
    ]


@pytest.mark.parametrize("decoder_quantize", ["weights", "dynamic"])
def test_the_int8_export_loads_and_tracks_float32(
    fp32: Exported,
    lexicon: Lexicon,
    legality: Legality,
    spans: SpanVocab,
    tmp_path: Path,
    decoder_quantize: str,
) -> None:
    quantized = export(
        fp32.model,
        legality,
        lexicon,
        tmp_path / decoder_quantize,
        quantize="int8",
        decoder_quantize=decoder_quantize,
    )
    record = json.loads(quantized.manifest.read_text(encoding="utf-8"))
    assert record["precision"] == {
        "context": {"dtype": "float32", "quantize": "dynamic-int8"},
        "keys": {"dtype": "float32", "quantize": "dynamic-int8"},
        "decoder": {
            "dtype": "float32",
            "quantize": "weights-int8" if decoder_quantize == "weights" else "dynamic-int8",
        },
    }
    decoder_ops = graph_ops(quantized.directory, "decoder.onnx")
    if decoder_quantize == "weights":
        assert decoder_ops["DequantizeLinear"] > 0 and "MatMulInteger" not in decoder_ops
    else:
        assert "DequantizeLinear" not in decoder_ops
        assert decoder_ops.get("MatMulInteger", 0) + decoder_ops.get("DynamicQuantizeLinear", 0) > 0
    assert graph_ops(quantized.directory, "keys.onnx").get("MatMulInteger", 0) > 0
    # The weights files shrink: int8 where there was float32.
    quantized_size = (quantized.directory / "keys.weights").stat().st_size
    assert quantized_size < (fp32.directory / "keys.weights").stat().st_size
    inputs = key_collator(spans, lexicon).inputs(["woaibeijing"], ["北京"])
    ours = OnnxKeyModel(quantized.directory)
    base = OnnxKeyModel(fp32.directory)
    got, expected = ours.encode(inputs), base.encode(inputs)
    assert np.isfinite(got.log_probs).all()
    np.testing.assert_allclose(got.log_probs, expected.log_probs, atol=0.5)
    rows = OnnxTransition(ours, [got])([0, 0], [(), (1,)])
    reference = OnnxTransition(base, [expected])([0, 0], [(), (1,)])
    assert np.isfinite(rows).all()
    np.testing.assert_allclose(rows, reference, atol=0.5)
    # The search runs over the quantised graphs too.
    hypotheses = onnx_decoder(quantized, lexicon, spans, legality).decode(RECORDS[:3], GRID[:1])
    assert all(len(point) == 3 and all(record for record in point) for point in hypotheses)


def test_a_qwen_context_a_warm_up_and_mismatched_tables_are_refused(
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    reader_vocab: CharVocab,
    tmp_path: Path,
) -> None:
    class QwenLike(torch.nn.Module):
        pass

    model = tiny_joint_model(lexicon, reader_vocab)
    model.context = QwenLike()  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r"MacBERT's tower.*QwenLike"):
        export_key_model(model, 1, lexicon, legality, KeyTokenizer(), tmp_path / "qwen")
    assert not (tmp_path / "qwen" / "context.onnx").exists()

    warm_up = tiny_key_model(lexicon, reader_vocab)
    with pytest.raises(ValueError, match="warm-up"):
        export_key_model(warm_up, 1, lexicon, legality, KeyTokenizer(), tmp_path / "warm")

    joint = tiny_joint_model(lexicon, reader_vocab)
    with pytest.raises(ValueError, match="quantize must be"):
        export_key_model(
            joint,
            1,
            lexicon,
            legality,
            KeyTokenizer(),
            tmp_path / "bad",
            quantize="int4",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="decoder_quantize must be"):
        export_key_model(
            joint,
            1,
            lexicon,
            legality,
            KeyTokenizer(),
            tmp_path / "bad",
            quantize="int8",
            decoder_quantize="fp16",  # type: ignore[arg-type]
        )
    other = SimpleNamespace(characters=tuple(reversed(lexicon.characters)))
    with pytest.raises(ValueError, match="another set than the lexicon"):
        export_key_model(joint, 1, lexicon, other, KeyTokenizer(), tmp_path / "set")  # type: ignore[arg-type]


def test_a_moved_digest_or_another_emission_order_is_refused(
    fp32: Exported, lexicon: Lexicon, legality: Legality
) -> None:
    manifest = KeyManifest.load(fp32.directory)
    index = next(index for index, probe in enumerate(manifest.probes) if probe.arcs)
    probe = manifest.probes[index]
    changed = replace(probe, digest=f"{int(probe.digest, 16) ^ 1:016x}")
    moved = replace(
        manifest, probes=(*manifest.probes[:index], changed, *manifest.probes[index + 1 :])
    )
    with pytest.raises(ValueError, match=rf"{probe.keys!r}.*pins"):
        moved.check_legality(legality)
    other = replace(manifest, characters=tuple(reversed(manifest.characters)))
    with pytest.raises(ValueError, match=r"another row"):
        other.check_lexicon(lexicon)
    layout = json.loads((fp32.directory / "key-model.json").read_text(encoding="utf-8"))
    layout["layout"] = "resident-pages"
    wrong = fp32.directory / "wrong"
    wrong.mkdir()
    (wrong / "key-model.json").write_text(json.dumps(layout), encoding="utf-8")
    with pytest.raises(ValueError, match="layout"):
        KeyManifest.load(wrong)


def test_the_keys_graph_runs_one_row_per_keystroke(
    fp32: Exported, spans: SpanVocab, lexicon: Lexicon
) -> None:
    two = key_collator(spans, lexicon).inputs(["wo", "wabj"], [None, None])
    with pytest.raises(ValueError, match="one row"):
        OnnxKeyModel(fp32.directory).encode(two)
