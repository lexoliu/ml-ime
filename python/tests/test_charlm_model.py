"""Both character LMs: the step graph agrees with the forward pass, the exported graphs
agree with torch, and a stream of batches resumes from a saved position."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import polars as pl
import pytest
import torch

from mlime.logging import log
from mlime.train.charlm import (
    BOS,
    SEP,
    CharLmConfig,
    CharLmTraining,
    CharVocab,
    Position,
    Progress,
    SequenceBatches,
    export_onnx,
    save_checkpoint,
)
from mlime.train.charlm_model import Restricted, build
from mlime.train.charlm_vocab import SPECIALS
from mlime.train.loop import Distributed

CHARS = tuple("你好吗我很再见谢")
VOCAB = CharVocab(chars=SPECIALS + CHARS)

CONFIGS = {
    "lstm": CharLmConfig(arch="lstm", embedding=8, hidden=16, layers=2, dropout=0.0),
    "transformer": CharLmConfig(
        arch="transformer",
        embedding=8,
        hidden=16,
        layers=2,
        heads=2,
        feedforward=32,
        max_positions=32,
        dropout=0.0,
    ),
}


def _tokens(*chars: str) -> list[int]:
    index = VOCAB.index
    return [index[ch] for ch in chars]


@pytest.mark.parametrize("arch", sorted(CONFIGS))
def test_prefill_and_steps_reproduce_the_forward_pass(arch: str) -> None:
    torch.manual_seed(0)
    model = build(len(VOCAB), CONFIGS[arch]).eval()
    prelude = [BOS, *_tokens("你", "好"), SEP]
    sentence = _tokens("我", "很", "好")
    full = torch.tensor([prelude + sentence])
    with torch.no_grad():
        expected = torch.log_softmax(model(full), dim=-1)[0]
        features, prefix, state = model.prefill(torch.tensor([prelude]))
        restricted = Restricted(model, None)
        got = [restricted(features)]
        for token in sentence:
            features, state = model.step(torch.tensor([token]), prefix, state)
            got.append(restricted(features))
    for position, log_probs in enumerate(got):
        want = expected[len(prelude) - 1 + position]
        torch.testing.assert_close(log_probs[0], want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("arch", sorted(CONFIGS))
def test_two_beams_over_one_prefix_score_as_two_sequences(arch: str) -> None:
    torch.manual_seed(1)
    model = build(len(VOCAB), CONFIGS[arch]).eval()
    prelude = [BOS, *_tokens("再", "见"), SEP]
    beams = (_tokens("谢", "谢"), _tokens("我", "好"))
    with torch.no_grad():
        _, prefix, state = model.prefill(torch.tensor([prelude]))
        state = tuple(tensor.expand(2, *tensor.shape[1:]).contiguous() for tensor in state)
        restricted = Restricted(model, None)
        for step in range(2):
            token = torch.tensor([beam[step] for beam in beams])
            features, state = model.step(token, prefix, state)
        got = restricted(features)
        for row, beam in enumerate(beams):
            expected = torch.log_softmax(model(torch.tensor([prelude + beam])), dim=-1)[0, -1]
            torch.testing.assert_close(got[row], expected, atol=1e-5, rtol=1e-5)


def _pad_left(tensor: torch.Tensor, width: int) -> torch.Tensor:
    """``[..., T, hd]`` padded to ``width`` on the left of the time axis."""
    pad = width - tensor.shape[-2]
    if pad == 0:
        return tensor
    return torch.cat([torch.zeros(*tensor.shape[:-2], pad, tensor.shape[-1]), tensor], dim=-2)


def test_a_padded_batch_step_equals_per_record_steps() -> None:
    """Two records whose preludes differ step in one padded batch as they do alone."""
    torch.manual_seed(3)
    model = build(len(VOCAB), CONFIGS["transformer"]).eval()
    restricted = Restricted(model, None)
    preludes = (
        torch.tensor([[BOS, *_tokens("你", "好", "很"), SEP]]),
        torch.tensor([[BOS, *_tokens("再"), SEP]]),
    )
    with torch.no_grad():
        runs = [model.prefill(prelude) for prelude in preludes]
        tokens = torch.tensor(_tokens("我", "吗"))
        solo = [model.step(tokens[row : row + 1], run[1], run[2]) for row, run in enumerate(runs)]
        # The batch's prefix stacks one row per record, the shorter one padded
        # on the left; the mask marks each row's real prelude positions.
        width = max(prefix[0].shape[3] for _, prefix, _ in runs)
        prefix = tuple(
            torch.cat([_pad_left(run[1][i], width) for run in runs]) for i in range(len(runs[0][1]))
        )
        mask = torch.zeros(len(runs), width, dtype=torch.bool)
        for row, (_, row_prefix, _) in enumerate(runs):
            mask[row, width - row_prefix[0].shape[3] :] = True
        state = tuple(torch.cat([run[2][i] for run in runs]) for i in range(2))
        features, next_state = model.step(tokens, prefix, state, mask)
        got = restricted(features)
        diffs = [
            (got[row] - restricted(solo[row][0])[0]).abs().max().item() for row in range(len(runs))
        ]
        log.info("padded batch vs per-record step", max_abs_diff=max(diffs))
        for row in range(len(runs)):
            torch.testing.assert_close(got[row], restricted(solo[row][0])[0], atol=1e-5, rtol=1e-5)
            for tensor, wanted in zip(next_state, solo[row][1], strict=True):
                torch.testing.assert_close(tensor[row], wanted[0], atol=1e-5, rtol=1e-5)


def _checkpoint(path: Path, arch: str) -> None:
    torch.manual_seed(2)
    model = build(len(VOCAB), CONFIGS[arch])
    training = CharLmTraining(max_steps=10)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress = Progress(step=3, tokens_seen=100, positions=(Position(0, 1, 2),))
    save_checkpoint(path, model, VOCAB, progress, optimizer, scheduler, scaler, training)


@pytest.mark.parametrize("arch", sorted(CONFIGS))
def test_exported_graphs_reproduce_torch(arch: str, tmp_path: Path) -> None:
    checkpoint = tmp_path / "charlm-final.pt"
    _checkpoint(checkpoint, arch)
    restrict = tmp_path / "emittable.txt"
    restrict.write_text("\n".join(CHARS[:4]) + "\n", encoding="utf-8")
    export_onnx(checkpoint, tmp_path / "export", restrict)
    manifest = json.loads((tmp_path / "export" / "charlm.json").read_text())
    assert manifest["arch"] == arch
    assert manifest["restricted_to"] == 5  # four characters plus <eos>
    weights = manifest["weights"]
    # One table shared by both graphs, or one per graph.
    tables = {"step": weights, "prefill": weights} if "file" in weights else weights
    for graph in ("step", "prefill"):
        file = tmp_path / "export" / tables[graph]["file"]
        assert file.is_file()
        blob = file.read_bytes()
        proto = onnx.load(
            tmp_path / "export" / ("prefill.onnx" if graph == "prefill" else "charlm.onnx"),
            load_external_data=False,
        )
        external = {
            t.name for t in proto.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL
        }
        assert external == {t["name"] for t in tables[graph]["tensors"]}
        for tensor in tables[graph]["tensors"]:
            size = np.dtype(tensor["dtype"]).itemsize
            assert np.prod(tensor["shape"]) * size == tensor["length"]
            extent = blob[tensor["offset"] : tensor["offset"] + tensor["length"]]
            assert len(extent) == tensor["length"]
    # ort reads the initializers out of the weights file, so loading the
    # graphs at all proves the external references resolve; the torch match
    # below proves they hold the right bytes.
    prefill = ort.InferenceSession(str(tmp_path / "export" / "prefill.onnx"))
    step = ort.InferenceSession(str(tmp_path / "export" / "charlm.onnx"))
    prelude = [BOS, *_tokens("你", "好"), SEP]
    beams = (_tokens("我", "吗"), _tokens("你", "好"))
    outputs = prefill.run(None, {"tokens": np.array([prelude], dtype=np.int64)})
    names = ["log_probs", *manifest["prefix"], *manifest["state"]]
    by_name = dict(zip(names, outputs, strict=True))
    prefix = {name: by_name[name] for name in manifest["prefix"]}
    mask = _prefix_mask(prefix)
    state = {name: np.repeat(by_name[name], 2, axis=0) for name in manifest["state"]}
    for position in range(2):
        token = np.array([beam[position] for beam in beams], dtype=np.int64)
        outputs = step.run(None, {"token": token, **prefix, **mask, **state})
        log_probs = outputs[0]
        state = dict(zip(manifest["state"], outputs[1:], strict=True))
    from mlime.train.charlm import load_model

    model, _, _ = load_model(checkpoint, torch.device("cpu"))
    model.eval()
    keep = torch.tensor(sorted({VOCAB.index[ch] for ch in CHARS[:4]} | {2}))
    restricted = Restricted(model, keep)
    with torch.no_grad():
        for row, beam in enumerate(beams):
            features = model.features(torch.tensor([prelude + beam]))[0, -1:]
            expected = restricted(features)[0].numpy()
            np.testing.assert_allclose(log_probs[row], expected, atol=1e-4, rtol=1e-4)
    assert (log_probs[:, VOCAB.index["谢"]] == Restricted.UNREACHABLE).all()


def _prefix_mask(prefix: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The step's ``prefix_mask`` input: one all-real row, as wide as the prefix."""
    if not prefix:
        return {}
    width = next(iter(prefix.values())).shape[3]
    return {"prefix_mask": np.ones((1, width), dtype=np.bool_)}


def _run_export(dir: Path) -> list[np.ndarray]:
    """The ``log_probs`` rows of an export over the fixture prelude and two beams."""
    manifest = json.loads((dir / "charlm.json").read_text())
    prefill = ort.InferenceSession(str(dir / "prefill.onnx"))
    step = ort.InferenceSession(str(dir / "charlm.onnx"))
    prelude = [BOS, *_tokens("你", "好"), SEP]
    outputs = prefill.run(None, {"tokens": np.array([prelude], dtype=np.int64)})
    names = ["log_probs", *manifest["prefix"], *manifest["state"]]
    by_name = dict(zip(names, outputs, strict=True))
    prefix = {name: by_name[name] for name in manifest["prefix"]}
    mask = _prefix_mask(prefix)
    state = {name: np.repeat(by_name[name], 2, axis=0) for name in manifest["state"]}
    rows = [by_name["log_probs"]]
    beams = (_tokens("我", "吗"), _tokens("你", "好"))
    for position in range(len(beams[0])):
        token = np.array([beam[position] for beam in beams], dtype=np.int64)
        outputs = step.run(None, {"token": token, **prefix, **mask, **state})
        rows.append(outputs[0])
        state = dict(zip(manifest["state"], outputs[1:], strict=True))
    return rows


def test_int8_export_scores_within_005_nats_of_fp32(tmp_path: Path) -> None:
    """The dynamic-quantized graphs load in onnxruntime and track the fp32 export."""
    checkpoint = tmp_path / "charlm-final.pt"
    _checkpoint(checkpoint, "transformer")
    restrict = tmp_path / "emittable.txt"
    restrict.write_text("\n".join(CHARS[:4]) + "\n", encoding="utf-8")
    export_onnx(checkpoint, tmp_path / "fp32", restrict)
    export_onnx(checkpoint, tmp_path / "int8", restrict, quantize="int8")
    quantized = _run_export(tmp_path / "int8")
    reference = _run_export(tmp_path / "fp32")
    for got, want in zip(quantized, reference, strict=True):
        np.testing.assert_allclose(got, want, atol=0.05)


def _shard(path: Path, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    texts = ["".join(rng.choice(list(CHARS), size=int(n))) for n in rng.integers(2, 6, rows)]
    contexts = ["".join(rng.choice(list(CHARS), size=int(n))) for n in rng.integers(0, 4, rows)]
    pl.DataFrame({"text": texts, "context": contexts}).write_parquet(path)


def test_batches_resume_from_a_saved_position(tmp_path: Path) -> None:
    shards = [tmp_path / f"shard-{i}.parquet" for i in range(3)]
    for i, shard in enumerate(shards):
        _shard(shard, rows=20, seed=i)
    world = Distributed()

    def stream() -> SequenceBatches:
        return SequenceBatches(shards, VOCAB, 4, 32, 24, world, seed=7, bucket_rows=8)

    first = stream()
    batches = iter(first)
    seen = [next(batches) for _ in range(12)]
    position = first.position
    assert position.epoch >= 0
    resumed = stream()
    resumed.skip_to(position)
    continued = iter(resumed)
    for expected in [next(batches) for _ in range(6)]:
        got = next(continued)
        assert torch.equal(got.tokens, expected.tokens)
        assert torch.equal(got.targets, expected.targets)
    assert len(seen) == 12
