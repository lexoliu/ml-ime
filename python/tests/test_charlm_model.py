"""Both character LMs: the step graph agrees with the forward pass, the exported graphs
agree with torch, and a stream of batches resumes from a saved position."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
import pytest
import torch

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
    manifest = __import__("json").loads((tmp_path / "export" / "charlm.json").read_text())
    assert manifest["arch"] == arch
    assert manifest["restricted_to"] == 5  # four characters plus <eos>
    prefill = ort.InferenceSession(str(tmp_path / "export" / "prefill.onnx"))
    step = ort.InferenceSession(str(tmp_path / "export" / "charlm.onnx"))
    prelude = [BOS, *_tokens("你", "好"), SEP]
    beams = (_tokens("我", "吗"), _tokens("你", "好"))
    outputs = prefill.run(None, {"tokens": np.array([prelude], dtype=np.int64)})
    names = ["log_probs", *manifest["prefix"], *manifest["state"]]
    by_name = dict(zip(names, outputs, strict=True))
    prefix = {name: by_name[name] for name in manifest["prefix"]}
    state = {name: np.repeat(by_name[name], 2, axis=0) for name in manifest["state"]}
    for position in range(2):
        token = np.array([beam[position] for beam in beams], dtype=np.int64)
        outputs = step.run(None, {"token": token, **prefix, **state})
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
