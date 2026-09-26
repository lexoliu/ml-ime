"""Write the `ime-lm` integration-test fixtures.

Run from `python/`:

    uv run python scripts/charlm_fixtures.py <out dir>

Builds a tiny LSTM and a tiny transformer checkpoint the way
`tests/test_charlm_model.py` does (same shapes, `torch.manual_seed`), exports
each restricted to a four-character list, and drives the exported graphs with
onnxruntime over one prelude and two beams of two tokens each. `<out
dir>/<arch>/` holds `charlm.json`, `prefill.onnx`, `charlm.onnx` and
`expected.json` (the context, the beams and every log-probability row), and
`<out dir>/char_pinyin.tsv` is the character table the Rust `Lexicon` parses.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mlime.logging import log
from mlime.train.charlm import (
    CharLmConfig,
    CharLmTraining,
    CharVocab,
    Position,
    Progress,
    export_onnx,
    save_checkpoint,
)
from mlime.train.charlm_model import build
from mlime.train.charlm_vocab import BOS, SEP, SPECIALS

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

# The export's emittable set: four characters, plus <eos> on export.
RESTRICTED = CHARS[:4]

# What the Rust test feeds `start` and `advance`.
CONTEXT = "你好"
BEAMS = ("我吗", "你好")

# `<char>\t<readings>`, strictly sorted by codepoint, covering the whole
# alphabet: the table `Lexicon::parse` reads.
CHAR_PINYIN = """\
你\tni
再\tzai
吗\tma
好\thao
很\then
我\two
见\tjian
谢\txie
"""


def _checkpoint(path: Path, arch: str) -> None:
    torch.manual_seed(2)
    model = build(len(VOCAB), CONFIGS[arch])
    training = CharLmTraining(max_steps=10)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress = Progress(step=3, tokens_seen=100, positions=(Position(0, 1, 2),))
    save_checkpoint(path, model, VOCAB, progress, optimizer, scheduler, scaler, training)


def _expected(export: Path) -> dict:
    manifest = json.loads((export / "charlm.json").read_text())
    prefill = ort.InferenceSession(str(export / "prefill.onnx"))
    step = ort.InferenceSession(str(export / "charlm.onnx"))
    index = VOCAB.index
    prelude = [BOS, *[index[ch] for ch in CONTEXT], SEP]
    outputs = prefill.run(None, {"tokens": np.array([prelude], dtype=np.int64)})
    names = ["log_probs", *manifest["prefix"], *manifest["state"]]
    by_name = dict(zip(names, outputs, strict=True))
    prefix = {name: by_name[name] for name in manifest["prefix"]}
    state = {name: np.repeat(by_name[name], len(BEAMS), axis=0) for name in manifest["state"]}
    steps = []
    for position in range(len(BEAMS[0])):
        token = np.array([index[beam[position]] for beam in BEAMS], dtype=np.int64)
        outputs = step.run(None, {"token": token, **prefix, **state})
        steps.append(outputs[0].tolist())
        state = dict(zip(manifest["state"], outputs[1:], strict=True))
    return {
        "arch": manifest["arch"],
        "context": CONTEXT,
        "beams": list(BEAMS),
        "prefill": by_name["log_probs"][0].tolist(),
        "steps": steps,
    }


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "char_pinyin.tsv").write_text(CHAR_PINYIN, encoding="utf-8")
    for arch in sorted(CONFIGS):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint = tmp_path / "charlm-final.pt"
            _checkpoint(checkpoint, arch)
            restrict = tmp_path / "emittable.txt"
            restrict.write_text("\n".join(RESTRICTED) + "\n", encoding="utf-8")
            export_onnx(checkpoint, out / arch, restrict)
        (out / arch / "expected.json").write_text(
            json.dumps(_expected(out / arch), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        log.info("wrote fixture", dir=str(out / arch))


if __name__ == "__main__":
    main()
