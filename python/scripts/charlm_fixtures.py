"""Write the `ime-lm` integration-test fixtures.

Run from `python/`:

    uv run python scripts/charlm_fixtures.py <out dir>

Builds a tiny LSTM and a tiny transformer checkpoint the way
`tests/test_charlm_model.py` does (same shapes, `torch.manual_seed`), exports
each restricted to a four-character list, and drives the exported graphs with
onnxruntime over one prelude and two beams of two tokens each. `<out
dir>/<arch>/` holds `charlm.json`, `prefill.onnx`, `charlm.onnx` and
`expected.json` (the context, the beams, the candidates each step was asked
for and their gathered scores, plus the full vocabulary rows the gather is
checked against), and `<out dir>/char_pinyin.tsv` is the character table the
Rust `Lexicon` parses.
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
from mlime.train.charlm_vocab import BOS, EOS, PAD, SEP, SPECIALS

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

# The candidates each produced state is asked to score, per call. The start
# answers the first position's candidates; a step's rows answer the
# following position's, so the last step asks only for `<eos>`. Uneven
# lengths exercise the [rows, K] padding.
START_ASKED = ("你好吗我", False)
STEP_ASKED = (
    (("很见谢", False), ("你吗", False)),
    (("", True), ("", True)),
)


def _asked(candidates: str, eos: bool, scores: np.ndarray) -> dict[str, object]:
    """One row's request and its gathered scores, keyed by what was asked.

    *scores* is the row's gathered row in the order the padded tensor
    named the ids; the record keeps them keyed by character (or ``<eos>``)
    so a caller never has to re-derive the sort. The Rust side sorts the
    request's ids together, ``<eos>`` among them, so this does too.
    """
    ids = sorted([VOCAB.index[ch] for ch in candidates] + ([EOS] if eos else []))
    asked = {"candidates": candidates, "eos": eos}
    return {
        **asked,
        "scores": {VOCAB.chars[i]: float(scores[slot]) for slot, i in enumerate(ids)},
    }


def _gathered(
    requests: tuple[tuple[str, bool], ...],
) -> tuple[np.ndarray, list[list[int]]]:
    """The ``candidates`` tensor for per-row ``(candidates, eos)`` requests.

    Returns the padded ``[rows, K]`` int64 array and, per row, the sorted
    alphabet ids it names.
    """
    per_row = [
        sorted([VOCAB.index[ch] for ch in candidates] + ([EOS] if eos else []))
        for candidates, eos in requests
    ]
    width = max(len(ids) for ids in per_row)
    columns = np.full((len(requests), width), PAD, dtype=np.int64)
    for row, ids in enumerate(per_row):
        columns[row, : len(ids)] = ids
    return columns, per_row


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


def _expected(export: Path) -> dict[str, object]:
    manifest = json.loads((export / "charlm.json").read_text())
    prefill = ort.InferenceSession(str(export / "prefill.onnx"))
    step = ort.InferenceSession(str(export / "charlm.onnx"))
    index = VOCAB.index
    prelude = [BOS, *[index[ch] for ch in CONTEXT], SEP]
    whole = np.arange(len(VOCAB), dtype=np.int64)

    def run_prefill(candidates: np.ndarray) -> dict[str, np.ndarray]:
        outputs = prefill.run(
            None,
            {"tokens": np.array([prelude], dtype=np.int64), "candidates": candidates},
        )
        return dict(zip((o.name for o in prefill.get_outputs()), outputs, strict=True))

    by_name = run_prefill(np.array([_gathered((START_ASKED,))[1][0]], dtype=np.int64))
    start = _asked(*START_ASKED, by_name["candidate_log_probs"][0])
    # The full row under the same prelude is what the gathered row is
    # checked against.
    prefill_full = run_prefill(whole.reshape(1, -1))["candidate_log_probs"][0]

    step_inputs = {item.name for item in step.get_inputs()}
    paged = "page_row" in step_inputs
    if paged:
        # The pages layout: the prefill emits one position per row, the
        # host writes each into a pool page, and every beam's prefix is
        # its chain of page ids. The beams share the prelude's pages.
        pk = by_name[manifest["prefix"][0]]
        layers, prelude_len = pk.shape[1], pk.shape[0]
        pool = {
            name: np.zeros((64, layers, *pk.shape[2:]), dtype=pk.dtype)
            for name in manifest["state"]
        }
        pool["keys"][1 : prelude_len + 1] = by_name["prefix_keys"]
        pool["values"][1 : prelude_len + 1] = by_name["prefix_values"]
        free_page = prelude_len + 1
        chains = [list(range(1, prelude_len + 1)) for _ in BEAMS]
    else:
        state = {name: np.repeat(by_name[name], len(BEAMS), axis=0) for name in manifest["state"]}

    def run_step(
        token: np.ndarray,
        candidates: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if paged:
            width = max(len(chain) for chain in chains)
            page_row = np.zeros((len(chains), width), dtype=np.int64)
            mask = np.zeros((len(chains), width), dtype=bool)
            for row, chain in enumerate(chains):
                page_row[row, : len(chain)] = chain
                mask[row, : len(chain)] = True
            outputs = step.run(
                None,
                {
                    "token": token,
                    "page_row": page_row,
                    "mask": mask,
                    "candidates": candidates,
                    **pool,
                    "state_keys": np.zeros(
                        (1, layers, pk.shape[2], width, pk.shape[3]), dtype=pk.dtype
                    ),
                    "state_values": np.zeros(
                        (1, layers, pk.shape[2], width, pk.shape[3]), dtype=pk.dtype
                    ),
                    "source_row": np.full(len(chains), -1, dtype=np.int64),
                },
            )
        else:
            outputs = step.run(
                None,
                {
                    "token": token,
                    "candidates": candidates,
                    "source_row": np.arange(len(BEAMS), dtype=np.int64),
                    **state,
                },
            )
        return dict(zip((o.name for o in step.get_outputs()), outputs, strict=True))

    steps = []
    full = []
    for position in range(len(BEAMS[0])):
        token = np.array([index[beam[position]] for beam in BEAMS], dtype=np.int64)
        requests = STEP_ASKED[position]
        candidates, _ = _gathered(requests)
        outputs = run_step(token, candidates)
        gathered = outputs["candidate_log_probs"]
        steps.append([_asked(*asked, gathered[row]) for row, asked in enumerate(requests)])
        # The same step again with the whole alphabet asked: its columns are
        # the row a full-vocabulary export would have returned.
        whole_row = run_step(token, np.tile(whole, (len(requests), 1)))["candidate_log_probs"]
        full.append(whole_row.tolist())
        if paged:
            for row, chain in enumerate(chains):
                for name in manifest["state"]:
                    pool[name][free_page] = outputs[f"next_{name}"][row]
                chain.append(free_page)
                free_page += 1
        else:
            state = {name: outputs[f"next_{name}"] for name in manifest["state"]}
    return {
        "arch": manifest["arch"],
        "context": CONTEXT,
        "beams": list(BEAMS),
        "start": start,
        "steps": steps,
        "prefill_full": prefill_full.tolist(),
        "full": full,
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
            # The fp16 variant exercises the loader's second dtype; its
            # expected.json is what the fp16 graphs themselves produce.
            export_onnx(checkpoint, out / f"{arch}-fp16", restrict, quantize="fp16")
            # The int8 variant is what the batch-independence test runs: its
            # scores must not move when a row shares its batch with strangers.
            export_onnx(checkpoint, out / f"{arch}-int8", restrict, quantize="int8")
        for variant in (arch, f"{arch}-fp16", f"{arch}-int8"):
            (out / variant / "expected.json").write_text(
                json.dumps(_expected(out / variant), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            log.info("wrote fixture", dir=str(out / variant))


if __name__ == "__main__":
    main()
