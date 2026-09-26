"""Write a randomly initialised char-LM export at the real model's size.

Run from `python/`:

    uv run python scripts/charlm_sized.py <out dir>

The fixtures under `crates/ime-lm/tests/fixtures` are tiny, so nothing they
exercise reproduces the memory of the trained transformer. This builds a
`TransformerCharLm` at the trained configuration -- 12 layers, hidden 512, 8
heads, feed-forward 2048, the full alphabet -- seeds it, and exports it
unrestricted. `<out dir>` holds `charlm.json`, `prefill.onnx`, `charlm.onnx`
(about 250 MB each) and a `char_pinyin.tsv` whose characters are a subset of
the alphabet, so a `Lexicon` parsed from it opens the model.
"""

import itertools
import sys
import tempfile
from pathlib import Path

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
from mlime.train.charlm_vocab import SPECIALS

# The trained model's alphabet, 41,928 ids: the five reserved ids, then the
# lexicon's 41,923 characters. CJK code points from U+4E00 stand in for them,
# skipping the UTF-16 surrogate block a straight range would cross.
CHARACTERS = 41_923
CODE_POINTS = (c for c in itertools.count(0x4E00) if not 0xD800 <= c <= 0xDFFF)
VOCAB = CharVocab(chars=SPECIALS + tuple(chr(c) for c in itertools.islice(CODE_POINTS, CHARACTERS)))

# The trained configuration, measured at 60M parameters.
CONFIG = CharLmConfig(
    arch="transformer",
    embedding=512,
    hidden=512,
    layers=12,
    heads=8,
    feedforward=2048,
    max_positions=512,
    dropout=0.0,
    context_chars=62,
)

# `<char>\t<readings>` for the first slice of the alphabet, strictly sorted by
# codepoint; enough characters for a beam's random walks, all read "a" so the
# lexicon and the syllable table agree.
CHAR_PINYIN = "".join(f"{ch}\ta\n" for ch in VOCAB.chars[len(SPECIALS) : len(SPECIALS) + 512])


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "char_pinyin.tsv").write_text(CHAR_PINYIN, encoding="utf-8")
    torch.manual_seed(2)
    model = build(len(VOCAB), CONFIG)
    training = CharLmTraining(max_steps=10)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    progress = Progress(step=3, tokens_seen=100, positions=(Position(0, 1, 2),))
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "charlm-final.pt"
        save_checkpoint(checkpoint, model, VOCAB, progress, optimizer, scheduler, scaler, training)
        export_onnx(checkpoint, out)
    log.info("wrote sized export", dir=str(out))


if __name__ == "__main__":
    main()
