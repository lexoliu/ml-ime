"""Write the `ime-neural` integration-test fixtures.

Run from the repository root:

    uv run --project python python/scripts/routea_fixtures.py crates/ime-neural/tests/fixtures

Builds a tiny `RouteAModel` the way `tests/test_routea_export.py` does (same
config, `torch.manual_seed`), over a lexicon of nine characters, and exports
it fp32 and int8 to `<out>/fp32` and `<out>/int8`. The lattice the export is
driven over is the real one: the script writes a two-record `eval.jsonl`,
runs `ime-cli emit-lattice` on it, and saves the resulting `lattice.jsonl`.
`<out>/expected.json` holds, per record, the raw fp32 score table the
exported graphs produce -- computed with onnxruntime, exactly what
`RouteA::emission` must return. No int8 table is recorded: the quantized
kernels' answers differ between platforms, so the Rust test checks the int8
export against the fp32 table at the bound `test_int8_export_tracks_fp32`
uses. `<out>/char_pinyin.tsv` is the character
table the Rust `Lexicon` parses, and `<out>/emittable.txt` the set
`Emittable::parse` reads.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig

from mlime.logging import log
from mlime.train.lexicon import build_lexicon
from mlime.train.model import RouteAConfig, RouteAModel
from mlime.train.routea import export_onnx
from mlime.train.samples import DEFAULT_CONTEXT_TOKENS, context_tail
from mlime.train.spans import SpanVocab

REPO = Path(__file__).resolve().parents[2]

# The test lexicon's readings, shared with `tests/conftest.py`.
READINGS = {
    "中": ("zhong",),
    "钟": ("zhong",),
    "重": ("zhong", "chong"),
    "我": ("wo",),
    "爱": ("ai",),
    "绿": ("lv", "lu"),
    "北": ("bei",),
    "京": ("jing",),
    "和": ("he", "hu", "huo"),
}

# `<char>\t<readings>`, strictly sorted by codepoint, in the layout
# `Lexicon::parse` reads.
CHAR_PINYIN = """\
中\tzhong
京\tjing
北\tbei
和\the,hu,huo
我\two
爱\tai
绿\tlv,lu
重\tzhong,chong
钟\tzhong
"""

# The eval set the fixture's lattice is emitted from: one record with a
# context, one without, so both `has_context` paths are exercised.
EVAL_SET = """\
{"pinyin":"zhongwo","text":"钟我","context":"北京大学"}
{"pinyin":"woaibeijing","text":"我爱北京","context":null}
"""

TINY = BertConfig(
    vocab_size=256,
    hidden_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
)


class FixtureTokenizer:
    """The context tower's vocabulary over just the fixture's characters.

    A `tokenizers.Tokenizer` with a `BertPreTokenizer`, so each CJK character
    is one token -- the shape the real MacBERT vocabulary produces -- plus a
    `BaseTokenizer`-compatible `__call__` and the `save` the export calls.
    """

    def __init__(self, characters: tuple[str, ...]) -> None:
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *characters]
        inner = Tokenizer(models.WordLevel({v: i for i, v in enumerate(vocabulary)}, "[UNK]"))
        # The real MacBERT pipeline: a normalizer isolates each CJK character
        # so every one is a token, as WordPiece splitting produces.
        inner.normalizer = normalizers.BertNormalizer(
            clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True
        )
        inner.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        inner.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", inner.token_to_id("[CLS]")),
                ("[SEP]", inner.token_to_id("[SEP]")),
            ],
        )
        inner.enable_truncation(max_length=DEFAULT_CONTEXT_TOKENS)
        self.inner = inner
        self.pad_token_id = inner.token_to_id("[PAD]")
        self.cls_token_id = inner.token_to_id("[CLS]")
        self.sep_token_id = inner.token_to_id("[SEP]")
        self.mask_token_id = inner.token_to_id("[MASK]")

    def __call__(
        self,
        texts,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 0,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Encode the texts the way the export's dummy inputs are built."""
        encoded = self.inner.encode_batch(list(texts), add_special_tokens=True)
        ids = torch.tensor([item.ids for item in encoded], dtype=torch.long)
        mask = torch.tensor([item.attention_mask for item in encoded], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def save(self, path: str) -> None:
        """Write `tokenizer.json`, the file the Rust loader reads."""
        Path(path).write_text(self.inner.to_str(pretty=True) + "\n", encoding="utf-8")


def _score_record(
    tokenizer: FixtureTokenizer,
    context_session: ort.InferenceSession,
    fill_session: ort.InferenceSession,
    record: dict,
    with_context: bool,
    span_ids: dict[str, int],
    emissions: dict[str, int],
) -> list[list[list[float]]]:
    """One record's score table, computed over the exported graphs.

    This mirrors `RouteA::emission`: the context tower once, the fill decoder
    once over the record's readings, and the masked log probabilities read at
    each asked candidate's emission index.
    """
    paths = record["paths"]
    rows = len(paths)
    width = max(len(path["spans"]) for path in paths) + 2

    input_ids = np.full((rows, width), tokenizer.pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((rows, width), dtype=np.int64)
    span_ids_array = np.zeros((rows, width), dtype=np.int64)
    span_positions = np.zeros((rows, width), dtype=np.bool_)
    asked: list[list[list[int]]] = []
    for row, path in enumerate(paths):
        input_ids[row, 0] = tokenizer.cls_token_id
        input_ids[row, len(path["spans"]) + 1] = tokenizer.sep_token_id
        input_ids[row, 1 : len(path["spans"]) + 1] = tokenizer.mask_token_id
        attention_mask[row, : len(path["spans"]) + 2] = 1
        positions = []
        for position, (span, candidates) in enumerate(
            zip(path["spans"], path["candidates"], strict=True)
        ):
            span_ids_array[row, position + 1] = span_ids[span]
            span_positions[row, position + 1] = True
            positions.append([emissions[character] for character in candidates])
        asked.append(positions)

    has = with_context and record["context"] is not None
    if has:
        # The tail cut and truncation the loader applies in Rust.
        encoded = tokenizer.inner.encode(
            context_tail(record["context"], DEFAULT_CONTEXT_TOKENS), add_special_tokens=True
        )
        context_ids = np.asarray([encoded.ids], dtype=np.int64)
        context_mask = np.asarray([encoded.attention_mask], dtype=np.int64)
        hidden = context_session.run(
            None, {"context_ids": context_ids, "context_mask": context_mask}
        )[0]
    else:
        # The gate zeroes the tower's contribution, so the row is zeros.
        hidden = np.zeros((1, 1, TINY.hidden_size), dtype=np.float32)
        context_mask = np.zeros((1, 1), dtype=np.int64)
    context = np.broadcast_to(hidden, (rows, *hidden.shape[1:])).copy()
    context_masks = np.broadcast_to(context_mask, (rows, context_mask.shape[1])).copy()
    has_context = np.full(rows, float(has), dtype=np.float32)

    log_probs = fill_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_ids": span_ids_array,
            "span_positions": span_positions,
            "context": context,
            "context_mask": context_masks,
            "has_context": has_context,
        },
    )[0]
    scores = []
    for row, positions in enumerate(asked):
        positions_scores = []
        for position, emissions_row in enumerate(positions):
            positions_scores.append(
                [float(log_probs[row, position + 1, emission]) for emission in emissions_row]
            )
        scores.append(positions_scores)
    return scores


def main() -> None:
    """Write the fixture directory."""
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "char_pinyin.tsv").write_text(CHAR_PINYIN, encoding="utf-8")
    (out / "emittable.txt").write_text(
        "".join(f"{c}\n" for c in sorted(READINGS)), encoding="utf-8"
    )
    (out / "eval.jsonl").write_text(EVAL_SET, encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        lattice_path = Path(tmp) / "lattice.jsonl"
        subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "-p",
                "ime-cli",
                "--",
                "emit-lattice",
                "--eval-set",
                str(out / "eval.jsonl"),
                "--emittable",
                str(out / "emittable.txt"),
                "--out",
                str(lattice_path),
            ],
            check=True,
            cwd=REPO,
        )
        lattice_lines = lattice_path.read_text(encoding="utf-8")
    lattice = [json.loads(line) for line in lattice_lines.splitlines()]
    (out / "lattice.jsonl").write_text(lattice_lines, encoding="utf-8")

    spans = SpanVocab.load()
    # The tokenizer's vocabulary covers the lattice's characters and the
    # context's -- `北京大学` is the record-0 context.
    characters = sorted(READINGS) + sorted(set("北京大学") - set(READINGS))
    tokenizer = FixtureTokenizer(tuple(characters))
    lexicon = build_lexicon(
        READINGS,
        {character: index + 100 for index, character in enumerate(sorted(READINGS))},
        spans,
    )
    torch.manual_seed(0)
    model = RouteAModel.from_config(
        TINY, lexicon, RouteAConfig(cross_attention_layers=2, cross_attention_dropout=0.0)
    )
    with torch.no_grad():
        for layer in model.gated_layers():
            layer.gate.fill_(0.4)
    model.eval()

    for variant, quantize in (("fp32", None), ("int8", "int8")):
        export_onnx(model, 7, lexicon, spans, tokenizer, out / variant, quantize=quantize)
        log.info("exported fixture", variant=variant)

    span_ids = {span: index for index, span in enumerate(spans)}
    emissions = {character: index for index, character in enumerate(lexicon.characters)}
    context_session = ort.InferenceSession(str(out / "fp32" / "context.onnx"))
    fill_session = ort.InferenceSession(str(out / "fp32" / "fill.onnx"))
    expected = []
    for with_context in (True, False):
        for record in lattice:
            expected.append(
                {
                    "variant": "fp32",
                    "with_context": with_context,
                    "record": record["record"],
                    "paths": _score_record(
                        tokenizer,
                        context_session,
                        fill_session,
                        record,
                        with_context,
                        span_ids,
                        emissions,
                    ),
                }
            )
    (out / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("wrote fixture", dir=str(out))


if __name__ == "__main__":
    main()
