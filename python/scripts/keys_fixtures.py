"""Write the key-model fixture the Rust Core ML runner is held to (#266).

Run from the repository root, on macOS:

    uv run --project python python/scripts/keys_fixtures.py crates/ime-session/tests/fixtures/keys

The script builds a tiny randomly initialised joint-stage key model over eighteen real characters
(`torch.manual_seed`, so the weights are the same on every run), writes its bundle
(`mlime.train.key_bundle`), and converts, verifies and packages it with the separate project
`python/coreml` (`uv run --project python/coreml --no-sync mlime-coreml convert`, the way
`tests/test_key_coreml.py` does; run `uv sync --project python/coreml` once before). Beside the
converted models it writes `expected.json`, the Python Core ML search's answer for a small list of
typed inputs.

`<out>/model/` is the unpacked archive: `key-model.json`, `tokenizer.json`, and one directory per
function family -- `context/`, `keys/`, `decoder/` -- each a `.mlmodelc` and a `manifest.json`.
The compiled models differ in bytes from machine to machine, so they live off the tree
(`crates/ime-fixtures/fixtures.json` pins them, `scripts/publish_fixtures.py` uploads them) and
only `expected.json` is committed. `--converted <models dir>` skips the conversion and reads the
models of an earlier run instead, which is how `expected.json` is rewritten over the pinned models
alone.

`expected.json` holds, per input, the keys and the context text, and for each of the weight points
the search ran at: the tree calls the search made, in order, as bucket and node count, and the
top eight hypotheses with their ids, characters, cuts and score terms. The models run on the CPU
alone, so a hosted runner without a Neural Engine reproduces them; a hypothesis's score is
reproduced to 1e-4, which float16 on the CPU keeps. The inputs are the three ways people type --
full pinyin, first letters, and the mix -- one with a typo, with and without a context, and the
contexts are Chinese, Japanese and Korean text.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import torch
from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig

from mlime.legality import LegalityOptions
from mlime.logging import log
from mlime.train.arbitration import ReadingArbitration
from mlime.train.charlm_model import CharLmConfig
from mlime.train.charlm_vocab import SPECIALS, CharVocab
from mlime.train.e2e_model import E2EConfig, emittable_ids
from mlime.train.key_bundle import REFERENCE_PROBES, write_key_bundle
from mlime.train.key_coreml import CoreMlKeyDecoder, CoreMlKeyModel, Encoding, Half
from mlime.train.key_eval import (
    DEFAULT_BEAM,
    DEFAULT_TOP_K,
    KeyRecord,
    KeyWeights,
    build_key_collator,
)
from mlime.train.key_model import KeyE2EModel
from mlime.train.key_search import Hypothesis
from mlime.train.key_tree import TreeCall
from mlime.train.keys import KeyFrontConfig
from mlime.train.lexicon import build_lexicon, read_char_readings
from mlime.train.run import Vocabularies
from mlime.train.samples import DEFAULT_CONTEXT_TOKENS
from mlime.train.spans import SpanVocab
from mlime.typo import load_syllables

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "crates/ime-pinyin/data"

#: The step the bundle claims, which the manifest and the converted models carry.
STEP = 266

#: The emittable characters: those of the key engine's synthetic fixture, so the pinyin
#: automaton over them is exercised the same way, and 钟 and 中 whose readings overlap.
CHARACTERS = "我爱北京被经窝哎西安先现线县显鲜钟中"

#: Contexts in the three languages the test data covers, and the inputs asked of the model:
#: `(name, style, keys, context)`.
CHINESE = "今天我们一起去北京玩"
JAPANESE = "今日は東京に行きます"
KOREAN = "오늘은 서울에 갑니다"
INPUTS: tuple[tuple[str, str, str, str | None], ...] = (
    ("full-chinese", "full", "woaibeijing", CHINESE),
    ("full-no-context", "full", "woaibeijing", None),
    ("full-japanese", "full", "xianxianbeijing", JAPANESE),
    ("initials-japanese", "initials", "wabj", JAPANESE),
    ("initials-no-context", "initials", "wxbj", None),
    ("mixed-korean", "mixed", "woaibj", KOREAN),
    ("typo-korean", "typo", "woaibeijign", KOREAN),
)

#: The search's weight points: a sum and a Viterbi one, as `tests/test_key_coreml.py` runs.
GRID = (KeyWeights(1.0, 1.0, False), KeyWeights(0.7, 1.3, True))

#: The weights and positions the tiny model's towers are built with.
TOWER = BertConfig(
    vocab_size=256,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=128,
)
READER = CharLmConfig(
    arch="transformer",
    embedding=8,
    hidden=16,
    layers=2,
    heads=4,
    feedforward=32,
    max_positions=64,
    dropout=0.0,
)

#: The widest keys function holds 126 keys; the front's positions bound the model to that.
MAX_KEYS = 126


class FixtureTokenizer:
    """The base model's tokenizer over the fixture's characters, one token per character.

    A `tokenizers.Tokenizer` over a word-level vocabulary whose pre-tokenizer isolates every
    character (the real WordPiece vocabulary gives one token per CJK character and kana or
    hangul syllable it knows), with accents kept so a hangul syllable is not decomposed. Written
    as `tokenizer.json`, which the Rust runner loads. Anything outside the vocabulary is `[UNK]`.
    """

    def __init__(self, characters: str) -> None:
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *sorted(set(characters))]
        inner = Tokenizer(models.WordLevel({v: i for i, v in enumerate(vocabulary)}, "[UNK]"))
        inner.normalizer = normalizers.BertNormalizer(
            clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=True
        )
        inner.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Split(Regex("."), "isolated")]
        )
        inner.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", inner.token_to_id("[CLS]")),
                ("[SEP]", inner.token_to_id("[SEP]")),
            ],
        )
        inner.enable_truncation(max_length=DEFAULT_CONTEXT_TOKENS)
        self.inner = inner
        self.vocabulary = {token: index for index, token in enumerate(vocabulary)}
        self.pad_token_id = self.vocabulary["[PAD]"]
        self.cls_token_id = self.vocabulary["[CLS]"]
        self.sep_token_id = self.vocabulary["[SEP]"]
        self.mask_token_id = self.vocabulary["[MASK]"]

    def __call__(
        self, text: list[str], padding: bool, truncation: bool, max_length: int
    ) -> dict[str, list[list[int]]]:
        """Encode a batch the way the collator asks: sentinels added, padded to the longest."""
        if not truncation or max_length != DEFAULT_CONTEXT_TOKENS:
            raise ValueError("the fixture tokenizer truncates to the context budget")
        encoded = self.inner.encode_batch(list(text), add_special_tokens=True)
        width = max(len(item.ids) for item in encoded) if padding else 0
        ids, mask = [], []
        for item in encoded:
            fill = width - len(item.ids)
            ids.append([*item.ids, *([self.pad_token_id] * fill)])
            mask.append([*item.attention_mask, *([0] * fill)])
        return {"input_ids": ids, "attention_mask": mask}

    def save(self, path: str) -> None:
        """Write `tokenizer.json`, the file the Rust runner reads."""
        Path(path).write_text(self.inner.to_str(pretty=True) + "\n", encoding="utf-8")


class RecordingModel(CoreMlKeyModel):
    """The converted model, logging every tree call it runs as `(bucket, nodes)`."""

    def __init__(self, directory: Path, units: ct.ComputeUnit) -> None:
        super().__init__(directory, units)
        self.calls: list[tuple[int, int]] = []

    def tree(self, call: TreeCall, encoding: Encoding) -> Half:
        """Run *call*, noting its bucket and its node count."""
        self.calls.append((call.bucket, len(call.nodes)))
        return super().tree(call, encoding)


def build_stack() -> tuple[KeyE2EModel, Vocabularies, FixtureTokenizer]:
    """The tiny model, the tables it is indexed by, and its tokenizer."""
    readings = read_char_readings(DATA / "char_pinyin.tsv")
    missing = sorted(set(CHARACTERS) - set(readings))
    if missing:
        raise SystemExit(f"the character table has no reading for {missing}")
    context_text = "".join(
        [CHARACTERS, *(context or "" for _, _, _, context in INPUTS)]
        + [context or "" for _, _, context in REFERENCE_PROBES]
    )
    tokenizer = FixtureTokenizer("".join(c for c in context_text if not c.isspace()))
    spans = SpanVocab.load()
    lexicon = build_lexicon(
        {character: readings[character] for character in CHARACTERS},
        tokenizer.vocabulary,
        spans,
    )
    vocabularies = Vocabularies(
        spans=spans,
        tokenizer=tokenizer,
        context_tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=ReadingArbitration.load(spans),
        syllables=frozenset(load_syllables(DATA / "syllables.txt")),
    )
    vocab = CharVocab(chars=SPECIALS + tuple(lexicon.characters))
    config = E2EConfig(
        cross_attention_layers=1,
        decoder=READER,
        keys=KeyFrontConfig(layers=2, max_keys=MAX_KEYS, transition_weight=0.5),
    )
    torch.manual_seed(0)
    model = KeyE2EModel.compose(TOWER, lexicon, config, len(vocab), emittable_ids(lexicon, vocab))
    # A trained model's gates are open; at zero the context and the decoder's memory would not
    # reach the output at all, and the fixture would test nothing of them.
    with torch.no_grad():
        for layer in model.gated_layers():
            layer.gate.fill_(0.5)
        for layer in model.decoder.gated_layers():
            layer.gate.fill_(0.5)
    return model.eval(), vocabularies, tokenizer


def convert(bundle: Path, work: Path) -> Path:
    """Convert, verify and package *bundle* with `python/coreml`; the unpacked models directory.

    The tiny model's operations all place on the CPU, so the Neural Engine placement gate is off,
    as in `tests/test_key_coreml.py`.
    """
    done = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "--project",
            str(REPO / "python/coreml"),
            "mlime-coreml",
            "convert",
            "--bundle",
            str(bundle),
            "--work-dir",
            str(work),
            "--no-neural-engine",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if done.returncode != 0:
        raise SystemExit(f"mlime-coreml convert failed:\n{done.stderr[-4000:]}")
    [digest] = work.glob("mlime-keys-coreml-step*.tar.zst.sha256")
    log.info("converted", archive=digest.read_text().strip())
    return work / "models"


def hypothesis_record(vocabularies: Vocabularies, hypothesis: Hypothesis) -> dict[str, object]:
    """One hypothesis as the JSON the Rust test reads."""
    characters = vocabularies.lexicon.characters
    return {
        "ids": list(hypothesis.ids),
        "text": "".join(characters[index] for index in hypothesis.ids),
        "cuts": list(hypothesis.cuts),
        "score": hypothesis.score,
        "ctc": hypothesis.ctc,
        "transition": hypothesis.transition,
        "correction": hypothesis.correction,
    }


def expected(
    models: Path, vocabularies: Vocabularies, tokenizer: FixtureTokenizer
) -> dict[str, object]:
    """The Python Core ML search over *models*, CPU only, for every input and weight point."""
    model = RecordingModel(models, ct.ComputeUnit.CPU_ONLY)
    manifest = model.manifest
    legality = vocabularies.legality()
    options = LegalityOptions(unfinished_tail=False, correct_syllables=manifest.correct_syllables)
    decoder = CoreMlKeyDecoder(
        model=model,
        lexicon=vocabularies.lexicon,
        collator=build_key_collator(
            vocabularies.spans,
            vocabularies.lexicon,
            tokenizer,
            tokenizer,
            with_context=True,
            max_context_tokens=manifest.context_tokens,
        ),
        legality=legality,
        options=options,
        replay=False,
    )
    records = []
    for index, (name, style, keys, context) in enumerate(INPUTS):
        searches = []
        for point, weights in enumerate(GRID):
            model.calls.clear()
            record = KeyRecord(record=index, keys=keys, text="", context=context)
            [[found]] = decoder.decode([record], [weights])
            if not found:
                raise SystemExit(f"{name} at weights {point} has no hypothesis")
            searches.append(
                {
                    "weights": point,
                    "tree_calls": [
                        {"bucket": bucket, "nodes": nodes} for bucket, nodes in model.calls
                    ],
                    "hypotheses": [hypothesis_record(vocabularies, each) for each in found],
                }
            )
        records.append(
            {"name": name, "style": style, "keys": keys, "context": context, "searches": searches}
        )
    return {
        "policy_version": manifest.policy_version,
        "compute_units": "cpu-only",
        "legality": {
            "unfinished_tail": options.unfinished_tail,
            "correct_syllables": options.correct_syllables,
        },
        "beam": DEFAULT_BEAM,
        "top_k": DEFAULT_TOP_K,
        "weights": [
            {"beta": each.beta, "gamma": each.gamma, "viterbi": each.viterbi} for each in GRID
        ],
        "records": records,
    }


def main() -> None:
    """Write the fixture directory."""
    arguments = sys.argv[1:]
    converted: Path | None = None
    if len(arguments) == 3 and arguments[1] == "--converted":
        converted = Path(arguments[2]).resolve()
    elif len(arguments) != 1:
        raise SystemExit(f"usage: {sys.argv[0]} <fixture dir> [--converted <models dir>]")
    out = Path(arguments[0])
    out.mkdir(parents=True, exist_ok=True)

    model, vocabularies, tokenizer = build_stack()
    with tempfile.TemporaryDirectory() as scratch:
        if converted is None:
            bundle = Path(scratch) / "bundle"
            write_key_bundle(
                model, STEP, vocabularies.lexicon, vocabularies.legality(), tokenizer, bundle
            )
            converted = convert(bundle, Path(scratch) / "work")
        destination = out / "model"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(converted, destination)
    result = expected(destination, vocabularies, tokenizer)
    (out / "expected.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    log.info("wrote fixture", dir=str(out), records=len(INPUTS))


if __name__ == "__main__":
    main()
