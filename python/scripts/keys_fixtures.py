"""Write the key-level model's `ime-neural` integration-test fixtures.

Run from the repository root:

    uv run --project python python/scripts/keys_fixtures.py crates/ime-neural/tests/fixtures-keys

Builds a tiny seeded `KeyE2EModel` over the nine characters of
`routea_fixtures.py`, and exports it with `export_key_model` three ways, each
into its own subdirectory: `fp32/`, `int8-weights/` (dynamic int8 context and
keys, weight-only int8 decoder) and `int8-dynamic/` (dynamic int8 throughout).
`<out>/expected.json` is computed over the fp32 export with onnxruntime
through `OnnxKeyModel`, `OnnxTransition` and the real `search`; for each probe
(a key string, its context and whether its tail may be unfinished) it holds:

* the legality lattice and, for outputs that more than one cover spells, those
  covers;
* the `keys.onnx` CTC log-probs (the whole table, blank last) and the sums,
  absolute sums and sampled values of `cross_keys` and `cross_values`;
* a scripted sequence of decoder calls (`decoder_calls`): the exact inputs of
  each `decoder.onnx` run -- the first carries a parent and its child in one
  call through `tree_mask`, the second reads paged history, the third is a cold
  deep chain -- the log-probs it returns, the sums and samples of the rows it
  appends, and the page each new prefix claims. Pages are claimed in row order
  from page 1 and page 0 is the masked scratch, so a consumer replays the
  sequence on its own pool;
* the search's top-8 at beta 1, gamma 0.5, beam 8 for `sum` and `viterbi`,
  with the prefixes each transition call was asked for (one call per key
  offset that has any).

`int8` holds the same search over each int8 export, with the fp32 export's
difference recorded beside it: nothing asserts it, since the quantized
kernels' answers differ between platforms, and a consumer checks its own int8
run against these values at a tolerance of its own.

Written into `crates/ime-neural/tests/fixtures-keys`, the text files (the
manifests, the tokenizers, `expected.json`) are committed; the `.onnx` and
`.weights` files are the ones `scripts/publish_fixtures.py` uploads. With
`--graphs <dir>` after the directory they are written under `<dir>` at the
same relative paths instead of next to the text files.
"""

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from routea_fixtures import READINGS, TINY, FixtureTokenizer
from transformers import BertConfig

from mlime.legality import LIVE, OFFLINE, KeyLattice, Legality, LegalityOptions
from mlime.logging import log
from mlime.train.arbitration import ReadingArbitration
from mlime.train.charlm import CharLmConfig
from mlime.train.charlm_vocab import SPECIALS, CharVocab
from mlime.train.e2e_model import E2EConfig, emittable_ids
from mlime.train.key_export import export_key_model
from mlime.train.key_model import KeyE2EModel
from mlime.train.key_onnx import KeyEncoding, OnnxKeyModel, OnnxTransition
from mlime.train.key_search import Hypothesis, search
from mlime.train.keys import KeyFrontConfig
from mlime.train.lexicon import Lexicon, build_lexicon
from mlime.train.run import Vocabularies
from mlime.train.samples import DEFAULT_CONTEXT_TOKENS
from mlime.train.spans import SpanVocab
from mlime.typo import load_syllables

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "crates/ime-pinyin/data"

#: The suffixes that leave the tree; `ime-fixtures` and `publish_fixtures.py`
#: use the same ones.
FETCHED = (".onnx", ".weights")

#: The variants: directory, `quantize`, `decoder_quantize`.
VARIANTS: tuple[tuple[str, Literal["int8"] | None, Literal["weights", "dynamic"]], ...] = (
    ("fp32", None, "weights"),
    ("int8-weights", "int8", "weights"),
    ("int8-dynamic", "int8", "dynamic"),
)

#: What the search runs at, for both combinations of covers.
BETA = 1.0
GAMMA = 0.5
BEAM = 8

#: The typed strings the fixtures are recorded over, as `(keys, context,
#: unfinished_tail)`. `zw` abbreviates two characters (`zhong`, `wo`), and
#: `zhonghe` has two covers that spell one output (`zhong|he` and `zhon|ghe`),
#: so `sum` and `viterbi` differ; `woaibj` mixes the two styles, `wobj` has no
#: context, and `zhongzh` ends in an unfinished syllable.
PROBES = (
    ("zw", "钟", False),
    ("zhonghe", None, False),
    ("woaibj", "北京", False),
    ("wobj", None, False),
    ("zhongzh", "钟", True),
)

#: The decoder calls scripted per probe, as the prefixes asked for in each
#: (emission indices over the sorted nine characters): a chain whose parent
#: and child share a call, history that has pages, and a cold deep chain.
DECODER_CALLS = (
    ((4, 5),),
    ((4, 5, 0), (1,)),
    ((2, 3, 7, 8),),
)

#: How many leading and trailing values of a tensor a record samples.
SAMPLES = 6

#: The reader the decoder is built from, as the unit tests' tiny one.
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

#: The key front over the fixture's tower: the joint stage's.
KEYS = KeyFrontConfig(layers=2, max_keys=62, transition_weight=0.5)

#: The tower is the route A fixture's, with the two sentinels' positions
#: for `max_keys` keys.
TOWER: BertConfig = TINY

Floats = npt.NDArray[np.float32]


class KeyFixtureTokenizer(FixtureTokenizer):
    """The route A fixture tokenizer, answering the collator's call with padded rows."""

    def __call__(  # type: ignore[override]
        self, text: list[str], padding: bool, truncation: bool, max_length: int
    ) -> dict[str, list[list[int]]]:
        """Encode each string with its sentinels, padded to the longest."""
        encoded = self.inner.encode_batch(list(text), add_special_tokens=True)
        width = max(len(item.ids) for item in encoded)
        return {
            "input_ids": [
                item.ids + [self.pad_token_id] * (width - len(item.ids)) for item in encoded
            ],
            "attention_mask": [
                item.attention_mask + [0] * (width - len(item.ids)) for item in encoded
            ],
        }


@dataclass(frozen=True)
class Setup:
    """What the model and the searches over it are built from."""

    spans: SpanVocab
    lexicon: Lexicon
    legality: Legality
    vocabularies: Vocabularies
    tokenizer: KeyFixtureTokenizer
    model: KeyE2EModel


def setup() -> Setup:
    """The seeded tiny model, with the tables and tokenizer it is exported over."""
    spans = SpanVocab.load()
    lexicon = build_lexicon(
        READINGS, {c: index + 100 for index, c in enumerate(sorted(READINGS))}, spans
    )
    # The context tower reads the lattice's characters and the contexts'.
    characters = sorted(READINGS) + sorted(set("北京大学") - set(READINGS))
    tokenizer = KeyFixtureTokenizer(tuple(characters))
    vocabularies = Vocabularies(
        spans=spans,
        tokenizer=tokenizer,
        context_tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=ReadingArbitration.load(spans),
        syllables=frozenset(load_syllables(DATA / "syllables.txt")),
    )
    reader_vocab = CharVocab(chars=SPECIALS + tuple(lexicon.characters))
    torch.manual_seed(0)
    model = KeyE2EModel.compose(
        TOWER,
        lexicon,
        E2EConfig(cross_attention_layers=1, decoder=READER, keys=KEYS),
        len(reader_vocab),
        emittable_ids(lexicon, reader_vocab),
    )
    with torch.no_grad():
        # A trained run's gates are open, and its blank is not zero: at zero
        # the decoder would not read the keys and every blank would be flat.
        for layer in (*model.gated_layers(), *model.decoder.gated_layers()):
            layer.gate.fill_(0.5)
        torch.nn.init.normal_(model.blank.weight, std=0.5)
        torch.nn.init.normal_(model.blank.bias, std=0.5)
    model.eval()
    return Setup(spans, lexicon, vocabularies.legality(), vocabularies, tokenizer, model)


class RecordedModel(OnnxKeyModel):
    """An export whose decoder runs are kept, with their inputs, for the fixture."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        self.runs: list[dict[str, Any]] = []

    def step(
        self,
        encoding: KeyEncoding,
        token: npt.NDArray[np.int64],
        position: npt.NDArray[np.int64],
        page_row: npt.NDArray[np.int64],
        page_mask: npt.NDArray[np.bool_],
        tree_mask: npt.NDArray[np.bool_],
        pool_keys: Floats,
        pool_values: Floats,
    ) -> tuple[Floats, Floats, Floats]:
        outputs = super().step(
            encoding, token, position, page_row, page_mask, tree_mask, pool_keys, pool_values
        )
        log_probs, next_keys, next_values = outputs
        self.runs.append(
            {
                "token": token.tolist(),
                "position": position.tolist(),
                "page_row": page_row.tolist(),
                "page_mask": page_mask.astype(int).tolist(),
                "tree_mask": tree_mask.astype(int).tolist(),
                "pool_pages": int(pool_keys.shape[0]),
                "log_probs": floats(log_probs),
                "next_keys": summary(next_keys),
                "next_values": summary(next_values),
            }
        )
        return outputs


class RecordedTransition:
    """A transition that keeps the prefixes each of its calls was asked for."""

    def __init__(self, inner: OnnxTransition) -> None:
        self.inner = inner
        self.queried: list[list[list[int]]] = []

    def __call__(self, records: Any, prefixes: Any) -> Floats:
        self.queried.append([list(prefix) for prefix in prefixes])
        return self.inner(records, prefixes)


def floats(array: Floats) -> Any:
    """*array* as nested lists of the float32 values, widened exactly."""
    return array.astype(np.float64).tolist()


def summary(array: Floats) -> dict[str, Any]:
    """The checksums and sampled values of a tensor: what a consumer compares."""
    flat = array.astype(np.float64).reshape(-1)
    return {
        "shape": list(array.shape),
        "sum": float(flat.sum()),
        "abs_sum": float(np.abs(flat).sum()),
        "head": flat[:SAMPLES].tolist(),
        "tail": flat[-SAMPLES:].tolist(),
    }


def covers(lattice: KeyLattice, characters: tuple[str, ...]) -> dict[str, list[list[int]]]:
    """The outputs more than one cover of the lattice spells, with each one's cuts."""
    by_start: dict[int, list[tuple[int, int]]] = {}
    for start, end, character in zip(lattice.start, lattice.end, lattice.character, strict=True):
        by_start.setdefault(int(start), []).append((int(end), int(character)))
    found: dict[str, list[list[int]]] = {}

    def walk(at: int, text: str, cuts: list[int]) -> None:
        if at == lattice.keys:
            found.setdefault(text, []).append(list(cuts))
            return
        for end, character in sorted(by_start.get(at, [])):
            walk(end, text + characters[character], [*cuts, end])

    walk(0, "", [])
    return {text: cuts for text, cuts in sorted(found.items()) if len(cuts) > 1}


def hypothesis(found: Hypothesis, characters: tuple[str, ...]) -> dict[str, Any]:
    """One search output as the fixture records it."""
    return {
        "ids": list(found.ids),
        "text": "".join(characters[index] for index in found.ids),
        "cuts": list(found.cuts),
        "score": found.score,
        "ctc": found.ctc,
        "transition": found.transition,
        "correction": found.correction,
    }


def decode(
    model: OnnxKeyModel,
    encoding: KeyEncoding,
    lattice: KeyLattice,
    characters: tuple[str, ...],
) -> dict[str, Any]:
    """The search's top-8 for both combinations of covers, with the prefixes it queried."""
    found: dict[str, Any] = {}
    for name, viterbi in (("sum", False), ("viterbi", True)):
        transition = RecordedTransition(OnnxTransition(model, [encoding]))
        results = search(
            [lattice],
            torch.from_numpy(encoding.log_probs).unsqueeze(0),
            transition,
            beta=BETA,
            gamma=GAMMA,
            beam=BEAM,
            top_k=BEAM,
            viterbi=viterbi,
        )[0]
        found[name] = {
            "hypotheses": [hypothesis(item, characters) for item in results],
            "queried": transition.queried,
        }
    return found


def decoder_calls(directory: Path, encoding: KeyEncoding) -> list[dict[str, Any]]:
    """The scripted decoder calls over *encoding*, each with its inputs and outputs."""
    model = RecordedModel(directory)
    transition = OnnxTransition(model, [encoding])
    calls = []
    for asked in DECODER_CALLS:
        before = len(model.runs)
        known = set(transition.pages[0])
        transition([0] * len(asked), [tuple(prefix) for prefix in asked])
        (run,) = model.runs[before:]
        run["queried"] = [list(prefix) for prefix in asked]
        run["claimed"] = [
            {"prefix": list(prefix), "page": page}
            for prefix, page in transition.pages[0].items()
            if prefix not in known
        ]
        calls.append(run)
    return calls


def difference(fp32: Floats, other: Floats) -> float:
    """The largest absolute difference between two same-shaped tensors."""
    return float(np.abs(fp32.astype(np.float64) - other.astype(np.float64)).max())


def probe_records(
    directory: Path, setup_: Setup, reference: dict[str, KeyEncoding] | None
) -> tuple[list[dict[str, Any]], dict[str, KeyEncoding]]:
    """Every probe through the export in *directory*; the encodings by probe, for a later diff."""
    from mlime.train.key_eval import build_key_collator

    model = OnnxKeyModel(directory)
    characters = tuple(setup_.lexicon.characters)
    collator = build_key_collator(
        setup_.spans,
        setup_.lexicon,
        setup_.tokenizer,
        setup_.tokenizer,
        with_context=True,
        max_context_tokens=DEFAULT_CONTEXT_TOKENS,
    )
    records = []
    encodings = {}
    for keys, context, tail in PROBES:
        options = LegalityOptions(unfinished_tail=tail, correct_syllables=KEYS.correct_syllables)
        if options not in (OFFLINE, LIVE):
            raise ValueError(f"the probes assume the offline or live options, got {options}")
        lattice = setup_.legality.lattice(keys, options)
        encoding = model.encode(collator.inputs([keys], [context]))
        encodings[keys] = encoding
        record: dict[str, Any] = {
            "keys": keys,
            "context": context,
            "unfinished_tail": tail,
            "lattice": [
                [int(s), int(e), int(c), float(w)]
                for s, e, c, w in zip(
                    lattice.start, lattice.end, lattice.character, lattice.cost, strict=True
                )
            ],
            "same_output_covers": covers(lattice, characters),
            "ctc_log_probs": floats(encoding.log_probs),
            "cross_keys": summary(encoding.cross_keys),
            "cross_values": summary(encoding.cross_values),
            "search": decode(model, encoding, lattice, characters),
        }
        if reference is None:
            record["decoder_calls"] = decoder_calls(directory, encoding)
        else:
            base = reference[keys]
            record["against_fp32"] = {
                "ctc_max_abs": difference(base.log_probs, encoding.log_probs),
                "cross_keys_max_abs": difference(base.cross_keys, encoding.cross_keys),
                "cross_values_max_abs": difference(base.cross_values, encoding.cross_values),
            }
        records.append(record)
    return records, encodings


def agreement(fp32: list[dict[str, Any]], int8: list[dict[str, Any]]) -> None:
    """Record, on each int8 probe, how its search differs from the fp32 one."""
    for base, other in zip(fp32, int8, strict=True):
        difference_ = other["against_fp32"]
        for name in ("sum", "viterbi"):
            ours = other["search"][name]["hypotheses"]
            theirs = base["search"][name]["hypotheses"]
            difference_[name] = {
                "top1_same": ours[0]["ids"] == theirs[0]["ids"]
                if ours and theirs
                else not (ours or theirs),
                "top8_same_ids": [item["ids"] for item in ours] == [item["ids"] for item in theirs],
                "top1_score_abs": abs(ours[0]["score"] - theirs[0]["score"])
                if ours and theirs
                else None,
            }


def manifest_checks(fp32: list[dict[str, Any]]) -> None:
    """Refuse a fixture whose probes lost what they were chosen for."""
    if not any(record["same_output_covers"] for record in fp32):
        raise ValueError("no probe has two covers spelling one output")
    for record in fp32:
        if not record["search"]["sum"]["hypotheses"]:
            raise ValueError(f"{record['keys']} has no complete cover")


def write_exports(setup_: Setup, committed: Path, graphs: Path) -> list[Path]:
    """Export every variant under *graphs*, and copy its text files into *committed*."""
    written = []
    for name, quantize, decoder_quantize in VARIANTS:
        target = graphs / name
        if target.exists():
            shutil.rmtree(target)
        export_key_model(
            setup_.model,
            7,
            setup_.lexicon,
            setup_.legality,
            setup_.tokenizer,
            target,
            quantize,
            decoder_quantize,
        )
        log.info("exported fixture", variant=name)
        written += sorted(path for path in target.rglob("*") if path.suffix in FETCHED)
        if committed != graphs:
            (committed / name).mkdir(parents=True, exist_ok=True)
            for path in sorted(target.iterdir()):
                if path.suffix not in FETCHED:
                    shutil.copy(path, committed / name / path.name)
    return written


def generate(committed: Path, graphs: Path) -> list[Path]:
    """Write the fixture set; returns the files that leave the tree, with their paths."""
    committed.mkdir(parents=True, exist_ok=True)
    graphs.mkdir(parents=True, exist_ok=True)
    setup_ = setup()
    written = write_exports(setup_, committed, graphs)

    fp32, encodings = probe_records(graphs / "fp32", setup_, None)
    manifest_checks(fp32)
    expected: dict[str, Any] = {
        "beta": BETA,
        "gamma": GAMMA,
        "beam": BEAM,
        "characters": list(setup_.lexicon.characters),
        "fp32": fp32,
        "int8": {},
    }
    for name, quantize, _ in VARIANTS:
        if quantize is None:
            continue
        records, _ = probe_records(graphs / name, setup_, encodings)
        agreement(fp32, records)
        expected["int8"][name] = records
    (committed / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return written


def main() -> None:
    """Write the fixture directory, and print the pins of the files that leave the tree."""
    arguments = sys.argv[1:]
    if len(arguments) not in (1, 3) or arguments[1:2] not in ([], ["--graphs"]):
        raise SystemExit(f"usage: {sys.argv[0]} <fixture dir> [--graphs <graph dir>]")
    committed = Path(arguments[0])
    graphs = Path(arguments[2]) if len(arguments) == 3 else committed
    written = generate(committed, graphs)
    pins = {
        path.relative_to(graphs).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in written
    }
    sys.stdout.write(json.dumps(pins, indent=2) + "\n")
    log.info(
        "wrote fixture",
        dir=str(committed),
        graphs=str(graphs),
        graph_bytes=sum(path.stat().st_size for path in written),
    )


if __name__ == "__main__":
    main()
