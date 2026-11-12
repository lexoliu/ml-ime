"""The key-level model through Core ML: the converted tiny model against the PyTorch one (#266).

A tiny randomly initialised joint-stage model is written as a key bundle and converted by
``mlime-coreml convert`` (the separate project ``python/coreml/``, whose torch and coremltools pins
differ from this one's), which these tests run as a subprocess; the converted models are then run
through :mod:`mlime.train.key_coreml`. They need macOS and Core ML and skip elsewhere. The
tiny model's ops all place on the CPU, so the conversion runs without the Neural Engine gate; the
numbers here are Core ML's float16 against PyTorch's float32.

The transition's rows must be the model's. The decoder, with and without keystroke replay, must
return the PyTorch search's top-1 sentence, a disagreement being forgiven only where the reference
itself separates the two by less than a float16 error (and warned about, so the near-ties are
listed). The statistics must add up.
"""

from __future__ import annotations

import copy
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from conftest import READINGS
from test_e2e_model import TINY
from test_key_eval import (
    SHARE,
    decoder,
    eval_rows,
    verifier_fixture,  # noqa: F401 -- the fixture, by name
    write_eval_set,
)
from test_keys import (
    JOINT_KEYS,
    KeyTokenizer,
    joint_config,
    key_collator,
    open_gates,
    read_the_encoder,
)
from typer.testing import CliRunner

from mlime.__main__ import app
from mlime.legality import Legality
from mlime.train.arbitration import ReadingArbitration
from mlime.train.charlm_vocab import SPECIALS, CharVocab
from mlime.train.e2e_model import emittable_ids
from mlime.train.key_bundle import write_key_bundle
from mlime.train.key_eval import (
    DecodeTimes,
    KeyRecord,
    KeyRun,
    KeyWeights,
    ModelTransition,
    report_keys,
    typed_keys,
)
from mlime.train.key_model import KeyE2EModel, KeyStates
from mlime.train.key_search import LegalityVerifier, Prefix
from mlime.train.key_tree import closure_size
from mlime.train.lexicon import Lexicon, build_lexicon
from mlime.train.run import Vocabularies
from mlime.train.samples import letter_id
from mlime.train.spans import SpanVocab
from mlime.typo import load_syllables

if TYPE_CHECKING:
    from mlime.train.key_coreml import CoreMlKeyDecoder, CoreMlKeyModel, Encoding

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "crates/ime-pinyin/data"

darwin = pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runs on macOS")

#: Core ML's float16 log-probabilities against PyTorch's float32, absolute. The largest
#: difference measured on this model (M1, CPU placement) is 1.6e-2 over a tree row and 5.9e-4 over
#: a CTC row; this is about three times the larger. A wrong mask, bucket or weight moves a row by
#: whole nats.
LOG_PROB_ATOL = 5e-2

#: Two sentences the reference scores within this are a tie as far as float16 can say: the
#: measured score error (1.5e-2 below), rounded up. No record disagreed in the runs measured.
NEAR_TIE = 5e-2

#: A top-1 sentence's score against the reference's. Measured: at most 1.5e-2 over the 12 records
#: at both weight points, with and without replay.
SCORE_ATOL = 0.1

#: The share of (record, weights) searches that must return the reference's top-1 outright, so
#: forgiven near-ties cannot hide a decoder that returns something else.
AGREE_SHARE = 0.75

GRID = [KeyWeights(1.0, 1.0, False), KeyWeights(0.7, 1.3, True)]


@dataclass(frozen=True)
class Stack:
    """The tiny model and everything its decode is indexed by."""

    model: KeyE2EModel
    lexicon: Lexicon
    spans: SpanVocab
    legality: Legality
    vocabularies: Vocabularies


def wide_joint_model(lexicon: Lexicon, vocab: CharVocab) -> KeyE2EModel:
    """The joint-stage fixture with positions for the widest keys function (126 keys, width 128)."""
    tower = copy.deepcopy(TINY)
    tower.max_position_embeddings = 128
    torch.manual_seed(0)
    model = KeyE2EModel.compose(
        tower,
        lexicon,
        joint_config(replace(JOINT_KEYS, max_keys=126)),
        len(vocab),
        emittable_ids(lexicon, vocab),
    )
    open_gates(model)
    read_the_encoder(model)
    return model


@pytest.fixture(name="stack", scope="module")
def stack_fixture() -> Stack:
    spans = SpanVocab.load()
    vocabulary = {character: index + 100 for index, character in enumerate(sorted(READINGS))}
    lexicon = build_lexicon(READINGS, vocabulary, spans)
    tokenizer = KeyTokenizer()
    vocabularies = Vocabularies(
        spans=spans,
        tokenizer=tokenizer,
        context_tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=ReadingArbitration.load(spans),
        syllables=frozenset(load_syllables(DATA / "syllables.txt")),
    )
    vocab = CharVocab(chars=SPECIALS + tuple(lexicon.characters))
    return Stack(
        model=wide_joint_model(lexicon, vocab).eval(),
        lexicon=lexicon,
        spans=spans,
        legality=vocabularies.legality(),
        vocabularies=vocabularies,
    )


@pytest.fixture(name="converted", scope="module")
def converted_fixture(stack: Stack, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny bundle converted by ``mlime-coreml``.

    The models directory it leaves, with the bundle's manifest and tokenizer beside the models as
    its archive carries them.
    """
    work = tmp_path_factory.mktemp("coreml")
    bundle = work / "bundle"
    write_key_bundle(stack.model, 11, stack.lexicon, stack.legality, KeyTokenizer(), bundle)
    done = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "--project",
            str(ROOT / "python/coreml"),
            "mlime-coreml",
            "convert",
            "--bundle",
            str(bundle),
            "--work-dir",
            str(work),
            "--no-neural-engine",
            "--no-package",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return work / "models"


@pytest.fixture(name="engine", scope="module")
def engine_fixture(converted: Path) -> CoreMlKeyModel:
    from mlime.train.key_coreml import CoreMlKeyModel

    return CoreMlKeyModel(converted)


def reference_states(stack: Stack, keys: str, context: str | None) -> KeyStates:
    with torch.no_grad():
        return stack.model.key_states(
            key_collator(stack.spans, stack.lexicon).inputs([keys], [context])
        )


def coreml_encoding(
    engine: CoreMlKeyModel, stack: Stack, keys: str, context: str | None
) -> Encoding:
    inputs = key_collator(stack.spans, stack.lexicon).inputs([keys], [context])
    return engine.encode([letter_id(key) for key in keys], engine.context(inputs))


def random_prefixes(rng: random.Random, emitted: int, count: int, depth: int) -> list[Prefix]:
    return [
        tuple(rng.randrange(emitted) for _ in range(rng.randint(0, depth))) for _ in range(count)
    ]


# --- the rows -----------------------------------------------------------------------------


@darwin
@pytest.mark.parametrize(("keys", "context"), [("woaibeijing", "北京"), ("zhonghe", None)])
def test_the_transitions_rows_are_the_models_within_float16(
    engine: CoreMlKeyModel, stack: Stack, keys: str, context: str | None
) -> None:
    from mlime.train.key_coreml import CoreMlTransition, log_softmax

    states = reference_states(stack, keys, context)
    encoding = coreml_encoding(engine, stack, keys, context)
    with torch.no_grad():
        ctc = stack.model.ctc_log_probs(states.states[0]).numpy()
    assert np.abs(log_softmax(encoding.ctc_logits) - ctc).max() <= LOG_PROB_ATOL
    rng = random.Random(5)
    emitted = stack.lexicon.size
    sets: list[list[Prefix]] = [
        [()],
        [(0, 1, 2)],
        random_prefixes(rng, emitted, 8, 5),
        # A chain past the first bucket, and a set whose closure needs several calls.
        [tuple(rng.randrange(emitted) for _ in range(40))],
        random_prefixes(rng, emitted, 300, 12),
    ]
    for prefixes in sets:
        transition = CoreMlTransition(engine, encoding, DecodeTimes())
        got = transition([0] * len(prefixes), prefixes)
        with torch.no_grad():
            want = ModelTransition(stack.model, states, DecodeTimes())(
                [0] * len(prefixes), prefixes
            )
        assert got.shape == tuple(want.shape)
        assert np.abs(got - want.numpy()).max() <= LOG_PROB_ATOL
        stats = transition.finish()
        assert stats.answered == len(prefixes) and stats.queried == len(set(prefixes))
        assert stats.closure == closure_size(prefixes)
    # The packing is the policy's: 300 random prefixes of depth 12 need more than one call.
    assert stats.miss_calls > 1 and stats.tree_calls == stats.miss_calls


@darwin
def test_a_cached_row_is_answered_without_a_call_and_the_first_value_stays(
    engine: CoreMlKeyModel, stack: Stack
) -> None:
    from mlime.train.key_coreml import CoreMlTransition

    encoding = coreml_encoding(engine, stack, "woaibeijing", "北京")
    transition = CoreMlTransition(engine, encoding, DecodeTimes())
    [first] = transition([0], [(1, 2)])
    assert transition.statistics.miss_calls == 1
    # The ancestors rode along in the call that answered the first query.
    assert transition([0, 0], [(), (1,)]).shape[0] == 2
    assert transition.statistics.miss_calls == 1
    [again] = transition([0], [(1, 2)])
    assert transition.statistics.miss_calls == 1
    assert np.array_equal(first, again)
    # A prefix that was not in it is a miss of its own.
    transition([0], [(3,)])
    assert transition.statistics.miss_calls == 2


# --- the decoder --------------------------------------------------------------------------


def records() -> list[KeyRecord]:
    return [
        KeyRecord(record=index, keys=typed_keys(pinyin), text=text, context=context)
        for index, (pinyin, text, context) in enumerate(eval_rows())
    ]


def coreml_decoder(engine: CoreMlKeyModel, stack: Stack, replay: bool) -> CoreMlKeyDecoder:
    from mlime.train.key_coreml import CoreMlKeyDecoder

    keys = stack.model.e2e_config.keys
    assert keys is not None
    return CoreMlKeyDecoder(
        model=engine,
        lexicon=stack.lexicon,
        collator=key_collator(stack.spans, stack.lexicon),
        legality=stack.legality,
        options=keys.options,
        replay=replay,
    )


@darwin
@pytest.mark.parametrize("replay", [False, True], ids=["one-keystroke", "replayed"])
def test_the_decoders_top_one_is_the_pytorch_searchs_up_to_near_ties(
    engine: CoreMlKeyModel, stack: Stack, replay: bool
) -> None:
    expected = decoder(stack.model, stack.lexicon, stack.spans, stack.legality).decode(
        records(), GRID
    )
    got = coreml_decoder(engine, stack, replay).decode(records(), GRID)
    agreed = total = 0
    ties: list[tuple[int, tuple[int, ...], tuple[int, ...], float]] = []
    for point, (want_point, got_point) in enumerate(zip(expected, got, strict=True)):
        for want, have in zip(want_point, got_point, strict=True):
            total += 1
            assert have and want
            if have[0].ids == want[0].ids:
                agreed += 1
                assert abs(have[0].score - want[0].score) <= SCORE_ATOL
                continue
            # The reference's own score of what Core ML ranked first; if its beam dropped it,
            # it scored below the reference's last, so that bounds the gap from below.
            scores: dict[tuple[int, ...], float] = {h.ids: h.score for h in want}
            gap = want[0].score - scores.get(have[0].ids, want[-1].score)
            ties.append((point, want[0].ids, have[0].ids, gap))
            assert gap < NEAR_TIE, (point, want[0].ids, have[0].ids, gap)
    if ties:
        warnings.warn(f"near-ties (point, reference, core ml, reference gap): {ties}", stacklevel=1)
    assert agreed >= total * AGREE_SHARE, (agreed, total)


# --- the statistics -----------------------------------------------------------------------


@darwin
def test_the_statistics_add_up_and_a_replayed_keystroke_speculates_on_the_last(
    engine: CoreMlKeyModel, stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlime.train.key_coreml import CoreMlTransition

    asked: list[int] = []
    original = CoreMlTransition.__call__

    def counting(self: CoreMlTransition, records_: list[int], prefixes: list[Prefix]) -> np.ndarray:
        asked.append(len(prefixes))
        return original(self, records_, prefixes)

    monkeypatch.setattr(CoreMlTransition, "__call__", counting)
    one = records()[3]  # 钟爱北京 typed in full, with its context
    weights = [GRID[0]]

    plain = coreml_decoder(engine, stack, replay=False)
    plain.decode([one], weights)
    [only] = plain.statistics.keystrokes
    assert only.keys == len(one.keys) and only.speculative_rows == 0
    assert only.answered == only.queried == sum(asked) and only.queried > 0
    assert only.miss_calls == only.tree_calls >= 1
    assert only.closure >= only.queried

    asked.clear()
    replayed = coreml_decoder(engine, stack, replay=True)
    replayed.decode([one], weights)
    strokes = replayed.statistics.keystrokes
    assert [each.keys for each in strokes] == list(range(1, len(one.keys) + 1))
    assert sum(each.answered for each in strokes) == sum(each.queried for each in strokes)
    assert sum(each.answered for each in strokes) == sum(asked)
    assert strokes[0].speculative_rows == 0
    for last, each in pairwise(strokes):
        # The speculative pass cached exactly the closure of what the last keystroke queried,
        # and its calls are among the keystroke's.
        assert each.speculative_rows == last.closure
        assert each.tree_calls >= each.miss_calls
        assert each.closure >= each.queried
    assert sum(each.speculative_rows for each in strokes) > 0
    rendered = replayed.statistics.render()
    assert f"{len(strokes)} keystrokes" in rendered and "miss calls" in rendered


@darwin
@pytest.mark.parametrize("replay", [False, True], ids=["one-keystroke", "replayed"])
def test_a_run_over_the_converted_model_reports_what_the_pytorch_run_does(
    converted: Path,
    stack: Stack,
    replay: bool,
    tmp_path: Path,
    verifier: LegalityVerifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlime.train.key_coreml import load_coreml_key_run

    monkeypatch.setattr(
        Vocabularies, "load", classmethod(lambda cls, *args, **kwargs: stack.vocabularies)
    )
    eval_set = write_eval_set(tmp_path / "eval.jsonl")
    run, coreml, step = load_coreml_key_run(
        converted,
        eval_set,
        char_table=DATA / "char_pinyin.tsv",
        typo_table=DATA / "typo.json",
        syllables=DATA / "syllables.txt",
        with_context=True,
        dev_share=SHARE,
        replay=replay,
    )
    assert step == 11
    weights = [KeyWeights(1.0, 1.0, False)]
    got = report_keys(run, tmp_path / "coreml", weights, slice_="all")
    reference = KeyRun(
        decoder=decoder(stack.model, stack.lexicon, stack.spans, stack.legality),
        verifier=verifier,
        eval_rows=eval_rows(),
        dev_share=SHARE,
    )
    assert got == report_keys(reference, tmp_path / "pytorch", weights, slice_="all")
    assert len(coreml.statistics.keystrokes) >= len(eval_rows())
    assert "miss calls" in coreml.statistics.render()


@darwin
def test_more_keys_than_the_largest_function_holds_are_refused(
    engine: CoreMlKeyModel, stack: Stack
) -> None:
    from mlime.train.key_coreml import Context

    empty = Context(
        np.zeros((1, 64, stack.model.fill.config.hidden_size), np.float16),
        np.zeros((1, 64), np.float16),
    )
    with pytest.raises(ValueError, match="do not fit the largest function"):
        engine.encode([0] * 127, empty)


# --- the command line ---------------------------------------------------------------------


def test_coreml_off_macos_is_refused_with_a_clear_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "keys",
            "--coreml",
            str(tmp_path),
            "--eval-set",
            str(tmp_path / "eval.jsonl"),
            "--dump-dir",
            str(tmp_path / "dumps"),
            "--slice",
            "dev",
        ],
    )
    assert result.exit_code != 0
    assert "needs macOS" in result.output


def test_a_run_names_one_source_and_replay_needs_core_ml(tmp_path: Path) -> None:
    base = ["eval", "keys", "--eval-set", str(tmp_path / "e"), "--dump-dir", str(tmp_path / "d")]
    neither = CliRunner().invoke(app, base)
    # Option names are styled where the output is coloured, so the plain words are what is held.
    assert neither.exit_code != 0 and "name exactly one of" in neither.output
    replay = CliRunner().invoke(app, [*base, "--checkpoint", str(tmp_path / "c"), "--replay"])
    assert replay.exit_code != 0 and "replays keystrokes" in replay.output
