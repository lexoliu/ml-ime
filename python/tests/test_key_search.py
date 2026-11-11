"""The constrained beam search over keys and the independent verifier (#216 §3, §5, §8).

The search is held to brute force: on short inputs over a small automaton,
every legal cover is enumerated and scored by the path formula, and the search
without a beam returns exactly those outputs at exactly those scores, summed
or maxed over covers; with a beam it returns legal outputs scored by covers it
names. The transition is asked once per key at most and once per distinct
prefix of a record. Every output on real sentences, typed in full, abbreviated
and mixed, passes the verifier; the verifier agrees with the shared legality
fixture and settles the §3 cases on its own.
"""

from __future__ import annotations

import json
import math
import random
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch

from mlime.legality import (
    LEGALITY_FIXTURE_RELATIVE,
    LIVE,
    OFFLINE,
    KeyLattice,
    Legality,
    LegalityOptions,
)
from mlime.train.key_search import (
    Hypothesis,
    LegalityVerifier,
    Prefix,
    Transition,
    search,
    search_keys,
)
from mlime.train.lexicon import read_char_readings

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "crates/ime-pinyin/data"

#: A small emittable set: homophones, polyphones, initials that collide, and
#: the syllables of §3's cases.
SMALL = "水饺家铺西安先沙拉啊是就爱他下嗯你个好们睡脚"

BETA = 0.7
GAMMA = 1.3


@pytest.fixture(scope="module")
def legality() -> Legality:
    return Legality.load(
        None,
        char_table=DATA / "char_pinyin.tsv",
        typo_table=DATA / "typo.json",
        syllables=DATA / "syllables.txt",
    )


@pytest.fixture(scope="module")
def small(legality: Legality) -> Legality:
    readings = read_char_readings(DATA / "char_pinyin.tsv")
    return Legality(tuple(SMALL), readings, legality.corrections)


@pytest.fixture(scope="module")
def verifier() -> LegalityVerifier:
    return LegalityVerifier.load(
        char_table=DATA / "char_pinyin.tsv",
        typo_table=DATA / "typo.json",
        syllables=DATA / "syllables.txt",
    )


class StubTransition:
    """A deterministic transition that remembers every call."""

    def __init__(self, width: int, seed: int = 0) -> None:
        self.width = width
        self.seed = seed
        self.calls: list[list[tuple[int, Prefix]]] = []

    def row(self, record: int, prefix: Prefix) -> npt.NDArray[np.float32]:
        rng = np.random.default_rng(zlib.crc32(repr((self.seed, record, prefix)).encode()))
        logits = rng.normal(0.0, 2.0, self.width)
        top = logits.max()
        log_probs = logits - (top + math.log(np.exp(logits - top).sum()))
        return log_probs.astype(np.float32)

    def __call__(self, records: Sequence[int], prefixes: Sequence[Prefix]) -> torch.Tensor:
        self.calls.append(list(zip(records, prefixes, strict=True)))
        return torch.from_numpy(
            np.stack([self.row(r, p) for r, p in zip(records, prefixes, strict=True)])
        )


def random_log_probs(rng: np.random.Generator, rows: int, width: int, columns: int) -> torch.Tensor:
    logits = torch.from_numpy(rng.normal(0.0, 2.0, (rows, width, columns)).astype(np.float32))
    return logits.log_softmax(dim=-1)


def type_small(rng: random.Random, readings: dict[str, tuple[str, ...]]) -> str:
    """One to three characters of :data:`SMALL`, each in full or as an initial, maybe slipped."""
    keys = []
    for _ in range(rng.randint(1, 3)):
        reading = rng.choice(readings[rng.choice(SMALL)])
        if rng.random() < 0.4:
            keys.append(reading[:2] if reading[:2] in ("zh", "ch", "sh") else reading[0])
        elif len(reading) >= 3 and rng.random() < 0.4:
            at = rng.randrange(1, len(reading))
            keys.append(reading[:at] + rng.choice("aeiouqwp") + reading[at + 1 :])
        else:
            keys.append(reading)
    return "".join(keys)[:7]


def brute_force(
    lattice: KeyLattice,
    log_probs: npt.NDArray[np.float64],
    transition: StubTransition,
    record: int,
    viterbi: bool,
) -> dict[Prefix, float]:
    """Every legal output's score, over every legal cover, by the path formula."""
    arcs: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for start, end, character, cost in zip(
        lattice.start.tolist(),
        lattice.end.tolist(),
        lattice.character.tolist(),
        lattice.cost.tolist(),
        strict=True,
    ):
        arcs.setdefault((start, end), []).append((character, cost))
    n = lattice.keys
    paths: dict[Prefix, list[float]] = {}

    def walk(at: int, ids: Prefix, ends: tuple[int, ...], costs: tuple[float, ...]) -> None:
        if at == n:
            paths.setdefault(ids, []).append(
                path_score(log_probs, transition, record, ids, ends, costs)
            )
            return
        for end in range(at + 1, n + 1):
            for character, cost in arcs.get((at, end), []):
                walk(end, (*ids, character), (*ends, end), (*costs, cost))

    walk(0, (), (), ())
    return {ids: combine(scores, viterbi) for ids, scores in paths.items()}


def path_score(
    log_probs: npt.NDArray[np.float64],
    transition: StubTransition,
    record: int,
    ids: Prefix,
    ends: Sequence[int],
    costs: Sequence[float],
) -> float:
    n = ends[-1]
    emitting = {end - 1 for end in ends}
    blanks = sum(log_probs[t, -1] for t in range(n) if t not in emitting)
    emits = sum(log_probs[end - 1, c] for end, c in zip(ends, ids, strict=True))
    steps = sum(float(transition.row(record, ids[:j])[c]) for j, c in enumerate(ids))
    return float(blanks + emits + BETA * steps - GAMMA * sum(costs))


def combine(scores: Sequence[float], viterbi: bool) -> float:
    top = max(scores)
    return top if viterbi else top + math.log(sum(math.exp(s - top) for s in scores))


def small_batch(
    small: Legality, seed: int, options: LegalityOptions, count: int = 24
) -> tuple[list[str], list[KeyLattice], torch.Tensor]:
    readings = read_char_readings(DATA / "char_pinyin.tsv")
    rng = random.Random(seed)
    keys = [type_small(rng, readings) for _ in range(count)]
    lattices = [small.lattice(each, options) for each in keys]
    log_probs = random_log_probs(
        np.random.default_rng(seed), count, max(map(len, keys)), len(SMALL) + 1
    )
    return keys, lattices, log_probs


@pytest.mark.parametrize("viterbi", [False, True], ids=["sum", "max"])
@pytest.mark.parametrize("options", [OFFLINE, LIVE], ids=["offline", "live"])
def test_an_unbounded_search_is_brute_force(
    small: Legality, viterbi: bool, options: LegalityOptions
) -> None:
    _, lattices, log_probs = small_batch(small, 7, options)
    transition = StubTransition(len(SMALL))
    found = search(
        lattices, log_probs, transition, beta=BETA, gamma=GAMMA, beam=None, viterbi=viterbi
    )
    host = log_probs.double().numpy()
    outputs = corrected = covered = 0
    for record, (lattice, hypotheses) in enumerate(zip(lattices, found, strict=True)):
        expected = brute_force(lattice, host[record], transition, record, viterbi)
        got = {hypothesis.ids: hypothesis.score for hypothesis in hypotheses}
        assert got.keys() == expected.keys()
        for ids, score in expected.items():
            assert got[ids] == pytest.approx(score, rel=1e-9, abs=1e-9)
        assert [h.score for h in hypotheses] == sorted((h.score for h in hypotheses), reverse=True)
        outputs += len(hypotheses)
        corrected += sum(h.correction > 0 for h in hypotheses)
        covered += sum(len(h.ids) < h.cuts[-1] for h in hypotheses)
    # The inputs reach many outputs, corrections and multi-key spans.
    assert outputs > 100
    assert corrected > 0
    assert covered > 0


def test_a_beam_keeps_legal_outputs_scored_by_their_covers(small: Legality) -> None:
    _, lattices, log_probs = small_batch(small, 11, OFFLINE)
    transition = StubTransition(len(SMALL))
    host = log_probs.double().numpy()
    pruned = 0
    for viterbi in (False, True):
        found = search(lattices, log_probs, transition, beta=BETA, gamma=GAMMA, viterbi=viterbi)
        for record, (lattice, hypotheses) in enumerate(zip(lattices, found, strict=True)):
            expected = brute_force(lattice, host[record], transition, record, viterbi)
            assert len(hypotheses) <= 8
            pruned += len(expected) > 8
            if expected:
                assert hypotheses, "a record with a legal output returns one"
            for hypothesis in hypotheses:
                assert hypothesis.ids in expected
                assert hypothesis.score <= expected[hypothesis.ids] + 1e-9
                if viterbi:
                    cover = cover_score(lattice, host[record], transition, record, hypothesis)
                    assert hypothesis.score == pytest.approx(cover, rel=1e-9, abs=1e-9)
                    split = (
                        hypothesis.ctc
                        + BETA * hypothesis.transition
                        - GAMMA * hypothesis.correction
                    )
                    assert hypothesis.score == pytest.approx(split, rel=1e-9, abs=1e-9)
    assert pruned > 0, "the beam must prune for the check to mean anything"


def cover_score(
    lattice: KeyLattice,
    log_probs: npt.NDArray[np.float64],
    transition: StubTransition,
    record: int,
    hypothesis: Hypothesis,
) -> float:
    """The path score of the hypothesis's own cover, its arcs read off the lattice."""
    costs = []
    start = 0
    for end, character in zip(hypothesis.cuts, hypothesis.ids, strict=True):
        hit = np.flatnonzero(
            (lattice.start == start) & (lattice.end == end) & (lattice.character == character)
        )
        assert hit.size == 1, "the cover's arcs are the automaton's"
        costs.append(float(lattice.cost[hit[0]]))
        start = end
    assert start == lattice.keys
    return path_score(log_probs, transition, record, hypothesis.ids, hypothesis.cuts, costs)


@pytest.mark.parametrize("beam", [8, None], ids=["beam8", "unbounded"])
def test_the_transition_is_asked_once_per_prefix(small: Legality, beam: int | None) -> None:
    keys, lattices, log_probs = small_batch(small, 3, OFFLINE)
    transition = StubTransition(len(SMALL))
    search(lattices, log_probs, transition, beta=BETA, gamma=GAMMA, beam=beam)
    assert 0 < len(transition.calls) <= max(map(len, keys))
    asked = [pair for call in transition.calls for pair in call]
    assert len(asked) == len(set(asked))
    # Batched across records: some call serves more than one.
    assert any(len({record for record, _ in call}) > 1 for call in transition.calls)


#: Real sentences, each typed in full, abbreviated and mixed.
TYPED = [
    ("我们吃水饺", ("womenchishuijiao", "wmcsj", "womencsj")),
    ("中国人民", ("zhongguorenmin", "zgrm", "zhongguorm")),
    ("今天天气很好", ("jintiantianqihenhao", "jttqhh", "jintiantqhh")),
    ("我想去北京", ("woxiangqubeijing", "wxqbj", "woxqbeijing")),
]


def test_every_output_is_legal_on_real_tables(
    legality: Legality, verifier: LegalityVerifier
) -> None:
    keys = [typed for _, styles in TYPED for typed in styles]
    width = len(legality.characters)
    log_probs = random_log_probs(
        np.random.default_rng(5), len(keys), max(map(len, keys)), width + 1
    )
    transition = StubTransition(width)
    found = search_keys(legality, keys, log_probs, transition, beta=BETA, gamma=GAMMA)
    for typed, hypotheses in zip(keys, found, strict=True):
        assert len(hypotheses) == 8
        for hypothesis in hypotheses:
            sentence = "".join(legality.characters[i] for i in hypothesis.ids)
            cost = verifier.cost(typed, sentence, OFFLINE)
            assert cost is not None, f"{typed} -> {sentence} is illegal"
            assert cost <= hypothesis.correction + 1e-9
            spans = [
                verifier.span_cost(typed[a:b], b == len(typed), character, OFFLINE)
                for a, b, character in zip(
                    (0, *hypothesis.cuts[:-1]), hypothesis.cuts, sentence, strict=True
                )
            ]
            assert sum(spans) == pytest.approx(hypothesis.correction, abs=1e-9)


def gold_log_probs(width: int, keys: int, ends: dict[int, int]) -> torch.Tensor:
    """Log-probs that favour emitting ``ends[t]`` at key ``t`` and blank elsewhere."""
    logits = torch.zeros(1, keys, width + 1)
    for t in range(keys):
        logits[0, t, ends.get(t, width)] = 12.0
    return logits.log_softmax(dim=-1)


def uniform(width: int) -> Transition:
    """A transition that knows nothing: every character equally likely."""

    def call(records: Sequence[int], prefixes: Sequence[Prefix]) -> torch.Tensor:
        return torch.full((len(prefixes), width), -math.log(width))

    return call


@pytest.mark.parametrize(
    ("keys", "sentence", "cuts"),
    [
        ("shuijiap", "水饺", (4, 8)),
        ("shuijiap", "水家铺", (4, 7, 8)),
        ("sj", "水饺", (1, 2)),
        ("sk", "烧烤", (1, 2)),
        ("sl", "沙拉", (1, 2)),
        ("sn", "酸奶", (1, 2)),
        ("zgrm", "中国人民", (1, 2, 3, 4)),
        ("xian", "西安", (2, 4)),
        ("xian", "先", (4,)),
        ("ng", "嗯", (2,)),
        ("ng", "你个", (1, 2)),
        ("hm", "噷", (2,)),
        ("hm", "好们", (1, 2)),
    ],
)
def test_the_search_reaches_the_legal_cases(
    legality: Legality, keys: str, sentence: str, cuts: tuple[int, ...]
) -> None:
    ids = legality.ids(sentence)
    assert ids is not None
    width = len(legality.characters)
    log_probs = gold_log_probs(
        width, len(keys), {end - 1: c for end, c in zip(cuts, ids, strict=True)}
    )
    [found] = search_keys(
        legality, [keys], log_probs, uniform(width), beta=BETA, gamma=GAMMA, viterbi=True
    )
    best = found[0]
    assert best.ids == ids
    assert best.cuts == cuts
    assert best.correction == legality.cost(keys, sentence, OFFLINE)


def test_the_typo_reading_pays_its_correction(legality: Legality) -> None:
    jiao = dict(legality.corrections.corrections("jiap"))["jiao"]
    ids = legality.ids("水饺")
    assert ids is not None
    width = len(legality.characters)
    log_probs = gold_log_probs(width, 8, {3: ids[0], 7: ids[1]})
    [found] = search_keys(
        legality, ["shuijiap"], log_probs, uniform(width), beta=BETA, gamma=GAMMA, viterbi=True
    )
    assert found[0].ids == ids
    assert found[0].correction == pytest.approx(jiao) and jiao > 0
    assert found[0].score == pytest.approx(found[0].ctc + BETA * found[0].transition - GAMMA * jiao)


def test_wmzyq_never_reads_as_women_chi_shuijiao(
    legality: Legality, verifier: LegalityVerifier
) -> None:
    ids = legality.ids("我们吃水饺")
    assert ids is not None
    width = len(legality.characters)
    log_probs = gold_log_probs(width, 5, dict(enumerate(ids)))
    for options in (OFFLINE, LIVE):
        [found] = search_keys(
            legality,
            ["wmzyq"],
            log_probs,
            uniform(width),
            beta=BETA,
            gamma=GAMMA,
            options=options,
            beam=64,
        )
        assert found
        assert all(hypothesis.ids != ids for hypothesis in found)
        for hypothesis in found:
            sentence = "".join(legality.characters[i] for i in hypothesis.ids)
            assert verifier.cost("wmzyq", sentence, options) is not None


def test_the_verifier_settles_the_section_3_cases(
    legality: Legality, verifier: LegalityVerifier
) -> None:
    jiao = dict(legality.corrections.corrections("jiap"))["jiao"]
    assert verifier.cost("wmzyq", "我们吃水饺", OFFLINE) is None
    assert verifier.cost("wmzyq", "我们吃水饺", LIVE) is None
    assert verifier.cost("wmcsj", "我们吃水饺", OFFLINE) == 0.0
    assert verifier.cost("shuijiap", "水饺", OFFLINE) == pytest.approx(jiao)
    assert jiao > 0.0
    assert verifier.cost("shuijiap", "水家铺", OFFLINE) == 0.0
    for keys, sentence in (("sj", "水饺"), ("sk", "烧烤"), ("sl", "沙拉"), ("sn", "酸奶")):
        assert verifier.cost(keys, sentence, OFFLINE) == 0.0
    assert verifier.cost("zgrm", "中国人民", OFFLINE) == 0.0
    assert verifier.cost("xian", "西安", OFFLINE) == 0.0
    assert verifier.cost("xian", "先", OFFLINE) == 0.0
    assert verifier.cost("ng", "嗯", OFFLINE) == 0.0
    assert verifier.cost("ng", "你个", OFFLINE) == 0.0
    assert verifier.cost("hm", "噷", OFFLINE) == 0.0
    assert verifier.cost("hm", "好们", OFFLINE) == 0.0
    # Every key consumed, none twice; the tail only when allowed.
    assert verifier.cost("a", "啊啊", OFFLINE) is None
    assert verifier.cost("ab", "啊", OFFLINE) is None
    assert verifier.cost("guozho", "国中", OFFLINE) is None
    assert verifier.cost("guozho", "国中", LIVE) == 0.0


def test_the_verifier_agrees_with_the_legality_fixture(verifier: LegalityVerifier) -> None:
    document = json.loads((ROOT / LEGALITY_FIXTURE_RELATIVE).read_text(encoding="utf-8"))
    assert document["cases"]
    for case in document["cases"]:
        options = LegalityOptions(
            unfinished_tail=bool(case["unfinished_tail"]),
            correct_syllables=bool(case["correct_syllables"]),
        )
        got = verifier.cost(str(case["keys"]), str(case["sentence"]), options)
        if case["cost"] is None:
            assert got is None, case
        else:
            assert got == pytest.approx(float(case["cost"]), abs=1e-9), case
