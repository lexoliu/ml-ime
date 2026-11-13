"""Word arcs in the constrained search over keys (#282).

The arc enumeration is held to a naive scan of the lexicon. The search with
arcs is held to brute force: with no beam, every output's score equals the
maximum (``viterbi``) or log-sum-exp (sum) over its legal covers of the path
score plus the prior's bonus, which the module docstring defines over *sets of
non-overlapping contained arcs*. Here the reference is written apart from the
search: it enumerates covers, finds the arcs each contains, enumerates the
subsets of them that do not overlap, and sums ``(e^pi - 1)^|S|`` or takes the
largest ``|S|``. At prior 0, with no lexicon entry matching, the search is the
search without arcs, hypothesis for hypothesis. An idiom every character of
which ranks below the beam is lost without the prior and wins with it.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from test_key_search import (
    BETA,
    DATA,
    GAMMA,
    SMALL,
    StubTransition,
    cover_score,
    path_score,
    random_log_probs,
    type_small,
)

from mlime.legality import OFFLINE, KeyLattice, Legality
from mlime.train.key_arcs import ArcLexicon, WordArc, frequent_entries
from mlime.train.key_search import (
    DEFAULT_ARC_WIDTH,
    ArcCounts,
    LegalityVerifier,
    Prefix,
    search,
    search_keys,
)
from mlime.train.lexicon import read_char_readings

PRIOR = 2.5

#: Six homophones of ``shi``: any of them is admitted by one typed ``s``.
SHI = "是时十石史师"


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
    return Legality(
        tuple(SMALL), read_char_readings(DATA / "char_pinyin.tsv"), legality.corrections
    )


@pytest.fixture(scope="module")
def homophones(legality: Legality) -> Legality:
    return Legality(tuple(SHI), read_char_readings(DATA / "char_pinyin.tsv"), legality.corrections)


@pytest.fixture(scope="module")
def verifier() -> LegalityVerifier:
    return LegalityVerifier.load(
        char_table=DATA / "char_pinyin.tsv",
        typo_table=DATA / "typo.json",
        syllables=DATA / "syllables.txt",
    )


Cover = tuple[Prefix, tuple[int, ...]]


def covers(lattice: KeyLattice) -> list[Cover]:
    """Every complete cover of the lattice: its characters and cut points."""
    arcs: dict[int, list[tuple[int, int]]] = {}
    for start, end, character in zip(
        lattice.start.tolist(), lattice.end.tolist(), lattice.character.tolist(), strict=True
    ):
        arcs.setdefault(start, []).append((end, character))
    found: list[Cover] = []

    def walk(at: int, ids: Prefix, cuts: tuple[int, ...]) -> None:
        if at == lattice.keys:
            found.append((ids, cuts))
            return
        for end, character in arcs.get(at, []):
            walk(end, (*ids, character), (*cuts, end))

    walk(0, (), ())
    return found


def typed_as_initial(span: str) -> bool:
    """One letter, or zh, ch or sh: the shape of an initial."""
    return len(span) == 1 or span in ("zh", "ch", "sh")


def contained(keys: str, cover: Cover, words: set[Prefix]) -> list[tuple[int, int]]:
    """The arcs *cover* contains, as character intervals ``[i, j)``.

    An entry is contained where its characters are the cover's, and at least
    one of their spans is typed as an initial.
    """
    ids, cuts = cover
    starts = (0, *cuts[:-1])
    found = []
    for word in words:
        for i in range(len(ids) - len(word) + 1):
            if ids[i : i + len(word)] != word:
                continue
            spans = [keys[starts[k] : cuts[k]] for k in range(i, i + len(word))]
            if any(typed_as_initial(span) for span in spans):
                found.append((i, i + len(word)))
    return found


def disjoint_sets(arcs: Sequence[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    """Every subset of *arcs* none of whose members overlap, the empty one included."""
    sets = []
    for size in range(len(arcs) + 1):
        for subset in itertools.combinations(arcs, size):
            if all(a[1] <= b[0] or b[1] <= a[0] for a, b in itertools.combinations(subset, 2)):
                sets.append(subset)
    return sets


def bonus(arcs: Sequence[tuple[int, int]], prior: float, viterbi: bool) -> tuple[float, int]:
    """The cover's gain from its contained *arcs*, and the size of the set that gives it.

    The max combination: ``prior`` times the largest set of non-overlapping
    arcs. The sum: the log of the mass factor ``sum_S (e^prior - 1)^|S|``.
    """
    sets = disjoint_sets(arcs)
    best = max(len(subset) for subset in sets)
    if viterbi:
        return prior * best, best
    return math.log(sum(math.expm1(prior) ** len(subset) for subset in sets)), best


def reference(
    lattice: KeyLattice,
    keys: str,
    host: np.ndarray,
    transition: StubTransition,
    record: int,
    words: set[Prefix],
    prior: float,
    viterbi: bool,
) -> dict[Prefix, float]:
    """Every legal output's score over its covers, the prior's bonus counted by definition."""
    scores: dict[Prefix, list[float]] = {}
    for ids, cuts in covers(lattice):
        starts = (0, *cuts[:-1])
        costs = [
            float(
                lattice.cost[
                    np.flatnonzero(
                        (lattice.start == a) & (lattice.end == b) & (lattice.character == c)
                    )[0]
                ]
            )
            for a, b, c in zip(starts, cuts, ids, strict=True)
        ]
        path = path_score(host, transition, record, ids, cuts, costs)
        gain, _ = bonus(contained(keys, (ids, cuts), words), prior, viterbi)
        scores.setdefault(ids, []).append(path + gain)
    out = {}
    for ids, values in scores.items():
        top = max(values)
        out[ids] = top if viterbi else top + math.log(sum(math.exp(v - top) for v in values))
    return out


def arcs_for(lexicon: ArcLexicon, lattices: Sequence[KeyLattice], keys: Sequence[str]):
    return [lexicon.arcs(lattice, each) for lattice, each in zip(lattices, keys, strict=True)]


def lexicon_from_covers(
    lattices: Sequence[KeyLattice], seed: int, per_record: int = 3
) -> tuple[ArcLexicon, set[Prefix]]:
    """Entries cut from the records' own covers, so they match, overlap and nest."""
    rng = random.Random(seed)
    words: set[Prefix] = set()
    for lattice in lattices:
        options = [ids for ids, _ in covers(lattice) if len(ids) >= 2]
        if not options:
            continue
        ids = rng.choice(options)
        for _ in range(per_record):
            length = rng.randint(2, min(4, len(ids)))
            at = rng.randint(0, len(ids) - length)
            words.add(ids[at : at + length])
    return ArcLexicon(sorted(words)), words


def batch(small: Legality, seed: int, count: int = 24):
    readings = read_char_readings(DATA / "char_pinyin.tsv")
    rng = random.Random(seed)
    keys = [type_small(rng, readings) for _ in range(count)]
    lattices = [small.lattice(each, OFFLINE) for each in keys]
    log_probs = random_log_probs(
        np.random.default_rng(seed), count, max(map(len, keys)), len(SMALL) + 1
    )
    return keys, lattices, log_probs


@pytest.mark.parametrize("viterbi", [False, True], ids=["sum", "max"])
def test_an_unbounded_search_with_arcs_is_brute_force(
    small: Legality, verifier: LegalityVerifier, viterbi: bool
) -> None:
    keys, lattices, log_probs = batch(small, 21)
    lexicon, words = lexicon_from_covers(lattices, 5)
    found = search(
        lattices,
        log_probs,
        transition := StubTransition(len(SMALL)),
        beta=BETA,
        gamma=GAMMA,
        beam=None,
        viterbi=viterbi,
        words=arcs_for(lexicon, lattices, keys),
        prior=PRIOR,
        arc_width=10**6,
    )
    host = log_probs.double().numpy()
    credited = outputs = 0
    for record, (lattice, each, hypotheses) in enumerate(zip(lattices, keys, found, strict=True)):
        expected = reference(lattice, each, host[record], transition, record, words, PRIOR, viterbi)
        got = {hypothesis.ids: hypothesis.score for hypothesis in hypotheses}
        assert got.keys() == expected.keys()
        for ids, score in expected.items():
            assert got[ids] == pytest.approx(score, rel=1e-9, abs=1e-9)
        for hypothesis in hypotheses:
            sentence = "".join(SMALL[i] for i in hypothesis.ids)
            assert verifier.cost(each, sentence, OFFLINE) is not None
            # The hypothesis's own cover contains exactly as many disjoint arcs as it says.
            own = contained(each, (hypothesis.ids, hypothesis.cuts), words)
            assert hypothesis.arcs == max(len(subset) for subset in disjoint_sets(own))
            if viterbi:
                split = (
                    hypothesis.ctc
                    + BETA * hypothesis.transition
                    - GAMMA * hypothesis.correction
                    + PRIOR * hypothesis.arcs
                )
                assert hypothesis.score == pytest.approx(split, rel=1e-9, abs=1e-9)
        outputs += len(hypotheses)
        credited += sum(hypothesis.arcs > 0 for hypothesis in hypotheses)
    assert outputs > 100
    assert credited > 20, "the entries must land for the check to mean anything"


@pytest.mark.parametrize("viterbi", [False, True], ids=["sum", "max"])
def test_a_bounded_search_with_arcs_keeps_legal_outputs_scored_by_their_covers(
    small: Legality, verifier: LegalityVerifier, viterbi: bool
) -> None:
    keys, lattices, log_probs = batch(small, 33)
    lexicon, words = lexicon_from_covers(lattices, 8)
    transition = StubTransition(len(SMALL))
    found = search(
        lattices,
        log_probs,
        transition,
        beta=BETA,
        gamma=GAMMA,
        viterbi=viterbi,
        words=arcs_for(lexicon, lattices, keys),
        prior=PRIOR,
    )
    host = log_probs.double().numpy()
    landed = 0
    for record, (lattice, each, hypotheses) in enumerate(zip(lattices, keys, found, strict=True)):
        expected = reference(lattice, each, host[record], transition, record, words, PRIOR, viterbi)
        assert len(hypotheses) <= 8
        for hypothesis in hypotheses:
            sentence = "".join(SMALL[i] for i in hypothesis.ids)
            assert verifier.cost(each, sentence, OFFLINE) is not None
            assert hypothesis.ids in expected
            assert hypothesis.score <= expected[hypothesis.ids] + 1e-9
            landed += hypothesis.arcs > 0
            if viterbi:
                cover = cover_score(lattice, host[record], transition, record, hypothesis)
                assert hypothesis.score == pytest.approx(
                    cover + PRIOR * hypothesis.arcs, rel=1e-9, abs=1e-9
                )
    assert landed > 0


def test_overlapping_arcs_are_alternatives_and_disjoint_arcs_multiply(
    homophones: Legality,
) -> None:
    """AB, ABC, ABCD, BC and CD over one cover: the sets that do not overlap are the bonus."""
    a, b, c, d = (SHI.index(character) for character in "石史师是")
    words = {(a, b), (a, b, c), (a, b, c, d), (b, c), (c, d)}
    lattice = homophones.lattice("ssss", OFFLINE)
    width = len(SHI)
    log_probs = random_log_probs(np.random.default_rng(3), 1, 4, width + 1)
    transition = StubTransition(width)
    lexicon = ArcLexicon(sorted(words))
    arcs = arcs_for(lexicon, [lattice], ["ssss"])
    # Every key admits every homophone, so an entry matches wherever it fits.
    assert {(w.start, w.entry, w.end) for w in arcs[0]} == {
        (start, word, start + len(word)) for word in words for start in range(5 - len(word))
    }
    host = log_probs.double().numpy()
    ids = (a, b, c, d)
    path = path_score(host[0], transition, 0, ids, (1, 2, 3, 4), [0.0] * 4)
    # By hand: the sets are {}, five singles, and {AB, CD} alone.
    factor = 1 + 5 * math.expm1(PRIOR) + math.expm1(PRIOR) ** 2
    for viterbi, expected, arcs_in_best in (
        (True, path + 2 * PRIOR, 2),
        (False, path + math.log(factor), 2),
    ):
        found = search(
            [lattice],
            log_probs,
            transition,
            beta=BETA,
            gamma=GAMMA,
            beam=None,
            viterbi=viterbi,
            words=arcs,
            prior=PRIOR,
            arc_width=100,
        )
        [hypothesis] = [h for h in found[0] if h.ids == ids]
        assert hypothesis.score == pytest.approx(expected, rel=1e-9, abs=1e-9)
        assert hypothesis.arcs == arcs_in_best
        assert hypothesis.cuts == (1, 2, 3, 4)
    # The reference agrees with the hand count.
    assert bonus(contained("ssss", (ids, (1, 2, 3, 4)), words), PRIOR, False)[0] == pytest.approx(
        math.log(factor)
    )


def test_an_entry_typed_in_full_syllables_is_no_arc(legality: Legality) -> None:
    """Without one initial-typed span a chain is just a sentence: no arc, no prior.

    ``zhongguo`` still has chains through other spans (a corrected ``z|ho``, a
    trailing ``g`` as an initial), but none from 0 to 8, where every span is a
    full syllable.
    """
    ids = legality.ids("中国")
    assert ids is not None
    lexicon = ArcLexicon([ids])
    full = lexicon.arcs(legality.lattice("zhongguo", OFFLINE), "zhongguo")
    assert (0, 8) not in {(arc.start, arc.end) for arc in full}
    half = lexicon.arcs(legality.lattice("zhongg", OFFLINE), "zhongg")
    assert (0, 6) in {(arc.start, arc.end) for arc in half}
    assert lexicon.arcs(legality.lattice("zg", OFFLINE), "zg")[0].end == 2


def scan(
    lattice: KeyLattice, keys: str, words: set[Prefix]
) -> dict[tuple[int, Prefix, int], list[tuple[int, ...]]]:
    """Every chain of every entry, by trying each entry from each cut: the slow, plain way."""
    arcs = list(
        zip(lattice.start.tolist(), lattice.end.tolist(), lattice.character.tolist(), strict=True)
    )
    found: dict[tuple[int, Prefix, int], list[tuple[int, ...]]] = {}

    def walk(
        word: Prefix, origin: int, at: int, taken: int, chain: tuple[int, ...], typed: bool
    ) -> None:
        if taken == len(word):
            if typed:
                found.setdefault((origin, word, at), []).append(chain)
            return
        for index, (start, end, character) in enumerate(arcs):
            if start == at and character == word[taken]:
                initial = typed or typed_as_initial(keys[start:end])
                walk(word, origin, end, taken + 1, (*chain, index), initial)

    for word in words:
        for origin in range(lattice.keys):
            walk(word, origin, origin, 0, (), False)
    return found


def test_the_enumeration_equals_a_scan_of_the_lexicon(small: Legality) -> None:
    keys, lattices, _ = batch(small, 9, count=40)
    lexicon, words = lexicon_from_covers(lattices, 4, per_record=6)
    total = 0
    for lattice, each in zip(lattices, keys, strict=True):
        got = {
            (w.start, w.entry, w.end): sorted(map(tuple, w.chains.tolist()))
            for w in lexicon.arcs(lattice, each)
        }
        expected = scan(lattice, each, words)
        assert got == {key: sorted(chains) for key, chains in expected.items()}
        total += len(got)
    assert total > 20


def test_no_arc_prior_zero_or_no_lexicon_is_the_search_without_arcs(small: Legality) -> None:
    keys, lattices, log_probs = batch(small, 14)
    matching, _ = lexicon_from_covers(lattices, 6)
    never = ArcLexicon([tuple(range(8))])  # eight characters over at most seven keys
    assert all(not w for w in arcs_for(never, lattices, keys))
    assert any(arcs_for(matching, lattices, keys))
    for viterbi in (False, True):
        for beam in (8, None):
            plain = search(
                lattices,
                log_probs,
                StubTransition(len(SMALL)),
                beta=BETA,
                gamma=GAMMA,
                beam=beam,
                viterbi=viterbi,
            )
            assert all(h.arcs == 0 for hypotheses in plain for h in hypotheses)
            for lexicon, prior in ((matching, 0.0), (never, PRIOR), (None, 0.0)):
                counts = ArcCounts()
                again = search(
                    lattices,
                    log_probs,
                    StubTransition(len(SMALL)),
                    beta=BETA,
                    gamma=GAMMA,
                    beam=beam,
                    viterbi=viterbi,
                    words=None if lexicon is None else arcs_for(lexicon, lattices, keys),
                    prior=prior,
                    counts=counts,
                )
                assert again == plain
                assert counts.landed == 0 and counts.rows == 0
    # An arc prior with nothing to propose is a mistake, not a quiet no-op.
    with pytest.raises(ValueError, match="needs the records' word arcs"):
        search(lattices, log_probs, StubTransition(len(SMALL)), beta=BETA, gamma=GAMMA, prior=1.0)


class Spy:
    """A uniform transition that remembers each call."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.calls: list[list[tuple[int, Prefix]]] = []

    def __call__(self, records: Sequence[int], prefixes: Sequence[Prefix]) -> torch.Tensor:
        self.calls.append(list(zip(records, prefixes, strict=True)))
        return torch.full((len(prefixes), self.width), -math.log(self.width))


def lowest_ranked(width: int, ranks: dict[int, int]) -> torch.Tensor:
    """Log-probs over four keys: at key ``t`` character ``ranks[t]`` is the least likely.

    Every other character gets logit 3 and the least likely one 0, the blank 0.
    """
    logits = torch.full((1, 4, width + 1), 3.0)
    logits[..., width] = 0.0
    for t, character in ranks.items():
        logits[0, t, character] = 0.0
    return logits.log_softmax(dim=-1)


def test_an_idiom_pruned_without_the_prior_wins_with_it(
    homophones: Legality, verifier: LegalityVerifier
) -> None:
    # Eleven homophones: more than the beam of 8, so the least likely character at a key
    # is always cut at the first step.
    eleven = Legality(
        tuple("是时十石史师世市事室式"),
        read_char_readings(DATA / "char_pinyin.tsv"),
        homophones.corrections,
    )
    chars = eleven.characters
    idiom = tuple(chars.index(character) for character in "石史事式")
    width = len(chars)
    lattice = eleven.lattice("ssss", OFFLINE)
    log_probs = lowest_ranked(width, dict(enumerate(idiom)))
    spy = Spy(width)
    lexicon = ArcLexicon([idiom])
    words = arcs_for(lexicon, [lattice], ["ssss"])
    assert len(words[0]) == 1
    for viterbi in (False, True):
        kwargs = {"beta": BETA, "gamma": GAMMA, "viterbi": viterbi}
        every = search([lattice], log_probs, Spy(width), beam=None, top_k=None, **kwargs)
        assert idiom in {h.ids for h in every[0]}, "it is a legal output"
        without = search([lattice], log_probs, Spy(width), words=words, prior=0.0, **kwargs)
        assert idiom not in {h.ids for h in without[0]}, "the beam loses every one of its prefixes"
        counts = ArcCounts()
        found = search(
            [lattice], log_probs, Spy(width), words=words, prior=20.0, counts=counts, **kwargs
        )
        best = found[0][0]
        assert best.ids == idiom and best.arcs == 1 and best.cuts == (1, 2, 3, 4)
        sentence = "".join(chars[i] for i in best.ids)
        assert verifier.cost("ssss", sentence, OFFLINE) == 0.0
        if viterbi:
            assert best.score == pytest.approx(
                best.ctc + BETA * best.transition - GAMMA * best.correction + 20.0
            )
        assert counts == ArcCounts(offered=1, landed=1, rows=3)
    # The inner prefixes' rows are asked through the transition, in rounds, each once.
    search([lattice], log_probs, spy, beta=BETA, gamma=GAMMA, words=words, prior=20.0)
    asked = [pair for call in spy.calls for pair in call]
    assert len(asked) == len(set(asked))
    rounds = [
        next(at for at, call in enumerate(spy.calls) if (0, idiom[:k]) in call) for k in (1, 2, 3)
    ]
    assert rounds == sorted(rounds) and len(set(rounds)) == 3, "round k asks the k-prefixes"


def test_arc_width_bounds_the_landings_to_the_best_by_optimistic_gain(
    homophones: Legality,
) -> None:
    eleven = Legality(
        tuple("是时十石史师世市事室式"),
        read_char_readings(DATA / "char_pinyin.tsv"),
        homophones.corrections,
    )
    width = len(eleven.characters)
    lattice = eleven.lattice("ssss", OFFLINE)
    rng = random.Random(4)
    entries = sorted({tuple(rng.randrange(width) for _ in range(4)) for _ in range(40)})[:20]
    assert len(entries) == 20
    log_probs = random_log_probs(np.random.default_rng(8), 1, 4, width + 1)
    host = log_probs.double().numpy()[0]

    def gain(entry: Prefix) -> float:
        return float(sum(host[t, c] - host[t, width] for t, c in enumerate(entry)))

    lexicon = ArcLexicon(entries)
    words = arcs_for(lexicon, [lattice], ["ssss"])
    assert len(words[0]) == 20
    for kept in (1, 3, 8):
        counts = ArcCounts()
        found = search(
            [lattice],
            log_probs,
            Spy(width),
            beta=BETA,
            gamma=GAMMA,
            viterbi=True,
            words=words,
            prior=30.0,
            arc_width=kept,
            counts=counts,
        )
        wanted = sorted(entries, key=lambda entry: -gain(entry))[:kept]
        landed = [h.ids for h in found[0] if h.arcs]
        assert set(landed) <= set(wanted), "only the best by gain are ever offered to land"
        assert counts.offered == 20 and counts.landed == kept
    assert DEFAULT_ARC_WIDTH == 8


def test_an_arc_adds_its_prior_to_exactly_its_own_output(homophones: Legality) -> None:
    lattice = homophones.lattice("ssss", OFFLINE)
    width = len(SHI)
    log_probs = random_log_probs(np.random.default_rng(1), 1, 4, width + 1)
    entry = (3, 4, 5, 0)
    words = arcs_for(ArcLexicon([entry]), [lattice], ["ssss"])
    # With beam=None the idiom is just another output; the arc adds its prior to it.
    plain = search(
        [lattice], log_probs, Spy(width), beta=BETA, gamma=GAMMA, beam=None, viterbi=True
    )
    with_arc = search(
        [lattice],
        log_probs,
        Spy(width),
        beta=BETA,
        gamma=GAMMA,
        beam=None,
        viterbi=True,
        words=words,
        prior=7.0,
    )
    before = {h.ids: h.score for h in plain[0]}
    after = {h.ids: h.score for h in with_arc[0]}
    assert after.pop(entry) == pytest.approx(before.pop(entry) + 7.0)
    assert after == pytest.approx(before)


def test_the_frequency_floor_keeps_the_entries_the_corpus_holds_often_enough(
    tmp_path: Path,
) -> None:
    table = tmp_path / "entries.tsv"
    header = "entry\tcategory\toccurrences\tsentences\tblocked\n"
    table.write_text(
        header
        + "一丁不识\tidiom\t0\t0\t0\n"
        + "一不做二不休\tidiom\t20\t19\t0\n"
        + "東京タワー\tplace\t400\t380\t0\n"
        + "서울\tplace\t5\t5\t0\n"
        + "一丝不挂\tidiom\t131\t120\t0\n",
        encoding="utf-8",
    )
    assert frequent_entries(table, ["idiom"], 20) == [
        ("一不做二不休", "idiom"),
        ("一丝不挂", "idiom"),
    ]
    assert frequent_entries(table, ["idiom", "place"], 100) == [
        ("東京タワー", "place"),
        ("一丝不挂", "idiom"),
    ]
    assert len(frequent_entries(table, ["idiom"], 0)) == 3
    with pytest.raises(ValueError, match="holds no entry"):
        frequent_entries(table, ["idiom"], 1000)
    with pytest.raises(ValueError, match="category"):
        frequent_entries(table, [], 20)
    with pytest.raises(FileNotFoundError):
        frequent_entries(tmp_path / "missing.tsv", ["idiom"], 20)
    table.write_text("entry\tcategory\n一丝不挂\tidiom\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no column"):
        frequent_entries(table, ["idiom"], 20)
    table.write_text(header + "一丝不挂\tidiom\tmany\t1\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a row"):
        frequent_entries(table, ["idiom"], 20)


def test_the_lexicon_loads_its_categories_and_counts_what_it_skips(
    legality: Legality, tmp_path: Path
) -> None:
    path = tmp_path / "knowledge.tsv"
    path.write_text(
        "水饺\tidiom\n"  # in
        "好\tidiom\n"  # too short
        "水龘\tidiom\n"  # a character with no place in the emittable set
        "你好\tperson\n"  # another category
        "你好\tidiom\n"  # in
        "水饺\tidiom\n",  # a repeat
        encoding="utf-8",
    )
    characters = tuple("水饺好你龘"[:4])
    lexicon = ArcLexicon.load(path, ["idiom"], characters)
    assert (lexicon.size, lexicon.skipped_short, lexicon.skipped_outside) == (2, 1, 1)
    both = ArcLexicon.load(path, ["idiom", "person"], characters)
    assert both.size == 2, "a repeated word is one entry"
    path.write_text("水饺 idiom\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row"):
        ArcLexicon.load(path, ["idiom"], characters)
    path.write_text("好\tidiom\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one entry"):
        ArcLexicon.load(path, ["idiom"], characters)
    with pytest.raises(FileNotFoundError):
        ArcLexicon.load(tmp_path / "missing.tsv", ["idiom"], characters)
    with pytest.raises(ValueError, match="category"):
        ArcLexicon.load(path, [], characters)


def test_real_abbreviations_land_real_idioms_legally(
    legality: Legality, verifier: LegalityVerifier
) -> None:
    words = [legality.ids(sentence) for sentence in ("中国人民", "一帆风顺", "天气")]
    assert all(word is not None for word in words)
    lexicon = ArcLexicon([word for word in words if word is not None])
    typed = ["zgrm", "yffs", "zhongguorm", "jttq", "jintiantq", "wxqbj"]
    width = len(legality.characters)
    log_probs = random_log_probs(np.random.default_rng(5), len(typed), 10, width + 1)
    counts = ArcCounts()
    found = search_keys(
        legality,
        typed,
        log_probs,
        StubTransition(width),
        beta=BETA,
        gamma=GAMMA,
        arcs=lexicon,
        prior=8.0,
        counts=counts,
    )
    assert counts.offered >= 5 and counts.landed >= 3
    landed = 0
    for each, hypotheses in zip(typed, found, strict=True):
        for hypothesis in hypotheses:
            sentence = "".join(legality.characters[i] for i in hypothesis.ids)
            cost = verifier.cost(each, sentence, OFFLINE)
            assert cost is not None and cost <= hypothesis.correction + 1e-9
            landed += hypothesis.arcs > 0
    assert landed > 0


def test_a_word_arc_is_a_value_of_the_lattice(small: Legality) -> None:
    keys, lattices, _ = batch(small, 2)
    lexicon, _ = lexicon_from_covers(lattices, 1)
    for lattice, each in zip(lattices, keys, strict=True):
        for arc in lexicon.arcs(lattice, each):
            assert isinstance(arc, WordArc)
            assert arc.chains.shape[1] == len(arc.entry)
            assert (lattice.start[arc.chains[:, 0]] == arc.start).all()
            assert (lattice.end[arc.chains[:, -1]] == arc.end).all()
            assert (lattice.character[arc.chains] == np.array(arc.entry)).all()


def test_the_enumeration_refuses_keys_that_are_not_the_lattices(small: Legality) -> None:
    lattice = small.lattice("sj", OFFLINE)
    with pytest.raises(ValueError, match="keys"):
        ArcLexicon([(0, 1)]).arcs(lattice, "sjx")
    with pytest.raises(ValueError, match="at least"):
        ArcLexicon([(0,)])
