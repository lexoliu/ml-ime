"""The shared fixture the Rust key search is held to (#265).

``crates/ime-decode/tests/fixtures/key-search.json`` pins :func:`mlime.train.key_search.search`
on synthetic inputs that both languages reproduce bit for bit, so the Rust port
needs no model: the emissions and the transition rows are functions of
SplitMix64 over integers, and the legality lattices are the real automaton's
over a small emittable set the fixture names.

Each record carries, per setting (combination, beam, gamma), the hypotheses the
Python search returned -- ids, cuts and the four score terms at ``repr``
precision -- and the sequence of transition calls it made. The arc records (#293)
also carry a small lexicon's load over their alphabet, the word arcs it
enumerates over their lattice, and settings over the arc prior and width whose
hypotheses carry their arc counts.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mlime.legality import OFFLINE, Legality
from mlime.legality import _splitmix as splitmix
from mlime.train.key_arcs import ArcLexicon, WordArc
from mlime.train.key_search import DEFAULT_ARC_WIDTH, ArcCounts, Hypothesis, Prefix, search
from mlime.train.lexicon import read_char_readings

#: Where the fixture lives, relative to the repository root.
SEARCH_FIXTURE_RELATIVE = Path("crates/ime-decode/tests/fixtures/key-search.json")

#: The legality fixture the training records are read from.
LEGALITY_RECORDS_RELATIVE = Path("crates/ime-pinyin/tests/fixtures/legality.json")

#: The transition weight of every setting.
BETA = 1.0

#: The correction weights of the grid.
GAMMAS = (0.0, 0.5)

#: The beam of the bounded settings.
BEAM = 8

#: A record with at most this many keys also runs without a beam.
UNBOUNDED_MAX_KEYS = 8

#: The arc priors of the arc records' grid; 0 pins the search without arcs.
ARC_PRIORS = (0.0, 1.5, 6.0)

#: The arc widths of the arc records' grid: one landing per offset, and the default.
ARC_WIDTHS = (1, DEFAULT_ARC_WIDTH)

#: The correction weight of the arc records' settings.
ARC_GAMMA = 0.5

#: The categories the arc lexicon is read for.
LEXICON_CATEGORIES = ("idiom",)

#: The arc lexicon's ``word<TAB>category`` rows. Besides the entries the records'
#: emittable sets hold, it carries what a load skips: a single character, entries
#: past an emittable set (Chinese, Japanese and Korean alike), a repeated entry
#: and entries of a category nobody asked for.
LEXICON_ROWS = (
    "逐光捡金\tidiom",
    "逐光\tidiom",
    "捡金\tidiom",
    "十周年快乐\tidiom",
    "十周年\tidiom",
    "周年\tidiom",
    "快乐\tidiom",
    "快乐\tidiom",
    "李云龙\tidiom",
    "太欢乐\tidiom",
    "欢乐\tidiom",
    "这个\tidiom",
    "这个女人\tidiom",
    "女人\tidiom",
    "晚上\tidiom",
    "他将\tidiom",
    "首发\tidiom",
    "抱起\tidiom",
    "就一个\tidiom",
    "一个\tidiom",
    "飞奔\tidiom",
    "可不\tidiom",
    "不好\tidiom",
    "好惹\tidiom",
    "玉米\tidiom",
    "浓汤\tidiom",
    "盖饭\tidiom",
    "数学家们\tidiom",
    "数学家\tidiom",
    "数学\tidiom",
    "拓展\tidiom",
    "概念\tidiom",
    "这些\tidiom",
    "是时\tidiom",
    "时事\tidiom",
    "十是\tidiom",
    "史石\tidiom",
    "市是时\tidiom",
    "事室\tidiom",
    "师世\tidiom",
    "是时十\tidiom",
    "时十\tidiom",
    "十是时\tidiom",
    "就金\tidiom",
    "金捡\tidiom",
    "时金\tidiom",
    "十就\tidiom",
    "捡金捡\tidiom",
    "金\tidiom",
    "中国\tidiom",
    "日本語\tidiom",
    "ありがとう\tidiom",
    "안녕하세요\tidiom",
    "张三丰\tname",
    "こんにちは\tname",
)

#: The fixture's own account of how Rust reproduces its inputs.
DEFINITIONS: dict[str, str] = {
    "draw": (
        "draw(seed, x) = -(splitmix64(seed ^ x) >> 44) / 65536, with splitmix64 the "
        "finaliser of SplitMix64 (add 0x9E3779B97F4A7C15, then xor-shift 30 and multiply by "
        "0xBF58476D1CE4E5B9, xor-shift 27 and multiply by 0x94D049BB133111EB, xor-shift 31), "
        "wrapping mod 2^64; the value is a multiple of 2^-16 in (-16, 0], exact in f32."
    ),
    "emission": (
        "log_probs[t][c] = draw(emission_seed, (t << 32) | c) for key t and emission index "
        "c < E, and log_probs[t][E] (the blank) = draw(emission_seed, (t << 32) | E). "
        "Under emission 'flat' every character's c is 0, so the characters of a key tie."
    ),
    "transition": (
        "A prefix of emission ids i_1..i_k hashes as h_0 = 0, h_j = splitmix64(h_(j-1) ^ i_j); "
        "its row is row[c] = draw(transition_seed, h_k ^ c) for c < E. Under transition 'flat' "
        "every row is 0.0."
    ),
    "settings": (
        "beta is 1 throughout; beam is 8, or null (no pruning) for records of at most "
        f"{UNBOUNDED_MAX_KEYS} keys; top_k is null, so the search returns the beam's worth, or "
        "every legal output without a beam. 'calls' lists the prefixes of each transition "
        "call, in call order, as emission-id lists."
    ),
    "arcs": (
        "A record with a 'lexicon' reads the header's lexicon rows over its alphabet for the "
        "header's categories (an entry is a row's word: at least two characters, all in the "
        "alphabet, as emission ids) and enumerates 'words' over its lattice: entry ids, start "
        "and end cut, and the chains, each the indices of its lattice arcs in the order the "
        "lattice's edges list them (by start, then end, then emission id), sorted by start, "
        "entry, end. 'lexicon' counts the entries kept and the rows skipped as too short or "
        "outside the alphabet. A setting of such a record adds 'prior' and 'arc_width', its "
        "hypotheses their 'arcs', and 'counts' the arcs offered, the arcs landed and the "
        "transition rows asked for landings alone. A record without a lexicon runs at prior 0."
    ),
}


@dataclass(frozen=True)
class FixtureRecord:
    """One record of the fixture: keys, the emittable set they are searched over, the inputs."""

    name: str
    keys: str
    #: The key of :data:`ALPHABETS` the record's emittable set is.
    alphabet: str
    emission: str
    transition: str
    seed: int
    #: Whether the record runs with the arc lexicon over the arc grid.
    lexicon: bool = False


#: Hand-picked characters the hand cases need, beyond the training sentences'.
_HAND_EXTRA = "水饺睡脚家铺铲西安先现线县显噷哼好们和吗哈美没是时十石史师世市事室式"

#: Characters a short training record's emittable set holds beyond its sentence's.
_SHORT_EXTRA = "洲年快乐是时逐光就"

#: The training records the fixture draws on, by key string: one of each style.
TRAINING_KEYS = (
    "shzhnkl",
    "zhgjj",
    "zhuguangjianjin",
    "liylngetaihlel",
    "zhgwshtajrdxshf",
    "baoqinijiuyigefeiben",
    "zhegenvrenkebuhaore",
    "yuminongtanggaifan",
    "shxjmentzhzhxgnian",
)

#: The hand cases: ``(name, keys, alphabet, emission, transition)``.
_HAND_CASES: tuple[tuple[str, str, str, str, str], ...] = (
    ("hm", "hm", "hm", "hash", "hash"),
    ("xian", "xian", "xian", "hash", "hash"),
    ("shuijiap", "shuijiap", "shuijiap", "hash", "hash"),
    ("tie-homophones", "shishishi", "shi", "flat", "flat"),
    ("tie-initials", "sjsjsjsjs", "initials", "flat", "flat"),
    ("tie-small", "shishi", "small", "flat", "flat"),
)

#: The arc cases: ``(name, keys, alphabet, emission, transition)``. The flat ones tie.
_ARC_CASES: tuple[tuple[str, str, str, str, str], ...] = (
    ("arc-small", "shsh", "small", "hash", "hash"),
    ("arc-tie-small", "shsh", "small", "flat", "flat"),
    ("arc-initials", "sjsjsjsjs", "initials", "hash", "hash"),
    ("arc-tie-initials", "sjsjsjsjs", "initials", "flat", "flat"),
    ("arc-shi", "shishishi", "shi", "hash", "hash"),
)

#: The training records (indices of :data:`TRAINING_KEYS`) that also run with arcs.
ARC_TRAINING = (0, 1, 2, 3, 4, 6, 8)

#: The small alphabets of the hand cases.
_HAND_ALPHABETS = {
    "hm": "噷哼好们和吗哈美没",
    "xian": "西安先现线县显鲜",
    "shuijiap": "水饺睡脚家铺铲",
    "shi": "是时十石史师世市事室式",
    "initials": "是时十石史师世市事室式就金捡",
    "small": "是时十",
}


def _splitmix_scalar(value: int) -> int:
    """SplitMix64's finaliser of one integer, through :func:`mlime.legality._splitmix`."""
    return int(splitmix(np.array([value], dtype=np.uint64))[0])


def draw(seed: int, values: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]:
    """``draw(seed, x)`` of the fixture's :data:`DEFINITIONS`, elementwise."""
    mixed = splitmix(np.bitwise_xor(values, np.uint64(seed)))
    return -(np.right_shift(mixed, np.uint64(44)).astype(np.float64)) / 65536.0


def emissions(keys: int, width: int, seed: int, flat: bool) -> npt.NDArray[np.float32]:
    """The record's ``[keys, width + 1]`` log-probs, the blank last."""
    rows = np.arange(keys, dtype=np.uint64)[:, None] << np.uint64(32)
    columns = np.arange(width + 1, dtype=np.uint64)
    if flat:
        columns = np.where(columns == width, columns, np.uint64(0))
    return draw(seed, rows | columns[None, :]).astype(np.float32)


class HashedTransition:
    """The fixture's transition: rows are SplitMix64 draws over a prefix's rolling hash."""

    def __init__(self, width: int, seed: int, flat: bool) -> None:
        self.width = width
        self.seed = seed
        self.flat = flat
        self.hashes: dict[Prefix, int] = {(): 0}
        #: The prefixes of each call, in call order.
        self.calls: list[list[list[int]]] = []

    def hash(self, prefix: Prefix) -> int:
        """The rolling hash of *prefix*."""
        known = self.hashes.get(prefix)
        if known is None:
            known = _splitmix_scalar(self.hash(prefix[:-1]) ^ prefix[-1])
            self.hashes[prefix] = known
        return known

    def __call__(self, records: Sequence[int], prefixes: Sequence[Prefix]) -> npt.NDArray[Any]:
        self.calls.append([list(prefix) for prefix in prefixes])
        if self.flat:
            return np.zeros((len(prefixes), self.width), dtype=np.float32)
        columns = np.arange(self.width, dtype=np.uint64)
        heads = np.array([self.hash(prefix) for prefix in prefixes], dtype=np.uint64)
        return draw(self.seed, heads[:, None] ^ columns[None, :]).astype(np.float32)


def _alphabets(training: Sequence[tuple[str, str]]) -> dict[str, str]:
    """The named emittable sets: the shared one and the hand cases' own."""
    shared = "".join(dict.fromkeys("".join(text for _, text in training) + _HAND_EXTRA))
    return {"shared": shared, **_HAND_ALPHABETS}


def _records(training: Sequence[tuple[str, str]]) -> list[FixtureRecord]:
    """The fixture's records: the training key strings, then the hand cases."""
    records: list[FixtureRecord] = []
    for index, (keys, _) in enumerate(training):
        short = len(keys) <= UNBOUNDED_MAX_KEYS
        # A short record runs without a beam, so it gets an emittable set small
        # enough for every legal output to be listed.
        alphabet = f"train-{index}" if short else "shared"
        records.append(
            FixtureRecord(f"train-{index}", keys, alphabet, "hash", "hash", 1000 + index)
        )
    for index, (name, keys, alphabet, emission, transition) in enumerate(_HAND_CASES):
        records.append(FixtureRecord(name, keys, alphabet, emission, transition, 2000 + index))
    for index, (name, keys, alphabet, emission, transition) in enumerate(_ARC_CASES):
        records.append(
            FixtureRecord(name, keys, alphabet, emission, transition, 3000 + index, lexicon=True)
        )
    for index in ARC_TRAINING:
        keys, _ = training[index]
        short = len(keys) <= UNBOUNDED_MAX_KEYS
        alphabet = f"train-{index}" if short else "shared"
        records.append(
            FixtureRecord(
                f"arc-train-{index}", keys, alphabet, "hash", "hash", 3100 + index, lexicon=True
            )
        )
    return records


def _hypothesis(hypothesis: Hypothesis, *, arcs: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ids": list(hypothesis.ids),
        "cuts": list(hypothesis.cuts),
        "score": hypothesis.score,
        "ctc": hypothesis.ctc,
        "transition": hypothesis.transition,
        "correction": hypothesis.correction,
    }
    if arcs:
        row["arcs"] = hypothesis.arcs
    return row


def _word(word: WordArc) -> dict[str, Any]:
    return {
        "entry": list(word.entry),
        "start": word.start,
        "end": word.end,
        "chains": word.chains.tolist(),
    }


def _search(
    lattice_legality: Legality,
    record: FixtureRecord,
    *,
    viterbi: bool,
    beam: int | None,
    gamma: float,
    words: Sequence[WordArc] | None = None,
    prior: float = 0.0,
    arc_width: int = DEFAULT_ARC_WIDTH,
) -> tuple[list[Hypothesis], HashedTransition, ArcCounts]:
    """The Python search over one record: hypotheses, the transition's calls, the arc counts."""
    width = len(lattice_legality.characters)
    lattice = lattice_legality.lattice(record.keys, OFFLINE)
    log_probs = torch.from_numpy(
        emissions(len(record.keys), width, record.seed, record.emission == "flat")
    )[None]
    transition = HashedTransition(width, record.seed ^ _TRANSITION_XOR, record.transition == "flat")
    counts = ArcCounts()
    [found] = search(
        [lattice],
        log_probs,
        transition,
        beta=BETA,
        gamma=gamma,
        beam=beam,
        viterbi=viterbi,
        words=None if words is None else [words],
        prior=prior,
        arc_width=arc_width,
        counts=counts,
    )
    return found, transition, counts


def run_setting(
    lattice_legality: Legality,
    record: FixtureRecord,
    *,
    viterbi: bool,
    beam: int | None,
    gamma: float,
) -> dict[str, Any]:
    """One record under one setting: the search's hypotheses and its transition calls."""
    found, transition, _ = _search(
        lattice_legality, record, viterbi=viterbi, beam=beam, gamma=gamma
    )
    return {
        "combine": "viterbi" if viterbi else "sum",
        "beam": beam,
        "gamma": gamma,
        "hypotheses": [_hypothesis(hypothesis, arcs=False) for hypothesis in found],
        "calls": transition.calls,
    }


def run_arc_setting(
    lattice_legality: Legality,
    record: FixtureRecord,
    words: Sequence[WordArc],
    *,
    viterbi: bool,
    beam: int | None,
    prior: float,
    arc_width: int,
) -> dict[str, Any]:
    """One arc record under one setting: :func:`run_setting` with the arcs, their counts too."""
    found, transition, counts = _search(
        lattice_legality,
        record,
        viterbi=viterbi,
        beam=beam,
        gamma=ARC_GAMMA,
        words=words,
        prior=prior,
        arc_width=arc_width,
    )
    return {
        "combine": "viterbi" if viterbi else "sum",
        "beam": beam,
        "gamma": ARC_GAMMA,
        "prior": prior,
        "arc_width": arc_width,
        "hypotheses": [_hypothesis(hypothesis, arcs=True) for hypothesis in found],
        "calls": transition.calls,
        "counts": {"offered": counts.offered, "landed": counts.landed, "rows": counts.rows},
    }


#: The transition seed is the emission seed xor this.
_TRANSITION_XOR = 0x5DEECE66D1CE4E5B


def build_fixture(
    legality: Legality, char_table: Path, legality_fixture: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The fixture's header and its records, searched under every setting."""
    document = json.loads(legality_fixture.read_text(encoding="utf-8"))
    sentences = {row["keys"]: row["sentence"] for row in document["records"]}
    training = [(keys, sentences[keys]) for keys in TRAINING_KEYS]
    alphabets = _alphabets(training)
    for index, (keys, text) in enumerate(training):
        if len(keys) <= UNBOUNDED_MAX_KEYS:
            alphabets[f"train-{index}"] = "".join(dict.fromkeys(text + _SHORT_EXTRA))
    readings = read_char_readings(char_table)
    automata = {
        name: Legality(tuple(characters), readings, legality.corrections)
        for name, characters in alphabets.items()
    }
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as scratch:
        rows = Path(scratch) / "lexicon.tsv"
        rows.write_text("\n".join(LEXICON_ROWS) + "\n", encoding="utf-8")
        for record in _records(training):
            records.append(_record(record, automata[record.alphabet], rows))
    header: dict[str, Any] = {
        "definitions": DEFINITIONS,
        "beta": BETA,
        "alphabets": alphabets,
        "lexicon": {"categories": list(LEXICON_CATEGORIES), "rows": list(LEXICON_ROWS)},
    }
    return header, records


def _record(record: FixtureRecord, automaton: Legality, rows: Path) -> dict[str, Any]:
    """One record of the fixture, searched under every setting of its grid."""
    beams = (BEAM, None) if len(record.keys) <= UNBOUNDED_MAX_KEYS else (BEAM,)
    document: dict[str, Any] = {
        "name": record.name,
        "keys": record.keys,
        "alphabet": record.alphabet,
        "emission": record.emission,
        "transition": record.transition,
        "emission_seed": f"{record.seed:016x}",
        "transition_seed": f"{record.seed ^ _TRANSITION_XOR:016x}",
    }
    if not record.lexicon:
        document["settings"] = [
            run_setting(automaton, record, viterbi=viterbi, beam=beam, gamma=gamma)
            for viterbi in (False, True)
            for beam in beams
            for gamma in GAMMAS
        ]
        return document
    lexicon = ArcLexicon.load(rows, LEXICON_CATEGORIES, automaton.characters)
    words = lexicon.arcs(automaton.lattice(record.keys, OFFLINE), record.keys)
    document["lexicon"] = {
        "entries": lexicon.size,
        "skipped_short": lexicon.skipped_short,
        "skipped_outside": lexicon.skipped_outside,
    }
    document["words"] = [
        _word(word) for word in sorted(words, key=lambda arc: (arc.start, arc.entry, arc.end))
    ]
    document["settings"] = [
        run_arc_setting(
            automaton,
            record,
            words,
            viterbi=viterbi,
            beam=beam,
            prior=prior,
            arc_width=arc_width,
        )
        for viterbi in (False, True)
        for beam in beams
        for prior in ARC_PRIORS
        for arc_width in ARC_WIDTHS
    ]
    return document


def write_search_fixture(
    legality: Legality, char_table: Path, legality_fixture: Path, out: Path
) -> int:
    """Write the shared key-search fixture and return how many (record, setting) runs it holds.

    The training key strings come from the legality fixture's records, so the
    file is rebuilt from the committed inputs. One line per record keeps the
    file diffable.
    """
    header, records = build_fixture(legality, char_table, legality_fixture)
    lines = ["{"]
    for name, value in header.items():
        lines.append(f"{json.dumps(name)}: {json.dumps(value, ensure_ascii=False)},")
    lines.append('"records": [')
    lines.append(
        ",\n".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records
        )
    )
    lines.append("]")
    lines.append("}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sum(len(record["settings"]) for record in records)
