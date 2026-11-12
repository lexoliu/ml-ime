"""The shared fixture the Rust key search is held to (#265).

``crates/ime-decode/tests/fixtures/key-search.json`` pins :func:`mlime.train.key_search.search`
on synthetic inputs that both languages reproduce bit for bit, so the Rust port
needs no model: the emissions and the transition rows are functions of
SplitMix64 over integers, and the legality lattices are the real automaton's
over a small emittable set the fixture names.

Each record carries, per setting (combination, beam, gamma), the hypotheses the
Python search returned -- ids, cuts and the four score terms at ``repr``
precision -- and the sequence of transition calls it made.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mlime.legality import OFFLINE, Legality
from mlime.legality import _splitmix as splitmix
from mlime.train.key_search import Hypothesis, Prefix, search
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
    return records


def _hypothesis(hypothesis: Hypothesis) -> dict[str, Any]:
    return {
        "ids": list(hypothesis.ids),
        "cuts": list(hypothesis.cuts),
        "score": hypothesis.score,
        "ctc": hypothesis.ctc,
        "transition": hypothesis.transition,
        "correction": hypothesis.correction,
    }


def run_setting(
    lattice_legality: Legality,
    record: FixtureRecord,
    *,
    viterbi: bool,
    beam: int | None,
    gamma: float,
) -> dict[str, Any]:
    """One record under one setting: the search's hypotheses and its transition calls."""
    width = len(lattice_legality.characters)
    lattice = lattice_legality.lattice(record.keys, OFFLINE)
    log_probs = torch.from_numpy(
        emissions(len(record.keys), width, record.seed, record.emission == "flat")
    )[None]
    transition = HashedTransition(width, record.seed ^ _TRANSITION_XOR, record.transition == "flat")
    [found] = search(
        [lattice], log_probs, transition, beta=BETA, gamma=gamma, beam=beam, viterbi=viterbi
    )
    return {
        "combine": "viterbi" if viterbi else "sum",
        "beam": beam,
        "gamma": gamma,
        "hypotheses": [_hypothesis(hypothesis) for hypothesis in found],
        "calls": transition.calls,
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
    for record in _records(training):
        automaton = automata[record.alphabet]
        settings = [
            run_setting(automaton, record, viterbi=viterbi, beam=beam, gamma=gamma)
            for viterbi in (False, True)
            for beam in ((BEAM, None) if len(record.keys) <= UNBOUNDED_MAX_KEYS else (BEAM,))
            for gamma in GAMMAS
        ]
        records.append(
            {
                "name": record.name,
                "keys": record.keys,
                "alphabet": record.alphabet,
                "emission": record.emission,
                "transition": record.transition,
                "emission_seed": f"{record.seed:016x}",
                "transition_seed": f"{record.seed ^ _TRANSITION_XOR:016x}",
                "settings": settings,
            }
        )
    header: dict[str, Any] = {
        "definitions": DEFINITIONS,
        "beta": BETA,
        "alphabets": alphabets,
    }
    return header, records


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
