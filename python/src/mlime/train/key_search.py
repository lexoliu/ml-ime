"""The constrained beam search over keys, and the independent legality verifier (#216 §3, §5, §8).

**The search.** The key model's head gives, at every key ``t`` of a record with
``n`` keys, a log-softmax over the ``E`` emittable characters and a blank (the
last column). An output is a cover of the keys by contiguous spans with one
character per span, emitted at the span's last key, every other key blank; it
is legal when each span matches its character under the legality automaton
(:class:`~mlime.legality.Legality`), possibly at a correction cost ``w``. A
path -- cut points ``0 = b_0 < ... < b_m = n`` and characters ``c_1..c_m`` --
scores

    sum of blank log-probs over the non-end keys + sum_j emit_{b_j - 1}(c_j)
    + beta * sum_j T(c_j | c_1..c_{j-1}) - gamma * sum_j w(span_j, c_j)

with ``T`` the transition the caller supplies. Illegal characters are masked,
never renormalised (#115): the search only ever reads the automaton's arcs,
at the head's own log-probabilities.

Written with ``P(t)``, the summed blank log-probability of keys ``0..t-1``, the
blank terms telescope: a path scores ``P(n) + sum_j g_j`` where the *gain* of
emitting ``c_j`` at key ``b_j - 1`` is ``emit - blank`` at that key, plus
``beta * T - gamma * w``. ``P(n)`` is the record's constant, so every
hypothesis of a record is carried as a sum of gains alone and two hypotheses
at the same key compare without any blank bookkeeping: each has paid, in
effect, the blanks of every key up to the present.

The search walks key offsets ``1..n`` frame-synchronously over the batch. A
hypothesis is a character prefix; it holds one state per open cut point (the
offset its last character ended at), and a state is the combination over the
covers of the prefix that end there -- log-sum-exp, the CTC sum, or with
``viterbi`` the max. At offset ``e`` every state whose cut ``c`` has an arc
``(c, e)`` emits each character the arc admits; extensions that land on the
same ``(prefix, e)`` combine, and an extension that lands on a prefix already
in the beam adds a cut to it. A state whose open span ``(c, e]`` cannot grow
into any arc any more is dead and dropped. The arcs are the automaton's
:meth:`~mlime.legality.Legality.lattice`: every legal arc of the keys, pruned
only to complete covers, so no reading list stands between the keys and the
search and a dropped state is one no complete output passes through. The only
other pruning is the beam: after each offset the ``beam`` best prefixes stay,
with all of their live states.

**Ranking an open span (decision 4).** A state whose span is still open has
not paid for the character it will emit, and one that just emitted has not
paid for the next one either. Each state is ranked by its gains so far plus
an admissible optimistic bound on the gain of its *next* character: the best
``emit - blank - gamma * w`` over every arc that starts at its cut and ends
after the current key, with the transition's share bounded by 0 (a log-prob
is at most 0, so the bound needs ``beta >= 0``, which the search requires).
That puts every state on the same footing -- all its committed characters
priced exactly, exactly one more character priced optimistically -- whether
its span is open or just closed, and a state whose span is about to close
cheaply is not outranked by one that merely has not paid yet. The bound is a
per-record table over (cut, key), built once from the gathered emissions; it
needs no transition row, so ranking never queries the transition for
prefixes that are about to be pruned. A prefix's rank is the combination
(log-sum-exp, or max with ``viterbi``) of its live states' bounded values.
At the last key only states with a cut at ``n`` remain and are ranked by
their exact scores. The bound changes only which prefixes the beam keeps:
with ``beam=None`` nothing is pruned and every legal output comes back at
exactly its brute-force score, in both combinations.

**The transition** is queried once per distinct prefix of a record, never per
key or per cut: a prefix's next-character distribution does not depend on
where its next character lands. Before each offset every prefix in the beam
that is about to extend and has no row yet is collected across the whole
batch, and the transition is called once with all of them. A row is kept
while the prefix, or a prefix of it, is in the beam -- the only way it can be
extended again -- so no prefix is asked twice.

**Emissions** are pulled to the host once per batch: one gather on the
tensor's own device collects every arc's emission log-prob and every key's
blank, and one transfer brings them over.

**The breakdown.** Each hypothesis carries ``transition``, the summed
``T`` of its characters, which is the same over all of its covers and so
exact; ``ctc`` and ``correction``, the CTC term and the summed correction
cost of its best cover (``cuts``). With ``viterbi`` the score is exactly
``ctc + beta * transition - gamma * correction`` (plus ``pi * arcs``, see
below); with the sum it is the log-sum-exp over the covers the beam kept.
Either way ``score - beta * transition`` does not depend on ``beta``, so the
returned list rescored under another ``beta`` is exact. Under another ``gamma`` the best cover may
change, so a rescored ``ctc - gamma' * correction`` is a lower bound (the old
cover's score). Neither re-weighting is the search re-run: which hypotheses
the beam keeps depends on both weights.

**Exactness under the beam.** With a finite beam a kept output's score need
not be its brute-force score in either combination: a prefix pruned while
one of its spans is open takes with it the covers of its descendants that
would have passed through that state, and a descendant already in the beam
keeps the covers it has. No prefix beam avoids that. What does hold: every
output is legal, and with ``viterbi`` its score is exactly the score of
``cuts``, one of its legal covers, so at most its brute-force score.

**Word arcs (#282).** With a lexicon (:mod:`mlime.train.key_arcs`) and a prior
``pi > 0`` the search proposes whole entries. A *word arc* is an entry
``c_1..c_L`` and a chain of the lattice's own arcs from a cut ``a`` to a cut
``b`` emitting it, at least one span typed as an initial; its cost is the
chain's, so it cannot make an illegal output. Arcs are enumerated per record
before the search and the best ``arc_width`` ending at each offset ``b`` are
kept, by their optimistic gain (the best chain's summed gains plus ``pi``);
the rest are never offered, so the bound below prices only what can land.

*The prior.* A cover's weight is raised by ``pi`` for the arcs it contains. A
cover is a character sequence with its cut points; it contains the arc of
entry ``E`` at characters ``i..j`` when ``E`` is those characters and one of
their spans is initial-typed. The landing mechanics below count a *set of
non-overlapping contained arcs*, not every contained arc, and that is the
definition: under ``viterbi`` the cover gains ``pi`` times the size of its
largest set of non-overlapping contained arcs; under the sum its mass is
multiplied by ``sum_S (e^pi - 1)^|S|`` over its sets ``S`` of non-overlapping
contained arcs. Arcs that do not overlap multiply (two of them: ``e^(2 pi)``),
so the sum is ``e^(pi k)`` for ``k`` of them; overlapping arcs (``AB`` and
``ABC``) are alternatives and do not stack. At ``pi = 0`` the search ignores
the lexicon altogether: a decode is the decode without arcs, hypothesis for
hypothesis, as it is when no arc matches.

*Landing.* At offset ``b`` every beam prefix ``P`` with a live cut at ``a``
lands each kept arc ``(E, a, b)`` whole: the prefix ``Q = P + E`` gets a state
at cut ``b`` holding the chain's gains, the transition's log-probs of ``E``'s
characters in turn (the rows of ``P + c_1..c_k``, ``k < L``, are asked through
the same :class:`Transition` in rounds, round ``k`` for every record at once,
and cached with every other row, so under the same eviction: a row stays while
a prefix of it is in the beam, hence as long as ``P`` can still land it and
not beyond), the chains' correction costs, the cut points and the prior.
Several chains of one arc combine as covers. The landed mass is the
*additional* mass the prior gives: ``mass * (e^pi - 1)``, in log space
``mass + pi + log(1 - e^-pi)``, added by log-add-exp to the state the
character-by-character path reached for ``(Q, b)``; under ``viterbi`` the
landing is ``best + pi`` merged by max. When the character-by-character path
did not reach ``(Q, b)`` -- its inner prefixes fell out of the beam, which is
what an arc is for -- the landing creates the state holding the chain's whole
mass, ``mass * e^pi``. Arcs landing on one absent ``Q`` from different entries
add their masses, which counts covers they share twice; that happens only for
a pruned lineage under a finite beam. With ``beam=None`` every ``(Q, b)`` is
reached both ways and the sum is exactly the formula above.

*Protection is the bound.* A cut with a kept arc that ends after the current
key is not dead, and the bound of an open span at that cut, for keys before
``b``, also takes the arc's optimistic gain: still exactly one more *event*
priced optimistically, here the whole arc, with the transition's share
bounded by 0. A prefix about to land an idiom therefore ranks by what the
idiom will pay, which keeps it in the beam; nothing is kept beside the beam.

*Counts.* :class:`ArcCounts` counts the arcs offered (enumerated), the arcs
landed (kept, with at least one prefix landing them) and the transition rows
asked only for landings.

**The breakdown with arcs.** ``arcs`` is the number of arcs in the best
cover's chosen set, so with ``viterbi`` the score is exactly ``ctc + beta *
transition - gamma * correction + pi * arcs``.

**The verifier** (#216 §8 item 5) is :class:`LegalityVerifier`: a plain
dynamic program over ``char_pinyin.tsv`` and the corrections function that
decides whether a sentence is a legal reading of keys, and at what cost. It
shares no code with the automaton or the search; it reads the same committed
tables.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from mlime.legality import OFFLINE, KeyLattice, Legality, LegalityOptions
from mlime.train.key_arcs import ArcLexicon, WordArc
from mlime.typo import (
    CORRECTIONS_MAX_EDITS,
    MAX_SYLLABLE_LEN,
    MULTI_LETTER_INITIALS,
    CorrectionTable,
    NoiseModel,
    load_syllables,
)

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]
Prefix = tuple[int, ...]

_NEGATIVE = -math.inf

#: Arcs landing per record per key offset (#282): the product's shape, as the beam's eight.
DEFAULT_ARC_WIDTH = 8


class Transition(Protocol):
    """The search's transition: next-character log-probs for a batch of prefixes.

    ``records[i]`` is the index, within the batch being searched, of the
    record prefix ``prefixes[i]`` (emission indices) belongs to; the pairs are
    distinct. Returns ``[len(prefixes), E]``: row ``i`` is ``log T(. |
    prefixes[i])`` over the emittable set, on any device.
    """

    def __call__(
        self, records: Sequence[int], prefixes: Sequence[Prefix]
    ) -> torch.Tensor | npt.NDArray[Any]: ...


@dataclass(frozen=True)
class Hypothesis:
    """One output of the search, with its score's terms."""

    #: The output's characters, as emission indices.
    ids: Prefix
    #: Its score: the combination over its kept covers of the path score.
    score: float
    #: The best cover's CTC term: its blanks and its characters' emissions.
    ctc: float
    #: The summed transition log-prob of its characters, the same on every cover.
    transition: float
    #: The best cover's summed correction cost.
    correction: float
    #: The best cover's cut points ``b_1..b_m``; ``b_m`` is the key count.
    cuts: tuple[int, ...]
    #: The arcs in the best cover's chosen set: its score carries ``pi`` for each.
    arcs: int


@dataclass
class ArcCounts:
    """What the word arcs cost a search: the work bound's evidence."""

    #: Arcs enumerated over the records' lattices, before the per-offset width.
    offered: int = 0
    #: Kept arcs that at least one beam prefix landed.
    landed: int = 0
    #: Transition rows asked for landings alone, on top of the beam's own.
    rows: int = 0


@dataclass
class _Cut:
    """A prefix's state at one cut point: the covers of the prefix ending there."""

    #: The combined gains, without the transition, over the covers.
    mass: float
    #: The best cover's gains, without the transition.
    best: float
    #: The best cover's ``sum (emit - blank)``.
    ctc: float
    #: The best cover's summed correction cost.
    correction: float
    #: The best cover's cut points.
    cover: tuple[int, ...]
    #: The arcs in the best cover's chosen set.
    arcs: int


@dataclass(frozen=True)
class _Landing:
    """A word arc priced against its record: what a prefix landing it gains, whichever it is."""

    entry: Prefix
    start: int
    end: int
    #: The chains' summed gains combined as covers, and the best chain's alone.
    mass: float
    best: float
    #: The best chain's ``sum (emit - blank)``, summed correction cost and cut points.
    ctc: float
    correction: float
    cuts: tuple[int, ...]
    #: The best chain's gains plus the prior: what the bound prices the arc at.
    optimistic: float


@dataclass
class _Arrival:
    """Every arc landing on one prefix at one offset, combined."""

    transition: float
    #: The additional mass the arcs give when the character-by-character path
    #: reached the prefix.
    extra: float
    #: The best landing's cut with its own whole mass and the prior: what
    #: stands in when that path did not reach the prefix. Landings that overlap
    #: are alternatives of one cover, so their whole masses are not summed and
    #: the state is a lower bound of the covers' mass.
    cut: _Cut


@dataclass
class _Prefix:
    """A hypothesis: a character prefix, its transition sum and its live cuts."""

    transition: float
    cuts: dict[int, _Cut]


@dataclass
class _Extensions:
    """Every character one beam prefix can emit at the current key, combined per character."""

    key: Prefix
    chars: Ints
    mass: Floats
    best: Floats
    ctc: Floats
    correction: Floats
    origin: Ints
    sources: dict[int, _Cut]
    transition: Floats
    score: Floats


def _combine(values: Sequence[float], viterbi: bool) -> float:
    """Log-sum-exp of *values*, or their max under *viterbi*."""
    top = max(values)
    if viterbi or len(values) == 1 or top == _NEGATIVE:
        return top
    return top + math.log(sum(math.exp(value - top) for value in values))


class _Record:
    """One record's arcs, gains, bound table and beam."""

    def __init__(
        self,
        lattice: KeyLattice,
        arc_emit: Floats,
        blank: Floats,
        gamma: float,
        words: Sequence[WordArc] = (),
        *,
        viterbi: bool = False,
        prior: float = 0.0,
        arc_width: int = DEFAULT_ARC_WIDTH,
        counts: ArcCounts | None = None,
    ) -> None:
        n = lattice.keys
        self.n = n
        self.blank_total = float(blank[:n].sum())
        self.chars: Ints = lattice.character.astype(np.int64)
        self.cost: Floats = lattice.cost.astype(np.float64)
        ends = lattice.end.astype(np.int64)
        starts = lattice.start.astype(np.int64)
        self.ctc_gain: Floats = arc_emit - blank[ends - 1]
        self.gain: Floats = self.ctc_gain - gamma * self.cost
        #: Arcs ending at each offset, by start: ``by_end[e][c] = (lo, hi)``.
        self.by_end: dict[int, dict[int, tuple[int, int]]] = {}
        #: The furthest end of any arc from each offset; -1 where none starts.
        self.max_end = np.full(n + 1, -1, dtype=np.int64)
        #: ``bound[c, t]``: the best gain, the transition's share bounded by 0,
        #: of an arc from ``c`` ending after key ``t``.
        self.bound = np.full((n + 1, n + 1), _NEGATIVE, dtype=np.float64)
        if self.chars.size:
            group = starts * (n + 1) + ends
            heads = np.flatnonzero(np.diff(group, prepend=-1))
            tails = np.append(heads[1:], group.size)
            best = np.maximum.reduceat(self.gain, heads)
            for lo, hi, top in zip(heads.tolist(), tails.tolist(), best.tolist(), strict=True):
                start, end = int(starts[lo]), int(ends[lo])
                self.by_end.setdefault(end, {})[start] = (lo, hi)
                self.max_end[start] = max(int(self.max_end[start]), end)
                row = self.bound[start, :end]
                np.maximum(row, top, out=row)
            root = _Prefix(0.0, {0: _Cut(0.0, 0.0, 0.0, 0.0, (), 0)})
            self.beam: dict[Prefix, _Prefix] = {(): root}
        else:
            self.beam = {}
        self.viterbi = viterbi
        self.prior = prior
        #: The extra mass a landed arc adds in the sum: ``log(e^pi - 1)``.
        self.bonus = prior + math.log1p(-math.exp(-prior)) if prior > 0 else _NEGATIVE
        self.counts = counts if counts is not None else ArcCounts()
        #: The kept arcs by their end offset, and the plan of the current offset:
        #: which beam prefix lands which arc from which state.
        self.landing: dict[int, list[_Landing]] = {}
        self.plan: list[tuple[Prefix, _Landing, _Cut]] = []
        if words and prior > 0 and self.chars.size:
            self._keep(lattice, words, arc_width)
        self.rows: dict[Prefix, npt.NDArray[np.float32]] = {}
        self.results: list[Hypothesis] = []

    def _keep(self, lattice: KeyLattice, words: Sequence[WordArc], width: int) -> None:
        """Price the arcs, keep the best *width* ending at each offset, and open their cuts.

        A kept arc keeps its start cut alive until it can land and raises the
        bound of the open span there by its optimistic gain.
        """
        ends = lattice.end.astype(np.int64)
        by_end: dict[int, list[_Landing]] = {}
        for word in words:
            gains = self.gain[word.chains].sum(axis=1)
            top = int(np.argmax(gains))
            by_end.setdefault(word.end, []).append(
                _Landing(
                    entry=word.entry,
                    start=word.start,
                    end=word.end,
                    mass=_combine(gains.tolist(), self.viterbi),
                    best=float(gains[top]),
                    ctc=float(self.ctc_gain[word.chains[top]].sum()),
                    correction=float(self.cost[word.chains[top]].sum()),
                    cuts=tuple(ends[word.chains[top]].tolist()),
                    optimistic=float(gains[top]) + self.prior,
                )
            )
        self.counts.offered += len(words)
        for end, group in by_end.items():
            group.sort(key=lambda arc: (-arc.optimistic, arc.start, arc.entry))
            self.landing[end] = group[:width]
            for arc in self.landing[end]:
                row = self.bound[arc.start, : arc.end]
                np.maximum(row, arc.optimistic, out=row)
                self.max_end[arc.start] = max(int(self.max_end[arc.start]), arc.end)

    def prepare(self, end: int) -> None:
        """Plan offset *end*'s landings: each kept arc lands from every prefix with its cut live."""
        self.plan = []
        landed = 0
        for arc in self.landing.get(end, ()):
            count = len(self.plan)
            self.plan.extend(
                (key, arc, prefix.cuts[arc.start])
                for key, prefix in self.beam.items()
                if arc.start in prefix.cuts
            )
            landed += len(self.plan) > count
        self.counts.landed += landed

    def requests(self, end: int, depth: int) -> tuple[list[Prefix], int]:
        """The prefixes whose rows round *depth* asks for, and how many only landings ask.

        Round 0 is the beam's own (:meth:`needs`) and each landing's source
        prefix; round ``k`` the inner prefixes ``P + c_1..c_k`` of the entries
        longer than ``k``.
        """
        asked = dict.fromkeys(self.needs(end)) if depth == 0 else {}
        beams = len(asked)
        for key, arc, _ in self.plan:
            if len(arc.entry) > depth:
                inner = (*key, *arc.entry[:depth])
                if inner not in self.rows:
                    asked.setdefault(inner)
        return list(asked), len(asked) - beams

    @property
    def depth(self) -> int:
        """The rounds of transition rows the current plan needs: its longest entry."""
        return max((len(arc.entry) for _, arc, _ in self.plan), default=0)

    def arrivals(self, beta: float) -> dict[Prefix, _Arrival]:
        """The planned landings combined per landed prefix, with their transition sums."""
        out: dict[Prefix, _Arrival] = {}
        for key, arc, state in self.plan:
            landed = (*key, *arc.entry)
            transition = self.transition_sum(key, arc.entry)
            base = state.mass + arc.mass
            whole = base + self.prior
            extra = whole if self.viterbi else base + self.bonus
            cut = _Cut(
                mass=whole,
                best=state.best + arc.best + self.prior,
                ctc=state.ctc + arc.ctc,
                correction=state.correction + arc.correction,
                cover=(*state.cover, *arc.cuts),
                arcs=state.arcs + 1,
            )
            held = out.get(landed)
            if held is None:
                out[landed] = _Arrival(transition, extra, cut)
                continue
            held.extra = _combine((held.extra, extra), self.viterbi)
            if cut.best > held.cut.best:
                held.cut = cut
        return out

    def transition_sum(self, key: Prefix, entry: Prefix) -> float:
        """``T(entry | key)`` summed character by character, then added to the prefix's own."""
        total = self.beam[key].transition
        for step, character in enumerate(entry):
            total = total + float(self.rows[(*key, *entry[:step])][character])
        return total

    def needs(self, end: int) -> list[Prefix]:
        """The beam prefixes that emit at *end* and have no transition row yet."""
        arcs = self.by_end.get(end)
        if not arcs:
            return []
        return [
            key
            for key, prefix in self.beam.items()
            if key not in self.rows and any(cut in arcs for cut in prefix.cuts)
        ]

    def extensions(self, end: int, beta: float, viterbi: bool) -> list[_Extensions]:
        """Every prefix's emissions at key ``end - 1``, combined per (prefix, character)."""
        arcs = self.by_end.get(end, {})
        out: list[_Extensions] = []
        for key, prefix in self.beam.items():
            sources = [(cut, state, arcs[cut]) for cut, state in prefix.cuts.items() if cut in arcs]
            if not sources:
                continue
            parts = []
            for cut, state, (lo, hi) in sources:
                gain = self.gain[lo:hi]
                parts.append(
                    (
                        self.chars[lo:hi],
                        state.mass + gain,
                        state.best + gain,
                        state.ctc + self.ctc_gain[lo:hi],
                        state.correction + self.cost[lo:hi],
                        np.full(hi - lo, cut, dtype=np.int64),
                    )
                )
            if len(parts) == 1:
                chars, mass, best, ctc, correction, origin = parts[0]
            else:
                chars, mass, best, ctc, correction, origin = (
                    np.concatenate(column) for column in zip(*parts, strict=True)
                )
                # By character, the best cover first within each.
                order = np.lexsort((-best, chars))
                chars, mass, best = chars[order], mass[order], best[order]
                ctc, correction, origin = ctc[order], correction[order], origin[order]
                heads = np.flatnonzero(np.diff(chars, prepend=-1))
                mass = (np.maximum if viterbi else np.logaddexp).reduceat(mass, heads)
                chars, best = chars[heads], best[heads]
                ctc, correction, origin = ctc[heads], correction[heads], origin[heads]
            transition = prefix.transition + self.rows[key][chars].astype(np.float64)
            out.append(
                _Extensions(
                    key=key,
                    chars=chars,
                    mass=mass,
                    best=best,
                    ctc=ctc,
                    correction=correction,
                    origin=origin,
                    sources={cut: state for cut, state, _ in sources},
                    transition=transition,
                    score=mass + beta * transition,
                )
            )
        return out

    def step(
        self, end: int, beta: float, beam: int | None, top_k: int | None, viterbi: bool
    ) -> None:
        """Consume key ``end - 1``: emit, drop dead cuts, merge, prune to the beam."""
        groups = self.extensions(end, beta, viterbi)
        by_parent = {group.key: index for index, group in enumerate(groups)}
        taken = [np.zeros(group.chars.size, dtype=bool) for group in groups]

        def landed(index: int, at: int) -> _Cut:
            group = groups[index]
            source = group.sources[int(group.origin[at])]
            return _Cut(
                mass=float(group.mass[at]),
                best=float(group.best[at]),
                ctc=float(group.ctc[at]),
                correction=float(group.correction[at]),
                cover=(*source.cover, end),
                arcs=source.arcs,
            )

        # Old prefixes: drop the cuts whose span cannot grow past this key, and
        # take the cut an extension of their parent lands on here.
        survivors: dict[Prefix, _Prefix] = {}
        for key, prefix in self.beam.items():
            cuts = {cut: state for cut, state in prefix.cuts.items() if self.max_end[cut] > end}
            if key:
                index = by_parent.get(key[:-1])
                if index is not None:
                    chars = groups[index].chars
                    at = int(np.searchsorted(chars, key[-1]))
                    if at < chars.size and chars[at] == key[-1]:
                        taken[index][at] = True
                        cuts[end] = landed(index, at)
            if cuts:
                survivors[key] = _Prefix(prefix.transition, cuts)
        final = end == self.n
        limit = top_k if final else beam
        # New prefixes: every new one has the one cut ``end``, so the bound
        # they are ranked by is common and the best ``limit`` by score are all
        # that can make the beam.
        fresh: list[tuple[int, int]] = []
        if groups:
            owner = np.concatenate(
                [np.full(group.chars.size, index) for index, group in enumerate(groups)]
            )
            position = np.concatenate([np.arange(group.chars.size) for group in groups])
            scores = np.concatenate([group.score for group in groups])
            open_ = ~np.concatenate(taken)
            owner, position, scores = owner[open_], position[open_], scores[open_]
            if limit is not None and scores.size > limit:
                # Exact ties at the boundary go to the earlier extension, so the
                # selection is fully determined.
                keep = np.sort(np.argsort(-scores, kind="stable")[:limit])
                owner, position = owner[keep], position[keep]
            fresh = list(zip(owner.tolist(), position.tolist(), strict=True))
        for index, at in fresh:
            group = groups[index]
            survivors[(*group.key, int(group.chars[at]))] = _Prefix(
                float(group.transition[at]), {end: landed(index, at)}
            )
        for key, arrival in self.arrivals(beta).items():
            held = survivors.get(key)
            if held is None:
                survivors[key] = _Prefix(arrival.transition, {end: arrival.cut})
            else:
                held.cuts[end] = self.merged(held.cuts.get(end), arrival)
        if final:
            self._finish(survivors, beta, top_k)
            return
        if beam is not None and len(survivors) > beam:
            ranks = {
                key: _combine(
                    [
                        state.mass + beta * prefix.transition + float(self.bound[cut, end])
                        for cut, state in prefix.cuts.items()
                    ],
                    viterbi,
                )
                for key, prefix in survivors.items()
            }
            kept = sorted(survivors, key=lambda key: -ranks[key])[:beam]
            survivors = {key: survivors[key] for key in kept}
        self.beam = survivors
        self.evict()

    def merged(self, reached: _Cut | None, arrival: _Arrival) -> _Cut:
        """The state at a landed prefix's cut: what the character-wise path reached, plus the arcs.

        Reached, the arcs add their additional mass to it; not reached, they
        stand for it and hold the covers' whole mass with the prior.
        """
        if reached is None:
            return arrival.cut
        mass = _combine((reached.mass, arrival.extra), self.viterbi)
        winner = arrival.cut if arrival.cut.best > reached.best else reached
        return _Cut(
            mass=mass,
            best=winner.best,
            ctc=winner.ctc,
            correction=winner.correction,
            cover=winner.cover,
            arcs=winner.arcs,
        )

    def _finish(self, outputs: dict[Prefix, _Prefix], beta: float, top_k: int | None) -> None:
        """The complete outputs at the last key, best first."""
        results = []
        for key, prefix in outputs.items():
            state = prefix.cuts[self.n]
            results.append(
                Hypothesis(
                    ids=key,
                    score=state.mass + beta * prefix.transition + self.blank_total,
                    ctc=state.ctc + self.blank_total,
                    transition=prefix.transition,
                    correction=state.correction,
                    cuts=state.cover,
                    arcs=state.arcs,
                )
            )
        results.sort(key=lambda hypothesis: -hypothesis.score)
        self.results = results if top_k is None else results[:top_k]
        self.beam = {}
        self.rows = {}

    def evict(self) -> None:
        """Drop the rows of prefixes no beam prefix can extend into again."""
        beam = self.beam
        self.rows = {
            key: row
            for key, row in self.rows.items()
            if any(key[:length] in beam for length in range(len(key) + 1))
        }


def _host_rows(output: torch.Tensor | npt.NDArray[Any], count: int, width: int) -> Any:
    """The transition's output as a host ``[count, width]`` float32 array."""
    if isinstance(output, torch.Tensor):
        rows = output.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        rows = np.asarray(output, dtype=np.float32)
    if rows.shape != (count, width):
        raise ValueError(f"the transition returned {rows.shape} for {count} prefixes of {width}")
    return rows


def search(
    lattices: Sequence[KeyLattice],
    log_probs: torch.Tensor,
    transition: Transition,
    *,
    beta: float,
    gamma: float,
    beam: int | None = 8,
    top_k: int | None = None,
    viterbi: bool = False,
    words: Sequence[Sequence[WordArc]] | None = None,
    prior: float = 0.0,
    arc_width: int = DEFAULT_ARC_WIDTH,
    counts: ArcCounts | None = None,
) -> list[list[Hypothesis]]:
    """Each record's best legal outputs over its keys' lattice, best first.

    *lattices* are the records' :meth:`~mlime.legality.Legality.lattice`, and
    *log_probs* ``[B, T, E + 1]`` the head's log-softmax at each record's keys,
    the blank last, padded past each record's key count. *beam* is the number
    of prefixes kept after each key (``None``: no pruning); *top_k* the number
    of outputs returned (``None``: *beam*, or all of them without a beam).
    *viterbi* combines a prefix's covers by max instead of log-sum-exp. A
    record whose lattice has no complete cover gets no outputs.

    *words* are each record's word arcs (:meth:`~mlime.train.key_arcs.ArcLexicon.arcs`)
    and *prior* the ``pi`` each contained arc adds (see the module docstring);
    at ``prior == 0`` they are ignored and the search is the search without
    arcs. At most *arc_width* arcs per record and offset are kept, and *counts*,
    given, accumulates what they cost.
    """
    rows, width, columns = log_probs.shape
    if rows != len(lattices):
        raise ValueError(f"{len(lattices)} lattices against {rows} rows of log-probs")
    if beta < 0:
        raise ValueError(f"the open-span bound prices the transition at most 0; beta is {beta}")
    if beam is not None and beam < 1:
        raise ValueError(f"the beam keeps at least one prefix, got {beam}")
    if not (math.isfinite(prior) and prior >= 0.0):
        raise ValueError(f"the arc prior is finite and not negative, got {prior}")
    if arc_width < 1:
        raise ValueError(f"at least one arc lands per offset, got {arc_width}")
    if words is not None and len(words) != rows:
        raise ValueError(f"{len(words)} records of word arcs against {rows} rows of log-probs")
    if prior > 0 and words is None:
        raise ValueError("an arc prior above 0 needs the records' word arcs")
    if top_k is None:
        top_k = beam
    emittable = columns - 1
    for lattice in lattices:
        if lattice.keys > width:
            raise ValueError(f"a record has {lattice.keys} keys against {width} log-prob rows")
        if lattice.character.size and int(lattice.character.max()) >= emittable:
            raise ValueError(f"an arc emits a character past the head's {emittable}")
    counts = counts if counts is not None else ArcCounts()
    records = _gather(
        lattices,
        log_probs,
        gamma,
        words=words if prior > 0 else None,
        viterbi=viterbi,
        prior=prior,
        arc_width=arc_width,
        counts=counts,
    )
    longest = max((record.n for record in records), default=0)
    for end in range(1, longest + 1):
        live = [index for index, record in enumerate(records) if record.n >= end and record.beam]
        for index in live:
            records[index].prepare(end)
        # Round 0 asks the beam's rows and each landing's source prefix; round k
        # the inner prefixes of the entries longer than k, across the whole batch.
        for depth in range(max(1, max((records[index].depth for index in live), default=0))):
            owners: list[int] = []
            prefixes: list[Prefix] = []
            for index in live:
                asked, extra = records[index].requests(end, depth)
                counts.rows += extra
                owners.extend([index] * len(asked))
                prefixes.extend(asked)
            if prefixes:
                fetched = _host_rows(transition(owners, prefixes), len(prefixes), emittable)
                for index, key, row in zip(owners, prefixes, fetched, strict=True):
                    records[index].rows[key] = row
        for index in live:
            records[index].step(end, beta, beam, top_k, viterbi)
    return [record.results for record in records]


def _gather(
    lattices: Sequence[KeyLattice],
    log_probs: torch.Tensor,
    gamma: float,
    *,
    words: Sequence[Sequence[WordArc]] | None,
    viterbi: bool,
    prior: float,
    arc_width: int,
    counts: ArcCounts,
) -> list[_Record]:
    """Every arc's emission log-prob and every key's blank, in one transfer to the host."""
    _, width, columns = log_probs.shape
    flat = log_probs.reshape(-1)
    pieces = [
        (row * width + lattice.end.astype(np.int64) - 1) * columns + lattice.character
        for row, lattice in enumerate(lattices)
    ]
    sizes = [piece.size for piece in pieces]
    index = torch.from_numpy(np.concatenate([*pieces, np.empty(0, dtype=np.int64)]))
    blank = log_probs[..., -1].reshape(-1)
    pulled = torch.cat([flat.index_select(0, index.to(flat.device)), blank])
    host = pulled.detach().to(device="cpu", dtype=torch.float64).numpy()
    arcs = np.split(host[: sum(sizes)], np.cumsum(sizes)[:-1])
    blanks = host[sum(sizes) :].reshape(len(lattices), width)
    return [
        _Record(
            lattice,
            emit,
            blanks[row],
            gamma,
            () if words is None else words[row],
            viterbi=viterbi,
            prior=prior,
            arc_width=arc_width,
            counts=counts,
        )
        for row, (lattice, emit) in enumerate(zip(lattices, arcs, strict=True))
    ]


def search_keys(
    legality: Legality,
    keys: Sequence[str],
    log_probs: torch.Tensor,
    transition: Transition,
    *,
    beta: float,
    gamma: float,
    options: LegalityOptions = OFFLINE,
    beam: int | None = 8,
    top_k: int | None = None,
    viterbi: bool = False,
    arcs: ArcLexicon | None = None,
    prior: float = 0.0,
    arc_width: int = DEFAULT_ARC_WIDTH,
    counts: ArcCounts | None = None,
) -> list[list[Hypothesis]]:
    """:func:`search` over the keys' lattices under *options*, offline by default.

    With *arcs* and a *prior* above 0 each record's word arcs are enumerated
    from the lexicon over its lattice and keys.
    """
    lattices = [legality.lattice(each, options) for each in keys]
    words = None
    if arcs is not None and prior > 0:
        words = [arcs.arcs(lattice, each) for lattice, each in zip(lattices, keys, strict=True)]
    return search(
        lattices,
        log_probs,
        transition,
        beta=beta,
        gamma=gamma,
        beam=beam,
        top_k=top_k,
        viterbi=viterbi,
        words=words,
        prior=prior,
        arc_width=arc_width,
        counts=counts,
    )


#: The longest span any rule can match: a six-letter syllable two edits wide.
_LONGEST_SPAN = MAX_SYLLABLE_LEN + CORRECTIONS_MAX_EDITS


class LegalityVerifier:
    """Whether a sentence is a legal reading of keys, by a plain dynamic program.

    #216 §8 item 5's checker: the §3 definition applied directly, span by span,
    over ``char_pinyin.tsv``'s readings and ``corrections``, sharing no code
    with :class:`~mlime.legality.Legality` or the search.
    """

    def __init__(self, readings: Mapping[str, Sequence[str]], corrections: CorrectionTable):
        self._readings = {character: tuple(own) for character, own in readings.items()}
        self._corrections = corrections

    @classmethod
    def load(cls, *, char_table: Path, typo_table: Path, syllables: Path) -> LegalityVerifier:
        """The verifier over the committed tables."""
        readings: dict[str, tuple[str, ...]] = {}
        for line in char_table.read_text(encoding="utf-8").splitlines():
            character, _, own = line.partition("\t")
            readings[character] = tuple(own.split(","))
        return cls(
            readings, CorrectionTable(NoiseModel.load(typo_table), load_syllables(syllables))
        )

    def span_cost(self, span: str, last: bool, character: str, options: LegalityOptions) -> float:
        """The cheapest rule by which *span* reads as *character*; ``inf`` when none does."""
        cost = math.inf
        for reading in self._readings[character]:
            if span == reading:
                return 0.0
            initial = len(span) == 1 or span in MULTI_LETTER_INITIALS
            tail = last and options.unfinished_tail
            if (initial or tail) and reading.startswith(span):
                return 0.0
        corrects = (
            len(span) >= 2
            and span not in MULTI_LETTER_INITIALS
            and (options.correct_syllables or span not in self._corrections.syllables)
        )
        if corrects:
            entries = dict(self._corrections.corrections(span))
            for reading in self._readings[character]:
                cost = min(cost, entries.get(reading, math.inf))
        return cost

    def cost(self, keys: str, sentence: str, options: LegalityOptions) -> float | None:
        """*sentence*'s cheapest legal cover of *keys*; ``None`` when it has none."""
        if not keys or not all("a" <= key <= "z" for key in keys):
            raise ValueError(f"keys must be a non-empty [a-z] string, got {keys!r}")
        if not sentence or not all(character in self._readings for character in sentence):
            return None
        width = len(keys)
        reached = [0.0] + [math.inf] * width
        for character in sentence:
            following = [math.inf] * (width + 1)
            for start in range(width):
                if reached[start] == math.inf:
                    continue
                for end in range(start + 1, min(start + _LONGEST_SPAN, width) + 1):
                    cost = self.span_cost(keys[start:end], end == width, character, options)
                    following[end] = min(following[end], reached[start] + cost)
            reached = following
        return None if reached[width] == math.inf else reached[width]
