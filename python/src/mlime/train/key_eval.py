"""Decoding a key-level checkpoint over evaluation records, and its report (#257).

The span model's evaluation decodes a record over the lattice's readings of
its keystrokes (:mod:`mlime.train.e2e_eval`). The key model reads the
keystrokes themselves: a record's keys are its pinyin with the syllable
separators removed -- exactly what a typist sends -- and its context, and
nothing about how the keys segment reaches the model. The search is the
constrained beam over the legality automaton (:mod:`mlime.train.key_search`),
scored by the CTC head plus ``beta`` times the decoder's transition minus
``gamma`` times the correction cost.

**Index space.** The search's lattices are built by the automaton the model
was trained against: :meth:`~mlime.train.run.Vocabularies.legality`, over the
lexicon's emittable set in emission order. An arc's character is therefore an
index into the lexicon, which is the CTC head's row and the decoder's
emission column, and an arc for a character the head cannot emit cannot be
built at all -- the automaton is constructed over the emittable set rather
than filtered afterwards. :class:`KeyDecoder` refuses an automaton over any
other set or a head of another width, and the search refuses an arc past the
head (:func:`~mlime.train.key_search.search`).

**Batching.** Records are grouped under a token budget over the keys' fill
row and the context row, as training groups them. Per group the encoder runs
once (:meth:`~mlime.train.key_model.KeyE2EModel.key_states`), the CTC head
once over every key, and the transition is the decoder's batched query: the
search asks once per key offset for every prefix of every record that is
about to extend, and :class:`ModelTransition` answers all of them in one
call. Only the search's bookkeeping is per record. A grid of weights reuses a
group's encoding and lattices -- neither depends on the weights -- and re-runs
the search per grid point: which hypotheses a beam keeps depends on both
weights, so rescoring one point's list under another is exact only for the
hypotheses it kept (:mod:`mlime.train.key_search`).

**Reports.** The counting, the slices and the dump are the span evaluation's
(:class:`~mlime.train.e2e_eval.SliceReports`, the fused-eval dump shape), so a
key model's numbers are comparable with the span model's on the same records.
Each dumped hypothesis also carries its terms -- ``ctc``, ``transition``,
``correction`` -- beside ``score``, and is checked by the independent
:class:`~mlime.train.key_search.LegalityVerifier` before it is written: an
illegal one raises with its record (#216 §8 item 5).

**Weights on dev.** :func:`select_on_dev` is ``fused-eval --select-on-dev``:
every grid point decodes the dev slice, the best by dev sentence top-1 is
chosen (ties: top-8, then the smaller ``beta``, then the first given), and
only then is the test slice decoded, at the chosen weights alone. The choice
is per evaluation set and context setting, as the notes' sweeps make it.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from itertools import batched
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from mlime.legality import KeyLattice, Legality, LegalityOptions
from mlime.logging import log
from mlime.train.e2e_eval import EvalRow, Report, SliceReports, read_eval_records
from mlime.train.key_model import KeyE2EModel, KeyStates
from mlime.train.key_search import Hypothesis, LegalityVerifier, Prefix, Transition, search
from mlime.train.keys import KeyCollator
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.samples import (
    DEFAULT_CONTEXT_TOKENS,
    BaseTokenizer,
    BudgetPacker,
    Collator,
    ContextTokenizer,
    context_width,
)
from mlime.train.spans import SpanVocab

#: The search's width and the report's depth: the product's shape, eight beams
#: a step (AGENTS.md), and fused-eval's top-8.
DEFAULT_BEAM = 8
DEFAULT_TOP_K = 8

#: Records decoded between two writes of the dumps: the progress a stopped run keeps.
DEFAULT_RECORDS_PER_CHUNK = 256

#: The syllable separator a pinyin string may carry; a typist's keys have none.
SEPARATOR = "'"

#: Slack on the verifier's cost against the search's float sums.
COST_TOLERANCE = 1e-6

#: The selection's record beside the grid's dumps.
SELECTION_FILE = "selection.json"


def typed_keys(pinyin: str) -> str:
    """The keystrokes *pinyin* stands for: its letters, the separators removed."""
    keys = pinyin.replace(SEPARATOR, "")
    if not keys or not all("a" <= key <= "z" for key in keys):
        raise ValueError(f"pinyin {pinyin!r} is not a run of [a-z] keystrokes")
    return keys


@dataclass(frozen=True)
class KeyRecord:
    """One evaluation record as the key model reads it."""

    #: Its zero-based line in the eval set, blank lines excluded.
    record: int
    keys: str
    text: str
    context: str | None


@dataclass(frozen=True)
class KeyWeights:
    """One point of the search's weights: the transition's, the corrections', the combination."""

    beta: float
    gamma: float
    viterbi: bool

    def __post_init__(self) -> None:
        if not (math.isfinite(self.beta) and self.beta >= 0.0):
            raise ValueError(f"beta must be finite and not negative, got {self.beta}")
        if not (math.isfinite(self.gamma) and self.gamma >= 0.0):
            raise ValueError(f"gamma must be finite and not negative, got {self.gamma}")

    @property
    def combination(self) -> str:
        """How a prefix's covers combine: ``viterbi`` (max) or ``sum`` (log-sum-exp)."""
        return "viterbi" if self.viterbi else "sum"

    @property
    def name(self) -> str:
        """A file-name-safe label, unique per point."""
        return f"beta{self.beta:g}-gamma{self.gamma:g}-{self.combination}"

    def as_record(self) -> dict[str, Any]:
        """The point as JSON fields."""
        return {"beta": self.beta, "gamma": self.gamma, "combination": self.combination}


def weight_grid(
    betas: Sequence[float], gammas: Sequence[float], combinations: Sequence[bool]
) -> list[KeyWeights]:
    """Every ``beta x gamma x combination``, in the order given: a dev tie goes to the first."""
    grid = [
        KeyWeights(beta, gamma, viterbi)
        for beta in betas
        for gamma in gammas
        for viterbi in combinations
    ]
    if not grid:
        raise ValueError("the weight grid is empty")
    if len(set(grid)) != len(grid):
        raise ValueError(f"the weight grid repeats a point: {[point.name for point in grid]}")
    return grid


@dataclass
class DecodeTimes:
    """Where a decode's wall time went, and how much it decoded."""

    #: Records decoded, and the grid points each was searched at.
    records: int = 0
    searches: int = 0
    #: Building the records' lattices on the host.
    lattice: float = 0.0
    #: The encoder and the CTC head, and pulling nothing back yet.
    encode: float = 0.0
    #: The decoder's transition queries, their transfer to the host included.
    transition: float = 0.0
    #: The search's bookkeeping, the transition's share taken out.
    search: float = 0.0
    #: The verifier's check of every hypothesis written.
    verify: float = 0.0

    def add(self, other: DecodeTimes) -> None:
        """Accumulate *other* into this."""
        for name, value in asdict(other).items():
            setattr(self, name, getattr(self, name) + value)

    @property
    def total(self) -> float:
        """Every phase together."""
        return self.lattice + self.encode + self.transition + self.search + self.verify

    def per_record(self, grid_points: int) -> float:
        """The seconds one record would take decoded at *grid_points* points.

        A record's lattice and encoding are paid once, its search, transition
        queries and verification once per point.
        """
        if not self.records or not self.searches:
            raise ValueError("nothing was decoded to measure")
        once = (self.lattice + self.encode) / self.records
        each = (self.transition + self.search + self.verify) / self.searches
        return once + grid_points * each


class ModelTransition:
    """The search's transition over one group's key states: the decoder's batched query.

    The search hands every prefix about to extend at a key offset, across the
    group's records, in one call; they go to
    :meth:`~mlime.train.key_model.KeyE2EModel.transition_log_probs` as one
    padded batch, and the rows come back to the host, where the search reads
    them.
    """

    def __init__(self, model: KeyE2EModel, states: KeyStates, times: DecodeTimes) -> None:
        self.model = model
        self.states = states
        self.times = times

    def __call__(self, records: Sequence[int], prefixes: Sequence[Prefix]) -> torch.Tensor:
        started = time.perf_counter()
        device = self.states.states.device
        lengths = np.fromiter((len(prefix) for prefix in prefixes), dtype=np.int64)
        padded = np.zeros((len(prefixes), int(lengths.max(initial=0))), dtype=np.int64)
        for row, prefix in enumerate(prefixes):
            padded[row, : len(prefix)] = prefix
        rows = self.model.transition_log_probs(
            self.states,
            torch.tensor(list(records), dtype=torch.long, device=device),
            torch.from_numpy(padded).to(device),
            torch.from_numpy(lengths).to(device),
        ).cpu()
        self.times.transition += time.perf_counter() - started
        return rows


def search_points(
    lattices: Sequence[KeyLattice],
    log_probs: torch.Tensor,
    transition: Transition,
    grid: Sequence[KeyWeights],
    *,
    beam: int,
    top_k: int,
    times: DecodeTimes,
) -> list[list[list[Hypothesis]]]:
    """The search at every grid point over one group's encoding: ``[point][record]``.

    Neither the lattices nor the emissions depend on the weights, so a group's
    encoding serves every point; which hypotheses a beam keeps does, so each
    point runs its own search. The transition's time is whatever the
    transition added to ``times.transition`` during the call, and is taken out
    of the search's.
    """
    found = []
    for weights in grid:
        queried = times.transition
        began = time.perf_counter()
        found.append(
            search(
                lattices,
                log_probs,
                transition,
                beta=weights.beta,
                gamma=weights.gamma,
                beam=beam,
                top_k=top_k,
                viterbi=weights.viterbi,
            )
        )
        spent = time.perf_counter() - began
        times.search += spent - (times.transition - queried)
    return found


class Decoding(Protocol):
    """What a :class:`KeyRun` asks of a decoder: the PyTorch model's, or an export's."""

    @property
    def lexicon(self) -> Lexicon:
        """The emittable set the hypotheses' indices name."""
        ...

    @property
    def options(self) -> LegalityOptions:
        """The automaton's options the lattices were built under."""
        ...

    @property
    def top_k(self) -> int:
        """Hypotheses kept per record."""
        ...

    @property
    def times(self) -> DecodeTimes:
        """Where the wall time goes."""
        ...

    def decode(
        self, records: Sequence[KeyRecord], grid: Sequence[KeyWeights]
    ) -> list[list[list[Hypothesis]]]:
        """Each grid point's hypotheses for each record, best first: ``[point][record]``."""
        ...


def build_key_collator(
    spans: SpanVocab,
    lexicon: Lexicon,
    tokenizer: BaseTokenizer,
    context_tokenizer: ContextTokenizer,
    *,
    with_context: bool,
    max_context_tokens: int,
) -> KeyCollator:
    """The collator a decode of typed records reads its keys and contexts through.

    *with_context* off drops every context, as a context dropout of 1 does in
    training.
    """
    return KeyCollator(
        Collator(
            tokenizer,
            context_tokenizer,
            CandidateSpace(spans, lexicon.candidate_mask),
            context_dropout=0.0 if with_context else 1.0,
            max_context_tokens=max_context_tokens,
        )
    )


def check_automaton(legality: Legality, lexicon: Lexicon, options: LegalityOptions) -> None:
    """Refuse an automaton a decode of complete records cannot use.

    Its arcs must name the head's rows, so it emits over the lexicon's set in
    the lexicon's order, and an evaluation record is complete input, so its
    tail rule is off.
    """
    if legality.characters != lexicon.characters:
        raise ValueError(
            "the legality automaton emits over a different set than the model's lexicon; "
            "its arcs would not name the head's rows"
        )
    if options.unfinished_tail:
        raise ValueError("an evaluation record is complete input; the tail rule is off")


@dataclass(frozen=True)
class KeyDecoder:
    """Everything a constrained decode of typed records needs, fixed for a run."""

    model: KeyE2EModel
    lexicon: Lexicon
    collator: KeyCollator
    legality: Legality
    options: LegalityOptions
    device: torch.device
    token_budget: int
    beam: int = DEFAULT_BEAM
    top_k: int = DEFAULT_TOP_K
    times: DecodeTimes = field(default_factory=DecodeTimes)

    def __post_init__(self) -> None:
        emittable = int(self.model.decoder.emit_ids.numel())
        if emittable != self.lexicon.size:
            raise ValueError(
                f"the model emits {emittable} characters against the lexicon's {self.lexicon.size}"
            )
        check_automaton(self.legality, self.lexicon, self.options)

    def groups(self, records: Sequence[KeyRecord]) -> Iterator[list[KeyRecord]]:
        """*records* in order, grouped under the token budget as training groups examples."""
        packer = BudgetPacker(self.token_budget)
        group: list[KeyRecord] = []
        limit = self.collator.max_context_tokens
        for record in records:
            if packer.add(len(record.keys) + 2, context_width(record.context, limit), 1):
                yield group
                group = []
            group.append(record)
        if group:
            yield group

    @torch.no_grad()
    def decode(
        self, records: Sequence[KeyRecord], grid: Sequence[KeyWeights]
    ) -> list[list[list[Hypothesis]]]:
        """Each grid point's hypotheses for each record, best first: ``[point][record]``."""
        if self.model.training:
            raise ValueError("decode a model in evaluation mode")
        found: list[list[list[Hypothesis]]] = [[] for _ in grid]
        width = self.lexicon.size
        for group in self.groups(records):
            started = time.perf_counter()
            lattices = [self.legality.lattice(record.keys, self.options) for record in group]
            encoding = time.perf_counter()
            inputs = self.collator.inputs(
                [record.keys for record in group], [record.context for record in group]
            ).to(self.device)
            states = self.model.key_states(inputs)
            log_probs = self.model.ctc_log_probs(states.states)
            if log_probs.shape[-1] != width + 1:
                raise ValueError(
                    f"the head emits {log_probs.shape[-1] - 1} characters against the "
                    f"lexicon's {width}"
                )
            if log_probs.is_cuda:
                torch.cuda.synchronize(log_probs.device)
            searching = time.perf_counter()
            self.times.lattice += encoding - started
            self.times.encode += searching - encoding
            transition = ModelTransition(self.model, states, self.times)
            for point, hypotheses in enumerate(
                search_points(
                    lattices,
                    log_probs,
                    transition,
                    grid,
                    beam=self.beam,
                    top_k=self.top_k,
                    times=self.times,
                )
            ):
                found[point].extend(hypotheses)
            self.times.records += len(group)
            self.times.searches += len(group) * len(grid)
        return found


def eval_key_records(rows: Sequence[EvalRow]) -> list[KeyRecord]:
    """An eval set's rows as the key model reads them."""
    return [
        KeyRecord(record=index, keys=typed_keys(row.pinyin), text=row.text, context=row.context)
        for index, row in enumerate(rows)
    ]


def checked_row(
    record: KeyRecord,
    hypotheses: Sequence[Hypothesis],
    characters: Sequence[str],
    verifier: LegalityVerifier,
    options: LegalityOptions,
) -> dict[str, Any]:
    """*record*'s dump row, every hypothesis first checked legal by the verifier.

    An illegal hypothesis, or one the search priced below the cheapest legal
    cover the verifier finds, is a defect in the automaton or the search, and
    raises with the record and the hypothesis.
    """
    written = []
    for rank, hypothesis in enumerate(hypotheses):
        sentence = "".join(characters[index] for index in hypothesis.ids)
        cost = verifier.cost(record.keys, sentence, options)
        if cost is None:
            raise ValueError(
                f"record {record.record} ({record.keys!r}, expected {record.text!r}): "
                f"hypothesis {rank} {sentence!r} (cuts {hypothesis.cuts}) is not a legal "
                "reading of its keys"
            )
        if cost > hypothesis.correction + COST_TOLERANCE:
            raise ValueError(
                f"record {record.record} ({record.keys!r}): hypothesis {rank} {sentence!r} "
                f"claims correction cost {hypothesis.correction} below the cheapest legal "
                f"cover's {cost}"
            )
        written.append(
            {
                "text": sentence,
                "score": hypothesis.score,
                "ctc": hypothesis.ctc,
                "transition": hypothesis.transition,
                "correction": hypothesis.correction,
                "cuts": list(hypothesis.cuts),
            }
        )
    return {"record": record.record, "text": record.text, "hypotheses": written}


@dataclass(frozen=True)
class KeyRun:
    """One evaluation set under one context setting: what every decode of it shares."""

    decoder: Decoding
    verifier: LegalityVerifier
    eval_rows: Sequence[EvalRow]
    dev_share: float
    records_per_chunk: int = DEFAULT_RECORDS_PER_CHUNK
    #: Decode only the first this-many records of each slice; ``None`` decodes it whole.
    limit: int | None = None

    def indices(self, slice_: str) -> list[int]:
        """The eval-set indices *slice_* decodes, in file order, cut to :attr:`limit`."""
        held = SliceReports(self.eval_rows, slice_, self.dev_share, self.decoder.top_k).indices
        return list(held if self.limit is None else held[: self.limit])

    def evaluate(
        self, slice_: str, grid: Sequence[KeyWeights], dumps: Sequence[Path]
    ) -> list[list[tuple[str, Report]]]:
        """Decode *slice_* at every grid point; each point's ``(name, report)`` pairs.

        Point ``i`` writes its dump to ``dumps[i]`` in the fused-eval shape,
        with each hypothesis's terms beside its score. A dump is also its
        point's progress: started again, a point folds the records its dump
        holds, and a record is decoded only at the points that lack it.
        """
        if len(dumps) != len(grid) or len(set(dumps)) != len(dumps):
            raise ValueError(f"{len(grid)} grid points need as many distinct dumps")
        decoder = self.decoder
        characters = decoder.lexicon.characters
        options = decoder.options
        records = eval_key_records(self.eval_rows)
        sliced = [SliceReports(self.eval_rows, slice_, self.dev_share, decoder.top_k) for _ in grid]
        done = [reports.resume(dump) for reports, dump in zip(sliced, dumps, strict=True)]
        pending = [
            records[index]
            for index in self.indices(slice_)
            if any(index not in held for held in done)
        ]
        decoded = 0
        started = time.monotonic()
        sinks = [dump.open("a", encoding="utf-8") for dump in dumps]
        try:
            for chunk in batched(pending, self.records_per_chunk):
                needed = [
                    point
                    for point, held in enumerate(done)
                    if any(record.record not in held for record in chunk)
                ]
                found = decoder.decode(chunk, [grid[point] for point in needed])
                began = time.perf_counter()
                for point, hypotheses in zip(needed, found, strict=True):
                    for record, own in zip(chunk, hypotheses, strict=True):
                        if record.record in done[point]:
                            continue
                        row = checked_row(record, own, characters, self.verifier, options)
                        sliced[point].fold(
                            record.record, [entry["text"] for entry in row["hypotheses"]]
                        )
                        sinks[point].write(json.dumps(row, ensure_ascii=False) + "\n")
                    sinks[point].flush()
                decoder.times.verify += time.perf_counter() - began
                decoded += len(chunk)
                log.info(
                    "key records decoded",
                    slice=slice_,
                    decoded=decoded,
                    pending=len(pending),
                    grid_points=len(needed),
                    records_per_second=round(decoded / (time.monotonic() - started), 2),
                    seconds={
                        name: round(value, 2) for name, value in asdict(decoder.times).items()
                    },
                )
        finally:
            for sink in sinks:
                sink.close()
        return [reports.results() for reports in sliced]


def dump_path(directory: Path, slice_: str, weights: KeyWeights) -> Path:
    """Where the decode of *slice_* at *weights* goes inside a run's dump directory."""
    return directory / f"{slice_}-{weights.name}.jsonl"


def better(candidate: tuple[KeyWeights, Report], incumbent: tuple[KeyWeights, Report]) -> bool:
    """Whether *candidate* beats *incumbent* on dev: top-1, then top-k, then the smaller beta."""
    (weights, report), (held, best) = candidate, incumbent
    return (report.top1_hits, report.topk_hits, -weights.beta) > (
        best.top1_hits,
        best.topk_hits,
        -held.beta,
    )


def choose(grid: Sequence[tuple[KeyWeights, Report]]) -> KeyWeights:
    """The dev winner (:func:`better`); a full tie goes to the point given first."""
    if not grid:
        raise ValueError("nothing to choose from")
    winner = grid[0]
    for entry in grid[1:]:
        if better(entry, winner):
            winner = entry
    return winner[0]


@dataclass(frozen=True)
class Selection:
    """A dev sweep and its outcome: every point's dev report and the chosen weights."""

    grid: tuple[tuple[KeyWeights, Report], ...]
    chosen: KeyWeights
    #: Records in the dev and test slices.
    dev_records: int
    test_records: int
    #: The test slice's decode at the chosen weights, estimated from this
    #: sweep's measured rate; None when nothing was decoded to measure.
    test_seconds_estimate: float | None

    def as_record(self) -> dict[str, Any]:
        """The selection as JSON."""
        return {
            "grid": [{**weights.as_record(), **vars(report)} for weights, report in self.grid],
            "chosen": self.chosen.as_record(),
            "dev_records": self.dev_records,
            "test_records": self.test_records,
            "test_seconds_estimate": self.test_seconds_estimate,
        }

    def render(self) -> str:
        """The dev sweep as a table, the chosen point marked."""
        lines = [
            f"{'beta':>6}  {'gamma':>6}  {'combine':<8}  top-1    top-k    char     MRR",
            f"{'-' * 6}  {'-' * 6}  {'-' * 8}  {'-' * 6}   {'-' * 6}   {'-' * 6}   {'-' * 6}",
        ]
        for weights, report in self.grid:
            records = report.records or 1
            mark = "  <- chosen" if weights == self.chosen else ""
            lines.append(
                f"{weights.beta:>6g}  {weights.gamma:>6g}  {weights.combination:<8}  "
                f"{report.top1_hits / records:.4f}   {report.topk_hits / records:.4f}   "
                f"{report.character_hits / (report.characters or 1):.4f}   "
                f"{report.reciprocal_ranks / (1 << 20) / records:.4f}{mark}"
            )
        return "\n".join(lines)


def select_on_dev(run: KeyRun, grid: Sequence[KeyWeights], directory: Path) -> Selection:
    """Decode the dev slice at every grid point and choose; nothing of test is decoded.

    The selection is written to :data:`SELECTION_FILE` in *directory*, beside
    the grid's dumps. A sweep resumed from its dumps decodes nothing and
    measures no rate; the estimate then comes from the selection file an
    earlier sweep of the same grid wrote, when there is one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    before = DecodeTimes(**asdict(run.decoder.times))
    sections = run.evaluate("dev", grid, [dump_path(directory, "dev", point) for point in grid])
    measured = DecodeTimes(**asdict(run.decoder.times))
    for name, value in asdict(before).items():
        setattr(measured, name, getattr(measured, name) - value)
    reports = tuple(
        (weights, dict(section)["dev"]) for weights, section in zip(grid, sections, strict=True)
    )
    chosen = choose(reports)
    dev = len(run.indices("dev"))
    test = len(run.indices("test"))
    estimate: float | None = None
    if measured.records:
        estimate = test * measured.per_record(1)
    else:
        previous = directory / SELECTION_FILE
        if previous.is_file():
            estimate = json.loads(previous.read_text(encoding="utf-8"))["test_seconds_estimate"]
    selection = Selection(
        grid=reports,
        chosen=chosen,
        dev_records=dev,
        test_records=test,
        test_seconds_estimate=estimate,
    )
    (directory / SELECTION_FILE).write_text(
        json.dumps(selection.as_record(), indent=2) + "\n", encoding="utf-8"
    )
    log.info(
        "weights chosen on dev",
        chosen=chosen.name,
        dev_records=dev,
        test_seconds_estimate=estimate,
    )
    return selection


def decode_test(run: KeyRun, selection: Selection, directory: Path) -> Report:
    """The test slice decoded at the weights *selection* chose, and nothing else."""
    chosen = selection.chosen
    [section] = run.evaluate("test", [chosen], [dump_path(directory, "test", chosen)])
    return dict(section)["test"]


def load_key_run(
    checkpoint: Path,
    eval_set: Path,
    *,
    char_table: Path,
    typo_table: Path,
    syllables: Path,
    with_context: bool,
    token_budget: int,
    dev_share: float,
    beam: int = DEFAULT_BEAM,
    top_k: int = DEFAULT_TOP_K,
    records_per_chunk: int = DEFAULT_RECORDS_PER_CHUNK,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    limit: int | None = None,
) -> tuple[KeyRun, int]:
    """Everything an evaluation of *checkpoint* on *eval_set* needs, and the checkpoint's step.

    The lexicon, the tokenizers and the automaton are built the way the run
    that trained the checkpoint built them (:class:`~mlime.train.run.Vocabularies`);
    the verifier reads the committed tables directly. *with_context* off
    drops every context, as a context dropout of 1 does in training.
    """
    from mlime.train.e2e import load_key_model, recorded_config
    from mlime.train.run import Vocabularies

    recorded = recorded_config(checkpoint)
    if recorded.keys is None:
        raise ValueError(f"{checkpoint} is a span model; `mlime eval e2e` decodes it")
    vocabularies = Vocabularies.load(char_table, recorded.base_model, recorded.context)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step = load_key_model(checkpoint, vocabularies.lexicon, device)
    collator = build_key_collator(
        vocabularies.spans,
        vocabularies.lexicon,
        vocabularies.tokenizer,
        vocabularies.context_tokenizer,
        with_context=with_context,
        max_context_tokens=max_context_tokens,
    )
    decoder = KeyDecoder(
        model=model,
        lexicon=vocabularies.lexicon,
        collator=collator,
        legality=vocabularies.legality(),
        options=recorded.keys.options,
        device=device,
        token_budget=token_budget,
        beam=beam,
        top_k=top_k,
    )
    run = build_key_run(
        decoder,
        eval_set,
        char_table=char_table,
        typo_table=typo_table,
        syllables=syllables,
        dev_share=dev_share,
        records_per_chunk=records_per_chunk,
        limit=limit,
    )
    return run, step


def build_key_run(
    decoder: Decoding,
    eval_set: Path,
    *,
    char_table: Path,
    typo_table: Path,
    syllables: Path,
    dev_share: float,
    records_per_chunk: int = DEFAULT_RECORDS_PER_CHUNK,
    limit: int | None = None,
) -> KeyRun:
    """*decoder* over *eval_set*, its hypotheses checked by the verifier over the tables."""
    return KeyRun(
        decoder=decoder,
        verifier=LegalityVerifier.load(
            char_table=char_table, typo_table=typo_table, syllables=syllables
        ),
        eval_rows=read_eval_records(eval_set),
        dev_share=dev_share,
        records_per_chunk=records_per_chunk,
        limit=limit,
    )


Ambiguity = Callable[[Path, str], str]


def report_keys(
    run: KeyRun,
    directory: Path,
    grid: Sequence[KeyWeights],
    *,
    slice_: str | None,
    selection_only: bool = False,
    ambiguity: Ambiguity | None = None,
) -> str:
    """Decode and render: a fixed point over *slice_*, or with *slice_* None the dev sweep.

    The sweep (:func:`select_on_dev`) chooses on dev and then decodes test at
    the choice, unless *selection_only* stops it after the choice.
    *ambiguity*, given, renders a dump's ambiguity slice by name; each
    reported section gets it.
    """
    directory.mkdir(parents=True, exist_ok=True)

    def section(title: str, report: Report, dump: Path, name: str) -> str:
        text = f"slice: {title}\n{report.render()}"
        if ambiguity is not None:
            text += "\n\n" + ambiguity(dump, name).rstrip("\n")
        return text

    if slice_ is not None:
        if selection_only:
            raise ValueError("only a dev sweep stops after its selection")
        if len(grid) != 1:
            raise ValueError(
                f"a fixed-weight run decodes at one point, got {len(grid)}; "
                "sweep several with the dev selection"
            )
        [weights] = grid
        dump = dump_path(directory, slice_, weights)
        [results] = run.evaluate(slice_, [weights], [dump])
        return "\n\n".join(
            section(f"{name} ({weights.name})", report, dump, name) for name, report in results
        )
    selection = select_on_dev(run, grid, directory)
    chosen = selection.chosen
    parts = [
        f"slice: dev, the weight sweep ({selection.dev_records} records)\n{selection.render()}",
        section(
            f"dev ({chosen.name}, chosen)",
            dict(selection.grid)[chosen],
            dump_path(directory, "dev", chosen),
            "dev",
        ),
    ]
    if not selection_only:
        report = decode_test(run, selection, directory)
        parts.append(
            section(
                f"test ({chosen.name}, chosen on dev)",
                report,
                dump_path(directory, "test", chosen),
                "test",
            )
        )
    return "\n\n".join(parts)
