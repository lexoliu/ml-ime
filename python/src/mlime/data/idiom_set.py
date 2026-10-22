"""The idiom selection set (#243): idioms at eval3's training frequencies, judged, to annotate.

#212's word arcs carry a lexicon prior that dev cannot choose: dev holds too
few idioms, and test has been looked at. The prior is chosen instead on a set
of its own. How well the decoder's models already know an idiom depends on how
often run3 trained them on it, so the set draws its idioms at the run3
frequencies of eval3's, from sentences no model has trained on: #225's
ingested FineWeb2-HQ, MNBVC forum and YYeTs samples.

**The draw** (:func:`draw`). eval3's idioms are the idiom entries of #215's
table that its gold texts hold. Their run3 occurrence counts, cut at their
:data:`STRATA`-quantiles, make the strata (:class:`Strata`); the draw's size is
shared among the strata in eval3's proportions, largest remainder first, and
each stratum takes its idioms in a seeded order, one sentence each. An idiom
is eligible when the table lists it unblocked with run3 occurrences, the
knowledge lexicon files it as an idiom, every character of it is emittable,
and eval3 does not hold it. Its sentence is the first, in a seeded order of
the sentences holding it, that passes the judge's word-boundary rule, shares
nothing with an evaluation or held-out record, and was not drawn for another
idiom. A stratum without enough idioms fails the draw and names the largest
size that fits. The draw is written as a candidate set for ``mlime corpus
judge --input candidates``, with its census beside it.

**The top-up** (:func:`top_up`). The judge rejects rare idioms more often, so
what it keeps drifts from eval3's proportions. One more round, drawn once the
judge has finished the first, puts them back: :func:`plan` finds the size the
strata can reach at their round-one keep rates and each stratum's shortfall at
it, and each short stratum draws that shortfall over its keep rate. A stratum
continues its seeded idiom order past the idioms the first round drew; where it
has none left, its rejected idioms give their next admissible sentence. The
first round is drawn again from the same inputs and must be what the judge
read, and no record or sentence of it is drawn again. The top-up's files are
numbered after the first round's, so the judge resumes over the set and judges
only them.

**The set** (:func:`build`). Once the judge has finished every file of both
rounds, each idiom
contributes the record it kept, checked against evaluation once more, to a
samples shard ``ime-cli g2p annotate`` reads; the annotation and ``ime-cli
export eval-set --typing abbreviated`` make it a set the way eval3's
abbreviated twin was made.

A record collides with evaluation when it is the same record, has the same
target sentence, or shares a run of :data:`WINDOW` Han characters with an
evaluation record, each read as its context followed by its target: the
near-duplicate rule the ingestion's leakage guard (``ime_corpus::leak``)
applies to whole documents.

The recipe, from the repository root, with ``<against>`` every evaluation and
held-out file the ingestion was guarded with, each after ``--against``::

    mlime eval idiom-draw --samples data/corpus-225/samples \\
        --source fineweb2-hq --source mnbvc-forum --source yyets \\
        --entries data/corpus-225/inputs/entries-215.tsv \\
        --knowledge data/lexicon/knowledge.tsv \\
        --emittable data/route-a-assets-v2/emittable.txt \\
        --ordinary-cache data/corpus-225/.knowledge-cache \\
        --eval3 data/run3_pool/eval3.jsonl <against> --size <n> --seed 243 \\
        --out data/idiom-select-243/pool
    mlime corpus judge --input candidates --data-dir data/idiom-select-243/pool \\
        --entries data/corpus-225/inputs/entries-215.tsv \\
        --ordinary-cache data/corpus-225/.knowledge-cache
    mlime eval idiom-topup <the draw's inputs, without --size and --seed> \\
        --pool data/idiom-select-243/pool
    mlime corpus judge --input candidates --data-dir data/idiom-select-243/pool \\
        --entries data/corpus-225/inputs/entries-215.tsv \\
        --ordinary-cache data/corpus-225/.knowledge-cache
    mlime eval idiom-set --judged data/idiom-select-243/pool \\
        --entries data/corpus-225/inputs/entries-215.tsv <against> --out <root>
    ime-cli g2p annotate --data-dir <root>
    ime-cli export eval-set --data-dir <root> --out <root>/idioms-abbreviated.jsonl \\
        --size <agreed> --seed 243 --typing abbreviated
"""

from __future__ import annotations

import json
import math
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from fractions import Fraction
from pathlib import Path
from typing import Any

import polars as pl

from mlime.logging import log

from . import judge
from .text import HAN

#: The category of #215's entry table the set is drawn from.
CATEGORY = "idiom"

#: Han characters two records must share in a row to count as the same text,
#: ``ime_corpus::leak::WINDOW``.
WINDOW = 13

#: The columns of a samples shard, as ``ime-cli g2p annotate`` reads them.
SAMPLE_COLUMNS = ("id", "source", "text", "context")

#: How many strata eval3's idiom frequencies are cut into.
STRATA = 10

#: The quantiles the census compares between the drawn idioms and eval3's.
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)

#: The draw's census, beside its candidate files.
DRAW_CENSUS = "draw.json"

#: The top-up's census, beside the candidate files of both rounds.
TOP_UP_CENSUS = "topup.json"


class SelectionError(RuntimeError):
    """The inputs contradict each other; the set is not built from them."""


# ---------------------------------------------------------------------------
# Collisions with evaluation.


@dataclass(frozen=True)
class Record:
    """One evaluation record's sentence and what preceded it."""

    text: str
    context: str | None


def read_records(path: Path) -> list[Record]:
    """The records of a JSON Lines evaluation file (or a pool, which has no context)."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records.append(Record(text=row["text"], context=row.get("context")))
    if not records:
        raise SelectionError(f"{path} holds no record")
    return records


def windows(record: Record, n: int = WINDOW) -> set[str]:
    """Every run of *n* Han characters in *record*'s context followed by its text."""
    han = "".join(HAN.findall((record.context or "") + record.text))
    return {han[start : start + n] for start in range(len(han) - n + 1)}


@dataclass
class Collisions:
    """What the set must not share with the evaluation records."""

    records: set[Record] = field(default_factory=set)
    texts: set[str] = field(default_factory=set)
    runs: set[str] = field(default_factory=set)

    @classmethod
    def of(cls, records: Iterable[Record]) -> Collisions:
        """The collisions of *records*."""
        index = cls()
        for record in records:
            index.records.add(record)
            index.texts.add(record.text)
            index.runs |= windows(record)
        return index

    def reason(self, record: Record) -> str | None:
        """Why *record* collides, or ``None`` when it does not."""
        if record in self.records:
            return "record"
        if record.text in self.texts:
            return "text"
        if not windows(record).isdisjoint(self.runs):
            return "window"
        return None


def collisions_of(against: Iterable[Path]) -> Collisions:
    """The collisions of every record of the files in *against*."""
    return Collisions.of(record for path in against for record in read_records(path))


# ---------------------------------------------------------------------------
# The strata.


def quantile(values: Sequence[int], share: float) -> int:
    """The value at rank ``floor(share * n)`` of sorted *values*, counting from zero."""
    return values[min(len(values) - 1, math.floor(share * len(values)))]


def ks_distance(left: Sequence[int], right: Sequence[int]) -> float:
    """The largest gap between the empirical distribution functions of two samples."""
    left, right = sorted(left), sorted(right)
    return max(
        abs(bisect_right(left, value) / len(left) - bisect_right(right, value) / len(right))
        for value in set(left) | set(right)
    )


@dataclass(frozen=True)
class Strata:
    """Half-open ranges of run3 occurrences, and how many of eval3's idioms each holds.

    Stratum ``i`` holds the counts from ``lowers[i]`` up to ``lowers[i + 1]``;
    the last has no upper bound.
    """

    lowers: tuple[int, ...]
    targets: tuple[int, ...]

    @classmethod
    def of(cls, occurrences: Sequence[int], count: int) -> Strata:
        """*count* strata cut at the quantiles of eval3's idioms' *occurrences*."""
        values = sorted(occurrences)
        if not values or values[0] < 1:
            raise SelectionError("every eval3 idiom must occur in run3 to be stratified")
        cuts = sorted({quantile(values, index / count) for index in range(1, count)} - {1})
        lowers = (1, *cuts)
        targets = Counter(bisect_right(lowers, value) - 1 for value in values)
        return cls(lowers=lowers, targets=tuple(targets[index] for index in range(len(lowers))))

    def __len__(self) -> int:
        return len(self.lowers)

    def index(self, occurrences: int) -> int:
        """The stratum of an idiom with *occurrences* in run3."""
        if occurrences < 1:
            raise SelectionError(f"an idiom with {occurrences} run3 occurrences has no stratum")
        return bisect_right(self.lowers, occurrences) - 1

    def upper(self, index: int) -> int | None:
        """The first count past stratum *index*, or ``None`` for the last."""
        return self.lowers[index + 1] if index + 1 < len(self.lowers) else None

    def allocate(self, size: int) -> list[int]:
        """*size* shared in eval3's proportions, the remainder by largest fraction."""
        total = sum(self.targets)
        shares = [target * size for target in self.targets]
        quotas = [share // total for share in shares]
        order = sorted(range(len(self)), key=lambda index: (-(shares[index] % total), index))
        for index in order[: size - sum(quotas)]:
            quotas[index] += 1
        return quotas

    def largest_size(self, available: Sequence[int]) -> int:
        """The largest size whose allocation no stratum's *available* idioms fall short of."""
        for size in range(sum(available), -1, -1):
            if all(
                quota <= have for quota, have in zip(self.allocate(size), available, strict=True)
            ):
                return size
        raise AssertionError("unreachable: a size of zero always fits")


# ---------------------------------------------------------------------------
# The draw.


def read_pool(samples: Path, sources: Sequence[str]) -> pl.DataFrame:
    """Every record of the named sources' samples shards (``<source>-*.parquet``)."""
    frames = []
    for source in sources:
        paths = sorted(samples.glob(f"{source}-*.parquet"))
        if not paths:
            raise SelectionError(f"no samples shard of {source} under {samples}")
        frame = pl.concat([pl.read_parquet(path, columns=list(SAMPLE_COLUMNS)) for path in paths])
        if frame.get_column("source").unique().to_list() != [source]:
            raise SelectionError(f"{source}'s shards hold records of another source")
        frames.append(frame)
    pool = pl.concat(frames)
    if pool.get_column("id").is_duplicated().any():
        raise SelectionError("the pool repeats an id")
    return pool


def held_entries(texts: pl.Series, entries: Sequence[str]) -> pl.Series:
    """For each text, every one of *entries* it holds, overlaps included."""
    return texts.str.extract_many(list(entries), overlapping=True)


@dataclass(frozen=True)
class Inputs:
    """What the draw reads besides the pool."""

    #: #215's table: ``entry``, ``category``, ``occurrences``, ``blocked``.
    table: pl.DataFrame
    #: The knowledge lexicon's idioms, the ones #212's word arcs carry.
    knowledge_idioms: frozenset[str]
    #: The characters the decoder can emit.
    emittable: frozenset[str]
    #: eval3's gold texts.
    eval_texts: Sequence[str]
    #: The evaluation and held-out records nothing may share text with.
    collisions: Collisions
    #: The judge's first rule.
    lexicon: judge.BoundaryLexicon


@dataclass
class DrawCensus:
    """How the draw came out."""

    seed: int = 0
    size: int = 0
    largest_size: int = 0
    eval3_idioms: int = 0
    pool_records: int = 0
    excluded_idioms: Counter[str] = field(default_factory=Counter)
    eligible_idioms: int = 0
    pool_records_holding_one: int = 0
    pool_idioms: int = 0
    rejected_sentences: Counter[str] = field(default_factory=Counter)
    idioms_without_sentence: int = 0
    strata: list[dict[str, Any]] = field(default_factory=list)
    sources: Counter[str] = field(default_factory=Counter)
    entry_lengths: Counter[int] = field(default_factory=Counter)
    quantiles: dict[str, dict[str, int]] = field(default_factory=dict)
    ks_distance: float = 0.0
    checks: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """The census as JSON-ready data, counters keyed by their value."""
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return {
            name: dict(sorted(value.items())) if isinstance(value, Counter) else value
            for name, value in values.items()
        }


def eval_idioms(inputs: Inputs) -> dict[str, int]:
    """eval3's idioms with their run3 occurrences."""
    idioms = inputs.table.filter(pl.col("category") == CATEGORY)
    occurrences = dict(zip(idioms["entry"], idioms["occurrences"], strict=True))
    held = held_entries(pl.Series(list(inputs.eval_texts), dtype=pl.String), list(occurrences))
    found = set(held.explode(empty_as_null=True).drop_nulls().to_list())
    if not found:
        raise SelectionError("eval3 holds no idiom")
    return {entry: int(occurrences[entry]) for entry in sorted(found)}


def eligible_idioms(inputs: Inputs, held_by_eval: Mapping[str, int]) -> tuple[dict[str, int], Counter[str]]:
    """The idioms the draw may take, with their run3 occurrences, and why the rest may not."""
    excluded: Counter[str] = Counter()
    eligible: dict[str, int] = {}
    rows = inputs.table.filter(pl.col("category") == CATEGORY)
    for entry, occurrences, blocked in rows.select("entry", "occurrences", "blocked").iter_rows():
        if occurrences < 1:
            excluded["absent from run3"] += 1
        elif blocked:
            excluded["blocked"] += 1
        elif entry not in inputs.knowledge_idioms:
            excluded["not a knowledge-lexicon idiom"] += 1
        elif not set(entry) <= inputs.emittable:
            excluded["a character not emittable"] += 1
        elif entry in held_by_eval:
            excluded["held by eval3"] += 1
        else:
            eligible[entry] = int(occurrences)
    return eligible, excluded


@dataclass
class Candidate:
    """One idiom's sentences in their seeded order, and how far the draw has looked."""

    entry: str
    stratum: int
    rows: list[tuple[str, str, str, str | None]]
    verdicts: dict[int, str | None] = field(default_factory=dict)


#: A pool record as the draw handles it: ``id``, ``source``, ``text``, ``context``.
Row = tuple[str, str, str, str | None]


@dataclass
class Taken:
    """The records drawn so far, by id and by sentence, each with the idiom it was drawn for."""

    ids: dict[str, str] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)

    def owner(self, row: Row) -> str | None:
        """The idiom *row*'s record or sentence was drawn for, or ``None`` when it is free."""
        return self.ids.get(row[0]) or self.texts.get(row[2])

    def add(self, row: Row, entry: str) -> None:
        """Mark *row* drawn for *entry*."""
        self.ids[row[0]] = entry
        self.texts[row[2]] = entry


@dataclass
class Prepared:
    """The pool read against the inputs: the strata, and the idioms each may draw from."""

    inputs: Inputs
    seed: int
    strata: Strata
    #: eval3's idioms with their run3 occurrences.
    in_eval: dict[str, int]
    #: The idioms a draw may take, with their run3 occurrences.
    eligible: dict[str, int]
    #: Per stratum, the eligible idioms with at least one admissible sentence.
    available: list[list[Candidate]]
    #: What the reading found, the census a draw goes on to fill.
    census: DrawCensus

    def verdict(self, candidate: Candidate, position: int) -> str | None:
        """Why *candidate*'s sentence at *position* may not be drawn, or ``None`` when it may."""
        if position not in candidate.verdicts:
            _, _, text, context = candidate.rows[position]
            if self.inputs.lexicon.false_match(text, candidate.entry) is not None:
                reason: str | None = "word boundary"
            else:
                collides = self.inputs.collisions.reason(Record(text=text, context=context))
                reason = None if collides is None else f"collides by {collides}"
            candidate.verdicts[position] = reason
            if reason is not None:
                self.census.rejected_sentences[reason] += 1
        return candidate.verdicts[position]

    def order(self, index: int) -> list[Candidate]:
        """Stratum *index*'s available idioms in the seeded order every round draws them in."""
        order = sorted(self.available[index], key=lambda candidate: candidate.entry)
        random.Random(f"{self.seed}:idioms:{index}").shuffle(order)
        return order

    def free(self, candidate: Candidate, taken: Taken) -> Row | None:
        """*candidate*'s first admissible sentence, in its seeded order, that is not drawn yet."""
        for position, row in enumerate(candidate.rows):
            if self.verdict(candidate, position) is None and taken.owner(row) is None:
                return row
        return None

    def take(self, candidate: Candidate, taken: Taken, skipped: Counter[str]) -> Row | None:
        """Draw :meth:`free`'s sentence for *candidate*, counting in *skipped* the drawn ones passed."""
        for position, row in enumerate(candidate.rows):
            if self.verdict(candidate, position) is not None:
                continue
            owner = taken.owner(row)
            if owner is None:
                taken.add(row, candidate.entry)
                return row
            skipped[
                "drawn for this idiom before" if owner == candidate.entry else "drawn for another idiom"
            ] += 1
        return None

    def draw(self, size: int) -> tuple[pl.DataFrame, DrawCensus]:
        """One sentence for each of *size* idioms, in eval3's proportions; and the census."""
        census = self.census
        census.size = size
        available = [len(group) for group in self.available]
        if size > census.largest_size:
            raise SelectionError(
                f"the pool holds too few idioms for {size} at eval3's frequencies: per stratum "
                f"{available} against eval3's {list(self.strata.targets)}; "
                f"the largest size that fits is {census.largest_size}"
            )
        quotas = self.strata.allocate(size)
        taken = Taken()
        chosen: list[tuple[str, str, str, str | None, str]] = []
        census.strata = []
        # The scarcest strata draw first, so a sentence two idioms share goes to the scarce one.
        for index in sorted(range(len(self.strata)), key=lambda i: available[i] - quotas[i]):
            drawn = 0
            for candidate in self.order(index):
                if drawn == quotas[index]:
                    break
                row = self.take(candidate, taken, census.rejected_sentences)
                if row is not None:
                    chosen.append((*row, candidate.entry))
                    drawn += 1
            if drawn < quotas[index]:
                raise SelectionError(
                    f"stratum {index} ran out of idioms at {drawn} of {quotas[index]}: the "
                    "sentences its idioms share are drawn for other idioms"
                )
            census.strata.append(
                {
                    "stratum": index,
                    "lower": self.strata.lowers[index],
                    "upper": self.strata.upper(index),
                    "eval3": self.strata.targets[index],
                    "available": available[index],
                    "drawn": drawn,
                }
            )
        census.strata.sort(key=lambda stratum: stratum["stratum"])
        frame = drawn_frame(chosen)
        census.checks = record_checks(frame, self.inputs, self.eligible, self.in_eval)
        drawn_occurrences = sorted(self.eligible[entry] for entry in frame["entry"])
        census.quantiles = compare_quantiles(drawn_occurrences, self.in_eval.values())
        census.ks_distance = round(
            ks_distance(drawn_occurrences, sorted(self.in_eval.values())), 4
        )
        census.sources = Counter(frame["source"].to_list())
        census.entry_lengths = Counter(len(entry) for entry in frame["entry"])
        return frame, census


def prepare(pool: pl.DataFrame, inputs: Inputs, strata_count: int, seed: int) -> Prepared:
    """Read *pool* against *inputs*: eval3's strata, and every eligible idiom's sentences."""
    census = DrawCensus(seed=seed, pool_records=pool.height)
    in_eval = eval_idioms(inputs)
    census.eval3_idioms = len(in_eval)
    strata = Strata.of(list(in_eval.values()), strata_count)
    eligible, census.excluded_idioms = eligible_idioms(inputs, in_eval)
    census.eligible_idioms = len(eligible)

    holding = pool.filter(pl.col("text").str.contains_any(list(eligible)))
    census.pool_records_holding_one = holding.height
    hits = (
        holding.with_columns(held_entries(holding["text"], list(eligible)).alias("entry"))
        .explode("entry", empty_as_null=True)
        .unique(["id", "entry"])
        .sort("entry", "id")
    )
    candidates: list[Candidate] = []
    for (entry,), group in hits.group_by("entry", maintain_order=True):
        rows = list(group.select(SAMPLE_COLUMNS).iter_rows())
        random.Random(f"{seed}:sentences:{entry}").shuffle(rows)
        candidates.append(Candidate(str(entry), strata.index(eligible[str(entry)]), rows))
    census.pool_idioms = len(candidates)

    prepared = Prepared(
        inputs=inputs,
        seed=seed,
        strata=strata,
        in_eval=in_eval,
        eligible=eligible,
        available=[[] for _ in range(len(strata))],
        census=census,
    )
    for candidate in candidates:
        if any(
            prepared.verdict(candidate, position) is None
            for position in range(len(candidate.rows))
        ):
            prepared.available[candidate.stratum].append(candidate)
        else:
            census.idioms_without_sentence += 1
    census.largest_size = strata.largest_size([len(group) for group in prepared.available])
    return prepared


def draw(
    pool: pl.DataFrame, inputs: Inputs, strata_count: int, size: int, seed: int
) -> tuple[pl.DataFrame, DrawCensus]:
    """One sentence for each of *size* idioms, stratified on eval3's frequencies; and the census.

    The result has a row per drawn idiom: the record's ``id``, ``source``,
    ``text`` and ``context``, and the ``entry``.
    """
    return prepare(pool, inputs, strata_count, seed).draw(size)


def drawn_frame(chosen: Sequence[tuple[str, str, str, str | None, str]]) -> pl.DataFrame:
    """Drawn rows as a frame of the record's columns and the ``entry``, by entry."""
    schema = {**dict.fromkeys(SAMPLE_COLUMNS, pl.String), "entry": pl.String}
    return pl.DataFrame(chosen, schema=schema, orient="row").sort("entry")


def compare_quantiles(drawn: Sequence[int], in_eval: Iterable[int]) -> dict[str, dict[str, int]]:
    """The :data:`QUANTILES` of the drawn idioms' run3 counts beside eval3's."""
    eval_occurrences = sorted(in_eval)
    return {
        f"p{round(share * 100)}": {
            "eval3": quantile(eval_occurrences, share),
            "drawn": quantile(sorted(drawn), share),
        }
        for share in QUANTILES
    }


def record_checks(
    frame: pl.DataFrame,
    inputs: Inputs,
    eligible: Mapping[str, int],
    in_eval: Mapping[str, int],
) -> dict[str, int]:
    """Check the drawn records from scratch; what each check found."""
    entries = frame["entry"].to_list()
    texts = " ".join(inputs.eval_texts)
    checks = {
        "records": frame.height,
        "idioms": len(set(entries)),
        "distinct ids": frame["id"].n_unique(),
        "distinct texts": frame["text"].n_unique(),
        "holding their idiom": sum(
            entry in text for entry, text in zip(entries, frame["text"], strict=True)
        ),
        "knowledge-lexicon idioms": sum(entry in inputs.knowledge_idioms for entry in entries),
        "every character emittable": sum(set(entry) <= inputs.emittable for entry in entries),
        "held by eval3": sum(entry in in_eval or entry in texts for entry in entries),
        "eligible": sum(entry in eligible for entry in entries),
        "colliding with evaluation": sum(
            inputs.collisions.reason(Record(text=text, context=context)) is not None
            for text, context in zip(frame["text"], frame["context"], strict=True)
        ),
        "crossed at the word boundary": sum(
            inputs.lexicon.false_match(text, entry) is not None
            for text, entry in zip(frame["text"], entries, strict=True)
        ),
    }
    expected = frame.height
    for name in (
        "idioms",
        "distinct ids",
        "distinct texts",
        "holding their idiom",
        "knowledge-lexicon idioms",
        "every character emittable",
        "eligible",
    ):
        if checks[name] != expected:
            raise SelectionError(f"{name}: {checks[name]} of {expected} drawn records")
    for name in ("held by eval3", "colliding with evaluation", "crossed at the word boundary"):
        if checks[name]:
            raise SelectionError(f"{name}: {checks[name]} drawn records")
    return checks


def write_draw(out: Path, frame: pl.DataFrame, census: DrawCensus) -> judge.CandidateSet:
    """Write the draw as a candidate set under *out*, its census beside it."""
    if (out / DRAW_CENSUS).exists() or judge.CandidateSet(out).ordinals():
        raise SelectionError(f"{out} already holds a draw; a draw is written into a fresh root")
    candidates = judge.write_candidates(out, frame)
    write_json(out / DRAW_CENSUS, census.as_dict())
    log.info(
        "idiom draw written",
        records=frame.height,
        largest_size=census.largest_size,
        ks_distance=census.ks_distance,
        out=str(out),
    )
    return candidates


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write *data* to *path* as indented JSON."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The top-up: one more round for the strata the judge left short.


@dataclass(frozen=True)
class JudgedRound:
    """A judged round: every unit it drew, and the ones the judge kept."""

    #: ``id``, ``source``, ``text``, ``context`` and ``entry`` of every drawn unit.
    units: pl.DataFrame
    #: The ``(id, entry)`` of every kept unit.
    kept: frozenset[tuple[str, str]]


def judged_round(layout: judge.CandidateSet) -> JudgedRound:
    """Every unit of *layout*'s files with the judge's verdict; every file must be judged."""
    ordinals = layout.ordinals()
    if not ordinals:
        raise SelectionError(f"{layout.candidates} holds no candidate file")
    unfinished = [ordinal for ordinal in ordinals if not judge.finished_at_threshold(layout, ordinal)]
    if unfinished:
        raise SelectionError(f"the judge has not finished files {unfinished}")
    units = []
    kept: set[tuple[str, str]] = set()
    for ordinal in ordinals:
        rows = layout.read_entry_rows(ordinal)
        records = layout.read_records(ordinal).select(SAMPLE_COLUMNS)
        units.append(rows.join(records, on="id", how="left").select(*SAMPLE_COLUMNS, "entry"))
        kept |= set(judge.read_entry_rows(layout.kept_rows(ordinal)).iter_rows())
    frame = pl.concat(units)
    if frame["text"].null_count():
        raise SelectionError("an entry row names a record its file does not hold")
    if not kept <= set(frame.select("id", "entry").iter_rows()):
        raise SelectionError("the judge kept a unit its input does not list")
    return JudgedRound(units=frame, kept=frozenset(kept))


@dataclass(frozen=True)
class RoundOne:
    """The first round under a root: the seed and size it was drawn at, and the judge's verdicts."""

    seed: int
    size: int
    judged: JudgedRound


def round_one(root: Path) -> RoundOne:
    """The judged first round under *root*, which must hold nothing past it."""
    if (root / TOP_UP_CENSUS).exists():
        raise SelectionError(f"{root} already holds a top-up; there is one top-up round")
    drawn = json.loads((root / DRAW_CENSUS).read_text(encoding="utf-8"))
    layout = judge.CandidateSet(root)
    listed = sum(layout.read_entry_rows(ordinal).height for ordinal in layout.ordinals())
    if listed != drawn["size"]:
        raise SelectionError(
            f"{layout.candidates} lists {listed} units, but the draw drew {drawn['size']}; "
            "were files added past the first round?"
        )
    return RoundOne(seed=int(drawn["seed"]), size=int(drawn["size"]), judged=judged_round(layout))


@dataclass(frozen=True)
class Plan:
    """How many each stratum draws in the top-up, from what the first round kept.

    A stratum is expected to keep its round-one share of what it draws: the
    round-one keep rate is the only yield the judge has shown on these idioms.
    The target is the largest size whose allocation in eval3's proportions no
    stratum falls short of when it draws every idiom it has left and keeps at
    that rate; a stratum's need is its quota at that size less what it kept,
    and it draws its need over its keep rate, rounded up, which the target keeps
    within the idioms it has left.
    """

    #: Per stratum: units the first round drew, and kept.
    drawn: tuple[int, ...]
    kept: tuple[int, ...]
    #: Per stratum: idioms it can still draw a sentence for.
    supply: tuple[int, ...]
    #: The size eval3's proportions are matched at, and each stratum's quota at it.
    size: int
    quotas: tuple[int, ...]
    #: Per stratum: kept units short of its quota, and how many it draws for them.
    needs: tuple[int, ...]
    draws: tuple[int, ...]
    #: Per stratum: kept units expected once the top-up is judged.
    expected: tuple[Fraction, ...]


def plan(strata: Strata, drawn: Sequence[int], kept: Sequence[int], supply: Sequence[int]) -> Plan:
    """The top-up's :class:`Plan` for what each stratum drew and kept, and has left to draw."""
    if not len(drawn) == len(kept) == len(supply) == len(strata):
        raise SelectionError("the top-up needs a count for every stratum")
    for index, (taken, held) in enumerate(zip(drawn, kept, strict=True)):
        if taken < 1:
            raise SelectionError(f"round one drew nothing in stratum {index}; its yield is unknown")
        if not 0 <= held <= taken:
            raise SelectionError(f"stratum {index} kept {held} of {taken} drawn")
    rates = [Fraction(held, taken) for held, taken in zip(kept, drawn, strict=True)]
    reach = [
        math.floor(held + rate * left) for held, rate, left in zip(kept, rates, supply, strict=True)
    ]
    size = strata.largest_size(reach)
    quotas = strata.allocate(size)
    needs = [max(0, quota - held) for quota, held in zip(quotas, kept, strict=True)]
    # quota <= floor(kept + rate * supply), so a positive need has a positive rate, and
    # need / rate <= supply: no stratum is asked for more idioms than it has left.
    draws = [math.ceil(need / rate) if need else 0 for need, rate in zip(needs, rates, strict=True)]
    return Plan(
        drawn=tuple(drawn),
        kept=tuple(kept),
        supply=tuple(supply),
        size=size,
        quotas=tuple(quotas),
        needs=tuple(needs),
        draws=tuple(draws),
        expected=tuple(
            held + rate * count for held, rate, count in zip(kept, rates, draws, strict=True)
        ),
    )


@dataclass
class TopUpCensus:
    """How the top-up came out."""

    seed: int = 0
    round_one_size: int = 0
    round_one_files: list[int] = field(default_factory=list)
    target_size: int = 0
    strata: list[dict[str, Any]] = field(default_factory=list)
    drawn: int = 0
    expected_kept: float = 0.0
    skipped_sentences: Counter[str] = field(default_factory=Counter)
    sources: Counter[str] = field(default_factory=Counter)
    checks: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """The census as JSON-ready data, counters keyed by their value."""
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return {
            name: dict(sorted(value.items())) if isinstance(value, Counter) else value
            for name, value in values.items()
        }


def top_up(
    prepared: Prepared, size: int, judged: JudgedRound
) -> tuple[pl.DataFrame, TopUpCensus]:
    """The second round's draw, for the strata whose kept units fall short of eval3's proportions.

    *prepared* must read the pool the first round was drawn from, and *size* be
    the first round's: the first round is drawn again from them and must be what
    the judge read. Each stratum then continues its seeded idiom order past the
    idioms the first round drew, one sentence each; once those run out, it goes
    through the idioms the judge rejected, in the same order, each giving the
    next admissible sentence in its own seeded order. No record or sentence of
    the first round is drawn again.
    """
    round_one, _ = prepared.draw(size)
    expected_units = set(round_one.select("id", "entry").iter_rows())
    if set(judged.units.select("id", "entry").iter_rows()) != expected_units:
        raise SelectionError(
            "the judged round is not the draw these inputs give at this seed and size; "
            "the top-up reads the pool and the inputs the first round was drawn from"
        )
    strata = prepared.strata
    census = TopUpCensus(seed=prepared.seed, round_one_size=size)
    taken = Taken()
    for identifier, source, text, context, entry in judged.units.iter_rows():
        taken.add((identifier, source, text, context), entry)
    kept_entries = {entry for _, entry in judged.kept}
    drawn_entries = set(judged.units["entry"].to_list())
    rejected_entries = drawn_entries - kept_entries

    unused: list[list[Candidate]] = []
    second: list[list[Candidate]] = []
    for index in range(len(strata)):
        order = prepared.order(index)
        unused.append([c for c in order if c.entry not in drawn_entries])
        second.append([c for c in order if c.entry in rejected_entries])
    supply = [
        sum(prepared.free(c, taken) is not None for c in unused[index] + second[index])
        for index in range(len(strata))
    ]
    drawn_per = Counter(strata.index(prepared.eligible[entry]) for entry in drawn_entries)
    kept_per = Counter(strata.index(prepared.eligible[entry]) for entry in kept_entries)
    the_plan = plan(
        strata,
        [drawn_per[index] for index in range(len(strata))],
        [kept_per[index] for index in range(len(strata))],
        supply,
    )

    chosen: list[tuple[str, str, str, str | None, str]] = []
    by_stratum: dict[int, dict[str, int]] = {}
    # The scarcest strata draw first, so a sentence two idioms share goes to the scarce one.
    for index in sorted(range(len(strata)), key=lambda i: supply[i] - the_plan.draws[i]):
        counts = {"unused": 0, "second": 0}
        for kind, group in (("unused", unused[index]), ("second", second[index])):
            for candidate in group:
                if counts["unused"] + counts["second"] == the_plan.draws[index]:
                    break
                row = prepared.take(candidate, taken, census.skipped_sentences)
                if row is not None:
                    chosen.append((*row, candidate.entry))
                    counts[kind] += 1
        if counts["unused"] + counts["second"] < the_plan.draws[index]:
            raise SelectionError(
                f"stratum {index} ran out of idioms at {counts['unused'] + counts['second']} of "
                f"{the_plan.draws[index]}: the sentences its idioms share are drawn for others"
            )
        by_stratum[index] = counts
    if not chosen:
        raise SelectionError(
            f"no stratum falls short of eval3's proportions at {the_plan.size}; nothing to top up"
        )

    frame = drawn_frame(chosen)
    checks = record_checks(frame, prepared.inputs, prepared.eligible, prepared.in_eval)
    round_one_ids = set(judged.units["id"].to_list())
    round_one_texts = set(judged.units["text"].to_list())
    checks |= {
        "kept in round one": sum(entry in kept_entries for entry in frame["entry"]),
        "a record of round one": sum(identifier in round_one_ids for identifier in frame["id"]),
        "a sentence of round one": sum(text in round_one_texts for text in frame["text"]),
    }
    for name in ("kept in round one", "a record of round one", "a sentence of round one"):
        if checks[name]:
            raise SelectionError(f"{name}: {checks[name]} top-up records")
    census.checks = checks
    census.target_size = the_plan.size
    census.strata = [
        {
            "stratum": index,
            "lower": strata.lowers[index],
            "upper": strata.upper(index),
            "eval3": strata.targets[index],
            "round_one_drawn": the_plan.drawn[index],
            "round_one_kept": the_plan.kept[index],
            "unused_idioms": len(unused[index]),
            "rejected_idioms": len(second[index]),
            "supply": the_plan.supply[index],
            "quota": the_plan.quotas[index],
            "need": the_plan.needs[index],
            "draw": the_plan.draws[index],
            "drawn_unused": by_stratum[index]["unused"],
            "drawn_second": by_stratum[index]["second"],
            "expected_kept": round(float(the_plan.expected[index]), 2),
        }
        for index in range(len(strata))
    ]
    census.drawn = frame.height
    census.expected_kept = round(float(sum(the_plan.expected)), 2)
    census.sources = Counter(frame["source"].to_list())
    return frame, census


def write_top_up(
    root: Path, frame: pl.DataFrame, census: TopUpCensus
) -> judge.CandidateSet:
    """Add the top-up to *root*'s candidate set as files after round one's, its census beside."""
    if (root / TOP_UP_CENSUS).exists():
        raise SelectionError(f"{root} already holds a top-up; there is one top-up round")
    census.round_one_files = judge.CandidateSet(root).ordinals()
    candidates = judge.write_candidates(root, frame)
    write_json(root / TOP_UP_CENSUS, census.as_dict())
    log.info(
        "idiom top-up written",
        records=frame.height,
        target_size=census.target_size,
        expected_kept=census.expected_kept,
        files=[ordinal for ordinal in candidates.ordinals() if ordinal not in census.round_one_files],
        root=str(root),
    )
    return candidates


def read_table(path: Path) -> pl.DataFrame:
    """#215's entry table, the columns the draw reads."""
    return pl.read_csv(path, separator="\t", quote_char=None).select(
        "entry", "category", "occurrences", "blocked"
    )


def read_knowledge_idioms(path: Path) -> frozenset[str]:
    """The idioms of ``knowledge.tsv`` (``entry<TAB>category``, no header)."""
    lexicon = pl.read_csv(
        path, separator="\t", has_header=False, new_columns=["entry", "category"], quote_char=None
    )
    return frozenset(lexicon.filter(pl.col("category") == CATEGORY)["entry"].to_list())


# ---------------------------------------------------------------------------
# The set, from what the judge kept.


def kept_units(layout: judge.Candidates) -> pl.DataFrame:
    """Every unit the judge kept, with its record; every file must be judged.

    A file counts as judged only when the judge finished it under the Jev
    threshold in force now, so the units are those the judge's own rules keep.
    """
    ordinals = layout.ordinals()
    if not ordinals:
        raise SelectionError("the judge's input holds no file")
    unfinished = [ordinal for ordinal in ordinals if not judge.finished_at_threshold(layout, ordinal)]
    if unfinished:
        raise SelectionError(f"the judge has not finished files {unfinished}")
    frames = []
    for ordinal in ordinals:
        rows = judge.read_entry_rows(layout.kept_rows(ordinal))
        kept = [layout.samples / shard.name for shard in layout.shards(ordinal)]
        records = (
            pl.concat([pl.read_parquet(path) for path in kept if path.exists()])
            if any(path.exists() for path in kept)
            else pl.DataFrame(schema=judge.RECORD_SCHEMA)
        )
        if set(rows["id"]) != set(records["id"]):
            raise SelectionError(f"file {ordinal:04d}'s kept rows and kept records disagree")
        frames.append(rows.join(records.select(SAMPLE_COLUMNS), on="id", how="left"))
    return pl.concat(frames)


@dataclass
class Census:
    """How the set came out of the judged records."""

    kept_units: int = 0
    kept_idiom_units: int = 0
    idioms: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    records: int = 0
    sources: Counter[str] = field(default_factory=Counter)
    entry_lengths: Counter[int] = field(default_factory=Counter)
    records_per_idiom_kept: Counter[int] = field(default_factory=Counter)
    run3_occurrences: Counter[int] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        """The census as JSON-ready data, counters keyed by their value."""
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return {
            name: dict(sorted(value.items())) if isinstance(value, Counter) else value
            for name, value in values.items()
        }


def select(
    units: pl.DataFrame, entries: pl.DataFrame, collisions: Collisions
) -> tuple[pl.DataFrame, Census]:
    """One record per kept idiom, none colliding with evaluation; and the census.

    *units* are the kept units (``id``, ``entry``, ``source``, ``text``,
    ``context``); *entries* is #215's table (``entry``, ``category``,
    ``occurrences``). Each idiom contributes its record with the smallest id.
    """
    census = Census(kept_units=units.height)
    category = dict(zip(entries["entry"], entries["category"], strict=True))
    occurrences = dict(zip(entries["entry"], entries["occurrences"], strict=True))

    candidates: dict[str, list[dict[str, Any]]] = {}
    for unit in units.sort("id", "entry").iter_rows(named=True):
        entry = unit["entry"]
        if entry not in category:
            raise SelectionError(f"the kept entry {entry!r} is not in the entry table")
        if entry not in unit["text"]:
            raise SelectionError(f"the kept record {unit['id']} does not hold {entry!r}")
        if category[entry] == CATEGORY:
            census.kept_idiom_units += 1
            candidates.setdefault(entry, []).append(unit)
    census.idioms = len(candidates)
    census.records_per_idiom_kept = Counter(len(rows) for rows in candidates.values())

    chosen: list[dict[str, Any]] = []
    taken_ids: set[str] = set()
    taken_texts: set[str] = set()
    for entry, rows in sorted(candidates.items()):
        pick = None
        for row in rows:
            if row["id"] in taken_ids or row["text"] in taken_texts:
                census.dropped["already taken"] += 1
                continue
            reason = collisions.reason(Record(text=row["text"], context=row["context"]))
            if reason is not None:
                census.dropped[f"collides by {reason}"] += 1
                continue
            pick = row
            break
        if pick is None:
            census.dropped["idiom left without a record"] += 1
            continue
        taken_ids.add(pick["id"])
        taken_texts.add(pick["text"])
        census.entry_lengths[len(entry)] += 1
        census.run3_occurrences[int(occurrences[entry])] += 1
        chosen.append({column: pick[column] for column in SAMPLE_COLUMNS} | {"entry": entry})
    census.records = len(chosen)
    if not chosen:
        raise SelectionError("no kept idiom record survived the collision checks")
    frame = pl.DataFrame(
        chosen, schema={**{column: pl.String for column in SAMPLE_COLUMNS}, "entry": pl.String}
    )
    census.sources = Counter(frame["source"].to_list())
    return frame, census


def build(
    layout: judge.Candidates,
    entry_table: Path,
    against: Iterable[Path],
    out: Path,
) -> Census:
    """Write the set's samples shards, its idiom list and its census under *out*."""
    units = kept_units(layout)
    entries = pl.read_csv(entry_table, separator="\t", quote_char=None).select(
        "entry", "category", "occurrences"
    )
    frame, census = select(units, entries, collisions_of(against))

    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    if any(samples.iterdir()):
        raise SelectionError(f"{samples} is not empty; a set is built once, into a fresh root")
    for (source,), shard in frame.group_by("source", maintain_order=True):
        shard.select(SAMPLE_COLUMNS).write_parquet(samples / f"{source}-00000.parquet")
    frame.select("id", "entry").write_csv(out / "idioms.tsv", separator="\t")
    (out / "census.json").write_text(
        json.dumps(census.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log.info(
        "idiom selection written",
        records=census.records,
        idioms=census.idioms,
        dropped=dict(census.dropped),
        out=str(out),
    )
    return census
