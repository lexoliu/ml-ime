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

**The set** (:func:`build`). Once the judge has finished every file, each idiom
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


def draw(
    pool: pl.DataFrame, inputs: Inputs, strata_count: int, size: int, seed: int
) -> tuple[pl.DataFrame, DrawCensus]:
    """One sentence for each of *size* idioms, stratified on eval3's frequencies; and the census.

    The result has a row per drawn idiom: the record's ``id``, ``source``,
    ``text`` and ``context``, and the ``entry``.
    """
    census = DrawCensus(seed=seed, size=size, pool_records=pool.height)
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

    def verdict(candidate: Candidate, position: int) -> str | None:
        if position not in candidate.verdicts:
            _, _, text, context = candidate.rows[position]
            if inputs.lexicon.false_match(text, candidate.entry) is not None:
                reason: str | None = "word boundary"
            else:
                collides = inputs.collisions.reason(Record(text=text, context=context))
                reason = None if collides is None else f"collides by {collides}"
            candidate.verdicts[position] = reason
            if reason is not None:
                census.rejected_sentences[reason] += 1
        return candidate.verdicts[position]

    available: list[list[Candidate]] = [[] for _ in range(len(strata))]
    for candidate in candidates:
        if any(verdict(candidate, position) is None for position in range(len(candidate.rows))):
            available[candidate.stratum].append(candidate)
        else:
            census.idioms_without_sentence += 1
    census.largest_size = strata.largest_size([len(group) for group in available])
    if size > census.largest_size:
        raise SelectionError(
            f"the pool holds too few idioms for {size} at eval3's frequencies: per stratum "
            f"{[len(group) for group in available]} against eval3's {list(strata.targets)}; "
            f"the largest size that fits is {census.largest_size}"
        )

    quotas = strata.allocate(size)
    taken_ids: set[str] = set()
    taken_texts: set[str] = set()
    chosen: list[tuple[str, str, str, str | None, str]] = []
    # The scarcest strata draw first, so a sentence two idioms share goes to the scarce one.
    for index in sorted(range(len(strata)), key=lambda i: len(available[i]) - quotas[i]):
        order = sorted(available[index], key=lambda candidate: candidate.entry)
        random.Random(f"{seed}:idioms:{index}").shuffle(order)
        drawn = 0
        for candidate in order:
            if drawn == quotas[index]:
                break
            for position, row in enumerate(candidate.rows):
                if verdict(candidate, position) is not None:
                    continue
                if row[0] in taken_ids or row[2] in taken_texts:
                    census.rejected_sentences["drawn for another idiom"] += 1
                    continue
                taken_ids.add(row[0])
                taken_texts.add(row[2])
                chosen.append((*row, candidate.entry))
                drawn += 1
                break
        if drawn < quotas[index]:
            raise SelectionError(
                f"stratum {index} ran out of idioms at {drawn} of {quotas[index]}: the sentences "
                "its idioms share are drawn for other idioms"
            )
        census.strata.append(
            {
                "stratum": index,
                "lower": strata.lowers[index],
                "upper": strata.upper(index),
                "eval3": strata.targets[index],
                "available": len(available[index]),
                "drawn": drawn,
            }
        )
    census.strata.sort(key=lambda stratum: stratum["stratum"])
    frame = pl.DataFrame(
        chosen, schema=[*SAMPLE_COLUMNS, "entry"], orient="row"
    ).sort("entry")
    record_checks(frame, inputs, eligible, in_eval, census)
    drawn_occurrences = sorted(eligible[entry] for entry in frame["entry"])
    eval_occurrences = sorted(in_eval.values())
    census.quantiles = {
        f"p{round(share * 100)}": {
            "eval3": quantile(eval_occurrences, share),
            "drawn": quantile(drawn_occurrences, share),
        }
        for share in QUANTILES
    }
    census.ks_distance = round(ks_distance(drawn_occurrences, eval_occurrences), 4)
    census.sources = Counter(frame["source"].to_list())
    census.entry_lengths = Counter(len(entry) for entry in frame["entry"])
    return frame, census


def record_checks(
    frame: pl.DataFrame,
    inputs: Inputs,
    eligible: Mapping[str, int],
    in_eval: Mapping[str, int],
    census: DrawCensus,
) -> None:
    """Check the drawn records from scratch, recording what each check found."""
    entries = frame["entry"].to_list()
    texts = " ".join(inputs.eval_texts)
    census.checks = {
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
        if census.checks[name] != expected:
            raise SelectionError(f"{name}: {census.checks[name]} of {expected} drawn records")
    for name in ("held by eval3", "colliding with evaluation", "crossed at the word boundary"):
        if census.checks[name]:
            raise SelectionError(f"{name}: {census.checks[name]} drawn records")


def write_draw(out: Path, frame: pl.DataFrame, census: DrawCensus) -> judge.CandidateSet:
    """Write the draw as a candidate set under *out*, its census beside it."""
    candidates = judge.write_candidates(out, frame)
    (out / "draw.json").write_text(
        json.dumps(census.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log.info(
        "idiom draw written",
        records=frame.height,
        largest_size=census.largest_size,
        ks_distance=census.ks_distance,
        out=str(out),
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
