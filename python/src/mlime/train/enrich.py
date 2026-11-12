"""Enrichment shards: the sentences of the units the corpus holds too rarely.

An idiom that occurs in a dozen sentences of a forty-million-sentence corpus is
typed as initials a handful of times in a whole epoch, however the typing
augmentation is tuned. Lifting repeats the sentences that hold it, so that each
unit reaches a floor of occurrences: every repeat is a row of its own, with its
own id, so ``example_rng(seed, epoch, id)`` draws it a typing of its own, and the
labels are the original rows' labels copied over, never recomputed.

One streaming pass reads every sample shard in worker processes and finds the
units of each sentence with :class:`~mlime.train.units.UnitMatcher` -- the
matcher the typing augmentation uses -- keeping only the sentences that hold
one and whose characters are all emittable. Their labels are then looked up by
id in the label shards: a corpus's labels need not be sharded as its samples
are, and only about one sentence in forty holds a unit, so nothing but those
ids is ever held. A sentence counts if the training builder would keep it: it
has a label of its length. Repeating a sentence the builder drops would add
nothing to the unit's exposure.

Extra sample shards -- sentences mined and judged elsewhere -- pass through
unchanged. Those that have label shards go beside the lifted rows; the rest are
written apart, counted in the census, because a sample shard without its labels
stops a run that reads it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from mlime.data.corpus import SAMPLE_SCHEMA
from mlime.data.shards import ShardWriter, shard_paths
from mlime.logging import log
from mlime.train.labels import LABEL_SCHEMA
from mlime.train.units import UnitMatcher

#: The occurrences each unit is lifted to, and how often one sentence may be repeated.
DEFAULT_FLOOR = 20
DEFAULT_MAX_REPEATS = 8

#: Rows per output shard.
DEFAULT_ROWS_PER_SHARD = 100_000

#: The shard prefixes the builder writes: lifted rows, and the rows passed through.
LIFT_PREFIX = "lift"
EXTRA_PREFIX = "mined"

#: Where the passed-through rows that have no labels go, under the output directory.
NEEDS_LABELS = "needs-labels"


@dataclass(frozen=True)
class Match:
    """A sentence that holds units: its sample row and how often each unit occurs in it."""

    id: str
    source: str
    text: str
    context: str | None
    units: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ShardMatches:
    """What one sample shard holds of the units, as one worker found it."""

    rows: int
    not_emittable: int
    matches: tuple[Match, ...]


def scan_shard(shard: Path, matcher: UnitMatcher, emittable: frozenset[str]) -> ShardMatches:
    """The sentences of *shard* that hold a unit and are emittable throughout."""
    samples = pl.read_parquet(shard)
    texts = samples["text"].to_list()
    found = {
        position: hits for position, text in enumerate(texts) if (hits := matcher.occurrences(text))
    }
    ids = samples["id"].to_list()
    sources = samples["source"].to_list()
    contexts = samples["context"].to_list()
    matches: list[Match] = []
    not_emittable = 0
    for position, hits in found.items():
        text = texts[position]
        if not emittable.issuperset(text):
            not_emittable += 1
            continue
        units = Counter(text[start:end] for start, end in hits)
        matches.append(
            Match(ids[position], sources[position], text, contexts[position], tuple(units.items()))
        )
    return ShardMatches(rows=samples.height, not_emittable=not_emittable, matches=tuple(matches))


def read_readings(
    label_dirs: Sequence[Path], wanted: Iterable[str]
) -> dict[str, tuple[str, ...] | None]:
    """The readings of the sentences *wanted*, by id, from every label shard under *label_dirs*.

    ``None`` where a label shard holds the id but refused it; an id no shard
    holds is absent from the result.
    """
    ids = pl.Series("id", list(wanted), dtype=pl.String)
    readings: dict[str, tuple[str, ...] | None] = {}
    for directory in label_dirs:
        shards = shard_paths(directory, "*")
        if not shards:
            raise FileNotFoundError(f"no label shards under {directory}")
        for shard in shards:
            held = pl.read_parquet(shard).filter(pl.col("id").is_in(ids.implode()))
            for sample_id, syllables in zip(
                held["id"].to_list(), held["syllables"].to_list(), strict=True
            ):
                readings[sample_id] = None if syllables is None else tuple(syllables)
    return readings


@dataclass(frozen=True)
class Row:
    """One sentence kept for lifting: its sample row and its reading."""

    id: str
    source: str
    text: str
    context: str | None
    syllables: tuple[str, ...]


@dataclass
class Corpus:
    """The trainable sentences that hold units, in the corpus's order.

    ``counts`` is each unit's occurrences over them. ``stash`` lists for each
    unit the first ``floor`` sentences that hold it, as ``(index, times)``: as
    many as a lifted unit can ever need, since a unit with ``floor`` sentences
    is not lifted. ``rows`` maps an index to the sentence.
    """

    floor: int
    rows_seen: int = 0
    not_emittable: int = 0
    matched: int = 0
    duplicates: int = 0
    unlabelled: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    stash: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    rows: dict[int, Row] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)

    def add(self, matches: Iterable[Match], readings: dict[str, tuple[str, ...] | None]) -> None:
        """Take the next sentences in corpus order.

        A sentence whose id was taken before is a duplicate and is skipped: the
        copies' ids are derived from it, and two would collide.
        """
        for match in matches:
            if match.id in self.seen:
                self.duplicates += 1
                continue
            self.seen.add(match.id)
            syllables = readings.get(match.id)
            if syllables is None or len(syllables) != len(match.text):
                self.unlabelled += 1
                continue
            index = len(self.seen)
            for entry, times in match.units:
                self.counts[entry] += times
                held = self.stash.setdefault(entry, [])
                if len(held) < self.floor:
                    held.append((index, times))
                    self.rows[index] = Row(
                        match.id, match.source, match.text, match.context, syllables
                    )


@dataclass(frozen=True)
class Plan:
    """How many extra copies each kept sentence gets, and where every unit ends up."""

    copies: dict[int, int]
    #: Each unit that reached the floor with copies, with its occurrences then.
    lifted: dict[str, int]
    #: Each unit that stays under the floor after its copies, with its occurrences.
    under_floor: dict[str, int]


def plan_lift(corpus: Corpus, blocked: frozenset[str], max_repeats: int) -> Plan:
    """Choose the copies that bring every present, unblocked unit under the floor up to it.

    Units are taken in order of their occurrences, rarest first, and each takes
    the copy of whichever of its sentences has the fewest so far -- the first
    such, in corpus order -- until it reaches the floor or every sentence is at
    *max_repeats*. A copy counts for every unit its sentence holds, so a sentence
    shared by two units serves both and the cap holds per sentence.
    """
    floor = corpus.floor
    copies: dict[int, int] = {}
    effective: dict[str, int] = {}
    wanting = sorted(
        (count, entry)
        for entry, count in corpus.counts.items()
        if 0 < count < floor and entry not in blocked
    )
    held_by: dict[int, list[tuple[str, int]]] = {}
    for _, entry in wanting:
        for index, times in corpus.stash[entry]:
            held_by.setdefault(index, []).append((entry, times))
    for count, entry in wanting:
        sentences = [index for index, _ in corpus.stash[entry]]
        have = effective.setdefault(entry, count)
        while have < floor:
            index = min(sentences, key=lambda sentence: copies.get(sentence, 0))
            if copies.get(index, 0) >= max_repeats:
                break
            copies[index] = copies.get(index, 0) + 1
            for other, times in held_by[index]:
                effective[other] = effective.setdefault(other, corpus.counts[other]) + times
            have = effective[entry]
    # A copy serves every unit its sentence holds, a rarer one's included, so
    # the final occurrences are read once all the copies are placed.
    lifted = {entry: have for entry, have in effective.items() if have >= floor}
    under_floor = {entry: have for entry, have in effective.items() if have < floor}
    return Plan(copies=copies, lifted=lifted, under_floor=under_floor)


def repeat_id(sample_id: str, repeat: int) -> str:
    """The id of the *repeat*-th copy of *sample_id*; each copy has its own, so its own draw."""
    return f"{sample_id}+{repeat}"


def lifted_rows(corpus: Corpus, plan: Plan) -> Iterator[Row]:
    """Every extra copy the plan makes, in a fixed order that mixes sources and units.

    The order is the copies' own id hashes: stable under any worker count, and
    it spreads the copies of one sentence, one unit and one source over the
    output rather than leaving them side by side.
    """
    copies = [
        (repeat_id(corpus.rows[index].id, repeat), corpus.rows[index])
        for index, count in plan.copies.items()
        for repeat in range(1, count + 1)
    ]
    copies.sort(key=lambda item: hashlib.blake2b(item[0].encode(), digest_size=8).digest())
    for new_id, row in copies:
        yield Row(new_id, row.source, row.text, row.context, row.syllables)


def sample_row(row: Row) -> dict[str, object]:
    """*row* as a ``SAMPLE_SCHEMA`` row."""
    return {"id": row.id, "source": row.source, "text": row.text, "context": row.context}


def label_row(row: Row) -> dict[str, object]:
    """*row*'s reading as a ``LABEL_SCHEMA`` row."""
    return {"id": row.id, "syllables": list(row.syllables), "refusal": None}


def write_lift(out: Path, rows: Iterator[Row], rows_per_shard: int) -> int:
    """Write *rows* as ``lift`` sample shards and their label shards, in step; return the count."""
    written = 0
    with (
        ShardWriter(out / "samples", LIFT_PREFIX, SAMPLE_SCHEMA, rows_per_shard) as samples,
        ShardWriter(out / "labels", LIFT_PREFIX, LABEL_SCHEMA, rows_per_shard) as labels,
    ):
        for row in rows:
            samples.write(sample_row(row))
            labels.write(label_row(row))
            written += 1
    return written


@dataclass(frozen=True)
class ExtraReport:
    """What became of one extra sample shard."""

    source: str
    written_as: str
    rows: int
    labelled: bool
    labelled_rows: int

    def as_dict(self) -> dict[str, Any]:
        """The report as a census record."""
        return dict(vars(self))


def pass_through(extras: Sequence[Path], extra_labels: Path | None, out: Path) -> list[ExtraReport]:
    """Copy each extra sample shard under *out*, with its labels when it has them.

    A shard with a label shard of the same name under *extra_labels* goes to
    ``samples/mined-NNNNN.parquet`` with the labels of its rows beside it. One
    without goes to ``needs-labels/samples/`` and is reported, so a training run
    that reads ``samples/`` never meets a shard it cannot label.
    """
    reports: list[ExtraReport] = []
    for index, path in enumerate(extras):
        frame = pl.read_parquet(path)
        if frame.schema != SAMPLE_SCHEMA:
            raise ValueError(f"{path} is not a sample shard: {dict(frame.schema)}")
        name = f"{EXTRA_PREFIX}-{index:05d}.parquet"
        label_path = None if extra_labels is None else extra_labels / path.name
        if label_path is not None and label_path.is_file():
            labels = pl.read_parquet(label_path)
            if labels.schema != LABEL_SCHEMA:
                raise ValueError(f"{label_path} is not a label shard: {dict(labels.schema)}")
            labels = labels.filter(pl.col("id").is_in(frame["id"].implode()))
            (out / "samples").mkdir(parents=True, exist_ok=True)
            (out / "labels").mkdir(parents=True, exist_ok=True)
            frame.write_parquet(out / "samples" / name)
            labels.write_parquet(out / "labels" / name)
            reports.append(
                ExtraReport(str(path), f"samples/{name}", frame.height, True, labels.height)
            )
        else:
            (out / NEEDS_LABELS / "samples").mkdir(parents=True, exist_ok=True)
            frame.write_parquet(out / NEEDS_LABELS / "samples" / name)
            reports.append(
                ExtraReport(str(path), f"{NEEDS_LABELS}/samples/{name}", frame.height, False, 0)
            )
    return reports


def blocked_entries(entries: Iterable[str], emittable: frozenset[str]) -> frozenset[str]:
    """The entries that hold a character outside the emittable set."""
    return frozenset(entry for entry in entries if not emittable.issuperset(entry))


def histogram(values: Iterable[int]) -> dict[str, int]:
    """*values* counted, keyed by value in ascending order, as JSON keys."""
    return {str(value): count for value, count in sorted(Counter(values).items())}


def enrich(
    samples: Path,
    labels: Sequence[Path],
    matcher: UnitMatcher,
    emittable: frozenset[str],
    out: Path,
    floor: int = DEFAULT_FLOOR,
    max_repeats: int = DEFAULT_MAX_REPEATS,
    rows_per_shard: int = DEFAULT_ROWS_PER_SHARD,
    workers: int | None = None,
    extras: Sequence[Path] = (),
    extra_labels: Path | None = None,
) -> dict[str, Any]:
    """Build the enrichment shards under *out* and return the census it wrote beside them.

    *labels* are the directories whose label shards hold the readings of the
    sentences under *samples*, however they are sharded. *out* must not exist.
    The pass is deterministic: the same inputs write the same bytes whatever
    *workers* is.
    """
    if floor < 1 or max_repeats < 1:
        raise ValueError(
            f"the floor and the repeat cap are positive, got {floor} and {max_repeats}"
        )
    if out.exists():
        raise FileExistsError(f"{out} exists; the builder never overwrites a directory")
    shards = shard_paths(samples, "*")
    if not shards:
        raise FileNotFoundError(f"no sample shards under {samples}")
    workers = workers or os.cpu_count()
    if workers is None:
        raise ValueError("the cores are not countable here; pass workers")
    scan = partial(scan_shard, matcher=matcher, emittable=emittable)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scanned = list(pool.map(scan, shards, chunksize=1))
    held = [match for shard in scanned for match in shard.matches]
    log.info("samples scanned", shards=len(shards), matched=len(held))
    readings = read_readings(labels, (match.id for match in held))
    corpus = Corpus(floor)
    corpus.rows_seen = sum(shard.rows for shard in scanned)
    corpus.not_emittable = sum(shard.not_emittable for shard in scanned)
    corpus.matched = corpus.not_emittable + len(held)
    corpus.add(held, readings)
    blocked = blocked_entries(matcher.entries, emittable)
    plan = plan_lift(corpus, blocked, max_repeats)
    out.mkdir(parents=True)
    written = write_lift(out, lifted_rows(corpus, plan), rows_per_shard)
    reports = pass_through(extras, extra_labels, out)
    census = census_of(
        corpus, len(shards), matcher, blocked, plan, written, reports, max_repeats, emittable
    )
    (out / "census.json").write_text(
        json.dumps(census, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "under-floor.tsv").write_text(
        "entry\toccurrences\tshortfall\n"
        + "".join(
            f"{entry}\t{have}\t{floor - have}\n" for entry, have in sorted(plan.under_floor.items())
        ),
        encoding="utf-8",
    )
    return census


def census_of(
    corpus: Corpus,
    shards: int,
    matcher: UnitMatcher,
    blocked: frozenset[str],
    plan: Plan,
    written: int,
    extras: Sequence[ExtraReport],
    max_repeats: int,
    emittable: frozenset[str],
) -> dict[str, Any]:
    """The census: what the pass saw, what it lifted and what stays under the floor."""
    floor = corpus.floor
    present = {entry for entry, count in corpus.counts.items() if count > 0} - blocked
    at_floor = {entry for entry in present if corpus.counts[entry] >= floor}
    final = {entry: corpus.counts[entry] for entry in present} | plan.lifted | plan.under_floor
    under = {entry: have for entry, have in final.items() if have < floor}
    shortfalls = [floor - have for have in under.values()]
    return {
        "settings": {
            "floor": floor,
            "max_repeats": max_repeats,
            "emittable_characters": len(emittable),
            "unit_entries": len(matcher),
            "unit_digest": matcher.digest,
        },
        "scan": {
            "shards": shards,
            "rows": corpus.rows_seen,
            "rows_holding_a_unit": corpus.matched,
            "of_which_not_emittable": corpus.not_emittable,
            "of_which_duplicate_ids": corpus.duplicates,
            "of_which_unlabelled": corpus.unlabelled,
            "trainable_unit_occurrences": sum(corpus.counts.values()),
        },
        "entries": {
            "total": len(matcher),
            "blocked": len(blocked),
            "absent": len(matcher) - len(blocked) - len(present),
            "present": len(present),
            "at_or_above_floor": len(at_floor),
            "lifted_to_floor": len(plan.lifted),
            "lifted_short_of_floor": len(plan.under_floor),
            "under_floor_after": len(under),
        },
        "lift": {
            "rows_written": written,
            "sentences_repeated": len(plan.copies),
            "repeats_histogram": histogram(plan.copies.values()),
        },
        "under_floor": {
            "entries": len(under),
            "total_shortfall": sum(shortfalls),
            "shortfall_histogram": histogram(shortfalls),
            "listed_in": "under-floor.tsv",
        },
        "extras": [report.as_dict() for report in extras],
        "extra_rows_needing_labels": sum(report.rows for report in extras if not report.labelled),
    }
