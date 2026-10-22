"""The idiom selection set (#243): mined sentences the full judge kept, as samples to annotate.

#212's word arcs carry a lexicon prior that dev cannot choose: dev holds too
few idioms, and test has been looked at. The prior is chosen instead on a set
of its own, built here from the FineWeb-2 mining of #225, which no model of the
fused decoder has trained on.

A mined record enters only when the judge kept it at all three of its rules
(the word boundary, Jev, and the review), and only for an entry of the idiom
category; every other verdict, an unfinished one included, keeps it out. Each
idiom contributes one record, the one with the smallest id (a pick that does
not depend on file order), so that the set measures idioms rather than the
handful the web repeats. A record that collides with any
evaluation record is dropped: the same record, the same target sentence, or
a run of :data:`WINDOW` Han characters in common between the two records'
context followed by their target, the near-duplicate rule the ingestion's
leakage guard (``ime_corpus::leak``) applies to whole documents.

What comes out is a samples shard in the layout ``ime-cli g2p annotate``
reads; the annotation and ``ime-cli export eval-set --typing abbreviated``
then make the set exactly the way eval3's abbreviated twin was made. The
census beside it records what was dropped and how the idioms are distributed.

The whole recipe, from the repository root, ``<root>`` a fresh data root::

    mlime eval idiom-set --mined data/corpus-225/ingest/fineweb2-mined \\
        --entries data/corpus-225/inputs/entries-215.tsv \\
        --against data/run3_pool/eval3.jsonl --against data/run3_pool/eval3-abbreviated.jsonl \\
        --against data/run3_pool/eval3-mixed.jsonl --against data/run3_pool/pool.jsonl \\
        --out <root>
    ime-cli g2p annotate --data-dir <root>
    ime-cli export eval-set --data-dir <root> --out <root>/idioms-abbreviated.jsonl \\
        --size 500 --seed 243 --typing abbreviated
    ime-cli emit-lattice --eval-set <root>/idioms-abbreviated.jsonl \\
        --out <root>/lattice.jsonl --emittable data/route-a-assets-v2/emittable.txt

The lattice is emitted without ``--arcs``, as #213's were, and is named
``lattice.jsonl`` because the route A kernel finds its assets by that name.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import polars as pl

from mlime.logging import log

from .text import HAN

#: The mined source, as the ingestion and the judge name it.
SOURCE = "fineweb2-mined"

#: The judge's rules, in the order a unit faces them; a unit is kept only when
#: its verdict log holds all three, in this order, each keeping it.
STAGES = ("boundary", "jev", "review")

#: The category of #215's entry table the set is drawn from.
CATEGORY = "idiom"

#: Han characters two records must share in a row to count as the same text,
#: ``ime_corpus::leak::WINDOW``.
WINDOW = 13

#: The columns of a samples shard, as ``ime-cli g2p annotate`` reads them.
SAMPLE_COLUMNS = ("id", "source", "text", "context")


class SelectionError(RuntimeError):
    """The inputs contradict each other; the set is not built from them."""


def kept_units(lines: Iterable[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """The (record, entry) units every rule of the judge kept.

    A unit whose verdicts are out of order or repeated is an error rather than
    a rejection: the verdict log is append-only and resumable, so either means
    two runs wrote into one log.
    """
    verdicts: dict[tuple[str, str], list[tuple[str, bool]]] = {}
    for line in lines:
        key = (str(line["id"]), str(line["entry"]))
        verdicts.setdefault(key, []).append((str(line["stage"]), bool(line["keep"])))
    kept = set()
    for key, stages in verdicts.items():
        names = tuple(stage for stage, _ in stages)
        if names != STAGES[: len(names)]:
            raise SelectionError(f"verdicts out of order for {key}: {names}")
        if any(not keep for _, keep in stages[:-1]):
            raise SelectionError(f"{key} was judged again after a rule rejected it: {stages}")
        if names == STAGES and stages[-1][1]:
            kept.add(key)
    return kept


def read_verdicts(judge_dir: Path) -> Iterator[dict[str, Any]]:
    """Every verdict line of the judge's logs, in file order."""
    paths = sorted(judge_dir.glob("*.jsonl"))
    if not paths:
        raise SelectionError(f"no verdict log under {judge_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


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


@dataclass
class Census:
    """How the set came out of the judged records."""

    kept_units: int = 0
    kept_idiom_units: int = 0
    idioms: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    records: int = 0
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
    rows: pl.DataFrame,
    kept: set[tuple[str, str]],
    entries: pl.DataFrame,
    collisions: Collisions,
) -> tuple[pl.DataFrame, Census]:
    """One record per kept idiom, none colliding with evaluation; and the census.

    *rows* are the mined records (``id``, ``source``, ``text``, ``context``);
    *entries* is #215's table (``entry``, ``category``, ``occurrences``).
    """
    census = Census(kept_units=len(kept))
    if rows["id"].n_unique() != rows.height:
        raise SelectionError("the mined records repeat an id")
    by_id = {row["id"]: row for row in rows.iter_rows(named=True)}
    category = dict(zip(entries["entry"], entries["category"], strict=True))
    occurrences = dict(zip(entries["entry"], entries["occurrences"], strict=True))

    candidates: dict[str, list[str]] = {}
    for record_id, entry in sorted(kept):
        if entry not in category:
            raise SelectionError(f"the kept entry {entry!r} is not in the entry table")
        if record_id not in by_id:
            raise SelectionError(f"the kept record {record_id} is not among the mined records")
        if entry not in by_id[record_id]["text"]:
            raise SelectionError(f"the kept record {record_id} does not hold {entry!r}")
        if category[entry] == CATEGORY:
            census.kept_idiom_units += 1
            candidates.setdefault(entry, []).append(record_id)
    census.idioms = len(candidates)
    census.records_per_idiom_kept = Counter(len(ids) for ids in candidates.values())

    chosen: list[dict[str, Any]] = []
    taken_ids: set[str] = set()
    taken_texts: set[str] = set()
    for entry, ids in sorted(candidates.items()):
        pick = None
        for record_id in ids:
            row = by_id[record_id]
            if record_id in taken_ids or row["text"] in taken_texts:
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
    sources = set(frame["source"])
    if sources != {SOURCE}:
        raise SelectionError(f"the mined records are tagged {sorted(sources)}, not {SOURCE!r}")
    return frame, census


def build(
    mined: Path,
    entry_table: Path,
    against: Iterable[Path],
    out: Path,
) -> Census:
    """Write the selection's samples shard, its idiom list and its census under *out*.

    *mined* is the ingestion's ``fineweb2-mined`` directory: ``unjudged/``
    holds the records and ``judge/`` the verdict logs.
    """
    kept = kept_units(read_verdicts(mined / "judge"))
    shards = sorted((mined / "unjudged").glob("*.parquet"))
    if not shards:
        raise SelectionError(f"no mined shard under {mined / 'unjudged'}")
    rows = pl.concat([pl.read_parquet(path, columns=list(SAMPLE_COLUMNS)) for path in shards])
    entries = pl.read_csv(entry_table, separator="\t", quote_char=None).select(
        "entry", "category", "occurrences"
    )
    collisions = Collisions.of(record for path in against for record in read_records(path))
    frame, census = select(rows, kept, entries, collisions)

    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    if any(samples.iterdir()):
        raise SelectionError(f"{samples} is not empty; a set is built once, into a fresh root")
    frame.select(SAMPLE_COLUMNS).write_parquet(samples / f"{SOURCE}-00000.parquet")
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
