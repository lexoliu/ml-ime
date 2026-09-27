"""Score a GUI input-method run exactly as ``mlime eval rime`` scores RIME.

Issue #59. Apple Pinyin, Sogou and Baidu expose no API a harness could bind,
so ``macos/ImeDrive`` types each record's keystrokes into a text view of its
own process through ``CGEvent``, reads the candidate window through the
accessibility tree, and accepts the first candidate until the composition is
consumed -- the sentence a user gets by pressing space. It writes one JSON
line per record: the record index, the engine, the committed text, the first
candidate page and the wall time.

This module joins those lines with the eval set (the committed text is scored
against the record's expected text, as everywhere else), restricts to a slice
with the same keyed split ``EvalRecord::digest`` implements -- BLAKE2b-64 of
the record's fields, dev share 0.0905 -- and reuses ``rime.evaluate`` so every
field means what the RIME row means.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlime.rescore import DumpedRecord
from mlime.rime import RimeResult, Typed, evaluate

#: The share of each eval set reserved for tuning, as in
#: ``ime-eval``'s ``--dev-share 0.0905``.
DEV_SHARE = 0.0905


def digest(pinyin: str, text: str, context: str | None) -> int:
    """The record's 64-bit digest, identical to ``EvalRecord::digest`` in Rust.

    BLAKE2b with an 8-byte output over the pinyin, a NUL, the text, a NUL and
    the context (empty string where the record has none), read big-endian.
    """
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(pinyin.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(text.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update((context or "").encode("utf-8"))
    return int.from_bytes(hasher.digest(), "big")


def slice_indices(rows: Sequence[dict[str, Any]], slice_: str, dev_share: float) -> list[int]:
    """The eval-set indices belonging to *slice_* ("test", "dev" or "all").

    A record is development data when ``digest / u64::MAX < dev_share`` — the
    same comparison ``Slice::holds`` makes.
    """
    assert 0.0 <= dev_share <= 1.0, f"dev_share {dev_share} is not a share"
    limit = dev_share * float(2**64 - 1)
    dev = {
        index
        for index, row in enumerate(rows)
        if digest(str(row["pinyin"]), str(row["text"]), row.get("context") or None) < limit
    }
    if slice_ == "dev":
        return sorted(dev)
    if slice_ == "test":
        return [index for index in range(len(rows)) if index not in dev]
    return list(range(len(rows)))


def read_eval_set(path: Path) -> list[dict[str, Any]]:
    """The eval set's rows, in file order."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@dataclass(frozen=True)
class GuiRow:
    """One line of an ime-drive results file."""

    index: int
    committed: str
    first_page: tuple[str, ...]
    wall_ms: int
    candidate_window: bool


def read_results(path: Path) -> tuple[dict[str, Any], list[GuiRow]]:
    """The run's meta line (if any) and one row per typed record."""
    meta: dict[str, Any] = {}
    rows: list[GuiRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "record" not in record:
            meta = record
            continue
        rows.append(
            GuiRow(
                index=int(record["record"]),
                committed=str(record["committed"]),
                first_page=tuple(str(s) for s in record.get("first_page", [])),
                wall_ms=int(record["wall_ms"]),
                candidate_window=bool(record.get("candidate_window", True)),
            )
        )
    return meta, rows


@dataclass(frozen=True)
class GuiReport:
    """What one engine did on one slice of one eval set."""

    engine: str
    meta: dict[str, Any]
    result: RimeResult
    rows: tuple[GuiRow, ...]
    typed: tuple[Typed, ...]

    @property
    def wall_ms(self) -> tuple[int, ...]:
        """Per-record wall times, in record order."""
        return tuple(row.wall_ms for row in self.rows)

    def as_dict(self) -> dict[str, object]:
        """A JSON-friendly view, with every sentence so the answers can be inspected."""
        return {
            "engine": self.engine,
            "run": self.meta,
            "result": self.result.as_dict(),
            "wall_ms": {
                "total": sum(self.wall_ms),
                "mean": statistics.mean(self.wall_ms) if self.wall_ms else 0.0,
                "median": statistics.median(self.wall_ms) if self.wall_ms else 0.0,
            },
            "typed": [
                {
                    "record": t.record.index,
                    "expected": t.record.text,
                    "committed": t.committed,
                    "first_page": list(t.first_page),
                    "wall_ms": row.wall_ms,
                    "candidate_window": row.candidate_window,
                }
                for t, row in zip(self.typed, self.rows, strict=True)
            ],
        }

    def render(self) -> str:
        """A few lines for the terminal."""
        r = self.result
        wall = f"{sum(self.wall_ms) / 60000:.1f} min" if self.wall_ms else "0 min"
        return "\n".join(
            [
                f"engine {self.engine} ({self.meta.get('input_source_id', 'no meta')})",
                f"{r.records} records: top-1 {r.rime_top1:.4f}, "
                f"first page exact {r.first_page_exact:.4f}, "
                f"characters right {r.characters_right:.4f} of {r.characters}, "
                f"length mismatches {r.length_mismatches}, wall {wall}",
            ]
        )


def measure(
    results: Path, eval_set: Path, engine: str, slice_: str = "test", dev_share: float = DEV_SHARE
) -> GuiReport:
    """Score the committed sentences of *results* against *eval_set*'s *slice_*.

    Only result lines naming a record in the slice are scored; a record the
    harness never typed contributes nothing. The file must have been typed
    with a --slice that covers what is scored ("all" covers any slice, and is
    what files written before --slice existed count as).
    """
    rows = read_eval_set(eval_set)
    keep = set(slice_indices(rows, slice_, dev_share))
    meta, gui_rows = read_results(results)
    ran = meta.get("slice", "all")
    if ran not in ("all", slice_):
        raise ValueError(f"{results} was typed with --slice {ran}; cannot score the {slice_} slice")
    kept = [row for row in gui_rows if row.index in keep]
    if not kept:
        raise ValueError(f"{results} holds no records of the {slice_} slice of {eval_set}")
    typed = [
        Typed(
            record=DumpedRecord(
                index=row.index,
                text=str(rows[row.index]["text"]),
                pinyin=str(rows[row.index]["pinyin"]),
                context=rows[row.index].get("context"),
                hypotheses=(),
            ),
            committed=row.committed,
            first_page=row.first_page,
        )
        for row in kept
    ]
    return GuiReport(
        engine=engine,
        meta=meta,
        result=evaluate(typed),
        rows=tuple(kept),
        typed=tuple(typed),
    )


def write_report(report: GuiReport, out: Path) -> None:
    """Write the report, with every committed sentence, as JSON."""
    out.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", "utf-8")
