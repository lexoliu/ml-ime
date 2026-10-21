"""Any system's answers as a dump, the row shape every slice of eval3 reads.

``fused-eval --dump`` and ``mlime eval e2e`` write each record's ranked
hypotheses as JSON Lines, one ``ime_eval::DumpRow`` per record. The other
systems keep their answers in their own files: the generator an answers file
of ``{"record", "answer"}`` lines (:func:`mlime.endpoint.read_answers`), the
GUI engines a journal of committed sentences (:func:`mlime.gui.read_results`),
RIME a report of committed sentences (:mod:`mlime.rime`). Each becomes rows of
the same shape here, holding the one sentence the system answered, so the
slices measure every system through one definition (``ime-knowledge``,
``ime-ambiguity``). A system that answered nothing for a record gets a row
with no hypotheses: a miss, as every report counts it.

A one-sentence system has no score; its hypothesis carries 0.0, which nothing
downstream reads when a row holds one hypothesis.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlime.endpoint import read_answers
from mlime.gui import DEV_SHARE, read_eval_set, read_results, slice_indices
from mlime.rescore import read_dump


@dataclass(frozen=True)
class DumpRow:
    """One record's expected text and the sentences a system answered, best first."""

    record: int
    text: str
    hypotheses: tuple[str, ...]

    def as_json(self) -> str:
        """The row as one dump line, without its newline."""
        return json.dumps(
            {
                "record": self.record,
                "text": self.text,
                "hypotheses": [{"text": text, "score": 0.0} for text in self.hypotheses],
            },
            ensure_ascii=False,
        )


def answered(record: int, text: str, sentence: str | None) -> DumpRow:
    """The row of a system that answered *sentence*, or nothing when it is empty or absent."""
    return DumpRow(record=record, text=text, hypotheses=(sentence,) if sentence else ())


def answer_rows(dump: Path, eval_set: Path, answers: Path) -> list[DumpRow]:
    """The generator's answers for the records of the *dump* it was run on.

    The dump names the records the generator was asked about -- a whole test
    slice, or a subset of it -- and a record the answers file lacks went
    unanswered.
    """
    known = read_answers(answers, str)
    return [
        answered(record.index, record.text, known.get(record.index))
        for record in read_dump(dump, eval_set)
    ]


def journal_rows(
    results: Path, eval_set: Path, slice_: str, dev_share: float = DEV_SHARE
) -> list[DumpRow]:
    """A GUI engine's committed sentences for the records of *slice_* its journal holds."""
    rows = read_eval_set(eval_set)
    keep = set(slice_indices(rows, slice_, dev_share))
    _, typed = read_results(results)
    return [
        answered(row.index, str(rows[row.index]["text"]), row.committed)
        for row in typed
        if row.index in keep
    ]


def report_rows(report: Path, eval_set: Path) -> list[DumpRow]:
    """RIME's committed sentences from the report ``mlime eval rime --out`` wrote."""
    rows = read_eval_set(eval_set)
    typed: Sequence[dict[str, Any]] = json.loads(report.read_text(encoding="utf-8"))["typed"]
    dumped = []
    for line in typed:
        index = int(line["record"])
        text = str(rows[index]["text"])
        if line["expected"] != text:
            raise ValueError(
                f"record {index} of {report} expects {line['expected']!r} but {eval_set} "
                f"holds {text!r}; the report was typed from another set"
            )
        dumped.append(answered(index, text, str(line["committed"])))
    return dumped


def write_dump(rows: Iterable[DumpRow], out: Path) -> int:
    """Write *rows* as a dump file; return how many there were."""
    lines = [row.as_json() + "\n" for row in rows]
    out.write_text("".join(lines), encoding="utf-8")
    return len(lines)
