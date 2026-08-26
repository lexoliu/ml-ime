"""Turn the context probe's JSONL log into an availability table.

The probe answers one question: which applications will tell an input method what
text sits around the caret. Milestone 3's headline number -- how much conditioning
on that text helps -- is only worth measuring in applications that supply it, so
this table decides how much of the corpus should carry context at all.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

DEFAULT_LOG = Path.home() / "Library" / "Logs" / "mlime-context-probe.jsonl"


@dataclass
class ClientSummary:
    """What one application was observed to provide."""

    bundle_identifier: str
    keystrokes: int = 0
    selected_range_available: int = 0
    document_length_available: int = 0
    context_before_returned: list[int] = field(default_factory=list)
    context_after_returned: list[int] = field(default_factory=list)
    _caret_positions: set[int] = field(default_factory=set)

    @property
    def caret_moves(self) -> bool:
        """Whether the caret was ever seen in more than one place.

        A client that always reports the same offset is answering, but not
        usefully -- the distinction a bare availability count would miss.
        """
        return len(self._caret_positions) > 1

    @property
    def verdict(self) -> str:
        """One word for how usable this client's context is."""
        if self.selected_range_available == 0:
            return "none"
        if not any(self.context_before_returned):
            return "caret only"
        if not self.caret_moves:
            return "static"
        return "full"


def summarise(log_path: Path) -> list[ClientSummary]:
    """Aggregate the probe log, most-observed client first."""
    if not log_path.exists():
        raise FileNotFoundError(
            f"no probe log at {log_path}; install the probe, select it, and type in a few apps"
        )
    clients: dict[str, ClientSummary] = defaultdict(lambda: ClientSummary(bundle_identifier="?"))
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = record["clientBundleIdentifier"] or "<unknown>"
        summary = clients[key]
        summary.bundle_identifier = key
        summary.keystrokes += 1
        location = record["selectedRange"]["location"]
        if location is not None:
            summary.selected_range_available += 1
            summary._caret_positions.add(location)
        if record["documentLength"] is not None:
            summary.document_length_available += 1
        for field_name, sink in (
            ("contextBefore", summary.context_before_returned),
            ("contextAfter", summary.context_after_returned),
        ):
            report = record[field_name]
            if report is not None and report["returnedCharacters"] is not None:
                sink.append(report["returnedCharacters"])
    return sorted(clients.values(), key=lambda c: -c.keystrokes)


def render(summaries: list[ClientSummary]) -> None:
    """Print the availability table."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Host context availability")
    for column in ("Application", "Keys", "Caret", "Length", "Before (median/max)", "Verdict"):
        table.add_column(column)
    for summary in summaries:
        before = summary.context_before_returned
        table.add_row(
            summary.bundle_identifier,
            str(summary.keystrokes),
            f"{summary.selected_range_available}/{summary.keystrokes}",
            f"{summary.document_length_available}/{summary.keystrokes}",
            f"{int(median(before))}/{max(before)}" if before else "-",
            summary.verdict,
        )
    Console().print(table)
