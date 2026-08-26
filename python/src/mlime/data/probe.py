"""Check both annotators against one sentence before spending a corpus on them.

The LLM annotator's failure modes are all silent from a distance: a wrong base
URL, an expired key, a model that ignores the JSON instruction, a proxy that
drops ``reasoning_effort``. Each of those turns into a wall of refusals an hour
into a run. So there is a command that annotates one polyphone-dense sentence and
prints both answers next to each other, and it is the first thing to run on a new
machine or in a fresh kernel.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from mlime.settings import LlmSettings

from .g2p import Annotator, Reading, Refusal, compare
from .text import han_characters


async def probe(sentence: str, g2pw_model: Path | None, console: Console | None = None) -> None:
    """Annotate *sentence* with both systems and print them side by side."""
    from .g2pw_annotator import DEFAULT_MODEL_DIR, G2pwAnnotator
    from .llm_annotator import LlmAnnotator

    console = console or Console()
    settings = LlmSettings.load()
    console.print(f"[bold]endpoint[/bold] {settings.base_url}  [bold]model[/bold] {settings.model}")

    first: Annotator = G2pwAnnotator(g2pw_model or DEFAULT_MODEL_DIR)
    second: Annotator = LlmAnnotator.from_settings(settings, concurrency=1)
    outcomes = [(await annotator.annotate([sentence]))[0] for annotator in (first, second)]

    for annotator, outcome in zip((first, second), outcomes, strict=True):
        if isinstance(outcome, Refusal):
            console.print(f"[red]{annotator.name} refused:[/red] {outcome.reason}")
    readings = [outcome for outcome in outcomes if isinstance(outcome, Reading)]
    if len(readings) != 2:
        raise RuntimeError("both annotators must answer before they can be compared")

    comparison = compare(sentence, readings[0], readings[1])
    table = Table(title=sentence)
    table.add_column("#", justify="right")
    table.add_column("Char")
    table.add_column(first.name)
    table.add_column(second.name)
    table.add_column("Agree")
    for index, character in enumerate(han_characters(sentence)):
        agrees = comparison.agree[index]
        table.add_row(
            str(index),
            character,
            comparison.first[index],
            comparison.second[index],
            "[green]yes[/green]" if agrees else "[red]no[/red]",
        )
    console.print(table)
    console.print(
        f"[bold]toneless agreement[/bold] {sum(comparison.agree)}/{len(comparison.agree)}"
    )
