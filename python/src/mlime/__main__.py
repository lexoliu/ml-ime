"""Command line entry point for the ml-ime data pipeline."""

from __future__ import annotations

from pathlib import Path

import typer

from mlime.logging import configure

app = typer.Typer(help="ml-ime data pipeline", no_args_is_help=True)


@app.callback()
def main() -> None:
    """ml-ime data pipeline."""


@app.command("gen-pinyin-tables")
def gen_pinyin_tables(
    out_dir: Path = typer.Option(  # noqa: B008
        Path("crates/ime-pinyin/data"), help="Directory to write the tables into"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Regenerate the syllable inventory and character->pinyin table from pypinyin."""
    configure(verbose)
    from mlime.data.pinyin_tables import build

    build(out_dir)


if __name__ == "__main__":
    app()
