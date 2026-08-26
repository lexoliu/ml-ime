"""Command line entry point for the ml-ime data pipeline.

The pipeline is four commands run in order -- ``corpus fetch``, ``corpus
prepare``, ``g2p annotate``, ``export eval-set`` -- with ``g2p probe`` and ``g2p
report`` alongside to check that the annotators work and to publish what they
agreed on. Every one of them takes ``--data-dir``, because the same commands run
against a checkout and inside a Kaggle kernel where no path is the same.

Heavy imports live inside the command bodies: loading an ONNX session or the
``datasets`` library to print ``--help`` would make the whole tool feel broken.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from mlime.data import DataLayout
from mlime.logging import configure

app = typer.Typer(help="ml-ime data pipeline", no_args_is_help=True)
corpus_app = typer.Typer(help="Acquire and filter training corpora", no_args_is_help=True)
g2p_app = typer.Typer(help="Dual pinyin annotation and its agreement report", no_args_is_help=True)
export_app = typer.Typer(help="Emit artefacts for the Rust side", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(g2p_app, name="g2p")
app.add_typer(export_app, name="export")

#: A polyphone-dense sentence: 重 chong/zhong, 还 huan/hai, 得 de/dei, 绿 lv.
PROBE_SENTENCE = "他还了钱还差一点，我得到了那件重要的绿色东西"

DATA_DIR = typer.Option(Path("data"), "--data-dir", help="Root the pipeline reads and writes")
VERBOSE = typer.Option(False, "--verbose", "-v")
SOURCE = typer.Option(None, "--source", help="Source to use; repeatable, defaults to all")
EXCLUDE = typer.Option(
    None,
    "--exclude",
    help="JSON Lines file whose `text` fields are held out of the corpus; repeatable",
)


@app.callback()
def main() -> None:
    """ml-ime data pipeline."""


@app.command("gen-pinyin-tables")
def gen_pinyin_tables(
    out_dir: Path = typer.Option(
        Path("crates/ime-pinyin/data"), help="Directory to write the tables into"
    ),
    verbose: bool = VERBOSE,
) -> None:
    """Regenerate the syllable inventory and character->pinyin table from pypinyin."""
    configure(verbose)
    from mlime.data.pinyin_tables import build

    build(out_dir)


@app.command("probe-report")
def probe_report(
    log: Path = typer.Option(
        None, help="Probe log to read (defaults to the location the probe writes to)"
    ),
    verbose: bool = VERBOSE,
) -> None:
    """Summarise which applications supply text around the caret."""
    configure(verbose)
    from mlime.data.probe_report import DEFAULT_LOG, render, summarise

    render(summarise(log or DEFAULT_LOG))


@corpus_app.command("fetch")
def corpus_fetch(
    source: list[str] = SOURCE,
    data_dir: Path = DATA_DIR,
    limit: int = typer.Option(None, help="Stop after this many upstream documents"),
    verbose: bool = VERBOSE,
) -> None:
    """Stream the upstream corpora into normalised document shards."""
    configure(verbose)
    from mlime.data.corpus import SOURCES, fetch

    layout = DataLayout(data_dir)
    for name in _requested(source):
        fetch(SOURCES[name], layout.documents, limit)


@corpus_app.command("prepare")
def corpus_prepare(
    source: list[str] = SOURCE,
    data_dir: Path = DATA_DIR,
    char_table: Path = typer.Option(
        None, help="ime-pinyin's char_pinyin.tsv; found in the repository when omitted"
    ),
    context_segments: int = typer.Option(3, help="Preceding sentences or turns kept as context"),
    max_context_characters: int = typer.Option(256, help="Longest context carried per sample"),
    min_characters: int = typer.Option(4, help="Shortest target sentence kept"),
    max_characters: int = typer.Option(64, help="Longest target sentence kept"),
    min_han_ratio: float = typer.Option(0.9, help="Least share of Han characters in a target"),
    limit: int = typer.Option(None, help="Stop after this many samples per source"),
    verbose: bool = VERBOSE,
) -> None:
    """Split fetched documents into filtered, deduplicated samples with context."""
    configure(verbose)
    from mlime.data.corpus import default_char_table, load_reference_characters, prepare

    table = char_table or default_char_table()
    if table is None:
        raise typer.BadParameter(
            "no char_pinyin.tsv found above the working directory; pass --char-table"
        )
    layout = DataLayout(data_dir)
    prepare(
        layout.documents,
        layout.samples,
        load_reference_characters(table),
        _requested(source),
        context_segments=context_segments,
        max_context_characters=max_context_characters,
        min_characters=min_characters,
        max_characters=max_characters,
        min_han_ratio=min_han_ratio,
        limit=limit,
    )


@g2p_app.command("probe")
def g2p_probe(
    sentence: str = typer.Option(PROBE_SENTENCE, help="Sentence to annotate with both systems"),
    g2pw_model: Path = typer.Option(None, help="Directory holding (or to hold) the g2pW model"),
    verbose: bool = VERBOSE,
) -> None:
    """Annotate one sentence with both systems and show them side by side."""
    configure(verbose)
    from mlime.data.probe import probe

    asyncio.run(probe(sentence, g2pw_model))


@g2p_app.command("annotate")
def g2p_annotate(
    data_dir: Path = DATA_DIR,
    limit: int = typer.Option(None, help="Stop after this many samples"),
    concurrency: int = typer.Option(8, help="Simultaneous requests in flight to the LLM"),
    batch_size: int = typer.Option(32, help="Sentences handed to the annotators at once"),
    g2pw_model: Path = typer.Option(None, help="Directory holding (or to hold) the g2pW model"),
    verbose: bool = VERBOSE,
) -> None:
    """Annotate prepared samples with both g2p systems and record where they agree."""
    configure(verbose)
    import itertools

    from mlime.data.corpus import Sample
    from mlime.data.g2p import annotate
    from mlime.data.g2pw_annotator import DEFAULT_MODEL_DIR, G2pwAnnotator
    from mlime.data.llm_annotator import LlmAnnotator
    from mlime.settings import LlmSettings

    layout = DataLayout(data_dir)
    samples = itertools.islice(Sample.read(layout.samples), limit)
    asyncio.run(
        annotate(
            samples,
            G2pwAnnotator(g2pw_model or DEFAULT_MODEL_DIR, batch_size=batch_size),
            LlmAnnotator.from_settings(LlmSettings.load(), concurrency=concurrency),
            layout.annotations,
            batch_size=batch_size,
        )
    )


@g2p_app.command("report")
def g2p_report(data_dir: Path = DATA_DIR, verbose: bool = VERBOSE) -> None:
    """Print the agreement rate, its frequency breakdown, and the worst characters."""
    configure(verbose)
    from mlime.data.g2p_report import report

    report(DataLayout(data_dir).annotations)


@export_app.command("eval-set")
def export_eval_set_command(
    data_dir: Path = DATA_DIR,
    out: Path = typer.Option(Path("data/eval.jsonl"), help="JSON Lines file to write"),
    size: int = typer.Option(1000, help="Number of evaluation items to draw"),
    seed: int = typer.Option(0, help="Sampling seed; the same seed reproduces the same set"),
    verbose: bool = VERBOSE,
) -> None:
    """Draw a seeded, source-stratified evaluation set from the agreed annotations."""
    configure(verbose)
    from mlime.data.export import export_eval_set
    from mlime.data.g2p import read_annotated

    export_eval_set(read_annotated(DataLayout(data_dir).annotations), out, size, seed)


@export_app.command("ngram-corpus")
def export_ngram_corpus_command(
    data_dir: Path = DATA_DIR,
    out: Path = typer.Option(Path("data/corpus.txt"), help="Plain text file to write"),
    exclude: list[Path] = EXCLUDE,
    verbose: bool = VERBOSE,
) -> None:
    """Dump prepared targets as one line each, minus the sentences held out for evaluation."""
    configure(verbose)
    from mlime.data.export import export_ngram_corpus, read_exclusions

    layout = DataLayout(data_dir)
    export_ngram_corpus(layout.samples, out, read_exclusions(exclude or ()))


def _requested(source: list[str] | None) -> tuple[str, ...]:
    """Validate the requested source names, defaulting to every source."""
    from mlime.data.corpus import SOURCES

    if not source:
        return tuple(SOURCES)
    unknown = set(source) - set(SOURCES)
    if unknown:
        raise typer.BadParameter(f"unknown source(s) {sorted(unknown)}; have {sorted(SOURCES)}")
    return tuple(source)


if __name__ == "__main__":
    app()
