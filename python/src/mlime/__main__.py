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
lexicon_app = typer.Typer(
    help="Manage external word-pinyin lexicons (Sogou scel, etc.)", no_args_is_help=True
)
train_app = typer.Typer(help="Route A training and the labels it needs", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(g2p_app, name="g2p")
app.add_typer(export_app, name="export")
app.add_typer(lexicon_app, name="lexicon")
app.add_typer(train_app, name="train")

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


@lexicon_app.command("fetch")
def lexicon_fetch(
    dict_slug: list[str] = typer.Option(
        None, "--dict", help="Dict slug to fetch; repeatable, defaults to all"
    ),
    data_dir: Path = DATA_DIR,
    verbose: bool = VERBOSE,
) -> None:
    """Download Sogou scel dictionaries and write word-pinyin lexicon shards."""
    configure(verbose)
    from mlime.data.scel import DICTS, fetch_lexicon

    slugs = tuple(dict_slug) if dict_slug else tuple(DICTS)
    unknown = set(slugs) - set(DICTS)
    if unknown:
        raise typer.BadParameter(f"unknown dict slug(s) {sorted(unknown)}; have {sorted(DICTS)}")

    layout = DataLayout(data_dir)
    counts = fetch_lexicon(slugs, layout.lexicon, layout.scel_cache)
    for slug, count in counts.items():
        typer.echo(f"{slug}: {count} entries")


@lexicon_app.command("list")
def lexicon_list() -> None:
    """Show the available Sogou dictionaries and their quality metadata."""
    from mlime.data.scel import DICTS

    for slug, d in DICTS.items():
        typer.echo(f"  {slug}  (id={d.id})  {d.name}")
        typer.echo(f"    quality: {d.quality}")
        typer.echo(f"    url: {d.detail_url}")
        typer.echo()


@train_app.command("gen-spans")
def train_gen_spans(
    out: Path = typer.Option(None, help="Where to write the table; defaults to the package's own"),
    syllables: Path = typer.Option(
        None, help="ime-pinyin's syllables.txt; found upwards when omitted"
    ),
    verbose: bool = VERBOSE,
) -> None:
    """Regenerate the typed-span table the fill tower is indexed by."""
    configure(verbose)
    from mlime.train.spans import TYPED_SPANS_PATH, build

    spans = build(out or TYPED_SPANS_PATH, syllables)
    typer.echo(f"{len(spans)} typed spans")


@train_app.command("labels")
def train_labels(
    data_dir: Path = DATA_DIR,
    out: Path = typer.Option(None, help="Where the label shards go; defaults to <data-dir>/labels"),
    shards: int = typer.Option(None, help="Stop after this many sample shards"),
    shard: list[str] = typer.Option(
        None, "--shard", help="Sample shard to label; repeatable, overrides --shards"
    ),
    batch_size: int = typer.Option(512, help="Sentences handed to g2pW at once"),
    onnx_batch_size: int = typer.Option(256, help="Query positions per ONNX call"),
    num_workers: int = typer.Option(
        None, help="DataLoader workers g2pW prepares batches with; platform default when omitted"
    ),
    g2pw_model: Path = typer.Option(None, help="Directory holding the g2pW model"),
    cuda: bool = typer.Option(False, help="Require the ONNX session to run on CUDA"),
    verbose: bool = VERBOSE,
) -> None:
    """Label the prepared samples with g2pW, one label shard per sample shard."""
    configure(verbose)
    from mlime.data.g2pw_annotator import DEFAULT_MODEL_DIR, G2pwAnnotator
    from mlime.train.labels import generate, load_cuda_annotator, select_shards

    layout = DataLayout(data_dir)
    model_dir = g2pw_model or DEFAULT_MODEL_DIR
    annotator = (
        load_cuda_annotator(model_dir, onnx_batch_size, num_workers)
        if cuda
        else G2pwAnnotator(model_dir, batch_size=onnx_batch_size, num_workers=num_workers)
    )
    labels = out or data_dir / "labels"
    counts = asyncio.run(
        generate(
            select_shards(layout.samples, shard, shards),
            labels,
            annotator,
            sentences_per_batch=batch_size,
            metrics=labels / "throughput.jsonl",
        )
    )
    typer.echo(
        f"{counts.labelled} labelled, {counts.refused} refused, "
        f"{counts.sentences_per_second:.1f} sentences/s"
    )


@train_app.command("route-a")
def train_route_a(
    data_dir: Path = DATA_DIR,
    labels: Path = typer.Option(None, help="Label shards; defaults to <data-dir>/labels"),
    out: Path = typer.Option(Path("runs/route-a"), help="Where checkpoints and metrics go"),
    char_table: Path = typer.Option(None, help="ime-pinyin's char_pinyin.tsv"),
    train_shard: list[str] = typer.Option(
        None,
        "--train-shard",
        help="Shard to train on; repeatable. Omitted, every shard that is not held out",
    ),
    held_out_shard: list[str] = typer.Option(
        None, "--held-out-shard", help="Shard to score but never train on; repeatable"
    ),
    max_steps: int = typer.Option(1000, help="Optimiser steps to run"),
    token_budget: int = typer.Option(8192, help="Padded positions per step, both towers together"),
    base_lr: float = typer.Option(3e-5, help="Learning rate for the pretrained weights"),
    new_lr: float = typer.Option(1e-4, help="Learning rate for the tables route A adds"),
    seed: int = typer.Option(0, help="Augmentation and initialisation seed"),
    fp16: bool = typer.Option(True, help="Train in fp16 with loss scaling"),
    checkpoint_every: int = typer.Option(500, help="Steps between checkpoints"),
    keep_checkpoints: int = typer.Option(2, help="Numbered checkpoints to keep on disk"),
    wall_budget_seconds: float = typer.Option(
        None,
        help="Pause and checkpoint after this many seconds, for a session that gets killed",
    ),
    resume: Path = typer.Option(
        None,
        help="Continue the run that wrote this checkpoint; every other option must match it",
    ),
    max_held_out: int = typer.Option(4096, help="Held-out examples to score"),
    verbose: bool = VERBOSE,
) -> None:
    """Train route A on the given shards and score the held-out ones, context on and off."""
    configure(verbose)
    from mlime.data.corpus import default_char_table
    from mlime.train.loop import TrainingConfig
    from mlime.train.run import RunPaths, Slices, describe_device, route_a

    table = char_table or default_char_table()
    if table is None:
        raise typer.BadParameter("no char_pinyin.tsv found above the working directory")
    layout = DataLayout(data_dir)
    typer.echo(f"device: {describe_device()}")
    slices = (
        Slices(
            train=tuple(train_shard),
            held_out=tuple(held_out_shard or ()),
            max_held_out_examples=max_held_out,
        )
        if train_shard
        else Slices.all_but(layout.samples, held_out_shard or (), max_held_out)
    )
    result = route_a(
        RunPaths(
            samples=layout.samples,
            labels=labels or data_dir / "labels",
            char_table=table,
            out=out,
        ),
        slices,
        TrainingConfig(
            max_steps=max_steps,
            base_lr=base_lr,
            new_lr=new_lr,
            token_budget=token_budget,
            seed=seed,
            fp16=fp16,
            checkpoint_every=checkpoint_every,
            keep_checkpoints=keep_checkpoints,
            wall_budget_seconds=wall_budget_seconds,
        ),
        resume=resume,
    )
    typer.echo(f"loss {result.first_loss:.4f} -> {result.last_loss:.4f} over {result.steps} steps")
    if not result.finished:
        typer.echo(
            f"paused at step {result.step} of {max_steps}; the next kernel continues it with "
            f"--resume {out / 'checkpoint-paused.pt'}"
        )
        return
    with_context, without_context = result.scored
    typer.echo(
        f"held-out character accuracy: context on {with_context.rate:.4f}, "
        f"off {without_context.rate:.4f} "
        f"({with_context.scored} characters)"
    )


@train_app.command("emittable")
def train_emittable(
    out: Path = typer.Option(..., help="Where the one-character-per-line set goes"),
    char_table: Path = typer.Option(None, help="ime-pinyin's char_pinyin.tsv"),
    base_model: str = typer.Option("hfl/chinese-macbert-base", help="The base checkpoint"),
    verbose: bool = VERBOSE,
) -> None:
    """Write the characters a route A model over *base_model* can emit."""
    configure(verbose)
    from mlime.data.corpus import default_char_table
    from mlime.train.lexicon import write_emittable
    from mlime.train.run import build_lexicon_for, load_tokenizer
    from mlime.train.spans import SpanVocab

    table = char_table or default_char_table()
    if table is None:
        raise typer.BadParameter("no char_pinyin.tsv found above the working directory")
    tokenizer = load_tokenizer(base_model)
    written = write_emittable(out, build_lexicon_for(table, tokenizer, SpanVocab.load()))
    typer.echo(f"{written} emittable characters")


@train_app.command("emit")
def train_emit(
    checkpoint: Path = typer.Option(..., help="A checkpoint written by `train route-a`"),
    lattice: Path = typer.Option(..., help="A lattice written by `ime-cli emit-lattice`"),
    out: Path = typer.Option(..., help="Where the JSON Lines scores go"),
    char_table: Path = typer.Option(None, help="ime-pinyin's char_pinyin.tsv"),
    context: bool = typer.Option(True, help="Let the model read each record's context"),
    token_budget: int = typer.Option(
        8192, help="Padded positions per forward, both towers together"
    ),
    records_per_chunk: int = typer.Option(256, help="Records held in memory before writing"),
    verbose: bool = VERBOSE,
) -> None:
    """Score a decoder's lattice with a trained route A model."""
    configure(verbose)
    from mlime.data.corpus import default_char_table
    from mlime.train.emit import emit
    from mlime.train.run import build_lexicon_for, load_tokenizer
    from mlime.train.spans import SpanVocab

    table = char_table or default_char_table()
    if table is None:
        raise typer.BadParameter("no char_pinyin.tsv found above the working directory")
    spans = SpanVocab.load()
    tokenizer = load_tokenizer(_base_model(checkpoint))
    written = emit(
        checkpoint=checkpoint,
        lattice=lattice,
        out=out,
        tokenizer=tokenizer,
        lexicon=build_lexicon_for(table, tokenizer, spans),
        spans=spans,
        with_context=context,
        token_budget=token_budget,
        records_per_chunk=records_per_chunk,
    )
    typer.echo(f"scores written to {written}")


def _base_model(checkpoint: Path) -> str:
    """The base checkpoint a trained model was built on, read out of its own file."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return str(state["route_a"]["base_model"])


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
