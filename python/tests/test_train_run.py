"""How a run decides what it trains on.

A full epoch names four hundred shards. Naming them one by one is how a run
quietly trains on three hundred and ninety-nine of them, so the run names what it
*withholds* and derives the rest -- and the derivation has to be checked, because
"everything except" silently including a held-out shard is exactly the leak the
held-out set exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mlime.train.lexicon import Lexicon
from mlime.train.run import RunPaths, Slices, held_out_examples
from mlime.train.samples import SampleBuilder
from mlime.train.spans import SpanVocab


@pytest.fixture(name="samples")
def samples_fixture(tmp_path: Path) -> Path:
    """A samples directory holding four empty shards from two sources."""
    directory = tmp_path / "samples"
    directory.mkdir()
    for name in ("dialogue-00000", "dialogue-00001", "wiki-00000", "wiki-00001"):
        pl.DataFrame(
            {"id": ["a"], "source": ["s"], "text": ["中"], "context": [None]}
        ).write_parquet(directory / f"{name}.parquet")
    return directory


def write_pair(root: Path, shard: str, source: str, rows: int) -> None:
    """A sample shard of *rows* one-character sentences, and the labels for it."""
    ids = [f"{source}-{index}" for index in range(rows)]
    (root / "samples").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "id": ids,
            "source": [source] * rows,
            "text": ["中"] * rows,
            "context": [None] * rows,
        }
    ).write_parquet(root / "samples" / shard)
    pl.DataFrame({"id": ids, "syllables": [["zhong"]] * rows}).write_parquet(
        root / "labels" / shard
    )


def test_the_held_out_slice_is_drawn_from_every_shard(
    tmp_path: Path, lexicon: Lexicon, spans: SpanVocab
) -> None:
    # The first shard alone could fill the whole quota, which is exactly the
    # situation that made the in-training evaluation one source's number.
    write_pair(tmp_path, "bilibili-00001.parquet", "bilibili", 400)
    write_pair(tmp_path, "wiki-00001.parquet", "wiki", 400)
    examples = held_out_examples(
        RunPaths(
            samples=tmp_path / "samples",
            labels=tmp_path / "labels",
            char_table=tmp_path / "unused.tsv",
            out=tmp_path / "out",
        ),
        SampleBuilder(lexicon, spans),
        Slices(
            train=("unused.parquet",),
            held_out=("bilibili-00001.parquet", "wiki-00001.parquet"),
            max_held_out_examples=100,
        ),
    )
    sources = {example.id.split("-")[0] for example in examples}
    assert sources == {"bilibili", "wiki"}
    assert len(examples) == 100


def test_everything_not_withheld_is_trained_on(samples: Path) -> None:
    slices = Slices.all_but(samples, ["wiki-00001.parquet"])
    assert slices.held_out == ("wiki-00001.parquet",)
    assert slices.train == (
        "dialogue-00000.parquet",
        "dialogue-00001.parquet",
        "wiki-00000.parquet",
    )


def test_a_held_out_shard_that_does_not_exist_is_refused(samples: Path) -> None:
    with pytest.raises(FileNotFoundError, match="news-00000"):
        Slices.all_but(samples, ["news-00000.parquet"])


def test_a_directory_with_no_shards_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no sample shards"):
        Slices.all_but(empty, [])


def test_a_shard_cannot_be_trained_on_and_held_out(samples: Path) -> None:
    with pytest.raises(ValueError, match="both trained on and held out"):
        Slices(train=("wiki-00000.parquet",), held_out=("wiki-00000.parquet",))
