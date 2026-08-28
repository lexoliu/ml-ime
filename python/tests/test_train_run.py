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

from mlime.train.run import Slices


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
