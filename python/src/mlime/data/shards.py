"""Fixed-size parquet shard writing.

Every stage of the pipeline emits the same shape -- a long stream of small rows
that has to survive a Kaggle kernel restart -- so they share one writer rather
than each growing its own buffering logic.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import polars as pl

from mlime.logging import log


class ShardWriter:
    """Buffer rows and flush them as ``<prefix>-00000.parquet`` under *directory*.

    Used as a context manager so an interrupted run still leaves whole, readable
    shards behind instead of one truncated file.
    """

    def __init__(self, directory: Path, prefix: str, schema: pl.Schema, rows_per_shard: int):
        if rows_per_shard <= 0:
            raise ValueError(f"rows_per_shard must be positive, got {rows_per_shard}")
        self._directory = directory
        self._prefix = prefix
        self._schema = schema
        self._rows_per_shard = rows_per_shard
        self._buffer: list[dict[str, object]] = []
        self._shards = 0
        self.rows_written = 0
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, object]) -> None:
        """Queue one row, flushing a shard once the buffer is full."""
        self._buffer.append(row)
        if len(self._buffer) >= self._rows_per_shard:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows out as one shard."""
        if not self._buffer:
            return
        path = self._directory / f"{self._prefix}-{self._shards:05d}.parquet"
        pl.DataFrame(self._buffer, schema=self._schema).write_parquet(path)
        self.rows_written += len(self._buffer)
        self._shards += 1
        log.debug("shard written", path=str(path), rows=len(self._buffer))
        self._buffer.clear()

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc is None:
            self.flush()


def shard_paths(directory: Path, prefix: str) -> list[Path]:
    """Every ``<prefix>-*.parquet`` shard in *directory*, in write order.

    Empty when the stage wrote none, which is a legitimate outcome for the
    optional outputs -- an annotation run that nothing refused writes no refusal
    shard -- so the caller decides whether emptiness is an error.
    """
    return sorted(directory.glob(f"{prefix}-*.parquet"))


def read_shards(directory: Path, prefix: str) -> pl.DataFrame:
    """Read every ``<prefix>-*.parquet`` shard in *directory* as one frame."""
    paths = shard_paths(directory, prefix)
    if not paths:
        raise FileNotFoundError(f"no {prefix}-*.parquet shards under {directory}")
    return pl.concat([pl.read_parquet(path) for path in paths])
