"""The Rust segmenter, held open over a pipe for the ranking loss (#116).

The ranking loss scores a sentence against the other readings of its
keystrokes, and those readings must be the ones the evaluation lattice offers.
The lattice is cut by ``ime-pinyin``'s k-best segmenter; reimplementing it here
would be a second segmenter that drifts from the first. So the training run
starts the ``ime-segment`` binary once and asks it one keystroke string at a
time: one line out, one line back.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import IO

#: What the ranking collator needs of a segmenter: keystrokes in, readings out,
#: cheapest first, each reading the typed spans.
Segmentations = Callable[[str], list[tuple[str, ...]]]


class Segmenter:
    """``ime-segment --max-paths N`` as a callable.

    The child process lives as long as this object's owner -- the training
    process -- and exits on its own when the pipe closes. A child that has
    exited or answered with nothing is an error at the call that found it.
    """

    def __init__(self, binary: Path, max_paths: int):
        self._process = subprocess.Popen(
            [str(binary), "--max-paths", str(max_paths)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("ime-segment started without its pipes")
        self._stdin: IO[str] = self._process.stdin
        self._stdout: IO[str] = self._process.stdout

    def __call__(self, keys: str) -> list[tuple[str, ...]]:
        """The readings of *keys*, cheapest first."""
        self._stdin.write(keys + "\n")
        self._stdin.flush()
        line = self._stdout.readline()
        if not line:
            raise RuntimeError(
                f"ime-segment exited ({self._process.wait()}) on {keys!r}; its stderr says why"
            )
        readings: list[list[str]] = json.loads(line)
        return [tuple(reading) for reading in readings]
