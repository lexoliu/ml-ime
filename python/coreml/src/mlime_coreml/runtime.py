"""Run a compiled function from Python, the way the host will: fixed feeds in, fp16 arrays out.

Core ML releases the input feature values on its own dispatch queue after ``predict`` returns, and
coremltools 9.0 drops the numpy reference without holding the GIL. A numpy *view* of memory some
other library owns (a torch tensor's) is then freed off the GIL and corrupts the heap. So every
feed is a numpy-owned copy, kept alive for the life of the runner, and every output is copied out.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import coremltools as ct
import numpy as np
import numpy.typing as npt

from mlime_coreml.feeds import Array
from mlime_coreml.signature import Signature

Float = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Timing:
    """Predict latency over repeated calls on one prepared feed, milliseconds."""

    median_ms: float
    p95_ms: float
    mean_ms: float
    calls: int


class Runner:
    """One function of a compiled model, loaded for *units*."""

    def __init__(self, compiled: Path, function: str, sig: Signature, units: ct.ComputeUnit):
        self.function = function
        self._sig = sig
        self._alive: list[dict[str, Array]] = []
        started = time.perf_counter()
        self._model = ct.models.CompiledMLModel(
            str(compiled), compute_units=units, function_name=function
        )
        self.load_seconds = time.perf_counter() - started

    def prepare(self, feed: Mapping[str, Array]) -> dict[str, Array]:
        """*feed* as numpy-owned arrays of the function's dtypes, kept alive with the runner."""
        owned: dict[str, Array] = {
            tensor.name: np.array(feed[tensor.name], dtype=tensor.numpy_dtype)
            for tensor in self._sig.inputs
        }
        self._alive.append(owned)
        return owned

    def predict_prepared(self, prepared: dict[str, Array]) -> dict[str, Float]:
        """Predict over a :meth:`prepare`d feed; the outputs are copies, in float32."""
        out = self._model.predict(prepared)
        return {name: np.array(value, dtype=np.float32) for name, value in out.items()}

    def predict(self, feed: Mapping[str, Array]) -> dict[str, Float]:
        """Predict over *feed*."""
        return self.predict_prepared(self.prepare(feed))

    def outputs(self, feed: Mapping[str, Array]) -> list[Float]:
        """Predict over *feed*; outputs in the signature's order, a batch dimension dropped."""
        out = self.predict(feed)
        return [
            out[tensor.name][0] if tensor.batched else out[tensor.name]
            for tensor in self._sig.outputs
        ]

    def time(self, prepared: dict[str, Array], warmup: int, calls: int) -> Timing:
        """Latency of *calls* predicts after *warmup*, on *prepared* (one feed, reused)."""
        for _ in range(warmup):
            self._model.predict(prepared)
        samples = []
        for _ in range(calls):
            started = time.perf_counter()
            self._model.predict(prepared)
            samples.append((time.perf_counter() - started) * 1e3)
        samples.sort()
        return Timing(
            median_ms=median(samples),
            p95_ms=samples[min(len(samples) - 1, int(np.ceil(0.95 * len(samples))) - 1)],
            mean_ms=float(np.mean(samples)),
            calls=len(samples),
        )
