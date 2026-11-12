"""The decision measurement: every function's latency on this machine, and where its ops run.

Timing does not depend on the weights, so a bundle of random weights at the real dimensions
answers it (#224 measured the ONNX design the same way). For each function and each compute-unit
set: the cold load (the load and the first predict, which is where a missing compile cache costs
its minutes), then ``WARMUP_CALLS`` warm-up predicts and ``TIMED_CALLS`` timed ones over one fixed
feed, reported as median and p95.

Core ML keys its compiled-bundle cache by the executable's name (``~/Library/Caches/<process>/
com.apple.e5rt.e5bundlecache``): two executables of one name evict each other's bundles, and a
name set at run time does not change the key.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np

from mlime_coreml import spec
from mlime_coreml.bundle import Bundle, Function
from mlime_coreml.convert import ConvertedFamily, compile_plan_variant, convert_family, sample_feed
from mlime_coreml.feeds import Array
from mlime_coreml.runtime import Runner, Timing
from mlime_coreml.signature import Signature, signature
from mlime_coreml.spec import Family
from mlime_coreml.verify import Placement, placement

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Measured:
    """One function's measurement under one compute-unit set."""

    units: str
    load_seconds: float
    first_predict_seconds: float
    timing: Timing

    def as_json(self) -> dict[str, object]:
        """The measurement as the bench report records it."""
        return {
            "units": self.units,
            "load_seconds": round(self.load_seconds, 3),
            "first_predict_seconds": round(self.first_predict_seconds, 3),
            "cold_seconds": round(self.load_seconds + self.first_predict_seconds, 3),
            "median_ms": round(self.timing.median_ms, 3),
            "p95_ms": round(self.timing.p95_ms, 3),
            "mean_ms": round(self.timing.mean_ms, 3),
            "calls": self.timing.calls,
        }


def bench_feed(bundle: Bundle, family: Family, function: Function) -> dict[str, Array]:
    """A full-bucket feed: the traced sample's structure, random context and cross activations."""
    feed = sample_feed(bundle, family, function)
    rng = np.random.default_rng(0)
    for name in ("context", "cross_keys", "cross_values"):
        if name in feed:
            feed[name] = rng.standard_normal(feed[name].shape).astype(np.float32)
    return feed


def measure(
    compiled: Path,
    function: str,
    sig: Signature,
    feed: dict[str, Array],
    units: ct.ComputeUnit,
    label: str,
) -> Measured:
    """Cold load, warm-up and timed predicts of *function* under *units*."""
    runner = Runner(compiled, function, sig, units)
    prepared = runner.prepare(feed)
    started = time.perf_counter()
    runner.predict_prepared(prepared)
    first = time.perf_counter() - started
    timing = runner.time(prepared, spec.WARMUP_CALLS, spec.TIMED_CALLS)
    return Measured(label, runner.load_seconds, first, timing)


def bench_family(
    bundle: Bundle, converted: ConvertedFamily, build_dir: Path
) -> dict[str, dict[str, object]]:
    """Every function of *converted*: placement, then CPU_AND_NE and CPU_ONLY timing."""
    family = converted.family
    sig = signature(bundle.manifest, family)
    results: dict[str, dict[str, object]] = {}
    for function in converted.functions:
        variant = compile_plan_variant(converted, function, build_dir)
        found: Placement = placement(variant, function.name)
        if variant != converted.compiled:
            shutil.rmtree(variant)
        feed = bench_feed(bundle, family, function)
        engine = measure(
            converted.compiled, function.name, sig, feed, ct.ComputeUnit.CPU_AND_NE, "CPU_AND_NE"
        )
        cpu = measure(
            converted.compiled, function.name, sig, feed, ct.ComputeUnit.CPU_ONLY, "CPU_ONLY"
        )
        results[function.name] = {
            "placement": found.as_json(),
            "convert_seconds": round(converted.convert_seconds[function.name], 2),
            "cpu_and_ne": engine.as_json(),
            "cpu_only": cpu.as_json(),
        }
        log.info(
            "%s: CPU_AND_NE median %.2f ms p95 %.2f ms cold %.1f s; CPU_ONLY median %.2f ms; "
            "ops %d/%d on the Neural Engine",
            function.name,
            engine.timing.median_ms,
            engine.timing.p95_ms,
            engine.load_seconds + engine.first_predict_seconds,
            cpu.timing.median_ms,
            found.engine,
            found.total,
        )
    return results


def bench(
    bundle: Bundle, work_dir: Path, families: tuple[Family, ...] = spec.FAMILIES
) -> dict[str, dict[str, dict[str, object]]]:
    """Convert each family, measure it, delete its models, write ``work_dir/bench.json``.

    One family at a time, so the working set is one family's packages and the disk holds one.
    """
    build_dir = work_dir / "build"
    models_dir = work_dir / "models"
    build_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, dict[str, object]]] = {}
    for family in families:
        converted = convert_family(bundle, family, build_dir, models_dir)
        report[family] = bench_family(bundle, converted, build_dir)
        shutil.rmtree(converted.package)
        shutil.rmtree(converted.directory)
    path = work_dir / "bench.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    log.info("wrote %s", path)
    return report
