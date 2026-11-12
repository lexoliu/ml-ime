"""Verification: the float32 port, Core ML against the port, padding, buckets and placement.

Every check is recorded in ``verification.json`` and any miss fails the run, after the report is
written. In order:

1. the float32 port against ``reference.npz`` (what the model's own framework computed);
2. the port at each bucket's width against its exact-length run (padding is invisible);
3. per function: Core ML (``CPU_AND_NE``) against the reference and against the exact-length
   port, within the fp16 tolerance, at both edges of the bucket;
4. buckets agree: the same real rows through the smallest and the largest function of a family;
5. per function, on its own compiled copy: where the plan placed each op, and no matrix product,
   normalisation or softmax on the CPU.

Notes on the harness. ``MLComputePlan`` reports compute-device usage only for the compiled model's
default function, so each function is counted on a copy compiled with it as default
(:func:`mlime_coreml.convert.compile_plan_variant`). Every array handed to ``predict`` is a
numpy-owned copy (:mod:`mlime_coreml.runtime`).
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
from coremltools.models.compute_plan import MLComputePlan

from mlime_coreml import spec
from mlime_coreml.bundle import Bundle, Function
from mlime_coreml.convert import ConvertedFamily, compile_plan_variant
from mlime_coreml.feeds import Array, ContextFeed, CrossFeed
from mlime_coreml.modules import build_module
from mlime_coreml.port import (
    Check,
    Edge,
    Float,
    check_padding,
    check_port,
    context_of,
    cross_of,
    edges,
    make_case,
    real_parts,
    run_case,
    tree_calls,
)
from mlime_coreml.runtime import Runner
from mlime_coreml.signature import signature
from mlime_coreml.spec import Family

log = logging.getLogger(__name__)

# --- placement --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """Where a function's ops run, by device, and the main ops that run off the Neural Engine."""

    function: str
    counts: dict[str, int]
    main_off_engine: dict[str, int]

    @property
    def total(self) -> int:
        """Non-const ops in the function."""
        return sum(self.counts.values())

    @property
    def engine(self) -> int:
        """Ops the plan prefers the Neural Engine for."""
        return self.counts.get("NeuralEngine", 0)

    def failures(self) -> list[str]:
        """What is wrong with the placement; empty when it is as the design requires."""
        found = []
        if self.main_off_engine:
            found.append(f"{self.function}: main ops off the Neural Engine: {self.main_off_engine}")
        if self.total and self.engine < spec.MIN_ANE_SHARE * self.total:
            found.append(
                f"{self.function}: {self.engine}/{self.total} ops on the Neural Engine, "
                f"under {spec.MIN_ANE_SHARE:.0%}"
            )
        return found

    def as_json(self) -> dict[str, object]:
        """The placement as the report records it."""
        return {
            "ops_by_device": self.counts,
            "total_ops": self.total,
            "neural_engine_ops": self.engine,
            "main_ops_off_neural_engine": self.main_off_engine,
        }


def _device_name(usage: Any) -> str:
    device = type(usage.preferred_compute_device).__name__
    return device.replace("ML", "").replace("ComputeDevice", "")


def placement(compiled: Path, function: str) -> Placement:
    """Count *function*'s ops by the device the plan prefers, under ``CPU_AND_NE``.

    *compiled* must have *function* as its default (see the module notes). A non-const op the
    plan reports no usage for is an error, not a skipped op.
    """
    plan = MLComputePlan.load_from_path(path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    operations = plan.model_structure.program.functions[function].block.operations
    counts: dict[str, int] = {}
    off: dict[str, int] = {}
    for operation in operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            if operation.operator_name == "const":
                continue
            raise RuntimeError(
                f"no compute device usage for non-const op {operation.operator_name}"
            )
        device = _device_name(usage)
        counts[device] = counts.get(device, 0) + 1
        # The plan names an op with its opset ("ios18.linear"); the allow-list does not.
        name = operation.operator_name.rpartition(".")[2]
        if name in spec.MAIN_OPS and device != "NeuralEngine":
            off[name] = off.get(name, 0) + 1
    del plan
    gc.collect()
    return Placement(function, counts, off)


# --- comparisons ------------------------------------------------------------------------------


def fp16_check(name: str, part: str, got: Float, want: Float) -> Check:
    """*got* (Core ML, fp16 widened) against *want* (float32) under the part's tolerance.

    The recorded gap is the largest excess of ``|got - want|`` over ``rtol * |want|``, to be
    within ``atol``; the raw largest difference is recorded beside it.
    """
    if got.shape != want.shape:
        raise ValueError(f"{name}: shapes differ: {got.shape} against {want.shape}")
    atol, rtol = spec.FP16_TOLERANCE[part]
    difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
    excess = difference - rtol * np.abs(want.astype(np.float64))
    return Check(
        name,
        float(np.max(excess)) if excess.size else 0.0,
        atol,
        raw=float(np.max(difference)) if difference.size else 0.0,
    )


def _pad_row(row: Array, width: int, fill: int | float) -> Array:
    padded = np.full(width, fill, dtype=row.dtype)
    padded[: len(row)] = row
    return padded


def reference_checks(
    bundle: Bundle,
    family: Family,
    function: Function,
    runner: Runner,
    reference: Mapping[str, Array],
) -> list[Check]:
    """*function* over the reference's probes that land in it, against what the reference holds."""
    manifest = bundle.manifest
    checks: list[Check] = []
    widths = [f.length for f in manifest.family(family).functions]
    for probe in manifest.reference:
        name = probe.name
        if family == "context":
            mask = np.asarray(reference[f"{name}.context_mask"], dtype=np.float32)
            real = int(mask.sum())
            if not real:
                continue
            [rows] = runner.outputs(
                {
                    "context_ids": np.asarray(reference[f"{name}.context_ids"])[None],
                    "context_mask": mask[None],
                }
            )
            want = np.asarray(reference[f"{name}.context"], dtype=np.float32)
            checks.append(
                fp16_check(f"reference.{name}.context", "context", rows[:real], want[:real])
            )
        elif family == "keys":
            ids = np.asarray(reference[f"{name}.input_ids"])
            if min(width for width in widths if width >= len(ids)) != function.length:
                continue
            width = function.length
            feed: dict[str, Array] = {
                "input_ids": _pad_row(ids, width, manifest.specials.pad)[None],
                "key_letters": _pad_row(
                    np.asarray(reference[f"{name}.key_letters"]), width, manifest.letter_pad
                )[None],
                "attention_mask": _pad_row(
                    np.asarray(reference[f"{name}.attention_mask"], dtype=np.float32), width, 0.0
                )[None],
            }
            context = context_of(reference, name)
            feed["context"], feed["context_mask"] = context.rows[None], context.mask[None]
            out = runner.outputs(feed)
            count = len(ids) - spec.KEY_SENTINELS
            got = real_parts("keys", count, out)
            wanted = {
                "ctc_logits": np.asarray(reference[f"{name}.ctc_logits"], dtype=np.float32),
                "cross_keys": np.asarray(reference[f"{name}.cross_keys"], dtype=np.float32),
                "cross_values": np.asarray(reference[f"{name}.cross_values"], dtype=np.float32),
            }
            for part, rows in got.items():
                want = wanted[part][:, :, :count] if part.startswith("cross") else wanted[part]
                checks.append(fp16_check(f"reference.{name}.{part}", part, rows, want))
            # Padded slots beyond the real keys are exactly zero.
            for part in ("cross_keys", "cross_values"):
                tail = np.abs(out[1 if part == "cross_keys" else 2][:, :, count:])
                checks.append(
                    Check(
                        f"reference.{name}.{part}.padded_slots_zero",
                        float(tail.max()) if tail.size else 0.0,
                        0.0,
                    )
                )
        else:
            cross = cross_of(reference, name)
            for call in tree_calls(reference, name):
                ids = np.asarray(reference[f"{call}.reader_ids"])
                if len(ids) != function.length:
                    continue
                [logits] = runner.outputs(
                    {
                        "reader_ids": ids[None],
                        "positions": np.asarray(reference[f"{call}.positions"])[None],
                        "tree_bias": np.asarray(reference[f"{call}.tree_bias"])[None, None],
                        "cross_keys": cross.keys,
                        "cross_values": cross.values,
                        "cross_bias": cross.bias,
                    }
                )
                want = np.asarray(reference[f"{call}.logits"], dtype=np.float32)
                checks.append(fp16_check(f"reference.{call}.logits", "logits", logits, want))
    return checks


def feed_of(
    family: Family,
    feed: Mapping[str, Array],
    case_context: ContextFeed | None,
    cross: CrossFeed | None,
) -> dict[str, Array]:
    """*feed* with the function's remaining inputs (the context rows, the cross keys) added."""
    full = dict(feed)
    if family == "keys":
        assert case_context is not None
        full["context"], full["context_mask"] = case_context.rows[None], case_context.mask[None]
    if family == "tree":
        assert cross is not None
        full["cross_keys"], full["cross_values"], full["cross_bias"] = (
            cross.keys,
            cross.values,
            cross.bias,
        )
    return full


# --- the report -------------------------------------------------------------------------------


@dataclass
class FunctionReport:
    """One function's placement and Core ML checks."""

    function: str
    convert_seconds: float
    placement: Placement | None
    checks: list[Check]

    def failures(self) -> list[str]:
        """Every miss in the function."""
        found = [] if self.placement is None else self.placement.failures()
        found += [
            f"{check.name}: gap {check.max_abs:.4g} > {check.tolerance:.4g}"
            for check in self.checks
            if not check.ok
        ]
        return found

    def as_json(self) -> dict[str, object]:
        """The function as the report records it."""
        return {
            "convert_seconds": round(self.convert_seconds, 2),
            "placement": None if self.placement is None else self.placement.as_json(),
            "checks": [check.as_json() for check in self.checks],
        }


def verify_family(
    bundle: Bundle,
    converted: ConvertedFamily,
    build_dir: Path,
    reference: Mapping[str, Array],
    require_neural_engine: bool,
) -> tuple[list[FunctionReport], list[Check]]:
    """Verify every function of *converted*; the reports and the cross-bucket agreement checks."""
    family = converted.family
    sig = signature(bundle.manifest, family)
    reports: list[FunctionReport] = []
    agreed: dict[str, dict[str, Float]] = {}
    common = min(function.capacity for function in converted.functions)
    for function in converted.functions:
        found = None
        if require_neural_engine:
            variant = compile_plan_variant(converted, function, build_dir)
            found = placement(variant, function.name)
            if variant != converted.compiled:
                shutil.rmtree(variant)
        runner = Runner(converted.compiled, function.name, sig, ct.ComputeUnit.CPU_AND_NE)
        checks = reference_checks(bundle, family, function, runner, reference)
        wanted = [
            edge
            for edge in edges(bundle)
            if (edge.family, edge.length) == (family, function.length)
        ]
        if family != "context" and Edge(family, function.length, common) not in wanted:
            wanted.append(Edge(family, function.length, common))
        for edge in wanted:
            case = make_case(bundle, edge)
            exact = build_module(bundle, family, case.exact_length)
            want = run_case(exact, family, case, case.exact)
            del exact
            out = runner.outputs(feed_of(family, case.padded, case.context, case.cross))
            got = real_parts(family, edge.real, out)
            for part, rows in got.items():
                checks.append(
                    fp16_check(
                        f"edge.{family}_{function.length}.real{edge.real}.{part}",
                        part,
                        rows,
                        want[part],
                    )
                )
            if family != "context" and edge.real == common:
                agreed[function.name] = got
        del runner
        gc.collect()
        reports.append(
            FunctionReport(function.name, converted.convert_seconds[function.name], found, checks)
        )
        log.info(
            "verified %s: %s Neural Engine ops, %d failing checks",
            function.name,
            "unchecked" if found is None else found.engine,
            sum(not check.ok for check in checks),
        )
    return reports, _agreement(agreed, converted)


def _agreement(agreed: dict[str, dict[str, Float]], converted: ConvertedFamily) -> list[Check]:
    """The same real rows through the family's smallest and largest function, compared."""
    if len(agreed) < 2:
        return []
    first, last = converted.functions[0].name, converted.functions[-1].name
    return [
        fp16_check(
            f"buckets.{first}_vs_{last}.{part}", part, agreed[first][part], agreed[last][part]
        )
        for part in agreed[first]
    ]


def verify(
    bundle: Bundle,
    converted: Mapping[Family, ConvertedFamily],
    build_dir: Path,
    work_dir: Path,
    require_neural_engine: bool = True,
) -> dict[str, object]:
    """Run every check, write ``work_dir/verification.json`` and fail if any missed.

    *require_neural_engine* off (a host without one, such as a hosted runner) skips placement and
    records that it did; every numeric check still runs, on the CPU.
    """
    reference = bundle.reference()
    port = check_port(bundle)
    padding = check_padding(bundle)
    families: dict[str, object] = {}
    failures = [
        f"{check.name}: gap {check.max_abs:.4g} > {check.tolerance:.4g}"
        for check in port + padding
        if not check.ok
    ]
    agreement: list[Check] = []
    for family in spec.FAMILIES:
        reports, joined = verify_family(
            bundle, converted[family], build_dir, reference, require_neural_engine
        )
        agreement += joined
        families[family] = {report.function: report.as_json() for report in reports}
        failures += [miss for report in reports for miss in report.failures()]
    failures += [
        f"{check.name}: gap {check.max_abs:.4g} > {check.tolerance:.4g}"
        for check in agreement
        if not check.ok
    ]
    report: dict[str, object] = {
        "step": bundle.manifest.step,
        "compute_units": "CPU_AND_NE",
        "placement_checked": require_neural_engine,
        "placement_note": "MLComputePlan reports usage only for the compiled model's default "
        "function; each function was counted on a copy compiled with it as default.",
        "tolerances": {
            "port_atol": spec.PORT_ATOL,
            "padding_atol_fp32": spec.PADDING_ATOL_FP32,
            "fp16 (atol, rtol)": spec.FP16_TOLERANCE,
            "min_ane_share": spec.MIN_ANE_SHARE,
        },
        "port_vs_reference": [check.as_json() for check in port],
        "port_padding": [check.as_json() for check in padding],
        "functions": families,
        "bucket_agreement": [check.as_json() for check in agreement],
        "passed": not failures,
        "failures": failures,
    }
    path = work_dir / "verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    log.info("wrote %s", path)
    if failures:
        raise RuntimeError("verification failed: " + "; ".join(failures))
    return report
