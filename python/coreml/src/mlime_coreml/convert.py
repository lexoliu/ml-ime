"""Trace each function at its fixed shape, convert to fp16 ML programs, merge a family into one
multifunction package (weights stored once) and compile it to ``.mlmodelc``.

The recipe is repo-search's: one traced graph per bucket, ``compute_precision=FLOAT16``, the
macOS 15 target, ``CPU_AND_NE``, ``save_multifunction`` per family, and a spec re-serialised with
deterministic map ordering so two runs compile to the same bytes.
"""

from __future__ import annotations

import gc
import logging
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import coremltools as ct
import numpy as np
import torch
from coremltools.proto import Model_pb2

from mlime_coreml import spec
from mlime_coreml.bundle import Bundle, Function
from mlime_coreml.feeds import (
    Array,
    context_row,
    key_row,
    random_context_ids,
    random_letters,
    random_tree,
    tree_row,
)
from mlime_coreml.modules import build_module
from mlime_coreml.package import write_directory_manifest
from mlime_coreml.signature import Signature, Tensor, functions, signature
from mlime_coreml.spec import Family

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvertedFamily:
    """A converted family: its compiled model, its package, and what each function cost."""

    family: Family
    directory: Path
    package: Path
    compiled: Path
    functions: tuple[Function, ...]
    convert_seconds: dict[str, float]


def _rewrite_spec_deterministic(package: Path, default_function: str | None = None) -> None:
    """Re-serialise the package spec with deterministic map ordering.

    protobuf randomises map order per process, and the compiled model embeds
    ``base64(sha256(model.mlmodel))``, so a non-deterministic spec makes the packaged archive
    differ between runs. *default_function* also sets the spec's default function.
    """
    path = package / "Data/com.apple.CoreML/model.mlmodel"
    model = Model_pb2.Model()
    model.ParseFromString(path.read_bytes())
    if default_function is not None:
        model.description.defaultFunctionName = default_function
    path.write_bytes(model.SerializeToString(deterministic=True))


def sample_feed(bundle: Bundle, family: Family, function: Function) -> dict[str, Array]:
    """Valid inputs for tracing *function*: values do not shape the graph, their dtypes do."""
    manifest = bundle.manifest
    rng = random.Random(0)
    sig = signature(manifest, family)
    if family == "context":
        ids = random_context_ids(manifest, min(8, function.length), rng)
        return context_row(manifest, ids, function.length)
    if family == "keys":
        feed = key_row(
            manifest, random_letters(manifest, min(5, function.capacity), rng), function.length
        )
        feed["context"] = np.zeros(sig.inputs[3].resolve(function.length), dtype=np.float32)
        feed["context_mask"] = np.ones(sig.inputs[4].resolve(function.length), dtype=np.float32)
        return feed
    feed = tree_row(manifest, random_tree(manifest, min(8, function.length), rng), function.length)
    cross = sig.inputs[3].resolve(function.length)
    feed["cross_keys"] = np.zeros(cross, dtype=np.float32)
    feed["cross_values"] = np.zeros(cross, dtype=np.float32)
    feed["cross_bias"] = np.zeros(sig.inputs[5].resolve(function.length), dtype=np.float32)
    return feed


def trace_inputs(sig: Signature, feed: dict[str, Array]) -> tuple[torch.Tensor, ...]:
    """The tracing example: *feed* as torch tensors, ids int32 and everything else float32."""
    return tuple(
        torch.from_numpy(
            np.array(feed[tensor.name], dtype=np.int32 if tensor.dtype == "int32" else np.float32)
        )
        for tensor in sig.inputs
    )


def _feature(tensor: Tensor, length: int) -> ct.TensorType:
    return ct.TensorType(name=tensor.name, shape=tensor.resolve(length), dtype=tensor.numpy_dtype)


def convert_function(
    bundle: Bundle, family: Family, function: Function, directory: Path
) -> tuple[Path, float]:
    """Trace and convert one function into an fp16 ML program package under *directory*.

    Returns the package and the seconds the trace and conversion took.
    """
    started = time.time()
    sig = signature(bundle.manifest, family)
    module = build_module(bundle, family, function.length)
    example = trace_inputs(sig, sample_feed(bundle, family, function))
    with torch.no_grad():
        traced = cast(Callable[..., Any], torch.jit.trace)(module, example)
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[_feature(tensor, function.length) for tensor in sig.inputs],
        outputs=[ct.TensorType(name=tensor.name, dtype=np.float16) for tensor in sig.outputs],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    package = directory / f"{family}-{function.name}.mlpackage"
    model.save(str(package))
    del module, traced, model
    gc.collect()
    return package, time.time() - started


def convert_family(
    bundle: Bundle, family: Family, build_dir: Path, models_dir: Path
) -> ConvertedFamily:
    """Convert every function of *family*, merge them into one package and compile it.

    One function at a time: each traced graph and converted model is freed before the next. The
    per-function packages are deleted once merged; the merged package stays for the placement
    variants, and the compiled model lands in ``models_dir/<directory>/``.
    """
    members = functions(bundle.manifest, family)
    packages: dict[str, Path] = {}
    seconds: dict[str, float] = {}
    for function in members:
        packages[function.name], seconds[function.name] = convert_function(
            bundle, family, function, build_dir
        )
        log.info("converted %s in %.1fs", function.name, seconds[function.name])
    descriptor = ct.utils.MultiFunctionDescriptor()
    for function in members:
        descriptor.add_function(str(packages[function.name]), "main", function.name)
    descriptor.default_function_name = members[0].name
    package = build_dir / f"{family}.mlpackage"
    ct.utils.save_multifunction(descriptor, str(package))
    for part in packages.values():
        shutil.rmtree(part)
    _rewrite_spec_deterministic(package)
    directory = models_dir / spec.DIRECTORIES[family]
    directory.mkdir(parents=True, exist_ok=True)
    compiled = directory / f"{spec.DIRECTORIES[family]}.mlmodelc"
    # compile_model nests its output inside a directory that exists; a rerun replaces instead.
    shutil.rmtree(compiled, ignore_errors=True)
    ct.utils.compile_model(str(package), str(compiled))
    write_directory_manifest(bundle.manifest, family, directory)
    log.info("compiled %s", compiled)
    return ConvertedFamily(family, directory, package, compiled, members, seconds)


def compile_plan_variant(converted: ConvertedFamily, function: Function, build_dir: Path) -> Path:
    """A compiled copy of the family with *function* as the default.

    ``MLComputePlan`` reports device usage only for the compiled model's default function, so
    placement per function needs one copy per function that differs solely in
    ``description.defaultFunctionName``; programs and weights are identical across copies. The
    family's own compiled model is the default's, and is returned as it is; any other copy is the
    caller's to delete once its plan is read (a copy holds the family's weights again).
    """
    if function == converted.functions[0]:
        return converted.compiled
    variant = build_dir / f"plan_{function.name}.mlpackage"
    shutil.copytree(converted.package, variant)
    _rewrite_spec_deterministic(variant, function.name)
    compiled = build_dir / f"plan_{function.name}.mlmodelc"
    ct.utils.compile_model(str(variant), str(compiled))
    shutil.rmtree(variant)
    return compiled
