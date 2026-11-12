"""Emit Core ML's outputs over the bundle's probes as JSON, for the Rust parity tests.

The Rust session's parity test runs this module in this project's environment and compares its
own outputs to these (the same compiled models, the same calls, the same packing): the context
rows, the keys function's CTC logits and cross keys and values over the real keys, and every tree
call's rows at the requests' nodes. Everything handed to ``predict`` is a numpy-owned copy kept
alive past the call (:mod:`mlime_coreml.runtime`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np

from mlime_coreml import spec
from mlime_coreml.bundle import Bundle, read_bundle
from mlime_coreml.feeds import Array
from mlime_coreml.port import context_of, cross_of, tree_calls
from mlime_coreml.runtime import Runner
from mlime_coreml.signature import signature
from mlime_coreml.spec import Family


def _runner(bundle: Bundle, models: Path, family: Family, name: str) -> Runner:
    directory = models / spec.DIRECTORIES[family]
    return Runner(
        directory / f"{spec.DIRECTORIES[family]}.mlmodelc",
        name,
        signature(bundle.manifest, family),
        ct.ComputeUnit.CPU_AND_NE,
    )


def emit(bundle: Bundle, models: Path) -> dict[str, object]:
    """Core ML's outputs over every probe of the bundle's reference."""
    manifest = bundle.manifest
    reference = bundle.reference()
    widths = [function.length for function in manifest.family("keys").functions]
    probes: list[dict[str, object]] = []
    for probe in manifest.reference:
        name = probe.name
        ids = np.asarray(reference[f"{name}.input_ids"])
        width = min(width for width in widths if width >= len(ids))
        count = len(ids) - spec.KEY_SENTINELS
        keys = _runner(bundle, models, "keys", f"keys_{width}")
        context = context_of(reference, name)

        def padded(row: Array, fill: float, width: int = width) -> Array:
            out = np.full(width, fill, dtype=row.dtype)
            out[: len(row)] = row
            return out[None]

        logits, cross_keys, cross_values = keys.outputs(
            {
                "input_ids": padded(ids, manifest.specials.pad),
                "key_letters": padded(
                    np.asarray(reference[f"{name}.key_letters"]), manifest.letter_pad
                ),
                "attention_mask": padded(
                    np.asarray(reference[f"{name}.attention_mask"], dtype=np.float32), 0.0
                ),
                "context": context.rows[None],
                "context_mask": context.mask[None],
            }
        )
        calls: list[dict[str, object]] = []
        cross = cross_of(reference, name)
        for call in tree_calls(reference, name):
            size = len(np.asarray(reference[f"{call}.reader_ids"]))
            tree = _runner(bundle, models, "tree", f"tree_{size}")
            [rows] = tree.outputs(
                {
                    "reader_ids": np.asarray(reference[f"{call}.reader_ids"])[None],
                    "positions": np.asarray(reference[f"{call}.positions"])[None],
                    "tree_bias": np.asarray(reference[f"{call}.tree_bias"])[None, None],
                    "cross_keys": cross.keys,
                    "cross_values": cross.values,
                    "cross_bias": cross.bias,
                }
            )
            answers = np.asarray(reference[f"{call}.answers"]).reshape(-1, 2)
            calls.append(
                {
                    "call": call,
                    "function": f"tree_{size}",
                    "answers": answers.tolist(),
                    "logits": rows[answers[:, 1]].tolist(),
                }
            )
        probes.append(
            {
                "name": name,
                "keys": probe.keys,
                "context": probe.context,
                "keys_function": f"keys_{width}",
                "ctc_logits": logits[1 : count + 1].tolist(),
                "cross_keys": cross_keys[:, :, :count].tolist(),
                "cross_values": cross_values[:, :, :count].tolist(),
                "tree": calls,
            }
        )
    return {"step": manifest.step, "probes": probes}


def main() -> None:
    """``python -m mlime_coreml.reference --bundle DIR --models DIR``: the JSON on stdout."""
    parser = argparse.ArgumentParser(prog="mlime_coreml.reference")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    args = parser.parse_args()
    json.dump(emit(read_bundle(args.bundle), args.models), sys.stdout)


if __name__ == "__main__":
    main()
