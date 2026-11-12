"""The float32 port: it reproduces the reference, and padding changes no real row."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mlime_coreml.bundle import MANIFEST, WEIGHTS, Bundle, read_bundle
from mlime_coreml.port import check_padding, check_port, edges


def test_the_port_reproduces_what_the_reference_holds(bundle: Bundle) -> None:
    checks = check_port(bundle)
    assert checks, "the reference holds no probe"
    names = {check.name.split(".")[0] for check in checks}
    assert names == {probe.name for probe in bundle.manifest.reference}
    # Every function is compared: the context, the keys and the tree rows, in both forms.
    kinds = {check.name.split(".")[-1] for check in checks}
    assert {"context", "ctc_logits", "cross_keys", "cross_values", "logits", "log_probs"} <= kinds
    failing = [(check.name, check.max_abs) for check in checks if not check.ok]
    assert not failing


def test_padding_is_invisible_at_both_edges_of_every_bucket(bundle: Bundle) -> None:
    found = edges(bundle)
    # Each function has its fullest case, and its smallest when that is another.
    assert {(edge.family, edge.length) for edge in found} == {
        (family.name, function.length)
        for family in bundle.manifest.families
        for function in family.functions
    }
    assert {(e.length, e.real) for e in found if e.family == "keys"} == {
        (16, 14), (16, 1), (32, 30), (32, 15), (64, 62), (64, 31), (128, 126), (128, 63),
    }  # fmt: skip
    failing = [(check.name, check.max_abs) for check in check_padding(bundle) if not check.ok]
    assert not failing


def test_a_bundle_the_code_does_not_speak_is_refused(bundle_dir: Path, tmp_path: Path) -> None:
    for change, message in (
        ({"policy_version": 2}, "packing policy"),
        ({"n_max": 64}, "n_max"),
        ({"neg": -3.0}, "mask"),
        ({"families": []}, "function families"),
        ({"surprise": 1}, "unknown"),
    ):
        broken = tmp_path / f"broken-{len(message)}-{next(iter(change))}"
        shutil.copytree(bundle_dir, broken)
        document = json.loads((broken / MANIFEST).read_text())
        document.update(change)
        (broken / MANIFEST).write_text(json.dumps(document))
        with pytest.raises(ValueError, match=message):
            read_bundle(broken)


def test_a_tensor_the_manifest_does_not_list_is_refused(bundle_dir: Path, tmp_path: Path) -> None:
    broken = tmp_path / "unlisted"
    shutil.copytree(bundle_dir, broken)
    document = json.loads((broken / MANIFEST).read_text())
    document["tensors"].pop()
    (broken / MANIFEST).write_text(json.dumps(document))
    with pytest.raises(ValueError, match=f"{WEIGHTS} and the manifest disagree"):
        read_bundle(broken)
