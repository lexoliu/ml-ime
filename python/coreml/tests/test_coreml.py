"""Conversion end to end on the tiny bundle; needs macOS and Core ML, and skips elsewhere."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mlime_coreml import spec
from mlime_coreml.bundle import Bundle

pytestmark = [
    pytest.mark.coreml,
    pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runs on macOS"),
]


def test_a_tiny_bundle_converts_verifies_and_packages_deterministically(
    bundle: Bundle, bundle_dir: Path, tmp_path: Path
) -> None:
    from mlime_coreml.convert import convert_family
    from mlime_coreml.package import build_archive
    from mlime_coreml.verify import verify

    build, models = tmp_path / "build", tmp_path / "models"
    build.mkdir()
    converted = {family: convert_family(bundle, family, build, models) for family in spec.FAMILIES}
    # The placement gate is off here: a host without a Neural Engine (a hosted runner) cannot be
    # held to it, and the tiny model's ops all place on the CPU even on one that has an engine
    # (measured on the M1: 0 of 62 to 163 ops per function). The gate runs on the real-size
    # bundle, in `mlime-coreml convert`, where 98% of the ops place on the engine.
    report = verify(bundle, converted, build, tmp_path, require_neural_engine=False)
    assert report["passed"] and json.loads((tmp_path / "verification.json").read_text())["passed"]
    for family in spec.FAMILIES:
        directory = models / spec.DIRECTORIES[family]
        assert (directory / f"{spec.DIRECTORIES[family]}.mlmodelc").is_dir()
        assert (directory / "manifest.json").is_file()
    first, digest = build_archive(
        bundle.manifest, bundle_dir, models, tmp_path / "verification.json", tmp_path
    )
    again = tmp_path / "again"
    again.mkdir()
    second, repeat = build_archive(
        bundle.manifest, bundle_dir, models, tmp_path / "verification.json", again
    )
    assert digest == repeat and first.read_bytes() == second.read_bytes()
