"""The key-level fixture set is deterministic, and the committed one is current."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import keys_fixtures  # noqa: E402 -- after the path it is found by

COMMITTED = keys_fixtures.REPO / "crates/ime-neural/tests/fixtures-keys"

#: Two runs on one machine agree to the bit; a tree written on another machine
#: agrees to onnxruntime's kernels, which this bounds.
TOLERANCE = 1e-4


def text_files(directory: Path) -> dict[str, bytes]:
    """Every committed file under *directory*, by relative path."""
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def assert_close(actual: Any, expected: Any, where: str) -> None:
    """*actual* equals *expected*, floats to :data:`TOLERANCE`."""
    if isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=TOLERANCE, rel=TOLERANCE), where
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), where
        for key, value in expected.items():
            assert_close(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for index, value in enumerate(expected):
            assert_close(actual[index], value, f"{where}[{index}]")
    else:
        assert actual == expected, where


@pytest.fixture(name="generated", scope="module")
def generated_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two runs of the generator, each into its own directories."""
    runs = []
    for name in ("first", "second"):
        root = tmp_path_factory.mktemp(name)
        keys_fixtures.generate(root / "committed", root / "graphs")
        runs.append(root / "committed")
    return runs[0], runs[1]


def test_two_runs_write_the_same_committed_files(generated: tuple[Path, Path]) -> None:
    first, second = (text_files(run) for run in generated)
    assert first.keys() == second.keys()
    assert "expected.json" in first
    for name, content in first.items():
        assert content == second[name], f"{name} differs between two runs"


def test_the_committed_set_is_current(generated: tuple[Path, Path]) -> None:
    fresh = text_files(generated[0])
    tree = text_files(COMMITTED)
    assert tree.keys() == fresh.keys(), "regenerate: python/scripts/keys_fixtures.py"
    for name, content in fresh.items():
        if name == "expected.json":
            continue
        assert tree[name] == content, f"{name} is stale: regenerate the fixtures"
    expected = json.loads(fresh["expected.json"])
    committed = json.loads(tree["expected.json"])
    assert_close(committed["fp32"], expected["fp32"], "fp32")
    # The int8 kernels differ between platforms: the tree's values are for a
    # consumer to check against, so only their shape is held current.
    assert committed["int8"].keys() == expected["int8"].keys()
    for name, records in expected["int8"].items():
        assert [r["keys"] for r in committed["int8"][name]] == [r["keys"] for r in records]
