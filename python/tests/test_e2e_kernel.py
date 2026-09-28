"""The e2e kernel's resume pick, exercised against mounted trios.

kaggle/e2e/kernel.py is a kernel script rather than a package module, so it
is imported by path; INPUTS and SEGMENT stand in for the mount layout and
the stamp a push applies.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import torch

KERNEL_PATH = Path(__file__).resolve().parents[2] / "kaggle" / "e2e" / "kernel.py"


def load_kernel(monkeypatch: pytest.MonkeyPatch, inputs: Path, segment: int) -> ModuleType:
    """kernel.py as a module, with its mounts and its stamp redirected."""
    spec = importlib.util.spec_from_file_location("e2e_kernel", KERNEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "INPUTS", inputs)
    monkeypatch.setattr(module, "SEGMENT", segment)
    return module


def trio(
    directory: Path,
    step: int,
    world: int = 2,
    segment: int = 0,
    max_steps: int = 100,
    last_step: int | None = None,
) -> Path:
    """A mounted resume trio whose checkpoint itself records *step*."""
    directory.mkdir(parents=True)
    torch.save(
        {"step": step, "positions": [{} for _ in range(world)]},
        directory / "checkpoint-paused.pt",
    )
    (directory / "run-config.json").write_text(
        json.dumps({"max_steps": max_steps, "epochs": 1, "segment": segment})
    )
    (directory / "run-summary.json").write_text(
        json.dumps({"last_step": step if last_step is None else last_step})
    )
    return directory


def test_segment_zero_with_a_mounted_trio_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A re-pushed segment 0 resumes what a Colab leg published."""
    kernel = load_kernel(monkeypatch, tmp_path, segment=0)
    mounted = trio(tmp_path / "mlime-e2e-resume", step=40, segment=5)
    picked = kernel.previous_segment()
    assert picked is not None
    assert picked["checkpoint"] == mounted / "checkpoint-paused.pt"
    assert picked["world"] == 2


def test_segment_zero_with_nothing_mounted_starts_fresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kernel = load_kernel(monkeypatch, tmp_path, segment=0)
    assert kernel.previous_segment() is None


def test_a_later_segment_with_nothing_mounted_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kernel = load_kernel(monkeypatch, tmp_path, segment=2)
    with pytest.raises(FileNotFoundError, match="no resumable trio"):
        kernel.previous_segment()


def test_the_pick_is_the_checkpoint_step_not_the_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A summary's last_step can claim more than its checkpoint holds."""
    kernel = load_kernel(monkeypatch, tmp_path, segment=3)
    ahead = trio(tmp_path / "mlime-e2e-resume", step=150, last_step=50)
    behind = trio(tmp_path / "mlime-e2e-s2" / "run", step=100, last_step=200)
    picked = kernel.previous_segment()
    assert picked is not None and picked["mount"] == ahead
    assert picked["checkpoint"] != behind / "checkpoint-paused.pt"


def test_two_lineages_of_run_config_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mounted trios that describe different runs fail the pick by name."""
    kernel = load_kernel(monkeypatch, tmp_path, segment=3)
    trio(tmp_path / "a", step=10, max_steps=100)
    trio(tmp_path / "b", step=20, max_steps=200)
    with pytest.raises(ValueError, match="max_steps"):
        kernel.previous_segment()
