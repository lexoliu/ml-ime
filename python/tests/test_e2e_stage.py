"""The Colab stager's choices, against a fake bucket and a scratch mount tree.

colab/e2e-stage.py and colab/e2e-publish.py are scripts rather than package
modules, so both are imported by path with their filesystem constants
redirected; the bucket is a fake that answers the listing and download calls
the stager makes.
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

COLAB = Path(__file__).resolve().parents[2] / "colab"


@dataclass
class Entry:
    type: str
    path: str


class FakeBucket:
    """The bucket calls the stager makes, over a dict of path to bytes."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def list_bucket_tree(self, bucket_id: str, prefix: str = "", *, recursive: bool = False):
        return [Entry("file", path) for path in sorted(self.files) if path.startswith(prefix)]

    def download_bucket_files(
        self, bucket_id: str, files: list[tuple[str, Path]], *, raise_on_missing_files: bool
    ) -> None:
        for source, destination in files:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.files[source])


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    module = load("e2e_stage", COLAB / "e2e-stage.py")
    monkeypatch.setattr(module, "INPUTS", tmp_path / "input")
    monkeypatch.setattr(module, "WORKING", tmp_path / "working")
    return module


@pytest.fixture
def publisher() -> ModuleType:
    return load("e2e_publish", COLAB / "e2e-publish.py")


def whole_trio(run: str, step: int) -> dict[str, bytes]:
    prefix = f"{run}/step-{step:07d}"
    files = {
        f"{prefix}/checkpoint-paused.pt": b"weights",
        f"{prefix}/run-config.json": json.dumps({"segment": 0}).encode(),
        f"{prefix}/run-summary.json": b"{}",
    }
    files[f"{prefix}/complete"] = f"{step}\n".encode()
    return files


def test_a_run_of_its_own_with_no_trio_starts_fresh(
    stager: ModuleType, publisher: ModuleType
) -> None:
    assert stager.bucket_trio(publisher, FakeBucket({}), "mlime-e2e-v2-a100", False) is None


def test_resuming_another_run_without_a_trio_refuses(
    stager: ModuleType, publisher: ModuleType
) -> None:
    """--resume-from names a run whose trio must exist; a fresh start is not a fallback."""
    with pytest.raises(FileNotFoundError, match="mlime-e2e-v2-t4-check"):
        stager.bucket_trio(publisher, FakeBucket({}), "mlime-e2e-v2-t4-check", True)


def test_resuming_another_run_stages_its_newest_whole_trio(
    stager: ModuleType, publisher: ModuleType, tmp_path: Path
) -> None:
    files = {**whole_trio("mlime-e2e-v2-a100", 40), **whole_trio("mlime-e2e-v2-a100", 80)}
    del files["mlime-e2e-v2-a100/step-0000080/complete"]
    mount = stager.bucket_trio(publisher, FakeBucket(files), "mlime-e2e-v2-a100", True)
    assert mount == tmp_path / "input" / "mlime-e2e-v2-a100-resume"
    assert sorted(p.name for p in mount.iterdir()) == sorted(stager.RESUME_MARKERS)


def test_a_leg_without_its_commit_or_session_length_is_refused(stager: ModuleType) -> None:
    with pytest.raises(SystemExit):
        stager.options(["--commit", "a" * 40])
    with pytest.raises(SystemExit):
        stager.options(["--session-seconds", "3600"])
    with pytest.raises(SystemExit):
        stager.options(["--commit", "a" * 40, "--session-seconds", "0"])


def test_the_leg_stamps_its_commit_and_session_into_the_kernel(
    stager: ModuleType, tmp_path: Path
) -> None:
    """Each stamp lands on its own line as a literal the kernel reads back."""
    leg = stager.options(
        ["--commit", "c" * 40, "--session-seconds", "5400", "--allow-commit-change"]
    )
    kernel = Path(__file__).resolve().parents[2] / "kaggle" / "e2e" / "kernel.py"
    stamped = tmp_path / "kernel.py"
    stamped.write_text(stager.stamp(kernel.read_text(), **stager.stamps(leg, 3)))
    module = load("stamped_kernel", stamped)
    assert module.COMMIT == "c" * 40
    assert module.SESSION_SECONDS == 5400
    assert module.ALLOW_COMMIT_CHANGE is True
    assert module.SEGMENT == 3
    assert module.CHECKPOINT_MINUTES == stager.CHECKPOINT_MINUTES
    with pytest.raises(RuntimeError, match="no `MISSING =` line"):
        stager.stamp(kernel.read_text(), MISSING=1)
