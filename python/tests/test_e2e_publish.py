"""The Colab publisher's create-or-version decision, against a stubbed kaggle.

colab/e2e-publish.py is a script rather than a package module, so it is
imported by path with its filesystem constants redirected; the `kaggle` the
decision drives is a stub on PATH that records its argv.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest
import torch

PUBLISHER_PATH = Path(__file__).resolve().parents[2] / "colab" / "e2e-publish.py"


def load_publisher(monkeypatch: pytest.MonkeyPatch, working: Path) -> ModuleType:
    """e2e-publish.py as a module, with its working dir redirected."""
    spec = importlib.util.spec_from_file_location("e2e_publish", PUBLISHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKING", working)
    monkeypatch.setattr(module, "RUN", working / "run")
    monkeypatch.setattr(module, "STATE", working / "publish-state.json")
    return module


@pytest.fixture
def kaggle_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A `kaggle` on PATH recording its argv; `datasets list` reads a flag."""
    log = tmp_path / "kaggle-calls.log"
    binary = tmp_path / "bin" / "kaggle"
    binary.parent.mkdir()
    exists_flag = tmp_path / "dataset-exists"
    binary.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{log}"\n'
        'if [ "$1 $2" = "datasets list" ]; then\n'
        '  echo "ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating"\n'
        '  echo "lexoliu/mlime-e2e-init,mlime-e2e-init,1,2026-09-28,0,0,0"\n'
        f'  if [ -f "{exists_flag}" ]; then '
        'echo "lexoliu/mlime-e2e-resume,mlime-e2e-resume,1,2026-09-28,0,0,0"; fi\n'
        "fi\n"
        "exit 0\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")
    return log


def write_trio(directory: Path, step: int) -> Path:
    """A resume trio whose checkpoint records *step*."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "positions": [{}, {}]}, directory / "checkpoint-paused.pt")
    (directory / "run-config.json").write_text(json.dumps({"max_steps": 100}))
    (directory / "run-summary.json").write_text(json.dumps({"last_step": step}))
    return directory


def test_an_absent_dataset_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kaggle_stub: Path
) -> None:
    """Without lexoliu/mlime-e2e-resume the first publish creates it."""
    working = tmp_path / "working"
    write_trio(working, step=30)
    publisher = load_publisher(monkeypatch, working)
    monkeypatch.setenv("ONCE", "1")
    publisher.main()
    calls = kaggle_stub.read_text()
    assert "datasets list" in calls and "datasets create" in calls
    assert "datasets version" not in calls
    state = json.loads((working / "publish-state.json").read_text())
    assert state["last_published_step"] == 30


def test_an_existing_dataset_is_versioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kaggle_stub: Path
) -> None:
    working = tmp_path / "working"
    write_trio(working, step=30)
    (tmp_path / "dataset-exists").touch()
    publisher = load_publisher(monkeypatch, working)
    monkeypatch.setenv("ONCE", "1")
    publisher.main()
    calls = kaggle_stub.read_text()
    assert "datasets version" in calls and "datasets create" not in calls


def test_a_second_poll_reads_no_checkpoint_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Polls over unchanged trios load each checkpoint once, not once per poll."""
    working = tmp_path / "working"
    interval = write_trio(working / "run" / "interval-000030", step=30)
    write_trio(working, step=40)
    publisher = load_publisher(monkeypatch, working)
    loads = []
    real = publisher._load_checkpoint

    def counting(path: Path):
        loads.append(path)
        return real(path)

    monkeypatch.setattr(publisher, "_load_checkpoint", counting)
    first = publisher.candidates()
    second = publisher.candidates()
    assert [step for _, step in first] == [step for _, step in second] == [30, 40]
    assert loads == [interval / "checkpoint-paused.pt", working / "checkpoint-paused.pt"]
