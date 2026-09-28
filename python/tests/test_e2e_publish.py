"""The Colab publisher's alternation between two resume slots, against a fake kaggle.

colab/e2e-publish.py is a script rather than a package module, so it is
imported by path with its filesystem constants redirected; the `kaggle` it
drives is a stub on PATH that keeps each "dataset" as a directory, so create,
delete, list and a single-file download behave the way the real service does
for the calls the publisher makes, and every argv is recorded.
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
    """A fake `kaggle` on PATH whose datasets are directories under ``store``."""
    log = tmp_path / "kaggle-calls.log"
    store = tmp_path / "store"
    store.mkdir()
    binary = tmp_path / "bin" / "kaggle"
    binary.parent.mkdir()
    binary.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{log}"\n'
        f'store="{store}"\n'
        'case "$1 $2" in\n'
        '  "datasets list")\n'
        '    echo "ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating"\n'
        '    for d in "$store"/*; do [ -d "$d" ] && '
        'echo "lexoliu/$(basename "$d"),x,1,2026-09-28,0,0,0"; done; true ;;\n'
        '  "datasets download")\n'
        '    cp "$store/${3#lexoliu/}/$5" "$7/$5" ;;\n'
        '  "datasets delete")\n'
        '    rm -rf "$store/${4#lexoliu/}" ;;\n'
        '  "datasets create")\n'
        "    id=$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1]))['id'])\" "
        '"$4/dataset-metadata.json")\n'
        '    mkdir -p "$store/${id#lexoliu/}" && cp "$4"/* "$store/${id#lexoliu/}/" ;;\n'
        '  *) echo "unexpected kaggle call: $*" >&2; exit 1 ;;\n'
        "esac\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary.parent}:{os.environ['PATH']}")
    return store


def write_trio(directory: Path, step: int) -> Path:
    """A resume trio whose checkpoint and summary record *step*."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "positions": [{}, {}]}, directory / "checkpoint-paused.pt")
    (directory / "run-config.json").write_text(json.dumps({"max_steps": 100}))
    (directory / "run-summary.json").write_text(
        json.dumps({"last_step": step, "checkpoint_step": step})
    )
    return directory


def held(store: Path) -> dict[str, int]:
    """The step each existing fake slot holds."""
    return {
        slot.name: json.loads((slot / "run-summary.json").read_text())["checkpoint_step"]
        for slot in sorted(store.iterdir())
    }


def publish_once(monkeypatch: pytest.MonkeyPatch, working: Path, step: int) -> None:
    """Put a trio at *step* in the working root and run one publisher pass."""
    write_trio(working, step=step)
    publisher = load_publisher(monkeypatch, working)
    monkeypatch.setenv("ONCE", "1")
    publisher.main()


def test_slots_fill_then_the_older_one_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kaggle_stub: Path
) -> None:
    """Two creates fill both slots; the third publish deletes and recreates the older."""
    working = tmp_path / "working"
    publish_once(monkeypatch, working, 30)
    assert held(kaggle_stub) == {"mlime-e2e-resume-a": 30}
    publish_once(monkeypatch, working, 40)
    assert held(kaggle_stub) == {"mlime-e2e-resume-a": 30, "mlime-e2e-resume-b": 40}
    publish_once(monkeypatch, working, 50)
    assert held(kaggle_stub) == {"mlime-e2e-resume-a": 50, "mlime-e2e-resume-b": 40}
    publish_once(monkeypatch, working, 60)
    assert held(kaggle_stub) == {"mlime-e2e-resume-a": 50, "mlime-e2e-resume-b": 60}
    calls = (tmp_path / "kaggle-calls.log").read_text()
    assert "datasets version" not in calls
    assert calls.count("datasets delete") == 2
    state = json.loads((working / "publish-state.json").read_text())
    assert state["last_published_step"] == 60


def test_nothing_newer_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kaggle_stub: Path
) -> None:
    """A trio no newer than what a slot already holds is not published again."""
    working = tmp_path / "working"
    publish_once(monkeypatch, working, 30)
    before = (tmp_path / "kaggle-calls.log").read_text()
    publish_once(monkeypatch, working, 30)
    after = (tmp_path / "kaggle-calls.log").read_text()
    assert "datasets create" not in after[len(before) :]
    assert held(kaggle_stub) == {"mlime-e2e-resume-a": 30}


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
