"""The Colab publisher's resume trios in the Hub bucket, against a fake bucket.

colab/e2e-publish.py is a script rather than a package module, so it is
imported by path with its filesystem constants redirected; the `HfApi` it
drives is a fake that keeps the bucket as a dict of path to bytes and records
every batch, so listing, adding and deleting behave the way the real bucket
does for the calls the publisher makes.
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch

PUBLISHER_PATH = Path(__file__).resolve().parents[2] / "colab" / "e2e-publish.py"
RUN = "mlime-e2e-rankw01b"


@dataclass
class Entry:
    type: str
    path: str


class FakeBucket:
    """The bucket calls the publisher makes, over a dict of path to bytes."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.batches: list[tuple[list[str], list[str]]] = []

    def list_bucket_tree(self, bucket_id: str, prefix: str = "", *, recursive: bool = False):
        return [Entry("file", path) for path in sorted(self.files) if path.startswith(prefix)]

    def batch_bucket_files(self, bucket_id: str, *, add=(), delete=()) -> None:
        for source, destination in add:
            self.files[destination] = source if isinstance(source, bytes) else source.read_bytes()
        for path in delete:
            del self.files[path]
        self.batches.append(([destination for _, destination in add], list(delete)))

    def steps(self) -> list[str]:
        """The trio directories the bucket holds."""
        return sorted({path.rsplit("/", 1)[0] for path in self.files})


def load_publisher(
    monkeypatch: pytest.MonkeyPatch, working: Path, bucket: FakeBucket | None = None
) -> ModuleType:
    """e2e-publish.py as a module, with its working dir and Hub client redirected."""
    spec = importlib.util.spec_from_file_location("e2e_publish", PUBLISHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKING", working)
    monkeypatch.setattr(module, "RUN", working / "run")
    monkeypatch.setattr(module, "STATE", working / "publish-state.json")
    metadata = working / "kernel-metadata.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps({"id": f"lexoliu/{RUN}-s0"}))
    monkeypatch.setattr(module, "METADATA", metadata)
    if bucket is not None:
        monkeypatch.setattr(module, "HfApi", lambda: bucket)
    return module


def write_trio(directory: Path, step: int) -> Path:
    """A resume trio whose checkpoint and summary record *step*."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "positions": [{}, {}]}, directory / "checkpoint-paused.pt")
    (directory / "run-config.json").write_text(json.dumps({"max_steps": 100}))
    (directory / "run-summary.json").write_text(
        json.dumps({"last_step": step, "checkpoint_step": step})
    )
    return directory


def publish_once(
    monkeypatch: pytest.MonkeyPatch, working: Path, bucket: FakeBucket, step: int
) -> None:
    """Put a trio at *step* in the working root and run one publisher pass."""
    write_trio(working, step=step)
    publisher = load_publisher(monkeypatch, working, bucket)
    monkeypatch.setenv("ONCE", "1")
    publisher.main()


def test_each_publish_leaves_exactly_the_newest_whole_trio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new trio lands, then its marker, and only then the older trio goes."""
    working = tmp_path / "working"
    bucket = FakeBucket()
    publish_once(monkeypatch, working, bucket, 30)
    assert bucket.steps() == [f"{RUN}/step-0000030"]
    publish_once(monkeypatch, working, bucket, 40)
    assert bucket.steps() == [f"{RUN}/step-0000040"]
    assert set(bucket.files) == {
        f"{RUN}/step-0000040/{name}"
        for name in ("checkpoint-paused.pt", "run-config.json", "run-summary.json", "complete")
    }
    trio, marker, deletion = bucket.batches[-3:]
    assert sorted(trio[0]) == sorted(
        f"{RUN}/step-0000040/{name}"
        for name in ("checkpoint-paused.pt", "run-config.json", "run-summary.json")
    )
    assert marker[0] == [f"{RUN}/step-0000040/complete"]
    assert all(path.startswith(f"{RUN}/step-0000030/") for path in deletion[1])
    state = json.loads((working / "publish-state.json").read_text())
    assert state["last_published_step"] == 40


def test_a_trio_without_its_marker_is_not_a_published_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A VM that died mid-publish leaves files without a marker; staging skips them."""
    bucket = FakeBucket()
    bucket.files[f"{RUN}/step-0000030/complete"] = b"30\n"
    bucket.files[f"{RUN}/step-0000040/checkpoint-paused.pt"] = b"partial"
    publisher = load_publisher(monkeypatch, tmp_path / "working")
    assert publisher.published_steps(bucket, RUN) == [30]


def test_nothing_newer_publishes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A trio no newer than the bucket's is not published again."""
    working = tmp_path / "working"
    bucket = FakeBucket()
    publish_once(monkeypatch, working, bucket, 30)
    batches = len(bucket.batches)
    publish_once(monkeypatch, working, bucket, 30)
    assert len(bucket.batches) == batches
    assert bucket.steps() == [f"{RUN}/step-0000030"]


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
