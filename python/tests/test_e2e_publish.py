"""The Colab publisher's resume trios in the Hub bucket, against a fake bucket.

colab/e2e-publish.py is a script rather than a package module, so it is
imported by path with its filesystem constants redirected; the `HfApi` it
drives is a fake that keeps the bucket as a dict of path to bytes and records
every batch, so listing, adding and deleting behave the way the real bucket
does for the calls the publisher makes.
"""

import importlib.util
import io
import json
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import httpx
import pytest
import torch

PUBLISHER_PATH = Path(__file__).resolve().parents[2] / "colab" / "e2e-publish.py"
RUN = "mlime-e2e-rankw01b"
CACHE = "compile-cache/Tesla-T4-torch2.13.0+cu128.tar"


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
        """The run's trio directories the bucket holds."""
        return sorted({path.rsplit("/", 1)[0] for path in self.files if path.startswith(RUN)})

    def run_batches(self) -> list[tuple[list[str], list[str]]]:
        """The batches that touched the run's trios, not the compile caches."""
        return [batch for batch in self.batches if batch != ([CACHE], [])]


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
    # No GPU here: the caches are a directory of the test's, keyed as a T4's.
    caches = working.parent / "inductor-cache"
    (caches / "fxgraph").mkdir(parents=True, exist_ok=True)
    (caches / "fxgraph" / "entry").write_bytes(b"compiled")
    monkeypatch.setattr(module, "INDUCTOR_CACHE", caches)
    monkeypatch.setattr(module, "compile_cache_path", lambda: CACHE)
    if bucket is not None:
        monkeypatch.setattr(module, "HfApi", lambda: bucket)
    return module


def write_trio(directory: Path, step: int) -> Path:
    """A resume trio whose checkpoint and summary record *step*."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "positions": [{}, {}]}, directory / "checkpoint-paused.pt")
    (directory / "run-config.json").write_text(json.dumps({"max_steps": 100}))
    (directory / "run-summary.json").write_text(
        json.dumps({"last_step": step, "checkpoint_step": step, "finished": False})
    )
    return directory


def write_finished(working: Path, summary: dict[str, object]) -> list[str]:
    """What a finished leg leaves in *working*; the outputs' paths under it."""
    (working / "run").mkdir(parents=True, exist_ok=True)
    outputs = {
        "run-summary.json": json.dumps(summary),
        "e2e-lattice-context-on.jsonl": "{}\n",
        "e2e-lattice-context-on-report.txt": "top-1\n",
        "eval-lattice-context-on.log": "done\n",
        # A key run's chosen-point dumps, gzipped per section by the kernel.
        "e2e-keys-eval3-typo-context-on-selection.json": "{}\n",
        "e2e-keys-eval3-typo-context-on-dev.jsonl.gz": "dev",
        "e2e-keys-eval3-typo-context-on-test.jsonl.gz": "test",
    }
    if "evaluated_checkpoint" not in summary:
        outputs |= {
            "run-config.json": "{}",
            "run/checkpoint-final.pt": "weights",
            "run/metrics.jsonl": json.dumps({"event": "summary", "finished": True}) + "\n",
        }
    for name, text in outputs.items():
        (working / name).write_text(text)
    return sorted(outputs)


def test_a_finished_run_ships_its_outputs_then_their_marker_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    working = tmp_path / "working"
    bucket = FakeBucket()
    outputs = write_finished(working, {"finished": True, "checkpoint_step": 120})
    publisher = load_publisher(monkeypatch, working, bucket)
    monkeypatch.setenv("ONCE", "1")
    publisher.main()
    final = f"final/{RUN}"
    files, marker = bucket.run_batches()
    assert sorted(files[0]) == [f"{final}/{name}" for name in outputs]
    assert marker[0] == [f"{final}/complete"]
    assert bucket.files[f"{final}/run/checkpoint-final.pt"] == b"weights"
    assert bucket.files[f"{final}/e2e-keys-eval3-typo-context-on-test.jsonl.gz"] == b"test"
    assert json.loads((working / "publish-state.json").read_text())["final_published"]

    again = load_publisher(monkeypatch, working, bucket)
    batches = len(bucket.batches)
    again.main()
    assert len(bucket.batches) == batches, "the final outputs went up again"


def test_the_trio_rotation_leaves_a_finished_runs_outputs_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket = FakeBucket()
    bucket.files[f"final/{RUN}/run/checkpoint-final.pt"] = b"weights"
    bucket.files[f"final/{RUN}/complete"] = b"7\n"
    publish_once(monkeypatch, tmp_path / "working", bucket, 30)
    publish_once(monkeypatch, tmp_path / "working", bucket, 40)
    assert bucket.files[f"final/{RUN}/run/checkpoint-final.pt"] == b"weights"
    assert bucket.steps() == [f"{RUN}/step-0000040"]


def test_a_leg_that_only_evaluated_ships_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its checkpoint and config were mounted from where they already live."""
    working = tmp_path / "working"
    bucket = finished_bucket()
    outputs = write_finished(working, {"finished": True, "evaluated_checkpoint": "/mnt/x.pt"})
    publisher = load_publisher(monkeypatch, working, bucket)
    monkeypatch.setenv("ONCE", "1")
    publisher.main()
    assert sorted(bucket.run_batches()[0][0]) == [f"final/{RUN}/{name}" for name in outputs]
    assert not {"run/checkpoint-final.pt", "run-config.json"} & set(outputs)


def finished_bucket() -> FakeBucket:
    """A bucket holding a finished run's final outputs, with an older evaluation's files."""
    final = f"final/{RUN}"
    bucket = FakeBucket()
    bucket.files |= {
        f"{final}/run/checkpoint-final.pt": b"weights",
        f"{final}/run-config.json": b'{"segment": 4}',
        f"{final}/run-summary.json": b'{"finished": true, "held_out": {}}',
        f"{final}/e2e-keys-eval3-context-on-dev.jsonl.gz": b"old dev",
        f"{final}/e2e-keys-eval3-context-on-test.jsonl.gz": b"old test",
        f"{final}/eval-keys-eval3-context-on-dev.log": b"old log",
        f"{final}/complete": b"9\n",
    }
    return bucket


def test_evaluating_a_finished_run_again_replaces_its_evaluation_beside_its_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bucket's marker is the earlier leg's: this leg still publishes its own files."""
    final = f"final/{RUN}"
    working = tmp_path / "working"
    bucket = finished_bucket()
    old = dict(bucket.files)
    outputs = write_finished(working, {"finished": True, "evaluated_checkpoint": "/mnt/x.pt"})
    publisher = load_publisher(monkeypatch, working, bucket)
    monkeypatch.setenv("ONCE", "1")
    assert not publisher.leg_published_final()
    publisher.main()
    assert publisher.leg_published_final()
    assert json.loads((working / "publish-state.json").read_text())["final_published"]
    # Byte-identical checkpoint and config, never re-uploaded.
    for kept in ("run/checkpoint-final.pt", "run-config.json"):
        assert bucket.files[f"{final}/{kept}"] == old[f"{final}/{kept}"]
        assert all(f"{final}/{kept}" not in added for added, _ in bucket.batches)
    # The new summary and dumps are up; the old evaluation's test dump, which this
    # one did not write, and its log are gone; the marker was never deleted.
    assert bucket.files[f"{final}/run-summary.json"] != old[f"{final}/run-summary.json"]
    assert bucket.files[f"{final}/e2e-keys-eval3-typo-context-on-test.jsonl.gz"] == b"test"
    assert f"{final}/e2e-keys-eval3-context-on-dev.jsonl.gz" not in bucket.files
    assert f"{final}/eval-keys-eval3-context-on-dev.log" not in bucket.files
    assert f"{final}/complete" in bucket.files
    assert all(f"{final}/complete" not in deleted for _, deleted in bucket.batches)
    assert sorted(path for path in bucket.files if path.startswith(f"{final}/e2e-")) == sorted(
        f"{final}/{name}" for name in outputs if name.startswith("e2e-")
    )

    again = load_publisher(monkeypatch, working, bucket)
    batches = len(bucket.batches)
    again.main()
    assert len(bucket.batches) == batches, "the evaluation went up again"


def test_an_evaluation_without_a_whole_final_beside_it_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    working = tmp_path / "working"
    bucket = FakeBucket()
    write_finished(working, {"finished": True, "evaluated_checkpoint": "/mnt/x.pt"})
    publisher = load_publisher(monkeypatch, working, bucket)
    monkeypatch.setenv("ONCE", "1")
    with pytest.raises(RuntimeError, match="no `complete` marker"):
        publisher.main()
    assert bucket.files == {}


def publish_once(
    monkeypatch: pytest.MonkeyPatch, working: Path, bucket: FakeBucket, step: int
) -> None:
    """Put a trio at *step* in the working root and run one publisher pass."""
    write_trio(working, step=step)
    publisher = load_publisher(monkeypatch, working, bucket)
    monkeypatch.setenv("ONCE", "1")
    publisher.main()


class Refusing(FakeBucket):
    """A bucket that answers every write with *status* while it is set."""

    def __init__(self, status: int | None) -> None:
        super().__init__()
        self.status = status

    def batch_bucket_files(self, bucket_id: str, *, add=(), delete=()) -> None:
        if self.status is None:
            super().batch_bucket_files(bucket_id, add=add, delete=delete)
            return
        request = httpx.Request("POST", "https://hub.invalid/batch")
        raise httpx.HTTPStatusError(
            f"{self.status} from the bucket",
            request=request,
            response=httpx.Response(self.status, request=request),
        )


def test_a_refused_publish_is_recorded_as_one_until_the_bucket_takes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A storage limit's 403 is a refusal; a 503 or a rate limit is retried quietly."""
    working = tmp_path / "working"
    state = working / "publish-state.json"
    bucket = Refusing(403)
    bucket.files["final/other-run/checkpoint-final.pt"] = b"weights"
    publish_once(monkeypatch, working, bucket, 30)
    recorded = json.loads(state.read_text())
    assert recorded["refused"] is True and "403" in recorded["last_error"]
    assert "bucket_bytes" in recorded
    assert "last_published_step" not in recorded

    for transient in (429, 503):
        bucket.status = transient
        publish_once(monkeypatch, working, bucket, 30)
        assert json.loads(state.read_text())["refused"] is False

    bucket.status = 403
    publish_once(monkeypatch, working, bucket, 30)
    assert json.loads(state.read_text())["refused"] is True
    bucket.status = None
    publish_once(monkeypatch, working, bucket, 30)
    recorded = json.loads(state.read_text())
    assert (recorded["refused"], recorded["last_published_step"]) == (False, 30)


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
    assert {path for path in bucket.files if path.startswith(RUN)} == {
        f"{RUN}/step-0000040/{name}"
        for name in ("checkpoint-paused.pt", "run-config.json", "run-summary.json", "complete")
    }
    trio, marker, deletion = bucket.run_batches()[-3:]
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
    assert len(bucket.batches) == batches, "neither a trio nor the caches went up again"
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


def test_the_caches_go_up_once_this_leg_has_logged_its_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Packed after the leg's fifth step record, whole lines only, and only once."""
    working = tmp_path / "working"
    bucket = FakeBucket()
    publisher = load_publisher(monkeypatch, working, bucket)
    monkeypatch.setenv("ONCE", "1")
    metrics = working / "run" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    records = [{"event": "start"}, *({"event": "step", "step": 10 * n} for n in range(1, 5))]
    metrics.write_text("".join(json.dumps(record) + "\n" for record in records))
    with metrics.open("a") as log:
        log.write('{"event": "st')
    publisher.main()
    assert CACHE not in bucket.files
    metrics.write_text("".join(json.dumps(record) + "\n" for record in records))
    with metrics.open("a") as log:
        log.write(json.dumps({"event": "step", "step": 50}) + "\n")
    publisher.main()
    assert bucket.batches == [([CACHE], [])]
    with tarfile.open(fileobj=io.BytesIO(bucket.files[CACHE])) as tar:
        entry = tar.extractfile("./fxgraph/entry")
        assert entry is not None and entry.read() == b"compiled"


def test_the_caches_are_keyed_by_card_and_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The card's name is made path-safe; the torch build keeps its local version."""
    spec = importlib.util.spec_from_file_location("e2e_publish", PUBLISHER_PATH)
    assert spec is not None and spec.loader is not None
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="NVIDIA A100-SXM4-40GB\n"
        ),
    )
    monkeypatch.setattr(publisher.importlib.metadata, "version", lambda name: "2.13.0+cu128")
    assert (
        publisher.compile_cache_path()
        == "compile-cache/NVIDIA-A100-SXM4-40GB-torch2.13.0+cu128.tar"
    )
