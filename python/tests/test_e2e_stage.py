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
    with pytest.raises(SystemExit):
        stager.options(["--commit", "a" * 40, "--session-seconds", "3600", "--segment-steps", "0"])


def test_the_leg_stamps_its_commit_and_session_into_the_kernel(
    stager: ModuleType, tmp_path: Path
) -> None:
    """Each stamp lands on its own line as a literal the kernel reads back."""
    leg = stager.options(
        [
            "--commit",
            "c" * 40,
            "--session-seconds",
            "5400",
            "--segment-steps",
            "70",
            "--allow-commit-change",
        ]
    )
    kernel = Path(__file__).resolve().parents[2] / "kaggle" / "e2e" / "kernel.py"
    stamped = tmp_path / "kernel.py"
    stamped.write_text(stager.stamp(kernel.read_text(), **stager.stamps(leg, 3)))
    module = load("stamped_kernel", stamped)
    assert module.COMMIT == "c" * 40
    assert module.SESSION_SECONDS == 5400
    assert module.SEGMENT_STEPS == 70
    assert module.ALLOW_COMMIT_CHANGE is True
    assert module.SEGMENT == 3
    assert module.CHECKPOINT_MINUTES == stager.CHECKPOINT_MINUTES
    # A leg that names no count runs to its clock, whatever the leg before it set.
    unbounded = stager.options(["--commit", "c" * 40, "--session-seconds", "5400"])
    restamped = tmp_path / "restamped.py"
    restamped.write_text(stager.stamp(stamped.read_text(), **stager.stamps(unbounded, 4)))
    assert load("restamped_kernel", restamped).SEGMENT_STEPS is None
    with pytest.raises(KeyError, match="no `MISSING =` line"):
        stager.stamp(kernel.read_text(), MISSING=1)


def test_a_v2_run_never_resumes_v1s_segments_or_trios(
    stager: ModuleType,
    publisher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stamped v2 kernel stages nothing of v1's chain or of a stage's.

    A leg finds earlier state three ways: what its metadata mounts, its run's
    trios in the bucket, and its base slug's Kaggle segments. v2 mounts none
    of v1's sources, and its run name is its own on the other two, where v1
    has 59,197 steps of trios and six complete segments.
    """
    from mlime.stamp import E2E_V2_SOURCES, E2EV2Run, stamp_e2e_v2

    root = COLAB.parent
    out = tmp_path / "a100"
    stamp_e2e_v2(root / "kaggle" / "e2e", out, E2EV2Run(run="mlime-e2e-v2-a100"))
    metadata = out / "kernel-metadata.json"
    stamped = json.loads(metadata.read_text())
    v1 = json.loads((root / "kaggle" / "e2e" / "kernel-metadata.json").read_text())
    assert stamped["kernel_sources"] == []
    assert stamped["dataset_sources"] == list(E2E_V2_SOURCES)
    assert set(v1["dataset_sources"]) - set(E2E_V2_SOURCES) == {
        "lexoliu/mlime-e2e-census",
        "lexoliu/mlime-e2e-resume-a",
        "lexoliu/mlime-e2e-resume-b",
    }

    run = publisher.run_name(metadata)
    assert run == "mlime-e2e-v2-a100"
    bucket = FakeBucket({**whole_trio("mlime-e2e", 59197), **whole_trio("mlime-e2e-rankw005", 900)})
    assert publisher.published_steps(bucket, "mlime-e2e") == [59197]
    assert stager.bucket_trio(publisher, bucket, run, False) is None

    asked: list[str] = []

    def status(slug: str) -> str | None:
        asked.append(slug)
        v1_segments = {f"lexoliu/mlime-e2e-s{n}" for n in range(6)}
        return "complete" if slug in v1_segments else None

    monkeypatch.setattr(stager, "kernel_status", status)
    assert stager.latest_segment(publisher.run_base(metadata)) is None
    assert asked == ["lexoliu/mlime-e2e-v2-a100-s0"]

    for taken in ("mlime-e2e", "mlime-e2e-rankw005", "mlime-e2e-v2", "mlime-e2e-v2-A100"):
        with pytest.raises(ValueError, match="not a v2 run name"):
            E2EV2Run(run=taken)


#: The finished v2 run #132's stages start from.
PARENT = "mlime-e2e-v2-g4"


def final_outputs(run: str, stage: dict[str, object], marked: bool) -> dict[str, bytes]:
    """*run*'s final outputs as the publisher leaves them, its marker last."""
    prefix = f"final/{run}"
    config = {"max_steps": 12000, "epochs": 1, **stage, "commit": "c" * 40, "segment": 4}
    files = {
        f"{prefix}/run/checkpoint-final.pt": b"final weights",
        f"{prefix}/run/metrics.jsonl": b"{}\n",
        f"{prefix}/run-config.json": json.dumps(config).encode(),
        f"{prefix}/run-summary.json": json.dumps({"finished": True}).encode(),
        f"{prefix}/e2e-lattice-context-on.jsonl": b"{}\n",
        f"{prefix}/eval-lattice-context-on.log": b"",
    }
    if marked:
        files[f"{prefix}/complete"] = b"6\n"
    return files


def stage_kernel(tmp_path: Path) -> Path:
    """#132's weight-0.05 stage of :data:`PARENT`, stamped from the committed kernel."""
    from mlime.stamp import E2EV2Run, E2EV2Stage, stamp_e2e_v2
    from mlime.train.e2e_model import RankingConfig

    stage = E2EV2Stage(
        init_from=PARENT,
        ranking=RankingConfig(weight=0.05),
        learning_rates=(2e-5, 2e-5),
        token_budget=12288,
        accumulate=2,
    )
    out = tmp_path / "rankw005"
    stamp_e2e_v2(
        COLAB.parent / "kaggle" / "e2e", out, E2EV2Run(run=f"{PARENT}-rankw005", stage=stage)
    )
    return out / "kernel.py"


def test_only_a_v2_stage_stages_its_start_from_the_bucket(
    stager: ModuleType, tmp_path: Path
) -> None:
    """v1's finished run is a Hub repo the metadata mounts; a v2 run's lives in the bucket."""
    from mlime.stamp import E2EV2Run, stamp, stamp_e2e_v2

    assert stager.init_from_bucket(stage_kernel(tmp_path)) == PARENT
    plain = tmp_path / "plain"
    stamp_e2e_v2(COLAB.parent / "kaggle" / "e2e", plain, E2EV2Run(run="mlime-e2e-v2-a100"))
    assert stager.init_from_bucket(plain / "kernel.py") is None
    v1_stage = tmp_path / "v1-stage.py"
    template = (COLAB.parent / "kaggle" / "e2e" / "kernel.py").read_text()
    v1_stage.write_text(stamp(template, INIT_FROM="mlime-e2e-final"))
    assert stager.init_from_bucket(v1_stage) is None


def test_a_stage_mounts_its_finished_v2_run_where_the_kernel_starts_from_it(
    stager: ModuleType,
    publisher: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stager's mount is the kernel's start, never its result or a trio to resume.

    Only what the kernel reads is downloaded. The stage's own paused trio,
    staged beside it on a later leg, is what that leg resumes, and the
    stage's own finished run is told apart from its parent by the stage its
    run-config records.
    """
    import torch

    kernel = load("stage_kernel", stage_kernel(tmp_path))
    monkeypatch.setattr(kernel, "INPUTS", stager.INPUTS)
    parent_stage = {"ranking": None, "whole_alphabet": False, "init_from": None}
    bucket = FakeBucket(final_outputs(PARENT, parent_stage, marked=True))
    mount = stager.bucket_final(publisher, bucket, PARENT)
    assert mount == stager.INPUTS / PARENT
    staged = sorted(str(p.relative_to(mount)) for p in mount.rglob("*") if p.is_file())
    assert staged == sorted(stager.FINAL_READS)

    assert kernel.init_checkpoint() == mount / "run" / "checkpoint-final.pt"
    assert kernel.finished_checkpoint() is None
    assert kernel.previous_segment() is None
    assert stager.trios() == []

    run = f"{PARENT}-rankw005"
    trio = whole_trio(run, 300)
    paused = tmp_path / "paused.pt"
    torch.save({"step": 300, "positions": [{}, {}]}, paused)
    trio[f"{run}/step-0000300/checkpoint-paused.pt"] = paused.read_bytes()
    config = {"max_steps": 900, "epochs": 1, **kernel.stage(), "segment": 0}
    trio[f"{run}/step-0000300/run-config.json"] = json.dumps(config).encode()
    resume = stager.bucket_trio(publisher, FakeBucket(trio), run, False)
    picked = kernel.previous_segment()
    assert picked is not None and picked["mount"] == resume and picked["step"] == 300
    assert kernel.init_checkpoint() == mount / "run" / "checkpoint-final.pt"

    own = stager.INPUTS / f"{run}-s1"
    (own / "run").mkdir(parents=True)
    (own / "run" / "checkpoint-final.pt").write_bytes(b"stage weights")
    (own / "run-config.json").write_text(json.dumps(config))
    assert kernel.finished_checkpoint() == own / "run" / "checkpoint-final.pt"
    assert kernel.init_checkpoint() == mount / "run" / "checkpoint-final.pt"


def test_a_stage_refuses_a_v2_run_whose_final_outputs_are_not_whole(
    stager: ModuleType, publisher: ModuleType
) -> None:
    """Without the marker the outputs may be half uploaded; nothing is staged from them."""
    stage = {"ranking": None, "whole_alphabet": False, "init_from": None}
    bucket = FakeBucket(final_outputs(PARENT, stage, marked=False))
    with pytest.raises(FileNotFoundError, match=f"final/{PARENT}: no `complete` marker"):
        stager.bucket_final(publisher, bucket, PARENT)
    assert not (stager.INPUTS / PARENT).exists()
