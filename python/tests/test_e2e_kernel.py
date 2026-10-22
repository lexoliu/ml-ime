"""The e2e kernel's resume pick, exercised against mounted trios.

kaggle/e2e/kernel.py is a kernel script rather than a package module, so it
is imported by path; INPUTS and SEGMENT stand in for the mount layout and
the stamp a push applies.
"""

import importlib.util
import json
import os
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


def finished(directory: Path, stage: dict[str, object] | None) -> Path:
    """A mounted finished run whose run-config records *stage* (v1's records none)."""
    (directory / "run").mkdir(parents=True)
    (directory / "run" / "checkpoint-final.pt").write_bytes(b"")
    config: dict[str, object] = {"max_steps": 10, "epochs": 1, "segment": 3}
    if stage is not None:
        config.update(stage)
    (directory / "run-config.json").write_text(json.dumps(config))
    return directory / "run" / "checkpoint-final.pt"


def test_a_ranking_stage_starts_from_v1_and_evaluates_only_its_own_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """v1's finished run is the start of a ranking stage, never its result."""
    kernel = load_kernel(monkeypatch, tmp_path, segment=0)
    ranking = {"segmentations": 8, "beam": 4, "negatives": 8, "weight": 1.0}
    monkeypatch.setattr(kernel, "RANKING", ranking)
    monkeypatch.setattr(kernel, "INIT_FROM", "mlime-e2e-final")
    monkeypatch.setattr(kernel, "WHOLE_ALPHABET", True)
    v1 = finished(tmp_path / "mlime-e2e-final", None)
    assert kernel.init_checkpoint() == v1
    assert kernel.finished_checkpoint() is None
    assert kernel.start_argv(None, v1) == ["--init-e2e", str(v1)]
    assert kernel.ranking_argv(Path("/bin/ime-segment")) == [
        "--ranking",
        "--segmenter",
        "/bin/ime-segment",
        "--rank-segmentations",
        "8",
        "--rank-beam",
        "4",
        "--rank-negatives",
        "8",
        "--rank-weight",
        "1.0",
    ]

    ranked = finished(tmp_path / "mlime-e2e-rank-s2", kernel.stage())
    assert kernel.finished_checkpoint() == ranked
    assert kernel.init_checkpoint() == v1

    # The v1 kernel sees v1's run as its own finished one, and ranks nothing.
    monkeypatch.setattr(kernel, "RANKING", None)
    monkeypatch.setattr(kernel, "INIT_FROM", None)
    monkeypatch.setattr(kernel, "WHOLE_ALPHABET", False)
    assert kernel.finished_checkpoint() == v1
    assert kernel.ranking_argv(Path("/bin/ime-segment")) == []
    assert kernel.start_argv((Path("/enc"), Path("/dec")), None) == [
        "--init-encoder",
        "/enc",
        "--init-decoder",
        "/dec",
    ]


def test_a_control_stage_is_told_apart_from_the_ranking_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fine-tune from v1 without ranking evaluates its own run, not a ranking run's."""
    kernel = load_kernel(monkeypatch, tmp_path, segment=0)
    monkeypatch.setattr(kernel, "INIT_FROM", "mlime-e2e-final")
    monkeypatch.setattr(kernel, "WHOLE_ALPHABET", True)
    v1 = finished(tmp_path / "mlime-e2e-final", None)
    ranking = {"segmentations": 8, "beam": 4, "negatives": 8, "weight": 1.0}
    finished(
        tmp_path / "mlime-e2e-rank2-s1",
        {"ranking": ranking, "whole_alphabet": True, "init_from": "mlime-e2e-final"},
    )
    assert kernel.init_checkpoint() == v1
    assert kernel.finished_checkpoint() is None
    assert kernel.ranking_argv(Path("/bin/ime-segment")) == []

    control = finished(tmp_path / "mlime-e2e-control-s0", kernel.stage())
    assert kernel.finished_checkpoint() == control


def test_a_stage_starts_from_an_earlier_stage_mounted_as_kernel_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """INIT_FROM picks its parent by mount name, and the parent is never the result."""
    kernel = load_kernel(monkeypatch, tmp_path, segment=0)
    finished(tmp_path / "datasets" / "lexoliu" / "mlime-e2e-final", None)
    parent = finished(
        tmp_path / "notebooks" / "lexoliu" / "mlime-e2e-control2-s0",
        {"ranking": None, "whole_alphabet": False, "init_from": "mlime-e2e-final"},
    )
    monkeypatch.setattr(kernel, "INIT_FROM", "mlime-e2e-control2-s0")
    assert kernel.init_checkpoint() == parent
    assert kernel.finished_checkpoint() is None
    assert kernel.start_argv(None, parent) == ["--init-e2e", str(parent)]

    own = finished(tmp_path / "notebooks" / "lexoliu" / "mlime-e2e-stage2-s1", kernel.stage())
    assert kernel.finished_checkpoint() == own

    monkeypatch.setattr(kernel, "INIT_FROM", "mlime-e2e-missing")
    with pytest.raises(FileNotFoundError, match="mlime-e2e-missing"):
        kernel.init_checkpoint()


def test_a_finished_run_keeps_only_its_final_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The numbered checkpoints go; the final one and the metrics stay."""
    kernel = load_kernel(monkeypatch, tmp_path, segment=0)
    out = tmp_path / "run"
    out.mkdir()
    for name in ("checkpoint-005000.pt", "checkpoint-010000.pt", "checkpoint-final.pt"):
        (out / name).write_bytes(b"")
    (out / "metrics.jsonl").write_text("")
    kernel.drop_numbered_checkpoints(out)
    assert sorted(path.name for path in out.iterdir()) == ["checkpoint-final.pt", "metrics.jsonl"]


def test_every_switch_the_kernel_passes_is_an_option_of_train_e2e(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The training command parses with every stamp switched on.

    The kernel builds `mlime train e2e`'s argv by hand; an option it passes
    that the command lacks otherwise surfaces only on Kaggle, after the
    install, as a usage error from every rank.
    """
    import typer
    from typer.core import TyperGroup

    from mlime.__main__ import app

    kernel = load_kernel(monkeypatch, tmp_path, segment=0)
    monkeypatch.setattr(
        kernel, "RANKING", {"segmentations": 8, "beam": 4, "negatives": 8, "weight": 1.0}
    )
    monkeypatch.setattr(kernel, "INIT_FROM", "mlime-e2e-final")
    monkeypatch.setattr(kernel, "WHOLE_ALPHABET", True)
    monkeypatch.setattr(kernel, "LEARNING_RATES", (2e-5, 2e-5))
    monkeypatch.setattr(kernel, "TYPOS", True)
    monkeypatch.setattr(kernel, "ACTIVATION_CHECKPOINTING", True)
    monkeypatch.setattr(kernel, "SEGMENT_STEPS", 70)
    monkeypatch.setattr(kernel, "SEARCH_GRAPHS", True)
    monkeypatch.setattr(
        kernel,
        "CONTEXT_TOWER",
        {
            "tower": "Qwen/Qwen3-0.6B-Base",
            "revision": "abc",
            "lora_rank": 0,
            "lora_alpha": 16.0,
            "layers": 18,
        },
    )
    monkeypatch.setattr(kernel, "CONTEXT_WARM_UP", True)
    monkeypatch.setattr(kernel, "TOWER_GRAPHS", True)
    argv = kernel.train_argv(
        tmp_path / "char_pinyin.tsv",
        tmp_path / "run",
        max_steps=100,
        wall_budget=3600,
        start=kernel.start_argv(None, tmp_path / "checkpoint-final.pt"),
        resume=tmp_path / "checkpoint-paused.pt",
        token_budget=12288,
        trace_steps=5,
        segmenter=tmp_path / "ime-segment",
    )
    command = argv.index("mlime")
    assert argv[command + 1 : command + 3] == ["train", "e2e"]
    assert argv[argv.index("--context-layers") + 1] == "18"
    group = typer.main.get_command(app)
    assert isinstance(group, TyperGroup)
    train = group.commands["train"]
    assert isinstance(train, TyperGroup)
    train.commands["e2e"].make_context("e2e", [str(arg) for arg in argv[command + 3 :]])


def commit_repository(root: Path) -> tuple[Path, str]:
    """A repository with one commit holding `python/src/mlime/__init__.py`, and that commit."""
    import subprocess

    def git(*argv: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), "-c", "commit.gpgsign=false", *argv],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    (root / "python" / "src" / "mlime").mkdir(parents=True)
    (root / "python" / "src" / "mlime" / "__init__.py").write_text("")
    git("init", "--quiet", "-b", "dev")
    git("-c", "user.email=t@invalid", "-c", "user.name=t", "add", ".")
    git("-c", "user.email=t@invalid", "-c", "user.name=t", "commit", "--quiet", "-m", "one")
    return root, git("rev-parse", "HEAD")


def test_the_leg_fetches_its_commit_and_refuses_any_other_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kernel = load_kernel(monkeypatch, tmp_path / "input", segment=0)
    origin, commit = commit_repository(tmp_path / "origin")
    checkout = tmp_path / "checkout"
    assert kernel.checkout(commit, checkout, str(origin)) == checkout
    assert (checkout / "python" / "src" / "mlime" / "__init__.py").is_file()
    # A second call verifies the checkout already there.
    assert kernel.checkout(commit, checkout, str(origin)) == checkout

    (checkout / "python" / "src" / "mlime" / "__init__.py").write_text("changed = True\n")
    with pytest.raises(RuntimeError, match="changed files"):
        kernel.checkout(commit, checkout, str(origin))
    with pytest.raises(RuntimeError, match=f"not the leg's {'0' * 40}"):
        kernel.checkout("0" * 40, checkout, str(origin))
    with pytest.raises(RuntimeError, match="no COMMIT is stamped"):
        kernel.checkout(None, checkout, str(origin))

    monkeypatch.setattr(kernel, "COMMIT", commit)
    monkeypatch.setattr(kernel, "CHECKOUT", tmp_path / "fresh")
    monkeypatch.setattr(kernel, "REPOSITORY", str(origin))
    monkeypatch.setattr(kernel.checkout, "__defaults__", (tmp_path / "fresh", str(origin)))
    assert kernel.package_root() == tmp_path / "fresh" / "python" / "src"


def test_a_resume_refuses_a_commit_the_run_did_not_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kernel = load_kernel(monkeypatch, tmp_path, segment=1)
    monkeypatch.setattr(kernel, "COMMIT", "b" * 40)
    kernel.refuse_commit_change({"commit": "b" * 40})
    with pytest.raises(RuntimeError, match=f"recorded commit {'a' * 40}"):
        kernel.refuse_commit_change({"commit": "a" * 40})
    # v1's run-config records no commit: that is a change like any other.
    with pytest.raises(RuntimeError, match="recorded commit None"):
        kernel.refuse_commit_change({"max_steps": 10})
    monkeypatch.setattr(kernel, "ALLOW_COMMIT_CHANGE", True)
    kernel.refuse_commit_change({"commit": "a" * 40})


def test_trios_of_one_run_on_two_allowed_commits_still_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The commit is checked against the trio picked, not between candidates."""
    kernel = load_kernel(monkeypatch, tmp_path, segment=3)
    older = trio(tmp_path / "mlime-e2e-v2-s1", step=10)
    newer = trio(tmp_path / "mlime-e2e-v2-resume", step=20)
    for mount, commit in ((older, "a" * 40), (newer, "b" * 40)):
        config = json.loads((mount / "run-config.json").read_text())
        (mount / "run-config.json").write_text(json.dumps({**config, "commit": commit}))
    picked = kernel.previous_segment()
    assert picked is not None and picked["mount"] == newer


def test_the_kernels_count_builds_on_every_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`count-batches` as the kernel runs it counts with one worker per usable core.

    The kernel passes no `--workers`, so the command's default is what sizes
    the census to the card's VM: twelve processes on an A100's, two on a T4's.
    """
    from typer.testing import CliRunner

    import mlime.__main__ as cli
    import mlime.train.count as count
    from mlime.train.run import Slices, Vocabularies

    kernel = load_kernel(monkeypatch, tmp_path, segment=0)
    argv = kernel.count_argv(tmp_path / "char_pinyin.tsv", tmp_path / "batch-counts.json", None)
    command = argv.index("mlime")
    assert argv[command + 1 : command + 3] == ["train", "count-batches"]
    assert "--workers" not in argv

    asked: dict[str, object] = {}

    class CountedError(Exception):
        pass

    def counting(*args: object, **kwargs: object) -> None:
        asked.update(kwargs)
        raise CountedError

    monkeypatch.setattr(count, "count_batches", counting)
    monkeypatch.setattr(Vocabularies, "load", classmethod(lambda cls, *_: None))
    monkeypatch.setattr(cli, "route_a_slices", lambda *_: Slices(train=("a",), held_out=()))
    (tmp_path / "char_pinyin.tsv").touch()
    result = CliRunner().invoke(cli.app, [str(arg) for arg in argv[command + 1 :]])
    assert isinstance(result.exception, CountedError), result.output
    assert asked["workers"] == os.cpu_count()
    assert asked["world_size"] == 2


def test_a_ranking_stage_replays_its_search_from_graphs_when_stamped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the trainer has a search; the count shares the ranking switches and refuses it."""
    kernel = load_kernel(monkeypatch, tmp_path, segment=0)
    monkeypatch.setattr(kernel, "RANKING", {"weight": 0.05})
    monkeypatch.setattr(kernel, "INIT_FROM", "mlime-e2e-v2-g4")
    monkeypatch.setattr(kernel, "SEARCH_GRAPHS", True)
    segmenter = tmp_path / "ime-segment"
    count = kernel.count_argv(tmp_path / "char_pinyin.tsv", tmp_path / "counts.json", segmenter)
    assert "--ranking" in count
    assert "--search-graphs" not in count
    train = kernel.train_argv(
        tmp_path / "char_pinyin.tsv",
        tmp_path / "run",
        max_steps=100,
        wall_budget=3600,
        start=kernel.start_argv(None, tmp_path / "checkpoint-final.pt"),
        resume=None,
        token_budget=12288,
        segmenter=segmenter,
    )
    assert "--search-graphs" in train


def test_the_segmenter_is_found_in_the_directory_named_after_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Colab driver mounts `ime-segment` at `<inputs>/ime-segment/ime-segment`."""
    kernel = load_kernel(monkeypatch, tmp_path / "input", segment=0)
    mount = tmp_path / "input" / "ime-segment"
    mount.mkdir(parents=True)
    (mount / "ime-segment").write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr(kernel, "WORKING", tmp_path / "working")
    staged = kernel.stage_segmenter()
    assert staged == tmp_path / "working" / "bin" / "ime-segment"
    assert staged.read_bytes() == b"#!/bin/sh\n"


def test_the_window_start_reaches_the_trainer_from_the_stamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`mlime e2e stamp --profile-start` lands in the kernel, its argv, and the trainer's config."""
    from typer.testing import CliRunner

    import mlime.train.e2e
    from mlime.__main__ import app
    from mlime.train.loop import TrainingConfig

    out = tmp_path / "kernel"
    argv = ["e2e", "stamp", "mlime-e2e-v2-g4-trace", "--out", str(out)]
    argv += ["--profile-steps", "30", "--profile-start", "400", "--trace-steps", "3"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert "  PROFILE_START = 400" in result.output

    spec = importlib.util.spec_from_file_location("stamped_kernel", out / "kernel.py")
    assert spec is not None and spec.loader is not None
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    assert (kernel.PROFILE_STEPS, kernel.PROFILE_START, kernel.TRACE_STEPS) == (30, 400, 3)
    monkeypatch.setattr(kernel, "TRAIN_SHARDS", ("news-00000.parquet",))
    command = kernel.train_argv(
        tmp_path / "char_pinyin.tsv",
        tmp_path / "run",
        max_steps=100,
        wall_budget=None,
        start=None,
        resume=None,
        token_budget=12288,
        profile_steps=kernel.PROFILE_STEPS,
        profile_start=kernel.PROFILE_START,
        trace_steps=kernel.TRACE_STEPS,
    )

    seen: list[TrainingConfig] = []

    class ReachedError(Exception):
        """The trainer was called; the test has what it came for."""

    def trainer(*args: object, **_: object) -> None:
        training = args[2]
        assert isinstance(training, TrainingConfig)
        seen.append(training)
        raise ReachedError

    monkeypatch.setattr(mlime.train.e2e, "e2e", trainer)
    mlime_at = command.index("mlime")
    result = CliRunner().invoke(app, [str(arg) for arg in command[mlime_at + 1 :]])
    assert isinstance(result.exception, ReachedError), result.output
    (training,) = seen
    assert (training.profile_steps, training.profile_start, training.trace_steps) == (30, 400, 3)
