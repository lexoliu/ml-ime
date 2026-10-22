"""The route A kernel's rescoring mode: which run it scores with, and how the
lattices are spread over the GPUs.

kaggle/route-a-v2/kernel.py is a kernel script rather than a package module,
so it is imported by path; INPUTS, WORKING and RESCORE_FROM stand in for the
mount layout and the stamp a push applies.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

KERNEL_PATH = Path(__file__).resolve().parents[2] / "kaggle" / "route-a-v2" / "kernel.py"
SOURCE = "mlime-route-a-v2-s6"


def load_kernel(monkeypatch: pytest.MonkeyPatch, inputs: Path, working: Path) -> ModuleType:
    """kernel.py as a module, with its mounts, its output and its stamp redirected."""
    spec = importlib.util.spec_from_file_location("route_a_kernel", KERNEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "INPUTS", inputs)
    monkeypatch.setattr(module, "WORKING", working)
    monkeypatch.setattr(module, "RESCORE_FROM", SOURCE)
    return module


def mounted_run(inputs: Path, name: str, finished: bool) -> Path:
    """A finished (or paused) segment's output, mounted the way Kaggle mounts one."""
    root = inputs / "notebooks" / "lexoliu" / name
    (root / "run").mkdir(parents=True)
    (root / "run" / "checkpoint-final.pt").write_bytes(b"")
    (root / "run-summary.json").write_text(json.dumps({"finished": finished, "max_steps": 7}))
    return root


def lattices(assets: Path) -> None:
    """Three lattices whose sizes order them full < mixed < abbreviated."""
    assets.mkdir(parents=True)
    for name, size in (("lattice", 1), ("lattice-mixed", 5), ("lattice-abbreviated", 7)):
        (assets / f"{name}.jsonl").write_bytes(b"x" * size)


def test_the_finished_run_named_by_the_stamp_is_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = mounted_run(tmp_path / "input", SOURCE, finished=True)
    kernel = load_kernel(monkeypatch, tmp_path / "input", tmp_path / "working")
    checkpoint, summary = kernel.finished_run()
    assert checkpoint == root / "run" / "checkpoint-final.pt"
    assert summary["max_steps"] == 7


def test_an_unfinished_or_other_run_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mounted_run(tmp_path / "paused", SOURCE, finished=False)
    kernel = load_kernel(monkeypatch, tmp_path / "paused", tmp_path / "working")
    with pytest.raises(RuntimeError, match="did not finish"):
        kernel.finished_run()
    mounted_run(tmp_path / "other", "mlime-route-a-v2-s5", finished=True)
    kernel = load_kernel(monkeypatch, tmp_path / "other", tmp_path / "working")
    with pytest.raises(FileNotFoundError, match="not mlime-route-a-v2-s6's"):
        kernel.finished_run()


def test_every_pair_is_scored_once_largest_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assets = tmp_path / "assets"
    lattices(assets)
    working = tmp_path / "working"
    working.mkdir()
    kernel = load_kernel(monkeypatch, tmp_path / "input", working)
    order = [
        (lattice.stem, context)
        for lattice, context in kernel.scoring_jobs(sorted(assets.glob("lattice*.jsonl")))
    ]
    assert order == [
        ("lattice-abbreviated", True),
        ("lattice-abbreviated", False),
        ("lattice-mixed", True),
        ("lattice-mixed", False),
        ("lattice", True),
        ("lattice", False),
    ]

    devices: dict[str, str] = {}

    def fake_run(argv: list[str], env: dict[str, str], log: str) -> tuple[int, float]:
        out = Path(argv[argv.index("--out") + 1])
        out.write_bytes(b"scores")
        devices[out.name] = env["CUDA_VISIBLE_DEVICES"]
        return 0, 1.0

    monkeypatch.setattr(kernel, "run", fake_run)
    scored = kernel.score_lattices(Path("final.pt"), assets, assets / "t.tsv", {}, 2)
    assert sorted(scored) == sorted(
        f"{stem}-context-{'on' if context else 'off'}" for stem, context in order
    )
    assert {entry["gpu"] for entry in scored.values()} <= {0, 1}
    assert {name: str(entry["gpu"]) for name, entry in scored.items()} == {
        name.removeprefix("scores-").removesuffix(".jsonl.gz"): device
        for name, device in devices.items()
    }


def test_a_failed_scoring_fails_the_kernel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    lattices(assets)
    kernel = load_kernel(monkeypatch, tmp_path / "input", tmp_path / "working")
    monkeypatch.setattr(kernel, "run", lambda argv, env, log: (1, 1.0))
    with pytest.raises(RuntimeError, match="failed"):
        kernel.score_lattices(Path("final.pt"), assets, assets / "t.tsv", {}, 2)
