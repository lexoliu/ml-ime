"""colab/e2e.sh end to end, against a fake `colab`, a fake `cargo` and a scratch repository.

The driver's checks run on the Mac before a runtime bills, so they are
exercised for real: a copy of `colab/` is committed to a scratch repository
with a bare origin, and `e2e.sh` runs from it. `colab` and `cargo` are fakes
on PATH that log every call -- `colab exec` with the Python it was fed -- and
answer the probe from a canned list.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

FAKE_COLAB = """\
import json, re, sys, time
from pathlib import Path
here = Path(__file__).parent
calls = here / "calls.jsonl"
stdin = sys.stdin.read() if sys.argv[1] == "exec" else ""
with calls.open("a") as log:
    log.write(json.dumps({"argv": sys.argv[1:], "stdin": stdin}) + "\\n")
hang = here / "hang-exec"
if sys.argv[1] == "exec" and hang.is_file() and hang.read_text() in stdin:
    time.sleep(3600)  # the CLI after a dropped connection
answers = here / "answers.json"
token = re.search(r'print\\("(remote-ok-[0-9]+)"\\)\\s*$', stdin)
fail = here / "fail-exec"
if token:
    # Like the real CLI: the kernel prints the traceback and the exit is 0.
    if fail.is_file() and fail.read_text() in stdin:
        print("RuntimeError: the code raised")
    else:
        print(token.group(1))
elif sys.argv[1] == "exec" and answers.is_file():
    print(json.loads(answers.read_text()).pop(0))
"""

FAKE_CARGO = """\
import json, sys
from pathlib import Path
here = Path(__file__).parent
(here / "cargo.json").write_text(json.dumps(sys.argv[1:]))
binary = here / "ime-segment"
binary.write_bytes(b"ELF")
for name, path in (("ime_pinyin", None), ("ime-segment", str(binary))):
    artifact = {"reason": "compiler-artifact", "target": {"name": name}, "executable": path}
    print(json.dumps(artifact))
print(json.dumps({"reason": "build-finished", "success": True}))
"""


def git(repo: Path, *argv: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *argv],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class Driver:
    """A scratch checkout of the driver, its origin, a stamped kernel dir and the fakes."""

    def __init__(self, root: Path) -> None:
        self.origin = root / "origin.git"
        subprocess.run(["git", "init", "--quiet", "--bare", str(self.origin)], check=True)
        self.repo = root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "--quiet", "-b", "dev")
        git(self.repo, "config", "user.email", "test@invalid")
        git(self.repo, "config", "user.name", "test")
        shutil.copytree(ROOT / "colab", self.repo / "colab")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "--quiet", "-m", "driver")
        git(self.repo, "remote", "add", "origin", str(self.origin))
        git(self.repo, "push", "--quiet", "origin", "dev")
        self.kernel = root / "run"
        self.kernel.mkdir()
        shutil.copy(ROOT / "kaggle" / "e2e" / "kernel.py", self.kernel / "kernel.py")
        (self.kernel / "kernel-metadata.json").write_text(
            json.dumps({"id": "lexoliu/mlime-e2e-v2-t4-s0", "dataset_sources": []})
        )
        self.bin = root / "bin"
        self.bin.mkdir()
        for name, source in (("colab", FAKE_COLAB), ("cargo", FAKE_CARGO)):
            (self.bin / name).write_text(f"#!{sys.executable}\n{source}")
            (self.bin / name).chmod(0o755)
        self.home = root / "home"
        for credential in (".kaggle/credentials.json", ".cache/huggingface/token"):
            (self.home / credential).parent.mkdir(parents=True)
            (self.home / credential).write_text("secret")

    def run(self, *argv: str, colab_seconds: int | None = None) -> subprocess.CompletedProcess[str]:
        bound = {} if colab_seconds is None else {"COLAB_SECONDS": str(colab_seconds)}
        return subprocess.run(
            ["bash", str(self.repo / "colab" / "e2e.sh"), str(self.kernel), *argv],
            env={
                **os.environ,
                "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                "HOME": str(self.home),
                **bound,
            },
            capture_output=True,
            text=True,
            timeout=120,
        )

    def calls(self) -> list[dict[str, Any]]:
        log = self.bin / "calls.jsonl"
        if not log.is_file():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def answer(self, *reports: dict[str, Any]) -> None:
        (self.bin / "answers.json").write_text(
            json.dumps(["LEG " + json.dumps(r) for r in reports])
        )

    def fail_exec(self, marker: str) -> None:
        """The code of the `colab exec` holding *marker* raises on the runtime."""
        (self.bin / "fail-exec").write_text(marker)

    def hang_exec(self, marker: str) -> None:
        """The `colab exec` holding *marker* never returns."""
        (self.bin / "hang-exec").write_text(marker)

    def stamp_ranking(self) -> None:
        kernel = self.kernel / "kernel.py"
        text = kernel.read_text()
        ranking = 'RANKING = {"segmentations": 8, "beam": 4, "negatives": 8, "weight": 0.05}'
        kernel.write_text(text.replace("\nRANKING = None\n", f"\n{ranking}\n", 1))


@pytest.fixture
def driver(tmp_path: Path) -> Driver:
    return Driver(tmp_path)


def stage_argv(calls: list[dict[str, Any]]) -> list[str]:
    """The stager's argv inside the `colab exec` that launched it."""
    [launch] = [c for c in calls if "e2e-stage.py" in c["stdin"]]
    tree = ast.parse(launch["stdin"])
    [starred] = [node for node in ast.walk(tree) if isinstance(node, ast.Starred)]
    argv: list[str] = ast.literal_eval(starred.value)
    return argv


def test_a_start_without_a_session_length_refuses_before_a_runtime_bills(
    driver: Driver,
) -> None:
    done = driver.run("start", "--gpu", "T4")
    assert done.returncode == 2
    assert "--session-hours is required" in done.stderr
    assert driver.calls() == []


def test_a_dirty_checkout_or_an_unpushed_commit_is_refused(driver: Driver) -> None:
    (driver.repo / "colab" / "leg.py").write_text("# edited\n")
    done = driver.run("start", "--session-hours", "1")
    assert done.returncode == 2
    assert "uncommitted changes" in done.stderr

    git(driver.repo, "commit", "--quiet", "-am", "local only")
    done = driver.run("start", "--session-hours", "1")
    assert done.returncode == 2
    assert "on no branch of origin" in done.stderr
    assert driver.calls() == []


def test_a_v2_start_runs_its_commit_on_a_stamped_session_and_builds_no_segmenter(
    driver: Driver,
) -> None:
    commit = git(driver.repo, "rev-parse", "HEAD")
    done = driver.run("start", "--session-hours", "1.5", "--no-publish", "--gpu", "L4")
    assert done.returncode == 0, done.stderr
    calls = driver.calls()
    assert calls[0]["argv"] == ["sessions"]
    assert calls[1]["argv"] == ["new", "-s", "mlime-e2e-v2-t4", "--gpu", "L4"]
    uploads = {c["argv"][-1]: c["argv"][-2] for c in calls if c["argv"][0] == "upload"}
    assert uploads["/kaggle/kernel.py"] == str(driver.kernel / "kernel.py")
    assert not any("ime-segment" in target for target in uploads)
    assert not (driver.bin / "cargo.json").exists()
    # The kernel's own checkout fetches exactly the leg's commit.
    [fetch] = [c for c in calls if '["checkout"]' in c["stdin"]]
    assert f"[\"checkout\"]('{commit}')" in fetch["stdin"]
    assert stage_argv(calls) == ["--commit", commit, "--session-seconds", "5400", "--no-publish"]
    # The CLI's own exec timeout is 30 s unless told; every cell gets its bound.
    assert {tuple(c["argv"][3:]) for c in calls if c["argv"][0] == "exec"} == {("--timeout", "600")}


def test_a_longer_bound_reaches_the_cli_s_exec_timeout(driver: Driver) -> None:
    done = driver.run("start", "--session-hours", "1", "--no-publish", colab_seconds=3600)
    assert done.returncode == 0, done.stderr
    execs = [c["argv"] for c in driver.calls() if c["argv"][0] == "exec"]
    assert execs
    assert all(argv[3:] == ["--timeout", "3600"] for argv in execs)


def test_a_start_whose_code_raises_on_the_runtime_releases_it(driver: Driver) -> None:
    """`colab exec` exits 0 when the code raises; the start must still fail and stop."""
    driver.fail_exec('["checkout"]')
    done = driver.run("start", "--session-hours", "1", "--no-publish")
    assert done.returncode != 0
    assert "did not run to its end" in done.stderr
    argv = [c["argv"] for c in driver.calls()]
    assert argv[-1] == ["stop", "-s", "mlime-e2e-v2-t4"]
    # Nothing after the failed step reached the runtime.
    assert not any("e2e-stage.py" in c["stdin"] for c in driver.calls())


def test_a_watchdog_that_does_not_start_releases_the_runtime(driver: Driver) -> None:
    driver.fail_exec("/root/watchdog.py")
    done = driver.run("start", "--session-hours", "1")
    assert done.returncode != 0
    assert [c["argv"] for c in driver.calls()][-1] == ["stop", "-s", "mlime-e2e-v2-t4"]


def test_a_cli_call_that_never_returns_is_cut_off_and_releases_the_runtime(
    driver: Driver,
) -> None:
    """After a dropped connection the CLI can wait forever; its bound ends the call."""
    driver.hang_exec("/root/watchdog.py")
    done = driver.run("start", "--session-hours", "1", colab_seconds=2)
    assert done.returncode == 124, done.stderr
    assert [c["argv"] for c in driver.calls()][-1] == ["stop", "-s", "mlime-e2e-v2-t4"]


def test_a_ranking_start_ships_the_segmenter_built_from_its_commit(driver: Driver) -> None:
    driver.stamp_ranking()
    done = driver.run("start", "--session-hours", "2", "--resume-from", "mlime-e2e-rankw005")
    assert done.returncode == 0, done.stderr
    cargo = json.loads((driver.bin / "cargo.json").read_text())
    assert cargo[:2] == ["build", "--release"]
    assert cargo[cargo.index("--target") + 1] == "x86_64-unknown-linux-musl"
    assert cargo[cargo.index("--bin") + 1] == "ime-segment"
    uploads = {c["argv"][-1]: c["argv"][-2] for c in driver.calls() if c["argv"][0] == "upload"}
    assert uploads["/kaggle/input/ime-segment/ime-segment"] == str(driver.bin / "ime-segment")
    assert stage_argv(driver.calls())[-2:] == ["--resume-from", "mlime-e2e-rankw005"]


def test_a_ranking_start_on_another_commit_than_the_checkouts_is_refused(
    driver: Driver,
) -> None:
    first = git(driver.repo, "rev-parse", "HEAD")
    (driver.repo / "colab" / "note").write_text("second\n")
    git(driver.repo, "add", ".")
    git(driver.repo, "commit", "--quiet", "-m", "second")
    git(driver.repo, "push", "--quiet", "origin", "dev")
    driver.stamp_ranking()
    done = driver.run("start", "--session-hours", "2", "--commit", first)
    assert done.returncode == 2
    assert f"check out {first}" in done.stderr
    assert driver.calls() == []


def report(refusal: str | None) -> dict[str, Any]:
    known = {"last_step": 3, "newest_trio": 3, "published": None}
    return {"facts": known, "outcome": "training", "reason": None, "refusal": refusal}


def test_stop_refuses_while_the_newest_trio_is_unpublished_unless_forced(
    driver: Driver,
) -> None:
    driver.answer(report("the trio at step 3 is not published yet"))
    done = driver.run("stop")
    assert done.returncode == 3
    assert "not published" in done.stdout
    assert not any(c["argv"][0] == "stop" for c in driver.calls())

    done = driver.run("stop", "--force")
    assert done.returncode == 0, done.stderr
    assert driver.calls()[-1]["argv"] == ["stop", "-s", "mlime-e2e-v2-t4"]

    driver.answer(report(None))
    done = driver.run("stop")
    assert done.returncode == 0, done.stderr
    assert driver.calls()[-1]["argv"] == ["stop", "-s", "mlime-e2e-v2-t4"]


def test_fetch_brings_back_the_census_and_the_samplers(driver: Driver, tmp_path: Path) -> None:
    done = driver.run("fetch", str(tmp_path / "out"))
    assert done.returncode == 0, done.stderr
    fetched = {c["argv"][-2] for c in driver.calls() if c["argv"][0] == "download"}
    for name in (
        "batch-counts.json",
        "gpu.csv",
        "host.jsonl",
        "run/metrics.jsonl",
        "run/trace-rank0.json",
    ):
        assert f"/kaggle/working/{name}" in fetched


def test_the_vm_checkout_is_one_place() -> None:
    """The kernel, the driver and the controller name the same checkout on the VM."""
    kernel = (ROOT / "kaggle" / "e2e" / "kernel.py").read_text()
    driver = (ROOT / "colab" / "e2e.sh").read_text()
    control = (ROOT / "colab" / "control.py").read_text()
    assert 'CHECKOUT = Path("/tmp/ml-ime")' in kernel
    assert "\ncheckout=/tmp/ml-ime\n" in driver
    assert 'CHECKOUT = "/tmp/ml-ime"' in control
