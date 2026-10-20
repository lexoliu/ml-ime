"""A Colab leg's probe and its waiter, against a scratch VM layout and a fake clock.

colab/leg.py runs on the VM and reads `/proc` and `/kaggle/working`; here both
are directories of the test's, with each process's `stat` line written the
way Linux writes it. colab/control.py runs on the Mac; its waiter is driven
with canned probe reports and a fake clock, and once through a fake `colab`
executable to cover the real `colab exec` round trip.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

COLAB = Path(__file__).resolve().parents[2] / "colab"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), COLAB / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


leg = load("leg")
control = load("control")


class Vm:
    """A working directory and a /proc of the test's, for one leg."""

    def __init__(self, root: Path) -> None:
        self.working = root / "working"
        self.proc = root / "proc"
        self.working.mkdir(parents=True)
        self.proc.mkdir(parents=True)
        self.pids = iter(range(100, 200))

    def process(self, pid_file: str, state: str | None) -> None:
        """A launched process in *state* (`R`, `S`, `Z`), or one gone from the table (None)."""
        pid = next(self.pids)
        (self.working / pid_file).write_text(f"{pid}\n")
        if state is not None:
            (self.proc / str(pid)).mkdir()
            # The command can hold spaces and parentheses; the state follows the last ')'.
            (self.proc / str(pid) / "stat").write_text(f"{pid} (python3 (x) y) {state} 1 2 3\n")

    def options(self, no_publish: bool = False) -> None:
        (self.working / "leg.json").write_text(json.dumps({"no_publish": no_publish}))

    def trio(self, directory: Path, step: int, summary: dict[str, Any] | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "checkpoint-paused.pt").write_bytes(b"")
        (directory / "run-config.json").write_text("{}")
        (directory / "run-summary.json").write_text(json.dumps(summary or {"step": step}))

    def interval(self, step: int) -> None:
        self.trio(self.working / "run" / f"interval-{step:06d}", step)

    def summary(self, **fields: Any) -> None:
        (self.working / "run-summary.json").write_text(json.dumps(fields))

    def paused(self, step: int) -> None:
        self.trio(self.working, step, {"finished": False, "checkpoint_step": step})

    def published(self, step: int, error: str | None = None) -> None:
        state: dict[str, Any] = {"last_published_step": step, "last_published_at": 1.0}
        if error is not None:
            state["last_error"] = error
        (self.working / "publish-state.json").write_text(json.dumps(state))

    def report(self) -> dict[str, Any]:
        result: dict[str, Any] = leg.report(self.working, self.proc)
        return result


@pytest.fixture
def vm(tmp_path: Path) -> Vm:
    return Vm(tmp_path)


def test_a_leg_still_staging_or_whose_staging_died(vm: Vm) -> None:
    vm.options()
    vm.process("stage.pid", "S")
    assert vm.report()["outcome"] == "staging"
    vm.process("stage.pid", "Z")
    report = vm.report()
    assert (report["outcome"], report["refusal"]) == ("failed", None)
    assert "stage.out" in report["reason"]


def test_a_training_leg_refuses_release_until_its_newest_interval_is_published(vm: Vm) -> None:
    vm.options()
    vm.process("stage.pid", None)
    vm.process("kernel.pid", "R")
    vm.process("publish.pid", "S")
    (vm.working / "run").mkdir()
    (vm.working / "run" / "metrics.jsonl").write_text(
        "".join(json.dumps({"event": "step", "step": s}) + "\n" for s in (10, 20))
        + '{"event": "step", "st'
    )
    report = vm.report()
    assert report["outcome"] == "training"
    assert report["facts"]["last_step"] == 20
    assert report["refusal"] is None

    vm.interval(15)
    # A trio still being staged under its hidden name is not one.
    (vm.working / "run" / ".interval-000020.tmp").mkdir()
    assert vm.report()["facts"]["newest_trio"] == 15
    assert "step 15 is not published" in vm.report()["refusal"]
    vm.published(15)
    assert vm.report()["refusal"] is None


def test_a_paused_leg_has_ended_once_its_paused_trio_is_published(vm: Vm) -> None:
    vm.options()
    vm.process("stage.pid", "Z")
    vm.process("kernel.pid", "Z")
    vm.process("publish.pid", "S")
    vm.interval(40)
    vm.published(40)
    vm.paused(52)
    report = vm.report()
    assert report["outcome"] == "publishing"
    assert report["facts"]["newest_trio"] == 52
    assert "step 52 is not published" in report["refusal"]

    vm.published(52)
    report = vm.report()
    assert (report["outcome"], report["refusal"]) == ("paused", None)


def test_a_publisher_that_died_with_the_paused_trio_unpublished_fails_the_leg(vm: Vm) -> None:
    vm.options()
    vm.process("kernel.pid", None)
    vm.process("publish.pid", None)
    vm.paused(52)
    vm.published(40, error="503 Service Unavailable")
    report = vm.report()
    assert report["outcome"] == "failed"
    assert "step 52 unpublished" in report["reason"]
    assert "503" in report["refusal"]


def test_a_check_leg_publishes_nothing_and_may_always_be_released(vm: Vm, tmp_path: Path) -> None:
    vm.options(no_publish=True)
    vm.process("kernel.pid", "Z")
    vm.interval(20)
    vm.paused(30)
    report = vm.report()
    assert (report["outcome"], report["refusal"]) == ("paused", None)

    short = Vm(tmp_path / "short")
    short.options(no_publish=True)
    short.process("kernel.pid", "Z")
    short.interval(20)
    short.summary(finished=True)
    report = short.report()
    assert (report["outcome"], report["refusal"]) == ("finished", None)


def test_a_kernel_that_died_without_a_summary_failed(vm: Vm) -> None:
    vm.options()
    vm.process("kernel.pid", None)
    report = vm.report()
    assert report["outcome"] == "failed"
    assert "run-summary.json" in report["reason"]


def test_a_finished_run_has_ended_only_once_its_outputs_are_in_the_bucket(vm: Vm) -> None:
    vm.options()
    vm.process("kernel.pid", "Z")
    vm.process("publish.pid", "S")
    vm.interval(100)
    vm.published(100)
    vm.summary(finished=True, checkpoint_step=120)
    report = vm.report()
    assert report["outcome"] == "publishing"
    assert "final checkpoint" in report["refusal"]

    state = json.loads((vm.working / "publish-state.json").read_text())
    state["final_published"] = True
    (vm.working / "publish-state.json").write_text(json.dumps(state))
    report = vm.report()
    assert (report["outcome"], report["refusal"]) == ("finished", None)


def test_a_publisher_that_died_with_the_final_outputs_unpublished_fails_the_leg(vm: Vm) -> None:
    vm.options()
    vm.process("kernel.pid", "Z")
    vm.process("publish.pid", None)
    vm.published(100, error="503 Service Unavailable")
    vm.summary(finished=True, checkpoint_step=120)
    report = vm.report()
    assert report["outcome"] == "failed"
    assert "finished run's outputs unpublished" in report["reason"]
    assert "503" in report["refusal"]


def test_the_probe_beats_the_watchdog_before_it_reports(
    vm: Vm, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    vm.options()
    vm.process("kernel.pid", "R")
    heartbeat = vm.working.parent / "heartbeat"
    monkeypatch.setattr(leg, "HEARTBEAT", heartbeat)
    monkeypatch.setattr(leg, "WORKING", vm.working)
    monkeypatch.setattr(leg, "PROC", vm.proc)
    monkeypatch.setattr(leg.report, "__defaults__", (vm.working, vm.proc))
    leg.main()
    assert heartbeat.is_file()
    [line] = capsys.readouterr().out.splitlines()
    assert line.startswith("LEG ")
    assert json.loads(line[4:])["outcome"] == "training"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def reported(outcome: str, reason: str | None = None) -> dict[str, Any]:
    known = {"last_step": 1, "newest_trio": None, "published": None}
    return {"facts": known, "outcome": outcome, "reason": reason, "refusal": None}


def waits(
    answers: list[dict[str, Any] | None], deadline: float, interval: float = 60
) -> tuple[int, Clock, int]:
    clock = Clock()
    asked = iter(answers)
    calls = 0

    def ask(session: str) -> dict[str, Any] | None:
        nonlocal calls
        calls += 1
        clock.now += 5  # a probe's own round trip
        return next(asked)

    code = control.wait("s", deadline, interval, ask, clock.time, clock.sleep)
    return code, clock, calls


@pytest.mark.parametrize(("outcome", "code"), [("paused", 10), ("finished", 11), ("failed", 12)])
def test_the_waiter_exits_with_the_code_of_the_legs_end(outcome: str, code: int) -> None:
    answers = [reported("staging"), reported("training"), reported("publishing"), reported(outcome)]
    result, clock, calls = waits(answers, deadline=3600)
    assert (result, calls) == (code, 4)
    # Every probe is a beat: none is further than the interval from the last.
    assert clock.sleeps == [60, 60, 60]


def test_the_waiter_gives_up_at_its_deadline() -> None:
    result, clock, calls = waits([reported("training")] * 100, deadline=200)
    assert result == control.DEADLINE
    assert calls == 4
    assert clock.now <= 200 + 5


def test_a_vm_unanswered_for_the_watchdogs_timeout_is_lost() -> None:
    answers: list[dict[str, Any] | None] = [reported("training"), *[None] * 100]
    result, clock, _ = waits(answers, deadline=24 * 3600, interval=120)
    assert result == control.LOST
    assert 600 <= clock.now <= 600 + 120 + 10


def test_the_waiter_never_beats_less_often_than_the_watchdog_needs() -> None:
    with pytest.raises(ValueError, match="interval"):
        control.wait("s", 3600, 121, lambda _: None)
    watchdog = (COLAB / "watchdog.py").read_text()
    assert "TIMEOUT_SECONDS = 10 * 60" in watchdog
    assert control.LOST_AFTER_SECONDS == 10 * 60


def test_release_is_refused_with_the_probes_reason() -> None:
    refused = {**reported("training"), "refusal": "the trio at step 15 is not published yet"}
    assert control.releasable("s", lambda _: refused) == control.REFUSED
    assert control.releasable("s", lambda _: reported("paused")) == 0
    assert control.releasable("s", lambda _: None) == control.UNREACHABLE


def fake_colab(directory: Path, answers: list[str]) -> Path:
    """A `colab` executable that logs each call and answers `exec` with the next line."""
    directory.mkdir()
    (directory / "answers.json").write_text(json.dumps(answers))
    script = directory / "colab"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "here = Path(__file__).parent\n"
        "calls = here / 'calls.jsonl'\n"
        "stdin = sys.stdin.read() if sys.argv[1] == 'exec' else ''\n"
        "with calls.open('a') as log:\n"
        "    log.write(json.dumps({'argv': sys.argv[1:], 'stdin': stdin}) + '\\n')\n"
        "if sys.argv[1] == 'exec':\n"
        "    answers = json.loads((here / 'answers.json').read_text())\n"
        "    count = sum(1 for _ in calls.open()) - 1\n"
        "    print(answers[min(count, len(answers) - 1)])\n"
    )
    script.chmod(0o755)
    return directory


def test_the_waiter_probes_the_vm_through_colab_exec(tmp_path: Path) -> None:
    fake = fake_colab(
        tmp_path / "bin",
        [
            "Traceback: the VM is busy",
            "LEG " + json.dumps(reported("training")),
            "LEG " + json.dumps(reported("paused")),
        ],
    )
    done = subprocess.run(
        [
            sys.executable,
            str(COLAB / "control.py"),
            "wait",
            "mlime-e2e-v2-t4",
            "--deadline",
            "60",
            "--interval",
            "0.05",
        ],
        env={**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 10, done.stdout + done.stderr
    calls = [json.loads(line) for line in (fake / "calls.jsonl").read_text().splitlines()]
    assert [call["argv"] for call in calls] == [["exec", "-s", "mlime-e2e-v2-t4"]] * 3
    # The probe runs the checkout's leg.py, which beats before it reports.
    assert calls[0]["stdin"] == control.PROBE
    assert f"{control.CHECKOUT}/colab/leg.py" in control.PROBE
