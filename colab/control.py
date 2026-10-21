"""Watch a Colab leg from the Mac until it ends, and say whether it may be released.

    python3 control.py wait SESSION --deadline SECONDS [--interval SECONDS]
    python3 control.py releasable SESSION

Both run `leg.py` on the VM, from the leg's checkout, through `colab exec`;
every such probe touches the watchdog's heartbeat before it reports. `wait`
probes every `--interval` seconds (at most two minutes, so the watchdog's
ten-minute timeout never fires under a live waiter) and exits when the leg has
ended, with the code of its end:

    10  paused   -- the kernel paused and its paused trio is published
                    (or the leg publishes nothing): stop the runtime
    11  finished -- the run reached max_steps and was evaluated
    12  failed   -- the stager, the kernel or the publisher ended wrongly;
                    the reason is printed
    13  deadline -- the leg was still running when --deadline passed
    14  lost     -- no probe answered for the watchdog's timeout: the runtime
                    has released itself, or cannot be reached

Codes 0 and 1 are left to a waiter that ran or crashed, so an outcome is
never mistaken for either. `releasable` exits 0 when releasing the runtime
loses nothing, 3 with the reason when it would, and 4 when the VM cannot be
reached to ask.

Standard library only, Python 3.9: it runs under the Mac's system Python.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

#: Where the VM checks out the leg's commit (the kernel's `CHECKOUT`).
CHECKOUT = "/tmp/ml-ime"
#: Bound to a name: IPython displays a cell's last value, and printing leg.py's
#: globals raises inside IPython's printer on the runtime's Python.
PROBE = f'import runpy\nleg = runpy.run_path("{CHECKOUT}/colab/leg.py", run_name="__main__")\n'
ENDS = {"paused": 10, "finished": 11, "failed": 12}
DEADLINE = 13
LOST = 14
REFUSED = 3
UNREACHABLE = 4
#: The watchdog's `TIMEOUT_SECONDS`: a runtime nobody probed for this long
#: has released itself.
LOST_AFTER_SECONDS = 10 * 60
#: The longest a waiter may go between beats: five of them fit the timeout.
MAX_INTERVAL_SECONDS = 2 * 60
#: How long one `colab exec` may take before it counts as unanswered.
PROBE_SECONDS = 90


def say(text: str) -> None:
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M:%S")
    print(f"{stamp} {text}", flush=True)


def probe(session: str) -> Optional[dict[str, Any]]:
    """The VM's `leg.py` report, or None when the probe did not answer."""
    try:
        done = subprocess.run(
            # The CLI ends a cell after its own timeout, 30 s unless told.
            ["colab", "exec", "-s", session, "--timeout", str(PROBE_SECONDS)],
            input=PROBE,
            capture_output=True,
            text=True,
            timeout=PROBE_SECONDS,
        )
    except subprocess.TimeoutExpired:
        say(f"probe of {session} took over {PROBE_SECONDS}s")
        return None
    answers = [line for line in done.stdout.splitlines() if line.startswith("LEG ")]
    if done.returncode != 0 or not answers:
        say(
            f"probe of {session} failed (exit {done.returncode}): {(done.stdout + done.stderr)[-1500:]}"
        )
        return None
    report: dict[str, Any] = json.loads(answers[-1][len("LEG ") :])
    return report


def progress(report: dict[str, Any]) -> str:
    known = report["facts"]
    return (
        f"{report['outcome']}: step {known['last_step']}, newest trio {known['newest_trio']}, "
        f"published {known['published']}"
    )


def wait(
    session: str,
    deadline: float,
    interval: float,
    ask: Callable[[str], Optional[dict[str, Any]]] = probe,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Probe until the leg ends, the deadline passes or the VM is lost; the exit code."""
    if not 0 < interval <= MAX_INTERVAL_SECONDS:
        raise ValueError(f"--interval must be in (0, {MAX_INTERVAL_SECONDS}], got {interval}")
    if deadline <= 0:
        raise ValueError(f"--deadline must be positive, got {deadline}")
    started = clock()
    answered = started
    while True:
        report = ask(session)
        now = clock()
        if report is not None:
            answered = now
            say(progress(report))
            if report["outcome"] in ENDS:
                if report["reason"] is not None:
                    say(report["reason"])
                return ENDS[report["outcome"]]
        elif now - answered >= LOST_AFTER_SECONDS:
            say(f"no probe answered for {now - answered:.0f}s; the runtime is lost")
            return LOST
        if now - started >= deadline:
            say(f"the deadline of {deadline:.0f}s passed with the leg still running")
            return DEADLINE
        sleep(min(interval, started + deadline - now))


def releasable(session: str, ask: Callable[[str], Optional[dict[str, Any]]] = probe) -> int:
    """0 when releasing the runtime loses nothing; the refusal's code otherwise."""
    report = ask(session)
    if report is None:
        say(f"{session} did not answer; nothing says releasing it is safe")
        return UNREACHABLE
    if report["refusal"] is not None:
        say(f"refusing to release {session}: {report['refusal']}")
        return REFUSED
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    waiting = commands.add_parser("wait", help="Probe the leg until it ends")
    waiting.add_argument("session")
    waiting.add_argument("--deadline", type=float, required=True, help="Seconds to wait at most")
    waiting.add_argument("--interval", type=float, default=60.0, help="Seconds between probes")
    releasing = commands.add_parser("releasable", help="Whether stopping loses nothing")
    releasing.add_argument("session")
    options = parser.parse_args(argv)
    if options.command == "wait":
        return wait(options.session, options.deadline, options.interval)
    return releasable(options.session)


if __name__ == "__main__":
    sys.exit(main())
