"""Where a Colab leg stands, read on its VM: one probe beats the watchdog and reports.

`colab/control.py` runs this on the VM through `colab exec` from the leg's
checkout, for the waiter (`e2e.sh wait`) and for `e2e.sh stop`. Running it is
the controller's heartbeat: it touches the watchdog's file first, so a probe
that reaches the VM keeps the runtime held. It then prints one line, `LEG `
and a JSON object:

- `facts`: what the VM holds -- whether the stager, the kernel and the
  publisher are alive (read from `/proc` by the pids the launchers wrote; a
  zombie has ended), the kernel's `run-summary.json`, the newest whole trio on
  the VM and the newest one published (`publish-state.json`), the leg's
  options (`leg.json`) and the last logged step;
- `outcome`: `staging`, `training`, `publishing` (the kernel paused, its trio
  is still going up), or one of the leg's ends: `paused`, `finished`,
  `failed`, with `reason` saying why for a failure;
- `refusal`: why releasing the runtime now would lose work, or null.

The outcome comes from the processes and the files the kernel and publisher
write as structured state, never from a log's text. A paused leg has ended
only once its paused trio is published (or the leg publishes nothing), so a
waiter that sees `paused` can stop the runtime at once.

Standard library only, and Python 3.9: the same file is imported by the
controller on the Mac.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

WORKING = Path("/kaggle/working")
PROC = Path("/proc")
#: The watchdog's heartbeat (`colab/watchdog.py`).
HEARTBEAT = Path("/root/heartbeat")
RESUME_MARKERS = ("checkpoint-paused.pt", "run-config.json", "run-summary.json")
INTERVAL = re.compile(r"^interval-(\d+)$")
#: How much of the metrics file's tail is read for the last logged step.
METRICS_TAIL_BYTES = 1 << 16


def alive(pid_file: Path, proc: Path) -> Optional[bool]:
    """Whether the process whose pid *pid_file* holds is running; None before it was started.

    A detached child whose parent never waits stays in the table as a zombie
    once it exits, so a zombie (or a dead task) counts as ended.
    """
    if not pid_file.is_file():
        return None
    stat = proc / pid_file.read_text().strip() / "stat"
    try:
        text = stat.read_text()
    except FileNotFoundError:
        return False
    state = text[text.rindex(")") + 2 :].split()[0]
    return state not in ("Z", "X")


def read_json(path: Path) -> Optional[Any]:
    """The JSON *path* holds, or None when it does not exist."""
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def newest_trio(working: Path) -> Optional[int]:
    """The checkpoint step of the newest whole trio on the VM, or None for none.

    Interval trios land whole under `run/interval-<step>` by rename; the
    paused trio is whole once the kernel has written its summary beside the
    checkpoint it moved to the working root.
    """
    steps = [
        int(match.group(1))
        for directory in (working / "run").glob("interval-*")
        if (match := INTERVAL.match(directory.name))
        and all((directory / marker).is_file() for marker in RESUME_MARKERS)
    ]
    if all((working / marker).is_file() for marker in RESUME_MARKERS):
        steps.append(int(json.loads((working / "run-summary.json").read_text())["checkpoint_step"]))
    return max(steps, default=None)


def last_step(working: Path) -> Optional[int]:
    """The last step the trainer logged, from the tail of its metrics."""
    metrics = working / "run" / "metrics.jsonl"
    if not metrics.is_file():
        return None
    with metrics.open("rb") as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - METRICS_TAIL_BYTES))
        lines = handle.read().decode().split("\n")
    # The first line may be cut by the seek and the last may be half written.
    whole = lines[1:-1] if size > METRICS_TAIL_BYTES else lines[:-1]
    steps = [record for record in map(json.loads, whole) if record.get("event") == "step"]
    return steps[-1]["step"] if steps else None


def facts(working: Path = WORKING, proc: Path = PROC) -> dict[str, Any]:
    """What the VM holds about the leg."""
    publish = read_json(working / "publish-state.json") or {}
    return {
        "leg": read_json(working / "leg.json"),
        "stage": alive(working / "stage.pid", proc),
        "kernel": alive(working / "kernel.pid", proc),
        "publisher": alive(working / "publish.pid", proc),
        "summary": read_json(working / "run-summary.json"),
        "newest_trio": newest_trio(working),
        "published": publish.get("last_published_step"),
        "publish_error": publish.get("last_error"),
        "final_published": publish.get("final_published", False),
        "last_step": last_step(working),
    }


def publishes(known: dict[str, Any]) -> bool:
    """Whether this leg runs a publisher; refused when the stager never said."""
    if known["leg"] is None:
        raise ValueError("no leg.json on the VM: the stager never started")
    return not known["leg"]["no_publish"]


def outcome(known: dict[str, Any]) -> tuple[str, Optional[str]]:
    """The leg's state and, for a failure, why."""
    if known["kernel"] is None:
        if known["stage"]:
            return "staging", None
        return "failed", "staging ended without launching the kernel; stage.out says why"
    if known["kernel"]:
        return "training", None
    summary = known["summary"]
    if summary is None:
        return "failed", "the kernel ended without writing run-summary.json; kernel.out says why"
    if summary["finished"]:
        if not publishes(known) or known["final_published"]:
            return "finished", None
        if known["publisher"]:
            return "publishing", None
        return "failed", (
            "the publisher ended with the finished run's outputs unpublished "
            f"(last error {known['publish_error']})"
        )
    step = summary["checkpoint_step"]
    if not publishes(known) or (known["published"] is not None and known["published"] >= step):
        return "paused", None
    if known["publisher"]:
        return "publishing", None
    return "failed", (
        f"the publisher ended with the paused trio at step {step} unpublished "
        f"(last published {known['published']}, last error {known['publish_error']})"
    )


def refusal(known: dict[str, Any]) -> Optional[str]:
    """Why releasing the runtime now would lose work, or None when nothing would be lost.

    A leg that publishes must have its newest trio in the bucket, and once
    the run has finished, its final checkpoint and evaluations too.
    """
    if known["leg"] is not None and not publishes(known):
        return None
    summary = known["summary"]
    if summary is not None and summary["finished"]:
        if known["final_published"]:
            return None
        return (
            "the run finished on this VM and its final checkpoint and evaluations are not in "
            f"the bucket yet (last error {known['publish_error']})"
        )
    newest = known["newest_trio"]
    if newest is not None and (known["published"] is None or known["published"] < newest):
        return (
            f"the trio at step {newest} is not published yet (last published "
            f"{known['published']}, last error {known['publish_error']})"
        )
    return None


def report(working: Path = WORKING, proc: Path = PROC) -> dict[str, Any]:
    known = facts(working, proc)
    state, reason = outcome(known)
    return {"facts": known, "outcome": state, "reason": reason, "refusal": refusal(known)}


def main() -> None:
    HEARTBEAT.touch()
    print("LEG " + json.dumps(report()), flush=True)  # noqa: T201 -- the probe's answer


if __name__ == "__main__":
    main()
