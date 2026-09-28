"""Publish this leg's checkpoints to two alternating Kaggle datasets, in a loop.

Runs inside the Colab VM beside the kernel: `colab/e2e.sh start` launches it
detached, and `colab/e2e.sh publish` runs a single pass through it with
`ONCE=1`. A Colab VM can die without warning, so each interval trio the
kernel writes under `run/interval-<step>/` is published as soon as it lands,
and the segment's own paused trio once the kernel has moved it to the working
root. The next leg -- Kaggle or Colab -- mounts both datasets and resumes the
newest checkpoint either holds.

Why two datasets rather than versions of one: every version of a Kaggle
dataset counts against the account's private quota and the API cannot delete
a version, so a multi-GB trio versioned every twenty minutes fills the quota
within a day. Each publish instead deletes and recreates whichever slot holds
the older checkpoint. The other slot keeps the previous one throughout, so a
VM that dies between the delete and the upload still leaves a resume point,
and the storage stays at two trios however long the run goes.

What it last did -- step and time of the last publish, or the last error -- is
written to `publish-state.json`, which `e2e.sh status` reads.
"""

import csv
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

WORKING = Path("/kaggle/working")
RUN = WORKING / "run"
STATE = WORKING / "publish-state.json"
#: The two datasets a publish alternates between; the kernel mounts both.
SLOTS = ("lexoliu/mlime-e2e-resume-a", "lexoliu/mlime-e2e-resume-b")
RESUME_MARKERS = ("checkpoint-paused.pt", "run-config.json", "run-summary.json")
#: How often the publisher looks for a newer trio than the one it last sent.
POLL_SECONDS = 60

#: Steps already read off checkpoints, keyed by the file's identity: path,
#: modified time, size. A trio lands whole via rename, so a rewrite makes a
#: new key rather than a stale read.
_step_cache: dict = {}


def record_state(**fields):
    """Merge *fields* into the publish-state file status reads."""
    state = {}
    if STATE.is_file():
        state = json.loads(STATE.read_text())
    state.update(fields)
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def _load_checkpoint(path):
    """torch.load with memory-mapping when this torch has it, for header reads."""
    import inspect

    import torch

    options = {}
    if "mmap" in inspect.signature(torch.load).parameters:
        options["mmap"] = True
    return torch.load(path, map_location="cpu", weights_only=False, **options)


def step_of(trio):
    """The step the checkpoint itself records; a summary's field can lag it.

    A checkpoint can be GBs, and this poll's host also runs the trainer's data
    loader, so the read happens once per file identity rather than per poll.
    """
    checkpoint = trio / "checkpoint-paused.pt"
    stat = checkpoint.stat()  # a vanishing trio is not a candidate this poll
    key = (checkpoint, stat.st_mtime_ns, stat.st_size)
    if key not in _step_cache:
        _step_cache[key] = int(_load_checkpoint(checkpoint)["step"])
    return _step_cache[key]


def candidates():
    """Every complete trio on the VM, oldest to newest by checkpoint step.

    Keep-newest rotation can remove an interval dir between this scan and the
    reads it does, so a trio that vanishes mid-poll is skipped, not an error.
    """
    found = []
    for directory in [*sorted(RUN.glob("interval-*")), WORKING]:
        if not all((directory / marker).is_file() for marker in RESUME_MARKERS):
            continue
        try:
            step = step_of(directory)
        except FileNotFoundError:
            continue
        found.append((directory, step))
    return sorted(found, key=lambda candidate: candidate[1])


def existing_slots():
    """The slots that exist now.

    Kaggle answers a metadata or status request for a private dataset that
    does not exist with 403, the same answer a credential problem gets, so the
    question is put to the account's own dataset list instead: a slot exists
    exactly when that list holds its ref.
    """
    probe = subprocess.run(
        ["kaggle", "datasets", "list", "--mine", "--search", "mlime-e2e-resume", "--csv"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        output = (probe.stdout + probe.stderr).strip()
        raise RuntimeError(f"kaggle datasets list --mine failed: {output}")
    refs = {row.get("ref") for row in csv.DictReader(io.StringIO(probe.stdout))}
    return [slot for slot in SLOTS if slot in refs]


def slot_step(slot):
    """The checkpoint step an existing *slot* holds, read off its summary."""
    scratch = Path(tempfile.mkdtemp(prefix="slot-", dir=WORKING))
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", slot, "-f", "run-summary.json", "-p", str(scratch)],
            check=True,
            capture_output=True,
        )
        summary = json.loads((scratch / "run-summary.json").read_text())
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return int(summary["checkpoint_step"])


def held_steps():
    """What each slot holds: its checkpoint step, or None when it does not exist."""
    present = existing_slots()
    return {slot: slot_step(slot) if slot in present else None for slot in SLOTS}


def target_slot(held):
    """The slot the next publish replaces: a missing one first, else the older."""
    return min(SLOTS, key=lambda slot: -1 if held[slot] is None else held[slot])


def publish(trio, held):
    """Replace the older slot's contents with a frozen *trio*; update *held*.

    The trio is hard-linked into its own fresh payload directory, so the
    keep-newest rotation deleting the original mid-publish never reaches the
    upload, and a second `publish` invocation never shares a payload dir with
    this loop. The payload is flat -- no subdirectories -- so no --dir-mode.
    """
    slot = target_slot(held)
    payload = Path(tempfile.mkdtemp(prefix="payload-", dir=WORKING))
    try:
        try:
            step = step_of(trio)
            for marker in RESUME_MARKERS:
                os.link(trio / marker, payload / marker)
        except FileNotFoundError:
            print(f"{trio} went away before it could be published", flush=True)
            return None
        (payload / "dataset-metadata.json").write_text(
            json.dumps(
                {"id": slot, "title": slot.split("/")[1], "licenses": [{"name": "unknown"}]},
                indent=2,
            )
            + "\n"
        )
        if held[slot] is not None:
            subprocess.run(["kaggle", "datasets", "delete", "-y", slot], check=True)
            held[slot] = None
        subprocess.run(["kaggle", "datasets", "create", "-p", str(payload)], check=True)
        held[slot] = step
    finally:
        shutil.rmtree(payload, ignore_errors=True)
    print(f"published {trio} (step {step}) to {slot}", flush=True)
    return step


def main():
    once = os.environ.get("ONCE") == "1"
    held = None
    while True:
        try:
            if held is None:
                held = held_steps()
            found = candidates()
            newest = found[-1] if found else None
            published = max((step for step in held.values() if step is not None), default=-1)
            if newest is not None and newest[1] > published:
                step = publish(newest[0], held)
                if step is not None:
                    record_state(
                        last_published_step=step, last_published_at=time.time(), slots=held
                    )
        except (subprocess.CalledProcessError, RuntimeError) as error:
            # A publish failure is transient until it is not: say so, keep
            # polling, and leave the error where status can read it. What the
            # slots hold is read again, since a failure can leave one deleted.
            held = None
            record_state(last_error=f"{error}", last_error_at=time.time())
            print(f"publish failed: {error}", flush=True)
        if once:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
