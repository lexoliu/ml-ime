"""Publish this leg's checkpoints to `lexoliu/mlime-e2e-resume`, in a loop.

Runs inside the Colab VM beside the kernel: `colab/e2e.sh start` launches it
detached, and `colab/e2e.sh publish` runs a single pass through it with
`ONCE=1`. A Colab VM can die without warning, so each interval trio the
kernel writes under `run/interval-<step>/` is versioned into the dataset as
soon as it lands, and the segment's own paused trio once the kernel has moved
it to the working root. The next leg -- Kaggle or Colab -- mounts the dataset
and resumes the newest checkpoint it holds.

When the dataset does not exist yet the publisher creates it, private, on the
first publish. What it last did -- step and time of the last publish, or the
last error -- is written to `publish-state.json`, which `e2e.sh status` reads.
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
DATASET = "lexoliu/mlime-e2e-resume"
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


def dataset_exists():
    """Whether the resume dataset is already there to version.

    Kaggle answers a metadata or status request for a private dataset that
    does not exist with 403, the same answer a credential problem gets, so the
    question is put to the account's own dataset list instead: the dataset
    exists exactly when that list holds its ref.
    """
    probe = subprocess.run(
        ["kaggle", "datasets", "list", "--mine", "--search", DATASET.split("/")[1], "--csv"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        output = (probe.stdout + probe.stderr).strip()
        raise RuntimeError(f"kaggle datasets list --mine failed: {output}")
    return any(row.get("ref") == DATASET for row in csv.DictReader(io.StringIO(probe.stdout)))


def publish(trio):
    """Version the dataset -- or create it, private -- from a frozen *trio*.

    The trio is hard-linked into its own fresh payload directory, so the
    keep-newest rotation deleting the original mid-publish never reaches the
    upload, and a second `publish` invocation never shares a payload dir with
    this loop. The payload is flat -- no subdirectories -- so no --dir-mode.
    """
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
                {
                    "id": DATASET,
                    "title": DATASET.split("/")[1],
                    "licenses": [{"name": "unknown"}],
                },
                indent=2,
            )
            + "\n"
        )
        if dataset_exists():
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "version",
                    "-p",
                    str(payload),
                    "-m",
                    f"e2e segment checkpoint at step {step}",
                ],
                check=True,
            )
        else:
            subprocess.run(
                ["kaggle", "datasets", "create", "-p", str(payload)],
                check=True,
            )
    finally:
        shutil.rmtree(payload, ignore_errors=True)
    print(f"published {trio} (step {step}) to {DATASET}", flush=True)
    return step


def main():
    once = os.environ.get("ONCE") == "1"
    published = -1
    while True:
        found = candidates()
        newest = found[-1] if found else None
        if newest is not None and newest[1] > published:
            try:
                step = publish(newest[0])
            except (subprocess.CalledProcessError, RuntimeError) as error:
                # A publish failure is transient until it is not: say so, keep
                # polling, and leave the error where status can read it.
                record_state(last_error=f"{error}", last_error_at=time.time())
                print(f"publish of {newest[0]} failed: {error}", flush=True)
            else:
                if step is not None:
                    published = step
                    record_state(last_published_step=step, last_published_at=time.time())
        if once:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
