"""Stage the e2e kernel's inputs on a Colab VM and start it detached.

Runs on the VM as its own process, which `colab/e2e.sh start` launches
(logging to `/kaggle/working/stage.out`) after it has uploaded the Kaggle
credentials, the kernel script, its metadata and the checkpoint publisher. It recreates the mount layout the
Kaggle kernel expects by downloading the same datasets, stages the resume
point -- the newest `checkpoint-paused` trio by step among the latest
COMPLETE Kaggle segment's output and the `mlime-e2e-resume` dataset -- stamps
the segment number and the interval checkpoint cadence into the kernel,
launches it detached, and launches the publisher beside it: every interval
trio lands in `mlime-e2e-resume`, so a VM that dies without warning loses
minutes of training, not the leg.
"""

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
KERNEL = Path("/kaggle/kernel.py")
METADATA = Path("/kaggle/kernel-metadata.json")
PUBLISHER = Path("/kaggle/e2e-publish.py")
RESUME_DATASET = "lexoliu/mlime-e2e-resume"
SEGMENT_SLUG = "lexoliu/mlime-e2e-s"
RESUME_MARKERS = ("checkpoint-paused.pt", "run-config.json", "run-summary.json")
#: The cadence a Colab leg checkpoints at: the VM can die without warning, so
#: the trio it would hand the next leg is re-written every twenty minutes.
CHECKPOINT_MINUTES = 20


def download(dataset, tolerate=False):
    """Fetch *dataset* into its mount directory unless it is already there."""
    mount = INPUTS / dataset.split("/")[1]
    if mount.is_dir() and any(mount.iterdir()):
        print(f"{dataset}: already staged", flush=True)
        return mount
    mount.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(mount)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if tolerate:
            print(
                f"{dataset}: not mounted this leg ({result.stderr.strip() or 'not found'})",
                flush=True,
            )
            return None
        print(result.stderr, flush=True)
        result.check_returncode()
    for archive in mount.glob("*.zip"):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(mount)
        archive.unlink()
    print(f"{dataset}: {sum(1 for _ in mount.rglob('*') if _.is_file())} files", flush=True)
    return mount


def kernel_status(slug):
    """The Kaggle worker status text for *slug*, or None when it was never pushed.

    "No such kernel" ends the chain's probe; any other nonzero -- auth,
    network, quota -- is an API error and raises, never a silent stop.
    """
    probe = subprocess.run(
        ["kaggle", "kernels", "status", slug],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (probe.stdout + probe.stderr).strip()
    lowered = output.lower()
    if "not found" in lowered or "404" in lowered or "cannot access kernel" in lowered:
        return None
    if probe.returncode != 0:
        raise RuntimeError(f"kaggle kernels status {slug} failed: {output}")
    return output


def latest_segment():
    """The newest COMPLETE Kaggle segment's slug, or None when none has one.

    A segment still RUNNING or QUEUED means the chain owns the run right now;
    starting a Colab leg on the same checkpoint would fork the lineage, so
    that refuses to stage at all.
    """
    n = 0
    latest = None
    while True:
        slug = f"{SEGMENT_SLUG}{n}"
        status = kernel_status(slug)
        if status is None:
            break
        lowered = status.lower()
        if "running" in lowered or "queued" in lowered:
            raise RuntimeError(f"{slug} is {status}; refusing to start a second lineage")
        if "complete" in lowered:
            latest = slug
        n += 1
    if latest is not None:
        print(f"latest complete Kaggle segment: {latest}", flush=True)
    return latest


def kernel_output(slug):
    """Download *slug*'s output under the input layout and return its mount."""
    mount = INPUTS / slug.split("/")[1]
    if (mount / "run-config.json").is_file():
        print(f"{slug}: output already staged", flush=True)
        return mount
    mount.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(mount)], check=True)
    print(f"{slug}: output staged under {mount}", flush=True)
    return mount


def step_of(trio):
    """The step the checkpoint itself records; a summary's field can lag it."""
    import torch

    state = torch.load(trio / "checkpoint-paused.pt", map_location="cpu", weights_only=False)
    return int(state["step"])


def trios():
    """Every mounted resume trio with the segment that wrote it and its step."""
    found = []
    for directory in [INPUTS, *[d for d in INPUTS.rglob("*") if d.is_dir()]]:
        if all((directory / marker).is_file() for marker in RESUME_MARKERS):
            config = json.loads((directory / "run-config.json").read_text())
            found.append((directory, int(config.get("segment", -1)), step_of(directory)))
    return found


def stamp(source, **fields):
    """Rewrite a ``NAME = value`` kernel line per field, refusing a missing one."""
    for key, value in fields.items():
        source, count = re.subn(
            rf"^{key} = .*$", f"{key} = {value}", source, count=1, flags=re.MULTILINE
        )
        if count != 1:
            raise RuntimeError(f"{KERNEL} holds no `{key} =` line to stamp")
    return source


def launch(argv, log_name, pid_name):
    """Start *argv* detached, teeing to *log_name* under the working dir."""
    log = open(WORKING / log_name, "ab")  # noqa: SIM115 -- handed to the child
    child = subprocess.Popen(
        argv,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(WORKING),
        env=dict(os.environ),
        start_new_session=True,
    )
    (WORKING / pid_name).write_text(f"{child.pid}\n")
    print(f"{' '.join(argv)} started, pid {child.pid}; log {WORKING / log_name}", flush=True)


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
    if not Path("/root/.kaggle/credentials.json").is_file():
        raise FileNotFoundError("upload ~/.kaggle/credentials.json to /root/.kaggle first")
    for uploaded in (KERNEL, METADATA, PUBLISHER):
        if not uploaded.is_file():
            raise FileNotFoundError(f"upload {uploaded.name} to {uploaded} first")
    WORKING.mkdir(parents=True, exist_ok=True)

    for dataset in json.loads(METADATA.read_text())["dataset_sources"]:
        # The resume dataset holds the Colab leg's checkpoints; before the
        # first Colab segment it does not exist, and the kernel tolerates its
        # absence the way it would on Kaggle.
        download(dataset, tolerate=(dataset == RESUME_DATASET))
    latest = latest_segment()
    if latest is not None:
        kernel_output(latest)

    # Every mounted trio is a resume candidate; the kernel picks the newest by
    # the step the checkpoint itself holds. The leg's number continues the
    # chain's: one past the segment that wrote the checkpoint being resumed,
    # or zero for a fresh run.
    candidates = trios()
    picked = max(candidates, key=lambda candidate: candidate[2], default=None)
    segment = 0 if picked is None else picked[1] + 1
    KERNEL.write_text(
        stamp(
            KERNEL.read_text(), SEGMENT=segment, CHECKPOINT_MINUTES=CHECKPOINT_MINUTES
        ),
        encoding="utf-8",
    )
    print(
        f"segment {segment}; resume candidates: "
        + ", ".join(f"{directory} at step {step}" for directory, _, step in candidates),
        flush=True,
    )

    launch([sys.executable, str(KERNEL)], "kernel.out", "kernel.pid")
    launch([sys.executable, str(PUBLISHER)], "publish.out", "publish.pid")


main()
