"""Stage the e2e kernel's inputs on a Colab VM and start it detached.

Runs on the VM as its own process, which `colab/e2e.sh start` launches
(logging to `/kaggle/working/stage.out`) after it has uploaded the Hugging
Face token, the Kaggle credentials, the kernel script, its metadata and the
checkpoint publisher. It recreates the mount layout the Kaggle kernel expects
under `/kaggle/input/<name>`: every dataset and every finished run the kernel
starts from comes from the private Hub dataset repo of the same name, all at
once. The resume point is the newest `checkpoint-paused` trio by step among
the run's trio in the `lexoliu/mlime-resume` bucket and the latest COMPLETE
Kaggle segment's output. It stamps the segment number and the interval
checkpoint cadence into the kernel, launches it detached, and launches the
publisher beside it: every interval trio lands in the bucket, so a VM that
dies without warning loses minutes of training, not the leg.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

INPUTS = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
KERNEL = Path("/kaggle/kernel.py")
METADATA = Path("/kaggle/kernel-metadata.json")
PUBLISHER = Path("/kaggle/e2e-publish.py")
OUTPUT = Path("/kaggle/output.py")
HF_TOKEN = Path("/root/.cache/huggingface/token")
#: The files of a kernel output the kernel reads: a finished run's final
#: checkpoint and config, and a paused segment's trio.
KERNEL_READS = (
    r"^(run-config\.json|run-summary\.json|checkpoint-paused\.pt|run/checkpoint-final\.pt)$"
)
RESUME_MARKERS = ("checkpoint-paused.pt", "run-config.json", "run-summary.json")
#: Inputs in flight while staging: every Hub repo, the resume trio and the
#: latest Kaggle segment's probe at once.
STAGING_TRANSFERS = 16
#: Parallel file downloads inside one Hub repo.
HUB_WORKERS_PER_REPO = 8
#: The cadence a Colab leg checkpoints at: the VM can die without warning, so
#: the trio it would hand the next leg is re-written every twenty minutes.
CHECKPOINT_MINUTES = 20


def hub_dataset(source):
    """Download the Hub dataset repo named like Kaggle *source* into its mount.

    The repo keeps the Kaggle slug's owner and name, so a kernel's metadata
    names its inputs once for both. A download into an existing mount
    fetches only what is missing or changed.
    """
    mount = INPUTS / source.split("/")[1]
    snapshot_download(
        source, repo_type="dataset", local_dir=mount, max_workers=HUB_WORKERS_PER_REPO
    )
    print(f"{source}: {sum(1 for _ in mount.rglob('*') if _.is_file())} files", flush=True)
    return mount


def bucket_trio(publisher, api, run):
    """Download the run's whole trio from the resume bucket, or None before its first."""
    steps = publisher.published_steps(api, run)
    if not steps:
        print(f"{publisher.BUCKET}/{run}: no trio published yet", flush=True)
        return None
    prefix = publisher.trio_prefix(run, steps[-1])
    mount = INPUTS / f"{run}-resume"
    api.download_bucket_files(
        publisher.BUCKET,
        [(f"{prefix}/{marker}", mount / marker) for marker in RESUME_MARKERS],
        raise_on_missing_files=True,
    )
    print(f"{publisher.BUCKET}/{prefix}: staged under {mount}", flush=True)
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


def latest_segment(base):
    """The newest COMPLETE Kaggle segment's slug, or None when none has one.

    A segment still RUNNING or QUEUED means the chain owns the run right now;
    starting a Colab leg on the same checkpoint would fork the lineage, so
    that refuses to stage at all.
    """
    n = 0
    latest = None
    while True:
        slug = f"{base}-s{n}"
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
    """Download what the kernel reads of *slug*'s output; return its mount.

    `output.py` continues a partial file where it stopped, so a restaged leg
    fetches only what an earlier attempt did not finish.
    """
    mount = INPUTS / slug.split("/")[1]
    mount.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(OUTPUT), slug, str(mount), "--pattern", KERNEL_READS], check=True
    )
    print(f"{slug}: output staged under {mount}", flush=True)
    return mount


def stage_latest_segment(base):
    """Stage the newest complete Kaggle segment's output, when the run has one."""
    latest = latest_segment(base)
    if latest is not None:
        kernel_output(latest)


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
    # The Hub client with bucket support, and aria2c, which `output.py`
    # downloads Kaggle segment outputs with.
    installs = [
        subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle", "huggingface_hub"]
        ),
        subprocess.Popen(["apt-get", "install", "-y", "-q", "aria2"]),
    ]
    for install in installs:
        if install.wait() != 0:
            raise subprocess.CalledProcessError(install.returncode, install.args)
    for credential in (HF_TOKEN, Path("/root/.kaggle/credentials.json")):
        if not credential.is_file():
            raise FileNotFoundError(f"upload {credential.name} to {credential} first")
    for uploaded in (KERNEL, METADATA, PUBLISHER, OUTPUT):
        if not uploaded.is_file():
            raise FileNotFoundError(f"upload {uploaded.name} to {uploaded} first")
    WORKING.mkdir(parents=True, exist_ok=True)

    # The run's names -- its Kaggle segments and its place in the resume
    # bucket -- come from the stamped kernel's slug, the way the publisher
    # reads them.
    spec = importlib.util.spec_from_file_location("publisher", PUBLISHER)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    base = publisher.run_base(METADATA)
    api = HfApi()
    metadata = json.loads(METADATA.read_text())
    # Every input downloads at once: the GPU is billed while it waits, and one
    # transfer at a time leaves most of the VM's bandwidth unused. The
    # finished runs a stage starts from (`INIT_FROM`) are Hub repos too.
    with ThreadPoolExecutor(max_workers=STAGING_TRANSFERS) as pool:
        transfers = [
            *(
                pool.submit(hub_dataset, source)
                for source in [*metadata["dataset_sources"], *metadata["kernel_sources"]]
            ),
            pool.submit(bucket_trio, publisher, api, publisher.run_name(METADATA)),
            pool.submit(stage_latest_segment, base),
        ]
        for transfer in as_completed(transfers):
            transfer.result()

    # Every mounted trio is a resume candidate; the kernel picks the newest by
    # the step the checkpoint itself holds. The leg's number continues the
    # chain's: one past the segment that wrote the checkpoint being resumed,
    # or zero for a fresh run.
    candidates = trios()
    picked = max(candidates, key=lambda candidate: candidate[2], default=None)
    segment = 0 if picked is None else picked[1] + 1
    KERNEL.write_text(
        stamp(KERNEL.read_text(), SEGMENT=segment, CHECKPOINT_MINUTES=CHECKPOINT_MINUTES),
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
