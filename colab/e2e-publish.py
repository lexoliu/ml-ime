"""Publish this leg's newest checkpoint to the run's place in a Hub bucket, in a loop.

Runs inside the Colab VM beside the kernel: `colab/e2e.sh start` launches it
detached, and `colab/e2e.sh publish` runs a single pass through it with
`ONCE=1`. A Colab VM can die without warning, so each interval trio the
kernel writes under `run/interval-<step>/` is published as soon as it lands,
and the segment's own paused trio once the kernel has moved it to the working
root. The next leg's staging downloads the newest trio the bucket holds.

The trios live in the private bucket `lexoliu/mlime-resume`, under
`<run>/step-<step>/`. A bucket holds files without history, so storage stays at
one trio per run however long the run goes: a publish uploads the new trio,
then writes its `complete` marker, and only then deletes every other trio of
the run. A VM that dies between those calls leaves the previous trio whole
and the new one without its marker, which staging passes over.

Once this leg has trained a few logged steps it has compiled everything the
training runs, so the publisher packs inductor's caches once and puts them in
the bucket under `compile-cache/<gpu>-torch<version>.tar`; the next leg on the
same card and torch restores them and starts training without compiling again
(#140).

What it last did -- step and time of the last publish, or the last error -- is
written to `publish-state.json`, which `e2e.sh status` reads.
"""

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import httpx2
from huggingface_hub import HfApi

WORKING = Path("/kaggle/working")
RUN = WORKING / "run"
STATE = WORKING / "publish-state.json"
METADATA = Path("/kaggle/kernel-metadata.json")
RESUME_MARKERS = ("checkpoint-paused.pt", "run-config.json", "run-summary.json")
#: The private bucket every run's resume trio lives in.
BUCKET = "lexoliu/mlime-resume"
#: Written after a trio's files, so a trio is whole exactly when it has one.
COMPLETE = "complete"
#: How often the publisher looks for a newer trio than the one it last sent.
POLL_SECONDS = 60
#: Where the kernel points inductor's caches (`TORCHINDUCTOR_CACHE_DIR`):
#: outside the working directory, which a Kaggle segment publishes as output.
INDUCTOR_CACHE = Path("/tmp/inductor-cache")
#: The bucket directory the packed caches live in, one archive per card and torch.
CACHE_PREFIX = "compile-cache"
#: Step records this leg logs before its caches are packed: by then every graph
#: the training runs has compiled, the recompile a second shape brings included.
CACHE_AFTER_RECORDS = 5

#: Steps already read off checkpoints, keyed by the file's identity: path,
#: modified time, size. A trio lands whole via rename, so a rewrite makes a
#: new key rather than a stale read.
_step_cache: dict = {}


def run_base(metadata):
    """The run a stamped kernel belongs to: its slug without the segment suffix.

    `lexoliu/mlime-e2e-s3` belongs to v1's run `lexoliu/mlime-e2e`, and
    `lexoliu/mlime-e2e-rankw005-s0` to `lexoliu/mlime-e2e-rankw005`.
    """
    slug = json.loads(metadata.read_text())["id"]
    base, separator, segment = slug.rpartition("-s")
    if not separator or not segment.isdigit():
        raise ValueError(f"{slug} is not a segment slug of the form <run>-s<n>")
    return base


def run_name(metadata):
    """The run's directory in the bucket: its base slug without the owner."""
    return run_base(metadata).split("/")[1]


def trio_prefix(run, step):
    """Where the trio at *step* lives in the bucket."""
    return f"{run}/step-{step:07d}"


def published_steps(api, run):
    """The steps of *run* whose trio is whole in the bucket, oldest first."""
    steps = []
    for entry in api.list_bucket_tree(BUCKET, prefix=f"{run}/", recursive=True):
        directory, _, name = entry.path.rpartition("/")
        if entry.type == "file" and name == COMPLETE:
            steps.append(int(directory.rpartition("/step-")[2]))
    return sorted(steps)


def compile_cache_path():
    """The bucket path of the compile caches for this VM's GPU and installed torch.

    Inductor keys every entry by its own inputs, so an archive from a different
    source tree only misses where the source changed; the card and the torch
    build decide whether any entry can hit at all.
    """
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[0]
    torch = importlib.metadata.version("torch")
    return f"{CACHE_PREFIX}/{re.sub(r'[^A-Za-z0-9.+-]+', '-', gpu)}-torch{torch}.tar"


def compiled():
    """Whether this leg has logged enough steps for its caches to be whole.

    The kernel starts every leg on an empty run directory, so the step records
    in its metrics are this leg's own. Only finished lines are read; the
    trainer may be writing the last one.
    """
    metrics = RUN / "metrics.jsonl"
    if not metrics.is_file():
        return False
    lines = metrics.read_text().split("\n")[:-1]
    steps = sum(1 for line in lines if json.loads(line)["event"] == "step")
    return steps >= CACHE_AFTER_RECORDS


def publish_compile_cache(api):
    """Pack the inductor caches and replace this card's archive in the bucket."""
    path = compile_cache_path()
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / "inductor-cache.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(INDUCTOR_CACHE, arcname=".")
        api.batch_bucket_files(BUCKET, add=[(archive, path)])
        size = archive.stat().st_size
    print(f"published the compile caches ({size} bytes) to {BUCKET}/{path}", flush=True)


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


def publish(api, run, trio):
    """Upload a frozen *trio*, mark it whole, then drop the run's other trios.

    The trio is hard-linked into its own fresh payload directory, so the
    keep-newest rotation deleting the original mid-publish never reaches the
    upload, and a second `publish` invocation never shares a payload dir with
    this loop.
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
        prefix = trio_prefix(run, step)
        api.batch_bucket_files(
            BUCKET, add=[(payload / marker, f"{prefix}/{marker}") for marker in RESUME_MARKERS]
        )
        api.batch_bucket_files(BUCKET, add=[(f"{step}\n".encode(), f"{prefix}/{COMPLETE}")])
    finally:
        shutil.rmtree(payload, ignore_errors=True)
    stale = [
        entry.path
        for entry in api.list_bucket_tree(BUCKET, prefix=f"{run}/", recursive=True)
        if entry.type == "file" and not entry.path.startswith(f"{prefix}/")
    ]
    if stale:
        api.batch_bucket_files(BUCKET, delete=stale)
    print(f"published {trio} (step {step}) to {BUCKET}/{prefix}", flush=True)
    return step


def main():
    once = os.environ.get("ONCE") == "1"
    api = HfApi()
    run = run_name(METADATA)
    published = None
    caches_published = False
    while True:
        try:
            if published is None:
                published = max(published_steps(api, run), default=-1)
            found = candidates()
            newest = found[-1] if found else None
            if newest is not None and newest[1] > published:
                step = publish(api, run, newest[0])
                if step is not None:
                    published = step
                    record_state(last_published_step=step, last_published_at=time.time())
            if not caches_published and compiled():
                publish_compile_cache(api)
                caches_published = True
        except (httpx2.HTTPError, OSError) as error:
            # A publish failure is transient until it is not: say so, keep
            # polling, and leave the error where status can read it. What the
            # bucket holds is read again, since a failure can stop mid-publish.
            published = None
            record_state(last_error=f"{error}", last_error_at=time.time())
            print(f"publish failed: {error}", flush=True)
        if once:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
