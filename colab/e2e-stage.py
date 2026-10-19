"""Stage the e2e kernel's inputs on a Colab VM and start it detached.

Runs on the VM as its own process, from the leg's checkout of the repository,
which `colab/e2e.sh start` fetches and then launches this from (logging to
`/kaggle/working/stage.out`) once it has uploaded the Hugging Face token, the
Kaggle credentials, the kernel script and its metadata; the publisher, the Hub
and output fetchers run from the same checkout. Its options -- the commit, the
session's length, the run to resume and whether to publish -- are recorded in
`/kaggle/working/leg.json` before anything else, where the waiter's probe
reads them. It recreates the mount layout the Kaggle kernel expects
under `/kaggle/input/<name>`: every dataset and every finished run the kernel
starts from comes from the private Hub dataset repo of the same name, all of
them in one `aria2c` run (`hub.py`). The resume point is the newest
`checkpoint-paused` trio by step among the run's trio in the
`lexoliu/mlime-resume` bucket and the latest COMPLETE Kaggle segment's output.
On a GPU it also restores the inductor caches an earlier leg on the same card
and torch packed into the bucket (#140), so the kernel does not compile again.
It stamps the segment number, the interval checkpoint cadence, the session's
length and the commit into the kernel, launches it detached, and launches the
publisher beside it: every interval trio lands in the bucket, so a VM that
dies without warning loses minutes of training, not the leg.

    python3 e2e-stage.py --commit SHA --session-seconds N
                         [--resume-from RUN] [--no-publish] [--allow-commit-change]
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
KERNEL = Path("/kaggle/kernel.py")
METADATA = Path("/kaggle/kernel-metadata.json")
#: The leg's options, as this stager parsed them; the waiter's probe reads them.
LEG = WORKING / "leg.json"
#: This file's directory in the leg's checkout, and the drivers beside it.
HERE = Path(__file__).resolve().parent
PUBLISHER = HERE / "e2e-publish.py"
HUB = HERE / "hub.py"
OUTPUT = HERE.parent / "kaggle" / "output.py"
ARIA2 = HERE.parent / "kaggle" / "aria2.py"
# The checkout's `mlime.stamp`, the one kernel rewrite the repository has; it
# is standard library only, so it imports before anything is installed.
sys.path.insert(0, str(HERE.parent / "python" / "src"))
from mlime.stamp import literal, stamp  # noqa: E402

HF_TOKEN = Path("/root/.cache/huggingface/token")
#: The files of a kernel output the kernel reads: a finished run's final
#: checkpoint and config, and a paused segment's trio.
KERNEL_READS = r"^(run-config\.json|run-summary\.json|checkpoint-paused\.pt|run/checkpoint-final\.pt)$"
RESUME_MARKERS = ("checkpoint-paused.pt", "run-config.json", "run-summary.json")
#: Transfers in flight while staging: the Hub repos, the resume trio, the
#: compile caches and the latest Kaggle segment's probe.
STAGING_TRANSFERS = 4
#: The cadence a Colab leg checkpoints at: the VM can die without warning, so
#: the trio it would hand the next leg is re-written every twenty minutes.
CHECKPOINT_MINUTES = 20


def bucket_trio(publisher, api, run, required):
    """Download *run*'s whole trio from the resume bucket, or None before its first.

    A leg started `--resume-from` another run *requires* that run's trio: one
    missing would start the leg fresh instead, on a schedule nobody asked for.
    """
    steps = publisher.published_steps(api, run)
    if not steps:
        if required:
            raise FileNotFoundError(
                f"{publisher.BUCKET}/{run}: no whole trio to resume from; "
                "--resume-from names a run that has published one"
            )
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


def compile_caches(publisher, api):
    """Restore the inductor caches packed for this GPU and torch, when a leg packed them."""
    if shutil.which("nvidia-smi") is None:
        print("no GPU: nothing compiles, so no compile caches", flush=True)
        return
    path = publisher.compile_cache_path()
    entries = api.list_bucket_tree(
        publisher.BUCKET, prefix=f"{publisher.CACHE_PREFIX}/"
    )
    if not any(entry.type == "file" and entry.path == path for entry in entries):
        print(
            f"{publisher.BUCKET}/{path}: not packed yet; the kernel compiles",
            flush=True,
        )
        return
    archive = publisher.INDUCTOR_CACHE.with_suffix(".tar")
    api.download_bucket_files(
        publisher.BUCKET, [(path, archive)], raise_on_missing_files=True
    )
    with tarfile.open(archive) as tar:
        tar.extractall(publisher.INDUCTOR_CACHE, filter="data")
    archive.unlink()
    print(
        f"{publisher.BUCKET}/{path}: restored under {publisher.INDUCTOR_CACHE}",
        flush=True,
    )


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
            raise RuntimeError(
                f"{slug} is {status}; refusing to start a second lineage"
            )
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
        [sys.executable, str(OUTPUT), slug, str(mount), "--pattern", KERNEL_READS],
        check=True,
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

    state = torch.load(
        trio / "checkpoint-paused.pt", map_location="cpu", weights_only=False
    )
    return int(state["step"])


def trios():
    """Every mounted resume trio with the segment that wrote it and its step."""
    found = []
    for directory in [INPUTS, *[d for d in INPUTS.rglob("*") if d.is_dir()]]:
        if all((directory / marker).is_file() for marker in RESUME_MARKERS):
            config = json.loads((directory / "run-config.json").read_text())
            found.append(
                (directory, int(config.get("segment", -1)), step_of(directory))
            )
    return found


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
    print(
        f"{' '.join(argv)} started, pid {child.pid}; log {WORKING / log_name}",
        flush=True,
    )


def requirements(kernel):
    """The kernel's pinned `REQUIREMENTS`, read from its source without running it."""
    return list(literal(kernel.read_text(), "REQUIREMENTS"))


def options(argv=None):
    """The leg's options; the commit and the session's length have no default."""
    parser = argparse.ArgumentParser(description="Stage and start one e2e leg.")
    parser.add_argument("--commit", required=True, help="The commit the leg runs")
    parser.add_argument(
        "--session-seconds",
        required=True,
        type=int,
        help="How long this runtime is held; the kernel pauses inside it",
    )
    parser.add_argument(
        "--resume-from", help="Resume this run's newest trio, not the leg's own"
    )
    parser.add_argument("--no-publish", action="store_true", help="Run no publisher")
    parser.add_argument(
        "--allow-commit-change",
        action="store_true",
        help="Let this leg resume a run recorded on a different commit",
    )
    parsed = parser.parse_args(argv)
    if parsed.session_seconds <= 0:
        parser.error(
            f"--session-seconds must be positive, got {parsed.session_seconds}"
        )
    return parsed


def stamps(leg, segment):
    """The kernel fields this leg stamps."""
    return {
        "SEGMENT": segment,
        "CHECKPOINT_MINUTES": CHECKPOINT_MINUTES,
        "SESSION_SECONDS": leg.session_seconds,
        "COMMIT": leg.commit,
        "ALLOW_COMMIT_CHANGE": leg.allow_commit_change,
    }


def main(argv=None):
    leg = options(argv)
    WORKING.mkdir(parents=True, exist_ok=True)
    LEG.write_text(json.dumps(vars(leg), indent=2) + "\n", encoding="utf-8")
    for needed in (HF_TOKEN, Path("/root/.kaggle/credentials.json"), KERNEL, METADATA):
        if not needed.is_file():
            raise FileNotFoundError(f"upload {needed.name} to {needed} first")
    # aria2c, which `hub.py` and `output.py` download with, beside the Python
    # packages: the Kaggle client, then the kernel's pins over whatever it
    # pulled in. The Hub client this stager and the publisher import is the
    # kernel's pinned one, so the kernel's own install moves nothing under a
    # publisher that is already running.
    apt = subprocess.Popen(["apt-get", "install", "-y", "-q", "aria2"])
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run([*pip, "-U", "kaggle"], check=True)
    subprocess.run([*pip, "--no-deps", *requirements(KERNEL)], check=True)
    if apt.wait() != 0:
        raise subprocess.CalledProcessError(apt.returncode, apt.args)
    # Imported only now: the Hub client is the one just installed, and
    # `hub.py` imports `aria2.py` from the checkout's kaggle/.
    sys.path[:0] = [str(HUB.parent), str(ARIA2.parent)]
    import hub
    from huggingface_hub import HfApi

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
    repos = [*metadata["dataset_sources"], *metadata["kernel_sources"]]
    # --resume-from races another run's checkpoint under this run's name, so
    # two cards can start one leg from the same trio.
    if leg.resume_from is not None:
        resumed_run, required = leg.resume_from, True
    else:
        resumed_run, required = publisher.run_name(METADATA), False
    with ThreadPoolExecutor(max_workers=STAGING_TRANSFERS) as pool:
        transfers = [
            pool.submit(hub.download, repos, INPUTS),
            pool.submit(bucket_trio, publisher, api, resumed_run, required),
            pool.submit(compile_caches, publisher, api),
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
        stamp(KERNEL.read_text(), **stamps(leg, segment)), encoding="utf-8"
    )
    print(
        f"segment {segment}; resume candidates: "
        + ", ".join(f"{directory} at step {step}" for directory, _, step in candidates),
        flush=True,
    )

    launch([sys.executable, str(KERNEL)], "kernel.out", "kernel.pid")
    # A check publishes nothing: its trios would enter the run's lineage.
    if not leg.no_publish:
        launch([sys.executable, str(PUBLISHER)], "publish.out", "publish.pid")


if __name__ == "__main__":
    main()
