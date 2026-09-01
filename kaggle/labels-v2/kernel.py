"""Label the rest of run3 -- the rows v1 did not draw -- with g2pW, on both T4s.

The rest shards are `<source>-rest-<index>.parquet`, 404 of them holding 31.2M
rows (`scripts/build_v2_rest.rs`), and one kernel labels a fifth of them: at the
265 rows/s a v1 kernel measured, a fifth is 6.6 hours, inside the budget with no
mop-up kernel needed. Push five copies, `KERNEL_INDEX` 0 to 4, two at a time.

The kernel owns what the package deliberately does not: installing the GPU
onnxruntime wheel, importing torch first so its CUDA libraries are resident when
onnxruntime looks for them, splitting the corpus across the two GPUs Kaggle
hands out, and deciding how much of it fits in the time available.

One process per GPU. `CUDA_VISIBLE_DEVICES` is the only way to pin an ONNX CUDA
session to a device and it is read when the session is built, so the split has to
be two processes rather than two sessions. The controller forks once per device
and never touches CUDA itself -- the device count comes from `nvidia-smi`, not
from torch -- so each child initialises CUDA for the first time in its own
process, on the one device its environment leaves visible.

The corpus is split across kernels too: `KERNEL_INDEX` of `KERNEL_COUNT` takes
every `KERNEL_COUNT`-th shard, so two kernels running side by side cover it
between them and neither depends on the other finishing. Shards are dealt round
robin within a kernel as well, which keeps every worker on a mixture of sources
rather than giving one of them all of the news.

A kernel that runs out of time loses its output entirely, so the budget is
enforced on measured rows per second: a shard is only started while the rate says
it will finish inside the budget, and a shard whose labels already exist is
skipped, which makes a re-push a resume.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

INPUTS = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/labels")

#: How deep the mount namespace is walked. `datasets/<user>/<slug>` is three.
MAX_DEPTH = 4

#: Which slice of the corpus this kernel owns, and how many kernels share it.
#: `KERNEL_INDEX` is stamped per copy at push time; see kaggle/README.md.
KERNEL_INDEX = 0
KERNEL_COUNT = 5

#: Query positions handed to the ONNX session at once, and sentences handed to
#: the annotator per call. Both come from the calibration sweep.
ONNX_BATCH = 256
SENTENCE_BATCH = 2048

#: Kaggle kills a kernel at 12 hours and its output dies with it. Nine leaves
#: room for the install, the commit and a shard that runs long.
BUDGET_SECONDS = 9 * 60 * 60

#: The shard every rest mount holds, by which the samples mount is found.
SAMPLES_MARKER = "dialogue-rest-00000.parquet"

#: onnxruntime-gpu is pinned to the last release built against CUDA 12: current
#: releases link libcublasLt.so.13 while the Kaggle image is CUDA 12.8, so an
#: unpinned install loads and then silently has no CUDA provider. `g2pw` is
#: installed without dependencies because it declares the *CPU* onnxruntime,
#: which would install over the GPU wheel's files.
REQUIREMENTS = ("onnxruntime-gpu==1.22.0", "polars", "structlog", "opencc", "regex")
NO_DEPENDENCY_REQUIREMENTS = ("g2pw",)


def directories(root: Path, depth: int = MAX_DEPTH) -> list[Path]:
    """Every directory at or under *root*, breadth first, to *depth* levels."""
    found = [root]
    frontier = [root]
    for _ in range(depth):
        children = [child for parent in frontier for child in parent.iterdir() if child.is_dir()]
        found.extend(children)
        frontier = children
        if not frontier:
            break
    return found


def describe() -> dict[str, list[str]]:
    """What is mounted, one line per directory, for a failure that has to be read."""
    if not INPUTS.is_dir():
        return {}
    return {
        str(directory): sorted(child.name for child in directory.iterdir())[:8]
        for directory in directories(INPUTS)
    }


def locate(*markers: str) -> Path:
    """The mounted directory holding every one of *markers*."""
    if not INPUTS.is_dir():
        raise FileNotFoundError(f"{INPUTS} does not exist; the kernel has no inputs at all")
    for directory in directories(INPUTS):
        if all((directory / marker).exists() for marker in markers):
            return directory
    raise FileNotFoundError(f"no mounted directory holds {markers}; mounts hold {describe()}")


def importable_package() -> Path:
    """A directory that can go on ``sys.path`` and make ``mlime`` importable."""
    try:
        return locate("mlime/__init__.py")
    except FileNotFoundError:
        mount = locate("train/spans.py", "__init__.py")
    root = Path("/kaggle/working/packages")
    root.mkdir(parents=True, exist_ok=True)
    link = root / "mlime"
    if not link.exists():
        link.symlink_to(mount, target_is_directory=True)
    return root


def install() -> None:
    """Install what the image does not ship, keeping the GPU onnxruntime intact."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", *NO_DEPENDENCY_REQUIREMENTS],
        check=True,
    )


def visible_devices() -> list[int]:
    """The GPUs `nvidia-smi` lists, without initialising CUDA in this process."""
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
    found = [line for line in listing.stdout.splitlines() if line.startswith("GPU ")]
    if not found:
        raise RuntimeError(f"nvidia-smi lists no GPU: {listing.stdout!r}")
    return list(range(len(found)))


def worker(shards: list[str], device: int) -> None:
    """Label *shards* on the one GPU this process can see."""
    import asyncio

    import polars as pl
    import torch  # noqa: F401  -- loads the CUDA libraries onnxruntime then finds

    sys.path.insert(0, str(importable_package()))
    from mlime.logging import configure, log
    from mlime.train.labels import generate, load_cuda_annotator, select_shards

    configure()
    samples = locate(SAMPLES_MARKER)
    model = locate("g2pw.onnx", "version")
    paths = select_shards(samples, shards)
    annotator = load_cuda_annotator(model, ONNX_BATCH, None)
    log.info("worker starting", device=device, shards=len(paths))

    started = time.monotonic()
    rows_done = 0
    for path in paths:
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        elapsed = time.monotonic() - started
        if rows_done and elapsed + rows * elapsed / rows_done > BUDGET_SECONDS:
            log.info(
                "stopping inside the budget",
                device=device,
                rows_done=rows_done,
                elapsed=round(elapsed, 1),
                next_shard=path.name,
            )
            break
        asyncio.run(
            generate(
                [path],
                OUTPUT,
                annotator,
                sentences_per_batch=SENTENCE_BATCH,
                metrics=OUTPUT / f"throughput-{device}.jsonl",
            )
        )
        rows_done += rows
    total = time.monotonic() - started
    log.info(
        "worker finished",
        device=device,
        rows=rows_done,
        seconds=round(total, 1),
        sentences_per_second=round(rows_done / total, 1) if total else 0.0,
    )


def fork_worker(shards: list[str], device: int) -> int:
    """Fork a child that labels *shards* with only *device* visible, and return its pid."""
    pid = os.fork()
    if pid:
        return pid
    code = 0
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        worker(shards, device)
    except BaseException as error:  # noqa: BLE001 -- a child must report, never propagate
        print(f"device {device} failed: {error!r}", flush=True)
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


def controller() -> None:
    """Split this kernel's shards across the GPUs and run one worker on each."""
    install()
    samples = locate(SAMPLES_MARKER)
    names = sorted(path.name for path in samples.glob("*.parquet"))
    mine = [name for index, name in enumerate(names) if index % KERNEL_COUNT == KERNEL_INDEX]
    devices = visible_devices()
    per_device = [mine[index :: len(devices)] for index in range(len(devices))]
    print(
        f"kernel {KERNEL_INDEX}/{KERNEL_COUNT}: {len(names)} shards mounted, {len(mine)} mine, "
        f"devices {devices}, {[len(shards) for shards in per_device]} shards each",
        flush=True,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    children = {
        fork_worker(shards, device) for device, shards in zip(devices, per_device) if shards
    }
    failures = 0
    for _ in list(children):
        _, status = os.wait()
        failures += 0 if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else 1
    written = sorted(path.name for path in OUTPUT.glob("*.parquet"))
    print(
        f"{len(written)} label shards written in {round(time.monotonic() - started, 1)}s",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"{failures} labelling worker(s) failed")
    if not written:
        raise RuntimeError("the kernel labelled nothing")


controller()
