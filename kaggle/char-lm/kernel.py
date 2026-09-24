"""Train the character language model the decoder's transition runs on.

One Kaggle session, 2xT4: `mlime train char-lm` over every run3 sample shard
(the v1 subset and the rest, staged side by side as `kaggle/route-a-v2` does),
then `mlime export char-lm`, so the kernel's output holds `charlm.onnx` and
`charlm.json` ready for `ime-cli --lm`, next to the final checkpoint and the
metrics. Kaggle kills a session at twelve hours and keeps nothing of it, so the
trainer is given a wall budget that ends the run with the export still to
spare; the step count is what two epochs need and the budget is the safety.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA = WORKING / "data"

#: How deep the mount namespace is walked. `datasets/<user>/<slug>` is three.
MAX_DEPTH = 4

#: Optimiser steps: two passes over the 41M lines at the token budget below,
#: 1.65 billion characters a pass, 32k per step across the two ranks.
MAX_STEPS = 100_000
#: Padded positions per step per rank.
MAX_TOKENS = 16384
CHECKPOINT_EVERY = 5000

#: One shard per source, withheld from training and scored as held-out nats per
#: character -- the same six the route A runs hold out.
HELD_OUT = (
    "bilibili-00001.parquet",
    "dialogue-00001.parquet",
    "douyin-00001.parquet",
    "moegirl-00001.parquet",
    "news-00001.parquet",
    "wiki-00001.parquet",
)

#: The shard by which each samples mount is recognised.
V1_MARKER = "dialogue-00000.parquet"
REST_MARKER = "dialogue-rest-00000.parquet"

#: Kaggle's session cap, and what is kept back from it: the install and staging
#: before training, the final checkpoint, the export, and the copy of the output
#: directory Kaggle makes after the script exits.
SESSION_SECONDS = 12 * 60 * 60
RESERVE_SECONDS = 40 * 60

REQUIREMENTS = (
    "polars",
    "structlog",
    "opencc",
    "regex",
    "typer",
    "onnx",
)


def directories(root, depth=MAX_DEPTH):
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


def describe():
    """What is mounted, for a failure that has to be read rather than guessed at."""
    if not INPUTS.is_dir():
        return {}
    return {
        str(directory): sorted(child.name for child in directory.iterdir())[:8]
        for directory in directories(INPUTS)
    }


def locate(*markers):
    """The mounted directory holding every one of *markers*."""
    if not INPUTS.is_dir():
        raise FileNotFoundError(f"{INPUTS} does not exist; the kernel has no inputs at all")
    for directory in directories(INPUTS):
        if all((directory / marker).exists() for marker in markers):
            return directory
    raise FileNotFoundError(f"no mounted directory holds {markers}; mounts hold {describe()}")


def samples_mount(marker):
    """The samples mount holding *marker*, told from a labels mount by its columns."""
    import polars as pl

    for directory in directories(INPUTS):
        shard = directory / marker
        if shard.is_file() and "text" in pl.scan_parquet(shard).collect_schema().names():
            return directory
    raise FileNotFoundError(f"no mount holds the samples shard {marker}; mounts hold {describe()}")


def importable_package():
    """A directory that can go on `PYTHONPATH` and make `mlime` importable.

    Only a package that carries the trainer counts: a mount holding an older
    `mlime` would otherwise be found first and fail at the command line, as one
    did when a source version still processing left the kernel bound to the
    one before it.
    """
    try:
        return locate("mlime/__init__.py", "mlime/train/charlm.py")
    except FileNotFoundError:
        mount = locate("__init__.py", "train/charlm.py")
    root = WORKING / "packages"
    root.mkdir(parents=True, exist_ok=True)
    link = root / "mlime"
    if not link.exists():
        link.symlink_to(mount, target_is_directory=True)
    return root


def install():
    """Install what the image does not ship."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS], check=True)


def stage_samples(sources):
    """One `samples` directory linking every shard of every mount."""
    target = DATA / "samples"
    target.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for shard in sorted(source.glob("*.parquet")):
            link = target / shard.name
            if link.exists():
                raise FileExistsError(f"{shard.name} is staged twice under {target}")
            link.symlink_to(shard)
    print(f"samples: {sum(1 for _ in target.glob('*.parquet'))} shards staged", flush=True)


def train_argv(char_table, out, wall_budget):
    """The training command both ranks run."""
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "mlime",
        "train",
        "char-lm",
        "--data-dir",
        str(DATA),
        "--char-table",
        str(char_table),
        "--out",
        str(out),
        "--max-steps",
        str(MAX_STEPS),
        "--max-tokens",
        str(MAX_TOKENS),
        "--checkpoint-every",
        str(CHECKPOINT_EVERY),
        "--wall-budget-seconds",
        str(wall_budget),
        "--seed",
        "0",
    ]
    for shard in HELD_OUT:
        argv += ["--held-out-shard", shard]
    return argv


def export_argv(checkpoint, out):
    """The command that writes the ONNX step graph and the manifest."""
    return [sys.executable, "-m", "mlime", "export", "char-lm", str(checkpoint), "--out", str(out)]


def run(argv, env, log):
    """Run *argv*, tee its output to *log*, and return (exit code, seconds)."""
    started = time.monotonic()
    finished = subprocess.run(
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    elapsed = time.monotonic() - started
    Path(log).write_text(finished.stdout, encoding="utf-8")
    print(f"--- {log} exited {finished.returncode} after {elapsed:.1f}s ---", flush=True)
    print(finished.stdout[-4000:], flush=True)
    return finished.returncode, elapsed


def records_of(metrics, event):
    """Every record of *event* in a metrics file, in order."""
    return [
        record
        for record in (json.loads(line) for line in Path(metrics).read_text().splitlines())
        if record.get("event") == event
    ]


def main():
    started = time.monotonic()
    install()
    package_root = importable_package()
    assets = locate("char_pinyin.tsv")
    stage_samples((samples_mount(V1_MARKER), samples_mount(REST_MARKER)))
    print(
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout,
        flush=True,
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(package_root)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    char_table = assets / "char_pinyin.tsv"
    out = WORKING / "run"
    shutil.rmtree(out, ignore_errors=True)
    wall_budget = SESSION_SECONDS - RESERVE_SECONDS - (time.monotonic() - started)
    if wall_budget <= 0:
        raise RuntimeError("the session was spent before training could start")
    code, seconds = run(train_argv(char_table, out, wall_budget), env, str(WORKING / "train.log"))
    if code != 0:
        raise RuntimeError("training failed; its log is in the kernel output")

    code, _ = run(
        export_argv(out / "charlm-final.pt", WORKING / "char-lm"), env, str(WORKING / "export.log")
    )
    if code != 0:
        raise RuntimeError("the export failed; its log is in the kernel output")
    # The numbered checkpoint is a copy of the final one's neighbourhood; only
    # the final one is worth Kaggle's output cap.
    (out / "charlm.pt").unlink(missing_ok=True)

    steps = records_of(out / "metrics.jsonl", "step")
    held = records_of(out / "metrics.jsonl", "held_out")
    summary = records_of(out / "metrics.jsonl", "summary")
    stopped = records_of(out / "metrics.jsonl", "stopped")
    report = {
        "max_steps": MAX_STEPS,
        "last_step": steps[-1]["step"],
        "first_loss": steps[0]["loss"],
        "last_loss": steps[-1]["loss"],
        "tokens": steps[-1]["tokens"],
        "tokens_per_second": steps[-1]["tokens_per_second"],
        "train_seconds": round(seconds, 1),
        "stopped_by_budget": bool(stopped),
        "held_out": [(record["step"], record["nats_per_char"]) for record in held],
        "final": summary[-1] if summary else None,
    }
    (WORKING / "run-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
