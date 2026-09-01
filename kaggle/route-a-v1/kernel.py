"""Route A's first full run: one epoch over the ten-million-segment v1 subset.

The kernel that proved the two ranks step together is this kernel with the step
count raised and the comparison removed, because what is being trusted here is
that it is the same run. It trains for one pass over the training shards, then
scores the evaluation lattice twice from the one finished checkpoint -- once with
each record's context and once without -- and writes both score files into the
kernel output.

Scoring happens here rather than on a laptop for one reason: the checkpoint is
most of a gigabyte and the lattice is twenty-one million log probabilities, and
the machine that has both already, with a warm GPU, is this one. What comes back
is the score files, which are a fifth of the size and the only thing the fused
decode needs.

The step count is not a guess. The batcher is arithmetic over sentence and
context lengths, so it was replayed over all 409 training shards to count exactly
how many batches one epoch produces per rank: 29,645 for the shards rank zero
reads and 28,977 for rank one's. The lower of the two is the run, because the
stream is endless and overshooting quietly starts a second epoch over data the
first one has already augmented.
"""

import json
import os
import re
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

#: One shard per source, withheld from training and scored at the end. Index one
#: rather than the last of each source, because a source's last shard is a
#: partial one and a held-out set should not be a different size by accident.
HELD_OUT = (
    "bilibili-00001.parquet",
    "dialogue-00001.parquet",
    "douyin-00001.parquet",
    "moegirl-00001.parquet",
    "news-00001.parquet",
    "wiki-00001.parquet",
)

#: One epoch, counted rather than estimated. The batcher is arithmetic over
#: sentence and context lengths, so it was replayed over all 409 training shards:
#: rank zero's share is 29,645 batches and rank one's is 28,977, and the run is
#: the shorter of the two because the stream is endless and overshooting quietly
#: starts a second pass over data the first has already augmented. A further one
#: per cent is left off for the drops the replay could not model -- a reading no
#: typed span admits is not visible in a length -- so this is 0.99 of an epoch,
#: which is the honest description of it.
STEPS = 28600

#: Padded positions per step per rank, both towers together. The first attempt
#: bounded the fill tower alone at 4,096 and the two-rank run went out of memory
#: on a batch whose contexts were four times that; this is the whole rectangle,
#: and it is a little under what the one-rank run was accidentally spending.
TOKEN_BUDGET = 16384

#: The base checkpoint both towers start from.
BASE_MODEL = "hfl/chinese-macbert-base"

#: `transformers` is pinned above 5.16 because route A's fill tower runs the
#: encoder layer by layer and needs `BertModel._create_attention_masks`, which
#: the 4.x series does not have. The rest is what the package imports and the
#: image does not ship.
REQUIREMENTS = (
    "polars",
    "structlog",
    "opencc",
    "regex",
    "typer",
    "transformers>=5.16",
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


def parquet_mounts():
    """The samples mount and the labels mount, told apart by what their shards hold.

    Their file names are set-equal by construction -- a label shard is named
    after the sample shard it labels -- so the only honest way to tell the two
    mounts apart is to open one shard of each and look at its columns. Guessing
    from the mount path would work until the day a dataset is renamed.
    """
    import polars as pl

    found = {}
    for directory in directories(INPUTS):
        shard = directory / "dialogue-00000.parquet"
        if not shard.is_file():
            continue
        columns = set(pl.scan_parquet(shard).collect_schema().names())
        if "text" in columns:
            found["samples"] = directory
        elif "syllables" in columns:
            found["labels"] = directory
    missing = {"samples", "labels"} - set(found)
    if missing:
        raise FileNotFoundError(
            f"no mount holds the {sorted(missing)} shards; mounts hold {describe()}"
        )
    return found["samples"], found["labels"]


def joined_ranks(output):
    """The ranks that logged joining the process group.

    The renderer colours its output, and a colour code sits between the key and
    the value, so the escapes come off before anything is read out of the line.
    """
    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
    ranks = set()
    for line in clean.splitlines():
        if "joined the process group" not in line:
            continue
        found = re.search(r"rank=(\d+)", line)
        if found:
            ranks.add(int(found.group(1)))
    return sorted(ranks)


def importable_package():
    """A directory that can go on `PYTHONPATH` and make `mlime` importable."""
    try:
        return locate("mlime/__init__.py")
    except FileNotFoundError:
        mount = locate("train/spans.py", "__init__.py")
    root = WORKING / "packages"
    root.mkdir(parents=True, exist_ok=True)
    link = root / "mlime"
    if not link.exists():
        link.symlink_to(mount, target_is_directory=True)
    return root


def install():
    """Install what the image does not ship."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS], check=True)


def stage_data(samples, labels):
    """A single data root whose `samples` and `labels` are the two mounts.

    The package addresses a corpus as one root with named subdirectories, and
    Kaggle mounts each dataset at its own path. Two symlinks reconcile the two
    without either side learning about the other.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    for name, mount in (("samples", samples), ("labels", labels)):
        link = DATA / name
        if not link.exists():
            link.symlink_to(mount, target_is_directory=True)
    return DATA


def warm_cache():
    """Download the base checkpoint once, in one process.

    Two ranks reaching for the same uncached checkpoint at the same moment is a
    race with a file lock at the bottom of it; pulling it here means the ranks
    only ever read.
    """
    from transformers import AutoTokenizer, BertForMaskedLM, BertModel

    AutoTokenizer.from_pretrained(BASE_MODEL)
    BertForMaskedLM.from_pretrained(BASE_MODEL)
    BertModel.from_pretrained(BASE_MODEL, add_pooling_layer=False)
    print("base checkpoint cached", flush=True)


def train_argv(out, char_table):
    """The one training command, which both launchers run."""
    argv = [
        "-m",
        "mlime",
        "train",
        "route-a",
        "--data-dir",
        str(DATA),
        "--labels",
        str(DATA / "labels"),
        "--char-table",
        str(char_table),
        "--out",
        str(out),
        "--max-steps",
        str(STEPS),
        "--token-budget",
        str(TOKEN_BUDGET),
        "--checkpoint-every",
        "5000",
        "--max-held-out",
        "4096",
        "--seed",
        "0",
    ]
    for shard in HELD_OUT:
        argv += ["--held-out-shard", shard]
    return argv


def run(argv, env, log, out=None):
    """Run *argv*, tee its output to *log*, and return (exit code, output, seconds).

    A run's output directory is cleared first: the metrics file is appended to,
    so a retry over a previous attempt's directory produces one file describing
    two runs, which is a thing that reads as a loss curve and is not one.
    """
    if out is not None:
        shutil.rmtree(out, ignore_errors=True)
    started = time.monotonic()
    finished = subprocess.run(
        argv,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    Path(log).write_text(finished.stdout, encoding="utf-8")
    print(f"--- {log} exited {finished.returncode} after {elapsed:.1f}s ---", flush=True)
    print(finished.stdout[-4000:], flush=True)
    return finished.returncode, finished.stdout, elapsed


def steps_of(metrics):
    """Every logged step of a run, in order."""
    records = [json.loads(line) for line in Path(metrics).read_text().splitlines()]
    return [record for record in records if record.get("event") == "step"]


def config_of(metrics):
    """The config record a run wrote before its first step."""
    records = [json.loads(line) for line in Path(metrics).read_text().splitlines()]
    for record in records:
        if record.get("event") == "config":
            return record
    raise RuntimeError(f"{metrics} has no config record; the run died before it started")


def emit_argv(checkpoint, lattice, out, char_table, context):
    """The command that scores a lattice with a trained checkpoint."""
    return [
        "-m",
        "mlime",
        "train",
        "emit",
        "--checkpoint",
        str(checkpoint),
        "--lattice",
        str(lattice),
        "--out",
        str(out),
        "--char-table",
        str(char_table),
        "--token-budget",
        str(TOKEN_BUDGET),
        "--context" if context else "--no-context",
        "--verbose",
    ]


def main():
    install()
    package_root = importable_package()
    samples, labels = parquet_mounts()
    assets = locate("char_pinyin.tsv", "lattice.jsonl")
    stage_data(samples, labels)
    print(f"samples {samples}\nlabels {labels}\nassets {assets}", flush=True)
    print(
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout,
        flush=True,
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(package_root)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, str(package_root))
    warm_cache()

    char_table = assets / "char_pinyin.tsv"
    out = WORKING / "run"

    launcher = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
    ]
    code, output, seconds = run(
        [*launcher, *train_argv(out, char_table)], env, str(WORKING / "train.log"), out
    )
    if code != 0:
        raise RuntimeError("the training run failed; its log is in the kernel output")

    steps = steps_of(out / "metrics.jsonl")
    ranks = joined_ranks(output)
    checkpoint = out / "checkpoint-final.pt"
    if not checkpoint.is_file():
        raise RuntimeError("the run finished without writing a final checkpoint")

    scored = {}
    for context in (True, False):
        name = "on" if context else "off"
        scores = WORKING / f"scores-context-{name}.jsonl.gz"
        code, _, elapsed = run(
            [
                sys.executable,
                *emit_argv(checkpoint, assets / "lattice.jsonl", scores, char_table, context),
            ],
            env,
            str(WORKING / f"emit-{name}.log"),
        )
        if code != 0:
            raise RuntimeError(f"scoring the lattice with context {name} failed")
        scored[name] = {"bytes": scores.stat().st_size, "seconds": round(elapsed, 1)}

    summary = {
        "world_size": config_of(out / "metrics.jsonl")["world_size"],
        "ranks_that_logged": ranks,
        "steps": steps[-1]["step"],
        "train_seconds": round(seconds, 1),
        "first_loss": steps[0]["loss"],
        "last_loss": steps[-1]["loss"],
        "examples_per_step": round(sum(step["examples"] for step in steps) / len(steps), 1),
        "steps_per_second": round(steps[-1]["step"] / steps[-1]["seconds"], 3),
        "final_gates": steps[-1]["gates"],
        "checkpoints": sorted(path.name for path in out.glob("checkpoint-*.pt")),
        "scores": scored,
    }
    (WORKING / "run-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)

    if summary["world_size"] != 2 or ranks != [0, 1]:
        raise RuntimeError(f"the run used world {summary['world_size']}, ranks {ranks}")


main()
