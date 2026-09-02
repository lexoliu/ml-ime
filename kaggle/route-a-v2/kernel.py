"""Route A v2: two epochs over all of run3, as a chain of Kaggle kernels.

One kernel is one *segment* of the run. Kaggle kills a session at twelve hours
and its output with it, so each segment trains on a wall budget, pauses with a
resumable checkpoint, and the next kernel -- this script again, `SEGMENT` one
higher, the previous kernel's output mounted -- continues from it. The package
refuses a resume whose configuration differs from the checkpoint's, so the chain
is one run or it is an error.

The data is the v1 subset and the rest of run3 side by side: two sample mounts
and two label mounts, staged into one `samples` and one `labels` directory by
per-file symlinks. Their shard names are disjoint by construction (`<source>-
<index>` and `<source>-rest-<index>`), so nothing is renamed and the held-out
shards are the same six v1 shards the v1 run held out.

The step count is not a guess and is not a constant either: the first segment
replays the batcher over every training shard at both epochs (`mlime train
count-batches`) and writes what it found to `run-config.json`; every later
segment reads that file from the previous output. The number has to be fixed
before the first step because the cosine schedule bends on it, and it has to be
exact because overshooting quietly starts a third pass.
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

#: Which segment of the chain this kernel is; stamped per copy at push time.
SEGMENT = 0

#: Passes over the training shards the whole chain makes.
EPOCHS = 2

#: One shard per source, withheld from training and scored at the end -- the same
#: six the v1 run held out, so the two runs' held-out numbers are comparable.
HELD_OUT = (
    "bilibili-00001.parquet",
    "dialogue-00001.parquet",
    "douyin-00001.parquet",
    "moegirl-00001.parquet",
    "news-00001.parquet",
    "wiki-00001.parquet",
)

#: The shard by which each mount is recognised.
V1_MARKER = "dialogue-00000.parquet"
REST_MARKER = "dialogue-rest-00000.parquet"

#: Padded positions per step per rank, both towers together, as in v1.
TOKEN_BUDGET = 16384

#: Kaggle's session cap, and what is kept back from it: the install and staging
#: before training, a two-gigabyte checkpoint written at the end, and the copy of
#: the output directory Kaggle makes after the script exits.
SESSION_SECONDS = 12 * 60 * 60
RESERVE_SECONDS = 35 * 60

#: Numbered checkpoints kept beside the paused one. One is enough: the paused
#: checkpoint is what the next segment resumes from, and every checkpoint is the
#: size of two models.
KEEP_CHECKPOINTS = 1
CHECKPOINT_EVERY = 5000

BASE_MODEL = "hfl/chinese-macbert-base"

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


def parquet_mounts(marker):
    """The samples mount and the labels mount holding *marker*, told apart by columns.

    A label shard is named after the sample shard it labels, so the two mounts
    hold the same file names and only their columns tell them apart.
    """
    import polars as pl

    found = {}
    for directory in directories(INPUTS):
        shard = directory / marker
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
            f"no mount holds the {sorted(missing)} shards for {marker}; mounts hold {describe()}"
        )
    return found["samples"], found["labels"]


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


def stage_data(mounts):
    """One `samples` and one `labels` directory linking every shard of every mount.

    Per-file links rather than per-directory ones because there are two mounts
    per kind and one directory has to hold both. A name that appears twice is a
    staging error, not a tie to break.
    """
    for kind, sources in mounts.items():
        target = DATA / kind
        target.mkdir(parents=True, exist_ok=True)
        for source in sources:
            for shard in sorted(source.glob("*.parquet")):
                link = target / shard.name
                if link.exists():
                    raise FileExistsError(f"{shard.name} is staged twice under {target}")
                link.symlink_to(shard)
        print(f"{kind}: {sum(1 for _ in target.glob('*.parquet'))} shards staged", flush=True)
    return DATA


def warm_cache():
    """Download the base checkpoint once, in one process, before the ranks race for it."""
    from transformers import AutoTokenizer, BertForMaskedLM, BertModel

    AutoTokenizer.from_pretrained(BASE_MODEL)
    BertForMaskedLM.from_pretrained(BASE_MODEL)
    BertModel.from_pretrained(BASE_MODEL, add_pooling_layer=False)
    print("base checkpoint cached", flush=True)


def data_argv(char_table):
    """The options that describe the data, shared by counting and training."""
    argv = [
        "--data-dir",
        str(DATA),
        "--labels",
        str(DATA / "labels"),
        "--char-table",
        str(char_table),
        "--token-budget",
        str(TOKEN_BUDGET),
        "--seed",
        "0",
    ]
    for shard in HELD_OUT:
        argv += ["--held-out-shard", shard]
    return argv


def count_argv(char_table, out):
    """Replay the batcher over the training shards and write the step budget."""
    return [
        sys.executable,
        "-m",
        "mlime",
        "train",
        "count-batches",
        *data_argv(char_table),
        "--world-size",
        "2",
        "--epochs",
        str(EPOCHS),
        "--out",
        str(out),
    ]


def train_argv(char_table, out, max_steps, wall_budget, resume):
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
        "route-a",
        *data_argv(char_table),
        "--out",
        str(out),
        "--max-steps",
        str(max_steps),
        "--checkpoint-every",
        str(CHECKPOINT_EVERY),
        "--keep-checkpoints",
        str(KEEP_CHECKPOINTS),
        "--max-held-out",
        "4096",
        "--wall-budget-seconds",
        str(wall_budget),
    ]
    if resume is not None:
        argv += ["--resume", str(resume)]
    return argv


def emit_argv(checkpoint, lattice, out, char_table, context):
    """The command that scores a lattice with a finished checkpoint."""
    return [
        sys.executable,
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


def previous_segment():
    """The previous kernel's output: its paused checkpoint and its run config."""
    if SEGMENT == 0:
        return None
    mount = locate("checkpoint-paused.pt", "run-config.json")
    return mount / "checkpoint-paused.pt", json.loads((mount / "run-config.json").read_text())


def main():
    started = time.monotonic()
    install()
    package_root = importable_package()
    v1_samples, v1_labels = parquet_mounts(V1_MARKER)
    rest_samples, rest_labels = parquet_mounts(REST_MARKER)
    assets = locate("char_pinyin.tsv", "lattice.jsonl")
    stage_data({"samples": (v1_samples, rest_samples), "labels": (v1_labels, rest_labels)})
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
    shutil.rmtree(out, ignore_errors=True)
    config_path = WORKING / "run-config.json"

    previous = previous_segment()
    if previous is None:
        counts_path = WORKING / "batch-counts.json"
        code, _ = run(count_argv(char_table, counts_path), env, str(WORKING / "count.log"))
        if code != 0:
            raise RuntimeError("counting the batches failed; its log is in the kernel output")
        counts = json.loads(counts_path.read_text())
        config = {"max_steps": int(counts["steps_for_epochs"]), "epochs": EPOCHS}
        resume = None
    else:
        resume, config = previous
    config["segment"] = SEGMENT
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config), flush=True)

    wall_budget = SESSION_SECONDS - RESERVE_SECONDS - (time.monotonic() - started)
    if wall_budget <= 0:
        raise RuntimeError("the session was spent before training could start")
    code, seconds = run(
        train_argv(char_table, out, config["max_steps"], wall_budget, resume),
        env,
        str(WORKING / "train.log"),
    )
    if code != 0:
        raise RuntimeError("the training segment failed; its log is in the kernel output")

    steps = records_of(out / "metrics.jsonl", "step")
    paused = records_of(out / "metrics.jsonl", "paused")
    summary = {
        "segment": SEGMENT,
        "max_steps": config["max_steps"],
        "resumed_from": None if resume is None else str(resume),
        "first_step": steps[0]["step"],
        "last_step": steps[-1]["step"],
        "first_loss": steps[0]["loss"],
        "last_loss": steps[-1]["loss"],
        "train_seconds": round(seconds, 1),
        "finished": not paused,
        "gates": steps[-1]["gates"],
    }

    if paused:
        if not (out / "checkpoint-paused.pt").is_file():
            raise RuntimeError("the segment paused without writing checkpoint-paused.pt")
        shutil.move(out / "checkpoint-paused.pt", WORKING / "checkpoint-paused.pt")
        print(
            f"segment {SEGMENT} paused at step {summary['last_step']} of {config['max_steps']}; "
            f"push segment {SEGMENT + 1} with this kernel's output mounted",
            flush=True,
        )
    else:
        checkpoint = out / "checkpoint-final.pt"
        if not checkpoint.is_file():
            raise RuntimeError("the run finished without writing a final checkpoint")
        lattices = sorted(assets.glob("lattice*.jsonl"))
        if not lattices:
            raise FileNotFoundError(f"no lattice*.jsonl under {assets}")
        scored = {}
        for lattice in lattices:
            for context in (True, False):
                name = f"{lattice.stem}-context-{'on' if context else 'off'}"
                scores = WORKING / f"scores-{name}.jsonl.gz"
                code, elapsed = run(
                    emit_argv(checkpoint, lattice, scores, char_table, context),
                    env,
                    str(WORKING / f"emit-{name}.log"),
                )
                if code != 0:
                    raise RuntimeError(f"scoring {lattice.name} with context {name} failed")
                scored[name] = {"bytes": scores.stat().st_size, "seconds": round(elapsed, 1)}
        summary["scores"] = scored
        summary["held_out"] = records_of(out / "metrics.jsonl", "summary")[-1]

    (WORKING / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


main()
