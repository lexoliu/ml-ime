"""E2E s0: the end-to-end model of issue #88, as a chain of Kaggle kernels.

The chain works exactly as `route-a-v2`'s: one kernel is one segment, each
segment trains on a wall budget and pauses to `checkpoint-paused.pt`, and the
next copy of this script -- `SEGMENT` one higher, the previous kernel's output
mounted -- continues it. The step count is replayed on segment 0 by the same
`count-batches` pass and written to `run-config.json`, so the cosine schedule
is fixed before the first step.

What differs is where the first segment starts: not from `hfl/chinese-macbert-
base`'s pretrained weights but from two finished checkpoints mounted with
`lexoliu/mlime-e2e-init` -- `route-a-v2-final.pt` for the encoder's towers and
`charlm-transformer-s1.pt` for the decoder's transformer. Both are located by
name under /kaggle/input; a mount without them is a configuration error, not
something to initialise around.

A segment that finishes the run evaluates it on every `lattice*.jsonl` in the
assets mount with `mlime eval e2e`, context on and off, and writes each
record's hypotheses and the report alongside the checkpoint.

For a smoke run -- the few-minute rate check that the cost estimate is made
from -- set `SMOKE_STEPS` nonzero: the kernel times each entry of
`SMOKE_CONFIGS` for that many steps after a warm-up at `TOKEN_BUDGET`, one
subprocess per configuration, and writes the rates, the peaks, the mean GPU
utilisation and the per-phase timing table to `run-summary.json`.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA = WORKING / "data"

#: How deep the mount namespace is walked. `datasets/<user>/<slug>` is three.
MAX_DEPTH = 4

#: Which segment of the chain this kernel is; stamped per copy at push time.
SEGMENT = 0

#: Set nonzero for a smoke run: each configuration in `SMOKE_CONFIGS` gets
#: this many timed steps after its warm-up, and the whole table lands in
#: run-summary.json.
SMOKE_STEPS = 0

#: A compiled run warms up until dynamo has added no graph for five
#: consecutive steps; this is the cap on that, since a dimension that still
#: specialises at that point keeps doing so. `max_steps` gets the cap plus
#: the timed steps -- the loop leaves on its own once the window is full.
SMOKE_WARMUP_CAP = 30

#: The configurations the smoke times against each other at `TOKEN_BUDGET`:
#: prefetch and compile on throughout, accumulate 2 against accumulate 4.
#: Accumulate 2 then also runs at `SMOKE_WIDE_BUDGET`.
SMOKE_CONFIGS = (
    ("compile-acc2", {"prefetch": True, "accumulate": 2, "compile": True}),
    ("compile-acc4", {"prefetch": True, "accumulate": 4, "compile": True}),
)

#: The wider budget accumulate-2 gets a timed run at. The gate is that run's
#: own timed peak: a 60-step probe was shown to under-report what a long
#: segment sees, so it no longer stands in.
SMOKE_WIDE_BUDGET = 12288

#: A configuration is safe while its timed peak stays at or under this share of
#: the smallest card, and competitive while its mean GPU utilisation reaches
#: `SMOKE_UTILISATION_FLOOR`; `recommended` says which it found.
SMOKE_PEAK_CEILING = 0.88
SMOKE_UTILISATION_FLOOR = 80.0

#: Recompute the encoder towers' layers in backward when even the smallest
#: budget cannot fit -- the kernel flag for it, off until the sweep says so.
ACTIVATION_CHECKPOINTING = False

#: Collate the next batches on a producer thread while the step runs, for the
#: real chain too; smoke v3's table is what this is checked against.
PREFETCH = True

#: Micro-batches an optimiser step accumulates over. The step count, the
#: schedule and `count-batches` all divide by it; smoke v6 measured 178.1
#: ex/s per rank at 94% utilisation with 4 against 162.2 with 2.
ACCUMULATE = 4

#: torch.compile the towers and the decoder on the real segments; smoke v6
#: showed zero recompiles and zero graph breaks in the timed window, so the
#: marked-dims compile is steady enough to be the default.
COMPILE = True
COMPILE_MODE = "default"

#: Passes over the training shards the whole chain makes.
EPOCHS = 1

#: One shard per source, withheld from training and scored at the end -- the
#: same six every run of this data held out, so the numbers stay comparable.
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

#: The files `lexoliu/mlime-e2e-init` holds: route A v2's final weights and the
#: transformer reader's.
INIT_ENCODER = "route-a-v2-final.pt"
INIT_DECODER = "charlm-transformer-s1.pt"

#: Weight of the encoder head's loss inside the decoder's total.
AUX_WEIGHT = 0.3

#: Padded positions per micro-batch per rank, both towers together -- the
#: largest budget the smoke sweep measured with headroom.
TOKEN_BUDGET = 8192

#: Kaggle's session cap, and what is kept back from it. `SESSION_SECONDS` is
#: the stamp a push overrides for a shorter session -- segment 0 ran with
#: `115 * 60` stamped into the pushed copy -- and the pause lands inside it
#: including the compile time, which is part of the segment's own clock.
SESSION_SECONDS = 12 * 60 * 60
RESERVE_SECONDS = 35 * 60

#: What the finishing segment needs after its last step: the held-out
#: evaluation and a beam decode of each lattice in both contexts. The decode is
#: autoregressive where emit's was one forward pass per path, so it is slower
#: than v2's scoring -- the reserve covers it.
SCORING_RESERVE_SECONDS = 3 * 60 * 60

#: Numbered checkpoints kept beside the paused one.
KEEP_CHECKPOINTS = 1
CHECKPOINT_EVERY = 5000

#: The world this run trains in: Kaggle segments run it as two processes and a
#: Colab leg as one process of two virtual ranks, and the checkpoints the two
#: legs write carry every rank's position either way, so they resume each
#: other. A fresh run trains in this world; a resumed one keeps the world's
#: size the checkpoint's positions record.
WORLD = 2

#: Minutes between out-of-band checkpoint trios inside the segment, in
#: addition to the step cadence and the wall-budget pause. A Kaggle leg is
#: never surprised -- its pause lands before the session dies -- so the
#: default is none; the Colab driver stamps 20 into the copy it uploads,
#: because a Colab VM can die without warning and the interval trio is what
#: its publisher ships.
CHECKPOINT_MINUTES = 0

#: The three files a leg hands the next one.
RESUME_MARKERS = ("checkpoint-paused.pt", "run-config.json", "run-summary.json")

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


def locate_all(*markers):
    """Every mounted directory holding every one of *markers*."""
    if not INPUTS.is_dir():
        raise FileNotFoundError(f"{INPUTS} does not exist; the kernel has no inputs at all")
    return [
        directory
        for directory in directories(INPUTS)
        if all((directory / marker).exists() for marker in markers)
    ]


def locate(*markers):
    """The first mounted directory holding every one of *markers*.

    For callers whose markers name one thing -- a table, a shard, a package --
    one match is the right one and the rest are the same files staged twice.
    Checkpoint trios are different: two mounts can hold real and different
    ones, and they go through `previous_segment`, which chooses by step.
    """
    found = locate_all(*markers)
    if not found:
        raise FileNotFoundError(f"no mounted directory holds {markers}; mounts hold {describe()}")
    return found[0]


def find_census():
    """A mounted `batch-counts.json` from an earlier count, if one was mounted.

    The census dataset saves counting the corpus again when this run's
    configuration equals the count's -- `count-batches --reuse-census` checks
    that, and counts the corpus itself when even one field differs.
    """
    if not INPUTS.is_dir():
        return None
    for directory in directories(INPUTS):
        candidate = directory / "batch-counts.json"
        if candidate.is_file():
            return candidate
    return None


def parquet_mounts(marker):
    """The samples mount and the labels mount holding *marker*, told apart by columns."""
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
    """A directory that can go on `PYTHONPATH` and make `mlime` importable.

    The marker is `train/e2e.py`, not just the package: a `mlime-src` dataset
    pushed before this branch exists still imports, and then the first `e2e`
    command fails for the wrong reason.
    """
    try:
        return locate("mlime/train/e2e.py")
    except FileNotFoundError:
        mount = locate("train/e2e.py", "__init__.py")
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
    """One `samples` and one `labels` directory linking every shard of every mount."""
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
    """Download the base model's small files once, before the ranks race for them.

    Only the config and the tokenizer: both towers' *weights* come from the
    initial checkpoints, so the pretrained tensors are never fetched.
    """
    from transformers import AutoTokenizer, BertConfig

    BertConfig.from_pretrained("hfl/chinese-macbert-base")
    AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
    print("base files cached", flush=True)


def data_argv(char_table, token_budget=TOKEN_BUDGET):
    """The options that describe the data, shared by counting and training."""
    argv = [
        "--data-dir",
        str(DATA),
        "--labels",
        str(DATA / "labels"),
        "--char-table",
        str(char_table),
        "--token-budget",
        str(token_budget),
        "--seed",
        "0",
    ]
    for shard in HELD_OUT:
        argv += ["--held-out-shard", shard]
    return argv


def count_argv(char_table, out, census=None):
    """Replay the batcher over the training shards and write the step budget.

    *census* is a mounted count from an earlier run; `count-batches` uses it
    only when every configuration field and the shard list equal this run's
    and says which field differed otherwise.
    """
    argv = [
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
        "--accumulate",
        str(ACCUMULATE),
        "--out",
        str(out),
    ]
    if census is not None:
        argv += ["--reuse-census", str(census)]
    return argv


def train_argv(
    char_table,
    out,
    max_steps,
    wall_budget,
    initials,
    resume,
    token_budget,
    max_held_out=4096,
    prefetch=PREFETCH,
    accumulate=ACCUMULATE,
    profile_steps=0,
    compile=COMPILE,
    compile_mode=COMPILE_MODE,
    nproc=WORLD,
    virtual_ranks=1,
    checkpoint_minutes=CHECKPOINT_MINUTES,
    log_every=10,
):
    """The training command the ranks of this segment run.

    *nproc* is how many processes the launcher's T4s host and
    *virtual_ranks* how many ranks each one stands for: on Kaggle two
    processes of one rank, on Colab one process of two, and the checkpoint
    does not know the difference.
    """
    argv = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "-m",
        "mlime",
        "train",
        "e2e",
        *data_argv(char_table, token_budget),
        "--out",
        str(out),
        "--aux-weight",
        str(AUX_WEIGHT),
        "--max-steps",
        str(max_steps),
        "--checkpoint-every",
        str(CHECKPOINT_EVERY),
        "--keep-checkpoints",
        str(KEEP_CHECKPOINTS),
        "--max-held-out",
        str(max_held_out),
        "--accumulate",
        str(accumulate),
        "--virtual-ranks",
        str(virtual_ranks),
        "--log-every",
        str(log_every),
        # The kernel's own config file, copied verbatim into interval trios.
        "--run-config",
        str(WORKING / "run-config.json"),
    ]
    if prefetch:
        argv.append("--prefetch")
    if compile:
        argv += ["--compile", "--compile-mode", str(compile_mode)]
    if profile_steps:
        argv += ["--profile-steps", str(profile_steps)]
    if wall_budget is not None:
        argv += ["--wall-budget-seconds", str(wall_budget)]
    if checkpoint_minutes:
        argv += ["--checkpoint-minutes", str(checkpoint_minutes)]
    if initials is not None:
        argv += ["--init-encoder", str(initials[0]), "--init-decoder", str(initials[1])]
    if resume is not None:
        argv += ["--resume", str(resume)]
    if ACTIVATION_CHECKPOINTING:
        argv.append("--activation-checkpointing")
    return argv


def eval_argv(checkpoint, lattice, eval_set, dump, report, char_table, context):
    """The command that beam-searches one lattice in one context setting."""
    return [
        sys.executable,
        "-m",
        "mlime",
        "eval",
        "e2e",
        "--checkpoint",
        str(checkpoint),
        "--lattice",
        str(lattice),
        "--eval-set",
        str(eval_set),
        "--char-table",
        str(char_table),
        "--slice",
        "all",
        "--context" if context else "--no-context",
        "--dump",
        str(dump),
        "--out",
        str(report),
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


def peak_memory(log_text):
    """Per-rank peak CUDA bytes, from the peak-memory lines the ranks print."""
    peaks = {}
    for line in log_text.splitlines():
        if '"peak-memory"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "peak-memory":
            peaks[int(record["rank"])] = int(record["bytes"])
    return peaks


def profile_phases(log_text):
    """Each rank's per-phase timing table, from the profile lines it prints.

    Beyond the phase stats the record carries ``warmup_steps`` (how long the
    adaptive warm-up took), ``window`` (the wall-clock bounds of the timed
    steps, for slicing the GPU samples) and ``recompiles``/``graph_breaks``
    (what dynamo did inside the window -- the compiled run's assertion of
    steadiness).
    """
    tables = {}
    for line in log_text.splitlines():
        if '"profile"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "profile":
            tables[int(record["rank"])] = {
                "steps": record["steps"],
                "warmup_steps": record["warmup_steps"],
                "window": record["window"],
                "recompiles": record["recompiles"],
                "graph_breaks": record["graph_breaks"],
                "phases": record["phases"],
            }
    return tables


def dynamo_counters(log_text):
    """Each rank's `torch._dynamo` counters, from the lines a compile run prints.

    `graph_breaks` is the Counter of reasons dynamo fell back to eager; the
    rest count cache hits, recompilations and frames. A run without --compile
    emits no such line and reads as absent.
    """
    counters = {}
    for line in log_text.splitlines():
        if '"dynamo"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "dynamo":
            counters[int(record["rank"])] = record["counters"]
    return counters


class GpuSampler:
    """`nvidia-smi -lms` running beside a subprocess: per-GPU utilisation.

    The check is the driver's, not torch's: a step that leaves the card idle
    while the host collates shows up here even though no device code knows
    about it. 500 ms is fine enough to catch a step boundary and coarse enough
    to keep the log small.
    """

    def __init__(self, interval_ms=500):
        self._argv = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,timestamp",
            "--format=csv,noheader,nounits",
            "-lms",
            str(interval_ms),
        ]
        self._process = None
        self.samples = {}

    def __enter__(self):
        self._process = subprocess.Popen(
            self._argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return self

    def __exit__(self, *_):
        self._process.terminate()
        try:
            out, _ = self._process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
            out, _ = self._process.communicate()
        for line in out.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            gpu, util, memory = int(fields[0]), int(fields[1]), int(fields[2])
            epoch = datetime.strptime(fields[3], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            self.samples.setdefault(
                gpu, {"utilization": [], "memory_mib": [], "times": []}
            )
            self.samples[gpu]["utilization"].append(util)
            self.samples[gpu]["memory_mib"].append(memory)
            self.samples[gpu]["times"].append(epoch)
        return False

    def means(self, since=None, until=None):
        """Mean utilisation and memory per GPU over the samples in the window.

        ``since``/``until`` are wall-clock epochs -- the profile record's
        ``window`` bounds -- so a run's mean covers its timed steps only, not
        the warm-up its compilation idles through.
        """
        out = {}
        for gpu, v in sorted(self.samples.items()):
            rows = [
                (u, m)
                for t, u, m in zip(
                    v["times"], v["utilization"], v["memory_mib"], strict=True
                )
                if (since is None or t >= since) and (until is None or t <= until)
            ]
            out[str(gpu)] = {
                "utilization": round(sum(u for u, _ in rows) / len(rows), 1)
                if rows
                else None,
                "memory_mib": round(sum(m for _, m in rows) / len(rows))
                if rows
                else None,
            }
        return out


def steps_per_second(steps):
    """Steps/s from the step records' own clock: last minus first over the span."""
    if len(steps) < 2:
        return None
    span = steps[-1]["seconds"] - steps[0]["seconds"]
    if span <= 0:
        return None
    return round((steps[-1]["step"] - steps[0]["step"]) / span, 3)


def gpu_total_bytes():
    """The smallest card's memory: the per-rank ceiling the headroom rule reads."""
    import torch

    return min(
        torch.cuda.get_device_properties(index).total_memory
        for index in range(torch.cuda.device_count())
    )


def smoke_run(char_table, init_files, env, name, token_budget, options, steps, timed):
    """One smoke subprocess: *steps* optimiser steps, the last *timed* measured.

    A configuration's run is a process of its own so an OOM neither corrupts
    nor misattributes the next configuration's measurement. Rates come off the
    timed window's own step records; `GpuSampler` brackets the run so a step
    that idles the card shows in the utilisation mean, and the loop's
    profiler -- on for the timed steps only -- splits them into phases.
    """
    run_dir = WORKING / f"smoke-{name}"
    log_file = WORKING / f"smoke-{name}.log"
    argv = train_argv(
        char_table,
        run_dir,
        steps,
        None,
        init_files,
        None,
        token_budget,
        max_held_out=256,
        prefetch=options.get("prefetch", False),
        accumulate=options.get("accumulate", 1),
        profile_steps=timed,
        compile=options.get("compile", False),
        compile_mode=options.get("compile_mode", "default"),
        log_every=1,
    )
    with GpuSampler() as sampler:
        code, seconds = run(argv, env, str(log_file))
    text = log_file.read_text(encoding="utf-8")
    profiles = profile_phases(text)
    # GPU means cover the timed window only: the profile record's epoch bounds
    # cut the sampler's log, so warm-up -- and the compilation inside it --
    # does not dilute the number the utilisation floor reads.
    bounds = [p["window"] for p in profiles.values() if len(p.get("window") or []) == 2]
    if bounds:
        gpu_mean = sampler.means(
            since=min(bound[0] for bound in bounds),
            until=max(bound[1] for bound in bounds),
        )
    else:
        gpu_mean = sampler.means()
    entry = {
        "name": name,
        "token_budget": token_budget,
        "prefetch": options.get("prefetch", False),
        "accumulate": options.get("accumulate", 1),
        "compile": options.get("compile", False),
        "seconds": round(seconds, 1),
        "gpu_mean": gpu_mean,
        "peak_bytes_per_rank": {
            str(rank): size for rank, size in sorted(peak_memory(text).items())
        },
        "phases": {str(rank): t for rank, t in sorted(profiles.items())},
        "recompiles": {
            str(rank): table.get("recompiles")
            for rank, table in sorted(profiles.items())
        },
        "graph_breaks": {
            str(rank): table.get("graph_breaks")
            for rank, table in sorted(profiles.items())
        },
        "dynamo": {str(rank): c for rank, c in sorted(dynamo_counters(text).items())},
    }
    if code != 0:
        entry["status"] = (
            "oom" if code == 3 or "out of memory" in text.lower() else "failed"
        )
    else:
        step_records = records_of(run_dir / "metrics.jsonl", "step")
        timed_records = step_records[-timed:] if timed else []
        entry["status"] = "ok"
        entry["timed_steps"] = len(timed_records)
        if len(timed_records) > 1:
            span = timed_records[-1]["seconds"] - timed_records[0]["seconds"]
            entry["steps_per_second"] = steps_per_second(timed_records)
            if span > 0:
                entry["examples_per_second_per_rank"] = round(
                    sum(step["examples"] for step in timed_records[1:]) / span, 1
                )
                entry["padded_tokens_per_second_per_rank"] = round(
                    sum(
                        step["tokens"] + step["context_tokens"]
                        for step in timed_records[1:]
                    )
                    / span,
                    1,
                )
        if step_records:
            # The first step carries the compile latency when one runs; as a
            # warm step its rate is excluded from every timing either way.
            entry["first_step_seconds"] = step_records[0]["seconds"]
            if options.get("compile", False):
                entry["compile_seconds"] = step_records[0]["seconds"]
            entry["first_loss"] = step_records[0]["loss"]
            entry["last_loss"] = step_records[-1]["loss"]
    print(json.dumps(entry), flush=True)
    shutil.rmtree(run_dir, ignore_errors=True)
    return entry


def rate_of(entry):
    """What "fastest" reads: padded tokens/s per rank over the timed window."""
    return entry.get("padded_tokens_per_second_per_rank") or 0.0


def peak_of(entry):
    """An entry's worst per-rank peak in bytes; None when it never ran far."""
    peaks = entry.get("peak_bytes_per_rank") or {}
    return max(peaks.values()) if peaks else None


def utilisation_of(entry):
    """An entry's worst-GPU mean utilisation; None when it was never sampled."""
    means = entry.get("gpu_mean") or {}
    values = [m["utilization"] for m in means.values() if m["utilization"] is not None]
    return min(values) if values else None


def settings_of(entry):
    """The switches a recommended entry would carry into the real segments."""
    return {
        "token_budget": entry["token_budget"],
        "prefetch": entry["prefetch"],
        "accumulate": entry["accumulate"],
        "compile": entry["compile"],
        "after": entry["name"],
    }


def smoke(char_table, out, init_files, env):
    """Time each `SMOKE_CONFIGS` entry at `TOKEN_BUDGET`, then acc2 wide.

    The wider run is always accumulate 2 -- the question it answers is memory,
    not rate -- and it is counted only if its own timed peak fits the ceiling.
    `recommended` is the fastest counted entry whose utilisation reaches the
    floor; without one it is the fastest memory-safe entry, marked so.
    """
    entries = [
        smoke_run(
            char_table,
            init_files,
            env,
            name,
            TOKEN_BUDGET,
            options,
            SMOKE_WARMUP_CAP + SMOKE_STEPS,
            SMOKE_STEPS,
        )
        for name, options in SMOKE_CONFIGS
    ]

    ceiling = gpu_total_bytes() * SMOKE_PEAK_CEILING

    def peak_ok(entry):
        peak = peak_of(entry)
        return peak is not None and peak <= ceiling

    wide = smoke_run(
        char_table,
        init_files,
        env,
        f"compile-acc2-{SMOKE_WIDE_BUDGET}",
        SMOKE_WIDE_BUDGET,
        {"prefetch": True, "accumulate": 2, "compile": True},
        SMOKE_WARMUP_CAP + SMOKE_STEPS,
        SMOKE_STEPS,
    )
    entries.append(wide)
    measured = [entry for entry in entries if entry["status"] == "ok"]

    # A run counts toward the recommendation only when it finished inside the
    # memory ceiling -- the wide run's timed peak is the gate, so a peak over
    # the share still shows in the table but cannot be recommended.
    for entry in entries:
        entry["counted"] = entry["status"] == "ok" and peak_ok(entry)

    summary = {
        "segment": SEGMENT,
        "smoke": True,
        "token_budget": TOKEN_BUDGET,
        "warmup_cap": SMOKE_WARMUP_CAP,
        "timed_steps": SMOKE_STEPS,
        "configurations": entries,
    }
    if not measured:
        (WORKING / "run-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            "every smoke configuration failed; the table is in run-summary.json"
        )

    counted = [entry for entry in measured if entry["counted"]]
    fed = [
        entry
        for entry in counted
        if (utilisation_of(entry) or 0.0) >= SMOKE_UTILISATION_FLOOR
    ]
    if fed:
        recommended = max(fed, key=rate_of)
        summary["recommended"] = settings_of(recommended)
    elif counted:
        recommended = max(counted, key=rate_of)
        summary["recommended"] = {
            **settings_of(recommended),
            "note": (
                f"no configuration reached {SMOKE_UTILISATION_FLOOR}% GPU "
                "utilisation; this is the fastest that fits the card"
            ),
        }
    else:
        recommended = max(measured, key=rate_of)
        summary["recommended"] = {
            **settings_of(recommended),
            "note": "every measured configuration peaked over the memory ceiling",
        }
    (WORKING / "run-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def torch_load(path):
    """torch.load with memory-mapping when this torch has it, for header reads."""
    import inspect

    import torch

    options = {}
    if "mmap" in inspect.signature(torch.load).parameters:
        options["mmap"] = True
    return torch.load(path, map_location="cpu", weights_only=False, **options)


def resume_candidates():
    """Every mounted trio that could resume this run, with the step and the
    world read off the checkpoint itself, not off its summary.

    A leg's checkpoint reaches the next segment two ways: inside the previous
    kernel's own output (which the chain mounts) and inside the
    `mlime-e2e-resume-{a,b}` slots a Colab leg publishes to. A segment's own
    interval checkpoints are found here too, inside the output's `run/`
    directory -- same run, older steps, so the pick by step passes over them.
    """
    found = []
    for mount in locate_all(*RESUME_MARKERS):
        state = torch_load(mount / "checkpoint-paused.pt")
        found.append(
            {
                "mount": mount,
                "checkpoint": mount / "checkpoint-paused.pt",
                "config": json.loads((mount / "run-config.json").read_text()),
                "summary": json.loads((mount / "run-summary.json").read_text()),
                "step": int(state["step"]),
                "world": len(state["positions"]),
            }
        )
    return found


def previous_segment():
    """The newest resumable checkpoint mounted, or None for a run not started.

    What is mounted decides fresh-or-resume, not SEGMENT: a re-pushed segment
    0 resumes the trio a Colab leg published just as a later segment would,
    and the stamp stays only a label on this leg's output. The pick is by the
    step the checkpoint itself records, so a Colab leg's progress is never
    silently discarded in favour of the Kaggle output that predates it. Trios
    whose shared run-config fields disagree describe different runs -- the
    refusal `refuse_mismatch` would make inside the checkpoint -- and are
    refused the same way, by field name. The segment stamp is exempt: it
    differs by construction.
    """
    candidates = resume_candidates()
    if not candidates:
        if SEGMENT == 0:
            return None
        raise FileNotFoundError(
            f"segment {SEGMENT} found no resumable trio mounted; mounts hold {describe()}"
        )
    shared = set.intersection(*(set(c["config"]) for c in candidates)) - {"segment"}
    differing = sorted(
        key
        for key in shared
        if len({json.dumps(c["config"].get(key), sort_keys=True) for c in candidates}) > 1
    )
    if differing:
        raise ValueError(
            "the mounted checkpoints disagree on the run: "
            + ", ".join(differing)
            + " differ between "
            + ", ".join(str(c["mount"]) for c in candidates)
        )
    picked = max(candidates, key=lambda c: c["step"])
    print(
        "resumable checkpoints: "
        + ", ".join(f"{c['mount']} at step {c['step']}" for c in candidates)
        + f"; resuming {picked['mount']}",
        flush=True,
    )
    return picked


def gpu_count():
    """How many accelerators this kernel can spawn a rank on; at least one."""
    import torch

    return max(1, torch.cuda.device_count())


def training_budget(elapsed, max_steps, previous, processes):
    """How long this segment may train.

    The rate the previous leg measured holds only for a leg on the same
    number of devices -- a Colab T4 runs at half the Kaggle pair's rate.
    A summary without the field, or one from a different process count,
    leaves the plain session budget in place with a line in the log.
    """
    budget = SESSION_SECONDS - RESERVE_SECONDS - elapsed
    if previous is None:
        return budget
    summary = previous["summary"]
    if summary.get("processes") != processes:
        print(
            f"the last leg trained on {summary.get('processes')} processes, this one has "
            f"{processes}; its rate does not transfer, using the plain budget",
            flush=True,
        )
        return budget
    steps = summary["last_step"] - summary["first_step"] + 1
    rate = steps / summary["train_seconds"]
    predicted = (max_steps - summary["last_step"]) / rate
    if predicted <= budget < predicted + SCORING_RESERVE_SECONDS:
        print(
            f"finishing here would take {predicted:.0f}s of {budget:.0f}s and leave no room to "
            f"score; pausing after {budget - SCORING_RESERVE_SECONDS:.0f}s instead",
            flush=True,
        )
        return budget - SCORING_RESERVE_SECONDS
    return budget


def main():
    started = time.monotonic()
    install()
    package_root = importable_package()
    v1_samples, v1_labels = parquet_mounts(V1_MARKER)
    rest_samples, rest_labels = parquet_mounts(REST_MARKER)
    assets = locate("char_pinyin.tsv", "lattice.jsonl")
    initials = locate(INIT_ENCODER, INIT_DECODER)
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

    if SMOKE_STEPS:
        init_files = (initials / INIT_ENCODER, initials / INIT_DECODER)
        smoke(char_table, WORKING / "run", init_files, env)
        return

    previous = previous_segment()
    if previous is None:
        counts_path = WORKING / "batch-counts.json"
        code, _ = run(
            count_argv(char_table, counts_path, find_census()),
            env,
            str(WORKING / "count.log"),
        )
        if code != 0:
            raise RuntimeError("counting the batches failed; its log is in the kernel output")
        counts = json.loads(counts_path.read_text())
        config = {"max_steps": int(counts["steps_for_epochs"]), "epochs": EPOCHS}
        resume = None
    else:
        resume, config = previous["checkpoint"], previous["config"]
    config["segment"] = SEGMENT
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config), flush=True)

    # The world the run trains in is the checkpoint's when resuming and WORLD
    # when starting; it splits into processes -- one per accelerator this
    # kernel can see, but never more than it has ranks -- and each process
    # stands for however many ranks are left over. On Kaggle that is two
    # processes of one rank each; on Colab's single T4, one process of two.
    world = WORLD if previous is None else previous["world"]
    nproc = min(gpu_count(), world)
    if world % nproc:
        raise RuntimeError(
            f"a world of {world} ranks does not split evenly over {nproc} processes"
        )
    virtual = world // nproc
    print(f"world of {world}: {nproc} processes x {virtual} virtual ranks", flush=True)

    init_files = (
        (initials / INIT_ENCODER, initials / INIT_DECODER)
        if resume is None
        else None
    )
    wall_budget = training_budget(
        time.monotonic() - started, config["max_steps"], previous, nproc
    )
    if wall_budget <= 0:
        raise RuntimeError("the session was spent before training could start")
    code, seconds = run(
        train_argv(
            char_table,
            out,
            config["max_steps"],
            wall_budget,
            init_files,
            resume,
            TOKEN_BUDGET,
            nproc=nproc,
            virtual_ranks=virtual,
        ),
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
        # A logged step is the last *recorded* one; the step the checkpoint
        # holds is the pause event's own.
        "checkpoint_step": paused[-1]["step"] if paused else steps[-1]["step"],
        "first_loss": steps[0]["loss"],
        "last_loss": steps[-1]["loss"],
        "train_seconds": round(seconds, 1),
        "finished": not paused,
        "world_size": world,
        "processes": nproc,
        "virtual_ranks": virtual,
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
        evaluated = {}
        for lattice in lattices:
            suffix = lattice.stem[len("lattice"):]
            eval_set = assets / f"eval3{suffix}.jsonl"
            if not eval_set.is_file():
                raise FileNotFoundError(
                    f"no eval set at {eval_set} for {lattice.name}; the two come in pairs"
                )
            for context in (True, False):
                name = f"{lattice.stem}-context-{'on' if context else 'off'}"
                dump = WORKING / f"e2e-{name}.jsonl"
                report = WORKING / f"e2e-{name}-report.txt"
                code, elapsed = run(
                    eval_argv(checkpoint, lattice, eval_set, dump, report, char_table, context),
                    env,
                    str(WORKING / f"eval-{name}.log"),
                )
                if code != 0:
                    raise RuntimeError(f"evaluating {lattice.name} with context {name} failed")
                evaluated[name] = {"bytes": dump.stat().st_size, "seconds": round(elapsed, 1)}
        summary["evaluated"] = evaluated
        summary["held_out"] = records_of(out / "metrics.jsonl", "summary")[-1]

    (WORKING / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
