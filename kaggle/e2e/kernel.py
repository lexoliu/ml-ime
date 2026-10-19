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
assets mount with `mlime eval e2e`, context on and off, one evaluation per GPU
at a time, and writes each record's hypotheses and the report alongside the
checkpoint. A segment that mounts a run whose training already finished --
`run/checkpoint-final.pt` in its output, the evaluation having stopped before
its report -- evaluates that checkpoint and trains nothing.

A stage is the same chain with `INIT_FROM` stamped: the first segment starts
from the finished run mounted under that name -- v1's dataset
`lexoliu/mlime-e2e-final`, or the kernel output of an earlier stage -- with its
weights, a fresh optimiser and schedule, and its one epoch runs over
`TRAIN_SHARDS`. With `RANKING` stamped as well it is a ranking stage (#116):
every segment adds the sequence-level ranking loss, reading each example's
other segmentations from the `ime-segment` binary the package dataset ships.
`WHOLE_ALPHABET` normalises the per-position loss over every emittable
character (#121). A finished run is told apart from the run it started from
by the stage its run-config records.

A leg measures itself while it trains: `GpuSampler` and `CpuSampler` log the
cards and the trainers' threads to `gpu.csv` and `cpu.jsonl` beside the run,
and with `PROFILE_STEPS` stamped the loop times that many steps after its
warm-up and trains on. `mlime train measure` turns the files into the leg's
steady-state rate, utilisation and phase table.
"""

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, SimpleQueue

INPUTS = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA = WORKING / "data"

#: How deep the mount namespace is walked. `datasets/<user>/<slug>` is three.
MAX_DEPTH = 4

#: Which segment of the chain this kernel is; stamped per copy at push time.
SEGMENT = 0

#: The `source_digest` of the `mlime` package this kernel was stamped against,
#: or None to run whatever is mounted. `datasets status` reports a new
#: `mlime-src` version ready before a kernel pushed right after it is sure
#: to mount it, and a kernel on the previous version can run without a
#: complaint; with the digest stamped it refuses instead.
SOURCE_DIGEST = None

#: Steps after the compile warm-up whose phases the loop times, and how many
#: of the first of them it also records as a torch.profiler trace
#: (`run/trace-rank<r>.json`, ~60 MB a step). The run trains on past the
#: window; `mlime train measure` reads it back with the samplers' logs.
PROFILE_STEPS = 0
TRACE_STEPS = 0

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
#: largest budget the smoke sweep measured with headroom. A ranking run counts
#: every reading's fill row and every ranked path's decoder row in it.
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

#: The ranking stage of #116: None trains #88's objective from route A and the
#: reader; a dict of `mlime train e2e --rank-*` settings (`segmentations`,
#: `beam`, `negatives`, `weight`) starts from the finished v1 run instead and
#: adds the sequence-level ranking loss. Stamped per push.
RANKING = None

#: The mount a stage starts from: a finished run's dataset or kernel-output
#: slug, such as "mlime-e2e-final" for v1. Its weights start the run with a
#: fresh optimiser and schedule; None starts from route A and the reader. A
#: ranking stage always names one. Stamped per push.
INIT_FROM = None

#: Normalise the decoder's per-position loss over every emittable character
#: rather than the span's candidates (#121). Stamped per push.
WHOLE_ALPHABET = False

#: The training shards a run makes its epoch over; None trains on every
#: mounted shard but the held-out ones. Stamped per push.
TRAIN_SHARDS = None

#: The (pretrained, new) learning rates; None keeps `mlime train e2e`'s.
LEARNING_RATES = None

#: The segmenter binary the ranking loss pipes keystrokes through, shipped
#: inside the package dataset.
SEGMENTER = "ime-segment"

#: Corrupt each example's typed keystrokes with the shared typo noise model.
#: Off by default so the running chain stays on clean input until a pushed
#: copy turns it on deliberately.
TYPOS = False

#: The three files a leg hands the next one.
RESUME_MARKERS = ("checkpoint-paused.pt", "run-config.json", "run-summary.json")

#: Inductor's caches, outside the working directory a Kaggle segment publishes
#: as output. A Colab leg's publisher packs them into the resume bucket and
#: the next leg's staging restores them here, so it skips compiling (#140).
INDUCTOR_CACHE = Path("/tmp/inductor-cache")

REQUIREMENTS = (
    "polars",
    "pydantic-settings",
    "pypinyin",
    "structlog",
    "opencc",
    "rapidfuzz",
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


def source_digest(package):
    """sha256 over a package's `.py` files: each relative path, then its bytes."""
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def importable_package():
    """A directory that can go on `PYTHONPATH` and make `mlime` importable.

    The marker is `train/e2e.py`, not just the package: a `mlime-src` dataset
    pushed before this branch exists still imports, and then the first `e2e`
    command fails for the wrong reason. With `SOURCE_DIGEST` set, a mounted
    package that is not the one stamped is refused.
    """
    try:
        root = locate("mlime/train/e2e.py")
    except FileNotFoundError:
        mount = locate("train/e2e.py", "__init__.py")
        root = WORKING / "packages"
        root.mkdir(parents=True, exist_ok=True)
        link = root / "mlime"
        if not link.exists():
            link.symlink_to(mount, target_is_directory=True)
    digest = source_digest(root / "mlime")
    print(f"mlime source {digest}", flush=True)
    if SOURCE_DIGEST is not None and digest != SOURCE_DIGEST:
        raise RuntimeError(
            f"the mounted mlime source is {digest}, not the stamped {SOURCE_DIGEST}: "
            "the kernel was pushed before the new mlime-src version was attached"
        )
    return root


def libcuda_link_dir():
    """A directory holding `libcuda.so`, for the compiled wrappers' link step.

    The training compiles its graphs' wrappers to C++, which links with
    `-lcuda`. The image ships the driver only as `libcuda.so.<version>`,
    found through the loader's path rather than ldconfig's cache, and no
    unversioned name, so the link fails. The library the loader resolves is
    linked here under the name the linker asks for.
    """
    ctypes.CDLL("libcuda.so.1")
    with open("/proc/self/maps", encoding="utf-8") as maps:
        real = next(line.split()[-1] for line in maps if "/libcuda.so." in line)
    directory = Path(tempfile.gettempdir()) / "libcuda"
    directory.mkdir(exist_ok=True)
    link = directory / "libcuda.so"
    if not link.exists():
        link.symlink_to(real)
    return directory


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
    for shard in TRAIN_SHARDS or ():
        argv += ["--train-shard", shard]
    return argv


def count_argv(char_table, out, segmenter, census=None):
    """Replay the batcher over the training shards and write the step budget.

    *census* is a mounted count from an earlier run; `count-batches` uses it
    only when every configuration field and the shard list equal this run's
    and says which field differed otherwise. A ranking run's batches weigh
    each example by every row the ranking step makes of it, so the count
    reads the staged *segmenter* too.
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
        *ranking_argv(segmenter),
    ]
    if census is not None:
        argv += ["--reuse-census", str(census)]
    return argv


def start_argv(initials, init_e2e):
    """Where a new run's weights come from: route A and the reader, or a finished run."""
    if INIT_FROM is None:
        return ["--init-encoder", str(initials[0]), "--init-decoder", str(initials[1])]
    return ["--init-e2e", str(init_e2e)]


def stage():
    """What tells this kernel's runs apart, as its run-config records it."""
    return {"ranking": RANKING, "whole_alphabet": WHOLE_ALPHABET, "init_from": INIT_FROM}


def stage_of(config):
    """The stage a mounted run-config records.

    v1's run-config predates the stage fields and records none of them; it is
    the plain run from route A and the reader.
    """
    return {
        "ranking": config.get("ranking"),
        "whole_alphabet": config.get("whole_alphabet", False),
        "init_from": config.get("init_from"),
    }


def ranking_argv(segmenter):
    """The ranking loss's switches, the same on every segment of a ranking run."""
    if RANKING is None:
        return []
    argv = ["--ranking", "--segmenter", str(segmenter)]
    for name, value in RANKING.items():
        argv += [f"--rank-{name}", str(value)]
    return argv


def train_argv(
    char_table,
    out,
    max_steps,
    wall_budget,
    start,
    resume,
    token_budget,
    max_held_out=4096,
    prefetch=PREFETCH,
    accumulate=ACCUMULATE,
    profile_steps=0,
    trace_steps=0,
    compile=COMPILE,
    compile_mode=COMPILE_MODE,
    nproc=WORLD,
    virtual_ranks=1,
    checkpoint_minutes=CHECKPOINT_MINUTES,
    log_every=10,
    segmenter=None,
):
    """The training command the ranks of this segment run.

    *nproc* is how many processes the launcher's T4s host and
    *virtual_ranks* how many ranks each one stands for: on Kaggle two
    processes of one rank, on Colab one process of two, and the checkpoint
    does not know the difference. *start* is `start_argv`'s for a new run and
    None for a resumed one; *segmenter* is the staged binary a ranking run
    reads.
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
    if trace_steps:
        argv += ["--trace-steps", str(trace_steps)]
    if wall_budget is not None:
        argv += ["--wall-budget-seconds", str(wall_budget)]
    if checkpoint_minutes:
        argv += ["--checkpoint-minutes", str(checkpoint_minutes)]
    if start is not None:
        argv += start
    if LEARNING_RATES is not None:
        argv += ["--base-lr", str(LEARNING_RATES[0]), "--new-lr", str(LEARNING_RATES[1])]
    argv += ranking_argv(segmenter)
    if WHOLE_ALPHABET:
        argv.append("--whole-alphabet")
    if resume is not None:
        argv += ["--resume", str(resume)]
    if ACTIVATION_CHECKPOINTING:
        argv.append("--activation-checkpointing")
    if TYPOS:
        argv.append("--typos")
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
    kernel's own output (which the chain mounts) and as the trio a Colab leg
    publishes to the `lexoliu/mlime-resume` bucket, which the Colab staging
    downloads under `<run>-resume`. A segment's own
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


def drop_numbered_checkpoints(out):
    """Delete a finished run's numbered checkpoints, leaving `checkpoint-final.pt`.

    They exist to recover a crash inside the session. Once the final one is
    written they are 3.5 GB each of kernel output that every download of the
    run pays for and nothing reads.
    """
    for numbered in out.glob("checkpoint-[0-9]*.pt"):
        numbered.unlink()


def finished_runs(wanted):
    """The final checkpoints of the mounted finished runs of stage *wanted*.

    A run's run-config records its stage; that is how a stage that starts
    from v1 tells its own finished run from the v1 run it starts from, both
    mounted at once.
    """
    return [
        mount / "run" / "checkpoint-final.pt"
        for mount in locate_all("run/checkpoint-final.pt", "run-config.json")
        if stage_of(json.loads((mount / "run-config.json").read_text())) == wanted
    ]


def finished_checkpoint():
    """The final checkpoint of a mounted run of this kernel's kind, or None.

    Two such mounts would be two finished runs, and which one to evaluate is
    not this kernel's to guess.
    """
    found = finished_runs(stage())
    if len(found) > 1:
        raise RuntimeError(f"{len(found)} finished runs are mounted: {found}")
    return found[0] if found else None


def init_checkpoint():
    """The final checkpoint of the finished run mounted as `INIT_FROM`; exactly one.

    A dataset mounts under its slug and a kernel output under its kernel's, so
    the name is one component of the mount's path.
    """
    found = [
        mount / "run" / "checkpoint-final.pt"
        for mount in locate_all("run/checkpoint-final.pt", "run-config.json")
        if INIT_FROM in mount.relative_to(INPUTS).parts
    ]
    if len(found) != 1:
        raise FileNotFoundError(
            f"a stage starts from one finished run mounted as {INIT_FROM}, found {found}; "
            f"mounts hold {describe()}"
        )
    return found[0]


def stage_segmenter():
    """The package dataset's `ime-segment`, copied out of the read-only mount, executable."""
    target = WORKING / "bin" / SEGMENTER
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(locate(SEGMENTER) / SEGMENTER, target)
    target.chmod(0o755)
    return target


def evaluate_lattices(checkpoint, assets, char_table, env):
    """`mlime eval e2e` over every lattice, context on and off, one per GPU at a time.

    Each worker owns one GPU and takes the next section off a shared queue,
    so a slow section -- the abbreviated lattice admits seven times the
    candidates of the full one -- does not leave the other GPU idle.
    """
    lattices = sorted(assets.glob("lattice*.jsonl"))
    if not lattices:
        raise FileNotFoundError(f"no lattice*.jsonl under {assets}")
    sections = SimpleQueue()
    for lattice in lattices:
        suffix = lattice.stem[len("lattice") :]
        eval_set = assets / f"eval3{suffix}.jsonl"
        if not eval_set.is_file():
            raise FileNotFoundError(
                f"no eval set at {eval_set} for {lattice.name}; the two come in pairs"
            )
        for context in (True, False):
            sections.put((lattice, eval_set, context))

    def worker(gpu):
        worker_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpu)}
        evaluated = {}
        while True:
            try:
                lattice, eval_set, context = sections.get_nowait()
            except Empty:
                return evaluated
            name = f"{lattice.stem}-context-{'on' if context else 'off'}"
            dump = WORKING / f"e2e-{name}.jsonl"
            report = WORKING / f"e2e-{name}-report.txt"
            code, elapsed = run(
                eval_argv(checkpoint, lattice, eval_set, dump, report, char_table, context),
                worker_env,
                str(WORKING / f"eval-{name}.log"),
            )
            if code != 0:
                raise RuntimeError(f"evaluating {name} on GPU {gpu} failed")
            evaluated[name] = {
                "bytes": dump.stat().st_size,
                "seconds": round(elapsed, 1),
                "gpu": gpu,
            }

    gpus = gpu_count()
    evaluated = {}
    with ThreadPoolExecutor(gpus) as pool:
        for part in pool.map(worker, range(gpus)):
            evaluated.update(part)
    return evaluated


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
    if RANKING is not None and INIT_FROM is None:
        raise RuntimeError("a ranking stage starts from a finished run; stamp INIT_FROM")
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
    env["LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (str(libcuda_link_dir()), env.get("LIBRARY_PATH")))
    )
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(INDUCTOR_CACHE)
    sys.path.insert(0, str(package_root))
    warm_cache()

    char_table = assets / "char_pinyin.tsv"
    out = WORKING / "run"
    shutil.rmtree(out, ignore_errors=True)
    config_path = WORKING / "run-config.json"

    segmenter = stage_segmenter() if RANKING is not None else None

    def start():
        if INIT_FROM is not None:
            return start_argv(None, init_checkpoint())
        initials = locate(INIT_ENCODER, INIT_DECODER)
        return start_argv((initials / INIT_ENCODER, initials / INIT_DECODER), None)

    final = finished_checkpoint()
    if final is not None:
        print(f"evaluating the finished run at {final}; nothing to train", flush=True)
        summary = {
            "segment": SEGMENT,
            "evaluated_checkpoint": str(final),
            "finished": True,
            "evaluated": evaluate_lattices(final, assets, char_table, env),
        }
        (WORKING / "run-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    previous = previous_segment()
    if previous is None:
        counts_path = WORKING / "batch-counts.json"
        code, _ = run(
            count_argv(char_table, counts_path, segmenter, find_census()),
            env,
            str(WORKING / "count.log"),
        )
        if code != 0:
            raise RuntimeError("counting the batches failed; its log is in the kernel output")
        counts = json.loads(counts_path.read_text())
        config = {
            "max_steps": int(counts["steps_for_epochs"]),
            "epochs": EPOCHS,
            **stage(),
            "train_shards": TRAIN_SHARDS,
            "learning_rates": LEARNING_RATES,
        }
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
        raise RuntimeError(f"a world of {world} ranks does not split evenly over {nproc} processes")
    virtual = world // nproc
    print(f"world of {world}: {nproc} processes x {virtual} virtual ranks", flush=True)

    wall_budget = training_budget(time.monotonic() - started, config["max_steps"], previous, nproc)
    if wall_budget <= 0:
        raise RuntimeError("the session was spent before training could start")
    # The package is importable from here on (`sys.path` above); its samplers
    # watch the whole segment, so any span of it can be measured afterwards.
    from mlime.train.measure import CpuSampler, GpuSampler

    if PROFILE_STEPS:
        # A recompile inside the window costs an idle card; dynamo's reason
        # for each lands in the log beside the step that paid it.
        env["TORCH_LOGS"] = "recompiles"
    with GpuSampler(WORKING / "gpu.csv"), CpuSampler(WORKING / "cpu.jsonl"):
        code, seconds = run(
            train_argv(
                char_table,
                out,
                config["max_steps"],
                wall_budget,
                start() if resume is None else None,
                resume,
                TOKEN_BUDGET,
                profile_steps=PROFILE_STEPS,
                trace_steps=TRACE_STEPS,
                nproc=nproc,
                virtual_ranks=virtual,
                segmenter=segmenter,
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
        drop_numbered_checkpoints(out)
        evaluated = evaluate_lattices(checkpoint, assets, char_table, env)
        summary["evaluated"] = evaluated
        summary["held_out"] = records_of(out / "metrics.jsonl", "summary")[-1]

    (WORKING / "run-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
