# Kaggle kernels

The scripts that ran on Kaggle, pulled back with `kaggle kernels pull -m` after the
run so the repository holds what actually executed. Each directory is one kernel:
`kernel.py` is the script and `kernel-metadata.json` its mounts and machine, as
`kaggle kernels push -p <dir>` expects them.

| kernel | what it did |
|---|---|
| `labels-v1` | g2pW labels for the run3 v1 subset, one of three kernels splitting the shards (`KERNEL_INDEX` of `KERNEL_COUNT`) |
| `route-a-v1` | the first full route A run, 2×T4, one epoch, then the eval3 lattice scored with context on and off |
| `labels-v2` | g2pW labels for the rest of run3 (`mlime-run3-rest-samples`), one of five kernels |

## Pushing a sharded kernel

`labels-v2` is one script pushed five times, with `KERNEL_INDEX` and the kernel
name stamped per copy:

```
for i in 0 1 2 3 4; do
  d=$(mktemp -d)
  sed "s/^KERNEL_INDEX = 0$/KERNEL_INDEX = $i/" kaggle/labels-v2/kernel.py > $d/kernel.py
  sed "s/mlime-rest-labels-0/mlime-rest-labels-$i/g" kaggle/labels-v2/kernel-metadata.json > $d/kernel-metadata.json
  kaggle kernels push -p $d
done
```

Kaggle runs two kernels at once per account and allows 30 GPU-hours a week, so
push two, and the next two when those finish.
| `route-a-v2` | two epochs over all of run3 as a chain of segments; segment 0 counts the steps, every segment trains on a wall budget and pauses resumably |
| `e2e` | the end-to-end model of issue #88 (`mlime train e2e`, 2×T4 DDP, wall budget, same segment chain as `route-a-v2`): route A's towers and the transformer reader initialised from `lexoliu/mlime-e2e-init`, decoded by its own beam search with `mlime eval e2e` on the finishing segment. Every segment logs its cards to `gpu.csv` and its trainers' threads and the host's memory to `host.jsonl`; a nonzero `PROFILE_STEPS` (the first `TRACE_STEPS` of them traced) times that many steps after the warm-up and trains on, and `mlime train measure` reads the leg's steady-state rate, utilisation and phase table back from those files; `ACTIVATION_CHECKPOINTING` recomputes the encoder towers when nothing fits. With `INIT_FROM` stamped, the same chain starts from the finished run mounted under that name (`mlime-e2e-final` for v1, or an earlier stage's kernel output) and trains one epoch over `TRAIN_SHARDS`; `WHOLE_ALPHABET` normalises the per-position loss over every emittable character (#121). With `RANKING` stamped as well, it is the ranking stage of #116: it adds the sequence-level ranking loss, and reads segmentations from `ime-segment`, which `colab/e2e.sh start` builds from the leg's commit (`cargo build --release -p ime-pinyin --bin ime-segment --target x86_64-unknown-linux-musl`; `.cargo/config.toml` links it with rust-lld) and uploads beside the inputs |

## Chaining a training run

`route-a-v2` is one script pushed once per segment. Segment 0 mounts the six
datasets in its metadata; segment `n` additionally mounts segment `n-1`'s output
as a kernel source and reads `checkpoint-paused.pt` and `run-config.json` from it:

```
n=1; d=$(mktemp -d)
sed "s/^SEGMENT = 0$/SEGMENT = $n/" kaggle/route-a-v2/kernel.py > $d/kernel.py
python3 -c "
import json,sys; m=json.load(open('kaggle/route-a-v2/kernel-metadata.json')); n=int(sys.argv[1])
m['id']=f'lexoliu/mlime-route-a-v2-s{n}'; m['title']=m['id'].split('/')[1]
m['kernel_sources']=[f'lexoliu/mlime-route-a-v2-s{n-1}']
json.dump(m,open('$d/kernel-metadata.json','w'),indent=2)" $n
kaggle kernels push -p $d
```

A segment that finishes the run scores every `lattice*.jsonl` in the assets (full,
abbreviated and mixed typing), with context on and off, and writes
`run-summary.json` with `"finished": true`; one that pauses says which segment
to push next.

## Colab

`colab/e2e.sh` runs the e2e kernel on a Colab GPU through the
[Colab CLI](https://github.com/googlecolab/google-colab-cli) (`uv tool install
google-colab-cli`, then `colab usage` once to log in). It provisions through
`colab/provision.sh`: `start` refuses a session name that is already running,
and a step that fails after the runtime exists stops that runtime and names the
step, so a failed start leaves nothing billing and can simply be run again.

`colab/e2e.sh <kernel dir> start --session-hours H [--commit REV] [--gpu G|--cpu]
[--census FILE] [--resume-from RUN] [--no-publish]` starts a leg of a stamped
e2e kernel (`kaggle/e2e`, or a run stamped from it with `mlime e2e stamp`).
`mlime e2e stamp mlime-e2e-v2-<card> --out DIR [--train-shard S ...]
[--profile-steps N] [--trace-steps N]` writes a v2 run's directory: v1's recipe
(refused when `kaggle/e2e` no longer holds it), the run's own slug, so its
trios and segments are its own, and the v2 sources alone (the assets, the four
shard repos and `mlime-e2e-init`). With `--init-from <finished v2 run>` the run
is a stage of that run, and `--ranking --rank-*`, `--base-lr --new-lr`,
`--token-budget` and `--accumulate` set the only fields of v1's recipe it
changes; without `--init-from` they are refused. The
code is one commit of this public repository: the VM fetches `--commit`
(default HEAD, refused when this checkout has uncommitted changes or no branch
of origin holds it) from GitHub, the kernel imports `mlime` from that checkout
and records the commit in `run-config.json` and the provenance, and a resume
on a different commit than the run's is refused unless the leg is started
`--allow-commit-change`. `--session-hours` is required: the stager stamps it
as `SESSION_SECONDS`, so the leg pauses inside the session it was given.
`colab/e2e-stage.py`, run from the checkout, downloads, all at once, the private Hugging Face dataset
repos named like its `dataset_sources` and `kernel_sources` (a v1 stage's
`INIT_FROM`), the run's newest trio in the `lexoliu/mlime-resume` bucket, the
newest complete Kaggle segment's output, and for a v2 stage the final
checkpoint and config of its `INIT_FROM` from `final/<run>/` in the bucket
(refused without that run's `complete` marker), then launches the kernel and
`colab/e2e-publish.py`, which puts each interval checkpoint in the bucket so a
lost VM costs minutes. Once a leg has logged fifty steps the publisher also packs
inductor's caches (`TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache`) into the
bucket as `compile-cache/<gpu>-torch<version>.tar`, and the next leg's staging
on the same card restores them, so it trains without compiling again. `--census` hands it a count of the same corpus and batch
shape so the GPU does not count again, and `--cpu` rehearses the staging on a
CPU runtime before a GPU is billed for it. `--resume-from RUN` starts the leg
from another run's newest trio (two cards racing one leg), and `--no-publish`
runs no publisher, for a check that must not enter the run's lineage. The
session is named after the run. `wait --deadline SECONDS` probes the VM every
minute through `colab/leg.py` -- each probe is the watchdog's heartbeat -- and
exits with the leg's end, read from the processes, `run-summary.json` and
`publish-state.json`: 10 paused with its trio published, 11 finished, 12
failed, 13 past the deadline, 14 runtime lost. `stop` refuses while the newest
trio is unpublished or a finished run lives only on the VM (`--force` releases
it anyway); `status`, `fetch <dir>` (the census `batch-counts.json`, the
metrics and both samplers' logs among them) and `publish` follow. The driver uploads only
the Hugging Face access token, never the OAuth refresh token, so no VM can
refresh it out from under the Mac's login.

The stager fetches the Hub repos with `colab/hub.py`: every file of every repo
over its `resolve` URL in one `aria2c` run, 32 files at once. The Hub client's
`snapshot_download` spends about two seconds on each file before its first
byte, and the e2e inputs are some 1,700 shards: staged that way a leg waited
twelve minutes with its GPU billed. Through `aria2c` the same 8.3 GB took 96
seconds on a CPU runtime, and a second run over files already whole 11.

Kernel outputs are fetched with `kaggle/output.py <owner>/<kernel> <dir>
[--pattern REGEX]`, never `kaggle kernels output`: the CLI reads each file
whole into memory, so a multi-gigabyte checkpoint whose connection drops fails
with `IncompleteRead` and can leave an empty file behind, and the output host
cuts each connection after about thirty seconds. `output.py` hands the files
to `aria2c` through `kaggle/aria2.py` (Homebrew or apt `aria2`): sixteen ranged connections per file,
each segment retried where it broke, a partial file continued on the next run.
The chain's harvest uses it too.
