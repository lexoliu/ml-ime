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
| `char-lm` | a character language model over the same 41M lines (`mlime train char-lm`, 2×T4 DDP, wall budget; the kernel's `MODEL` picks the architecture, a 12-layer transformer since `notes/generate-ceiling.md`, the LSTM of `notes/char-lm-v1.md` before it), exported to ONNX (`charlm.onnx`, `prefill.onnx`, the `charlm.weights` shared weights file, `charlm.json`) for `ime-cli fused-eval --lm`. A run longer than one session resumes: push the next session with the previous output added as a kernel source and it continues from that output's `run/charlm.pt` |
| `e2e` | the end-to-end model of issue #88 (`mlime train e2e`, 2×T4 DDP, wall budget, same segment chain as `route-a-v2`): route A's towers and the transformer reader initialised from `lexoliu/mlime-e2e-init`, decoded by its own beam search with `mlime eval e2e` on the finishing segment. A nonzero `SMOKE_STEPS` instead times each `SMOKE_CONFIGS` entry at `TOKEN_BUDGET`, and accumulate 2 at `SMOKE_WIDE_BUDGET`, for that many steps after warm-up, writing rates, peaks and GPU utilisation to `run-summary.json`; `ACTIVATION_CHECKPOINTING` recomputes the encoder towers when nothing fits. With `INIT_FROM` stamped, the same chain starts from the finished run mounted under that name (`mlime-e2e-final` for v1, or an earlier stage's kernel output) and trains one epoch over `TRAIN_SHARDS`; `WHOLE_ALPHABET` normalises the per-position loss over every emittable character (#121). With `RANKING` stamped as well, it is the ranking stage of #116: it adds the sequence-level ranking loss, and reads segmentations from the `ime-segment` binary the package dataset ships at `bin/ime-segment` (`cargo build --release -p ime-pinyin --bin ime-segment --target x86_64-unknown-linux-musl`; `.cargo/config.toml` links it with rust-lld) |

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

## Keeping the chain moving without a session

`kaggle/chain.sh` runs hourly from launchd on the Mac mini
(`~/Library/LaunchAgents/cool.lexo.mlime-chain.plist`, one log line per run in
`data/route-a-v2/chain.log`). Whenever the highest pushed segment is COMPLETE it
downloads that segment's output once into `data/route-a-v2/s<n>`, evaluates the
segment that finished the run with `kaggle/finish.sh` (six fused evals, results
in `s<n>/results.md`), and otherwise pushes the next segment; a refused push
(quota, an expired token) is retried an hour later. Only the notes stay manual.
`notes/HANDOFF.md` is the operator's manual.

The chain is the same for `e2e`, parameterised rather than copied:

```
kaggle/chain.sh --slug lexoliu/mlime-e2e-s --kernel-dir kaggle/e2e \
                --data-dir data/e2e --finish none
```

`--finish none` is because the e2e kernel evaluates in-kernel (the trigram
fusion `finish.sh` performs does not apply): the chain only gathers the
segment's `e2e-*-report.txt` files into `s<n>/results.md`. Bare
`kaggle/chain.sh` is route A v2, unchanged.

A segment that would reach `max_steps` but could not also fit the scoring pauses
early (`SCORING_RESERVE_SECONDS`), so the run always finishes in a segment with
room to score. To spend a partial quota week, stamp `SESSION_SECONDS` smaller at
push time the way `SEGMENT` is stamped.
The chain itself pushes a segment only with `MLIME_SESSION_HOURS` (default 12)
of quota left; set it lower in the job's environment when the next segment
needs less, as the last segment of a run that pauses with an hour to go does.

## The same kernel on Colab

`colab/char-lm.sh` runs `char-lm/kernel.py` unchanged on a Colab GPU through the
[Colab CLI](https://github.com/googlecolab/google-colab-cli) (`uv tool install
google-colab-cli`, then `colab usage` once to log in). `start` provisions the
runtime (an A100 by default), uploads the Hugging Face token and the kernel,
and runs `colab/stage.py` on the VM, which downloads the four datasets from
their private Hugging Face repos into `/kaggle/input/<slug>` and launches the
kernel as a detached process; `status`, `fetch <dir>` and `stop` follow. The kernel sizes itself to
the GPUs it finds (one rank per GPU, `TOKENS_PER_STEP` split across them), so
one A100 trains the same batches as Kaggle's two T4s in roughly a third of the
time, for about 35 compute units.

`colab/e2e.sh <kernel dir> start [--gpu G|--cpu] [--census FILE]` does the
same for a stamped e2e kernel (`kaggle/e2e` or a stage stamped from it):
`colab/e2e-stage.py` downloads, all at once, the private Hugging Face dataset
repos named like its `dataset_sources` and `kernel_sources` (a stage's
`INIT_FROM`), the run's newest trio in the `lexoliu/mlime-resume` bucket, and
the newest complete Kaggle segment's output, then launches the kernel and
`colab/e2e-publish.py`, which puts each interval checkpoint in the bucket so a
lost VM costs minutes. Once a leg has logged fifty steps the publisher also packs
inductor's caches (`TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache`) into the
bucket as `compile-cache/<gpu>-torch<version>.tar`, and the next leg's staging
on the same card restores them, so it trains without compiling again. `--census` hands it a count of the same corpus and batch
shape so the GPU does not count again, and `--cpu` rehearses the staging on a
CPU runtime before a GPU is billed for it. The session is named after the run;
`status`, `fetch <dir>`, `publish` and `stop` follow. Both drivers upload only
the Hugging Face access token, never the OAuth refresh token, so no VM can
refresh it out from under the Mac's login.

Both stagers fetch the Hub repos with `colab/hub.py`: every file of every repo
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
