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
| `char-lm` | a character language model over the same 41M lines (`mlime train char-lm`, 2×T4 DDP, wall budget; the kernel's `MODEL` picks the architecture, a 12-layer transformer since `notes/generate-ceiling.md`, the LSTM of `notes/char-lm-v1.md` before it), exported to ONNX (`charlm.onnx`, `prefill.onnx`, `charlm.json`) for `ime-cli fused-eval --lm`. A run longer than one session resumes: push the next session with the previous output added as a kernel source and it continues from that output's `run/charlm.pt` |

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

A segment that would reach `max_steps` but could not also fit the scoring pauses
early (`SCORING_RESERVE_SECONDS`), so the run always finishes in a segment with
room to score. To spend a partial quota week, stamp `SESSION_SECONDS` smaller at
push time the way `SEGMENT` is stamped.

## The same kernel on Colab

`colab/char-lm.sh` runs `char-lm/kernel.py` unchanged on a Colab GPU through the
[Colab CLI](https://github.com/googlecolab/google-colab-cli) (`uv tool install
google-colab-cli`, then `colab usage` once to log in). `start` provisions the
runtime (an A100 by default), uploads the Kaggle credentials and the kernel,
and runs `colab/stage.py` on the VM, which downloads the four datasets with the
Kaggle API into `/kaggle/input/<slug>` and launches the kernel as a detached
process; `status`, `fetch <dir>` and `stop` follow. The kernel sizes itself to
the GPUs it finds (one rank per GPU, `TOKENS_PER_STEP` split across them), so
one A100 trains the same batches as Kaggle's two T4s in roughly a third of the
time, for about 35 compute units.
