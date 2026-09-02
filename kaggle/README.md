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
