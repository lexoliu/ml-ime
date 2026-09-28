# Route A in Rust (2026-09-28)

Issue #81, part of the macOS input method (#80). Before this, route A's
context encoder and fill decoder ran only in Python. `mlime train emit` wrote
score files, and `fused-eval` read them. An input method cannot wait for a
score file, so the towers now run inside the Rust decoder over ONNX Runtime.

## What exists

- **`mlime export route-a <checkpoint> --out <dir> [--quantize int8]`**
  writes the following files:
  - `context.onnx`, the context tower, run once per record;
  - `fill.onnx`, the fill decoder over every reading of the lattice, which
    returns log probabilities under the lexicon's candidate mask;
  - the weights of both as external data;
  - `tokenizer.json`;
  - `route-a.json`, a manifest with `layout: "towers"` that names every
    tensor and carries the span and emission vocabularies and the checkpoint's
    step.
- **`crates/ime-neural` `RouteA`** loads that directory:
  - its manifest parser refuses unknown keys and other layouts;
  - it keeps one memory-mapped copy of the weights, shared by every thread's
    sessions;
  - fp32 sessions also share pre-packed matrices;
  - backends are CPU, or Core ML, WebGPU and CUDA behind features, and a
    provider that fails is an error.
- **`RouteA::emission`** returns raw `f32` log probabilities for each reading,
  position and asked candidate. Rounding to the score file's four decimals
  happens only in the file writer, `examples/emit.rs`.
- **`fused-eval --route-a <dir>`** replaces `--scores` with live emissions.
  - `--no-context` gives the context-off twin.
  - `--route-a-threads` sets the intra-op threads per session.
  - Each record's emissions are computed once per slice and reused across
    every `--weight` and the dev and test passes.

## Identity with the Python score file

`python/scripts/routea_identity.py` checks the eval3 dev slice (498 records,
fp32 export of route A v2 at step 244,797). It diffs every candidate score
of a Rust-emitted file against `mlime train emit`'s. It then decodes the
slice twice, neural only, once from the file and once live, and compares the
two record by record.

| | candidate slots | max \|Δ\| | top-1 flips | top-1 / top-8 / char |
| --- | --- | --- | --- | --- |
| context on | 1,920,990 | 4.28e-4 | 0 | 0.6908 / 0.7008 / 0.9407 |
| context off | | 1.15e-4 | 0 | 0.5924 / 0.6084 / 0.9057 |

The file was emitted by PyTorch and the live scores by ONNX Runtime. The
residual is therefore the file's rounding plus fp32 drift between the two
engines. The script fails past 1e-3 on any slot, or on any flip.

The fixture test in `crates/ime-neural/tests` holds the fp32 export within
1e-4 of the recorded table on every platform. The int8 export is held within
0.05 of the fp32 table, because int8 kernels differ between x86 and arm64.
The Python side pins `onnxruntime==1.28.0`, the runtime `ort` links.

## int8 against fp32

The int8 export decodes 0.61 points lower on top-1 with context on, at
0.6847. With context off it is 0.20 points lower. Python ONNX Runtime on the
int8 graph reproduces the Rust output bit for bit, so this is quantization
error and not the loader.

## Cost per record

The cost of one record is one context forward plus one fill forward over its
readings. It was measured on the eval3 dev slice, on Apple CPU on a Devin
macOS VM:

| export | threads | median | p95 |
| --- | --- | --- | --- |
| fp32, context on | 1 | 56.1 ms | 85.7 ms |
| fp32, context on | 4 | 44.2 ms | 65.7 ms |
| fp32, context off | 1 | 36.8 ms | 60.7 ms |
| int8, context on | 1 | 95.2 ms | 177.1 ms |
| int8, context on | 4 | 91.8 ms | 165.4 ms |
| int8, context off | 1 | 62.2 ms | 113.6 ms |

int8 is about 1.7 times slower than fp32 on this CPU. At one sentence the
batch is at most eight readings, and dequantization costs more than NEON fp32
matrix multiplication saves. For the input method, fp32 is the export to use
on Apple CPU.

## Reproduce

```
mlime export route-a data/route-a-v2/s6/run/checkpoint-final.pt --out data/route-a-export/fp32
mlime export route-a data/route-a-v2/s6/run/checkpoint-final.pt --out data/route-a-export/int8 --quantize int8
uv run --project python python/scripts/routea_identity.py \
  --eval-set data/run3_pool/eval3.jsonl --lattice data/route-a-assets-v2/lattice.jsonl \
  --scores data/route-a-v2/s6/scores-lattice-context-on.jsonl.gz \
  --export data/route-a-export/fp32 --emittable data/route-a-assets-v2/emittable.txt
ime-cli fused-eval --route-a data/route-a-export/fp32 --model data/run3/ngram.bin \
  --eval-set data/run3_pool/eval3.jsonl --emittable data/route-a-assets-v2/emittable.txt --select-on-dev
```

## Next

The per-record cost above is for a whole sentence at once. Issue #82 measures
per-keystroke latency on the M1 with incremental decoding, and the cost of
the towers is the first thing it has to bring down.
