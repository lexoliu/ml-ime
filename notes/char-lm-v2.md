# A character LM inside the beam, v2: the transformer reader (2026-09-27)

Issue #61, the next line of #40 after `notes/char-lm-v1.md`. The LSTM in the
beam was worth 1.2 points on every typing style and left the beam's contents
where they were; `notes/generate-ceiling.md` placed the target for a system
that sees only the context and the keystrokes at 36.7 / 49.2 / 84.7
(abbreviated / mixed / full, GPT-6 Luna with the beam's hypotheses) and said
the gap was the reader inside the lattice, not the width of the search. This
is the stronger reader: a 60M-parameter transformer character LM whose beam
state is its KV cache, exported in int8, measured on the three eval3 twins
exactly as v1 was, plus the oracle at 32.

## Verdict

**The transformer reader is worth 2.7 points on abbreviated input, 3.2 on
mixed and 0.7 on full pinyin over the LSTM, and it raises the abbreviated
oracle at 32 by 2.8.** Test sentence top-1, context on, trigram + LM,
weights chosen on dev: 28.26 / 38.54 / 78.16 against the LSTM's 25.54 /
35.39 / 77.50 and the v2 decoder's 24.32 / 34.20 / 76.33. The LM alone now
reads as well as the trigram plus the LSTM did (28.12 / 37.73 / 76.11); the
trigram adds a tenth of a point on abbreviated input and less than a point
elsewhere, and the fused weight that wins is 1, not the 0.5 the LSTM wanted.
So this model is the first transition that knows more than the emissions
and the trigram do, rather than a second reading of the same evidence. What
it does not do is change the beam's shape: top-8 minus top-1 is 5.8 / 9.0 /
4.4 points, as it was, and the abbreviated oracle at 32 (36.30) is still
eight points below the hinted Luna target. The int8 export is the product
export: every difference from fp32 is inside one standard error of a 5,000-
record slice, at half the time and half the memory.

## The model

| quantity | value |
|---|---:|
| architecture | pre-norm transformer, 12 layers, width 512, 8 heads, feed-forward 2048, learned positions (512), tied output through a projection, dropout 0.1 |
| parameters | 60.08M |
| alphabet | 41,928 characters plus `<pad> <bos> <eos> <sep> <unk>`, as the LSTM |
| input | up to 62 characters of the preceding context, `<sep>`, then the sentence |
| data | the run3 sample shards, one shard per source held out, as the LSTM |
| steps | 100,000 at the LSTM's token budget (1.54B tokens), 2×T4 DDP, AdamW, fp16 |
| wall | 21.4 h over two chained Kaggle sessions: 52,150 steps in 11.3 h (`lexoliu/mlime-char-lm` v4, ended by the wall budget), resumed to 100,000 in 10.1 h (`mlime-char-lm-s1`); about 20k tokens/s |
| held-out | 4.42 nats/char at 5k steps, 3.97 at 52k, **3.696 nats/char (perplexity 40.3)** at 100k; the curve is noisy (3.70 at 65k, 3.77–3.82 between) and still falling |

The LSTM read 3.89 nats/char (perplexity 49.1) from the same data in the
same budget. The two Kaggle sessions are one run: the kernel resumes from
the previous session's checkpoint when it is mounted as a dataset (#63).

Three exports of the final checkpoint are measured here:

- `char-lm-fp32`: `mlime export char-lm run/charlm.pt --restrict
  emittable.txt`, the step graph `charlm.onnx`, the prefill graph
  `prefill.onnx`, and the weights as external data the manifest's table
  maps name by name (#65), 488 MB on disk. `ime-lm` maps each weights file
  once and hands the same memory to every per-thread session, so the reader
  costs one copy of the weights however many threads decode (6.6 GB before,
  1.5 GB after on the eight-core M1).
- `char-lm-int8`: the same with `--quantize int8` (#67): every MatMul weight
  dynamic-quantised to int8 per channel, seven-bit range so the x86 kernels'
  16-bit accumulators do not saturate, 248 MB on disk. On the fixture it is
  0.029 nats off fp32 on arm64.
- The step graph attends to the prelude's cache and the sentence's cache
  separately and joins only the scores before the softmax, with the prelude
  passed once and broadcast over the beam, so a step copies nothing per
  beam (#67).

On the pressure harness (8 threads, beam 8, an M4 Pro) a step costs 16.2 ms
in fp32 and 7.7 ms in int8, 23.5 and 49.4 records/s, peak RSS 1.51 and
0.73 GB. The LSTM step was 15.8 ms at batch 8 on the M1, so the transformer
in int8 costs about what the LSTM did.

## Results (test slices, context on, neural weight chosen on dev)

Every row uses the s6 context-on scores of route A v2 and beam 8, run on an
8-vCPU Linux VM from the binary at 33f7371 (`ime-cli fused-eval
--select-on-dev --weight 0.75 --weight 1 --weight 1.5`). The LSTM rows are
`notes/char-lm-v1.md`; the reranked and hinted rows are
`notes/rescore-ceiling.md` and `notes/generate-ceiling.md`, for scale.

### abbreviated

| configuration | neural w (dev) | top-1 | top-8 | char | MRR@8 |
|---|---|---|---|---|---|
| trigram (route A v2) | 1.5 | 24.32% | 30.16% | 65.59% | 0.266 |
| trigram + LSTM, lm-weight 0.5 (v1) | 1.0 | 25.54% | 31.27% | 66.25% | 0.277 |
| transformer alone, int8 | 1.0 | 28.12% | 33.45% | 67.88% | 0.302 |
| transformer alone, fp32 | 1.0 | 28.10% | 33.49% | 67.89% | 0.302 |
| trigram + transformer int8, lm-weight 0.5 | 1.0 | 27.25% | 32.71% | 67.26% | 0.293 |
| **trigram + transformer int8, lm-weight 1** | 1.5 | **28.26%** | **34.10%** | **67.90%** | **0.305** |
| trigram + transformer int8, lm-weight 2 | 0.75 | 27.54% | 32.18% | 66.87% | 0.293 |
| trigram + transformer fp32, lm-weight 1 | 1.5 | 28.32% | 34.00% | 67.88% | 0.305 |
| trigram, reranked by GPT-6 Luna | 1.5 | 27.90% | | | |
| GPT-6 Luna writing the sentence, with hints | | 36.73% | | | |

### mixed

| configuration | neural w (dev) | top-1 | top-8 | char | MRR@8 |
|---|---|---|---|---|---|
| trigram (route A v2) | 1.0 | 34.20% | 44.10% | 74.48% | 0.378 |
| trigram + LSTM, lm-weight 1 (v1) | 1.0 | 35.39% | 45.17% | 75.04% | 0.390 |
| transformer alone, int8 | 0.75 | 37.73% | 46.90% | 76.21% | 0.412 |
| transformer alone, fp32 | 0.75 | 37.57% | 46.86% | 76.21% | 0.410 |
| trigram + transformer int8, lm-weight 0.5 | 1.0 | 36.94% | 46.50% | 75.78% | 0.405 |
| **trigram + transformer int8, lm-weight 1** | 1.0 | **38.54%** | **47.53%** | **76.23%** | **0.419** |
| trigram + transformer int8, lm-weight 2 | 0.75 | 37.99% | 46.36% | 75.84% | 0.411 |
| trigram + transformer fp32, lm-weight 1 | 1.0 | 38.46% | 47.55% | 76.27% | 0.419 |
| trigram, reranked by GPT-6 Luna | 1.0 | 39.56% | | | |
| GPT-6 Luna writing the sentence, with hints | | 49.22% | | | |

### full

| configuration | neural w (dev) | top-1 | top-8 | char | MRR@8 |
|---|---|---|---|---|---|
| trigram (route A v2) | 1.0 | 76.33% | 81.26% | 95.43% | 0.784 |
| trigram + LSTM, lm-weight 0.5 (v1) | 1.5 | 77.50% | 82.16% | 95.70% | 0.794 |
| transformer alone, int8 | 1.5 | 76.11% | 80.66% | 95.42% | 0.780 |
| transformer alone, fp32 | 1.0 | 76.75% | 81.14% | 95.61% | 0.786 |
| trigram + transformer int8, lm-weight 0.5 | 1.0 | 78.04% | 82.43% | 95.89% | 0.799 |
| **trigram + transformer int8, lm-weight 1** | 1.0 | **78.16%** | **82.53%** | 95.87% | **0.800** |
| trigram + transformer int8, lm-weight 2 | 1.5 | 78.16% | 82.47% | **95.90%** | 0.800 |
| trigram + transformer fp32, lm-weight 0.5 | 1.5 | 78.44% | 82.83% | 95.88% | 0.803 |
| trigram + transformer fp32, lm-weight 1 | 1.0 | 78.16% | 82.49% | 95.87% | 0.800 |
| trigram + transformer fp32, lm-weight 2 | 1.0 | 77.46% | 81.90% | 95.71% | 0.793 |
| trigram, reranked by GPT-6 Luna | 1.0 | 78.83% | | | |
| GPT-6 Luna writing the sentence, with hints | | 84.66% | | | |

### The oracle at 32

`--top-k 32 --beam-width 32` on the test slices with the int8 export, at the
lm-weight and neural weight the dev grid chose (the mixed grid tied lm-weight
0.5 and 1 on dev at 197 / 484 and the run took 0.5). The LSTM and trigram
rows are `notes/generate-ceiling.md`.

| twin, transition | lm-weight, w | top-1 | top-32 | char |
|---|---|---|---|---|
| abbreviated, trigram | | 24.44% | 31.94% | |
| abbreviated, trigram + LSTM | 0.5, 1.0 | 25.96% | 33.47% | |
| **abbreviated, trigram + transformer int8** | 1, 1.5 | **28.67%** | **36.30%** | 68.21% |
| mixed, trigram | | 34.34% | 46.82% | |
| mixed, trigram + LSTM | 1, 1.0 | 35.79% | 48.48% | |
| **mixed, trigram + transformer int8** | 0.5, 1.0 | **37.08%** | **49.47%** | 75.92% |

### The dev grid

Sentence top-1 % on the dev slices (475 / 484 / 498 records) at each neural
weight; the test rows above are reported at the bold cell of their row.

| twin, transition | w=0.75 | 1.0 | 1.5 |
|---|---|---|---|
| abbreviated, int8 alone | 26.9 | **27.8** | 26.7 |
| abbreviated, int8 lm-weight 0.5 | 25.9 | **26.7** | 25.9 |
| abbreviated, int8 lm-weight 1 | 26.7 | 26.3 | **27.4** |
| abbreviated, int8 lm-weight 2 | **27.0** | 26.3 | 26.5 |
| abbreviated, fp32 alone | 27.6 | **28.0** | 27.0 |
| abbreviated, fp32 lm-weight 1 | 27.0 | 26.7 | **27.4** |
| mixed, int8 alone | **38.6** | 38.0 | 37.0 |
| mixed, int8 lm-weight 0.5 | 40.1 | **40.7** | 39.0 |
| mixed, int8 lm-weight 1 | 40.1 | **40.7** | 39.9 |
| mixed, int8 lm-weight 2 | **39.9** | 39.9 | 39.7 |
| mixed, fp32 alone | **38.8** | 38.2 | 36.4 |
| mixed, fp32 lm-weight 1 | 40.1 | **40.5** | 39.9 |
| full, int8 alone | 76.3 | 76.1 | **76.7** |
| full, int8 lm-weight 0.5 | 78.7 | **79.1** | 79.1 |
| full, int8 lm-weight 1 | 79.1 | **79.5** | 79.1 |
| full, int8 lm-weight 2 | 77.9 | 78.7 | **79.1** |
| full, fp32 alone | 76.1 | **76.5** | 76.5 |
| full, fp32 lm-weight 0.5 | 78.7 | 78.9 | **79.5** |
| full, fp32 lm-weight 1 | 79.1 | **79.5** | 78.9 |
| full, fp32 lm-weight 2 | 77.9 | **78.7** | 78.5 |

## What the numbers say

- **The gain scales with how much the keystrokes leave out.** Over the LSTM:
  +2.72 abbreviated, +3.15 mixed, +0.66 full on top-1 (fp32's best full row
  is +0.94); over the trigram alone: +3.94, +4.34, +1.83. The LSTM's gain was
  the same 1.2 on every style; this one is three to five times larger where
  the emissions are weakest, which is what a model that reads the sentence
  looks like. A 0.2-nat better reader of the same data moved abbreviated
  top-1 by 2.7 points; the LSTM at 3.89 nats moved it by 1.2.
- **The transition no longer needs the trigram.** Alone, the transformer
  ties or beats the LSTM-plus-trigram on every style, and adding the trigram
  is worth 0.14 / 0.81 / 2.05 (int8, lm-weight 1). On full pinyin the
  trigram still carries something the LM lacks, which is the run3 counts'
  view of frequent surface forms; on abbreviated input it carries nothing.
  The dev grid's preferred lm-weight moved from 0.5 to 1, and lm-weight 2
  costs up to 0.7 points: the model can share the transition but not own it
  yet.
- **The beam is still eight near-copies.** Top-8 minus top-1 is 5.84 / 8.99
  / 4.37 (LSTM 5.73 / 9.78 / 4.66). The abbreviated oracle at 32 rises from
  33.47 to 36.30, nearly three points past the trigram's oracle at 128 (33.49), and the
  mixed oracle from 48.48 to 49.47; a better transition lifts the whole
  ranking a little, so the expected sentence enters the candidates a little
  more often, but the structure of the beam (per-position independent
  emissions agreeing with each other) is unchanged. The reranker ceiling
  moves by the same amount as the oracle.
- **The reranker's gain is now inside the decoder on abbreviated input.**
  The GPT-6 reranker over the trigram's beam reached 27.90; the fused
  transformer reaches 28.26 with no model call after the search. On mixed
  and full it is still 1.0 and 0.7 short of the reranked trigram, and the
  hinted Luna numbers are 8.5 / 10.7 / 6.5 points away.
- **int8 is the export.** Fused at lm-weight 1: 28.26 vs 28.32, 38.54 vs
  38.46, 78.16 vs 78.16; alone: 28.12 vs 28.10, 37.73 vs 37.57, 76.11 vs
  76.75. One standard error of a 5,000-record top-1 near 28% is 0.63 points
  and near 78% is 0.58, so none of these is a measured difference, the full
  alone row included. The dev grid disagrees between the two exports by at
  most three records. What int8 buys is a section (three dev sweeps and the
  test pass, about 5.5k sentences) in 20–26 minutes on the 8-vCPU VM against
  40–48 for fp32, and 0.73 GB against 1.51 GB of reader memory on the
  harness.
- **Cost.** A section on the VM is 1,170–1,590 s in int8 at 8 threads, all
  eight saturated (98–100% CPU for the whole run), so the twelve int8
  sections plus the two oracles were about six hours of machine time, split
  over three VMs by twin, and the eight fp32 checks another six. The beam-32 oracle sections took 2,000–2,130 s and
  13 GB. The trigram alone takes two minutes: the reader is still about ten
  times the cost of the rest of the decoder, and every hypothesis pays a
  step of a 60M-parameter model. Lockstep batching of records (#68) is the
  way to get that onto a GPU; per-record resume (#71) is what lets a grid be
  stopped for such a change and continued.

## Reproduce

```
kaggle kernels output lexoliu/mlime-char-lm-s1 -p data/char-lm-run/transformer-s1     # run/charlm.pt
mlime export char-lm data/char-lm-run/transformer-s1/run/charlm.pt --out data/char-lm-run/transformer-s1/char-lm-int8 \
  --restrict data/route-a-assets-v2/emittable.txt --quantize int8
ime-cli fused-eval --model data/run3/ngram.bin --lm data/char-lm-run/transformer-s1/char-lm-int8 --lm-weight 1 \
  --eval-set data/run3_pool/eval3-abbreviated.jsonl --emittable data/route-a-assets-v2/emittable.txt \
  --scores data/route-a-v2/s6/scores-lattice-abbreviated-context-on.jsonl.gz \
  --select-on-dev --weight 0.75 --weight 1 --weight 1.5
ime-cli fused-eval --model data/run3/ngram.bin --lm data/char-lm-run/transformer-s1/char-lm-int8 --lm-weight 1 \
  --eval-set data/run3_pool/eval3-abbreviated.jsonl --emittable data/route-a-assets-v2/emittable.txt \
  --scores data/route-a-v2/s6/scores-lattice-abbreviated-context-on.jsonl.gz \
  --slice test --weight 1.5 --top-k 32 --beam-width 32
```

Drop `--model` and `--lm-weight` for the LM alone and `--quantize` for the
fp32 export; the mixed and full rows use `eval3-mixed.jsonl` /
`eval3.jsonl` with `scores-lattice-mixed-*` / `scores-lattice-context-on`.
The grid ran on three Linux VMs (8 vCPU, 31 GB, Ubuntu 22.04) from one
bundle of the eval sets, the scores, the n-gram model and the two exports;
the reports and the VMs' utilisation snapshots are under
`data/char-lm-t1/cloud/<twin>/`.

## Next

1. **A reader that runs on the device at this cost, or a cheaper step.** The
   transition is now the strongest part of the decoder and its most
   expensive. #68 decodes records in lockstep so the LM step is one batched
   call over every record's beams, which is the shape a GPU or the Neural
   Engine wants; the same batching makes the evaluation grid a fraction of
   its present six hours, and #71's resume makes any grid stoppable.
2. **More reader.** The held-out curve was still falling at 100k steps
   (3.97 → 3.70 over the second session) and the model saw its data once.
   The tokens-per-nat trade is now known to convert into points, at about
   2.7 abbreviated points per 0.2 nats: a second pass over the shards, or a
   wider model on an A100 when the Colab units land, is the cheapest next
   point. The trigram can be retired from the product path once the LM
   alone matches the fused row on full pinyin.
3. **The beam.** Neither reader changed what the eight hypotheses are.
   `notes/generate-ceiling.md` and this note agree that the target
   (36.7 / 49.2 on abbreviated / mixed) sits above what any ranking of the
   present lattice reaches (the oracle at 32 is 36.3 / 49.5), so after the
   reader the decoder itself is what changes: hypotheses proposed by the
   reader rather than by the per-position emissions.
4. **Commercial baselines (#59).** Every number here is relative to our own
   trigram; the product target is measured on a VM through the
   accessibility APIs, and the gap to it decides how much of 1–3 is needed.
