# Route A v1 — first full training and the kill-gate verdict (2026-09-01)

Kernel `lexoliu/mlime-route-a-v1`, 2×T4 DDP, launched 2026-08-28, evaluated
2026-09-01 on `lexos-mac-mini`.

## Verdict

**The kill gate is passed.** With context on, the fused decoder beats the
run3 trigram baseline by 16.1 points of sentence top-1 on the held-out test slice
(target was +8), and the trigram trained on the neural model's own data by 18.2. Context is worth +5.6 points inside the fused system and +8.6
points for the neural emissions alone. The neural route as designed lives; the
next step is scaling, not rethinking.

## Training

| quantity | value |
|---|---:|
| data | run3-v1 subset, 10.03M labelled segments (Kaggle `mlime-run3-v1-samples` + `-labels`) |
| epochs | 1 (each rank: 4.92M seen, 4.75M kept) |
| steps | 28,600, 166.5 examples/step, 1.17 steps/s |
| wall | 6h48m train + 11 min scoring |
| loss | 4.088 → 1.329 (label smoothing 0.05) |
| gates (top 4 layers) | 0.121, 0.105, 0.192, 0.232 |
| held-out char acc (mixed typing aug.) | context on **80.14%**, off 74.92% (39,958 chars) |

Build counts per rank: kept 4,757,058; target_not_admitted 151,186 (3.1%);
unemittable_character 17,894; unknown_span 817; unlabelled 15.

The held-out accuracy is on the training distribution, which is 45%
abbreviated/mixed typing, so it is not comparable to the eval3 numbers below,
which are full pinyin.

## eval3 ablation

`data/run3_pool/eval3.jsonl`, 5,525 records, 81% with context. The fusion
weight was tuned on the dev slice (498 records, hash-selected, `--dev-share
0.0905`) and everything below is reported on the disjoint test slice (5,027
records). Decoder settings: beam 16, 8 readings, top-k 8, unscored −30.

Two trigrams. The *subset* trigram is trained on the same 10.03M-line v1 subset
the neural model saw (21.5M trigram types, 272 MiB, 16 s; no eval3 sentence is in
that corpus, checked by exact match). The *run3* trigram is the laptop's 41M-line
model from `notes/baseline-run1.md`'s lineage (53.9M trigram types, 643 MiB), the
55.29% baseline the kill gate was written against.

| configuration | top-1 | top-8 | char | MRR@8 |
|---|---:|---:|---:|---:|
| subset trigram only | 51.84% | 59.32% | 87.20% | 0.548 |
| run3 trigram only | 55.10% | 62.58% | 88.61% | 0.581 |
| neural only, context off | 44.84% | 48.40% | 85.33% | 0.462 |
| neural only, context on | 53.45% | 57.15% | 89.04% | 0.550 |
| fused with subset trigram, context off, w=1 | 64.47% | 71.12% | 92.18% | 0.672 |
| fused with subset trigram, context on, w=1 | 70.02% | 75.47% | 93.96% | 0.722 |
| fused with run3 trigram, context off, w=1 | 65.55% | 71.93% | 92.59% | 0.681 |
| **fused with run3 trigram, context on, w=1** | **71.18%** | **76.69%** | **94.15%** | **0.735** |

Against the run3 trigram the fused, context-on decoder is **+16.1** top-1; against
the matched subset trigram it is +18.2. Context is worth +5.6 inside the fused
system either way.

Dev-slice weight sweep (top-1, 498 records):

| weight | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| subset trigram, context on | 65.5 | 70.5 | 72.1 | 72.9 | 72.5 | 73.1 | 69.5 |
| subset trigram, context off | 62.0 | 65.9 | 66.7 | 67.3 | 65.5 | 64.7 | 60.6 |
| run3 trigram, context on | — | 71.9 | 73.9 | 74.3 | 73.1 | 72.7 | — |

The plateau is 0.75–2.0 with context and peaks at 1.0 without and with the
stronger trigram; 1.0 is used throughout.

## What the numbers say

- **The n-gram is still load-bearing.** Neural-only top-8 is barely above
  top-1 (57.2 vs 53.5): per-position independent distributions give a beam
  almost no useful alternatives. The trigram supplies the sequence coherence,
  and fusion is where the +16.6 over neural-only comes from. That is the
  design, but it means the transition model ships with the product.
- **Context is real.** Gates trained to 0.10–0.23 from zero, and every
  context-on row beats its context-off twin: +8.6 standalone, +5.6 fused.
- **Character accuracy of the fused system is 94%**, so most remaining
  sentence errors are one wrong character in a long sentence.
- **Training had not converged.** Loss was still falling at the end of the
  cosine schedule, one epoch over a quarter of the available data.

## Reproduce

```
kaggle kernels output lexoliu/mlime-route-a-v1 -p data/route-a-v1
kaggle datasets download lexoliu/mlime-route-a-assets -p data/route-a-assets --unzip
kaggle datasets download lexoliu/mlime-run3-v1-samples -p data/run3-v1/samples --unzip
cd python && uv run mlime export ngram-corpus --data-dir ../data/run3-v1 --out ../data/run3-v1/ngram-corpus.txt
ime-cli train-ngram --corpus data/run3-v1/ngram-corpus.txt --out data/run3-v1/ngram.bin
ime-cli fused-eval --model data/run3-v1/ngram.bin --eval-set data/run3_pool/eval3.jsonl \
  --emittable data/route-a-assets/emittable.txt --scores data/route-a-v1/scores-context-on.jsonl.gz \
  --slice test --weight 1
```

`--no-transition` drops the trigram for the neural-only rows; omitting
`--scores` gives the trigram-only row.

## Next (v2)

1. Train on the full 41M run3 labels for 2–3 epochs; the loss curve says the
   model is data-limited, not capacity-limited.
2. Fix the 3.1% `target_not_admitted` build loss: those are label/lexicon
   disagreements, the Sogou lexicon arbitration deferred from v1.
3. Add an abbreviated and mixed-typing eval set: eval3 is full pinyin only,
   and the training held-out numbers say abbreviation is where the model is
   weakest.
4. Retire the trigram as the only transition: an autoregressive rescorer or
   an iterative refinement pass over the NAR output would recover the top-k
   diversity the beam currently lacks.
