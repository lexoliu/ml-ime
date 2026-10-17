# Sequence-level ranking loss (2026-10-15, #116)

## Verdict

The ranking stage improves full pinyin and makes abbreviated and mixed input worse. Test slices, context on, against e2e v1 (`notes/e2e-v1.md`) and the fused decoder (`notes/char-lm-v2.md`):

| style | ranked top-1 | v1 top-1 | fused top-1 | ranked top-8 | v1 top-8 | fused top-8 |
|---|---|---|---|---|---|---|
| full | **74.40%** | 73.52% | 78.16% | **90.31%** | 89.72% | 82.53% |
| abbreviated | 22.14% | **25.05%** | 28.26% | 36.32% | **39.33%** | 34.10% |
| mixed | 32.04% | **34.70%** | 38.54% | 48.40% | **50.47%** | 47.53% |

The full-pinyin top-1 gap to the fused decoder narrows from 4.64 to 3.76 points. On abbreviated and mixed input the model loses 2.9 and 2.7 points of top-1, and more with context off. This checkpoint does not replace v1, and the end-to-end model still does not replace the fused decoder.

The loss did what it was built to do. Answers of the wrong length almost disappear. What it took away elsewhere is explained below, and the next step follows from that.

## The loss

The loss is in `python/src/mlime/train/ranking.py`; its docstring states the design. Briefly:

1. A batch types every example each way the `ime-segment` binary reads its keystrokes, up to 8 readings.
2. A no-grad beam of width 4 runs over every reading. The hypotheses are merged per sentence, keeping the better reading's score, as the evaluation does.
3. The gold path is scored with gradient on the decode's own path score: the sum of each step's log-probability over the 7,322 emittable characters. So are the gold sentence on another reading, if the beam found it there, and the 8 best other sentences.
4. The loss is a softmax over those path scores with the gold ones as positives. It is added at weight 1.0 beside v1's per-position loss over the homophone candidates and route A's auxiliary loss.

Training starts from v1's final checkpoint (`lexoliu/mlime-e2e-final`) with a fresh optimiser. The cosine schedule runs at 2e-5 for both parameter groups. Each step is 12,288 tokens per rank × 2 ranks × accumulate 2.

## The run

One epoch over 42 shards: the 40-pick of `data/e2e-census/rank-shard-picks.json`, spread across every source. The count replayed 2,190,297 kept examples into 19,856 optimiser steps. It ran as two Kaggle segments (`lexoliu/mlime-e2e-rank-s0`, `-s1`) on 2×T4 DDP:

| segment | steps | training time |
|---|---|---|
| s0 | 1 – 18,040 | 11.30 h |
| s1 | 18,049 – 19,856, then the evaluation | 1.37 h |

Held-out per-position character accuracy is unchanged: 91.00% with context and 88.26% without, against v1's 90.92% and 88.12%. The change is entirely in how whole paths rank.

### The epoch stopped early on one rank

The training stream dealt shards to ranks round-robin. Over 813 shards that is even (v1's ranks counted 237,751 and 236,787 micro-batches). Over 42 shards of uneven size it is not: rank 0 owned 39,711 micro-batches and rank 1 owned 47,617. DDP ends every rank's epoch at the shorter count, so 7,905 of rank 1's batches, the tail of its shard list, were never trained on. That is 8.6% of the epoch's batches.

`deal_shards` now gives each shard, largest first, to the reader holding the least text. The census format moved to version 3, so a count taken under the old dealing is not reused. The run measured here predates the fix.

## Results

Test slices: 5,027 full, 5,050 abbreviated and 5,041 mixed records. The dev slice is `--dev-share 0.0905`. Nothing was tuned on it: the run has no weight to select.

### Context on, test

| style | top-1 | top-8 | character | MRR@8 |
|---|---|---|---|---|
| full | 74.40% (+0.88) | 90.31% (+0.59) | 94.44% (+0.75) | 80.71% (+0.84) |
| abbreviated | 22.14% (−2.91) | 36.32% (−3.01) | 62.12% (−1.13) | 27.07% (−2.92) |
| mixed | 32.04% (−2.66) | 48.40% (−2.07) | 71.48% (−0.91) | 37.88% (−2.47) |

Differences are against v1.

### Context off, test

| style | top-1 | top-8 | character |
|---|---|---|---|
| full | 66.28% (+0.42) | 85.96% (+0.66) | 92.11% (+0.84) |
| abbreviated | 9.76% (−4.42) | 18.87% (−6.10) | 47.59% (−4.37) |
| mixed | 20.43% (−4.39) | 35.53% (−4.32) | 62.06% (−3.16) |

### Context on, dev

| style | top-1 | top-8 | character |
|---|---|---|---|
| full | 74.50% (+0.60) | 90.36% (+1.00) | 94.67% (+0.06) |
| abbreviated | 21.89% (−4.43) | 37.26% (−2.11) | 58.69% (−3.27) |
| mixed | 31.61% (−2.89) | 48.14% (−4.96) | 71.41% (−2.21) |

## What the numbers say

Top-1 answers over every record (dev and test), context on, sorted by whether the answer is right, wrong at the gold length, or wrong at another length:

| style | v1 right / same length / other length | ranked right / same length / other length |
|---|---|---|
| full | 4,064 / 1,330 / 131 | 4,111 / 1,385 / 29 |
| abbreviated | 1,390 / 3,308 / 827 | 1,222 / 3,778 / 525 |
| mixed | 1,916 / 2,992 / 617 | 1,768 / 3,383 / 374 |

The loss removed 78% of the wrong-length answers on full pinyin and 37–39% on the other two styles. That is the segmentation ranking it was built for.

On abbreviated and mixed input it also turned right answers into wrong sentences of the right length. Examples (truth → ranked top-1, both right length):

- 晚上六点 → 挽手落地
- 不是周杰伦 → 不食之精灵
- 也会感动很多人 → 也会干掉很多人
- 不了了之 → 埔里来着

The wrong characters it chooses are rarer than v1's. Character rarity here is ranked by frequency over the six local run3 sample shards (59,872 characters, 2,964 distinct). The table covers the characters that differ from the truth in same-length wrong answers, context on:

| style | v1: share unseen in the sample | ranked: share unseen | v1 median frequency rank | ranked median frequency rank |
|---|---|---|---|---|
| full | 2.7% | 11.4% | 433 | 690 |
| abbreviated | 6.4% | 9.1% | 267 | 403 |
| mixed | 6.8% | 10.6% | 281 | 419 |

This fits how the loss is built. The softmax over a handful of path scores is invariant to a shift of all of them, so it trains only differences between the paths the beam found.

- **Negatives.** The beam's competitors are common, fluent words. Their characters are pushed down wherever they occur as a negative.
- **Never-offered characters.** Rare characters are almost never offered as negatives, and the per-position loss only covers each span's homophone candidates. So nothing holds rare characters down in the emittable-set normaliser that the path score uses, and their share of it grows.

Abbreviated and mixed spans admit hundreds of characters, against a handful for a full syllable. That is where the drift shows.

## Cost

| job | where | time |
|---|---|---|
| s0 | 2×T4 | 12.0 h session, 11.30 h of training |
| s1 | 2×T4 | ~2.2 h session: 1.37 h of training and the six evaluations (8–11 min each, three per GPU) |
| the run in all | | 13.35 GPU-session hours, against an estimate of 13.7 |

The throughput was 24.5 examples/s per rank, 49.0 for the world, at 91.4% and 93.0% GPU utilisation. Both cards spent 94–97% of samples on the software power cap. The table of each optimisation is in #116.

## Reproduce

1. Version the package dataset from this branch: `kaggle datasets version -p <dir holding mlime/ and bin/ime-segment> --dir-mode tar`.
2. Stamp `kaggle/e2e/kernel.py`:
   - `SOURCE_DIGEST`, the package's `source_digest`;
   - `TOKEN_BUDGET = 12288`;
   - `ACCUMULATE = 2`;
   - `RANKING = {"segmentations": 8, "beam": 4, "negatives": 8, "weight": 1.0}`;
   - `TRAIN_SHARDS`, the 40-pick;
   - `LEARNING_RATES = (2e-5, 2e-5)`.
3. Add `lexoliu/mlime-e2e-final` to its `dataset_sources`.
4. Push it as `lexoliu/mlime-e2e-rank-s0`, and let `kaggle/chain.sh --slug lexoliu/mlime-e2e-rank-s --kernel-dir <stamped dir> --data-dir data/e2e-rank --finish none` carry it on.

With less than a whole session of quota left in the week, set `MLIME_SESSION_HOURS` in the chain's environment, as for the s1 push here. The final segment evaluates in-kernel. Its output, with `run/checkpoint-final.pt`, is the kernel output of `lexoliu/mlime-e2e-rank-s1`.

## The per-position term over the emittable set (#121)

The ranking term should train the whole distribution the path score normalises over, not only the paths the beam found. So the second run changes one thing: the decoder's per-position term is normalised over the 7,322 emittable characters (`smoothed_cross_entropy`) instead of over each span's homophones. Its gradient then pushes down every character in proportion to its probability, rare ones included. The smoothing mass stays on the candidates.

Everything else is the first run's: v1's final checkpoint, the same 42 shards and settings. The shards are now dealt by their text, so both ranks counted 43,859 and 43,468 micro-batches and the epoch is 21,734 steps over all 2,190,297 examples. It ran as `lexoliu/mlime-e2e-rank2-s0` and `-s1`.

Held-out per-position accuracy is 90.54% with context and 87.78% without. That is about 0.35 points below both v1 (90.92% / 88.12%) and #116 (91.00% / 88.26%).

### Results

Test slice, context on, top-1:

| style | #121 | #116 | v1 | fused |
|---|---|---|---|---|
| full | **74.58%** | 74.40% | 73.52% | 78.16% |
| abbreviated | 22.99% | 22.14% | **25.05%** | 28.26% |
| mixed | 33.39% | 32.04% | **34.70%** | 38.54% |

Test slice, context on, top-8 and character accuracy:

| style | #121 top-8 | v1 top-8 | #121 character | v1 character |
|---|---|---|---|---|
| full | 90.21% | 89.72% | 94.72% | 93.69% |
| abbreviated | 35.90% | 39.33% | 61.46% | 63.25% |
| mixed | 48.30% | 50.47% | 71.82% | 72.39% |

Test slice, context off, top-1:

| style | #121 | #116 | v1 |
|---|---|---|---|
| full | 66.24% | 66.28% | 65.86% |
| abbreviated | 11.80% | 9.76% | 14.18% |
| mixed | 22.95% | 20.43% | 24.82% |

Dev slice, context on, top-1:

| style | #121 | #116 | v1 |
|---|---|---|---|
| full | 73.29% | 74.50% | 73.90% |
| abbreviated | 23.58% | 21.89% | 26.32% |
| mixed | 33.88% | 31.61% | 34.50% |

The top-1 answers over every record, context on, sorted by right / wrong at the gold length / wrong at another length. Rarity is measured on the same-length wrong answers, over the characters that differ from the truth, with the same local frequency sample as above:

| style | run | right / same length / other length | unseen in the sample | median frequency rank |
|---|---|---|---|---|
| full | v1 | 4,064 / 1,330 / 131 | 2.7% | 433 |
| full | #116 | 4,111 / 1,385 / 29 | 11.4% | 690 |
| full | #121 | 4,114 / 1,396 / 15 | 5.9% | 482 |
| abbreviated | v1 | 1,390 / 3,308 / 827 | 6.4% | 267 |
| abbreviated | #116 | 1,222 / 3,778 / 525 | 9.1% | 403 |
| abbreviated | #121 | 1,273 / 3,736 / 516 | 7.9% | 369 |
| mixed | v1 | 1,916 / 2,992 / 617 | 6.8% | 281 |
| mixed | #116 | 1,768 / 3,383 / 374 | 10.6% | 419 |
| mixed | #121 | 1,847 / 3,323 / 355 | 9.0% | 384 |

### What it says

The whole-alphabet term did what it was built for:

- On full pinyin it halved the rare-character drift (11.4% → 5.9% unseen).
- It kept the ranking stage's segmentation gain: wrong-length answers fell from 131 to 15.
- Full-pinyin top-1 is the end-to-end model's best so far, 74.58% on test. The fused decoder is still 3.58 points ahead.

On abbreviated and mixed input it recovers 0.85 and 1.35 points of #116's loss, but both stay below v1: −2.06 and −1.31 on test, and further below with context off. The drift toward rarer characters is smaller there but not gone (7.9% and 9.0% unseen, against v1's 6.4% and 6.8%). Abbreviated input still has 428 more right-length wrong answers than v1.

So the anchoring explains part of #116's regression, not all of it. Both stages fine-tune v1 on 2.2M examples (5% of v1's epoch) with a fresh optimiser and schedule, beside the ranking term. These runs cannot separate what the ranking term costs from what that fine-tuning costs.

### Cost

| job | where | time |
|---|---|---|
| s0 | 2×T4 | 12.0 h session: 11.31 h of training, 19,450 steps at 0.478 steps/s |
| s1 | 2×T4 | 2.27 h session: a 7 min resume skip, 92 min of training, 34 min for the six evaluations |
| the first s0 attempt | 2×T4 | ~10 min: DDP rejected `forward_sync_buffers`, which Kaggle's torch does not take (#123) |

The s1 push at 20:40 EDT, right after the weekly quota reset, waited 83 minutes in Kaggle's queue before its container started, with `kernels status` already reporting RUNNING.

## Next

Separate the two causes with a control. Run the same fine-tuning from v1's final checkpoint on the same 42 shards, with the same optimiser, schedule and whole-alphabet per-position term, but without the ranking term.

- If the control also loses abbreviated and mixed top-1, the fine-tuning is the cost. The ranking term then belongs in v1's full epoch rather than in a short stage after it.
- If the control holds v1's numbers, the ranking term is the cost, and its weight is chosen on the dev slice.
