# Sequence-level ranking loss (2026-10-15, #116)

## Verdict

The ranking term pays at a small weight. The best end-to-end model so far is e2e v2 with a ranking stage at weight 0.05, chosen on dev (#132, below). It beats the fused decoder on abbreviated input and trails it by 0.72 points on full pinyin and 0.12 on mixed input. Test slices, context on, top-1:

| style | v2 ranked at 0.05 (#132) | v2 (#155) | ranked at 0.1 (#130) | fine-tuned (#127) | ranked at 1.0 (#116) | v1 | fused |
|---|---|---|---|---|---|---|---|
| full | **77.44%** | 75.77% | 77.06% | 75.37% | 74.40% | 73.52% | 78.16% |
| abbreviated | 28.89% | 27.31% | **29.01%** | 26.63% | 22.14% | 25.05% | 28.26% |
| mixed | **38.42%** | 36.84% | 37.73% | 35.69% | 32.04% | 34.70% | 38.54% |

The sections below record each run in order. The first verdict, written after #116 alone, follows.

### After #116

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

So the anchoring explains part of #116's regression, not all of it. Both stages fine-tune v1 on 2.2M examples (11% of v1's 20.3M-example epoch) with a fresh optimiser and schedule, beside the ranking term. These runs cannot separate what the ranking term costs from what that fine-tuning costs.

### Cost

| job | where | time |
|---|---|---|
| s0 | 2×T4 | 12.0 h session: 11.31 h of training, 19,450 steps at 0.478 steps/s |
| s1 | 2×T4 | 2.27 h session: a 7 min resume skip, 92 min of training, 34 min for the six evaluations |
| the first s0 attempt | 2×T4 | ~10 min: DDP rejected `forward_sync_buffers`, which Kaggle's torch does not take (#123) |

The s1 push at 20:40 EDT, right after the weekly quota reset, waited 83 minutes in Kaggle's queue before its container started, with `kernels status` already reporting RUNNING.

## The control without the ranking term (#124)

The control fine-tunes v1's final checkpoint exactly as #121 did, minus the ranking term. It uses the same 42 shards, dealt by text, the same 2,190,297 examples, the same learning rates (2e-5 for both groups, cosine), and the same 12,288 tokens per rank × 2 ranks × accumulate 2. The decoder's per-position term is normalised over the emittable set (`train e2e --whole-alphabet`), and route A's auxiliary term is unchanged. It ran as `lexoliu/mlime-e2e-control-s0`.

One setting does not carry over. The token budget counts the tokens a step trains on, and a ranking step types each example up to 8 ways, so #121 fit about 100 examples into a step and the control about 512. Over the same epoch the control took 4,277 optimiser steps against #121's 21,734, at the same peak rate. It moved v1's weights through 5× fewer updates.

Held-out per-position accuracy is 90.80% with context and 88.39% without, against v1's 90.92% / 88.12% and #121's 90.54% / 87.78%.

### Results

Test slice, context on:

| style | metric | control | #121 | v1 |
|---|---|---|---|---|
| full | top-1 | 69.15% | **74.58%** | 73.52% |
| full | top-8 | 84.01% | **90.21%** | 89.72% |
| full | character | 90.37% | **94.72%** | 93.69% |
| abbreviated | top-1 | 21.41% | 22.99% | **25.05%** |
| abbreviated | top-8 | 35.25% | 35.90% | **39.33%** |
| abbreviated | character | 60.42% | 61.46% | **63.25%** |
| mixed | top-1 | 31.50% | 33.39% | **34.70%** |
| mixed | top-8 | 45.73% | 48.30% | **50.47%** |
| mixed | character | 68.72% | 71.82% | **72.39%** |

Top-1 on the other slices:

| style | test, context off: control / #121 / v1 | dev, context on: control / #121 / v1 |
|---|---|---|
| full | 60.87% / 66.24% / 65.86% | 69.28% / 73.29% / 73.90% |
| abbreviated | 12.79% / 11.80% / 14.18% | 21.47% / 23.58% / 26.32% |
| mixed | 22.95% / 22.95% / 24.82% | 34.09% / 33.88% / 34.50% |

The top-1 answers over every record, context on, measured as in the tables above:

| style | run | right / same length / other length | unseen in the sample | median frequency rank |
|---|---|---|---|---|
| full | v1 | 4,064 / 1,330 / 131 | 2.7% | 433 |
| full | #121 | 4,114 / 1,396 / 15 | 5.9% | 482 |
| full | control | 3,821 / 1,251 / 453 | 1.5% | 356 |
| abbreviated | v1 | 1,390 / 3,308 / 827 | 6.4% | 267 |
| abbreviated | #121 | 1,273 / 3,736 / 516 | 7.9% | 369 |
| abbreviated | control | 1,183 / 3,191 / 1,151 | 4.1% | 201 |
| mixed | v1 | 1,916 / 2,992 / 617 | 6.8% | 281 |
| mixed | #121 | 1,847 / 3,323 / 355 | 9.0% | 384 |
| mixed | control | 1,753 / 2,797 / 975 | 4.8% | 209 |

### What it says

The fine-tuning is the cost, and the ranking term is a gain on top of it. Against the control, the ranking term adds 5.43, 1.58 and 1.89 points of test top-1 on full, abbreviated and mixed input. The control itself loses 4.37, 3.64 and 3.20 points against v1.

The control's loss has a different shape from the ranking stages'. Its per-position accuracy is v1's, and its wrong characters are more common than v1's (1.5–4.8% unseen). What breaks is segmentation: answers of the wrong length rise from 131 to 453 on full pinyin, from 827 to 1,151 on abbreviated input and from 617 to 975 on mixed. The path score sums one log-probability per character, so readings of different lengths compete through the scale of those log-probabilities. The control changes that scale and nothing retrains the comparison. The ranking term trains exactly that comparison, which is why #121 has the fewest wrong-length answers of all four runs.

Two changes from v1 are folded together in the control, and the run cannot separate them:

- **The whole-alphabet term.** v1 trained the decoder over each span's homophones only, so the logits outside them were never trained. Normalised over the emittable set, the control's decoder loss starts at 3.48 nats (#121's at 3.39) and ends at 1.18, while #121's, trained beside the ranking term, ends at 2.22. The control spends its updates pulling the untrained mass back onto the candidates, which is a direct change to the per-step scale.
- **The short fine-tune itself.** A fresh optimiser and schedule over 11% of v1's epoch, drawn from 42 of its 813 shards.

### Cost

| job | where | time |
|---|---|---|
| s0 | 2×T4 | ~2.5 h session: 1.81 h of training at 336 examples/s for the world, then the six evaluations (8.5–11.4 min each, three per GPU) |
| the first s0 attempt | 2×T4 | ~3 min: `train e2e` lacked the `--whole-alphabet` option the kernel passed; every rank exited on the usage error |

## The candidates-only control (#126)

The same control once more, with the per-position term over each span's homophone candidates (`--candidates-only`), which is v1's own loss. Everything else is #124's: the same source, shards, examples, learning rates, token budget and 4,277 optimiser steps. It ran as `lexoliu/mlime-e2e-control2-s0`.

Held-out per-position accuracy is 91.62% with context and 89.19% without, against v1's 90.92% / 88.12%. The decoder loss over the candidates starts at 0.94 and ends at 0.83.

### Results

Test slice, context on:

| style | metric | #126 | #124 | #121 | v1 | fused |
|---|---|---|---|---|---|---|
| full | top-1 | **75.31%** | 69.15% | 74.58% | 73.52% | 78.16% |
| full | top-8 | **90.85%** | 84.01% | 90.21% | 89.72% | 82.53% |
| full | character | 94.19% | 90.37% | **94.72%** | 93.69% | |
| abbreviated | top-1 | **26.20%** | 21.41% | 22.99% | 25.05% | 28.26% |
| abbreviated | top-8 | **41.13%** | 35.25% | 35.90% | 39.33% | 34.10% |
| abbreviated | character | **64.61%** | 60.42% | 61.46% | 63.25% | |
| mixed | top-1 | **35.73%** | 31.50% | 33.39% | 34.70% | 38.54% |
| mixed | top-8 | **51.93%** | 45.73% | 48.30% | 50.47% | 47.53% |
| mixed | character | **73.51%** | 68.72% | 71.82% | 72.39% | |

Top-1 on the other slices, #126 against v1:

| style | test, context off | dev, context on | dev, context off |
|---|---|---|---|
| full | 68.43% / 65.86% | 75.70% / 73.90% | 70.68% / 66.67% |
| abbreviated | 15.58% / 14.18% | 26.95% / 26.32% | 18.32% / 16.42% |
| mixed | 26.48% / 24.82% | 36.57% / 34.50% | 26.45% / 25.83% |

The top-1 answers over every record, context on:

| style | run | right / same length / other length | unseen in the sample | median frequency rank |
|---|---|---|---|---|
| full | v1 | 4,064 / 1,330 / 131 | 2.7% | 433 |
| full | #126 | 4,163 / 1,229 / 133 | 3.2% | 410 |
| abbreviated | v1 | 1,390 / 3,308 / 827 | 6.4% | 267 |
| abbreviated | #126 | 1,451 / 3,263 / 811 | 6.5% | 276 |
| mixed | v1 | 1,916 / 2,992 / 617 | 6.8% | 281 |
| mixed | #126 | 1,978 / 2,929 / 618 | 7.1% | 284 |

### What it says

The whole-alphabet term was the whole of #124's loss. Without it, the same short fine-tune improves v1 on every style and slice: +1.79, +1.15 and +1.03 points of test top-1 with context on, and more with context off. The answers keep v1's shape. The wrong-length count is unchanged and so is the rarity of the wrong characters, so the gain is in choosing characters, not in segmenting.

With the four runs, each term's cost can be read against the right baseline:

- **The ranking term**, with the per-position term over the candidates, costs 0.91, 4.06 and 3.69 points of test top-1 (#116 against #126). It fixes segmentation, but not by enough to pay for the rare-character drift.
- **The whole-alphabet term** costs 6.16, 4.79 and 4.23 points without the ranking term (#124 against #126). With the ranking term beside it, it costs much less, because the ranking term trains the comparison between readings that the whole-alphabet term disturbs.
- **The short fine-tune** gains 1.03–1.79 points by itself.

So the ranking loss, in the form #116 built, does not belong in the product model, and the fine-tune does.

### Cost

| job | where | time |
|---|---|---|
| s0 | 2×T4 | 2.5 h session: 1.81 h of training at 337 examples/s for the world, then the six evaluations (8.6–11.6 min each, three per GPU) |

## A second fine-tune stage (#127)

The second stage starts from #126's final checkpoint and repeats its fine-tune, with v1's own loss, fresh optimiser and schedule and the same settings, on shards #126 never saw: the 80-pick minus the 40-pick, 51 shards and 2,544,331 examples in 5,025 optimiser steps. The kernel now names the finished run a stage starts from (`INIT_FROM`), so the parent mounts as the kernel output `lexoliu/mlime-e2e-control2-s0`. It ran as `lexoliu/mlime-e2e-stage2-s0`.

Held-out per-position accuracy is 91.81% with context and 89.52% without, against #126's 91.62% / 89.19%.

### Results

Test slice, top-1:

| style | context on: stage 2 / #126 / v1 | context off: stage 2 / #126 / v1 |
|---|---|---|
| full | 75.37% / 75.31% / 73.52% | 68.97% / 68.43% / 65.86% |
| abbreviated | 26.63% / 26.20% / 25.05% | 16.36% / 15.58% / 14.18% |
| mixed | 35.69% / 35.73% / 34.70% | 27.22% / 26.48% / 24.82% |

Top-8 with context on is 90.81% / 41.80% / 52.43%, against #126's 90.85% / 41.13% / 51.93%. On dev with context on, top-1 is 77.71% / 26.53% / 36.98%, against #126's 75.70% / 26.95% / 36.57%.

The top-1 changes on the test slice with context on, record by record. "Gained" counts records the later run answers right and the earlier one wrong; z is McNemar's statistic over those two counts:

| style | v1 → #126: gained / lost, z | #126 → stage 2: gained / lost, z |
|---|---|---|
| full | 176 / 86, 5.56 | 79 / 76, 0.24 |
| abbreviated | 141 / 83, 3.88 | 94 / 72, 1.71 |
| mixed | 149 / 97, 3.32 | 82 / 84, −0.16 |

### What it says

The second stage is flat. It reshuffles about 160 answers per style both ways and nets nothing measurable with context on. With context off it gains 0.5–0.8 points, which is consistent, but small beside the first stage's gain. The first stage's gain over v1 is well outside noise on all three styles.

So the gain was not "more training". Some 2.5M more examples of fresh data at the same settings add nothing. What the first stage changed is what v1's end left behind: v1 read its shards in order, and its last, near-zero-rate steps saw only the corpus's hardest tail (`notes/e2e-v1.md`). A short pass spread across every source, at a small rate, rebalances that once, and doing it again does not pay.

The answers keep v1's shape across both stages. Over every record, context on, the wrong-length counts are 133 / 818 / 616 (v1: 131 / 827 / 617), and the rarity of the wrong characters is unchanged. The fine-tunes choose characters better. They do nothing for the gap between the beam's top-8 and its top-1: 90.81% against 75.37% on full pinyin, 15.4 points, where the fused decoder's top-1 is 78.16%.

### Cost

| job | where | time |
|---|---|---|
| s0 | 2×T4 | 2.6 h session: 1.92 h of training at 368 examples/s for the world, then the six evaluations (8.0–10.3 min each, three per GPU) |

## The ranking weight, chosen on dev (#130)

Two ranking stages start from #127's final checkpoint, `INIT_FROM = "mlime-e2e-stage2-s0"`, and differ only in the ranking weight: 0.1 and 0.3. The rest is #116's ranking (8 segmentations, beam 4, 8 negatives) beside the candidates-only per-position term, with learning rates (2e-5, 2e-5), 12,288 tokens per rank × 2 ranks × accumulate 2. Both train one epoch over the 20-pick: 22 shards and 1,192,633 examples in 11,403 optimiser steps. They ran at once as `lexoliu/mlime-e2e-rankw01-s0` and `-rankw03-s0`.

The ranks counted 22,805 and 23,884 micro-batches. Dealing whole shards cannot balance 22 of them, so 1,079 of rank 1's micro-batches (4.5%) were never trained on (#133).

Held-out per-position accuracy is 91.83% / 89.36% (with / without context) at weight 0.1, and 91.49% / 88.99% at 0.3, against #127's 91.81% / 89.52%.

### Choosing on dev

Dev slice, context on, top-1. Weight 0 is #127 itself:

| style | 0 | 0.1 | 0.3 |
|---|---|---|---|
| full | 77.71% | **78.71%** | 77.51% |
| abbreviated | 26.53% | **26.95%** | 25.68% |
| mixed | 36.98% | **38.64%** | 36.36% |

Weight 0.1 is best on every style. With context off it is also best on full and mixed input, and 0.64 points below #127 on abbreviated input.

### Results at weight 0.1

Test slice, context on:

| style | metric | weight 0.1 | #127 | fused |
|---|---|---|---|---|
| full | top-1 | 77.06% (+1.69) | 75.37% | **78.16%** |
| full | top-8 | **91.63%** (+0.82) | 90.81% | 82.53% |
| full | character | 95.44% (+1.34) | 94.10% | |
| abbreviated | top-1 | **29.01%** (+2.38) | 26.63% | 28.26% |
| abbreviated | top-8 | **43.29%** (+1.49) | 41.80% | 34.10% |
| abbreviated | character | 67.63% (+2.36) | 65.27% | |
| mixed | top-1 | 37.73% (+2.04) | 35.69% | **38.54%** |
| mixed | top-8 | **53.20%** (+0.77) | 52.43% | 47.53% |
| mixed | character | 75.19% (+1.51) | 73.68% | |

Test slice, context off, top-1: 70.70% / 17.25% / 27.97% on full / abbreviated / mixed, against #127's 68.97% / 16.36% / 27.22%. Weight 0.3 on test, context on, top-1: 76.45% / 27.11% / 36.44%.

The top-1 changes against #127 on the test slice with context on, record by record, and the shape of the answers over every record:

| style | #127 → 0.1: gained / lost, z | #127 → 0.3: gained / lost, z |
|---|---|---|
| full | 171 / 86, 5.30 | 193 / 139, 2.96 |
| abbreviated | 218 / 98, 6.75 | 224 / 200, 1.17 |
| mixed | 211 / 108, 5.77 | 239 / 201, 1.81 |

| style | run | right / same length / other length | unseen in the sample | median frequency rank |
|---|---|---|---|---|
| full | #127 | 4,176 / 1,216 / 133 | 3.1% | 404 |
| full | 0.1 | 4,266 / 1,238 / 21 | 4.2% | 459 |
| full | 0.3 | 4,229 / 1,279 / 17 | 7.2% | 525 |
| abbreviated | #127 | 1,471 / 3,236 / 818 | 6.6% | 270 |
| abbreviated | 0.1 | 1,593 / 3,468 / 464 | 7.3% | 316 |
| abbreviated | 0.3 | 1,491 / 3,584 / 450 | 8.1% | 367 |
| mixed | #127 | 1,978 / 2,931 / 616 | 7.2% | 279 |
| mixed | 0.1 | 2,089 / 3,095 / 341 | 7.5% | 325 |
| mixed | 0.3 | 2,013 / 3,187 / 325 | 8.9% | 378 |

### What it says

The weight trades the two effects #116 found. Both weights remove most wrong-length answers: on full pinyin from 133 to 21 and 17, on abbreviated input from 818 to 464 and 450. The drift toward rare characters grows with the weight: 4.2% of substituted characters unseen at 0.1 on full pinyin, 7.2% at 0.3. At 0.1 the segmentation gain is nearly all there and the drift is small, so most of the records that stop being wrong-length become right. At 0.3 more of them become right-length wrong answers instead.

Weight 0.1 is the first end-to-end checkpoint to beat the fused decoder on any style: abbreviated input, by 0.75 points of top-1. Its top-8 is far above the fused decoder's on all three. The remaining top-1 gaps are 1.10 points on full pinyin and 0.81 on mixed input.

The dev curve over 0, 0.1 and 0.3 peaks at 0.1, which does not rule out a lower optimum (#132).

### Cost

| job | where | time |
|---|---|---|
| weight 0.1, s0 | 2×T4 | 7.6 h session: 7.03 h of training at 47 examples/s for the world, then the six evaluations (8.3–11.0 min each, three per GPU) |
| weight 0.3, s0 | 2×T4 | 7.7 h session: 7.09 h of training, then the six evaluations (8.4–11.5 min each) |

The estimate was 6.2 h of training each: it assumed 1.1M examples at #116's 49 examples/s, against 1.19M at 47.

The first harvest of weight 0.1 broke off with the Mac mini's disk at 15 GB free. The kernel output carried the run's numbered checkpoint beside the final one, 3.5 GB that nothing reads. A finished run now publishes only its final checkpoint (#131).

## The ranking stage on e2e v2 (#132)

e2e v2 (#155) is one epoch from v1's initialisation on the interleaved stream (#133). Two ranking stages start from its final checkpoint, `INIT_FROM = "mlime-e2e-v2-g4"`, at weights 0.05 and 0.1. Everything else is #130's recipe: 8 segmentations, beam 4, 8 negatives, the candidates-only per-position term, learning rates (2e-5, 2e-5), 12,288 tokens per rank × 2 ranks × accumulate 2, one epoch over the 20-pick (22 shards, 1,192,633 examples). The interleaved stream balances the ranks to 23,585 and 23,567 micro-batches, so the epoch is 11,784 optimiser steps and none are dropped. Both ran on a Colab G4 as one process of two virtual ranks with the search replayed from CUDA graphs (#137): `mlime-e2e-v2-rank-g4` at 6ae0132 and `mlime-e2e-v2-rank01-g4` at 066850c, which adds #181 and changes nothing that is computed.

Held-out per-position accuracy (with / without context) is 92.30% / 90.02% at 0.05 and 92.18% / 89.83% at 0.1, against v2's 92.14% / 89.92%.

### Choosing on dev

Dev slice, top-1. Weight 0 is v2 itself:

| style | context on: 0 / 0.05 / 0.1 | context off: 0 / 0.05 / 0.1 |
|---|---|---|
| full | 77.51% / **80.72%** / 80.52% | 72.89% / 74.30% / **74.50%** |
| abbreviated | 28.42% / **29.89%** / 28.00% | 18.74% / **20.00%** / 19.79% |
| mixed | 36.36% / 39.26% / **39.88%** | 26.86% / 30.37% / **30.79%** |

Both weights beat v2 on every style. Weight 0.05 is chosen: with context on it answers 734 of the 1,457 dev records right against 727 at 0.1. That margin is small, and with context off it is 612 against 614 the other way. Between 0.05 and 0.1 the dev slice cannot tell the weights apart.

### Results at weight 0.05

Test slice, context on:

| style | metric | weight 0.05 | v2 | fused |
|---|---|---|---|---|
| full | top-1 | 77.44% (+1.67) | 75.77% | **78.16%** |
| full | top-8 | **91.96%** (+0.99) | 90.97% | 82.53% |
| full | character | 95.61% (+1.23) | 94.38% | |
| abbreviated | top-1 | **28.89%** (+1.58) | 27.31% | 28.26% |
| abbreviated | top-8 | **44.20%** (+0.64) | 43.56% | 34.10% |
| abbreviated | character | 68.14% (+1.83) | 66.31% | |
| mixed | top-1 | 38.42% (+1.58) | 36.84% | **38.54%** |
| mixed | top-8 | **54.65%** (+1.31) | 53.34% | 47.53% |
| mixed | character | 76.25% (+2.02) | 74.23% | |

Test slice, context off, top-1: 70.82% / 18.16% / 28.74% on full / abbreviated / mixed, against v2's 69.41% / 17.05% / 27.97% and the fused decoder's 71.85% / 17.27% / 29.14% (#168). Weight 0.1 on test is level with 0.05: 77.34% / 29.19% / 38.44% with context on, 70.94% / 17.98% / 28.45% with context off.

The top-1 answers on the test slice, context on, by kind, and the rarity of the substituted characters, measured as in the tables above (`mlime eval drift`):

| style | run | right / same length / other length | unseen in the sample | median frequency rank |
|---|---|---|---|---|
| full | v2 | 3,809 / 1,096 / 122 | 3.1% | 420 |
| full | 0.05 | 3,893 / 1,116 / 18 | 4.5% | 452.5 |
| full | 0.1 | 3,888 / 1,124 / 15 | 4.4% | 456 |
| abbreviated | v2 | 1,379 / 2,955 / 716 | 6.3% | 258 |
| abbreviated | 0.05 | 1,459 / 3,131 / 460 | 7.0% | 288 |
| abbreviated | 0.1 | 1,474 / 3,151 / 425 | 7.2% | 310 |
| mixed | v2 | 1,857 / 2,616 / 568 | 7.0% | 268 |
| mixed | 0.05 | 1,937 / 2,774 / 330 | 7.6% | 309 |
| mixed | 0.1 | 1,938 / 2,790 / 313 | 7.5% | 326 |

The earlier tables count every record. Over every record these shares move by at most 0.2 points.

### What it says

The ranking stage pays on v2 as it did on #127. At 0.05 it adds 1.58–1.67 points of test top-1 with context on and 0.77–1.41 with context off. No earlier end-to-end checkpoint comes as close to the fused decoder on full and mixed input. With context on it beats the fused decoder on abbreviated input by 0.63 points and trails it by 0.72 on full pinyin and 0.12 on mixed. With context off it beats it on abbreviated input by 0.89 and trails by 1.03 and 0.40. #149's first two acceptance criteria are not met yet.

The drift toward rare characters is back, at about the size #130 found. The share of substituted characters unseen in the sample rises from 3.1% to 4.5% on full pinyin, from 6.3% to 7.0% on abbreviated input and from 7.0% to 7.6% on mixed. #130 measured rises of 1.1, 0.7 and 0.3 points at 0.1 on #127. It does not grow between 0.05 and 0.1: the unseen shares are within 0.2 points of each other. Only the median rank creeps up on abbreviated and mixed input (288 → 310 and 309 → 326). In this range the drift is a fixed cost of having the ranking term, not a function of its weight.

Where the segmentation gain goes differs by style. On full pinyin, 84 of the 104 answers that stop being wrong-length become right. On abbreviated and mixed input only 80 of 256 and 80 of 238 do, and the rest become right-length wrong answers. That is where the fused decoder's lead on mixed input is left, and where the drift costs most.

### Cost

| leg | commit | rate | GPU | training time |
|---|---|---|---|---|
| 0.05 | 6ae0132 | 0.542 s/step over steps 300–2130 | 85.2% busy, 304 W, no throttling | 6,789 s |
| 0.1 | 066850c | 0.478 s/step over steps 1530–1960 | 97.1% busy, 346 W, no throttling | 5,925 s |

#181 removed the ranking step's device syncs. The steady rate is 12% faster and the card went from 85% to 97% busy. Over the whole run the same 11,780 steps took 13% less training time, two minutes of it the first step's compile (203 s against 84 s). Each of the six evaluations took 86–97 s on the same card.

### Reproduce

1. Stamp the kernel: `uv run --project python mlime e2e stamp mlime-e2e-v2-rank-g4 --out <dir> --init-from mlime-e2e-v2-g4 --ranking --rank-weight 0.05 --base-lr 2e-5 --new-lr 2e-5 --token-budget 12288 --accumulate 2 --search-graphs`, with one `--train-shard` for each of the 20-pick's 22 shards (the `train_shards` of `run-config.json` below).
2. Run it: `colab/e2e.sh <dir> start --session-hours 5 --gpu G4`, then `colab/e2e.sh <dir> wait --deadline 18000` until it finishes. The final checkpoint, `run-config.json` and the six evaluations with their dumps are in the `lexoliu/mlime-resume` bucket under `final/mlime-e2e-v2-rank-g4/` (`final/mlime-e2e-v2-rank01-g4/` for 0.1).
3. The drift table, per twin and run: `uv run --project python mlime eval drift --dump <final>/e2e-lattice-context-on.jsonl --eval-set data/run3_pool/eval3.jsonl --slice test`, with `-abbreviated` and `-mixed` in both names for the other twins. It reproduces #130's table exactly with `--slice all`.

## Next

- **More ranking data at 0.05 (#149).** The dev gain holds on v2 and the drift does not grow with the weight, so the next stage trains the ranking term at 0.05 on more than the 20-pick. It continues only while the dev gain holds and the drift, measured as here against the 0.05 rows, does not grow.
- **Where right-length wrong answers come from.** On abbreviated and mixed input two thirds of the segmentation gain becomes right-length wrong answers. The hypothesis is #116's: the term's negatives come from a beam of 4 over 8 readings, so on spans that admit hundreds of characters it never offers most of them as negatives. That is the place to look for the remaining gap to the fused decoder on mixed input.
