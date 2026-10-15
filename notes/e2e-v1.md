# The end-to-end model, v1 (2026-10-10, #88)

## Verdict

The end-to-end model wins on top-8 and loses on top-1. On the eval3 test slices with context on, its top-8 beats the best fused decoder in all three typing styles:

| style | e2e top-8 | fused top-8 |
|---|---|---|
| full | 89.72% | 82.53% |
| abbreviated | 39.33% | 34.10% |
| mixed | 50.47% | 47.53% |

Its top-1 is still behind:

| style | e2e top-1 | fused top-1 |
|---|---|---|
| full | 73.52% | 78.16% |
| abbreviated | 25.05% | 28.26% |
| mixed | 34.70% | 38.54% |

The fused decoder is route A plus the run3 trigram plus the int8 transformer reader. Its numbers are from `notes/char-lm-v2.md`.

So the right sentence is more often among the model's eight answers than among the fused decoder's, but the model ranks it first less often. Per #88's gate ("once it wins, it replaces route A plus the fused transitions"), this version does not replace the product path.

The case #88 was opened for is solved. With no context, `tashiwojiejie` decodes to 她是我姐姐 (−0.50 against 他是我姐姐 at −1.60). Its siblings decode right as well: 她是我妹妹, 他是我哥哥, 他是我弟弟. The abbreviated `tswjj` does not: its top answer is 他死亡将近.

## The model

The model is the design in #88 as built.

- **Encoder.** Route A's MacBERT fill tower and its context tower, joined by gated cross-attention. It is initialised from route A v2's final checkpoint.
- **Decoder.** The 12-layer, 512-wide transformer reader. Each block has a zero-initialised, gated cross-attention into the encoder's span vectors, and each step adds the current span's encoder vector to its input.
- **Loss.** Per-position cross-entropy over each span's homophone candidates, with route A's own head kept as an auxiliary loss.

Training ran one epoch over run3 v1 and the rest of run3: 20.3M examples, 59,197 steps, 2×T4 DDP. It was split into Kaggle segments, plus a first Colab leg that ended at step 4,225:

| segment | steps |
|---|---|
| s0 | 4,225 – 25,984 |
| s1 | 25,985 – 47,808 |
| s2 | 47,809 – 58,240 |
| s3 | 58,241 – 59,197 |

Each segment resumed from the previous segment's paused checkpoint.

Character accuracy on the run's own held-out shards (4,092 examples, 40,412 characters) was 90.92% with context and 88.12% without.

### The training loss rises after step 36k, and that is the data

The smoothed loss per 2k steps does not drift. It moves in steps:

| steps | decoder loss | encoder loss |
|---|---|---|
| 16k–22k | 0.81 | 0.98 |
| 24k | 0.67 | 0.80 |
| 30k–36k | 0.60–0.62 | 0.71–0.73 |
| 38k | 0.72 | 0.90 |
| 44k–58k | 0.81–0.84 | 1.00–1.03 |

None of the jumps falls on a resume point (4,225, 25,984, 47,808). The loss stays level from 50k on, where the learning rate is below 1.2e-6 and ends at 1e-12, so the weights barely change there and the curve is following the data. `CorpusStream` reads the train shards in order, and the later shards are harder text.

The training loss alone cannot say whether the late shards hurt the model. The eval3 numbers above are from the final checkpoint.

## Scoring segmentations: normalise over the emittable set

The first evaluation renormalised each decode step over the characters its span admits. That made the scores of two segmentations of the same keystrokes incomparable.

A narrow span such as `o` admits a handful of characters. Splitting `zao` into `za o` therefore costs almost nothing, and the over-segmented reading wins. Examples of top-1 answers this produced:

- 杂哦 for 早;
- 亚欧 for 腰;
- 嘎哦 for 搞;
- 即阿里顿 for 家里蹲.

On the full-pinyin test slice, 610 of the 1,710 top-1 misses were longer than the truth.

The search still admits only the lattice's characters. Since #115, a step's score is its log-probability over all 7,322 emittable characters.

| full pinyin, context on, test | top-1 | top-8 | character |
|---|---|---|---|
| renormalised over the admitted | 68.95% | 82.57% | 89.20% |
| over the emittable set | **73.52%** | **89.72%** | **93.69%** |

## Results

Test slices (5,027 full / 5,050 abbreviated / 5,041 mixed records). The dev slice is `--dev-share 0.0905`, as everywhere else. Nothing was tuned on it: the model has no weight to select.

### Context on

| style | e2e top-1 | e2e top-8 | e2e character | fused top-1 | fused top-8 | fused character |
|---|---|---|---|---|---|---|
| full | 73.52% | **89.72%** | 93.69% | **78.16%** | 82.53% | **95.87%** |
| abbreviated | 25.05% | **39.33%** | 63.25% | **28.26%** | 34.10% | **67.90%** |
| mixed | 34.70% | **50.47%** | 72.39% | **38.54%** | 47.53% | **76.23%** |

### Context off

| style | top-1 | top-8 | character |
|---|---|---|---|
| full | 65.86% | 85.30% | 91.27% |
| abbreviated | 14.18% | 24.97% | 51.96% |
| mixed | 24.82% | 39.85% | 65.22% |

### Dev slice, context on

| style | top-1 | top-8 | character |
|---|---|---|---|
| full | 73.90% | 89.36% | 94.61% |
| abbreviated | 26.32% | 39.37% | 61.96% |
| mixed | 34.50% | 53.10% | 73.62% |

## What the numbers say

The model finds the right sentence: its top-8 is 4 to 7 points above the fused decoder's in every style. It loses at choosing among what it found. On the full-pinyin test slice, 814 records have the truth in the top-8 but not first; choosing right among the model's own eight would give 89.72%.

The reason is how the model was trained. It only ever saw the gold segmentation, teacher-forced, with each position's loss over that position's candidates. Nothing in training compares two segmentations of the same keystrokes, or two sentences that differ in more than one position. Yet at decode time the score of a whole path, summed over positions and compared across segmentations, is exactly what decides top-1.

Emittable-set normalisation removed the worst of it: candidate-set size no longer rewards over-segmentation. But the per-position probabilities were never trained to add up to a sentence score that ranks segmentations.

## Cost

| job | where | time | notes |
|---|---|---|---|
| training | 2×T4 | 29.5 h of training time in s0–s2, plus the Colab leg to step 4,225 and s3 | October 3–10 |
| s3, two attempts | 2×T4 | ~2.7 h | the run had finished on the first; see below |
| evaluation (`mlime eval e2e`, batched since #113) | both T4s in parallel | 11–14 min per lattice and context setting, ~30 min for all six | |
| the same, before batching | M1 CPU | 1.6 s a record | hours |

The last segment trained twice because `e2e()` crashed while writing its summary after a finished run. Kaggle marked the kernel ERROR, and the chain re-pushed the segment as dead (#113).

## Reproduce

The final checkpoint, with the optimiser stripped (1.19 GB), is the Kaggle dataset `lexoliu/mlime-e2e-final` (`run/checkpoint-final.pt`, `run-config.json`).

1. Version the package dataset from `dev`: `kaggle datasets version -p <dir holding python/src/mlime and dataset-metadata.json> --dir-mode tar`.
2. Push `kaggle/e2e/kernel.py` as a new segment with `lexoliu/mlime-e2e-final` among its `dataset_sources` and no `kernel_sources`. A kernel that finds `run/checkpoint-final.pt` mounted evaluates it on every `lattice*.jsonl` of the route A assets, context on and off, one section per GPU, and trains nothing.

Locally, one section:

```
uv run --project python mlime eval e2e \
  --checkpoint data/e2e/final/run/checkpoint-final.pt \
  --lattice data/route-a-assets-v2/lattice.jsonl \
  --eval-set data/route-a-assets-v2/eval3.jsonl \
  --char-table data/route-a-assets-v2/char_pinyin.tsv \
  --slice all --dump e2e-full-on.jsonl
```

The dump is the progress record: a run started again on the same dump continues where it stopped.

## Next

Train the model to rank sentences, not positions (#116). A sequence-level loss over each training example's lattice teaches the summed path score to put the gold sentence first among competing segmentations and readings:

- **Positive:** the gold path.
- **Negatives:** the model's own beam over the example's lattice, or the lattice's other segmentations.
- **Starting point:** this checkpoint.

The target is the top-1 gap: 4.6 / 3.2 / 3.8 points on full / abbreviated / mixed, with the right answer already inside the top-8 more often than the fused decoder's.
