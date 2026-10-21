# The end-to-end model, v2 (2026-10-21, #155)

## Verdict

The interleaved stream was the lever. v2 is v1 trained again from the same initialisation with the same recipe on the same data, in a different order. It beats v1 on every twin, context on and off, by 2.1 to 3.6 points of test top-1, and matches or beats #127, which got there with v1's epoch plus two fine-tune stages. Test slices, top-1:

| style | v2 | v1 | #127 | fused | v2 − v1 | v2 − fused |
|---|---|---|---|---|---|---|
| full, context on | 75.77% | 73.52% | 75.37% | **78.73%** | +2.25 | −2.96 |
| abbreviated, context on | 27.31% | 25.05% | 26.63% | **28.42%** | +2.26 | −1.11 |
| mixed, context on | 36.84% | 34.70% | 35.69% | **38.23%** | +2.14 | −1.39 |
| full, context off | 69.41% | 65.86% | 68.97% | **71.85%** | +3.55 | −2.44 |
| abbreviated, context off | 17.05% | 14.18% | 16.36% | **17.27%** | +2.87 | −0.22 |
| mixed, context off | 27.97% | 24.82% | 27.22% | **29.14%** | +3.15 | −1.17 |

v2 still trails the fused decoder at top-1 and at the characters everywhere. Its top-8 leads the fused decoder by 3.8 to 6.9 points, so the gap left is still in ranking what it finds. v2 is the base for the ranking stage, which ran on it as #132 (`notes/sequence-ranking.md`).

The fused decoder here is route A v2 + the run3 trigram + the int8 transformer reader, with the emission weight chosen on dev: the CI runs at 246d25b (context on) and 2b29cd3 (context off) under `data/ambiguity-slice/fused/`. #149 and the #155 results comment quote an older fused run, 78.16% / 28.26% / 38.54%, at weight 1.0 from 33f7371; #170 explains the difference.

## The model and the run

The model is v1's (`notes/e2e-v1.md`). The encoder is route A's MacBERT fill tower and context tower, initialised from route A v2's final checkpoint. The decoder is the 12-layer, 512-wide transformer reader with gated cross-attention into the encoder's span vectors. The loss is per-position cross-entropy over each span's homophone candidates, with route A's head as an auxiliary loss at 0.3.

The recipe is v1's: a world of 2, 8,192 tokens per micro-batch per rank, accumulate 4, learning rates 3e-5 / 1e-4 on the cosine schedule, one epoch. `mlime e2e stamp` refuses a template whose recipe has drifted from v1's. The stream is the only change. v1's `CorpusStream` read the train shards in order. v2's strides rows across readers over interleaved shards (#133, #135). The census counts 59,577 steps for the epoch over 813 shards, 238,307 and 238,320 micro-batches on the two ranks. v1's epoch was 59,197 steps.

v1 trained on 2×T4 DDP. v2 trains on one card, where the kernel runs the world of two as one process of two virtual ranks.

### The card

Two legs ran the identical kernel at a88f36e from `mlime-e2e-init`, each under its own run name: `mlime-e2e-v2-g4` on a G4 (RTX PRO 6000 Blackwell) and `mlime-e2e-v2-a100` on an A100 40 GB. Both used the same census, profiled steps 1–30 with the first 3 traced, sampled the GPU and the host throughout, and paused at step 300. Their losses at step 300 were 1.8857 and 1.8917. The rate is over steps 40–300 (`measure.json` of each leg):

| | G4 | A100 |
|---|---|---|
| s/step | 0.352 | 0.719 |
| examples/s per rank | 981.4 | 480.4 |
| GPU utilisation, power | 92.6%, 378.0 W | 75.2%, 260.5 W |
| device busy in the trace | 74.8% | 52.8% |
| GPU idle per step, backward / forward | 57.7 ms / 41.8 ms | 246.7 ms / 202.8 ms |
| GPU memory | 10,124.6 MiB | 9,986.0 MiB |
| host | 48 cores, 3.6% busy | 12 cores, 14.6% busy |
| trainer peak resident | 4.45 GB | 4.42 GB |
| first step, with compile | 314.4 s | 779.7 s |
| the epoch at this rate | 5.83 h | 11.90 h |

The G4 is 2.04 times as fast and ran the epoch. The A100's steps were dropped (#149). Both cards launch the same 8,368 forward kernels per step. The A100 spends 450 ms of every step idle between them against the G4's 100 ms, so the A100 VM loses on how fast its host issues launches, not on the card. Memory is far from a limit on either. The two legs and the comparison cost 2.87 CU, as recorded on #155 from Colab's compute-unit balance; no file under `data/` holds it.

### The run

The G4 run resumed its own step-300 trio at 0b83d4a and trained steps 301–59,577 in one Colab session. The only commit between a88f36e and 0b83d4a is #162, which changes what a finished run publishes and nothing that is computed. From `data/e2e-v2/g4/leg2/measure.json` and `run-summary.json`:

| | |
|---|---|
| rate | 0.355 s/step over steps 400–59,570, 963.5 examples/s per rank |
| GPU | 91.5% utilisation, 381.6 W, no throttling, at most 69 °C, 10,835.7 MiB |
| device busy in the trace (steps after the resume) | 76.8%: 56.9 ms of backward and 31.4 ms of forward idle per step |
| recompiles, graph breaks | 0, 0 |
| trainer peak resident | 4.68 GB, on a 48-core host 3.5% busy |
| first step | 49.4 s, from the compile cache in the resume bucket |
| training time | 21,195.5 s (5.89 h) |
| evaluation | six sections, 86.5–97.5 s each on the one G4 (v1: 661.8–849.7 s on a T4) |
| loss | 1.7037 at step 1, 1.0045 at step 59,570 |

The whole session cost 54.16 CU (#155, from Colab's balance).

GPU utilisation is above the 80% bar, but the trace puts the device busy for only 76.8% of the traced steps. The rest is idle time between kernel launches. The levers on #155 address them: fewer, larger micro-batches, CUDA graphs over fixed shapes, compiled autograd. The first two change the recipe, so v2 did not pull them.

Held-out per-position accuracy on the run's own held-out shards (the same 4,092 examples and 40,412 characters as v1) is 92.14% with context and 89.92% without, against v1's 90.92% and 88.12%.

### The loss curve is level

The smoothed loss per 2k steps, binned the same way for both runs from their `metrics.jsonl` (v1's log starts at step 4,225):

| steps | v1 decoder / encoder | v2 decoder / encoder |
|---|---|---|
| 4k | 0.82 / 0.89 | 0.85 / 0.92 |
| 8k | 0.89 / 1.02 | 0.80 / 0.91 |
| 16k–22k | 0.81–0.82 / 0.97–0.98 | 0.75–0.79 / 0.89–0.95 |
| 24k | 0.67 / 0.80 | 0.78 / 0.93 |
| 30k–34k | 0.60–0.61 / 0.71 | 0.74–0.75 / 0.89–0.92 |
| 38k | 0.72 / 0.90 | 0.80 / 0.99 |
| 44k–58k | 0.81–0.84 / 0.99–1.03 | 0.78–0.80 / 0.95–0.99 |

v1's curve moves in steps as its shards change. v2's stays between 0.74 and 0.80 from 8k on, so every bin sees the same mix of sources, the last, near-zero-rate steps included. Late in the epoch v2's loss is lower than v1's, but the two runs see different examples in each bin, so that is not a comparison of the models.

## Results

Eval3 test slices (5,027 full / 5,050 abbreviated / 5,041 mixed records) and the dev slice (`--dev-share 0.0905`; 498 / 475 / 484 records, where one record is 0.2 points). v2 has no weight to choose. The fused decoder's dev row is the one at its chosen weight, so it is the best of three on that slice. The end-to-end systems leave 13 abbreviated and 14 mixed test records unanswered, and 1 dev record per twin; the fused decoder leaves none. The best value in each row is bold.

| system | directory |
|---|---|
| v2 | `data/e2e-v2/g4/final` |
| v1 | `data/e2e/s5` |
| #127 (v1 + two fine-tune stages) | `data/e2e-stage2/s0` |
| #130 (#127 + ranking at 0.1) | `data/e2e-rankw01/s0` |
| fused | `data/ambiguity-slice/fused/<twin>-context-<on/off>` |

### Test, context on

| style | metric | v2 | v1 | #127 | #130 | fused |
|---|---|---|---|---|---|---|
| full | top-1 | 75.77% | 73.52% | 75.37% | 77.06% | **78.73%** |
| full | top-8 | 90.97% | 89.72% | 90.81% | **91.63%** | 85.88% |
| full | character | 94.38% | 93.69% | 94.10% | 95.44% | **96.00%** |
| full | MRR | 0.8178 | 0.7987 | 0.8142 | **0.8279** | 0.8173 |
| abbreviated | top-1 | 27.31% | 25.05% | 26.63% | **29.01%** | 28.42% |
| abbreviated | top-8 | **43.56%** | 39.33% | 41.80% | 43.29% | 36.83% |
| abbreviated | character | 66.31% | 63.25% | 65.27% | 67.63% | **67.71%** |
| abbreviated | MRR | 0.3303 | 0.2999 | 0.3197 | **0.3405** | 0.3170 |
| mixed | top-1 | 36.84% | 34.70% | 35.69% | 37.73% | **38.23%** |
| mixed | top-8 | **53.34%** | 50.47% | 52.43% | 53.20% | 48.86% |
| mixed | character | 74.23% | 72.39% | 73.68% | 75.19% | **76.08%** |
| mixed | MRR | 0.4274 | 0.4035 | 0.4170 | **0.4334** | 0.4222 |

### Test, context off

| style | metric | v2 | v1 | #127 | #130 | fused |
|---|---|---|---|---|---|---|
| full | top-1 | 69.41% | 65.86% | 68.97% | 70.70% | **71.85%** |
| full | top-8 | 87.13% | 85.30% | 86.87% | **87.77%** | 80.86% |
| full | character | 92.38% | 91.27% | 92.01% | 93.73% | **94.34%** |
| full | MRR | 0.7624 | 0.7337 | 0.7578 | **0.7734** | 0.7561 |
| abbreviated | top-1 | 17.05% | 14.18% | 16.36% | 17.25% | **17.27%** |
| abbreviated | top-8 | **30.42%** | 24.97% | 28.36% | 28.12% | 23.50% |
| abbreviated | character | 56.28% | 51.96% | 54.73% | 56.10% | **57.38%** |
| abbreviated | MRR | **0.2164** | 0.1791 | 0.2042 | 0.2105 | 0.1962 |
| mixed | top-1 | 27.97% | 24.82% | 27.22% | 27.97% | **29.14%** |
| mixed | top-8 | **43.60%** | 39.85% | 42.29% | 42.99% | 39.83% |
| mixed | character | 67.64% | 65.22% | 67.10% | 68.47% | **69.78%** |
| mixed | MRR | **0.3341** | 0.3017 | 0.3244 | 0.3336 | 0.3306 |

### Dev, context on

| style | metric | v2 | v1 | #127 | #130 | fused |
|---|---|---|---|---|---|---|
| full | top-1 | 77.51% | 73.90% | 77.71% | 78.71% | **79.52%** |
| full | top-8 | 91.16% | 89.36% | 90.76% | **92.17%** | 85.54% |
| full | character | 94.51% | 94.61% | 95.21% | 95.84% | **96.26%** |
| full | MRR | 0.8286 | 0.8017 | 0.8280 | **0.8377** | 0.8187 |
| abbreviated | top-1 | **28.42%** | 26.32% | 26.53% | 26.95% | 27.79% |
| abbreviated | top-8 | **43.58%** | 39.37% | 42.32% | 42.11% | 36.42% |
| abbreviated | character | 64.09% | 61.96% | 63.72% | 65.21% | **65.41%** |
| abbreviated | MRR | **0.3379** | 0.3069 | 0.3205 | 0.3283 | 0.3145 |
| mixed | top-1 | 36.36% | 34.50% | 36.98% | 38.64% | **40.70%** |
| mixed | top-8 | 56.20% | 53.10% | 54.13% | **56.61%** | 52.27% |
| mixed | character | 75.52% | 73.62% | 74.50% | 76.25% | **76.97%** |
| mixed | MRR | 0.4351 | 0.4111 | 0.4274 | **0.4523** | 0.4490 |

### Dev, context off

| style | metric | v2 | v1 | #127 | #130 | fused |
|---|---|---|---|---|---|---|
| full | top-1 | **72.89%** | 66.67% | 70.08% | 72.69% | **72.89%** |
| full | top-8 | 87.75% | 85.34% | 87.75% | **88.15%** | 81.53% |
| full | character | 93.12% | 92.40% | 93.00% | 94.09% | **94.51%** |
| full | MRR | 0.7840 | 0.7391 | 0.7722 | **0.7894** | 0.7651 |
| abbreviated | top-1 | 18.74% | 16.42% | 18.53% | 17.89% | **19.16%** |
| abbreviated | top-8 | 29.89% | 26.11% | **30.32%** | 29.26% | 25.68% |
| abbreviated | character | 55.53% | 51.54% | 54.20% | 55.05% | **56.57%** |
| abbreviated | MRR | **0.2279** | 0.1972 | 0.2275 | 0.2202 | 0.2173 |
| mixed | top-1 | 26.86% | 25.83% | 26.03% | 30.17% | **30.79%** |
| mixed | top-8 | **45.66%** | 40.91% | 43.18% | 44.83% | 41.32% |
| mixed | character | 68.10% | 66.01% | 67.74% | 68.96% | **70.55%** |
| mixed | MRR | 0.3324 | 0.3076 | 0.3171 | 0.3493 | **0.3500** |

### Record by record

Test top-1, record by record. "Gained" counts records v2 answers right and the other system wrong. z is McNemar's statistic over the two counts. The test records are those of the fused decoder's test dump, and every count of correct answers equals its report.

| style | v2 against v1 | v2 against #127 | v2 against #130 | v2 against fused |
|---|---|---|---|---|
| full, context on | 258 / 145, 5.63 | 164 / 144, 1.14 | 147 / 212, −3.43 | 178 / 327, −6.63 |
| abbreviated, context on | 244 / 130, 5.89 | 167 / 133, 1.96 | 176 / 262, −4.11 | 235 / 291, −2.44 |
| mixed, context on | 264 / 156, 5.27 | 201 / 143, 3.13 | 195 / 240, −2.16 | 239 / 309, −2.99 |
| full, context off | 333 / 155, 8.06 | 186 / 164, 1.18 | 188 / 253, −3.10 | 266 / 389, −4.81 |
| abbreviated, context off | 248 / 103, 7.74 | 141 / 106, 2.23 | 177 / 187, −0.52 | 206 / 217, −0.53 |
| mixed, context off | 287 / 128, 7.81 | 173 / 135, 2.17 | 220 / 220, 0.00 | 217 / 276, −2.66 |

### The #150 slice

`notes/ambiguity-slice.md` measures v2 on the slice against the fused decoder's own test dumps, beside v1, #127, #130 and #132. The outputs are in `data/ambiguity-slice/vs-fused/`. For v2, context on, full / abbreviated / mixed:

| | v2 | fused | records gained / lost, z | positions gained / lost, z |
|---|---|---|---|---|
| slice top-1 | 75.74% / 26.08% / 35.11% | 78.35% / 26.66% / 36.95% | −3.56 / −0.84 / −2.51 | |
| character at the positions | 94.46% / 78.94% / 82.84% | 96.03% / 81.65% / 85.15% | | 18 / 54, −4.24; 100 / 163, −3.88; 82 / 136, −3.66 |

With context off, v2 is level by records on abbreviated input (z −0.55), behind on full and mixed input (−2.86 and −1.97), and behind at the characters on all three (z −3.66 / −3.04 / −4.37). Over v1 the stream gains at the characters on every twin, 0.22 points on full pinyin and 1.33 and 1.58 on abbreviated and mixed input with context on. The pronouns do not move: v2 writes 他 for 她 15 times out of 53 with context on, as v1 does. Only the ranking stage changes that (10 for #132). The three #86 records are still unmeasured, since no system has a dump of the named set.

## What the numbers say

- **The order of the data was the cause.** v2 beats v1 by z 5.3 to 8.1 on every twin and context, with the same initialisation, recipe and epoch. The gain is larger with context off (2.9 to 3.6 points) than with it (2.1 to 2.3), and larger at top-8 on abbreviated input (4.23 points with context on, 5.45 off) than on full pinyin (1.25 and 1.83). That fits `notes/e2e-v1.md`'s reading: v1 ended its epoch on the corpus's hardest tail, at a near-zero rate.
- **The fine-tunes were a repair.** #127 is v1 plus #126's fine-tune and a second stage on fresh shards, about 4.7M more examples. v2 gets there with none of them. It is ahead of #127 at test top-1 in every row, significantly on mixed input with context on (z 3.13) and on abbreviated and mixed input with context off (2.23 and 2.17), and level on full pinyin (1.14 and 1.18). #127's note concluded that its gain was the rebalancing of v1's ending, not more training; v2 confirms it from the other side.
- **Ranking is still the gap.** v2's top-8 is 15.2 to 16.5 points above its top-1 with context on; the fused decoder's spread is 7.2 to 10.6 points. MRR is the one ranking measure where v2 is level with or above the fused decoder: 0.8178 against 0.8173 on full pinyin with context on, and above it on all six test rows. But it is behind #130, which is #127 with the ranking term, in every test row except context-off abbreviated and mixed, where it is level (z −0.52 and 0.00). So the stream does not replace the ranking stage. It is the better base for it.
- **Characters.** The fused decoder keeps a lead of 1.4 to 1.9 points of test character accuracy with context on and 1.1 to 2.1 off. The #150 slice shows the same at the ambiguous characters.
- **Against #149's acceptance.** v2 meets none of the criteria, and it was not expected to: it is the base, and #132 is the stage measured against them.

## Cost

| job | where | commit | time |
|---|---|---|---|
| the G4 leg, steps 1–300 | Colab G4 | a88f36e | 438.0 s of training, 314.4 s of it the first step |
| the A100 leg, steps 1–300 | Colab A100 | a88f36e | 1,034.5 s of training, 779.7 s of it the first step |
| the epoch, steps 301–59,577, and the six evaluations | Colab G4 | 0b83d4a | 5.89 h of training, 550.9 s of evaluation |

Compute units, from Colab's balance as recorded on #155: 2.87 CU for both legs, 54.16 CU for the G4 session that trained the epoch and evaluated it.

## Reproduce

1. Stamp the kernel: `uv run --project python mlime e2e stamp mlime-e2e-v2-g4 --out <dir>`. Its defaults are v1's recipe and the 30-step profile with 3 traced; `mlime-e2e-v2-a100` for the other leg.
2. A card leg: `colab/e2e.sh <dir> start --commit a88f36e --gpu G4 --session-hours 2 --census <batch-counts.json> --segment-steps 300`, then `colab/e2e.sh <dir> wait --deadline 3600`, `fetch <dir>/leg1` and `stop` (`data/e2e-v2/run-leg1.sh`, with `--gpu A100` for the other card). The census is the leg's own `batch-counts.json` in `data/e2e-v2/g4/leg1/`. `measure.json` in the fetched directory holds the rate, the GPU and host samples and the trace's summary.
3. The epoch: `colab/e2e.sh <dir> start --commit 0b83d4a --gpu G4 --session-hours 8 --allow-commit-change` resumes the step-300 trio. Then `wait --deadline 30600`, `fetch <dir>/leg2` and `stop` (`data/e2e-v2/run-g4-full.sh`). A finished run evaluates every `lattice*.jsonl`, context on and off, and publishes its final checkpoint, `run-config.json`, `run-summary.json` and the six evaluations with their dumps under `final/mlime-e2e-v2-g4/` in the `lexoliu/mlime-resume` bucket. A local copy is in `data/e2e-v2/g4/final/`.
4. One evaluation section locally, as for v1: `uv run --project python mlime eval e2e --checkpoint <final>/run/checkpoint-final.pt --lattice data/route-a-assets-v2/lattice.jsonl --eval-set data/route-a-assets-v2/eval3.jsonl --char-table data/route-a-assets-v2/char_pinyin.tsv --slice all --dump e2e-full-on.jsonl`.
5. The #150 slice: `ime-ambiguity` over the dumps, as `notes/ambiguity-slice.md` gives it.

## Next

- **The ranking stage on v2 (#132)** has already run: weight 0.05, chosen on dev, brings test top-1 with context on to 77.44% / 28.89% / 38.42% (`notes/sequence-ranking.md`). It is ahead of the fused decoder on abbreviated input and behind on full and mixed. The end-to-end line continues from there under #149.
- **The step's launch gaps.** The device is busy 76.8% of a traced step. Larger micro-batches with less accumulation, and CUDA graphs over fixed shapes, both change the recipe, so they belong to the next full run rather than to a rerun of this one. Compiled autograd does not change the recipe.
- **The #86 records by name** are still to be decoded under v2 and the fused decoder (`notes/ambiguity-slice.md`, Next).
