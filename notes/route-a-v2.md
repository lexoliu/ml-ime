# Route A v2 — two epochs over all of run3 (2026-09-19)

Kernels `lexoliu/mlime-route-a-v2-s0` … `-s6`, 2×T4 DDP, launched 2026-09-09,
finished 2026-09-19 and evaluated on `lexos-mac-mini` by `kaggle/finish.sh`.
Same architecture, augmentation and decoder as v1 (`notes/route-a-v1.md`);
what changed is the data (all 41.28M labelled run3 segments instead of the
10.03M subset, with the reading arbitration of `notes/v2-data-prep.md`), the
length of training (two epochs instead of one), and the evaluation
(abbreviated and mixed typing next to full pinyin).

## Verdict

**Scaling the data did what the v1 loss curve said it would.** With context
on, the fused decoder reaches 76.33% sentence top-1 on full pinyin, +5.2 over
v1 and +21.2 over the run3 trigram; the context-off fused system alone (71.20%)
now matches v1's context-on number. **Abbreviations are where the neural route
earns its place and where it is still not enough**: the trigram alone gets
7.0% of fully abbreviated sentences and 16.2% of mixed ones, the fused
context-on decoder 24.3% and 34.2%, with 65.6% and 74.5% of characters right.
That is 3.5× and 2.1× the baseline, and it is not a usable input method for
whole abbreviated sentences yet. The limiting factor is visible in the same
table: neural-only top-8 barely exceeds top-1 in every setting, so the beam
has no useful alternatives to rank. The next modelling step is a rescorer over
the NAR output, not more data.

## Training

| quantity | value |
|---|---:|
| data | run3, 41.28M labelled segments (`mlime-run3-v1-*` + `mlime-run3-rest-*`), 20.26M kept per rank per epoch |
| epochs | 2 |
| steps | 244,797 (`mlime train count-batches`), 165.5 examples/step/rank, 1.10 steps/s |
| wall | 61.8 h of training over seven Kaggle sessions, + 1.1 h scoring |
| loss | 4.069 → ≈1.1 (label smoothing 0.05; per-batch, see the segment table) |
| gates (top 4 layers) | 0.190, −0.129, 0.230, −0.224 |
| held-out char acc (mixed typing aug.) | context on **85.56%**, off 80.58% (40,412 chars, 4,092 examples) |

Build counts per rank: kept 20,256,825 of 20,350,656 seen; reading_arbitrated
966,652 (4.7%, the Taiwan→mainland map); unemittable_character 87,350;
unknown_span 3,586; target_not_admitted 2,843 (0.014%, down from 3.1% in v1);
unlabelled 52.

The run was a chain of resumable segments (`kaggle/route-a-v2/kernel.py`,
`kaggle/chain.sh`): each segment restores the optimiser, scheduler, scaler,
per-rank data position and every RNG from the previous segment's checkpoint,
trains on a wall budget and pauses; the segment that reaches 244,797 steps
scores the six lattices. Segment 4 was cut short by the weekly quota, segment 6
finished with the 3 h scoring reserve intact.

| segment | steps | loss first → last | train h | note |
|---|---:|---:|---:|---|
| s0 | 1 → 37,960 | 4.069 → 1.018 | 8.9 | counted the batches, then trained |
| s1 | 37,965 → 84,100 | 1.067 → 0.959 | 11.6 | |
| s2 | 84,102 → 132,750 | 0.874 → 0.780 | 11.9 | |
| s3 | 132,754 → 177,590 | 0.907 → 0.910 | 11.5 | |
| s4 | 177,595 → 191,350 | 0.917 → 0.662 | 3.7 | 4 h session, end of quota week |
| s5 | 191,353 → 237,960 | 0.716 → 1.061 | 11.8 | paused for the scoring reserve |
| s6 | 237,962 → 244,790 | 1.307 → 1.135 | 2.5 | finished, scored |

The per-batch loss is noisy (batches differ in length and typing style), so the
first/last columns say little beyond the s0 → s6 trend; the held-out accuracy
is the number to read, +5.4 (context on) and +5.7 (off) over v1.

## Evaluation

Three eval sets of the same 5,525 sentences (`notes/v2-data-prep.md`): full
pinyin (`eval3.jsonl`), abbreviated (`eval3-abbreviated.jsonl`, each syllable
reduced to its initial with p=0.7) and mixed (`eval3-mixed.jsonl`, a full
prefix and an abbreviated suffix). For every set and context setting the fusion
weight was swept over 0.5, 0.75, 1, 1.5, 2 on the dev slice (`--dev-share
0.0905`, 475–498 records) and the chosen weight is reported on the disjoint
test slice (≈5,030 records). Decoder settings as in v1: beam 16, 8 readings,
top-k 8, unscored −30. The trigram is the run3 trigram (41M lines, 643 MiB)
throughout. Full table: `data/route-a-v2/s6/results.md`.

### Full typing

| configuration | top-1 | top-8 | char | MRR@8 |
|---|---:|---:|---:|---:|
| trigram only | 55.10% | 62.58% | 88.61% | 0.581 |
| neural only, context off | 59.36% | 62.78% | 90.48% | 0.607 |
| fused, context off, w=1 | 71.20% | 76.73% | 94.17% | 0.734 |
| neural only, context on | 69.03% | 72.19% | 93.38% | 0.703 |
| **fused, context on, w=1** | **76.33%** | **81.26%** | **95.43%** | **0.784** |

Against v1 (same trigram, same slice): neural only, context off 44.84 → 59.36
(+14.5); neural only, context on 53.45 → 69.03 (+15.6); fused, context off
65.55 → 71.20 (+5.7); fused, context on 71.18 → 76.33 (+5.2).

### Abbreviated typing

| configuration | top-1 | top-8 | char | MRR@8 |
|---|---:|---:|---:|---:|
| trigram only | 7.01% | 10.14% | 40.31% | 0.081 |
| neural only, context off | 8.02% | 9.27% | 50.24% | 0.085 |
| fused, context off, w=0.5 | 14.83% | 19.49% | 55.64% | 0.165 |
| neural only, context on | 16.38% | 18.53% | 61.84% | 0.173 |
| **fused, context on, w=1.5** | **24.32%** | **30.16%** | **65.59%** | **0.266** |

### Mixed typing

| configuration | top-1 | top-8 | char | MRR@8 |
|---|---:|---:|---:|---:|
| trigram only | 16.19% | 25.09% | 57.16% | 0.194 |
| neural only, context off | 16.64% | 20.17% | 63.21% | 0.179 |
| fused, context off, w=0.5 | 25.97% | 35.81% | 67.97% | 0.295 |
| neural only, context on | 25.67% | 29.76% | 71.26% | 0.272 |
| **fused, context on, w=1** | **34.20%** | **44.10%** | **74.48%** | **0.378** |

### Dev-slice weight sweep

Sentence top-1 on the dev slice (hits / records in parentheses), the numbers
the reported weights were chosen from:

| set, context (dev records) | w=0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|---:|
| full, off (498) | 70.9 (353) | 71.3 (355) | **71.7 (357)** | 71.3 (355) | 70.1 (349) |
| full, on (498) | 75.5 (376) | 76.7 (382) | **77.7 (387)** | 77.3 (385) | 77.5 (386) |
| abbreviated, off (475) | **17.3 (82)** | 17.3 (82) | 17.1 (81) | 16.4 (78) | 15.2 (72) |
| abbreviated, on (475) | 23.6 (112) | 24.4 (116) | 24.4 (116) | **24.6 (117)** | 23.4 (111) |
| mixed, off (484) | **27.1 (131)** | 26.5 (128) | 24.8 (120) | 24.6 (119) | 23.8 (115) |
| mixed, on (484) | 36.0 (174) | 37.4 (181) | **37.6 (182)** | 36.4 (176) | 34.7 (168) |

The bold cell is the weight `--select-on-dev` picked (ties go to the first
weight in the given order). With context the plateau is 0.75–2.0 as in v1;
without it the best weight falls to 0.5 as soon as the typing is abbreviated,
that is, the sharper the emissions have to be about ambiguous keystrokes, the
less of them the trigram should be allowed to override.

## What the numbers say

- **The model, not the trigram, carries v2.** Neural-only rows gained 14–16
  points over v1 while the fused rows gained 5–6, so the trigram's share of
  the fused result shrank: fused minus neural-only with context is 7.3 points
  on full pinyin (17.7 in v1). The n-gram still matters, and more so without
  context (+11.8), but it is no longer half the system.
- **Context is worth more the less the user types.** Fused, context on minus
  off: +5.1 full, +8.2 mixed, +9.5 abbreviated; neural-only on versus off
  doubles the abbreviated top-1 (8.0 → 16.4). The gates moved further from
  zero than in v1 (0.19–0.23 against 0.10–0.23) and two went negative, which
  is a sign of the model using the cross-attention rather than a problem.
- **Abbreviation is a beam-diversity problem.** In every setting neural-only
  top-8 is within 1–3 points of top-1 (72.2 vs 69.0 full, 18.5 vs 16.4
  abbreviated): the per-position independent distributions give the beam one
  good hypothesis and seven near-copies of it. The trigram adds coherence but
  reads two characters back. A rescorer that sees the whole candidate (an
  autoregressive pass over the top-k, or an iterative refinement of the NAR
  output) is the missing piece, and abbreviated input is where it will show.
- **Character accuracy tells the product story.** 95.4% of characters are
  right on full pinyin, 74.5% mixed, 65.6% abbreviated. Most remaining
  full-pinyin errors are one wrong character in a long sentence; on
  abbreviated input a third of the characters are wrong, which is a different
  regime and the reason the sentence numbers are what they are.
- **The fusion weight drifts with typing style**, 1.0 on full pinyin, 1.5
  with context on abbreviated, 0.5 without: the emissions are sharper in some
  regimes than others. The dev slice is a coarse instrument for choosing it
  (475 records; abbreviated context-off ties 82/475 at w=0.5 and 0.75, and
  the two weights differ by 0.1 points on test). One weight per typing style,
  or a calibrated emission that needs no per-style weight, is cheap to do.

## Reproduce

```
kaggle kernels output lexoliu/mlime-route-a-v2-s6 -p data/route-a-v2/s6
kaggle datasets download lexoliu/mlime-route-a-assets -p data/route-a-assets-v2 --unzip
kaggle/finish.sh data/route-a-v2/s6            # writes data/route-a-v2/s6/results.md, 33 min on the M1
```

`kaggle/finish.sh` needs `target/release/ime-cli`, `data/run3/ngram.bin` and
`data/run3_pool/eval3{,-abbreviated,-mixed}.jsonl`; single rows come from
`ime-cli fused-eval` as documented in `notes/route-a-v1.md` §Reproduce, with
`scores-lattice{,-abbreviated,-mixed}-context-{on,off}.jsonl.gz` from the
segment directory, and `--select-on-dev` replaces `--slice` to sweep the dev
slice and report test at the winning weight in one run.

## Next

1. A rescorer over the NAR output: an autoregressive pass over the beam's
   top-k or an iterative refinement of the fill tower's distribution, measured
   first on the abbreviated set where top-8 − top-1 is the whole gap.
2. Issue #16: hash text and context only in `EvalRecord::digest` so the three
   typing twins share one dev/test split; then choose the fusion weight per
   typing style on a larger dev share.
3. Inference on macOS: the model is trained, `notes/inputmethodkit.md` and
   `notes/compute.md` hold what was known when it was parked.
