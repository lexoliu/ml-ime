# The key-level model, sequence-level ranking S4 (2026-11-13, #278)

## Verdict

Ranking the search's own candidates lifts every section read with context and the typo twin by five points, and it does not close the gap on full pinyin. With context the model is now above the fused decoder on the abbreviated and mixed styles and 2.6 points under it on full pinyin; without context it trails on all three. #216 §8's gate on the clean twins is therefore not met: of the six sections two pass.

`mlime-e2e-keys-s4` ran from `mlime-e2e-keys-s3` (`notes/key-level-s3.md`) at fd32b86, typos kept on. The evaluation is S3's: betas 1 / 2 / 4 / 8, the sum combination, weights chosen on dev per section, beam 8.

**The stage.** The training score of a sentence is the search's: `S(y) = −constrained_ctc(y; γ·cost) + β·Σ log p_transition`, at β 2 and γ 1. To S3's terms the stage adds 0.05 × the cross-entropy of the gold among itself and eight negatives, which a no-grad lane beam on the GPU draws from the current model every update. One epoch over 22 natural shards disjoint from S2's and S3's, 1,194,526 examples kept of 1,199,963 seen; typos at .35 (411,129 sentences, 632,875 edits), unit initials at 0.5; lr 2e-5 / 2e-5, 36,864 tokens × 2 ranks, accumulate 8, 2,087 updates of about 535 examples.

**Sentence top-1 on test**, context on / off:

| set | S4 | S3 | S4 − S3 | fused | S4 − fused | #132 (on) |
|---|---|---|---|---|---|---|
| full | 76.15 / 68.57 | 75.49 / 67.87 | +0.66 / +0.70 | 78.73 / 71.85 | −2.58 / −3.28 | 77.44 |
| abbreviated | 28.48 / 16.30 | 27.60 / 16.57 | +0.88 / −0.27 | 28.42 / 17.27 | +0.06 / −0.97 | 28.89 |
| mixed | 39.56 / 28.76 | 38.68 / 28.19 | +0.88 / +0.57 | 38.23 / 29.14 | +1.33 / −0.38 | 38.42 |
| eval3-typo | 49.06 / 40.16 | 43.66 / 35.01 | +5.40 / +5.15 | | | |

Top-8 with context: S4 90.69 / 44.93 / 57.23 / 65.80, S3 90.31 / 44.42 / 55.19 / 61.47. No record unanswered in any section. The fused decoder and #132 are as in `notes/key-level-s2.md`.

**Before the run**, two 800-update legs from S3 on the same stream, one with the ranking term and one without, were decoded on dev with context: 921 against 898 records right of 1,955 over the four sets (+1.18 points; typo +14, full +5, abbreviated +2, mixed +2). That was the gate for spending the epoch, and the full run's gain over S3 has the same shape: most of it on typos.

**The knowledge slice** (test, with context; slice top-1, then entity exact for idioms):

| | S4 | S3 | fused |
|---|---|---|---|
| full, all entities | 76.83% | 76.64% | 80.12% |
| abbreviated, all | 27.73% | 24.41% | 25.59% |
| mixed, all | 37.74% | 36.97% | 37.36% |
| abbreviated, idiom entity exact | 43.27% | 45.19% | 52.88% |
| mixed, idiom entity exact | 54.55% | 52.73% | 58.18% |

Abbreviated records with an entity gain 22 and lose 5 against S3 (z 3.27). Idiom entities do not move (abbreviated 2 gained, 4 lost): the stage ranks what the search can already reach, and an idiom typed as initials is not among it. That stays with word arcs (#282).

**The ambiguity slice** (test, with context; S4 then S3): sentence top-1 75.64% / 75.64% full, 28.92% / 27.92% abbreviated, 39.33% / 38.38% mixed; the confusable characters themselves 95.77% / 95.98%, 81.91% / 82.29%, 85.49% / 85.79%. 他/她/它 on full pinyin is 81.82% against 84.19% (207 against 213 of 253). None of the character differences separates (|z| ≤ 1.3).

**Against the gate.** §8 asks for the fused decoder's 78.73 / 28.42 / 38.23 with context and 71.85 / 17.27 / 29.14 without, and #132's 77.44 / 28.89 / 38.42.

- Abbreviated and mixed with context pass the fused decoder; mixed passes #132 too, abbreviated is 0.41 under it.
- Full pinyin is 2.58 and 3.28 under the fused decoder and 1.29 under #132. The ranking stage gave #132 its full-pinyin points; here it gives 0.66.
- Without context all three trail, by 0.4 to 3.3.

What remains on full pinyin is not something this stage's weight or another epoch of it is likely to find: the comparison legs and the full run agree on a gain of about a point on clean input. The open levers are the ones #216 lists after it, word arcs for entries the network does not spell from initials and the context-off half of the stream.

**Sweep.** With context the chosen beta is 1 on full, 2 on abbreviated and mixed, 8 on typo with gamma 2; S3 chose 8 on mixed. The stage trains at beta 2, and the search now prefers a beta near it.

**Run.** An A100: 9,093 s for updates 141 to 2,087 after an earlier leg's 140, about 4.7 s an update, GPU 75% busy over the training window with 7% of samples under half. The eight sections' sweep and test decode took 49 minutes. Three starts were lost before it: a captured search graph invalidated by another thread's allocation (#283), a runtime released while its waiter starved, and a compiled kernel collected inside a capture (#286). The finished run then waited an hour on a storage quota before its outputs could be published.

## Reproduce

```sh
uv run --project python mlime e2e stamp mlime-e2e-keys-s4 --out DIR \
  --init-from mlime-e2e-keys-s3 --keys --key-ranking --base-lr 2e-5 --new-lr 2e-5 \
  --token-budget 36864 --accumulate 8 --search-graphs \
  --typos --unit-initials 0.5 --unit-lexicon unit-entries.tsv \
  --train-shard ...   # 22 natural shards outside S2's, S3's and the held-out lists
colab/e2e.sh DIR start --session-hours 4 --gpu A100
colab/e2e.sh DIR wait && colab/e2e.sh DIR fetch && colab/e2e.sh DIR stop
```

The logs, selections and each section's dev and test dumps are in the `lexoliu/mlime-resume` bucket under `final/mlime-e2e-keys-s4/`. The slice rows are `ime-knowledge --slice test` and `ime-ambiguity --slice test` over the test dumps, S3's dump the baseline.

## Next

- Word arcs over keys (#282) on this checkpoint: the prior and the lexicon chosen on dev, then the six clean sections and the idiom rows with and without them.
- The Rust constrained search and the session engine (#224, #151) on this checkpoint, and the per-keystroke latency on an M1.
