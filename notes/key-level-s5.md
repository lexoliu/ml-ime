# The key-level model, the character LM as a teacher S5 (2026-11-17, #298)

## Verdict

Teaching the transition the character LM's ordering of the ranking list passes the stage's gate and lifts every section, most of all where the input is noisiest: eval3-typo gains 5.4 points with context and 7.1 without. On clean full pinyin it gives 0.2 and 0.6, far less than rescoring S4's beam with the same LM did (1.6 and 2.9 on part of the test slice, #298). S5 replaces S4 as the model. The gap to the fused decoder on full pinyin stays: 2.4 points with context and 2.7 without.

`mlime-e2e-keys-s5` ran from `mlime-e2e-keys-s4` (`notes/key-level-s4.md`) at 93b2c14 with S4's settings. The evaluation is S4's: betas 1 / 2 / 4 / 8, the sum combination, weights chosen on dev per section, beam 8.

**The stage.** S4's list for an example is the gold and eight negatives of the no-grad beam, scored `S(y) = −constrained_ctc(y; γ·cost) + β·Σ log p_transition` at β 2 and γ 1. S5 has the frozen character LM (the 60M transformer of `notes/char-lm-v2.md`, read from `charlm-transformer-s1.pt`) score every sentence of the list under the example's context and adds `KL(q ‖ p)` over the list at weight 1, with `p ∝ exp(S)` and `q ∝ exp(sg(S) + 0.6 · log p_LM)`. The ranking cross-entropy on the gold stays at 0.05. One epoch over 22 natural shards disjoint from the earlier stages' and the held-out lists, 1,127,296 examples kept of 1,135,198 seen, 387,217 of them with a typo; 1,973 updates.

**Sentence top-1 on test**, context on / off:

| set | S5 | S4 | S5 − S4 | fused | S5 − fused | #132 (on) |
|---|---|---|---|---|---|---|
| full | 76.37 / 69.15 | 76.15 / 68.57 | +0.22 / +0.58 | 78.73 / 71.85 | −2.36 / −2.70 | 77.44 |
| abbreviated | 29.58 / 17.62 | 28.48 / 16.30 | +1.10 / +1.32 | 28.42 / 17.27 | +1.16 / +0.35 | 28.89 |
| mixed | 40.01 / 29.44 | 39.56 / 28.76 | +0.45 / +0.68 | 38.23 / 29.14 | +1.78 / +0.30 | 38.42 |
| eval3-typo | 54.49 / 47.26 | 49.06 / 40.16 | +5.43 / +7.10 | | | |

Top-8 with context: S5 91.05 / 45.35 / 57.67 / 70.30, S4 90.69 / 44.93 / 57.23 / 65.80. No record unanswered in any section. The fused decoder and #132 are as in `notes/key-level-s2.md`.

**The gate** is on dev: full pinyin top-1 above S4's on both context settings, no twin down by more than one record. Records right, S5 against S4:

| set | context on | context off |
|---|---|---|
| full (498) | 390 / 387 | 355 / 351 |
| abbreviated (475) | 133 / 131 | 95 / 85 |
| mixed (484) | 195 / 194 | 147 / 142 |
| eval3-typo (498) | 278 / 263 | 250 / 228 |

It passes, on full pinyin by three and four records.

**Against #216 §8.** Abbreviated and mixed are now above the fused decoder on both context settings and above #132 with context. Full pinyin is not: 2.36 and 2.70 under the fused decoder, 1.07 under #132.

**What the teacher did.** Through training the model's distribution over the list sat about 0.2 nats from the teacher's (0.22, 0.18 and 0.22 over the leg's thirds), and the teacher's first choice differed from the model's on 10% to 14% of lists, the gold being the teacher's first on 54% to 65%. The term did not shrink over the epoch. On clean full pinyin the test gain is 11 and 29 records of 5,027; the LM's ordering of S4's finished top-8 was worth several times that, so most of what the LM knows about those lists did not reach the transition in one epoch at this weight.

**Sweep.** With context the chosen beta is 2 on full, 1 on abbreviated and mixed, 2 on typo with gamma 2; without, 1, 1, 2 and 4. S4 chose 8 on both typo sections and on mixed without context: the transition now carries more of the ordering at a small weight.

**Run.** An A100. The leg that carried updates 1 to 1,890 ran at 5.52 s an update against S4's 4.7, GPU 70% busy over the training window with 9% of samples under half, and ended out of memory as the process grew from 28.5 to 40.4 GB (#304). A fresh process resumed from the trio at 1,689 and finished; the eight sections' sweep and test decode took about 50 minutes. Two starts were lost before the long leg to runtimes that disappeared during staging, and one to the teacher's loader importing the ONNX exporter, which the training image does not carry (#303).

## Reproduce

```sh
uv run --project python mlime e2e stamp mlime-e2e-keys-s5 --out DIR \
  --init-from mlime-e2e-keys-s4 --keys --key-ranking --base-lr 2e-5 --new-lr 2e-5 \
  --token-budget 36864 --accumulate 8 --search-graphs \
  --typos --unit-initials 0.5 --unit-lexicon unit-entries.tsv \
  --key-rank-teacher-weight 1.0 --key-rank-teacher-scale 0.6 \
  --train-shard ...   # 22 natural shards outside the earlier stages' and the held-out lists
colab/e2e.sh DIR start --session-hours 6 --gpu A100
colab/e2e.sh DIR wait && colab/e2e.sh DIR fetch && colab/e2e.sh DIR stop
```

The logs, selections and each section's dev and test dumps are in the `lexoliu/mlime-resume` bucket under `final/mlime-e2e-keys-s5/`.

## Next

- The memory growth of a ranking leg (#304) before another one longer than about 1,500 updates.
- The teacher at a larger weight or over a second epoch is the cheap follow-on if full pinyin is to be pushed this way; what the list never contains stays with word arcs (#282).
- The session engine's latency on an M1 (#151); S5 has S4's architecture, so S4's Core ML conversion applies unchanged.
