# The key-level model, data enrichment S3 (2026-11-12, #261)

## Verdict

S3 teaches the key-level model to read typos and costs it under a point on clean input; it does nothing for idioms typed as initials. Sentence top-1 on eval3-typo goes from 4.32% to 43.66% with context. The six clean sections fall by 0.3 to 0.95. The knowledge slice does not move, and idiom entities are exactly where S2 left them. S3 is the base of the ranking stage, with typos kept on through it; idioms go to word arcs in the search (#216 §5, #224).

`mlime-e2e-keys-s3` ran from `mlime-e2e-keys-s2` (`notes/key-level-s2.md`) at e28513f. S2's final was decoded again under the same evaluation (betas 1 / 2 / 4 / 8, the typo twin, #273, #276), so the two are compared like for like; S2's numbers here therefore differ slightly from its own note's, which chose among betas 0.5 / 1 / 2.

**Data.** One epoch over 25 shards, 1,339,753 examples kept of 1,346,782 seen: the 20-pick's 22 natural shards, the two lift shards (117,961 rows: 10,880 idiom and internet entries under 20 occurrences lifted, 4,752 of them to the floor, 6,128 short of it at 8 repeats a sentence) and the judged mined sentences (28,148 rows of the 146,480 mined). Natural text is 89% of the stage. Typos at .35: 452,698 sentences carry 696,463 edits. Unit initials at 0.5. lr 2e-5 / 2e-5, 24,576 tokens × 2 ranks, accumulate 2, 2,506 updates.

**Sentence top-1 on test**, beam 8, weights chosen on dev per section, context on / off:

| set | S3 | S2 | S3 − S2 |
|---|---|---|---|
| full | 75.49 / 67.87 | 75.81 / 68.61 | −0.32 / −0.74 |
| abbreviated | 27.60 / 16.57 | 28.46 / 17.45 | −0.86 / −0.88 |
| mixed | 38.68 / 28.19 | 39.50 / 29.14 | −0.82 / −0.95 |
| eval3-typo | 43.66 / 35.01 | 4.32 / 3.20 | +39.34 / +31.81 |

Top-8 with context: S3 90.31 / 44.42 / 55.19 / 61.47, S2 91.15 / 45.94 / 57.27 / 9.77. No record unanswered in any section.

**The knowledge slice** (test, with context; slice top-1, then entity exact for idioms):

| | S3 | S2 | fused |
|---|---|---|---|
| full, all entities | 76.64% | 75.87% | 80.12% |
| abbreviated, all | 24.41% | 25.00% | 25.59% |
| mixed, all | 36.97% | 37.93% | 37.36% |
| abbreviated, idiom entity exact | 45.19% | 45.19% | 52.88% |
| mixed, idiom entity exact | 52.73% | 52.73% | 58.18% |

Over all entities no difference between S3 and S2 separates (|z| ≤ 1.1 by records).

**Against the gate.**

- eval3-typo is higher by 39 points. S2 cannot read typed errors at all; S3 can.
- The clean twins are lower in all six sections, by 0.3 to 0.95. No single section separates at this size, but they all point the same way, so the stage does cost clean accuracy. That is the price of a third of the stream carrying typos at a low rate for 2,506 updates.
- The knowledge slice is not higher. Idioms typed as initials are exactly where S2 left them, to the entity. Lifting 10,880 rare entries and typing units as initials for 2,506 updates at 2e-5 does not teach the network twenty-odd thousand idioms by their initials. This is the case #216 §5 keeps word arcs for: the lexicon offers the idiom as a legal emission and the network scores it.

So S3 is taken as the base of the ranking stage for what it adds, with the typo augmentation kept on through that stage so it is not unlearned, and the clean twins are what the ranking stage has to win back. Idioms go to word arcs in the search (#224), not to more data.

**Sweep.** On the typo twin the best beta is 8, the grid's top, in both contexts; on the clean twins it is 1 to 8 with dev top-1 nearly flat across them. Gamma matters on typos (0.5 to 1 best, 2 worse at low beta).

**Run.** A G4, 1,184 s of training at 0.47 s per update, GPU 96% busy over the training window, 16.5 GB peak; the eight sections' sweep and test decode took about 29 minutes. 9.1 CU for the stage, 4.6 CU for S2's evaluation leg.

## Reproduce

```sh
uv run --project python mlime e2e stamp mlime-e2e-keys-s3 --out DIR \
  --init-from mlime-e2e-keys-s2 --keys --base-lr 2e-5 --new-lr 2e-5 \
  --token-budget 24576 --accumulate 2 \
  --typos --unit-initials 0.5 --unit-lexicon unit-entries.tsv \
  --train-shard ...   # the 20-pick's 22 shards, lift-00000, lift-00001, lift-mined-00000
colab/e2e.sh DIR start --session-hours 4 --gpu G4
colab/e2e.sh DIR wait && colab/e2e.sh DIR fetch && colab/e2e.sh DIR stop
```

The logs, selections and each section's dev and test dumps are in the `lexoliu/mlime-resume` bucket under `final/mlime-e2e-keys-s3/` and, for S2 under this evaluation, `final/mlime-e2e-keys-s2/`. The slice rows are `ime-knowledge --slice test` over the test dumps, the first dump the baseline (`notes/knowledge-slice.md`).

## Next

- The ranking stage S4 from this checkpoint, typos on, gated on #216 §8's clean twins.
- Word arcs over keys in the constrained search for idioms typed as initials.
