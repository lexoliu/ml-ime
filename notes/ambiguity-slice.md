# The ambiguity slice (2026-10-20, #150)

## Verdict

The slice exists, and the fused decoder and the end-to-end evaluation both report it from their dumps, through one implementation. Its main comparison is not measured yet. No dump of the fused decoder's best configuration exists: route A v2 + the run3 trigram + the int8 transformer reader, `notes/char-lm-v2.md`. The tables below set the end-to-end checkpoints against a reference that does have dumps: route A v2 + the trigram, without the reader (76.33 / 24.32 / 34.20 test top-1).

Among the end-to-end checkpoints, #130 (ranking weight 0.1) is best on the slice in every typing style with context on. On full pinyin it is level with the trigram reference at the ambiguous characters (95.98% against 95.90%, McNemar z 0.26). v1, #127 and v2 are below the reference there (z −4.20, −4.48, −3.46). Almost all of that gap is 的 / 地 / 得, not the pronouns.

On the pronouns, the end-to-end line does what #86 asked of it: it writes 他 for 她 less often. With context on full pinyin, 她→他 falls from 24 of 53 (trigram reference) to 15 (v1) and 11 (#130). #130 then writes 她 for 他 12 times, so 他 / 她 / 它 accuracy is level: 83.00% against 83.40%.

In eval3 the context, not later input, decides the pronoun. Without context, v1 writes 他 for 48 of the 53 她. So this slice mostly measures how the context is used. The later-input case `tashiwojiejie` is rare in eval3, and it is reported by name only once the named set is decoded (below).

## The slice

The sets are in code, `ime_eval::AMBIGUITIES`:

| set | members |
|---|---|
| pronouns | 他 / 她 / 它 |
| particles | 的 / 地 / 得 |

A position of a record's expected text is in the slice when:
1. its character is a member of a set, and
2. the span typed for it also admits another member of the same set.

The second condition drops what the keystrokes already decide. 得 typed `dei` admits no other member, so it stays out. 的 typed `di` (的确) admits 地 and stays in. The slice does not depend on any system: which span typed which character is recovered from the keystrokes and the expected text alone. `SegmentLattice::cheapest_reading` finds the cheapest reading that spells the text, searching every edge of the lattice with abbreviations on and no half-typed tail. A record whose keystrokes spell no reading of its text is counted as unspelled. eval3 has none.

Per twin, test slice:

| twin | records | positions | 他 / 她 / 它 | 的 / 地 / 得 |
|---|---|---|---|---|
| full | 1,880 | 2,291 | 253 | 2,038 |
| abbreviated | 1,902 | 2,327 | 246 | 2,081 |
| mixed | 1,897 | 2,337 | 264 | 2,073 |

The metrics are:
- **slice top-1:** sentence top-1 over the records holding at least one position;
- **character accuracy at the positions:** the top hypothesis's character at the position, compared positionally as `ime-eval`'s character accuracy is;
- **swapped:** of the errors at the positions, those that wrote another member of the same set.

The dev/test split is the usual hash at `--dev-share 0.0905`. eval3's slices, weights, beam width and metrics are unchanged.

## Results

Test slices. "trigram" is route A v2 + the run3 trigram at the dev-chosen weights (1 / 1.5 / 1), from `data/rescore-v2`. It is context-on and test-only, and it is not the fused decoder of #149.

### Context on

| style | system | slice top-1 | character | 他 / 她 / 它 | 的 / 地 / 得 | swapped |
|---|---|---|---|---|---|---|
| full | trigram | 76.49% | 95.90% | 83.00% | 97.50% | 86 |
| full | v1 | 74.15% | 94.24% | 81.03% | 95.88% | 84 |
| full | #127 | 75.74% | 94.06% | 80.24% | 95.78% | 89 |
| full | #130 | **77.29%** | **95.98%** | **83.40%** | **97.55%** | 83 |
| full | v2 | 75.74% | 94.46% | 82.21% | 95.98% | 82 |
| abbreviated | trigram | 22.71% | 80.58% | 75.20% | 81.21% | 117 |
| abbreviated | v1 | 25.50% | 77.61% | **77.24%** | 77.65% | 96 |
| abbreviated | #127 | 26.29% | 78.21% | 76.83% | 78.38% | 101 |
| abbreviated | #130 | **28.86%** | **80.75%** | 75.61% | **81.36%** | 104 |
| abbreviated | v2 | 26.08% | 78.94% | 76.02% | 79.29% | 96 |
| mixed | trigram | 32.58% | **84.55%** | 76.89% | **85.53%** | 119 |
| mixed | v1 | 33.58% | 81.26% | 76.52% | 81.86% | 100 |
| mixed | #127 | 34.63% | 81.94% | 74.24% | 82.92% | 107 |
| mixed | #130 | **36.58%** | 83.14% | **77.27%** | 83.89% | 101 |
| mixed | v2 | 35.11% | 82.84% | 75.00% | 83.84% | 105 |

### Context off

No fused dump exists context-off.

| style | system | slice top-1 | character | 他 / 她 / 它 | 的 / 地 / 得 | swapped |
|---|---|---|---|---|---|---|
| full | v1 | 65.37% | 91.40% | 67.98% | 94.31% | 124 |
| full | #127 | 68.30% | 91.66% | **69.17%** | 94.46% | 124 |
| full | #130 | **69.79%** | **93.50%** | 68.38% | **96.61%** | 132 |
| full | v2 | 68.56% | 92.36% | 68.77% | 95.29% | 122 |
| abbreviated | v1 | 14.51% | 68.63% | 55.69% | 70.16% | 149 |
| abbreviated | #127 | 16.04% | 70.73% | **62.20%** | 71.74% | 146 |
| abbreviated | #130 | **17.03%** | 71.51% | 56.91% | 73.23% | 166 |
| abbreviated | v2 | 16.14% | **71.81%** | 58.94% | **73.33%** | 130 |
| mixed | v1 | 23.72% | 75.87% | 60.61% | 77.81% | 152 |
| mixed | #127 | 26.25% | 77.06% | **62.88%** | 78.87% | 139 |
| mixed | #130 | **27.36%** | 77.75% | 59.09% | **80.13%** | 152 |
| mixed | v2 | 26.73% | **77.83%** | 60.98% | 79.98% | 141 |

### Record by record, context on

"Gained" counts what the first system gets right and the second wrong. z is McNemar's statistic over the two counts. Positions within one record are not independent, so the position z overstates the evidence where a record holds several.

| style | comparison | slice records: gained / lost, z | positions: gained / lost, z |
|---|---|---|---|
| full | v1 vs trigram | 87 / 131, −2.98 | 22 / 60, −4.20 |
| full | #130 vs trigram | 121 / 106, 1.00 | 31 / 29, 0.26 |
| full | v2 vs trigram | 107 / 121, −0.93 | 29 / 62, −3.46 |
| full | #130 vs v1 | 105 / 46, 4.80 | 46 / 6, 5.55 |
| full | v2 vs #130 | 53 / 82, −2.50 | 15 / 50, −4.34 |
| abbreviated | v1 vs trigram | 120 / 67, 3.88 | 93 / 162, −4.32 |
| abbreviated | #130 vs trigram | 174 / 57, 7.70 | 127 / 123, 0.25 |
| abbreviated | v2 vs trigram | 132 / 68, 4.53 | 131 / 169, −2.19 |
| abbreviated | #130 vs v1 | 108 / 44, 5.19 | 149 / 76, 4.87 |
| abbreviated | v2 vs #130 | 51 / 104, −4.26 | 99 / 141, −2.71 |
| mixed | v1 vs trigram | 106 / 87, 1.37 | 83 / 160, −4.94 |
| mixed | #130 vs trigram | 155 / 79, 4.97 | 96 / 129, −2.20 |
| mixed | v2 vs trigram | 123 / 75, 3.41 | 103 / 143, −2.55 |
| mixed | #130 vs v1 | 106 / 49, 4.58 | 107 / 63, 3.37 |
| mixed | v2 vs #130 | 65 / 93, −2.23 | 93 / 100, −0.50 |

### Which way the pronouns go

Full pinyin, test: 178 他, 53 她, 22 它.

| system | 她→他 | 他→她 | 它→他 | 他→它 | the rest |
|---|---|---|---|---|---|
| trigram | 24 | 5 | 11 | 2 | 1 |
| v1, context on | 15 | 9 | 10 | 5 | 9 |
| #130, context on | 11 | 12 | 10 | 5 | 4 |
| v2, context on | 15 | 9 | 10 | 4 | 7 |
| v1, context off | 48 | 4 | 15 | 6 | 8 |
| #130, context off | 39 | 18 | 12 | 6 | 5 |

"The rest" is 她 ↔ 它 and characters outside the set.

## What the numbers say

- **The pronouns.** The end-to-end model reads the context for them. The trigram reference, which has the same context through route A, writes 他 for 她 nearly half the time; v1 and v2 do it 15 times in 53, #130 11. #130's ranking stage pushes past the balance point: it is the first system with more 他→她 than 她→他 errors.
- **What decides them.** Without context, 他 / 她 / 它 accuracy falls 11 to 15 points in every checkpoint on full pinyin, and most of the loss is 她 written as 他: all 33 extra errors for v1, 28 of 38 for #130. The context decides eval3's pronouns; the sentence's own later syllables rarely do. The #86 pattern, where `jiejie` decides `ta`, is reported by name through the named set, not by this aggregate.
- **的 / 地 / 得.** These are where v1, #127 and v2 lose the slice to the trigram reference on full pinyin: 95.8–96.0% against 97.50%. The ranking stage closes that gap entirely (#130: 97.55%), as it does with context off (96.61% against v1's 94.31%). It is the clearest per-character gain #130 shows.
- **Abbreviated and mixed input.** #130's slice top-1 is far above the reference's (z 7.70 and 4.97), but at the ambiguous characters it is level on abbreviated input and behind on mixed (z 0.25, −2.20). Its gain there is whole sentences, not these characters.
- **v2** is below #130 on the slice in all three styles (record z −2.50 / −4.26 / −2.23). That is the same gap as its whole-test top-1, and v2 has no ranking stage yet.

## The #86 records by name

The named set is `crates/ime-eval/data/pronoun-cases.jsonl`:
- the three #86 records;
- the siblings `notes/e2e-v1.md` names: 她是我妹妹, 他是我哥哥, 他是我弟弟;
- the abbreviated `tswjj`.

Every one of them is in the slice at its pronoun, which a unit test checks with the real tables. No system has a dump of them, so the table of each system's top answer and the wanted sentence's rank is still to be filled. `ime-ambiguity --by-record` prints it from the dumps.

## Implementation

- **`ime-eval`.** It owns the slice: `AmbiguitySlice::select` / `measure`, `AmbiguityReport`, and the McNemar pairing. It also owns the dump row type that `fused-eval` writes, so writing and reading share one definition.
- **`fused-eval --dump`.** It prints the slice under each section's report, measured from the rows it writes to the dump file. A test checks that this equals rescoring the written file.
- **`ime-ambiguity`** (`ime-eval`, feature `cli`). It rescores any dump file against its eval set, compares several dumps record by record, and lists records with `--by-record`.
- **`mlime eval e2e --ambiguity <ime-ambiguity>`.** It hands its dump to the same binary and appends the slice to each reported section's report. Python keeps no copy of the selection or the metrics.

## Reproduce

```
cargo build --release -p ime-eval --features cli --bin ime-ambiguity
target/release/ime-ambiguity --eval-set data/run3_pool/eval3.jsonl --slice test \
  --dump data/rescore-v2/eval3/neural-w1.000-kn-trigram-test.jsonl \
  --dump data/e2e/s5/e2e-lattice-context-on.jsonl \
  --dump data/e2e-rankw01/s0/e2e-lattice-context-on.jsonl
```

The first dump is the baseline, and each later one is compared with it. The other twins use `eval3-abbreviated.jsonl` / `eval3-mixed.jsonl` with `e2e-lattice-abbreviated-*` / `e2e-lattice-mixed-*`. The trigram reference uses `rescore-v2/eval3-abbreviated/neural-w1.500-…` and `rescore-v2/eval3-mixed/neural-w1.000-…`.

The dumps:

| system | directory |
|---|---|
| v1 | `data/e2e/s5` |
| #127 | `data/e2e-stage2/s0` |
| #130 | `data/e2e-rankw01/s0` |
| v2 | `data/e2e-v2/g4/final` |

Each directory holds `e2e-lattice{,-abbreviated,-mixed}-context-{on,off}.jsonl` over all 5,525 records. Every output behind these tables is in `data/ambiguity-slice/`.

Each takes well under a second; nothing is decoded.

## Appendix: the other checkpoints, context on

Slice top-1 / character accuracy at the positions:

| system | full | abbreviated | mixed |
|---|---|---|---|
| #116 (ranking 1.0) | 74.52% / 95.33% | 22.71% / 75.81% | 31.73% / 81.00% |
| #121 | 74.68% / 95.85% | 22.66% / 75.93% | 32.53% / 81.43% |
| #124 | 70.16% / 91.36% | 20.77% / 74.13% | 30.73% / 77.41% |
| #126 | 75.90% / 94.41% | 26.08% / 78.56% | 34.21% / 82.16% |
| #130 at 0.3 | 76.38% / 95.77% | 26.71% / 79.16% | 35.53% / 82.71% |

## Next

- **The fused decoder's dumps.** Run `fused-eval --dump` for the best configuration over the existing route A v2 s6 score files, on every twin, context on and off. The score files exist for both. Then fill the fused rows above and run #149's acceptance 3 against them.
- **The named set.** Decode the named set under the fused decoder and each end-to-end checkpoint: `emit-lattice`, then route A scores, then `fused-eval --dump`; and `mlime eval e2e --dump --ambiguity` per checkpoint. Then fill the by-name table.
- **The kernel.** Ship `ime-ambiguity` beside `ime-segment` in the e2e kernel, so every in-kernel evaluation reports the slice.
