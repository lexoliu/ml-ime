# The ambiguity slice (2026-10-21, #150)

## Verdict

The slice exists. The fused decoder and the end-to-end evaluation both report it from their dumps, through one implementation, and every system below is measured against the fused decoder's own test dumps, context on and off.

No end-to-end checkpoint meets #149's third acceptance criterion on this slice. Its best, v2 ranked at 0.05 (#132), is level with the fused decoder on full pinyin: 96.12% against 96.03% at the ambiguous characters (McNemar z 0.30) and 77.55% against 78.35% slice top-1 (z −1.17). On abbreviated and mixed input it is significantly behind at the characters (z −2.49 and −2.43) and level on whole sentences (z 0.65 and −0.14). Context off tells the same story. The one significant end-to-end win anywhere is #130's slice top-1 on abbreviated input (28.86% against 26.66%, z 2.94), and #130 is behind at those same characters. The three #86 records are not measured: no system has a dump of the named set.

The pronouns are level. The end-to-end line writes 他 for 她 about half as often as the fused decoder: 10 of 53 for #132 against 19 with context on, 29 against 47 with context off. But it writes 她 for 他 more often (14 against 8, 22 against 7), so 他 / 她 / 它 accuracy comes out the same: 83.40% against 83.79% with context on.

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

## The fused decoder

The fused decoder is route A v2 + the run3 trigram + the int8 transformer reader (`notes/char-lm-v2.md`), with the emission weight chosen on dev from 0.75 / 1.0 / 1.5. Its dumps come from `fused-eval --dump`, context on at 246d25b and context off at 2b29cd3. A context-off run never feeds a context to the reader, so #193, which dropped context positions from the reader's prefill, cannot affect it.

| twin | context on: weight, test top-1 | context off: weight, test top-1 |
|---|---|---|
| full | 1.5, 78.73% | 0.75, 71.85% |
| abbreviated | 1.5, 28.42% | 0.75, 17.27% |
| mixed | 0.75, 38.23% | 1.0, 29.14% |

The context-on top-1 is not the 78.16% / 28.26% / 38.54% that #149 quotes from `notes/char-lm-v2.md`. That run was at weight 1.0 from 33f7371 with the older int8 export. Dev separates the weights by one or two records, so the two runs' test rows are not at the same weight (#170).

## Results

Test slices. The end-to-end rows are each checkpoint's `e2e-lattice*` dumps.

### Context on

| style | system | slice top-1 | character | 他 / 她 / 它 | 的 / 地 / 得 | swapped |
|---|---|---|---|---|---|---|
| full | fused | **78.35%** | 96.03% | **83.79%** | 97.55% | 86 |
| full | v1 | 74.15% | 94.24% | 81.03% | 95.88% | 84 |
| full | #127 | 75.74% | 94.06% | 80.24% | 95.78% | 89 |
| full | #130 | 77.29% | 95.98% | 83.40% | 97.55% | 83 |
| full | v2 | 75.74% | 94.46% | 82.21% | 95.98% | 82 |
| full | #132 | 77.55% | **96.12%** | 83.40% | **97.69%** | 80 |
| abbreviated | fused | 26.66% | **81.65%** | 76.02% | **82.32%** | 111 |
| abbreviated | v1 | 25.50% | 77.61% | **77.24%** | 77.65% | 96 |
| abbreviated | #127 | 26.29% | 78.21% | 76.83% | 78.38% | 101 |
| abbreviated | #130 | **28.86%** | 80.75% | 75.61% | 81.36% | 104 |
| abbreviated | v2 | 26.08% | 78.94% | 76.02% | 79.29% | 96 |
| abbreviated | #132 | 27.13% | 80.06% | 74.39% | 80.73% | 106 |
| mixed | fused | **36.95%** | **85.15%** | **78.41%** | **86.01%** | 113 |
| mixed | v1 | 33.58% | 81.26% | 76.52% | 81.86% | 100 |
| mixed | #127 | 34.63% | 81.94% | 74.24% | 82.92% | 107 |
| mixed | #130 | 36.58% | 83.14% | 77.27% | 83.89% | 101 |
| mixed | v2 | 35.11% | 82.84% | 75.00% | 83.84% | 105 |
| mixed | #132 | 36.85% | 83.74% | 76.89% | 84.61% | 105 |

### Context off

| style | system | slice top-1 | character | 他 / 她 / 它 | 的 / 地 / 得 | swapped |
|---|---|---|---|---|---|---|
| full | fused | **71.01%** | **94.02%** | 70.75% | **96.91%** | 130 |
| full | v1 | 65.37% | 91.40% | 67.98% | 94.31% | 124 |
| full | #127 | 68.30% | 91.66% | 69.17% | 94.46% | 124 |
| full | #130 | 69.79% | 93.50% | 68.38% | 96.61% | 132 |
| full | v2 | 68.56% | 92.36% | 68.77% | 95.29% | 122 |
| full | #132 | 70.05% | 93.89% | **71.54%** | 96.66% | 124 |
| abbreviated | fused | 16.51% | **74.34%** | 61.38% | **75.88%** | 159 |
| abbreviated | v1 | 14.51% | 68.63% | 55.69% | 70.16% | 149 |
| abbreviated | #127 | 16.04% | 70.73% | **62.20%** | 71.74% | 146 |
| abbreviated | #130 | 17.03% | 71.51% | 56.91% | 73.23% | 166 |
| abbreviated | v2 | 16.14% | 71.81% | 58.94% | 73.33% | 130 |
| abbreviated | #132 | **17.14%** | 72.45% | 56.91% | 74.29% | 158 |
| mixed | fused | **28.15%** | **80.92%** | **63.64%** | **83.12%** | 154 |
| mixed | v1 | 23.72% | 75.87% | 60.61% | 77.81% | 152 |
| mixed | #127 | 26.25% | 77.06% | 62.88% | 78.87% | 139 |
| mixed | #130 | 27.36% | 77.75% | 59.09% | 80.13% | 152 |
| mixed | v2 | 26.73% | 77.83% | 60.98% | 79.98% | 141 |
| mixed | #132 | 27.78% | 78.31% | 58.33% | 80.85% | 158 |

### Record by record, against the fused decoder

"Gained" counts what the checkpoint gets right and the fused decoder wrong. z is McNemar's statistic over the two counts; #149 asks for z ≥ 1.96. Positions within one record are not independent, so the position z overstates the evidence where a record holds several.

Context on:

| style | system | slice records: gained / lost, z | positions: gained / lost, z |
|---|---|---|---|
| full | v1 | 58 / 137, −5.66 | 16 / 57, −4.80 |
| full | #127 | 70 / 119, −3.56 | 17 / 62, −5.06 |
| full | #130 | 87 / 107, −1.44 | 23 / 24, −0.15 |
| full | v2 | 70 / 119, −3.56 | 18 / 54, −4.24 |
| full | #132 | 75 / 90, −1.17 | 23 / 21, 0.30 |
| abbreviated | v1 | 72 / 94, −1.71 | 82 / 176, −5.85 |
| abbreviated | #127 | 79 / 86, −0.54 | 88 / 168, −5.00 |
| abbreviated | #130 | 123 / 81, 2.94 | 104 / 125, −1.39 |
| abbreviated | v2 | 80 / 91, −0.84 | 100 / 163, −3.88 |
| abbreviated | #132 | 99 / 90, 0.65 | 92 / 129, −2.49 |
| mixed | v1 | 73 / 137, −4.42 | 74 / 165, −5.89 |
| mixed | #127 | 75 / 119, −3.16 | 75 / 150, −5.00 |
| mixed | #130 | 105 / 112, −0.48 | 75 / 122, −3.35 |
| mixed | v2 | 80 / 115, −2.51 | 82 / 136, −3.66 |
| mixed | #132 | 102 / 104, −0.14 | 76 / 109, −2.43 |

Context off:

| style | system | slice records: gained / lost, z | positions: gained / lost, z |
|---|---|---|---|
| full | v1 | 85 / 191, −6.38 | 30 / 90, −5.48 |
| full | #127 | 102 / 153, −3.19 | 31 / 85, −5.01 |
| full | #130 | 122 / 145, −1.41 | 35 / 47, −1.33 |
| full | v2 | 106 / 152, −2.86 | 35 / 73, −3.66 |
| full | #132 | 118 / 136, −1.13 | 44 / 47, −0.31 |
| abbreviated | v1 | 60 / 98, −3.02 | 123 / 256, −6.83 |
| abbreviated | #127 | 66 / 75, −0.76 | 127 / 211, −4.57 |
| abbreviated | #130 | 86 / 76, 0.79 | 148 / 214, −3.47 |
| abbreviated | v2 | 77 / 84, −0.55 | 159 / 218, −3.04 |
| abbreviated | #132 | 86 / 74, 0.95 | 144 / 188, −2.41 |
| mixed | v1 | 55 / 139, −6.03 | 80 / 198, −7.08 |
| mixed | #127 | 67 / 103, −2.76 | 86 / 176, −5.56 |
| mixed | #130 | 91 / 106, −1.07 | 85 / 159, −4.74 |
| mixed | v2 | 80 / 107, −1.97 | 100 / 172, −4.37 |
| mixed | #132 | 97 / 104, −0.49 | 102 / 163, −3.75 |

### Which way the pronouns go

Full pinyin, test: 178 他, 53 她, 22 它. "Trigram" is the fused decoder without the reader: route A v2 + the run3 trigram at weight 1.0, context on, from `data/rescore-v2`.

| system | 她→他 | 他→她 | 它→他 | 他→它 | the rest |
|---|---|---|---|---|---|
| trigram, context on | 24 | 5 | 11 | 2 | 1 |
| fused, context on | 19 | 8 | 11 | 2 | 1 |
| v1, context on | 15 | 9 | 10 | 5 | 9 |
| #130, context on | 11 | 12 | 10 | 5 | 4 |
| v2, context on | 15 | 9 | 10 | 4 | 7 |
| #132, context on | 10 | 14 | 10 | 4 | 4 |
| fused, context off | 47 | 7 | 19 | 0 | 1 |
| v1, context off | 48 | 4 | 15 | 6 | 8 |
| #130, context off | 39 | 18 | 12 | 6 | 5 |
| v2, context off | 43 | 10 | 14 | 7 | 5 |
| #132, context off | 29 | 22 | 13 | 2 | 6 |

"The rest" is 她 ↔ 它 and characters outside the set. The counts compare the wanted text and the top answer position by position, from `--by-record`.

## What the numbers say

- **#149's third criterion is not met.** Against the fused decoder, no checkpoint reaches z ≥ 1.96 on all three twins, by records or by positions, context on or off. #132 is the closest: level on full pinyin, level by records on abbreviated and mixed input, and behind at the characters there. The #86 records are a separate part of the criterion and stay unmeasured.
- **Abbreviated and mixed input.** Here the fused decoder keeps its lead at the ambiguous characters: 1.6 and 1.4 points over #132 with context on, 1.9 and 2.6 with context off, most of it 的 / 地 / 得. The end-to-end checkpoints win whole sentences on abbreviated input (#130, #132) but not these characters, so their gain there lies elsewhere in the sentence.
- **的 / 地 / 得 on full pinyin.** The ranking stage brought the end-to-end line up to the fused decoder: v1, #127 and v2 trail it by 1.6 to 1.8 points, #130 is level at 97.55%, #132 slightly ahead at 97.69%.
- **The pronouns.** The left-to-right transitions do cost the fused decoder the 她 that #86 found: the reader lowers the trigram's 24 她→他 to 19, and the end-to-end checkpoints reach 10 to 15. The ranking stages then push past the balance point, to more 他→她 than 她→他 errors for #130 and #132, so pronoun accuracy ends level with the fused decoder's.
- **What decides them.** Without context, every system writes 他 for most 她: 47 of 53 for the fused decoder, 48 for v1, 29 for #132. In eval3 the context decides the pronoun far more often than the sentence's later syllables do, so this aggregate mostly measures how the context is used. The `jiejie` case of #86 belongs to the named set.
- **v2 without its ranking stage** is behind the fused decoder at the characters on all three twins, context on and off (z −4.24 / −3.88 / −3.66 with context on). #132's ranking stage closes it on full pinyin and narrows it on abbreviated and mixed input.

## The #86 records by name

The named set is `crates/ime-eval/data/pronoun-cases.jsonl`:
- the three #86 records;
- the siblings `notes/e2e-v1.md` names: 她是我妹妹, 他是我哥哥, 他是我弟弟;
- the abbreviated `tswjj`.

Every one of them is in the slice at its pronoun, which a unit test checks with the real tables. No system has a dump of them, so the table of each system's top answer and the wanted sentence's rank is still to be filled. `ime-ambiguity --by-record` prints it from the dumps.

## Implementation

- **`ime-eval`.** It owns the slice: `AmbiguitySlice::select` / `measure`, `AmbiguityReport`, and the McNemar pairing. It also owns the dump row type that `fused-eval` writes, so writing and reading share one definition.
- **`fused-eval --dump`.** It prints the slice under each section's report, measured from the rows it writes to the dump file. A test checks that this equals rescoring the written file.
- **`ime-ambiguity`** (`ime-eval`, feature `cli`). It rescores any dump file against its eval set, compares several dumps record by record, and lists records with `--by-record`, one per line with tabs and newlines in a context escaped.
- **`mlime eval e2e --ambiguity <ime-ambiguity>`.** It hands its dump to the same binary and appends the slice to each reported section's report. Python keeps no copy of the selection or the metrics.

## Reproduce

```
cargo build --release -p ime-eval --features cli --bin ime-ambiguity
target/release/ime-ambiguity --eval-set data/run3_pool/eval3.jsonl --slice test \
  --dump data/ambiguity-slice/fused/full-context-on/dump/neural-w1.500-kn-trigram+char-lm-test.jsonl \
  --dump data/e2e/s5/e2e-lattice-context-on.jsonl \
  --dump data/e2e-stage2/s0/e2e-lattice-context-on.jsonl \
  --dump data/e2e-rankw01/s0/e2e-lattice-context-on.jsonl \
  --dump data/e2e-v2/g4/final/e2e-lattice-context-on.jsonl \
  --dump data/e2e-v2-rank-g4/final/e2e-lattice-context-on.jsonl
```

The first dump is the baseline, and each later one is compared with it. The other twins use `eval3-abbreviated.jsonl` / `eval3-mixed.jsonl` with `e2e-lattice-abbreviated-*` / `e2e-lattice-mixed-*`. The fused decoder's test dump for each twin and context is the one `*-test.jsonl` under `data/ambiguity-slice/fused/<twin>-context-<on|off>/dump/`, at the weight chosen on dev.

The end-to-end dumps:

| system | directory |
|---|---|
| v1 | `data/e2e/s5` |
| #127 | `data/e2e-stage2/s0` |
| #130 | `data/e2e-rankw01/s0` |
| v2 | `data/e2e-v2/g4/final` |
| #132 | `data/e2e-v2-rank-g4/final` |

Each directory holds `e2e-lattice{,-abbreviated,-mixed}-context-{on,off}.jsonl` over all 5,525 records. The outputs behind these tables, the `--by-record` listings included, are in `data/ambiguity-slice/vs-fused/`. Each takes well under a second; nothing is decoded.

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

- **The named set.** Decode the named set under the fused decoder and each end-to-end checkpoint: `emit-lattice`, then route A scores, then `fused-eval --dump`; and `mlime eval e2e --dump --ambiguity` per checkpoint. Then fill the by-name table, which is the rest of #149's third criterion. Until then the by-name table stays pending.
- **The kernel.** Ship `ime-ambiguity` beside `ime-segment` in the e2e kernel, so every in-kernel evaluation reports the slice.
