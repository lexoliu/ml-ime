# The key-level model with decoder prelude, sequence-level ranking and Char-LM teacher S4/S5-prelude (2026-11-22, #320)

## Verdict

Combining sequence-level ranking and Char-LM teacher distillation in one epoch over the decoder-prelude architecture (`--decoder-prelude --score-eos`, #309) beats the previous best key-level model S5 (#298) on all eight evaluation sections. With context, test sentence top-1 reaches `77.12%` on full pinyin (`+0.75` over S5, within `0.32` of #132), `31.45%` on abbreviated (`+1.87` over S5, `+3.03` above fused), `41.88%` on mixed (`+1.87` over S5, `+3.65` above fused), and `56.61%` on `eval3-typo` (`+2.12` over S5, `+11.95` over S3-prelude). Without context, all four twins also beat S5 (`69.54% / 19.11% / 29.82% / 47.34%`), with abbreviated (`+1.84`) and mixed (`+0.68`) above the fused decoder.

`mlime-e2e-keys-s4-prelude` replaces S5 as the Phase 1 model. Against #216 §8's clean-twin gate, four of six sections now clear the fused decoder by wide margins (both context settings on abbreviated and mixed); full pinyin remains `1.61` points under fused with context (`77.12%` vs `78.73%`) and `2.31` without (`69.54%` vs `71.85%`), which is the gap Phase 2 offline large-teacher distillation (#191) targets.

`mlime-e2e-keys-s4-prelude` ran from `mlime-e2e-keys-s3-prelude` (`notes/key-level-s3-prelude.md`) across three G4 legs: leg 1 at `d46ae5a` (steps `1–294` published before hitting #322 at step `560`) and legs 2–3 at `be8af74` (after #322 fixed `SearchGraphs` eviction synchronisation and pinned prelude width to a single bucket). The evaluation is S3-prelude's: `β ∈ {1, 2, 4, 8}`, sum combination, weights chosen on dev per section, beam 8, using #316's incremental KV-cache key-search decoder.

**Data and stage.** One epoch over the 44 natural shards disjoint from S2, S3, and held-out (the union of S4's 22 shards and S5's 22 shards): `2,320,186` examples kept of `2,333,511` seen (`0` illegal gold, `0` keys too long), `797,809` sentences with typos (`1,227,825` edits), unit initials at `0.5`. Objective: S3-prelude's CTC and transition terms plus `0.05 ×` ranking cross-entropy over the gold and 8 no-grad beam negatives (`β = 2`, `γ = 1`), plus `1.0 × KL(q ‖ p)` from the frozen 60M Char-LM teacher (`charlm-transformer-s1.pt`, `teacher_scale = 0.6`). lr `2e-5 / 2e-5`, `36,864` tokens × 2 virtual ranks, accumulate 8, `4,058` updates.

**Sentence top-1 on test**, beam 8, weights chosen on dev per section, context on / off:

| set | S4/S5-prelude | S3-prelude | S4-p − S3-p | S5 (#298) | S4-p − S5 | fused | S4-p − fused | #132 (on) |
|---|---|---|---|---|---|---|---|---|
| full | **77.12 / 69.54** | 75.65 / 68.29 | +1.47 / +1.25 | 76.37 / 69.15 | +0.75 / +0.39 | 78.73 / 71.85 | −1.61 / −2.31 | 77.44 |
| abbreviated | **31.45 / 19.11** | 28.71 / 17.29 | +2.74 / +1.82 | 29.58 / 17.62 | +1.87 / +1.49 | 28.42 / 17.27 | +3.03 / +1.84 | 28.89 |
| mixed | **41.88 / 29.82** | 39.62 / 29.08 | +2.26 / +0.74 | 40.01 / 29.44 | +1.87 / +0.38 | 38.23 / 29.14 | +3.65 / +0.68 | 38.42 |
| eval3-typo | **56.61 / 47.34** | 44.66 / 35.29 | +11.95 / +12.05 | 54.49 / 47.26 | +2.12 / +0.08 | | | |

Top-8 with context: S4/S5-prelude `91.43 / 46.77 / 58.04 / 71.81`, S3-prelude `90.71 / 45.49 / 57.13 / 62.54`, S5 `91.05 / 45.35 / 57.67 / 70.30`. Character accuracy with context: `95.51% / 70.67% / 77.90% / 88.69%`. MRR@8 with context: `0.8278 / 0.3720 / 0.4811 / 0.6254`. No record unanswered (`0`) in any section.

**Dev gate** (records right on dev, S4/S5-prelude vs S3-prelude):

| set | context on | context off |
|---|---|---|
| full (498) | 389 / 387 | 356 / 352 |
| abbreviated (475) | 135 / 123 | 99 / 89 |
| mixed (484) | 208 / 189 | 153 / 145 |
| eval3-typo (498) | 288 / 235 | 258 / 193 |

All eight dev sections improve over S3-prelude, and all eight also beat S5's dev hits (`390/355` full on/off vs `389/356` — 1 record apart on full-on, ahead on all other 7 sections).

**The knowledge slice** (test, with context; slice top-1, then entity exact for idioms):

| | S4/S5-prelude | S3-prelude | S5 | fused |
|---|---|---|---|---|
| full, all entities | 77.61% | 75.87% | 76.83% | 80.12% |
| abbreviated, all | 30.08% | 25.39% | 28.12% | 25.59% |
| mixed, all | 40.23% | 36.40% | 38.89% | 37.36% |
| abbreviated, idiom entity exact | 50.96% | 45.19% | 47.12% | 52.88% |
| mixed, idiom entity exact | 57.27% | 51.82% | 55.45% | 58.18% |

Against S3-prelude, record-level slice top-1 separates cleanly on abbreviated (`34` gained, `10` lost, `z = 3.62`) and mixed (`32` gained, `12` lost, `z = 3.02`), and full-pinyin idiom characters gain `14` vs `1` lost (`z = 3.36`). Against S5, abbreviated idiom entities gain `4` and lose `0` (`47.12% → 50.96%`, `z = 2.00`), while both abbreviated (`30.08%` vs `25.59%`) and mixed (`40.23%` vs `37.36%`) sit well above the fused decoder.

**The ambiguity slice** (test, with context; S4/S5-prelude then S3-prelude): sentence top-1 `76.76% / 76.28%` full, `31.49% / 28.97%` abbreviated (`96` gained, `48` lost, `z = 4.00`), `41.22% / 39.43%` mixed (`92` gained, `58` lost, `z = 2.78`). Confusable characters: `95.90% / 95.90%` full (`他/她/它` `84.19%`, `的/地/得` `97.35%`), `82.85% / 83.11%` abbreviated, `85.84% / 86.95%` mixed.

**Sweep.** With context, dev selects `β = 1` on full and abbreviated, `β = 2` on mixed, and `β = 2, γ = 0.5` on `eval3-typo`. Without context, dev selects `β = 1` on full and abbreviated, `β = 4` on mixed, and `β = 2, γ = 0.5` on `eval3-typo`. Compared with S3-prelude (which chose `β = 8` on typo and context-off abbreviated/mixed), ranking and teacher distillation calibrate the transition scale down to `β = 1–2`.

**Run.** G4 (RTX PRO 6000 Blackwell, 96 GB), `46.92 CU` total across three legs (`2378.14 → 2331.22 CU`):
- Leg 1 (`d46ae5a`, `SEARCH_GRAPHS = 96`): ran steps `1–560`, published `step-0000294`, then hit `CUDA_ERROR_LAUNCH_FAILED` under 13× prelude-bucket graph thrashing; fixed in #322 (PR #323, `be8af74`).
- Leg 2 (`be8af74`, `SEARCH_GRAPHS = 192`): resumed from `step 294`, ran steps `295–1,788` at `3.288 s/step` (`91.8%` GPU utilisation, `323.3 W`, `60°C`, `26.3 GiB` VRAM, `0` graph breaks / `0` recompiles), paused cleanly.
- Leg 3 (`be8af74`, `SEARCH_GRAPHS = 192`): resumed from `step 1,788`, finished steps `1,789–4,058` at `3.228 s/step` (`93.0%` GPU utilisation over `14,328` samples, `317.9 W`, `54°C`, `27.1 GiB` VRAM, `0` graph breaks / `0` recompiles). The 16-section dev sweep and test decode finished in `1,034.3 s` (`17.24` min: `5.89` min dev, `11.35` min test).

## Reproduce

```sh
uv run --project python mlime e2e stamp mlime-e2e-keys-s4-prelude --out DIR \
  --init-from mlime-e2e-keys-s3-prelude --keys --decoder-prelude --score-eos \
  --key-ranking --key-rank-teacher-weight 1.0 --key-rank-teacher-scale 0.6 \
  --search-graphs 192 --base-lr 2e-5 --new-lr 2e-5 \
  --token-budget 36864 --accumulate 8 \
  --typos --unit-initials 0.5 --unit-lexicon unit-entries.tsv \
  --train-shard ...   # 44 natural shards disjoint from S2, S3, and held-out
colab/e2e.sh DIR start --session-hours 7 --gpu G4
colab/e2e.sh DIR wait && colab/e2e.sh DIR fetch && colab/e2e.sh DIR stop
```

The logs, selections, and each section's dev and test dumps are in the `lexoliu/mlime-resume` bucket under `final/mlime-e2e-keys-s4-prelude/`.

## Word arcs (#282) and M1 latency (#151)

**Word arcs (`prior = 4`, 5,119 idioms `≥ 20` occurrences, L4 `1.44 CU`).** Adding the #282 word-arc lexicon at `prior = 4` produces zero regressions across all six clean test sections while lifting initial-typed idiom entity exactness by `+27.89` points on abbreviated and `+14.55` on mixed:

| section | no arcs | arcs (π=4) | records | vs fused |
|---|---|---|---|---|
| full, context on / off | 77.12 / 69.54 | **77.12 / 69.60** | 0 / +3 | −1.61 / −2.25 |
| abbreviated, on / off | 31.45 / 19.11 | **31.54 / 19.21** | +5 / +5 | **+3.12 / +1.94** |
| mixed, on / off | 41.88 / 29.82 | **42.00 / 29.84** | +6 / +1 | **+3.77 / +0.70** |

Knowledge slice (test, with context):

| | no arcs | arcs (π=4) | S4 + arcs | fused |
|---|---|---|---|---|
| abbreviated, all entities, slice top-1 | 30.08% | **31.64%** (`z = 2.31`) | 28.71% | 25.59% |
| abbreviated, idiom entity exact | 50.96% | **78.85%** (`30` gained, `1` lost, `z = 5.21`) | 69.23% | 52.88% |
| mixed, all entities, slice top-1 | 40.23% | **41.95%** (`9` gained, `0` lost, `z = 3.00`) | 38.89% | 37.36% |
| mixed, idiom entity exact | 57.27% | **71.82%** (`17` gained, `1` lost, `z = 3.77`) | 67.27% | 58.18% |

On the idiom selection set (`306` sentences, context off), idiom entity exact across `π ∈ {0, 4, 8, 12, 16}` is `44.44% / 70.79% / 77.46% / 81.59% / 82.54%`.

**M1 Core ML latency (`d961379`, `step 4,058`, Mac mini M1 8 GB).** Exported via `mlime export keys` and `mlime-coreml convert`: all 10 Core ML functions (`ctx_64`, `prelude_64`, `keys_{16,32,64,128}`, `tree_{32,64,128,256}`) verified with `0` failing checks on the Neural Engine. Replaying `eval3` dev (`498` records, `14,625` keystrokes, `β = 1, γ = 1, sum`, `0` records too long):

| engine | median | p95 | max | phys footprint | peak RSS |
|---|---|---|---|---|---|
| S4/S5-prelude (Core ML ANE) | **28.71 ms** | **61.54 ms** | 152.01 ms | **146 MB** | 376 MB |
| #216 §8 target | ≤ 109 ms | ≤ 236 ms | — | — | — |
| fused (M1 product) | 130.6 ms | 270.8 ms | 3,450 ms | 1.32 GB | 2.89 GB |

By typed length (median / p95): `1–4` `16.2 / 29.6 ms`, `5–8` `20.6 / 27.2 ms`, `9–16` `26.1 / 35.0 ms`, `17–24` `32.8 / 45.2 ms`, `25–32` `40.6 / 56.0 ms`, `33–48` `48.2 / 71.8 ms`, `49–64` `62.6 / 95.0 ms`, `65+` `75.7 / 121.6 ms`. Warmup `0.153 s`, speculated row share `79.2%`.

## Next

- Phase 1 is complete: `S2-prelude` (#313) → `S3-prelude` (#315) → `S4/S5-prelude` (#320) beats S5 on all 8 sections and clears fused on abbreviated and mixed in both context settings.
- Closing the remaining `1.61 / 2.31` full-pinyin gap to fused moves to Phase 2 (#191): large-teacher offline distillation (`K2` + T1/T2/T3) and word arcs over keys (#282) for idioms typed as initials.
