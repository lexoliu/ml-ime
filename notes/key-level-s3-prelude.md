# The key-level model with decoder prelude, data enrichment S3-prelude (2026-11-22, #315)

## Verdict

S3-prelude teaches the decoder-prelude key-level model (`--decoder-prelude --score-eos`, #309) to read typos while beating the original S3 (#261) on all eight evaluation sections. Sentence top-1 on `eval3-typo` rises from 4.10% to 44.66% with context (+40.56; +1.00 over S3) and from 3.08% to 35.29% without (+32.21; +0.28 over S3). On the six clean sections, S3-prelude outperforms S3 by +0.16 to +1.11 points, with the largest gains on abbreviated (+1.11 / +0.72) and mixed (+0.94 / +0.89). S3-prelude seeds the combined sequence-level ranking and Char-LM teacher stage S4/S5-prelude (#320), with typos kept on.

`mlime-e2e-keys-s3-prelude` ran from `mlime-e2e-keys-s2-prelude` (`notes/key-level-s2-prelude.md`) at `a7d409f` (with the incremental KV-cache key-search evaluation of #316). S2-prelude, S3, and S3-prelude are compared under the same evaluation protocol: `β ∈ {1, 2, 4, 8}`, sum combination, weights chosen on dev per section, beam 8.

**Data.** One epoch over 26 shards, 1,355,955 examples kept of 1,363,924 seen (0 illegal gold, 0 keys too long): S3's 25 shards (22 natural shards, `lift-00000`, `lift-00001`, `lift-mined-00000`) plus `lift-swe2-00000.parquet` (17,290 grounded synthetic rows covering rare idioms, internet expressions, and entities, leak-checked against all `eval3*` sets and held-out shards, #215 / #308). Typos at 0.35: 458,249 sentences carry 704,926 edits. Unit initials at 0.5. lr `2e-5 / 2e-5`, `24,576` tokens × 2 virtual ranks, accumulate 2, `2,530` updates.

**Sentence top-1 on test**, beam 8, weights chosen on dev per section, context on / off:

| set | S3-prelude | S2-prelude | S3-p − S2-p | S3 (#261) | S3-p − S3 |
|---|---|---|---|---|---|
| full | 75.65 / 68.29 | 76.17 / 69.09 | −0.52 / −0.80 | 75.49 / 67.87 | +0.16 / +0.42 |
| abbreviated | 28.71 / 17.29 | 29.21 / 17.43 | −0.50 / −0.14 | 27.60 / 16.57 | +1.11 / +0.72 |
| mixed | 39.62 / 29.08 | 39.85 / 29.04 | −0.23 / +0.04 | 38.68 / 28.19 | +0.94 / +0.89 |
| eval3-typo | 44.66 / 35.29 | 4.10 / 3.08 | +40.56 / +32.21 | 43.66 / 35.01 | +1.00 / +0.28 |

Top-8 with context: S3-prelude `90.71 / 45.49 / 57.13 / 62.54`, S2-prelude `91.47 / 46.93 / 59.02 / 9.65`, S3 `90.31 / 44.42 / 55.19 / 61.47`. No record unanswered (`0`) in any section.

**The knowledge slice** (test, with context; slice top-1, then entity exact for idioms):

| | S3-prelude | S2-prelude | S3 | fused |
|---|---|---|---|---|
| full, all entities | 75.87% | 75.68% | 76.64% | 80.12% |
| abbreviated, all | 25.39% | 25.98% | 24.41% | 25.59% |
| mixed, all | 36.40% | 36.40% | 36.97% | 37.36% |
| abbreviated, idiom entity exact | 45.19% | 46.15% | 45.19% | 52.88% |
| mixed, idiom entity exact | 51.82% | 51.82% | 52.73% | 58.18% |

Against S2-prelude, no record-level knowledge slice separates (`|z| ≤ 0.52`). Against S3 on abbreviated input, S3-prelude improves entity exactness from 63.97% to 66.78% (`22` gained, `6` lost, `z = 3.02`) and entity character accuracy from 67.95% to 70.96% (`64` gained, `17` lost, `z = 5.22`), driven by person names (`60.56% → 65.74%`, `z = 3.36`) and places (`81.18% → 82.80%`). Idiom exactness typed as initials stays flat (45.19%), matching S3's finding that initial-typed idioms belong to word arcs (#282).

**The ambiguity slice** (test, with context; S3-prelude then S2-prelude): sentence top-1 `76.28% / 76.38%` full, `28.97% / 29.23%` abbreviated, `39.43% / 39.64%` mixed; confusable characters `95.90% / 95.90%`, `83.11% / 85.17%`, `86.95% / 87.76%`. `他/她/它` on full pinyin is `84.19%` against `83.40%` (`213` vs `211` of `253`).

**Sweep.** On `eval3-typo`, dev selects `β = 8, γ = 0.5` in both context settings. On the clean twins with context, dev selects `β = 1` on full and `β = 2` on abbreviated and mixed; without context, `β = 2` on full and `β = 8` on abbreviated and mixed.

**Run.** A G4 (RTX PRO 6000 Blackwell, 96 GB): `1,107.9 s` steady-state training (`0.445 s` per update, `608.1` examples/s/rank, `110,264.9` padded tokens/s/rank), GPU `94.5%` busy (`430.9 W`, `71°C`, `0` throttle), `18.92 GB` peak VRAM, `0` graph breaks and `0` recompiles after warmup. With #316's incremental KV-cache key-search evaluation, the 16-section dev sweep and test decode finished in `1,049.2 s` (`17.49` min: `5.74` min dev, `11.75` min test), `1.44×` faster than the `1,514.6 s` (`25.24` min) measured on the identical checkpoint before #316 (`1.68×` on dev sweep, `1.33×` on test decode).

## Reproduce

```sh
uv run --project python mlime e2e stamp mlime-e2e-keys-s3-prelude --out DIR \
  --init-from mlime-e2e-keys-s2-prelude --keys --decoder-prelude --score-eos \
  --base-lr 2e-5 --new-lr 2e-5 --token-budget 24576 --accumulate 2 \
  --typos --unit-initials 0.5 --unit-lexicon unit-entries.tsv \
  --train-shard ...   # S3's 25 shards + lift-swe2-00000.parquet
colab/e2e.sh DIR start --session-hours 4 --gpu G4
colab/e2e.sh DIR wait && colab/e2e.sh DIR fetch && colab/e2e.sh DIR stop
```

The logs, selections, and each section's dev and test dumps are in the `lexoliu/mlime-resume` bucket under `final/mlime-e2e-keys-s3-prelude/`. The slice rows are `ime-knowledge --slice test` and `ime-ambiguity --slice test` over the decompressed test dumps.

## Next

- Combined sequence-level ranking and Char-LM teacher stage S4/S5-prelude (#320) from `mlime-e2e-keys-s3-prelude` over the 44 disjoint natural shards from S4 and S5, typos kept on.
- Word arcs over keys (#282) on the finished S4/S5-prelude checkpoint for idioms typed as initials.
