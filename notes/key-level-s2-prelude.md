# The key-level model, joint stage with context prelude and `<eos>` (2026-11-21, #313)

## Verdict

Giving the reader decoder a causal context prelude (`<bos> c_1 … c_k <sep>`) and an explicit `<eos>` transition (#309), and training across all 769 clean `run3` shards outside the held-out and ranking sets, lifts every clean section over S2 (#257). Before any typo enrichment or sequence-level ranking, the single network is already ahead of S4 (#278) on clean full pinyin (`76.17% / 69.09%` against `76.15% / 68.57%`) and above both the fused pipeline and #132 on abbreviated (`29.21% / 17.43%`) and mixed (`39.85%` with context).

Sentence top-1 on test, beam 8, weights chosen on dev per section (S2 under the same `β ∈ {1, 2, 4, 8}` sum sweep as `notes/key-level-s3.md`; its `0.5 / 1 / 2` numbers in `notes/key-level-s2.md` were `75.91 / 68.65`, `28.46 / 17.15`, `39.04 / 28.96`):

| style | S2-prelude | S2 | fused | #132 | − S2 | − fused | − #132 |
|---|---|---|---|---|---|---|---|
| full, context on | 76.17% | 75.81% | **78.73%** | 77.44% | +0.36 | −2.56 | −1.27 |
| abbreviated, context on | **29.21%** | 28.46% | 28.42% | 28.89% | +0.75 | +0.79 | +0.32 |
| mixed, context on | **39.85%** | 39.50% | 38.23% | 38.42% | +0.35 | +1.62 | +1.43 |
| full, context off | 69.09% | 68.61% | **71.85%** | | +0.48 | −2.76 | |
| abbreviated, context off | **17.43%** | 17.45% | 17.27% | | −0.02 | +0.16 | |
| mixed, context off | 29.04% | **29.14%** | **29.14%** | | −0.10 | −0.10 | |

The fused decoder and #132 are as in `notes/e2e-v2.md` and `notes/sequence-ranking.md`. Against S2's original `0.5 / 1 / 2` grid every clean section rises (`+0.26 / +0.44` full, `+0.75 / +0.28` abbreviated, `+0.81 / +0.08` mixed).

## The run

`mlime-e2e-keys-s2-prelude` started from `mlime-e2e-keys-s1-frame3-a100` at `f70c46a` with `--keys --decoder-prelude --score-eos`. Everything is unfrozen. The 12-layer causal decoder prefills `<bos>` and up to 63 context characters into a prelude KV cache, steps from `<sep>` (`root_logits`) while attending causally over that prelude as well as cross-attending to the key states, masks the prelude alongside the encoder's context dropout (`p = 0.2`) so context-off inference matches a trained path, and scores `<eos>` (`vocab = 5,863`) when a beam covers the final key (#309).

- **Data.** One epoch over all 769 natural `run3` shards outside `HELD_OUT` (6) and the 44 shards reserved for S4 (22) and S5 (22), in the three typing styles (`0.55` full, `0.25` abbreviated, `0.20` mixed), clean. `38,461,758` examples kept of `38,640,186` seen (`166,378` dropped for an unemittable character, `6,496` for an unknown span, `5,424` not admitted, `130` unlabelled, `0` illegal gold; `1,838,752` readings arbitrated).
- **Shape.** `24,576` tokens per micro-batch per rank across two virtual ranks on one card (`[145,932, 145,929]` micro-batches), `accumulate 2`, learning rates `2e-5` (pretrained) and `1e-4` (new) on the cosine schedule; `72,965` updates.
- **Card.** Three G4 (`RTX PRO 6000 Blackwell`, 96 GB, 48-core host) legs of `43,244`, `19,558` (`43,245..62,802`), and `10,163` (`62,803..72,965`) updates. Steady-state rate `0.442 s/update` (`2.264 updates/s`, `581–600 examples/s` per rank; `0.134 s` forward, `0.307 s` backward, `0.005 s` H2D, `0.0005 s` batch wait), GPU `95.6%`, `95.4%`, and `95.4%` busy over the training windows (`19.0–19.6 GB` VRAM, `435–443 W`, `0` recompiles and `0` graph breaks after step 1). Resumed legs 2 and 3 reached their first update in `55.6 s` and `55.7 s` from process start by seeking the batch producer to the saved shard cursor (#271).
- **Held-out greedy character accuracy** at step `72,965` (`4,092` examples, `40,412` characters): `91.98%` with context, `89.67%` without (`ctc_loss` `0.5068` / `0.6743`), up from S2's `91.73%` and `89.41%`.

## Results

Test and dev at the chosen point per section (`β ∈ {1, 2, 4, 8}`, sum combination, beam 8, `0` unanswered):

| section | records | chosen `(β, γ)` | dev top-1 | test top-1 | top-8 | character | MRR@8 |
|---|---|---|---|---|---|---|---|
| full, context on | 5,027 | `(2, 1)` | 77.91% (388/498) | 76.17% | 91.47% | 95.46% | .8213 |
| full, context off | 5,027 | `(8, 1)` | 71.49% (356/498) | 69.09% | 87.11% | 93.63% | .7609 |
| abbreviated, context on | 5,050 | `(2, 1)` | 27.79% (132/475) | 29.21% | 46.93% | 71.45% | .3551 |
| abbreviated, context off | 5,050 | `(2, 1)` | 19.79% (94/475) | 17.43% | 31.47% | 60.80% | .2232 |
| mixed, context on | 5,041 | `(4, 1)` | 40.70% (197/484) | 39.85% | 59.02% | 78.47% | .4671 |
| mixed, context off | 5,041 | `(2, 1)` | 29.13% (141/484) | 29.04% | 46.68% | 71.41% | .3524 |
| eval3-typo, context on | 5,027 | `(8, 0.5)` | 4.22% (21/498) | 4.10% | 9.65% | 52.38% | .0563 |
| eval3-typo, context off | 5,027 | `(8, 0.5)` | 3.21% (16/498) | 3.08% | 7.18% | 49.87% | .0418 |

Against `#313`'s dev gate over S2 (`386 / 350` full, `129 / 95` abbreviated, `196 / 141` mixed under `β ∈ {1, 2, 4, 8}`), full pinyin rises on both context settings (`388` and `356`, `+2` and `+6` records), abbreviated and mixed rise with context (`132` and `197`), and without context stay within one record (`94` and `141`).

### What the sweep shows

Dev sentence top-1 across `β`:

| section | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| full, context on | .7731 | **.7791** | .7751 | .7771 |
| full, context off | .6988 | .7108 | .7129 | **.7149** |
| abbreviated, context on | .2674 | **.2779** | .2737 | .2758 |
| abbreviated, context off | .1895 | **.1979** | .1958 | .1895 |
| mixed, context on | .4029 | .4050 | **.4070** | **.4070** |
| mixed, context off | .2789 | **.2913** | .2831 | **.2913** |

Where S2 under the same grid chose `β = 4` or `8` on five of the six clean sections, four of the six now peak at `β = 2` (and `full, context on` at `β = 1` already matches S2's `β = 2`): with direct causal attention over the context tokens and `<eos>` closing the sentence, the transition carries sharper probabilities without needing a high multiplier.

### Slices against S2

- **Knowledge slice** (test, against S2's `β ∈ {1, 2, 4, 8}` dumps): with context, slice top-1 is `75.68%` (`−0.19`, `z = −0.24`), `25.98%` (`+0.98`, `z = +1.09`), and `36.40%` (`−1.53`, `z = −1.71`) on full, abbreviated, and mixed; abbreviated entity exact rises from `64.67%` to `66.26%` (`15` gained, `6` lost, `z = +1.96`; characters `70.13%` vs `69.10%`, `z = +2.00`). Without context, full slice top-1 rises from `68.53%` to `71.43%` (`person` `62.66%` to `67.22%`, `place` `75.00%` to `77.38%`), and abbreviated entity characters rise from `52.24%` to `53.59%` (`42` gained, `21` lost, `z = +2.65`).
- **Ambiguity slice** (test, against S2): with context, sentence top-1 rises on all three clean twins (`76.38%` vs `76.17%` full, `29.23%` vs `28.86%` abbreviated, `39.64%` vs `38.90%` mixed; `他/她/它` `83.40%` vs `82.61%` full and `78.46%` vs `76.83%` abbreviated). On mixed without context, confusable character accuracy rises from `82.46%` to `83.40%` (`49` gained, `27` lost, `z = +2.52`, driven by `的/地/得` `86.01%` vs `84.90%`).

## Cost

`96.90 CU` across the three G4 legs (`55.03`, `24.62`, and `17.25 CU`), matching #313's `92–98 CU` estimate. Stepping took `32,294.7 s` (`8.97 G4-h`); the eight sections' dev sweeps (`579.8 s`) and test decodes (`948.9 s`) took `25.5 minutes` of the third leg, with zero batch-stream re-walk waste across the two resumes (#271).

## Reproduce

```sh
uv run --project python mlime e2e stamp mlime-e2e-keys-s2-prelude --out DIR \
  --init-from mlime-e2e-keys-s1-frame3-a100 --keys \
  --decoder-prelude --score-eos \
  --base-lr 2e-5 --new-lr 1e-4 --token-budget 24576 --accumulate 2 \
  --train-shard ...   # the 769 natural shards outside HELD_OUT, S4 (22), and S5 (22)
colab/e2e.sh DIR start --session-hours 6 --gpu G4
colab/e2e.sh DIR wait && colab/e2e.sh DIR fetch && colab/e2e.sh DIR stop
```

The final checkpoint, selections, logs, and dumps are in `lexoliu/mlime-resume` under `final/mlime-e2e-keys-s2-prelude/`.

## Next

- `S3-prelude`: data enrichment from this checkpoint with typo augmentation (`--typos`), unit initials (`--unit-initials 0.5`), the lifted and mined shards, and the 17,290-row grounded/contrastive synthesis shard (`lift-swe2-00000.parquet`, #308).
- `S4/S5-prelude`: sequence-level ranking over the constrained beam with the frozen 60M character-LM teacher KL (#278, #298) across the 44 reserved natural shards, plus word arcs (#282).
