# The key-level model, joint stage S2 (2026-11-12, #257)

## Verdict

One network that reads keys is level with the fused pipeline on the abbreviated and mixed styles and three points short on full pinyin. It has no reading list, no lattice and no separate language model: a two-layer key front feeds the fill tower one position per key, a CTC head scores a character or a blank at every key, and the reader decoder, cross-attending to every key's state, is the transition. A constrained per-key beam of 8 decodes it (#216 §5, #221).

Sentence top-1 on test, weights chosen on dev:

| style | S2 | fused | #132 | S2 − fused | S2 − #132 |
|---|---|---|---|---|---|
| full, context on | 75.91% | **78.73%** | 77.44% | −2.82 | −1.53 |
| abbreviated, context on | 28.46% | 28.42% | **28.89%** | +0.04 | −0.43 |
| mixed, context on | **39.04%** | 38.23% | 38.42% | +0.81 | +0.62 |
| full, context off | 68.65% | **71.85%** | | −3.20 | |
| abbreviated, context off | 17.15% | **17.27%** | | −0.12 | |
| mixed, context off | 28.96% | **29.14%** | | −0.18 | |

The fused decoder and #132 are as in `notes/e2e-v2.md` and `notes/sequence-ranking.md`. #132 is the lineage's previous model with its ranking stage; S2 has had none, and S4 is that stage for this model. S3 (#261) comes first: typos, unit initials and the lifted and mined sentences.

## The run

`mlime-e2e-keys-s2` started from `mlime-e2e-keys-s1-frame3-a100`, the front warm-up whose gate was the frozen decoder's held-out accuracy (91.80% with context). Everything is unfrozen. The loss is the constrained CTC term plus the transition's teacher-forced cross-entropy at weight 1.0; S1's distillation and frame terms are gone.

- Data: one epoch over 460 of run3's 813 shards, strided evenly so the source mix is the corpus's, in the three typing styles, clean. 23,057,063 examples kept of 23,164,350 seen; 100,025 dropped for a character outside the emittable set (#214), 4,051 for an unknown span.
- Shape: 24,576 tokens per micro-batch per rank, a world of 2 as two virtual ranks on one card, accumulate 2; learning rates 2e-5 (pretrained) and 1e-4 (new) on the cosine schedule; 43,705 updates.
- Card: three A100 legs of 16,165, 16,910 and 10,630 updates at 6e97363. GPU 85.5%, 88.3% and 87.8% busy over the training windows, 16.3 GB peak, no recompiles after warm-up.
- Held-out greedy character accuracy at the end: 91.73% with context, 89.41% without.

A short run over S1's 22 shards (`mlime-e2e-keys-s2a`, 2,210 updates, 3% of the corpus) exercised the stage first. It reached 72.07% / 25.94% / 36.30% with context, so the other 97% of the epoch was worth 3.8, 2.5 and 2.7 points.

## Results

Test, beam 8, the constrained decode. Every section chose beta 2, gamma 0.5 and the sum on dev. No record went unanswered.

| section | records | top-1 | top-8 | character | MRR@8 |
|---|---|---|---|---|---|
| full, context on | 5,027 | 75.91% | 91.21% | 95.45% | .8181 |
| full, context off | 5,027 | 68.65% | 87.11% | 93.51% | .7569 |
| abbreviated, context on | 5,050 | 28.46% | 45.94% | 70.61% | .3472 |
| abbreviated, context off | 5,050 | 17.15% | 31.05% | 60.32% | .2188 |
| mixed, context on | 5,041 | 39.04% | 57.77% | 77.97% | .4577 |
| mixed, context off | 5,041 | 28.96% | 46.06% | 71.19% | .3502 |

Dev top-1 at the chosen point: 77.31% / 27.16% / 39.46% with context, 69.68% / 19.37% / 28.10% without.

### What the sweep shows

Dev top-1 by beta, the transition's weight (the six gamma and combination settings at each beta agree to four digits):

| section | 0.5 | 1 | 2 |
|---|---|---|---|
| full, context on | .7671 | .7691 | .7731 |
| full, context off | .6867 | .6928 | .6968 |
| abbreviated, context on | .2526 | .2632 | .2716 |
| abbreviated, context off | .1768 | .1874 | .1937 |
| mixed, context on | .3843 | .3760 | .3946 |
| mixed, context off | .2603 | .2686 | .2810 |

- The best beta is the top of the grid in every section and top-1 is still rising in five of the six, so the grid is too low. The next stage sweeps higher.
- Gamma, the correction cost's weight, and the combination of a prefix's covers do not move top-1 on clean input. The clean twins keep one gamma; it stays swept where the input carries typos.

## Cost

79.76 CU from the stage's first leg to its publish, against #216 §7's estimate of ~73 for a G4 epoch. No G4 was to be had, so the epoch ran on A100s. About 7 CU of it bought nothing: a resumed leg re-walks the batch stream to its position, 50 minutes at step 33,075, and the budget rule counted that time as training rate and paused the leg after one update (#271). The six sections' sweep and test decode took 47 minutes of the last leg.

## Reproduce

```sh
uv run --project python mlime e2e stamp mlime-e2e-keys-s2 --out DIR \
  --init-from mlime-e2e-keys-s1-frame3-a100 --keys \
  --base-lr 2e-5 --new-lr 1e-4 --token-budget 24576 --accumulate 2 \
  --train-shard ...   # the 460 strided shards
colab/e2e.sh DIR start --session-hours 4 --gpu A100
colab/e2e.sh DIR wait && colab/e2e.sh DIR fetch && colab/e2e.sh DIR stop
```

A leg that ends before the epoch does is continued by starting the next one; the run's checkpoints are in the `lexoliu/mlime-resume` bucket. The final checkpoint and the twelve evaluation logs are under `final/mlime-e2e-keys-s2/` there.

## Next

- S3 (#261): the same model trained on with typo augmentation, unit initials and the lifted and mined sentences, gated on the clean twins not falling and eval3-typo and the knowledge slice rising.
- S4: the sequence-level ranking stage over the constrained beam, where #132 gained its full-pinyin points.
- The decode in Rust and on Core ML (#224, #266, #267) and its per-keystroke latency on the M1 (#151).
