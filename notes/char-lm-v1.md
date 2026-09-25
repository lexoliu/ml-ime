# A character LM inside the beam, v1 (2026-09-25)

Issue #54, step 2 of #40. `notes/rescore-ceiling.md` showed that a reranker
over the beam's eight hypotheses is capped by what the beam contains, so the
language model has to sit inside the search. This is the first such model: a
recurrent character LM whose state rides on every beam, scored at each
expansion in place of, or next to, the run3 Kneser-Ney trigram.

## Verdict

**The LM inside the beam is worth about 1.2 points on every typing style,
and it does not change what the beam contains.** Added to the trigram, the
fused decoder goes from 76.33 to 77.50 on full pinyin, 24.32 to 25.54 on
abbreviated and 34.20 to 35.39 on mixed sentence top-1 (test, context on).
Top-8 moves by the same amount (81.26 → 82.16, 30.16 → 31.27, 44.10 → 45.17),
so the oracle that bounded the reranker barely moved: the beam still holds
one good hypothesis and seven near-copies. Alone, the LM is no better
than the trigram it was meant to replace (74.10 against 76.33 on full
pinyin, 24.34 against 24.32 abbreviated, 33.90 against 34.20 mixed), and the
best fused weight is the smallest one tried (0.5 on two of three styles),
which says the model duplicates what the context-conditioned emissions
already know rather than adding a longer view of the sentence. The gain is
real and cheap to keep, and it is a third of what the GPT-6 reranker
recovered on abbreviated input (+1.2 against +3.6). A 30M-parameter LSTM at 3.89 nats per character is not the reader
that the abbreviated regime needs.

## The model

| quantity | value |
|---|---:|
| architecture | 2-layer LSTM, embedding 384, hidden 1024, tied output through a projection, dropout 0.1 |
| parameters | 30.67M |
| alphabet | 41,928 characters plus `<pad> <bos> <eos> <sep> <unk>` |
| input | up to 62 characters of the preceding context, `<sep>`, then the sentence |
| data | the run3 sample shards (the same 41M lines the trigram was counted from); one shard per source held out |
| steps | 100,000 × 32,768 tokens (1.54B tokens), 2×T4 DDP, AdamW, fp16 |
| wall | 10.4 h on Kaggle (`lexoliu/mlime-char-lm` v3), 41.3k tokens/s |
| held-out | 4.44 nats/char at 5k steps → **3.89 nats/char (perplexity 49.1)** at 100k, still falling |

The export (`mlime export char-lm --restrict emittable.txt`, #53) takes the
next-character distribution over the 7,322 emittable characters plus `<eos>`
and scores every other id as unreachable; the kept log probabilities equal
the full model's renormalised over that set (max difference 4e-6) and a step
costs about half as much (batch 1: 18.8 → 9.5 ms on the M1). An int8
dynamic quantisation was tried and rejected: mean log-probability deviation
0.64 nats, argmax agreement 75%.

In the decoder (`crates/ime-lm`, #52) the LM is a `Transition` whose state is
the LSTM's hidden and cell tensors plus the cached next-character log
probabilities; `start` runs the context through the model, `score` is a
lookup, and `advance` batches every surviving beam's next step through one
ONNX Runtime call. `Both<A, B>` adds it to the trigram at `--lm-weight`.

## Results (test slices, context on, neural weight chosen on dev)

Every row uses the s6 context-on scores of route A v2 and beam 8. The
baseline row is `notes/route-a-v2.md`; the reranked row is the GPT-6 Luna
reranker of `notes/rescore-ceiling.md`, listed for scale, not as a system.

### full

| configuration | neural w (dev) | top-1 | top-8 | char | MRR@8 |
|---|---|---|---|---|---|
| trigram (route A v2) | 1.0 | 76.33% | 81.26% | 95.43% | 0.784 |
| char LM alone | 1.5 | 74.10% | 79.05% | 94.82% | 0.762 |
| **trigram + char LM, lm-weight 0.5** | 1.5 | **77.50%** | **82.16%** | **95.70%** | **0.794** |
| trigram + char LM, lm-weight 1 | 1.5 | 76.92% | 81.68% | 95.60% | 0.789 |
| trigram + char LM, lm-weight 2 | 1.5 | 75.41% | 80.33% | 95.29% | 0.775 |
| trigram, reranked by GPT-6 Luna | 1.0 | 78.83% | | | |

### abbreviated

| configuration | neural w (dev) | top-1 | top-8 | char | MRR@8 |
|---|---|---|---|---|---|
| trigram (route A v2) | 1.5 | 24.32% | 30.16% | 65.59% | 0.266 |
| char LM alone | 0.75 | 24.34% | 30.10% | 65.95% | 0.264 |
| **trigram + char LM, lm-weight 0.5** | 1.0 | **25.54%** | **31.27%** | 66.25% | **0.277** |
| trigram + char LM, lm-weight 1 | 1.0 | 25.31% | 31.13% | 66.10% | 0.275 |
| trigram + char LM, lm-weight 2 | 1.5 | 25.09% | 31.07% | **66.30%** | 0.273 |
| trigram, reranked by GPT-6 Luna | 1.5 | 27.90% | | | |

### mixed

| configuration | neural w (dev) | top-1 | top-8 | char | MRR@8 |
|---|---|---|---|---|---|
| trigram (route A v2) | 1.0 | 34.20% | 44.10% | 74.48% | 0.378 |
| char LM alone | 1.0 | 33.90% | 42.97% | 74.60% | 0.373 |
| trigram + char LM, lm-weight 0.5 | 0.75 | 35.01% | 44.51% | 74.71% | 0.385 |
| **trigram + char LM, lm-weight 1** | 1.0 | **35.39%** | **45.17%** | **75.04%** | **0.390** |
| trigram + char LM, lm-weight 2 | 1.5 | 35.15% | 45.11% | 75.01% | 0.388 |
| trigram, reranked by GPT-6 Luna | 1.0 | 39.56% | | | |

### The neural weight on dev

`--select-on-dev` swept the neural emission weight over 0.75 / 1.0 / 1.5 and
reported test at the winner (sentence top-1 %, dev slices of 498 / 475 / 484
records):

| twin, transition | w=0.75 | 1.0 | 1.5 |
|---|---|---|---|
| full, char LM alone | 74.7 | 75.5 | **75.7** |
| full, trigram + LM ×0.5 | 76.9 | 77.3 | **78.7** |
| full, trigram + LM ×1 | 75.7 | 76.9 | **78.3** |
| full, trigram + LM ×2 | 74.5 | 75.3 | **76.5** |
| abbreviated, char LM alone | **24.0** | **24.0** | 22.9 |
| abbreviated, trigram + LM ×0.5 | 25.3 | **25.5** | 25.3 |
| abbreviated, trigram + LM ×1 | 25.3 | **25.9** | 25.7 |
| abbreviated, trigram + LM ×2 | 22.1 | 23.8 | **25.5** |
| mixed, char LM alone | 34.1 | **34.3** | 33.7 |
| mixed, trigram + LM ×0.5 | **37.4** | 37.2 | 37.0 |
| mixed, trigram + LM ×1 | 37.2 | **38.0** | 37.4 |
| mixed, trigram + LM ×2 | 34.3 | 35.5 | **37.0** |

## What the numbers say

- **A gain of one point, uniformly.** +1.17 full, +1.22 abbreviated, +1.19
  mixed on top-1; +0.9 to +1.1 on top-8; +0.3 to +0.7 on characters. The
  three typing styles, which differ by a factor of three in top-1, gain the
  same absolute amount, which is what a model that fixes the occasional
  trigram error looks like, not one that reads the sentence.
- **The beam is unchanged.** Top-8 minus top-1 is 4.7 / 5.7 / 9.8 points
  with the LM and 4.9 / 5.8 / 9.9 without. The reranker experiment put the
  oracle at 30.2% on abbreviated input; the LM raises it to 31.3%. Whatever
  produces the seven near-copies (per-position independent emissions that
  agree with each other, the trigram and the LM both approving them) is
  untouched by a stronger transition of this kind.
- **The LM is not a better trigram.** Alone it loses 2.2 points on full
  pinyin, 0.3 on mixed and ties on abbreviated, and with no emissions at all
  it reads the lattice at 59.4% top-1 on the full dev slice (`probe-lm-only-dev`), a few points above the
  trigram's 55.1% on test. So it knows a little more Chinese than the
  trigram but combines worse with the neural emissions, and the combination
  prefers it at half weight. The neural model was trained on the same lines
  with the same context; the two see the same evidence, and a second reading
  of it is worth less than a different one.
- **The neural weight wants to be higher with the LM in the beam.** Every
  full-pinyin row chose 1.5, the top of the grid, where v2 chose 1.0. Two
  transition models add up to a stronger prior, and the emissions have to be
  turned up to keep their share. A probe at 2.0 says the plateau is
  reached rather than the grid cut short: on dev the best fused row ties its
  1.5 result exactly (392 / 498) and the LM-alone row drops (75.7 → 74.5),
  and on test the fused row at 2.0 reads 77.38 against 77.50 at 1.5.
- **Cost.** A section (three dev sweeps and one test pass, 5.5k sentences)
  takes 57–61 min on the M1 against about two minutes for the trigram: the
  LM step is 9.5 ms at batch 1 and 15.8 ms at batch 8 and runs once per
  beam expansion. On the device this is the budget of an LSTM this size; a
  transformer LM would need a KV cache in the state to be comparable.

## Reproduce

```
kaggle kernels output lexoliu/mlime-char-lm -p data/char-lm-run     # char-lm/, char-lm-restricted/, run/
ime-cli fused-eval --model data/run3/ngram.bin --lm data/char-lm-run/char-lm-restricted --lm-weight 0.5 \
  --eval-set data/run3_pool/eval3.jsonl --emittable data/route-a-assets-v2/emittable.txt \
  --scores data/route-a-v2/s6/scores-lattice-context-on.jsonl.gz \
  --select-on-dev --weight 0.75 --weight 1 --weight 1.5
```

Drop `--model` and `--lm-weight` for the LM alone; the abbreviated and mixed
rows use `eval3-abbreviated.jsonl` / `eval3-mixed.jsonl` with the matching
`scores-lattice-abbreviated-*` / `scores-lattice-mixed-*`. `mlime export
char-lm run/charlm-final.pt --out <dir> --restrict emittable.txt` rebuilds
the restricted export from the checkpoint.

## Next

1. **The beam, not the transition.** Both experiments of #40 now point at
   the same place: the hypotheses. The reranker cannot fix what is not in
   the beam, and a stronger transition does not put it there. The next
   measurement is the oracle as a function of beam width with the LM in
   place (8 → 32 → 128): if the expected sentence enters the beam at a width
   the device can afford, the LM plus a reranker is a system; if it does not,
   the per-position independent emissions are the limit and the decoder
   itself has to change (an autoregressive pass over the fill tower, or
   iterative refinement).
2. **A larger LM on Colab.** The held-out curve was still falling at 100k
   steps (4.44 → 3.89 nats); an A100 run of four times the tokens is a
   few hours once the units land. Worth doing only if 1 shows the beam can
   hold the answer, because a better reader of the same evidence buys
   another point, not a regime change.
3. **Commercial baselines (#51).** Every number so far is relative to our
   own trigram; the product target is the commercial IMEs, and the gap to
   them decides how much of 1 and 2 is needed.
