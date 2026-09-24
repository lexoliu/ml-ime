# The ceiling of reranking the beam (2026-09-24)

Issue #42, step 1 of #40. Route A v2 (`notes/route-a-v2.md`) ends with a
diagnosis: neural-only top-8 sits within a few points of top-1, so the beam
has little worth ranking. This experiment measures that claim directly by
handing the beam's eight hypotheses to a far stronger reader and asking how
many sentences it can fix *without touching the search*.

## Verdict

**Reranking recovers 2.5 / 3.6 / 5.4 points; the hypotheses cap it at
4.9 / 5.9 / 9.9.** A GPT-6-class model reading the context and the typed
pinyin closes roughly half of the gap between the beam's top-1 and the oracle
on every typing style, and the oracle itself is the problem: on abbreviated
input the expected sentence is absent from all eight hypotheses 70% of the
time, on mixed input 56%. No reranker can reach the product target from
here. The search has to produce different hypotheses, which is what a
language model *inside* the beam (#40, step 2) is for.

## Setup

- Hypotheses: `fused-eval --dump` (#43) on the s6 context-on scores, fused
  with the run3 trigram at the dev-chosen weight (1.0 / 1.5 / 1.0), beam 8,
  test slices of the three eval3 twins.
- Reranker: `mlime eval rescore` sends each record to the `MLIME_LLM_*`
  endpoint (`gpt-6-luna` through CLIProxyAPI over the Codex subscription,
  `reasoning_effort=high`) with the context, the typed pinyin, and the eight
  hypotheses in beam order, and asks for the number of the one the user
  meant. The expected text is never shown. Answers are kept in
  `<slice>.picks.jsonl`, so a cut-off run resumes.
- Oracle: the share of records whose expected text is among the hypotheses.

## Results (test slices)

| typing | records | beam top-1 | reranked top-1 | oracle | unanswered |
|---|---|---|---|---|---|
| full | 5,027 | 76.33 | **78.83** | 81.26 | 0 |
| abbreviated | 5,050 | 24.32 | **27.90** | 30.16 | 0 |
| mixed | 5,041 | 34.20 | **39.56** | 44.10 | 0 |

The model keeps the beam's first hypothesis 93% of the time on full pinyin
and 74% / 72% on abbreviated / mixed, and its picks reach down the whole
list: on abbreviated input 59 records are answered with the eighth
hypothesis.

### Reasoning effort

Abbreviated dev slice (475 records, beam top-1 24.63, oracle 29.26), the
same prompt at three efforts:

| effort | reranked top-1 | wall time |
|---|---|---|
| high | 27.37 | 1,510 s |
| xhigh | 27.37 | 1,881 s |
| max | 27.58 | 2,565 s |

`max` answers one more record, inside noise, at 70% more time; the test
slices ran at `high`.

## What the numbers say

- **The beam is the bottleneck, and increasingly so the less the user
  types.** A perfect reranker would gain 4.9 points on full pinyin and 9.9 on
  mixed; the sentences a reader could fix are a minority of the errors.
- **Reranking is still worth its share.** Half the oracle gap on every
  slice is what an autoregressive reader recovers from candidates the NAR
  model plus a trigram already put in front of it. Once the search produces
  better hypotheses, the same reader (or a small distilled one) is the second
  stage.
- **The pinyin matters.** The prompt gives the model the keystrokes; on
  abbreviated input the candidates differ by whole words, and the initials
  are the only tie-breaker the context does not supply.

## Reproduce

```
# dumps (about 6 minutes for the three twins on the M1)
target/release/ime-cli fused-eval --model data/run3/ngram.bin \
  --eval-set data/run3_pool/eval3.jsonl --emittable data/route-a-assets-v2/emittable.txt \
  --select-on-dev --scores data/route-a-v2/s6/scores-lattice-context-on.jsonl.gz \
  --weight 0.5 --weight 0.75 --weight 1 --weight 1.5 --weight 2 --dump data/rescore-v2/eval3
# rerank one slice (the endpoint's burst limit tolerates about 10 in flight)
cd python && uv run mlime eval rescore \
  --dump ../data/rescore-v2/eval3/neural-w1.000-kn-trigram-test.jsonl \
  --eval-set ../data/run3_pool/eval3.jsonl --picks ../data/rescore-v2/eval3.picks.jsonl \
  --concurrency 10 --effort high --out ../data/rescore-v2/eval3.rerank.json
```

`eval3-abbreviated` / `eval3-mixed` pair with `scores-lattice-abbreviated-*`
/ `scores-lattice-mixed-*` and their own dumps. A slice takes about an hour.
