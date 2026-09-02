# v2 data preparation (2026-09-02)

What changed between the v1 run (`notes/route-a-v1.md`) and the v2 training
chain, and why.

## Labels for the rest of run3

`scripts/build_v2_rest.rs` cuts the 31,247,616 rows of run3 the v1 draw did not
take (404 shards, `<source>-rest-<index>.parquet`; 8,298 held-out texts dropped,
the same set v1 dropped). They are labelled by `kaggle/labels-v2` five kernels
at a time on Kaggle; a T4 labels 120–135 rows/s, two per kernel. Rows already
labelled are mounted from `mlime-run3-rest-labels` and skipped, so a failed or
clock-limited round is resumed by re-pushing.

## Reading arbitration (issue #14)

4.0% of v1 sentences never reached training: `target_not_admitted`, the label's
reading for a character not among the readings the character table lists. Over
all 415 v1 shards, 418,313 positions fail, 130 distinct (character, reading)
pairs cover every one of them, and 和/`han` alone is 83%. The cause is a
standards split: g2pW is Taiwan-trained and emits Taiwan MOE readings (和 hàn,
垃圾 lèsè, 暂 zhàn, 圳 zùn, 姊 jiě…) against a mainland table. Tone is never the
reason (both sides are toneless), and no character is missing from the table.

Adding the Taiwan readings to the table would have taught the model Taiwan
keystrokes. Instead `python/src/mlime/train/data/reading_arbitration.tsv` maps
51 (character, Taiwan reading) pairs to the mainland reading at build time,
and two readings the table genuinely lacked are added through the generator's
`pinyin_overrides.tsv` (嗯 `en` — the table's `n`/`ng` are untypeable — and
乐 `yao`). Residual drop rate over the same shards: **0.017%** (1,654
sentences), the largest survivor being 呵/`o`.

## Eval sets for abbreviated and mixed typing (issue #13)

`ime-cli export eval-set --typing abbreviated|mixed` draws the same 5,525
sentences as eval3 (`--seed 11`) with keystrokes typed the way training
augments them: each syllable independently reduced to its initial with p=0.7,
or a full prefix with an abbreviated suffix. The draw is seeded per record
from the text, so a sentence types the same way at any set size.

| typing | keystrokes | trigram top-1 (all) | top-8 | char | lattice slots |
|---|---:|---:|---:|---:|---:|
| full | 159,750 | 55.29% | 62.62% | 88.62% | 21.4M |
| mixed | 111,333 | 16.22% | 25.29% | 57.16% | 111.0M |
| abbreviated | 91,833 | 7.11% | 10.33% | 40.28% | 143.5M |

The trigram alone collapses on abbreviations, which is the case v2 is built
for. The dev/test slices of the three sets differ by ~20 records because
`EvalRecord::digest` hashes the keystrokes (issue #16); `--slice all` is the
exact twin comparison.

## Kaggle inputs for the v2 chain

- `mlime-src` — the package at dev `d5cdd02` (resume, wall budget,
  count-batches, arbitration).
- `mlime-route-a-assets` — the patched character table, the three eval sets
  and their lattices; the finishing segment scores every `lattice*.jsonl`.
- `mlime-run3-v1-samples` / `-labels` and `mlime-run3-rest-samples` /
  `-labels`, staged side by side by `kaggle/route-a-v2`.
