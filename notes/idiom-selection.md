# The idiom selection set and the arc prior (2026-10-24, #243)

## Verdict

Done. The word arcs' lexicon prior (#212) is chosen on a private set of 306 abbreviated-typed sentences with idioms, by a rule fixed before the set was scored: **prior 12**. #212 reports test once at 12.

- **Why a set.** Dev cannot choose the prior: whole-sentence top-1 is nearly flat across priors and dev holds 15 idioms. Test had already been looked at. The set gives the idiom rows enough weight to choose while dev top-1 guards the rest.
- **The rule** (approved before scoring): priors {0, 4, 8, 12, 16}. A prior is admissible when every twin/context configuration's dev top-1 is within one record of that configuration's best over the grid. Among the admissible priors, the highest idiom exact on the set (abbreviated, context off) wins, ties to the smaller prior. One prior serves all six configurations.
- **Result.** Mixed without context rules out 0 and 4 (158 against 160 on dev) and abbreviated without context rules out 16 (103 against 106), leaving 8 and 12. On the set, idiom exact is 79.68% at 8 and 84.44% at 12.

| arc prior | 0 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| idiom exact on the set (315 entities) | 67.62% | 75.56% | 79.68% | **84.44%** | 88.25% |
| sentence top-1 on the set (306) | 24.84% | 27.12% | 28.10% | 28.76% | 29.41% |

## The set

Kept under `data/idiom-select-243/` and never committed, like eval3.

- **Pool.** #225's FineWeb2-HQ, MNBVC forum and YYeTs sentences: never trained on, and their idioms' run3 frequencies close to eval3's. A first attempt drew from #225's mined FineWeb-2 rows instead, whose idioms all have zero run3 occurrences; it would have favoured a higher prior than eval3 warrants and was dropped.
- **Draw.** One sentence per idiom, stratified on eval3's run3-frequency deciles, seed 243. Excluded: eval3's own idioms, #215's blocked entries, idioms outside the knowledge lexicon or with a character route A cannot emit, and sentences that cross a word boundary. Nothing collides with eval3, its pools or the other evaluation sets on record, sentence or 13-character run. The draw stops at 504: above about 42 run3 occurrences the pool holds every eligible idiom already.

  | | p10 | p25 | p50 | p75 | p90 |
  |---|---|---|---|---|---|
  | eval3's 119 idioms | 42 | 98 | 266 | 553 | 1,176 |
  | draw (504) | 46 | 100 | 269 | 490 | 1,175 |

- **Judge.** #225's three stages: word boundary, Jev at 0.7, then the review model. Round one kept 310 of 504; Jev rejects rare idioms more often, so the kept set drifted toward frequent ones. A top-up round drew 203 more for the short deciles (second sentences in the top decile, which had no unused idiom) and kept 131: 441 of 707 over both rounds.
- **Proportions.** `idiom-set --match-strata` keeps, per decile in a seeded order, the largest set in eval3's decile proportions: 401. Annotation (#244, #246) agreed in full on 363 (90.5%; character agreement 99.31%). The export keeps the agreed sentences, as eval3's does, and `idiom-trim` restores the proportions: **306 records**, 28 in decile 0, 31 in deciles 1–8, 30 in decile 9. By source, before the last trim: 196 FineWeb2-HQ, 196 forum, 9 YYeTs.

## Scoring

The set's abbreviated lattice is emitted by the #212 build with `--arcs`; the same build re-emits eval3's abbreviated lattice byte-identically to #213's. Route A (s6's final checkpoint, #213's package and token budget) scored it on CPU, context on and off, since Kaggle's GPU quota was spent; every prior decodes against the same score file, so the platform does not move the comparison. `fused-eval --slice all --weight 0.75 --no-context` with the run3 trigram, the int8 char LM at lm-weight 1 and `--arcs data/lexicon/knowledge.tsv --arc-prior {0,4,8,12,16}`, then `ime-knowledge` on the dumps.

## Reproduce

```
# against: --against each of eval3, its pools and the other evaluation sets
uv run --project python mlime eval idiom-draw --samples data/corpus-225/samples \
  --source fineweb2-hq --source mnbvc-forum --source yyets \
  --entries data/corpus-225/inputs/entries-215.tsv --knowledge data/lexicon/knowledge.tsv \
  --emittable data/route-a-assets-v2/emittable.txt --ordinary-cache data/corpus-225/.knowledge-cache \
  $AGAINST --seed 243 --out data/idiom-select-243/pool
uv run --project python mlime corpus judge --input candidates --data-dir data/idiom-select-243/pool \
  --entries data/corpus-225/inputs/entries-215.tsv --ordinary-cache data/corpus-225/.knowledge-cache \
  --review-concurrency 1
uv run --project python mlime eval idiom-topup <the draw's inputs> $AGAINST --pool data/idiom-select-243/pool
# judge again: it resumes over the top-up without re-asking round one
uv run --project python mlime eval idiom-set --judged data/idiom-select-243/pool --match-strata \
  --entries data/corpus-225/inputs/entries-215.tsv $AGAINST --out data/idiom-select-243/pool-set-396
ime-cli g2p annotate --data-dir data/idiom-select-243/pool-set-396 --concurrency 1 --sentences-per-batch 4
# export the agreed abbreviated sentences to idioms-abbreviated-agreed.jsonl, then
uv run --project python mlime eval idiom-trim --eval <agreed export> --set data/idiom-select-243/pool-set-396 \
  --draw data/idiom-select-243/pool --entries data/corpus-225/inputs/entries-215.tsv \
  --out data/idiom-select-243/pool-set-396/idioms-abbreviated.jsonl
```

Every stage resumes. The judge's review runs at concurrency 1 against the local proxy, about 12 s a sentence.

## Next

- #212 carries the test numbers at 12 and makes 12 the default.
- The set is small for anything finer than this choice: 315 idiom entities, so a gap under about 4 points between two priors is noise.
