# Word arcs and the lexicon prior (2026-10-21, #212)

## Verdict

Partial: one of the six configurations is measured. Abbreviated input without context is the one where the gap to the commercial engines is widest.

- **Arcs lift the idioms and cost nothing elsewhere.** On the abbreviated twin without context, the knowledge slice's idiom rows go from 40.38% entity exact to 60.58% (21 gained, 0 lost, z 4.58). Whole-test top-1 is 17.31% (874 / 5,050), against 17.27% for today's fused decoder. Against a live baseline in the same build, over the 2,855 test records it has finished, arcs get 485 records right where the baseline gets 482 (4 gained, 1 lost).
- **Dev cannot choose the prior.** Dev holds 15 idiom entities in 475 records. Dev top-1 is flat across the prior (92 / 91 / 91 / 91 / 90 at 0 / 4 / 8 / 12 / 16), while dev idiom exact rises from 66.67% to 80.00% at 12 and 93.33% at 16. `--select-on-dev` picks the largest top-1, so it chooses 0. The arcs alone account for the gain above. The prior's gain is visible only on the idiom rows, and dev is too small to show it. Choosing the prior on the idiom rows would change the protocol, and that needs its own issue.
- Still to measure: the other five configurations (full and mixed without context, and all three twins with context), plus a test check at prior 12 and 16 on the abbreviated twin, reported as a sensitivity run and not as a dev choice. The commands are under Reproduce, and the queue resumes where it stopped.

## The design

An idiom typed by its initials, `yffs`, has to survive four steps of character beam search before 一帆风顺 is whole. Its first character, 一 under `y`, competes with every `y` character. The trigram and the reader give the idiom's mass only once its later characters arrive, which is after the beam has already dropped it. A word arc proposes the whole idiom at the position where it starts:

- **Proposal** (`ime_decode::WordLexicon`, `crates/ime-decode/src/arcs.rs`). The lexicon is an input file, `word<TAB>category` (`data/lexicon/knowledge.tsv`, categories chosen with `--arc-category`, `idiom` by default). It is indexed by its first letters. Each reading of the typed string looks up the first letters of its spans from every start position. A match is kept only when every span spells its character: either exactly one of the character's syllables, or an abbreviation that is a strict prefix of one. Readings are matched by their keystroke boundaries, not by which lattice edge they came through, so a span that the reading took as the syllable `m` still admits 马 by its initial. When that happens, the arc pays what reading the span as an initial costs over the reading taken (`ambiguity_weight · (ln |prefix range| − ln ambiguity)`). That is the cost the lattice gives the abbreviation edge. Entries holding a character outside the emittable set are skipped (counted in the load log; #214), as are entries holding a character with no reading.
- **Scoring.** No new scorer. Each of an arc's characters is scored by route A at its position, and by the trigram and the char LM stepped through it, exactly as a free character is. Characters outside the reading's mask are added to the position's asked set (`CandidatePath::asked`), so the char LM and route A score them like any other.
- **Protection.** A lineage inside an arc is merged only with lineages inside the same arc at the same offset (`MergeKey` carries the arc's cursor). The best lineage of each arc in flight is kept beside the beam, up to `arc_width` (8) arcs a step, in order of their start, until it lands. After it lands, the arc competes with the free candidates.
- **The prior.** On landing, an arc with at least one character typed as an initial adds `arc_prior`. The test is whether the typed span is shorter than the character's syllable, which a lattice that gives `a e o m n` a single abbreviation edge (#213) does not change. With `arc_prior` at 0 and no arc in the lexicon matching, a batch decodes bit-identically to one without arcs (unit test).

**The per-step bound.** At most `beam_width + arc_width` lineages survive a step for each reading (8 + 8). A free survivor expands into its mask plus the first characters of the arcs starting there. A survivor inside an arc expands into exactly one character. The extra transition work in a step is therefore at most `arc_width` single-character asks, plus one ask per arc starting at that position. `Breakdown` and the replay report count both as `arc_starts` and `arc_slots`.

The interface changes with it:

- `Candidates::with_arcs` attaches the arcs.
- `BeamOptions` gains `arc_prior` and `arc_width`.
- `SessionOptions` gains `words`.
- `ime-cli fused-eval`, `emit-lattice` and `replay` take `--arcs <tsv> [--arc-category …] [--arc-width N]`. `fused-eval` and `replay` also take `--arc-prior`, which `fused-eval` sweeps.

## Tables

Every number below is from live route A (the fp32 export of `route-a-v2-final.pt`, step 244797), the run3 trigram and the int8 char LM at weight 1. The emission weight is today's dev choice for each configuration. Score files cannot answer characters outside the masks, so arcs need the live towers. "Today" is the score-file dump behind `notes/knowledge-slice.md`.

### Abbreviated, context off (emission weight 0.75)

Dev sweep of the prior:

| prior | dev top-1 | dev idiom exact (15) |
|---|---|---|
| 0 | 92 / 475 | 66.67% |
| 4 | 91 | 66.67% |
| 8 | 91 | 66.67% |
| 12 | 91 | 80.00% |
| 16 | 90 | 93.33% |

Test, at the dev choice (prior 0):

| system | top-1 | idiom exact (104) | slice top-1 |
|---|---|---|---|
| today (score files) | 17.27% | 40.38% | 13.28% |
| arcs, prior 0 | 17.31% (874 / 5,050) | **60.58%** (21 / 0, z 4.58) | 14.26% |
| Apple Pinyin | — | 78.85% | — |
| RIME | — | 77.88% | — |

The ambiguity slice on test with arcs: 16.67% top-1 (317 / 1,902), 74.47% character, 他/她/它 61.38%, 的/地/得 76.02%.

### Pending

| twin | context | arcs (dev sweep + test) | live baseline (test) |
|---|---|---|---|
| abbreviated | off | done | 2,855 / 5,050 |
| mixed | off | — | — |
| full | off | — | — |
| abbreviated | on | — | — |
| mixed | on | — | — |
| full | on | — | — |

## Reproduce

```
uv run --project python mlime export route-a route-a-v2-final.pt --out data/route-a-export/fp32
cargo build --release -p ime-cli
# one configuration: the prior chosen on dev, reported on test
target/release/ime-cli fused-eval --model data/run3/ngram.bin \
  --lm data/char-lm-run/transformer-s1/char-lm-int8-pages --lm-weight 1 \
  --eval-set data/run3_pool/eval3-abbreviated.jsonl \
  --emittable data/route-a-assets-v2/emittable.txt \
  --route-a data/route-a-export/fp32 --no-context --weight 0.75 \
  --arcs data/lexicon/knowledge.tsv \
  --arc-prior 0 --arc-prior 4 --arc-prior 8 --arc-prior 12 --arc-prior 16 \
  --select-on-dev --progress data/arcs-212/arcs/abbreviated-context-off/progress \
  --dump data/arcs-212/arcs/abbreviated-context-off/dump
# the baseline: the same without --arcs/--arc-prior, with --slice test
target/release/ime-knowledge --eval-set data/run3_pool/eval3-abbreviated.jsonl \
  --lexicon data/lexicon/knowledge.tsv --slice test \
  --dump <baseline test dump> --dump <arcs test dump>
```

The emission weights are full 1.5 / 0.75, abbreviated 1.5 / 0.75 and mixed 0.75 / 1.0 (context on / off). Every run resumes from `--progress`. On a loaded MacBook Pro the decode runs at about 2.5–3.5 records a second, so one configuration (5 × 475 dev + 5,050 test with arcs, 5,050 without) is about an hour. The prior is re-chosen with the same command once the lattices are re-emitted (#213).

**Latency** is measured on the M1 with the product configuration of `notes/keystroke-latency.md`. Run it with and without `--arcs`:

```
ime-cli replay --route-a <int8 export of route-a-v2-final.pt, --quantize int8> --route-a-threads 4 \
  --lm data/char-lm-run/transformer-s1/char-lm-int8-pages --model <ngram.mmap from pack-ngram> \
  --emittable data/route-a-assets-v2/emittable.txt --eval-set data/run3_pool/eval3-abbreviated.jsonl \
  --slice dev --records 40 --incremental \
  --arcs data/lexicon/knowledge.tsv --arc-prior 0 --out replay-arcs.json
```

The report gives `beam_arc_starts` and `beam_arc_slots` per keystroke, and `beam_arcs_unprotected` in the summary.

## Next

- Finish the five pending configurations and the sensitivity check at prior 12 and 16.
- A protocol issue on how to choose the prior, since dev top-1 cannot see it.
- #214: the idioms with a character outside the emittable set.
