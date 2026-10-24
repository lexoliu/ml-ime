# Word arcs and the lexicon prior (2026-10-24, #212)

## Verdict

Done. Idioms typed by their initials now come out right more often than in any commercial engine measured, and no configuration loses more than one test record.

- **Idioms.** On the knowledge slice (abbreviated twin, context off, test, 104 idioms), entity exact goes from 42.31% to 82.69% at the chosen prior, above Apple Pinyin (78.85%) and RIME (77.88%).
- **Whole test.** Against the same build without arcs, top-1 moves by −1, +8, +1, +1, +9 and +6 records across the six configurations, +24 in all. The −1 is abbreviated without context, 979 against 980 of 5,050.
- **The prior is 12**, chosen on the idiom selection set (#243) by the rule set there: a prior is admissible when every configuration's dev top-1 is within one record of its best, and the admissible prior with the highest idiom exact on the set wins. Dev top-1 alone cannot choose it: it is nearly flat across priors, and dev holds 15 idioms. `BeamOptions`, `fused-eval` and `replay` default to it.

## The design

An idiom typed by its initials, `yffs`, has to survive four steps of character beam search before 一帆风顺 is whole. Its first character, 一 under `y`, competes with every `y` character. The trigram and the reader give the idiom's mass only once its later characters arrive, which is after the beam has already dropped it. A word arc proposes the whole idiom at the position where it starts:

- **Proposal** (`ime_decode::WordLexicon`, `crates/ime-decode/src/arcs.rs`). The lexicon is an input file, `word<TAB>category` (`data/lexicon/knowledge.tsv`, categories chosen with `--arc-category`, `idiom` by default). It is indexed by its first letters. Each reading of the typed string looks up the first letters of its spans from every start position. A match is kept only when every span spells its character: either exactly one of the character's syllables, or, when abbreviations are allowed, an initial one of its syllables starts with. Since #213 the lattice reads an initial through its whole prefix range, so an arc's character always lies in its position's mask. Entries holding a character outside the emittable set are skipped (counted in the load log; #214), as are entries holding a character with no reading.
- **Scoring.** No new scorer. Each of an arc's characters is scored by route A at its position, and by the trigram and the char LM stepped through it, exactly as a free character is, and the models are asked about the masks alone, as without arcs.
- **Protection.** A lineage inside an arc is merged only with lineages inside the same arc at the same offset (`MergeKey` carries the arc's cursor). The best lineage of each arc in flight is kept beside the beam, up to `arc_width` (8) arcs a step, in order of their start, until it lands. After it lands, the arc competes with the free candidates.
- **The prior.** On landing, an arc with at least one character typed as an initial adds `arc_prior`. The test is whether the typed span is shorter than the character's syllable, which a lattice that gives `a e o m n` a single abbreviation edge (#213) does not change. With `arc_prior` at 0 and no arc in the lexicon matching, a batch decodes bit-identically to one without arcs (unit test).

**The per-step bound.** At most `beam_width + arc_width` lineages survive a step for each reading (8 + 8). A free survivor expands into its mask plus the first characters of the arcs starting there. A survivor inside an arc expands into exactly one character. The extra transition work in a step is therefore at most `arc_width` single-character asks, plus one ask per arc starting at that position. `Breakdown` and the replay report count both as `arc_starts` and `arc_slots`.

The interface changes with it:

- `Candidates::with_arcs` attaches the arcs.
- `BeamOptions` gains `arc_prior` and `arc_width`.
- `SessionOptions` gains `words`.
- `ime-cli fused-eval`, `emit-lattice` and `replay` take `--arcs <tsv> [--arc-category …] [--arc-width N]`. `fused-eval` and `replay` also take `--arc-prior`, which `fused-eval` sweeps.

## Tables

Every number below decodes #213's lattices with route A's score files for them (`fused-213`, the files the CI decode reads), the run3 trigram and the int8 char LM at weight 1, with each configuration's dev emission weight: full 1.5 / 0.75, abbreviated 1.5 / 0.75, mixed 0.75 / 1.0 (context on / off). A score file answers every arc character, since an arc's character always lies in its position's mask. All runs are on one Apple Silicon machine; the CI's x86 runners differ by a few records through platform drift, so only same-platform runs are compared.

### Choosing the prior

Dev top-1 by prior:

| configuration | 0 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| abbreviated, off | 106 | 105 | 105 | 105 | 103 |
| abbreviated, on | 155 | 155 | 155 | 155 | 154 |
| full, off | 362 | 362 | 362 | 362 | 362 |
| full, on | 395 | 395 | 395 | 395 | 395 |
| mixed, off | 158 | 158 | 160 | 159 | 159 |
| mixed, on | 214 | 214 | 215 | 215 | 214 |

0 and 4 are inadmissible on mixed without context (158 against 160), 16 on abbreviated without context (103 against 106), which leaves 8 and 12. On the selection set (306 idioms in sentences drawn to eval3's run3-frequency deciles, abbreviated, context off), idiom exact is 67.62 / 75.56 / 79.68 / 84.44 / 88.25% at 0 / 4 / 8 / 12 / 16, so the rule picks 12.

### Test at prior 12

| configuration | without arcs | arcs, prior 12 | Δ records |
|---|---|---|---|
| abbreviated, off | 19.41% (980 / 5,050) | 19.39% (979) | −1 |
| abbreviated, on | 32.08% (1,620 / 5,050) | 32.24% (1,628) | +8 |
| full, off | 71.87% (3,613 / 5,027) | 71.89% (3,614) | +1 |
| full, on | 78.77% (3,960 / 5,027) | 78.79% (3,961) | +1 |
| mixed, off | 32.28% (1,627 / 5,041) | 32.45% (1,636) | +9 |
| mixed, on | 41.84% (2,109 / 5,041) | 41.96% (2,115) | +6 |

### Idioms on the knowledge slice

Abbreviated, context off, test, 104 idioms, entity exact:

| system | idiom exact |
|---|---|
| without arcs | 42.31% |
| arcs, prior 0 | 61.54% |
| arcs, prior 12 | **82.69%** |
| arcs, prior 16 | 84.62% (whole test −8 records) |
| Apple Pinyin | 78.85% |
| RIME | 77.88% |

## Reproduce

```
cargo build --release -p ime-cli
cargo build --release -p ime-eval --features cli --bin ime-knowledge
# one configuration, mixed with context shown; for context off add --no-context and read the context-off score file
target/release/ime-cli fused-eval --model data/run3/ngram.bin \
  --lm data/char-lm-run/transformer-s1/char-lm-int8-pages --lm-weight 1 \
  --eval-set data/run3_pool/eval3-mixed.jsonl \
  --emittable data/route-a-assets-v2/emittable.txt \
  --scores data/fused-eval-inputs/fused-213/scores-lattice-mixed-context-on.jsonl.gz \
  --weight 0.75 --arcs data/lexicon/knowledge.tsv --slice test \
  --progress data/arcs-212/p12/mixed-context-on/progress --dump data/arcs-212/p12/mixed-context-on/dump
# the dev sweep: --arc-prior 0 --arc-prior 4 --arc-prior 8 --arc-prior 12 --arc-prior 16 --select-on-dev
# without arcs: the same command without --arcs
target/release/ime-knowledge --eval-set data/run3_pool/eval3-abbreviated.jsonl \
  --lexicon data/lexicon/knowledge.tsv --slice test \
  --dump <without-arcs test dump> --dump <arcs test dump>
```

`--arc-prior` defaults to 12. Every run resumes from `--progress`. With the char LM, the decode's working set grows with the thread count (#248); on a 32 GB machine already under load, run it at `RAYON_NUM_THREADS=1`. The selection set and how it is built are #243's.

## Next

- #248: the char LM's step graph gathers every layer's history at once, which bounds how many threads an evaluation can use.
- #214: idioms with a character outside the emittable set.
- Keystroke latency with arcs on the M1, through `ime-cli replay --arcs`, which reports `beam_arc_starts` and `beam_arc_slots` per keystroke.
