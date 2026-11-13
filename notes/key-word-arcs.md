# Word arcs in the key-level search (2026-11-13, #282)

## Verdict

Idioms typed by their initials come out right far more often and nothing else moves. On the knowledge slice (test, with context) idiom entity exact goes from 43.27% to 69.23% on the abbreviated twin and from 54.55% to 67.27% on the mixed twin, above the fused decoder's 52.88% and 58.18%. Whole-sentence top-1 over the six clean sections changes by +5, +6, +3, −3, −1 and 0 records. The lexicon is the idioms the corpus holds at least 20 times, and the prior is 4.

The model is S4's final (`notes/key-level-s4.md`); the search is the Python reference, beam 8, sum combination. The fused decoder's arcs are `notes/word-arcs.md`.

## The design

A word arc is a lexicon entry and a chain of the keystroke lattice's own arcs that emits it character by character, at least one of whose spans is typed as an initial. The lattice is the only source of chains, so an arc cannot make an illegal output, and its corrections cost what they cost any path.

- **Landing.** At the key an entry ends on, every beam prefix with a live cut at its start lands the whole entry: the characters' CTC gains along the chain, the transition's log-probabilities of its characters in turn, the correction costs and the prior. The landed prefix merges with the same prefix reached character by character.
- **The prior** π is counted once per arc of a cover's largest set of non-overlapping arcs; overlapping entries over the same keys do not stack. In the sum combination a cover with arc set A has its mass multiplied by the sum over non-overlapping subsets S of A of (e^π − 1)^|S|, so at π = 0 the decode is the decode without arcs exactly.
- **Protection is the bound.** A cut with an arc in flight is live while the arc can still land, and its optimistic bound includes the arc's gain with the prior. Nothing is kept beside the beam.
- **The bound on work.** At most 8 arcs land per record and key, the best by optimistic gain.

## The lexicon

A flat prior over the whole idiom category does harm. The category has 40,687 entries, 87% of which the corpus holds under 20 times, and on eval3-abbreviated it offers 39 arcs a record: with S2's checkpoint, abbreviated dev top-1 went 129 / 127 / 125 / 126 of 475 at priors 0 / 4 / 8 / 12, and at 12 only 5 of the 74 top answers holding an arc were right. Obscure entries land wherever their initials happen to be typed.

Cut to the entries with at least 20 occurrences (5,119) the twins no longer fall. That list beats the 2,323 entries with at least 100 on the idiom selection set at every prior: idiom exact 69.84 / 78.73 / 81.27% against 62.54 / 68.89 / 70.79% at 4 / 8 / 12.

## The prior

By the rule of `notes/idiom-selection.md`: a prior is admissible when every configuration's dev top-1 is within one record of that configuration's best, and the admissible prior with the highest idiom exact on the selection set (306 sentences, context off) wins.

| prior | abbreviated on / off (475) | mixed on / off (484) | set, idiom exact | set, sentence top-1 |
|---|---|---|---|---|
| 0 | 131 / 84 | 194 / 138 | 43.17% | 45 |
| 4 | 132 / 85 | 195 / 139 | 69.84% | 64 |
| 8 | 133 / 86 | 194 / 137 | 78.73% | 73 |
| 12 | 132 / 86 | 192 / 135 | 81.27% | 77 |
| 16 | 132 / 85 | 190 / 133 | 82.86% | 77 |

Only 4 is admissible: 0 is two records under the best on abbreviated with context, and 8 and above lose mixed without context. The fused decoder took 12 under the same rule. Why this search loses mixed records as the prior rises has not been looked into; it is what keeps the idiom rows short of the 80% a higher prior reaches on the set.

## Test

Sentence top-1 at prior 4, each section at the weights its dev sweep chose without arcs:

| section | no arcs | arcs | records |
|---|---|---|---|
| full, context on / off | 76.15 / 68.57 | 76.13 / 68.57 | −1 / 0 |
| abbreviated, on / off | 28.48 / 16.30 | 28.57 / 16.36 | +5 / +3 |
| mixed, on / off | 39.56 / 28.76 | 39.67 / 28.70 | +6 / −3 |

Full pinyin is not arc-free: a syllable typed as one letter counts as an initial.

The knowledge slice, test with context:

| | no arcs | arcs | fused |
|---|---|---|---|
| abbreviated, all entities, slice top-1 | 27.73% | 28.71% | 25.59% |
| abbreviated, idiom entity exact | 43.27% | 69.23% | 52.88% |
| mixed, all entities, slice top-1 | 37.74% | 38.89% | 37.36% |
| mixed, idiom entity exact | 54.55% | 67.27% | 58.18% |

Idiom entities: 27 gained and none lost on abbreviated, 14 and none on mixed. Sentence top-1 moves little because idioms are a small share of eval3 and a sentence holding one is usually wrong elsewhere too.

Against #216 §8 the arcs change no verdict: abbreviated and mixed with context stay above the fused decoder, the rest stay under.

## Reproduce

```sh
uv run --project python mlime lexicon arcs \
  --entries data/corpus-225/inputs/entries-215.tsv --out ARCS.tsv   # 5,119 idioms, >= 20 occurrences
uv run --project python mlime eval keys --checkpoint FINAL.pt \
  --char-table crates/ime-pinyin/data/char_pinyin.tsv --beta 2 --gamma 1 --sum \
  --eval-set data/run3_pool/eval3-abbreviated.jsonl --select-on-dev --selection-only --context \
  --arcs ARCS.tsv --arc-prior 0 --arc-prior 4 --arc-prior 8 --arc-prior 12 --arc-prior 16
uv run --project python mlime eval keys --checkpoint FINAL.pt ... \
  --eval-set data/idiom-select-243/pool-set-396/idioms-abbreviated.jsonl --slice all --no-context \
  --arcs ARCS.tsv --arc-prior 4 --dump-dir DIR
ime-knowledge --eval-set ... --lexicon data/lexicon/knowledge.tsv --slice all --dump DIR/...
```

Everything here was decoded on a CPU: about two minutes a prior on a dev slice, 20 to 37 minutes a test section.

## Next

- The Rust search carries the same arcs and is held to this one (#224).
- Why mixed input pays for a higher prior, which is what stands between 69.84% and the 80% the set shows is there.
