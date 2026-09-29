# Reading typed input that contains typos: the Rust inference side (2026-09-29)

Issue #104, the decoder half of #101; the training half landed in #108
(`notes/typo-training.md`), the measurement twin in #102
(`notes/typo-baseline.md`). This stage teaches the Rust pipeline the same noise
model: a span the keystrokes did not spell can still be read, and a position's
candidates are the union of homophones over everything the span could have
meant -- the same widening `CandidateSpace.resolve` applies on the training
side, so the product finally offers the candidate set the network was trained
against. Everything behind `--typos`; without it the pipeline is byte-for-byte
the clean path.

## The correction table in Rust

`CorrectionTable` (`crates/ime-pinyin/src/corrections.rs`) is the inference
twin of `python/src/mlime/typo.py`'s `CorrectionTable`, reading the same
generated `data/typo.json` (embedded with `include_str!`, like the syllable
tables). `corrections(span)` returns up to `CORRECTIONS_K = 8` `(syllable,
cost)` entries: every inventory syllable that reaches the span in at most
`CORRECTIONS_MAX_EDITS = 2` edits, cost `-log P(span | y)` summed over every
one- and two-edit path, sorted by cost with ties broken by syllable. A span
that already is a valid syllable or prefix gets its cost-0 self entry first.
The computation holds every syllable's one-edit outcome distribution once
(`outcomes`, insertion-ordered -- a `HashMap` sum permutes the adds and moves
the last ULPs) and resolves a span by intersecting it with the span's
predecessors, so each query is a small lookup, and a `Mutex`-guarded memo per
span serves the repeat asks a session makes every keystroke.

Bit-compatibility is pinned by `tests/fixtures/typo-corrections.json`: every
listed span's entry list must match the Python table's output -- syllables
exactly, costs within 1e-9 relative.

## The correction-aware lattice

`SegmentLattice::build_corrections` (a sibling constructor; `build` is
untouched) admits a second kind of segment: a typed stretch that is not a
syllable or prefix but that `corrections` reaches. Its readings are the
`Readings::Corrections` entries -- the same shape `Syllables` carries, so
`Segment::syllables()` is now `Option` (`None` when the segment spells
nothing) and `Segment::corrections()`/`corrected()` expose the entry list.
Corrected spans are capped at `MAX_SYLLABLE_LEN + CORRECTIONS_MAX_EDITS`
letters.

`k_best` ranks each path by the summed segmentation cost plus the cheapest
entry's cost of every corrected segment -- the correction cost as a prior, as
the spec prescribes. Clean paths are untouched: an all-syllable reading keeps
exactly its clean cost, so a clean input that admits corrections too can still
see a corrected reading crack the top-8 (the additive prior is the only
ranking under which the feature lives at all; a lexicographic
`(n_corrections, cost)` rule was tried and rejects every correction whenever
an abbreviation path exists). The invariant that does hold, tested on the
dev slice's clean records and the lattice unit tests: filtering corrected
paths out of the top-8 yields a subsequence of `build`'s list, in order, and
identical segmentations appear nowhere reordered.

`Segment::cost` counts corrected segments by their cheapest entry's
syllable's ambiguity like any other one-character reading, so the
`emit-lattice` record format and the progress-key accounting did not change
shape -- `Progress`'s `Key` did gain a `corrections: bool` so a resumed run
cannot mix lattice modes.

## Wider candidates

`Candidates::build_corrections` (`crates/ime-decode`) replaces a position's
homophone mask with the union over its span's corrections' prefix ranges --
row-union in the lexicon table, exactly `CandidateSpace.resolve`'s set.
Corrected segments union over their entry list; ordinary segments union over
`corrections(their own span)` too, because even a valid syllable's neighbours
are legal readings under the model. The union is memoised per call (k-best
paths repeat the same spans), which is most of the speed in the table below.
`build` takes the clean path unchanged when no noise model is loaded. An
empty union still fails `EmptyCandidateSet`; nothing falls back.

`RouteA::emission` (`crates/ime-neural`) grew a matching
`Option<&CorrectionTable>` parameter. With it, an off-inventory span embeds
through the `<unk>` row of the span table (`SpanVocab.id_or_unknown`'s
numbering -- `<unk>` is the last row, resolved at open from the manifest's
vocabulary list, `ManifestShape` when absent) and a position's admission
check ORs over every correction's mask row instead of reading only the span's
own row.

The union alone is not enough: a character the keystrokes only reach through
a correction would otherwise compete with the span's own readings at zero
cost -- the original table's clean-input regression, below. So each candidate
also carries its correction prior (`Lexicon::corrections_priors`): the
minimum `-log P(span | y)` over the corrections *y* whose homophones include
it, 0 for the span's own readings under the cost-0 self entry. The prior
rides `CandidatePath` beside `positions()` and enters the beam score as a
subtractive term scaled by `--typo-weight`, so emission, transition and
prior combine the way fused-eval already combines its other terms. The
Python side computes the same term (`CandidateSpace.priors`) for the e2e
decoder; the shared candidates fixture pins `(id, prior)` pairs rather than
ids alone -- by digest, per span -- so both languages' priors are held to the
same values.

## `span_letters` through the fill graph

The fill graph now takes `span_letters [B, W, 12]` int64 between `span_ids`
and `span_positions` -- a-z as 0-25, 26 as pad, exactly the collator's
encoding -- and `export_onnx` no longer refuses a trained letter encoder:
`(span_embeddings(span_ids) + letter_encoder(span_letters)) * span_positions`
is what the graph traces. `FILL_INPUTS` in `ime-neural` lists it, `fill_inputs`
builds it from the typed letters, and a graph without the input fails to load
per the manifest contract. The `tests/fixtures` export was regenerated -- its
first regeneration since #92, so the fixture model now carries the
letter-encoder RNG draws in construction order: the context tower's expected
values shifted ~0.01 while the context-off values match to float jitter (the
letter projection is zero-initialised, so the fill math itself is unchanged).

## The flag

`fused-eval` and `emit-lattice` take `--typos`: the `Reader` loads
`CorrectionTable` once and threads it through lattice construction, candidate
masks, and live emissions. `fused-eval` additionally takes `--typo-weight`
(repeatable, default 1.0, requires `--typos`): the prior's scale inside the
beam, swept on the dev slice through `--select-on-dev` exactly the way
`--weight` is -- a select run decodes the `weight x typo-weight` product and
reports the test slice at whichever pair wins dev top-1. `BeamOptions` gained
`typo_weight` (default 1.0, the noise model's own rate); `decode`, `eval` and
the score-file path run at that default. Without the flag everything is the
clean path. The score-file producer (`mlime train emit`) still writes
clean-path lattices only, and `Scored::attach` refuses a shape mismatch
honestly rather than pairing typo segments with clean scores.

## Evaluation

The export was regenerated for this run: `route-a-v2-final.pt` (step 244797)
through the branch's `export_onnx`, int8, so the fill graph carries
`span_letters` and the grown `<unk>` span row. That checkpoint's letter
projection is zero, so on clean input the new export must score identically to
the old one -- and it does: on the first 50 dev records of `eval3` the new and
old `fill.onnx` graphs differ by `max |Δ log_prob| = 0.0` over every asked
candidate and the whole output tensor (same inputs both sides, the new graph
additionally fed `span_letters`; `onnxruntime` on both graphs, identical int8
weights). `eval3-typo.jsonl` was rebuilt with `mlime eval typo-twin` (5525
records, 8554 edits, seed per `notes/typo-baseline.md`), so its dev/test split
is `eval3`'s.

The first measurement's clean-input regression was a defect in the inference
side, not a trade-off to wait out: the widened candidates entered the search
at zero cost. The spec's correction cost is per correction -- a character the
keystrokes reach only through a correction *y* of the typed span *s* pays that
correction's `-log P(s|y)` -- and the first implementation priced corrected
*segments* but never the widened characters of an honest span, so every free
rider competed with the span's own readings. Each position's candidate set now
carries a per-character prior (`Lexicon::corrections_priors`): the minimum
correction cost over the corrections whose homophones include the character,
charged into the beam scaled by `--typo-weight` next to the fusion weights, so
emission, transition and prior combine the way fused-eval already combines its
terms.

The prior's zero set is the span's *honest* own readings, not the self entry's
row: for a full-syllable span the union's self row covers the prefix
completions too (every `s*` syllable's homophones), and crediting that whole
row at zero leaves the prior toothless. A full-syllable span owns its exact
homophones, an incomplete tail owns its prefix completions, a corrected
segment owns nothing, and every other character covered only by the self row
pays the cheapest cost its own readings reach in the uncapped cost map -- a
character no correction reaches at all prices at infinity and is dead in the
beam whenever `--typo-weight` is on (the decoder guards `0 * inf` at weight
zero). The Python side computes the same term (`CandidateSpace.priors`, for
the e2e decoder; the restricted training loss is unchanged) and the shared
fixture pins the pairs against both languages.

`--typo-weight` is chosen on the dev slice with `--select-on-dev` over
{0, 0.25, 0.5, 1, 2} at the product's `--weight 0.5`, 498 dev records per
section, product configuration (kn-trigram + int8 char LM at `--lm-weight 1`,
route-a int8 emission). **Selection rule: the weight with the larger minimum
of the two sets' dev top-1** -- jointly maximising eval3 and eval3-typo, so a
weight cannot win by spending clean accuracy the typo set does not need --
with ties broken toward the higher clean top-1. Dev top-1 over the grid:

| `--typo-weight` | eval3 top-1 | eval3-typo top-1 | min |
| --- | --- | --- | --- |
| 0 | .7490 | .2651 | .2651 |
| 0.25 | .7550 | .2691 | .2691 |
| 0.5 | .7610 | .2751 | .2751 |
| 1 | .7651 | .2771 | .2771 |
| 2 | **.7671** | **.2771** | **.2771** |
| 4* | .7711 | .2610 | .2610 |
| flag off | .7751 | .0161 | -- |

(The `4` row is a probe past the asked grid, run because both curves were
still rising at `2`: it is where the trade-off turns -- clean keeps recovering
toward the flag-off number while the typo gain starts eroding, so the
min-optimal sits at the knee, `1` or `2`.) `1` and `2` tie the minimum
(.2771, 138/498 records each); the clean tie-break picks **`--typo-weight
2`**.

Test slice at the chosen weight (5027 records per section), with the flag-off
cells and the chosen weight's dev rows for comparison:

| eval set | `--typos` | top-1 | top-8 | char | MRR@8 | unanswered |
| --- | --- | --- | --- | --- | --- | --- |
| eval3 test | off | .7627 | .8448 | .9552 | .7974 | 0 |
| eval3 test | on, `tw=2` | .7563 | .8401 | .9526 | .7919 | 0 |
| eval3-typo test | off | .0119 | .0131 | .4256 | .0125 | 860 |
| eval3-typo test | on, `tw=2` | .2684 | .3135 | .7193 | .2867 | 0 |
| eval3 dev | on, `tw=2` | .7671 | .8394 | .9565 | .8000 | 0 |
| eval3-typo dev | on, `tw=2` | .2771 | .3153 | .7238 | .2923 | 0 |

(The `--select-on-dev` run's own per-set argmax for eval3-typo broke the
dev tie toward `1` and also measured test there -- .2668 top-1, indistinct
from `2`'s .2684; the tie really is a tie.)

**Acceptance: not met -- no weight in the grid brings clean eval3 within 0.3
of flag-off while keeping the typo gain.** At the chosen `tw=2` clean trails
by 0.64 points on test (.7563 vs .7627; dev: 0.80). What limits it, in order:

- **The same weight prices both directions of the trade-off.** Raising it
  recovers clean by charging widened characters more, but past the knee
  (`~2`) it starts suppressing real corrections too: the `tw=4` probe's
  clean dev rises to .7711 (still 0.40 short of .7751) while its typo dev
  falls to .2610 -- the minimum only degrades. The min-optimal cannot sit
  where clean parity lives.
- **The residual clean flips are priced widened characters that honestly
  out-score under the untrained checkpoint.** Comparing dev hypotheses at
  `tw=2` against flag-off, exactly 4 of 498 records flip where off was
  right -- every one a same-length substitution (`几天`→`今天`,
  `感激`→`感觉`, `高反`→`高仿`): a completion or 1-2-edit correction of the
  typed span pays its real cost and still beats the reference on emission
  plus transition. Two of them persist even at `tw=4`. The noise model
  prices these readings as plausible -- because they are; a live typist
  might have meant them. Ranking them below the intended text is the
  letter encoder's job, not the prior's.
- **One flip has no price at all:** `七七`→`琪琪` exchanges two of the
  span's *own* readings (prior 0 on both sides) -- beam-composition churn
  from the extra corrected paths competing for k-best slots, which no
  candidate prior can reach.

So the decoder-side defect -- widened candidates competing at zero cost --
is fixed end to end, and eval3-typo test keeps the whole gain at the chosen
weight (.0119 → .2684, `unanswered` 860 → 0). The remaining clean delta is
emission ranking under a checkpoint that never trained on typos: the term
that could close it is #103's `TYPOS` run, measured on these same cells
when the typo-trained model lands.

### The union parity test found real bugs

The shared fixture `crates/ime-decode/tests/fixtures/typo-candidates.json`
(189 fixture spans → Python `CandidateSpace.priors`'s `(emission id, prior)`
pairs, written by `mlime.train.lexicon.write_candidates_fixture`,
recency-pinned by `test_candidates_fixture_is_current`) holds each span's
pair count, a SHA-256 over the pairs' canonical serialisation (`id:prior`
ascending, the prior `{:.9e}`, `inf` for the unreachable) and the first 16
pairs verbatim -- the Rust test recomputes the digest and, on a mismatch,
names the span, both counts and the first differing verbatim pair. It is
asserted against the Rust path in `crates/ime-decode/tests/corrections.rs`:
`Lexicon::corrections_priors` per span and `Candidates::build_corrections` on
the lattice's k-best segmentations both reproduce the fixture exactly. It has
caught a real bug in every round: the first union implementation folded every
correction's `mask_into` into the *same* buffer (and `mask_into` clears its
output, so the "union" was silently the last correction's range); this round's
pair comparison caught the entry-merge loop pricing a kept entry through its
merged neighbour's cost, and a duplicate cost-0 self entry crowding the
eighth correction off the `K = 8` list and shrinking every union it priced.

## Latency

`ime-decode`'s `examples/keystroke-cost` replays every keystroke prefix of a
clean and a corrupted record set through lattice construction plus candidate
resolution -- the two stages the flag touches -- 100 iterations in release
mode, median per keystroke (corrections cache warm, as a session keeps it;
measured while the dev sweep saturated the box's other cores):

| path | lattice µs | candidates µs | total µs/keystroke |
| --- | --- | --- | --- |
| clean | 18.8 | 144.8 | ~164 |
| `--typos` | 24.2 | 778.5 | ~803 |

The flag costs ~640µs a keystroke, essentially all in candidate widening with
the per-character priors -- each span's prior is a dense `f64` scratch over
the lexicon (one `is_nan` write per touched character plus the uncapped-costs
top-up over each character's readings), computed once per distinct span per
batch through the same memo as the union. Before the memoisation and the dense
scratch this was ~2.7ms of candidate work.

## Reproduce

```sh
# unit level: fixture-pinned corrections, lattice, candidates, span_letters
cargo test -p ime-pinyin -p ime-decode -p ime-neural -p ime-cli

# the wider lattice and the widened candidates on a record
ime-cli emit-lattice --eval-set <eval.jsonl> --emittable <emittable.txt> \
  --out lattice.jsonl --typos

# end-to-end (the four cells of the table; same BASE with/without --typos;
# --select-on-dev decodes the weight x typo-weight product on dev, then the
# test slice at the winner)
ime-cli fused-eval --model data/run3/ngram.bin \
  --lm data/char-lm-run/transformer-s1/char-lm-int8 --lm-weight 1 \
  --eval-set data/run3_pool/eval3-typo.jsonl \
  --emittable data/route-a-assets-v2/emittable.txt \
  --route-a data/route-a-export/int8 --weight 0.5 \
  --typos --typo-weight 0 --typo-weight 0.25 --typo-weight 0.5 \
  --typo-weight 1 --typo-weight 2 --select-on-dev \
  --progress <dir> --dump <dir>

# the export itself, from the checkpoint (int8, span_letters + <unk> row)
mlime export route-a route-a-v2-final.pt --out data/route-a-export/int8 \
  --quantize int8

# the latency table above
cargo run --release -p ime-decode --example keystroke-cost -- 100
```

## What comes next

- The decoder now prices the widened candidates the way the spec prices them;
  what the untrained checkpoint cannot supply is an emission that *prefers*
  the right one. The letter encoder that learns that ranking is #103's
  `TYPOS` run, measured on the same cells when it lands.
- The score-file path (`mlime train emit` → `scores-lattice*.jsonl.gz`)
  writes clean-path lattices only; teaching it corrections (or retiring it
  for live emissions) is the last place the typo pipeline and the eval
  harness disagree.
- #105's follow-on work builds the reader that pairs with this candidate
  space; the `--typos` numbers here are the decoder-side share the trained
  letter encoder then has to build on.
