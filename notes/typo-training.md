# Reading typos: the training side (2026-09-29)

Issue #103, the training twin of #101's plan. The end-to-end network (#88)
should read typed input that contains typos and still produce the intended
sentence -- in `wo zai jai`, the preceding 我在 is what makes `jai` read as 家.
#102 built the noise model and the eval3-typo twin; this stage teaches the
network to read them. The inference side (#104) will apply the same
`corrections` function in the Rust decoder.

## `corrections(span)`

`CorrectionTable` in `python/src/mlime/typo.py` is the Python half of the
shared spec: for a typed span `s`, every syllable `y` in `syllables.txt` the
noise model can bend into `s` within at most two edits, costed
`-log P(s | y)` under the model's edit-type probabilities. `P` sums every
one- and two-edit path that produces `s`; the per-sentence coin and the
edit-count distribution are sampling policy, not path probability, so they
do not enter. A valid syllable or a valid prefix is its own correction at
cost 0 -- reading the typed letters as meant is always an option the decoder
may price. Entries sort by cost, ties by syllable, and `K = 8` survive --
the cap keeps a high-entropy typo from exploding the candidate union.

The parity contract with #104 is
`crates/ime-pinyin/tests/fixtures/typo-corrections.json`, 189 fixed spans --
valid syllables, prefixes, one- and two-edit typos, fuzzy swaps -- written by
`write_corrections_fixture`. The test fails when regenerating the file would
change it; the Rust test reads the same file and matches costs within 1e-9.

## Augmentation

`SampleBuilder` with `Augmentation(typos=True)` corrupts the typed keystrokes
per example after `type_syllables` picks the style: the noise model edits the
span list abbreviation or the mixed pass produced, never the other way round.
Corrupting first would rewrite syllables that abbreviation then replaces --
the model would see typos the typist never typed, and the target's span would
drift from what its letters say. The target stays the clean sentence; the
example's own RNG (seeded per example id, epoch and index) drives the draws,
so a corrupted example replays identically across a resume.

`Corruption.spans` -- the corrupted text re-sliced per input span -- keeps
the keystroke-to-target alignment: each target character knows the letters
typed for it. An edit landing on a span seam now joins the bound that ends
there, so `"".join(spans) == text` always holds; previously a doubling at a
boundary dropped the pressed key from every bound and a deletion at a
bound's first letter stole the previous span's tail.

A corrupted span can admit no character (two edits can produce a string no
syllable reaches). The builder counts those as `target_not_admitted` drops,
the same accounting a span that outlived its homophones already used; the
no-fallbacks rule holds -- an impossible example is removed, not guessed.

## Letter-level span input

A `LetterEncoder` -- 27 letter embeddings plus 12 position embeddings summed
per key, masked mean-pool, one `64 -> hidden` projection -- produces a vector
per span from the letters actually typed. The pool is the cheapest encoding
that separates `na` from `an`: the summed position embeddings put `n`'s
vector in a different slot for each order, so the mean differs; a bag of
letters cannot. The projection is zero-initialised and its output adds into
the existing span embedding, so at init the letter path contributes exactly
nothing -- a fresh model loaded from an #88-shaped checkpoint produces
bit-identical outputs and losses, which is what "the clean-input loss must
not change at init" means here.

Off-inventory spans -- typos -- map to a reserved `<unk>` id appended to
`typed_spans.txt` (507 rows now, `<unk>` last) plus the letter encoder. The
`<unk>` row starts at zero: a typoed span's signal is its letters, which the
encoder supplies; the table row stays a placeholder rather than a fabricated
mean. When typos are off the encoder is frozen (`requires_grad_(False)`), so
clean runs keep the old parameter set, the old step shape, and the old ONNX
export graph -- the export refuses a nonzero projection rather than ship a
graph that could not express it.

## Widened candidates

Each position's candidate set is the union of the homophones over
`corrections(span)`, applied by `CandidateSpace.resolve` in both places the
spec names: the restricted loss's `[P, E]` mask (rebuilt per batch by
`candidates_at`) and the e2e decoder's candidate path (now the batch-carried
`candidate_ids`/`candidate_counts`, replacing the decoder's fixed buffers).
A valid span keeps exactly its old candidates plus its corrections' -- the
cost-0 self entry covers the old row.

Measured on the eval3 lattice (5,525 records, 53,823 positions), the real
emittable mask:

| positions | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| before | 78.2 | 40 | 172 | 602 |
| after | 301.4 | 215 | 682 | 1047 |
| typo-sampled, before | 76.4 | 39 | 165 | 815 |
| typo-sampled, after | 302.6 | 216 | 682 | 1543 |

99.0% of positions widen (every span has neighbours in the inventory). The
typo-sampled row applies `maybe_corrupt` at the authored .35 rate to the same
records: under the clean space 1,793 of the 19,424 typo-carrying positions
resolved to the empty `<unk>` row; under the widened space 24 do -- those 24
are the drops the builder counts. A collate-and-step CPU smoke (tiny model,
4-example batch, 30 steps, median) shows the widening costs nothing at train
time: 5.7 ms before, 5.5 ms after, 5.6 ms with corrupted inputs -- the
resolve cache and the zero-projection keep the step identical in shape.

## Resume compatibility

A checkpoint written by the #88 code resumes under the new code with typos
on: `load_resumed` merges the one-row-shorter `span_embeddings` and
`candidate_mask` (the `<unk>` tail prefixes at the end, resuming at zero),
drops the retired `decoder.span_candidates` buffers, and lets the letter
parameters load as missing keys at their zero init -- anything else missing
or unexpected still raises. `parameter_groups` puts the letter encoder last
in the new group, so a saved optimiser state pads cleanly: the old group is a
strict prefix of the new one, epoch and example positions and the collator's
RNG carry over, and the resumed run's losses continue identically -- the
test trains three clean steps, prunes the checkpoint to #88 shape, and
resumes twice under the typo lane for the same losses and a `resume` metrics
record.

## The switch

`mlime train e2e --typos` enables the lane; `kaggle/e2e/kernel.py` sets
`TYPOS = False` by default, appending `--typos` only when a pushed copy turns
it on -- the running #88 lineage is untouched until we choose to switch. The
three pieces must come together: `augmentation.typos` without the noise model
or without the table-bearing `CandidateSpace` raises rather than silently
corrupting keystrokes the decoder cannot price.

## Reproduce

- Gates: `ruff check python/`, `ruff format --check python/`,
  `python -m mypy python/src/mlime`, `python -m pytest python/tests/`.
- The fixture: regenerate with `write_corrections_fixture` in
  `python/src/mlime/typo.py`; the test fails if the file drifts.
- The measurement scripts are the ones in this stage's session log; the
  lattice is `data/route-a-assets-v2/lattice.jsonl`, the mask is built from
  `char_pinyin.tsv` intersected with `emittable.txt`.
