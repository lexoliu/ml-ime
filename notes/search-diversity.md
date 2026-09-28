# Keeping a runner-up reading alive in the beam (2026-09-28)

Issue #86. The decoder behind `notes/char-lm-v2.md` fills its candidate list
with tail variants of one reading. When the candidates at one position score
close together, the runner-up character's lineage falls out of the beam
within a position or two. For example, 她 behind 他 at the start of a sentence
is gone long before the cue that decides between them arrives. No reordering
of the finished list can bring it back.

A first version only rearranged the final list. It kept the best hypothesis
that differs from the winner at exactly one position. It changed nothing
measurable, because the missing sentence was not in the finished pool at all.
Widening the beam and the list to 32 recovered only 3 to 5 of them per slice.

## Verdict

**Guarding close positions inside the search adds 2.2 to 2.8 top-1 points
per thousand sentences, and 19 to 33 top-8 points per thousand, on every
typing style. The correct one-character-off sentence goes missing from the
list 11 to 23% less often.** Test sentence top-1, context on, trigram plus the
transformer reader, goes from 28.24 / 38.42 / 78.16 with the old search to
28.46 / 38.70 / 78.38 (abbreviated / mixed / full).

This is a change to the search, not to a model. The whole-sentence evidence
that decides 她 against 他 still comes from the left-to-right reader. Issue
#88's end-to-end network is what gives the decoder that evidence. This change
only stops the beam from discarding the alternative before the evidence
arrives.

## Mechanism

- **When a position is guarded.** It is guarded when the best candidate with
  a different character trails the winner by at most `--diversity-gap` nats of
  cumulative score. The default is 1.5.
- **What the beam reserves.** It reserves one slot for the best lineage of each
  of the top `--diversity-chars` characters there. The default is 3.
- **Reserved slots come out of the beam width, not on top of it.** When more
  guards would fit than there is room for, the earliest positions keep theirs.
  At most 8 positions per path are guarded.
- **Marks keep a guarded lineage distinct.** A candidate carries its lineage's
  character at each guarded position, and these marks join the history merge
  key. So 她x and 他x are not merged even when the transition's history cannot
  tell them apart.
- **The list keeps the single-substitution variants on top.** The finished list
  still keeps the best variant per divergent position, so whatever survives the
  beam can be shown.
- **Turning it off.** `--diversity-gap 0` disables the guard and reproduces the
  old search.

## Choosing the defaults on dev

The table uses the three eval3 twins' dev slices, with kn-trigram plus
char-lm, LM weight 1 and neural weight 1. Each cell is top-1 / top-8 / MRR /
one-char-off absent:

| | eval3 (498) | abbreviated (475) | mixed (484) |
| --- | --- | --- | --- |
| old | 396 / 413 / .809 / 43 | 125 / 154 / .287 / 35 | 196 / 243 / .443 / 29 |
| gap 0.5, 2 chars | 398 / 419 / .817 / 40 | 127 / 161 / .294 / 35 | 196 / 247 / .445 / 28 |
| gap 1.5, 2 chars | 395 / 428 / .821 / 36 | 130 / 173 / .309 / 27 | 197 / 256 / .452 / 27 |
| **gap 1.5, 3 chars** | **396 / 428 / .822 / 36** | **130 / 172 / .309 / 28** | **197 / 253 / .451 / 27** |
| gap 2.0, 3 chars | 395 / 435 / .827 / 30 | 126 / 170 / .302 / 31 | 197 / 255 / .451 / 27 |
| gap 2.5, 3 chars | 395 / 436 / .828 / 29 | 131 / 177 / .313 / 26 | 197 / 259 / .453 / 24 |
| gap 3.0, 2 chars | 395 / 438 / .830 / 29 | 133 / 184 / .323 / 23 | 197 / 261 / .455 / 25 |
| gap 3.0, 3 chars | 395 / 438 / .830 / 29 | 132 / 181 / .318 / 24 | 195 / 258 / .450 / 24 |

Gap 1.5 with three characters is the widest guard that drops top-1 on no dev
slice. Wider gaps push top-8 and the absent count further. However, a reserved
lineage then sometimes finishes first with a wrong homophone, and top-1 slips by
one or two records.

## Test slices

These use the chosen defaults against the old search. Eval3 and mixed use
neural weight 1, and abbreviated uses its dev weight of 1.5 from
`notes/char-lm-v2.md`:

| | top-1 | top-8 | char | MRR | one-char-off absent |
| --- | --- | --- | --- | --- | --- |
| eval3 (5027) old | .7816 | .8249 | .9587 | .7997 | 456 |
| eval3 new | **.7838** | **.8582** | .9592 | **.8150** | **350** |
| abbreviated (5050) old | .2824 | .3410 | .6787 | .3045 | 463 |
| abbreviated new | **.2846** | **.3679** | .6772 | **.3168** | **402** |
| mixed (5041) old | .3842 | .4745 | .7624 | .4182 | 413 |
| mixed new | **.3870** | **.4936** | .7631 | **.4267** | **366** |

The "old" rows are the search on the current decoder, and they sit below
`notes/char-lm-v2.md`: .2824 against 28.26% abbreviated and .3842 against
38.54% mixed. The note's rows came from the binary at 33f7371, which advanced
the character LM one reading at a time. Since #76 the LM advances every
reading of a position in one batch. The int8 export's dynamic quantisation
takes one activation scale per batch, so a beam's score depends on its
batchmates by 0.002 to 0.13 nats. That redistributes near-ties: on the mixed
slice, 33f7371 gives .3854, 160a3ee .3846, and a380edc and current dev .3842.
Current dev and a380edc agree record for record. Issue #93 makes the int8
export batch-independent. The comparison here holds the decoder fixed, so the
old and new rows differ only by the guard.

These records changed top-1:

| slice | changed | gained | lost | wrong both ways |
| --- | --- | --- | --- | --- |
| eval3 | 101 | 25 | 14 | 62 |
| abbreviated | 590 | 32 | 21 | 537 |
| mixed | 427 | 22 | 8 | 397 |

The only regression is abbreviated character accuracy, which falls by 72
characters of about 49k. On abbreviated and mixed input most flips move between
two wrong answers. The net gains are what the table shows.

## Reproduce

```
ime-cli fused-eval --model data/run3/ngram.bin --lm data/char-lm-run/transformer-s1/char-lm-int8 --lm-weight 1 \
  --eval-set data/run3_pool/eval3-abbreviated.jsonl --emittable data/route-a-assets-v2/emittable.txt \
  --scores data/route-a-v2/s6/scores-lattice-abbreviated-context-on.jsonl.gz \
  --slice test --weight 1.5                        # the new defaults
ime-cli fused-eval ... --slice test --weight 1.5 --diversity-gap 0   # the old search
```

The other twins use `eval3.jsonl` / `eval3-mixed.jsonl` and their own score
files at `--weight 1`. The dev sweep adds `--slice dev` and sets
`--diversity-gap` and `--diversity-chars`.

## Next

The guard keeps an alternative alive. It cannot make the decoder prefer it,
because the reader sees the sentence left to right. Issue #88 trains the
network that reads the whole sentence before choosing.
