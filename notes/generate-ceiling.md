# The ceiling of generating the sentence (2026-09-25)

Issue #56, step 3 of #40. The reranker (`notes/rescore-ceiling.md`) is capped
by what the beam holds; the character LM inside the beam
(`notes/char-lm-v1.md`) moves that by a point. Two measurements here close
the question from both sides: how much a wider beam holds, and how often a
far stronger model *writes* the sentence from the same evidence the decoder
has, the context and the keystrokes.

## Verdict

**A frontier model reading the same evidence as the decoder does not beat
it on abbreviated input; shown the decoder's hypotheses as well, it reaches
36.7%, and that is the ceiling this data has.** From the context and the
abbreviated keystrokes alone, GPT-6 Luna writes the exact sentence for 23.94%
of the test records against the fused decoder's 24.32%, and gets the length
wrong on a third of them. Given the beam's eight hypotheses as hints it
reaches 36.73%, past the reranker (27.90%) and the beam's own oracle
(30.16%), because a tenth of its right answers are sentences the beam never
proposed. On mixed input the same two numbers are 39.16% and 49.22% against
34.20%; on full pinyin 79.17% and 84.66% against 76.33%. The beam
itself is not the bottleneck either: 128 candidates hold the abbreviated
answer 33.5% of the time, 32 with the char LM 33.5%. So the abbreviated
target for any system that sees only the context and the keystrokes is about
40% of whole sentences, and the decoder's 25.5% is fifteen points short of it. The
distance is modelling, not search: the reader inside the lattice has to be
much stronger than a 30M-parameter LSTM, and the lattice has to stay,
because the strongest reader without it loses a third of the sentences to
the wrong length.

## The beam does not hold the answer

`fused-eval --top-k N --beam-width N` on the test slices, s6 context-on
scores, neural weight as chosen on dev. The top-N line is the oracle: the
share of records whose expected sentence is anywhere in the N candidates.

| twin, transition | N=8 (beam 16) | N=32 | N=128 |
|---|---|---|---|
| abbreviated, trigram | 30.16% (top-1 24.32) | 31.94% (24.44) | 33.49% (24.48) |
| abbreviated, trigram + char LM | 31.27% (25.54) | 33.47% (25.96) | |
| mixed, trigram | 44.10% (34.20) | 46.82% (34.34) | 49.32% (34.38) |
| mixed, trigram + char LM | 45.17% (35.39) | 48.48% (35.79) | |

Sixteen times the candidates raise the abbreviated oracle by 3.3 points and
the mixed one by 5.2; the char LM is worth a further 1.5 at 32. Two thirds
of abbreviated sentences are not among 128 candidates ranked by the fused
score, which means the score puts them below 128 others, not that the search
missed them. That is a modelling limit, and the width of the beam is not the
lever.

## Generating from context and keystrokes

`mlime eval generate` sends each record's context and typed keystrokes to
`gpt-6-luna` (CLIProxyAPI over the Codex subscription, reasoning effort
high) with the typing conventions explained, and asks for the sentence
alone, with one character per syllable. A second variant adds the beam's
eight hypotheses as hints. The expected text is never shown. The dumps are
the ones the reranker used (fused with the trigram at the dev weight, beam
8, test slices).

Test slices, sentence exact match unless said otherwise. "Reranked" is
`notes/rescore-ceiling.md`; "oracle (8)" is how often the expected sentence
is among the beam's eight; "either" counts a record right when the beam's
top-1 or the plain generation is.

| twin | beam top-1 | reranked (#42) | oracle (8) | generated, plain | generated, with hints | either plain or beam | length mismatches plain / hints |
|---|---|---|---|---|---|---|---|
| full | 76.33% | 78.83% | 81.26% | 79.17% | 84.66% | 87.33% | 94 / 20 |
| abbreviated | 24.32% | 27.90% | 30.16% | 23.94% | 36.73% | 35.21% | 1775 / 584 |
| mixed | 34.20% | 39.56% | 44.10% | 39.16% | 49.22% | 49.08% | 1199 / 433 |

What the generated sentences look like. "Same-length answers" have as many
characters as the expected sentence; among them the positional character
accuracy is comparable to the decoder's character metric (95.43 / 65.59 /
74.48 in `notes/route-a-v2.md`). "Right and outside the beam" is the number
of records the model wrote correctly that none of the eight hypotheses had.

| twin, variant | characters right (edit distance) | same-length answers | exact among them | positional characters among them | right and outside the beam |
|---|---|---|---|---|---|
| full, plain | 95.56% | 4933 / 5027 | 80.68% | 96.07% | 404 |
| full, hints | 97.35% | 5007 / 5027 | 85.00% | 97.47% | 337 |
| abbreviated, plain | 56.01% | 3275 / 5050 | 36.92% | 69.78% | 469 |
| abbreviated, hints | 73.37% | 4466 / 5050 | 41.54% | 76.57% | 493 |
| mixed, plain | 69.90% | 3842 / 5041 | 51.38% | 80.91% | 535 |
| mixed, hints | 80.63% | 4608 / 5041 | 53.84% | 83.52% | 538 |

## What the numbers say

- **On abbreviated input the model and the decoder tie.** 23.94% against
  24.32%, from the same context and keystrokes. The one is a frontier model
  with the conventions explained, the other a 100M-parameter fill tower with
  a trigram. What separates the two is not what they know: their right
  answers overlap only partly (the union is 35.21%), so each is guessing a
  different third of an ambiguous set. Initials alone under-determine the
  sentence, and the context does not pin it down often enough.
- **The lattice is worth a third of the sentences.** Of the plain answers,
  1,775 abbreviated and 1,199 mixed have the wrong length: the model
  mis-segments the initials or writes what the context suggests rather than
  what was typed. With the hypotheses in view, which give the length and a
  pinyin-consistent skeleton, the mismatches fall to 584 / 433 and exact
  matches rise by 12.8 / 10.1 points. A decoder gets that constraint from
  the lattice for free; a generator that leaves the lattice pays for it.
- **The hinted number is the target.** 36.73% abbreviated, 49.22% mixed,
  84.66% full: the strongest reader available, the keystrokes enforced by
  the beam's material, the context in view. The decoder is 11 / 14 / 8
  points under it. On full pinyin the reader alone adds 3 points over the
  decoder and the hints 5 more, which says the emissions are right most of
  the time and the reader fixes the ranking; on abbreviated input the
  hinted model still writes 41.5% of same-length answers exactly, so the
  reader's contribution is bounded by the data before it is bounded by the
  model.
- **The beam width is not the lever.** 128 candidates hold 33.5% of the
  abbreviated answers, 32 with the char LM the same; the hinted model with
  8 candidates is at 36.7 because it can leave them. What matters is how the
  candidates are scored and whether the reader may write past them, not how
  many survive.
- **Cost.** 5,050 abbreviated records took 96 min plain and 71 min hinted at
  24 requests in flight, with no rate limit hit; the model thinks longer
  when it has to segment the keystrokes itself. Full pinyin runs in 41 / 31
  min.

## Reproduce

```
ime-cli fused-eval --model data/run3/ngram.bin --eval-set data/run3_pool/eval3-abbreviated.jsonl \
  --emittable data/route-a-assets-v2/emittable.txt \
  --scores data/route-a-v2/s6/scores-lattice-abbreviated-context-on.jsonl.gz \
  --slice test --weight 1.5 --top-k 128 --beam-width 128
mlime eval generate --dump data/rescore-v2/eval3-abbreviated/neural-w1.500-kn-trigram-test.jsonl \
  --eval-set data/run3_pool/eval3-abbreviated.jsonl --answers abbreviated.answers.jsonl \
  --concurrency 24 --effort high --out abbreviated.json          # add --with-hypotheses for the hinted run
```

## Next

1. **Issue #51, the commercial baselines.** The ceiling is now placed
   (about 40% of abbreviated sentences, about 55% of mixed ones, from context
   and keystrokes); where Sogou, Baidu, Apple and RIME sit between the
   decoder's 25.5 / 35.8 and that ceiling decides how much of the gap has to
   be closed to lead.
2. **A stronger reader inside the lattice.** The LSTM at 3.89 nats/char is
   the weakest part of the decoder; a transformer character LM of 100M+
   parameters trained on more than one epoch of run3, on a Colab A100 when
   the units land, carried in the beam with a KV-cache state and measured by
   the same two numbers, top-1 and the oracle at 32. The hinted Luna run says
   a strong enough reader over the beam's material is worth twelve points on
   abbreviated input.
3. **A generator behind the beam, not a reranker.** Luna's right answers
   outside the beam (493 abbreviated, 535 mixed records) are the part of the
   ceiling no reranker reaches; on device that is an autoregressive pass
   that may leave the beam's hypotheses, constrained by the lattice so it
   cannot leave the keystrokes.
