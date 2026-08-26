# ml-ime

A neural Chinese pinyin input method for macOS, conditioned on the text already on
screen.

## Why

Sogou, Baidu, Microsoft Pinyin and RIME all decode with a pinyin lattice and an
n-gram language model. That architecture cannot carry long-range dependencies, has
to memorise abbreviations in a dictionary, and — the largest waste — ignores the
hundreds of characters already sitting in the document you are typing into. An
input method is the piece of software closest to a user's intent and the one that
uses the least information about it.

## Approach

```
keystrokes ──> segmentation lattice ──> per-position homophone masks
                                               │
  host text + commit history ──> context encoder (K/V cached across keystrokes)
                                               │ K,V
                                               ▼
                                   non-autoregressive fill decoder
                                               │
                                   per-position distributions ⊗ mask
                                               │
                                   Viterbi with n-gram transitions ──> candidates
```

Three properties fall out of this shape:

- **The model cannot hallucinate.** A per-position mask restricts each output to
  the characters that syllable can actually be written as, so a reading that does
  not match the keystrokes is unreachable rather than merely unlikely.
- **Latency does not grow with context.** The context tower runs when the context
  changes, not when a key is pressed; per-keystroke work is the fill decoder over
  ~20 positions plus cross-attention into a cached K/V.
- **Segmentation ambiguity is a batch, not a branch.** `xian` is 西安 or 咸; both
  lengths go into one encoder forward.

## Layout

| Path | Contents |
| --- | --- |
| `crates/ime-pinyin` | Syllable inventory, character lexicon, segmentation lattice, masks |
| `crates/ime-decode` | Emission/transition traits and the Viterbi pass |
| `crates/ime-ngram` | n-gram model: baseline comparator and transition scores |
| `crates/ime-neural` | Model inference |
| `crates/ime-eval` | Evaluation harness |
| `crates/ime-cli` | Command line driver |
| `python/` | uv-managed data pipeline and training |
| `notes/` | Parked, unverified platform notes |

## Generated data

`crates/ime-pinyin/data` is generated, not written:

```
cd python && uv run mlime gen-pinyin-tables --out-dir ../crates/ime-pinyin/data
```

CI fails if the committed copies are stale.

## Status

Milestone 0 (scaffold, CI) and the segmentation half of milestone 1 are done.
Milestone 3 is a kill gate: if top-1 sentence accuracy does not beat RIME's
octagram, and if conditioning on context does not measurably help, the project
stops there.
