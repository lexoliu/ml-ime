# Route A base encoder selection

Decision (2026-08-26): **`hfl/chinese-macbert-base`** initializes both towers for route A.

## The constraint that drove it

The fill tower emits one distribution per typed syllable, restricted by the
per-position homophone mask over *characters*. That requires:

1. char-level tokenization (1 hanzi = 1 token) so positions align with syllables;
2. a pretrained MLM head to restrict, so we inherit pretraining instead of
   training a 41k-way head from scratch.

## Candidates considered

| model | char-aligned | MLM head | verdict |
|---|---|---|---|
| hfl/chinese-macbert-base (2020) | yes | yes; MLM-as-correction objective is *the same shape* as pinyin fill (recover the true char at a corrupted position) | **chosen** |
| hfl/chinese-roberta-wwm-ext | yes | yes | fallback, near-identical; CNMBert lineage |
| Chinese ModernBERT (arXiv 2510.12285, 2025) | **no — 32k BPE merges compounds** | — | disqualified for the fill tower; candidate for a later *context-tower* swap (context tower only feeds K/V via cross-attention, so its tokenizer is unconstrained). Weights "to be released" as of the paper. |
| Qwen3 family | decoder-only | no MLM | not an encoder; stays relevant as annotation/distillation teacher |

## Asymmetric-tower note

The two towers have independent tokenizer constraints. Milestone 3 uses one
base for both (fewer variables); swapping the context tower for a modern
long-context encoder is a post-milestone-3 ablation, not part of the gate.

## Measured: lexicon ∩ MacBERT vocab (2026-08-26)

- Lexicon (pypinyin-derived, all of Unihan): 41,923 chars.
- Present in MacBERT's 21,128-token vocab: **7,322 chars (17.5%)**.
- The missing 34k are CJK Ext-A/B rarities (㐀-class) nobody types; the number
  that matters is corpus-frequency-weighted coverage, to be measured once the
  corpus lands (expected >99.9%).
- Design consequence: route A's emittable set = lexicon ∩ base vocab. The
  homophone mask must be built against the *model's* vocab ids, not raw lexicon
  ids — mask construction takes the intersection at model-load time and the
  eval must report OOV-target sentences separately rather than silently
  scoring them wrong.
- Verified char-level tokenization: hanzi always 1 token each ([MASK]=103,
  vocab 21,128); latin words split to subwords, which only affects the context
  tower, not fill positions.
