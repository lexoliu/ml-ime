# Milestone 3 — training plan (drafted 2026-08-26)

Two routes, one eval set (data/run1_pool/eval2.jsonl lineage), decided by
numbers. Baseline to beat: top-1 54.81% / char 88.84% / MRR@5 0.577.

## Sample construction (shared by both routes)

One training example: (typed, target, context?) where
- target = a prepared corpus sentence (all-Han after target cleaning);
- typed = pinyin keystrokes derived from the target's per-char readings;
- context = the sample's stored context, dropped with p=0.3 at batch time
  (model must work with and without it).

### Labels (per-char readings) at scale

- v1 labels: g2pW alone, batched on Kaggle T4 (ONNX CUDA EP) — measured
  98.97% char agreement with the dual annotation, and its dominant error
  (和→han4) is a *reading* error that mostly still maps to a valid typed
  key sequence; label noise at this level is acceptable for training.
  Sogou lexicon overrides where a covered word disagrees.
- Eval labels stay dual-annotated (g2pW × luna, agreement-filtered) — never
  train-labelled.

### Typing-style augmentation (per example, sampled at build time)

- full pinyin (p=0.55): every syllable typed out;
- abbreviation (p=0.25): each syllable independently reduced to its initial
  with p=0.7 (multi-letter initials zh/ch/sh kept whole);
- mixed (p=0.2): first syllables full, a random suffix abbreviated (models
  the real "type full then get lazy" pattern).
The typed string never carries tone. Augmentation happens in the sample
builder so one target yields one typed form per epoch pass (re-sampled per
epoch on the fly, not materialised).

## Route A — MacBERT-initialised two towers

- Fill tower: hfl/chinese-macbert-base, input = one slot per typed *syllable
  span*: [CLS] s1 s2 ... sn [SEP]. Each si = the base [MASK] embedding plus an
  additive **typed-span embedding** from a new learned table. The typed spans
  form a closed set (every prefix of every valid syllable plus the initials,
  ~1.3k forms — enumerable from ime-pinyin's SyllableTable), so a dedicated
  table is exact; a mean-of-letter-embeddings would collide anagrams (na/an).
  Initialise each entry from the mean of its letters' base embeddings (order
  ambiguity at init only; training separates them).
  Output head = base MLM head restricted to lexicon ∩ vocab (7,322 chars);
  per-position homophone mask applied at loss and at decode.
- Context tower: same base, encoding the context string; K/V exposed via
  cross-attention added to the fill tower's top 4 layers (new, zero-init
  gates so route A starts as pure fill and learns to use context).
- Loss: per-position CE over the masked candidate set (softmax restricted to
  homophones), label smoothing 0.05.
- Optimiser: AdamW, lr 3e-5 base / 1e-4 for new parameters, cosine, warmup 4%.
  Batch: pack to ~8k tokens per step per GPU; fp16; T4 x2 via DDP.
- v1 data: run1 (5M) + run2 smoke-scale extension; one epoch first, measure,
  then scale corpus before scaling epochs.

## Route B — from scratch

Same two-tower shape, but char-level vocab built from corpus frequency
(coverage ≥99.99% of corpus chars) + our syllable-slot inputs; 12 layers,
d=512, 8 heads (~45M params) as the first point — deliberately smaller than
route A to test the "task doesn't need pretraining" hypothesis honestly.
Tokeniser-free (chars + syllables are the units already).

## Decode & eval integration

Both routes export per-position log-probs over candidate ids for each of the
k-best segmentations → ime-decode's Emission trait (path, position, candidate).
Fusion with the KN trigram transition (weight λ tuned on a dev split of the
eval pool, NOT the test half) → same beam Viterbi → same report as the
baseline. Report with-context vs without-context on the same records.

## Infrastructure

- Kaggle: kernel per experiment, `machine_shape: NvidiaTeslaT4` (2x), datasets
  uploaded as Kaggle datasets (corpus parquet + eval jsonl + lexicon tables).
- Checkpoints + metrics land in kernel output; a small mlime-train python
  package lives under python/ (the sanctioned torch exception).
- Every run logs: corpus hash, label source, augmentation seed, eval commit.

## Kill gate (unchanged)

If neither route beats the n-gram baseline meaningfully with context ON
(target: +8pt top-1 or better), the neural route as designed is dead and we
stop and rethink rather than scale.
