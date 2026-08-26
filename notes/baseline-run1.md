# Baseline scoreboard — run1 (2026-08-26)

First end-to-end numbers. The line every neural route must beat.

## Corpus
5.0M prepared sentences (dialogue 2M / news 2M / wiki 1M) from 1.85M documents;
1.6GB parquet. Eval pool of 4,000 (stratified, seed 7) drawn and exhaustively
excluded from training text (0 leaks over all 4,995,994 lines).

## Dual annotation (g2pW × luna@high, concurrency 32, 87 min, 0 refusals)
- Raw: sentence agreement 82.47%, char agreement 98.97% (84,667 positions).
- Hard-set decomposition (701 rows): **45 = v/ü spelling artefact** (toneless()
  fails to fold `yu`/`yv` — comparison bug, fixed in the Rust port), **309 =
  和 han4-vs-he2 alone** (g2pW systematic Taiwan reading; luna correct in
  context), **347 genuinely hard**.
- **Corrected sentence agreement: 91.3%.** Agreement is recomputable offline
  from the stored parquet (both annotators' readings kept) — no LLM cost.

## KN trigram baseline (char-level, interpolated)
Train: 14.2s, 1.95GB peak RSS, 341MB text → 236MiB model
(18.6M trigram / 2.4M bigram types, vocab 41,923).

## Eval (1,562 all-Han agreed sentences: dialogue 856, wiki 525, news 181)
| metric | value |
|---|---|
| sentence top-1 | **54.93%** |
| sentence top-5 / top-8 | 61.65% / 62.23% |
| character | 88.87% |
| MRR@5 | 0.579 |

81.8% of eval records carry context; the n-gram ignores it by design — that
headroom belongs to the neural route.

Decode spot-checks: full pinyin strong (你好/北京大学/重要的决定 all top-1);
abbreviations are the visible weakness (`zgrm` → 中国人民 ranked 2nd by 0.07;
`bjdx` → 北京大学 top-1). Exactly the gap context + neural emissions target.

## Ops lessons
- The "hung annotate" alarm was a wrong-pid measurement (sockets counted on the
  uv wrapper, not the python worker). The implementing agent verified directly
  (198% CPU, 34 live sockets, advancing logs), declined the kill, and saved the
  84%-complete run. Measure the right process before killing anything.
- g2pw upstream falsy-zero bug (`num_workers=0` → 2) is real and fixed
  (20ffe67), but the 4,000-run completed WITH workers=2 — the fix removes
  fragility; it was not the cause of any observed failure.
