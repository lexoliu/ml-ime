# Internet-authentic corpus expansion (decided 2026-08-26)

The user wants the corpus to sound like the actual Chinese internet, not just
wiki/news. Slang, 梗, abbreviations and homophone play are precisely where
n-gram IMEs fail hardest, so this is the differentiation, not garnish.

## Sources, in priority order

1. **中国大陆网络用语列表** (zh.wikipedia) — a *seed lexicon*, not a corpus:
   term + 1-3 sentence explanation + origin, categorised (拼音缩写/谐音/缩句…).
   Fetch via the MediaWiki API (proper route, no scraping). Feeds the synthesis
   path below.
2. **萌娘百科** — full WikiTeam dump on Internet Archive
   (archive.org/details/wiki-zhmoegirlorg), CC BY-NC-SA 3.0 CN (fine for a
   research prototype). Wikitext → prose; its 梗-explanation pages are dense
   with real usage examples.
3. **Bilibili comments/danmaku** — `Midsummra/bilibilicomment` (HF) and/or
   VideoIC (5M comments). Real informal typing, short and noisy.
4. **梗百科 (gengbaike.cn)** — SKIPPED: serves 403 with an anti-bot session
   cookie even for robots.txt. Not worth fighting; coverage overlaps 1+2.

## Luna synthesis path (the cheap-LLM leverage)

Seed lexicon (source 1, plus 梗 titles from 2) → gpt-5.6-luna generates N
realistic usage sentences per term with a preceding-context turn, IME-register.
Rules:
- synthetic samples carry `source=synthetic-luna` and are **training-only**;
  the eval set must never contain LLM-generated text (contamination).
- keep per-term provenance so a bad generation batch can be dropped wholesale.

## Performance directive (user, binding)

Heavy corpus processing (normalisation, splitting, filtering, dedup at
millions-of-sentences scale) runs in Rust — `rust-script` for one-offs
(rayon + memchr-class crates; verified working locally, 12 threads), workspace
crates for recurring stages. Python stays for orchestration, HF datasets I/O,
and LLM calls.
