# Internet-authentic corpus expansion (decided 2026-08-26)

The user wants the corpus to sound like the actual Chinese internet, not just
wiki/news. Slang, 梗, abbreviations and homophone play are precisely where
n-gram IMEs fail hardest, so this is the differentiation, not garnish.

## Sources, in priority order

1. **中国大陆网络用语列表** (zh.wikipedia) — a *seed lexicon*, not a corpus:
   term + 1-3 sentence explanation + origin, categorised (拼音缩写/谐音/缩句…).
   Fetch via the MediaWiki API (proper route, no scraping). Feeds the synthesis
   path below.
2. **萌娘百科** — **the entire article text is training corpus** (user
   directive) — the prose itself is live internet Chinese. Source selection
   (corrected 2026-08-26 after the user pointed out fresh dumps exist; the
   2019 IA dump is superseded):
   - **primary**: HF `YCWTG/MoeGirlPedia_zh_cleaned_latest` — cleaned JSONL
     {title, text} from the 2025-10 full dump. Sampled: prose is clean;
     residue = flattened infobox `key=value` lines, removed by a line-level
     Rust filter (`^\S+=` pattern) plus the standard han-ratio filter.
   - **freshness top-up**: HF `milashkaarshif/MoeGirlPedia_wikitext_raw_archive`
     — monthly raw wikitext archives, still updated (through 2026-02); process
     increments with our own wikitext stripper when we need post-2025-10 text.
   - IA full dumps (2019, 2023, 2025 slices) exist but are superseded by the
     above. Entry titles additionally seed the synthesis path below
     (`outloudvi/mw2fcitx` and `suiginko/moetype` already build IME dicts from
     moegirl titles — reuse before reinventing).

3. **Bilibili comments/danmaku** — `Midsummra/bilibilicomment` (HF) and/or
   VideoIC (5M comments). Real informal typing, short and noisy.
4. **梗百科 (gengbaike.cn)** — SKIPPED: serves 403 with an anti-bot session
   cookie even for robots.txt. Not worth fighting; coverage overlaps 1+2.

## Sogou scel dictionaries (seed lexicon, added 2026-08-26)

Downloaded via `mlime lexicon fetch`. These are **word-pinyin lexicons**, not
running text — they skip the corpus/g2p pipeline and land in `data/lexicon/`
as parquet shards with schema `{word, pinyin, dict_id, dict_name, rank}`.

| slug | id | name | quality |
|---|---|---|---|
| `wangluo-liuxing-xinci` | 4 | 网络流行新词 | **premium** — manually scanned 2026-08-26, very high quality |
| `bilibili-wanggeng` | 177287 | 哔哩网梗 | unreviewed, appears reasonable |

Source: https://pinyin.sogou.com/dict/ — auto-updated weekly by Sogou.
Download quirk: the server drops HTTP/2 connections; force HTTP/1.1.
CLI: `mlime lexicon fetch` (all) or `mlime lexicon fetch --dict <slug>`.

These lexicons are high-priority seed data for the Luna synthesis path and
directly usable as n-gram vocabulary. The 网络流行新词 dict in particular was
manually reviewed and found to be very high quality internet vocabulary — treat
it as a premium data source in any downstream pipeline.

## Luna synthesis path (the cheap-LLM leverage)

Seed lexicon (source 1, plus entry titles from 2) → gpt-5.6-luna generates N
realistic usage sentences per term with a preceding-context turn, IME-register.
Rules:
- synthetic samples carry `source=synthetic-luna` and are **training-only**;
  the eval set must never contain LLM-generated text (contamination).
- keep per-term provenance so a bad generation batch can be dropped wholesale.

## Performance directive (user, binding — escalated 2026-08-26)

**Everything is Rust except the GPU training loop.** After the g2p hang
(g2pw's torch DataLoader multiprocessing deadlock), the user ruled: "please
always use rust". Data processing, g2p annotation (`ort` + `tokenizers`
against the same g2pw.onnx), LLM annotation (`async-openai`), eval and
inference all live in workspace crates; `rust-script` (rayon) for one-offs.
Python remains ONLY as HF dataset download shims and torch training scripts
on Kaggle/Colab.
