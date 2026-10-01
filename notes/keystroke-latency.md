# Per-keystroke latency (2026-09-28)

Issue #82, part of the macOS input method (#80). The product's cost is the
time from a keystroke to the updated top 8. This stage builds the session
API the input method (#84) calls through a C ABI, and measures that cost
on the decoder exactly as `fused-eval` runs it.

## What exists

- **`crates/ime-session` `Session`** — the keystroke-side API:
  - `Session::new(engine, options)` takes the loaded models by reference:
    the lexicon and the syllable table in `SessionOptions`, everything else
    through the engine;
  - `set_context(text)`, `key(ch)` for letters and `'`, `backspace()`,
    `candidates() -> &[Candidate]` (each carries its text and how many
    typed characters it consumes), `select(index) -> Commit` (the committed
    text joins the context, the unconsumed keystrokes stay pending),
    `reset()`;
  - the list is the `SessionOptions::sentences` best whole-input
    hypotheses (3 by default), then prefix candidates covering the first
    k positions of the best reading's segmentation for k = n-1 … 1 — the
    best text among the surviving hypotheses that is not already listed,
    scored up to position k — capped at 8. A prefix's `consumes` is the
    keystrokes of its syllables, `'`s between them included; `select` on
    one commits its text and leaves the rest pending, so the IME (#84)
    gets its partial commit;
  - `'` is a segmentation boundary typed by hand: the pending input splits
    into pieces at `'`, each piece segments alone, only the last piece may
    end in a half-typed syllable, and the pieces' readings join by the
    cross-product of their k-best paths (`Segmentation::concat`);
  - `stages()` reports cumulative segmentation / emission / decode /
    candidate-assembly nanos; `invalidation()` reports how the ranked
    lattice paths moved.
- **`Engine`** — the trait the session is generic over: "score this lattice
  given this context". `transition()` hands the beam the `Transition`, and
  `emission(&Reading)` returns its `Emission`. #88's end-to-end net is a
  second implementation, not a rewrite.
- **`Fused`/`Timed`** in `crates/ime-session/src/fused.rs` — today's engine:
  route A live (the towers from #81) fused with a `Transition` at the fused
  weight, exactly `fused-eval`'s pipeline. `Timed` wraps any `Transition`
  and times `start`/`advance` without touching its `State` or `HISTORY`,
  because the beam's dedup key truncates the history to `T::HISTORY`.
- **`ime-cli replay`** — the harness. It feeds a slice's records one
  keystroke at a time into a fresh session per record, takes candidates
  after each keystroke, and reports:
  - per-keystroke latency: median, p95, max, overall and bucketed by
    pending-keystroke count;
  - peak RSS;
  - where the time went: segmentation, emission (split into context tower
    and fill tower), beam decode, candidate assembly, transition
    `start`/`advance`;
  - the lattice invalidation histogram;
  - a JSON report at `--out` plus a short table on stderr via tracing.
  - The equality gate: every record is also replayed through a second
    session with every reuse lever off, and the last keystroke's top
    candidate must equal it. `replay` reports a `mismatches` list.
- **`ime-session/tests/fixture.rs`** — the session's last-keystroke top 1
  equals a literal `decode_many` (what `fused-eval` runs) on the fixture
  towers, with and without incremental reuse, plus `'` handling,
  `select` semantics, and the prefix candidates: selecting a
  two-character prefix of "woaibeijing" commits it and leaves "beijing"
  pending, and a mixed full/abbreviated input yields prefixes only at
  syllable boundaries.

The prefix candidates cost no decode — only a walk over the beam's
finished hypotheses; `Hypothesis::steps` carries the running score so a
prefix's score is the hypothesis's up to its last position. The walk
shows in the stage split as `candidates_ns` and is negligible (per
keystroke: single-digit microseconds, under a hundredth of a percent of
emission).

## What "incremental" means here

The fill tower sees the whole reading, so its outputs at earlier positions
can move when a syllable is added. The issue allowed measuring rather than
assuming: the fill graph re-runs on each keystroke as one batched forward,
and the stage split is the evidence for what that costs.

What is actually reused across keystrokes is the context tower's output,
whose inputs genuinely do not change while the context stands: `Fused`
caches the `EncodedContext` keyed on the context text, so `set_context`
runs the tower once and `key`/`backspace` never re-run it. The session
also keeps the pending lattice paths and the transition's resident state
between refreshes; the beam itself re-runs wholesale.

The invalidation counters say why partial-lattice reuse would not have
paid on this data: over 2,497 comparable refreshes, zero kept every ranked
path's spans identical. The first divergence lands at span index 0 for
23% of keystrokes — mostly the leading incomplete-tail segment growing —
and within the first eight spans for 96%.

### The LM state cache

The character LM was ~90% of a keystroke on the M1 (see below): the beam
asks the transition for a state per `(character prefix, request)`, ~900
`advance` rows a keystroke, almost all of which recur keystroke after
keystroke because the text so far is mostly the same. `ime-session`'s
`Cached` wraps any `Transition` in a map from that pair to the produced
state:

- `start` is keyed by context: the first path asking for a context's
  start state prefills once, every later start — the other paths', and
  every subsequent keystroke's — is a hit on the stored entry.
- Entries live under an LRU bound (`--lm-cache`, resident rows: for the
  LM each compacted entry is one row, so the cap is the layout's natural
  unit). The whole map is dropped when `set_context` changes the context.
- An entry is stored as produced and moved into its compact form — one
  resident row in the LM's `CachePool` — only on its first hit, so a state
  pays the row copy only once it proves it recurs.
- The request's candidate set is part of the key: a stored state answers
  only requests its scores cover (`AskedSig::covers`), so an
  `unasked-candidate` panic cannot be hidden by the cache.
- The equality gate is unchanged: `--incremental` off wraps the model in
  `Cached` at capacity zero, every reuse off, and the last top 1 must
  match on every record. The hit rate is in the stage split's `cache`
  line and the JSON report.

## Measurements

Replay on the eval3 dev slice (`--slice dev`, share 0.0905), first 100
records, 2,597 keystrokes, neural-only (`--no-transition`): this VM has no
trigram or char-LM export, so the transition is `NoTransition` and the
same command line on the M1 mini carries `--model` and `--lm`. Each row is
one run; `key` wall time per keystroke.

| config | median | p95 | max |
| --- | --- | --- | --- |
| fp32, 1 thread, baseline | 41.7 ms | 76.0 ms | 150.1 ms |
| fp32, 1 thread, incremental | 35.8 ms | 66.9 ms | 115.8 ms |
| int8, 1 thread, incremental | 32.9 ms | 65.9 ms | 154.4 ms |
| fp32, 4 threads, incremental | 31.3 ms | 56.7 ms | 100.5 ms |
| int8, 4 threads, incremental | **26.2 ms** | **50.3 ms** | **86.1 ms** |
| fp32, Core ML, incremental | 58.5 ms | 106.9 ms | 235.1 ms |

The 50 ms p95 target lands at int8 + 4 threads + incremental (50.3 ms),
and is missed by every other configuration. Per-band medians at the best
configuration: 7.5 ms at 1–4 typed chars, 25.2 ms at 9–16, 41.8 ms at
33–48 — the tail grows with the reading, since the fill tower re-runs
over the whole lattice each keystroke.

Stage split at the best configuration, cumulative over the 2,597
keystrokes: fill tower 52.2 s (78% of emission), context tower 0.46 s
(1,096 cache hits of the 1,140 requests — the tower ran only 44 times:
once per distinct context, not once per keystroke), beam decode 16.1 s,
candidate assembly 0.01 s, segmentation 0.26 s, transition `advance`
0.5 ms (no-op). The fill tower is the lever that matters.

Notes per lever:

- **incremental** (context-tower cache): context_ns 15.85 s → 0.58 s on
  fp32/1t — about 6 ms per keystroke, the whole gain. Fill is unchanged.
- **int8**: fill 78.8 s → 69.7 s at 1 thread and 64.8 s → 52.2 s at 4 —
  faster than fp32 per keystroke here, the opposite of #81's per-record
  emission (the cached context tower is out of the int8 comparison);
  RSS halves (2.1 GB → 1.0 GB).
- **4 threads**: fp32 fill 78.8 s → 64.8 s.
- **Core ML**: 1.8× slower than CPU fp32 on this VM (fill 141 s vs 78.8 s).
  The VM has no real GPU/ANE; the attached `gpu-coreml` build still lets
  the M1 measure it.

Zero mismatches in every run — the incremental engine's last-keystroke
top 1 equals the baseline re-decode's on all 100 records, and the fixture
test ties it to `decode_many` itself.

### The M1, full engine

The same binary on the Mac mini's M1, with the kn-trigram plus the int8
char LM at lm-weight 1 (the `resident-candidates` export), fp32 route A on
4 threads, `--incremental`, eval3 dev, 40 records, 1,036 keystrokes:

| | median | p95 | max |
| --- | --- | --- | --- |
| per keystroke, fp32 route A | 565 ms | 1,876 ms | 3,707 ms |
| per keystroke, int8 route A | 522 ms | 1,757 ms | |
| typed 1–4 | 84 | 300 | 483 |
| typed 5–8 | 345 | 487 | 657 |
| typed 9–16 | 520 | 732 | 979 |

Peak RSS 2.17 GB, 0 mismatches. The stage totals over the 1,036
keystrokes:

| stage | total | per keystroke |
| --- | --- | --- |
| LM advance (8,258 calls, 934,989 states) | 566.8 s | 547 ms |
| LM start (7,227 calls) | 49.2 s | 47 ms |
| fill tower, fp32 | 70.6 s | 68 ms |
| fill tower, int8 | | 26 ms |
| context tower (16 runs, 429 cached) | 1.1 s | |
| segmentation | 0.18 s | |

Two findings. The character LM is ~90% of a keystroke: each keystroke
re-decodes the whole sentence — eight paths by the beam width by every
position, ~900 advanced states a keystroke, recomputed from scratch though
most were computed the keystroke before. `start` alone prefills the same
context about seven times a keystroke. And route A flips its int8 verdict
on this silicon: the int8 fill tower costs 26.0 ms against fp32's 68.2 —
2.6 times faster where the VM measured the reverse (see
`notes/route-a-rust.md`). The LM is what the state cache (`--lm-cache`)
targets.

### The cache, on the real export

The same runs on the VM against the int8 `resident-candidates` char LM
the M1 uses (`--lm char-lm-int8`, no trigram on the VM, int8 towers on 4
threads, eval3 dev, 40 records, 1,036 keystrokes). The measured session
is `incremental` — context tower cache and `--lm-cache` both on — against
the all-off baseline in the same run; 0 mismatches in every row.

| config | median | p95 | max | LM advance | hit rate | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (no reuse) | 328.7 ms | 1,016.6 ms | 2,003.0 ms | 340.2 s | — | 2.55 GB |
| cache, 512 rows | 502.6 | 1,396.1 | 3,030.5 | 537.1 s | 36.2% | 5.20 GB |
| cache, 2,048 rows | 276.1 | 677.9 | 1,361.0 | 272.8 s | 76.3% | 8.28 GB |
| cache, 4,096 rows | 275.5 | 633.0 | 2,013.2 | 254.7 s | 80.7% | 10.47 GB |
| + `--lm-threads 4` | 294.1 | 841.0 | 2,064.3 | 291.2 s | 80.7% | 11.18 GB |
| + `--max-paths 4` | **104.0** | **268.5** | **558.4** | 99.7 s | 79.6% | 10.66 GB |

Three findings in the rows.

- **The cache pays, once it is lazy.** At 4,096 rows the hit rate is 80.7%
  and the LM's `advance` falls 340 s to 255 s; the first version, which
  copied every produced state into a pool row at store time, paid the copy
  ~180 times a keystroke and measured *slower* than no cache (412 s of
  `advance`). Compacting only on a hit — the state that provably recurs —
  is what turns it. At 512 rows the working set thrashes (36% hits,
  587k evictions, worse than baseline); the live prefix set is roughly a
  few thousand rows, so the bound wants ~2,048 rows and up here.
- **The bound is a memory trade.** At 4,096 rows the pool plus the pinned
  step generations peak at 10.5 GB — over budget for a 16 GB product
  machine; 2,048 rows keeps 94% of the hits at 8.3 GB. On the M1 the cap
  is a lever to tune, not a default to keep.
- **The shape levers beat the cache anyway.** Halving `max_paths` to 4
  halves the LM's rows: 104 ms median against the cache's own 275 ms, and
  `--lm-threads 4` costs instead of paying (294 ms median) — the per-row
  steps are too small for intra-op parallelism at one sentence. Neither
  touches the hit rate.

`max_paths` on the dev slice's accuracy (498 records, `fused-eval`, the
same int8 towers and LM): 8 paths reads top-1 0.7631 / top-8 0.8052 /
char 0.9582 / MRR 0.7802; 4 paths reads top-1 0.7610 / top-8 0.8092 /
char 0.9582 / MRR 0.7803 — one sentence moved each way, a wash.

### The cache, per position — the `resident-pages` export

The row cache above cannot ship: its memory is the pinned step
generations, and the mini it runs on has 8 GB, not 16. The fix is a new
export layout, `resident-pages`, which replaces `resident-candidates`
(the loader's layout pin moved; the old layout is removed, not kept):

the step graph owns a resident KV pool `[L, capacity, H, D]` bound as
graph inputs, and a cached state is a chain of page ids — a radix trie
over character prefixes whose nodes each own one position's K and V for
every layer. One node is **49,152 bytes** on the real export (`[12
layers, 8 heads, 1, 64]` × K,V × fp32); the 4,096-node default bound is
**192 MB**. The context prefill writes the context's positions into
claimed pages once — that chain is the trie's shared root.

Per step the host passes only indices and flags — `token`, `page_row`
(the i64 page id of each position on each row's path), `mask`,
`candidates`, `source_row`, and a scratch `state_*` block for rows
whose newest positions are still unresolved. The graph gathers K/V by
page index, attends, scores the candidates, and returns the new
position's K/V per row — which the host writes into the claimed page;
no KV bytes cross the host boundary per step and no step generations
are produced or pinned.

- Produced states resolve eagerly to a fresh trie node (one page claim,
  one 48 KiB write). When the pool is exhausted the claim fails and the
  state stays `Pending`: it records its ancestor leaf plus only its own
  tip position's K/V, and the next step materialises that contiguous
  tail into a scratch slot — the `state_keys`/`state_values` input, slot
  0 reserved as the pad `source_row.clamp(0)` lands on.
- The LRU bound (`--lm-cache`) counts nodes; when claims run out a
  garbage pass sweeps the insertion order for nodes whose hypotheses
  and cache entries have all dropped (leaf nodes only — a claimed page
  is stolen, the node tombstoned, and stale memos self-heal on the next
  lookup).
- The equality gate is unchanged: the baseline session is all-off at
  the engine level — the pool is the layout's state representation, not
  a cache toggle (a `cache_rows` of 0 still builds a 1,024-page floor,
  which the context prefill's claims need).

Product configuration (the one `fused-eval` reports — kn-trigram plus
the weight-only int8 char LM from #95 at lm-weight 1, int8 route A on 4
threads, neural weight 1), eval3 dev, 40 records, 1,036 keystrokes,
`--lm-cache 4096`, `--incremental`, **0 mismatches**:

| layout | median | p95 | max | transition advance | hit rate | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| resident-candidates | 145.5 ms | 300.2 ms | 473.2 ms | 98.2 s | 81.8% | 3.94 GB |
| **resident-pages** | **216.8 ms** | **509.5 ms** | **764.6 ms** | **175.9 s** | **81.0%** | **5.53 GB** |

Same 81% hit rate (757,774 step hits, 178,894 misses; `start` 7,186
hits against 41 prefills) at the same 192 MB bound — the hit count is
the prefix statistics, which did not move. The VM's per-keystroke cost
did: the in-graph append is slower per row on CPU than the old host-
memcpy-into-scratch was (~0.69 ms a missed state against ~0.56 ms), and
the page bookkeeping the old layout never needed is real. The layout
trade was never about this VM's CPU — it removes the ~100 MB per-step
generations and makes the cache shape fixed, which is what a GPU or the
ANE needs (below).

#### Where a keystroke goes

The stage totals over the run's 1,036 keystrokes, product
configuration, per keystroke (means of the totals; the median keystroke
is 216.8 ms):

| stage | total | per keystroke | share |
| --- | --- | --- | --- |
| LM misses — compute | 122.6 s | 118.3 ms | 51% |
| LM bookkeeping | 53.4 s | 51.5 ms | 22% |
| beam selection | 48.9 s | 47.2 ms | 21% |
| fill tower | 21.0 s | 20.3 ms | 9% |
| context tower (429 cached) | 0.36 s | 0.35 ms | |
| LM `start` | 0.40 s | 0.38 ms | |
| segmentation | 0.11 s | 0.11 ms | |
| candidate assembly | 0.005 s | ~0 | |

- **LM misses** dominate: 178,894 missed states ≈ 173 per keystroke at
  ~0.69 ms each on this VM's CPU — the step graph's own matmuls, the
  part no host-side fix touches.
- **Bookkeeping** is the `bookkeeping_ns` counter: the `page_row`/`mask`
  builds per step, pending-tail scratch materialisation, claims and
  page writes — 51.5 ms a keystroke, one third of it the per-step
  `page_row` index fill over ~8 path × 16 beam rows.
- **Beam selection** is `decode` minus the transition's `advance` and
  `start` — 47.2 ms a keystroke of hypothesis churn the LM numbers sit
  inside the decode of.

#### Core ML, re-tested

The earlier Core ML refusal was the empty-KV-cache input shape. The
resident pool is a fixed-shape bound input, so the `gpu-coreml` build
opens the paged graphs now: the 4-record run completes with **0
mismatches**. On this VM — "Apple M4 Pro (Virtual)", no real GPU or
ANE — it is slower than CPU anyway, 443.4 ms median and 441 ms a
keystroke of `advance` over the same 91 keystrokes (~2.6× the CPU
row's). The VM cannot price the provider; the attached
`ime-cli-aarch64-apple-darwin-coreml` build is for the M1 run, where
the step graph's 128–170-row steps are the question.

#### The M1, ef80cca

Same workload on the Mac mini — kn-trigram plus the weight-only int8 LM
at lm-weight 1, int8 route A on 4 threads, `--incremental`, `--lm-cache
4096`, lm-threads 1, eval3 dev, 40 records, 1,036 keystrokes
(resident-candidates layout, the preceding revision):

| | median | p95 | max | peak RSS |
| --- | --- | --- | --- | --- |
| per keystroke | 232 ms | 481 ms | 850 ms | 1.48 GB |

0 mismatches. Per keystroke: LM advance 180.2 ms, fill tower 25.8,
context 0.4, segmentation 0.2 — 759,284 step hits against 175,623
misses (81%), so ~170 missed states a keystroke at ~1 ms each on the
M1's CPU: ~76 MFLOP a row-step, ~13 GFLOP a keystroke. No CPU
bookkeeping fix reaches 50 ms from there; the GPU or the Neural Engine
has to run the LM step, which is exactly what the resident-append pool
makes possible — the cache is a fixed-shape graph input, and the
`gpu-coreml` binary above is the measurement.

### The beam's shape

`decode_many` advances 16 beams on each of up to 8 segmentation paths —
128 rows a step — although the product's shape is one sentence, eight
beams (`AGENTS.md`). The grid, `fused-eval --slice dev` on all three
twins, product configuration:

| shape | eval3 top-1 | abbrev top-1 | mixed top-1 | eval3 top-8 | abbrev top-8 | mixed top-8 |
| --- | --- | --- | --- | --- | --- | --- |
| beam 4, paths 2 | 0.7691 | 0.1874 | 0.3347 | 0.7912 | 0.2105 | 0.3864 |
| beam 4, paths 4 | 0.7671 | 0.2126 | 0.3636 | 0.7892 | 0.2400 | 0.4215 |
| beam 4, paths 8 | 0.7671 | 0.2337 | 0.3864 | 0.7892 | 0.2674 | 0.4483 |
| beam 8, paths 2 | 0.7912 | 0.1979 | 0.3409 | 0.8273 | 0.2400 | 0.4174 |
| beam 8, paths 4 | 0.7892 | 0.2274 | 0.3698 | 0.8253 | 0.2737 | 0.4504 |
| beam 8, paths 8 | 0.7892 | 0.2484 | 0.3905 | 0.8233 | 0.3053 | 0.4814 |
| beam 16, paths 2 | 0.7912 | 0.2063 | 0.3533 | 0.8273 | 0.2505 | 0.4298 |
| beam 16, paths 4 | 0.7892 | 0.2358 | 0.3822 | 0.8253 | 0.2863 | 0.4669 |
| beam 16, paths 8 | 0.7892 | 0.2589 | 0.4070 | 0.8233 | 0.3200 | 0.5000 |
| shared beam 8, paths 8 | 0.6506 | 0.1937 | 0.3182 | 0.6727 | 0.2168 | 0.3843 |

(Character accuracy moves with top-1 in the same direction everywhere;
the full table is in `beam-grid.txt` on the box.)

Two facts fall out:

- **On eval3, beam 8 × 2 paths already matches 16 × 8** (0.7912 vs
  0.7892 — a tie within a record). The other twins decide: on the
  abbreviated and mixed slices every shrinking costs real top-1 —
  8×8 loses 1.1 and 1.7 points respectively — because the cut lattice
  paths are where the right reading lives there. No shape below 16 × 8
  stays within 0.3 top-1 points on every twin, so **the session default
  stays `beam_width = 16`, `max_paths = 8`**.
- **A joint beam of 8 shared across paths is much worse** (0.65 / 0.19 /
  0.32 top-1): pooling the cut across workers at each position lets the
  high-scoring paths starve the rest, and hypotheses from different
  readings are not interchangeable — pruning one path's tail removes its
  future descendants, which a per-path beam keeps. The `--shared-beam`
  flag measured the contrast and is removed; the numbers stand.

The 50 ms p95 target is not met by the shape grid — the smallest safe
shape is the current one — so the keystroke cost rides on the LM step
and its bookkeeping, which is what the resident-pages layout and the
Core ML retest above are about.

### The M1, 53d509c — and the equality-gate failure

Same product configuration, resident-pages layout, `ime-cli` at 53d509c
on the Mac mini (eval3 dev, 40 records, 1,036 keystrokes,
`--incremental --lm-cache 4096`, int8 towers on 4 threads):

| | median | p95 | max | peak RSS |
| --- | --- | --- | --- | --- |
| per keystroke | 334.9 ms | 972.9 ms | 1,815 ms | 3.03 GB (harness) |

**The gate fails on the M1: 1 mismatch** the VM's 0-mismatch run never
produced. Record 180
(`shiyeyihouweilenengliangkuaizhinengqudangjiaodoushi`): the incremental
session returns 失业以后为了能量快只能去当角斗士, the all-off baseline
失业以后为了能量会只能去当角斗士. Per keystroke: transition advance 291.5 ms
of which LM bookkeeping 157.7 ms (≈3× the VM's figure), 180.3 misses;
emission 26.3 (fill 25.3). Cache 749,876 step hits vs 186,812 misses
(80.1%), 96,607 evictions. `/usr/bin/time`: 2,304 s real, 1,854 s user,
394 s sys.

#### Root cause: `PagePool::reclaim` double-ownership

`PageSlot::drop` pushes its index back to `free` unconditionally. The
old `reclaim()` tombstoned a victim leaf and returned `Some(slot.index)`
— the index, not the slot. The victim's slot then dropped and pushed
the same index into `free` a second time, so the *next* `claim()`
issued the page to a second live node while the claimant already owned
it. Two live chains interleaved `write_page`s on one page: the older
chain's later positions read the younger chain's K/V — a corrupted
history that scores differently from a fresh prefill. It only fires
under eviction pressure — `reclaim` is the single path that both frees
and hands out — so the VM's runs, which never evicted at the M1's
96,607-eviction rate, saw 0 mismatches. The fix moves the slot itself
to the claimant (`reclaim() -> Option<PageSlot>`); the regression test
`reclaim_hands_a_stolen_page_to_one_claimant` fails on the old code and
passes now. Verification on the fixed binary: the same 40 records
replayed at `--lm-cache` 64 and 256 — forced reclaim pressure, 704,224
and 686,298 evictions, ~7× the M1 run's 96,607 — with 0 mismatches.

The same bug was the fused-eval accuracy gap, not a pages-layout
scoring difference: a `fused-eval --slice dev` run on the pre-fix code
scored top-1 0.7570 against the resident-candidates layout's 0.7892 on
the same 498 records — its records share one pool across rayon
workers, so the double-issue corrupted live states continuously.
Re-run on the fixed code (serial, same configuration): **top-1
0.7912** (394/498), top-8 0.8554, char 0.9616, MRR@8 0.8186 — the
pages layout scores within jitter of the row layout it replaced. A
40-record dev subset scored 0.6250 top-1 on *both* layouts, which is
why the gap only showed at full-slice scale. Paired per-record on a
160-record subset (`fused-eval --dump`): pages 0.8063 vs candidates
0.8000, with top-1 differing on 6 records — 3 flips each way, all
near-ties — the signature of fp-boundary jitter, not a systematic
divergence.

The same window fixed the `fused-eval` prefill panic on a degraded
pool: the context prelude claimed its ~64 pages up front, and parallel
records sharing one 4,096-page pool exhausted it. `resolve_at` now
accepts context descs (`ch: None` — two contexts share nothing, so they
never enter the parent's `children` map), and `prefill_pages` claims
greedily: positions past exhaustion degrade to `Pending` rows — the
same fallthrough `advance` uses — holding the last claimed leaf plus
tip `RowRef`s into the prefill generation.

#### Profiling the Rust side

samply over a product-configuration replay (1,277.9 s of stacks): the
top items are `MlasQ8Int8GemmKernelNeon` 429.5 s, `memcpy` 307.6 s,
`DefaultHasher` 29.6 s, `rustc_entry` 36.3 s, quicksort 20.0 s and
`NgramModel::score` 19.6 s. The fixes against the hot items:

- `decode_many` (`crates/ime-decode/src/beam.rs`): the survivors sort is
  a `select_nth_unstable` partition to `keep + width` deep, then only
  the head sorts; the diversity guard's `next.iter().position()` scans —
  quadratic in the candidate count per guarded slot — become one
  unsorted pass building `first_of`/`first_char` best-index maps (the
  guard picks then read exactly what a scan over the sorted `next`
  found); `kept` is a `Vec<bool>`; hash maps are `hashbrown`'s raw
  entry on precomputed keys.
- `ime-lm`: the pending rows' `state_*` scratch generation — a
  `[slots, L, H, T, D]` allocation on *every* step, the 394 s of `sys`
  the M1 spent faulting fresh pages — is now `PagePool::scratch`,
  cached and grown, never re-allocated at the same depth; `asked_ids`
  is an `Arc<[u32]>` shared with the binding; `fill_page_row` writes the
  chain's page indices without re-walking it.
- `ime-session/src/cache.rs`: the coverage check stops re-walking the
  produced chain.

Before/after, same replay (eval3 dev, 40 records, product
configuration, `--lm-cache 4096`):

| | before (53d509c) | after |
| --- | --- | --- |
| median / p95 / max | 216.8 / 509.5 / 764.6 ms | 199.0 / 588.3 / 971.0 ms |
| beam selection | 47.2 ms/ks | 28.4 ms/ks (−40%) |
| LM bookkeeping | 51.5 ms/ks | 42.4 ms/ks (−18%) |
| `advance` total | 169.8 ms/ks | 190.3 ms/ks (+12%) |
| cache step hits | 81.0% | 80.9% |
| mismatches | 0 | 0 |

The `advance` residual is honest: bookkeeping fell 9 ms while the
session-side compute inside `advance` rose ~30 ms — the scratch write
now transposes position-major K/V into the `[L,H,T,D]` slabs the graph
gathers on; it is the price of the fixed-shape input and the resident
device graph removes the scratch path entirely. The p95 tail went up
while the median fell — the after-run shared the box with two more
replays and a fused-eval, so the tail is contention, not a regression
in kind.

#### Core ML on the M1: dead end

Measured on the M1, 53d509c, fp32 export, `--backend coreml`, same
product configuration — stopped after 80 minutes where the CPU int8
run finished the same 40 records in 38. The GPU sat at ~0% the whole
time (`ioreg` Device Utilization), the process used one core at ~55%.
A 3 s sample shows the Core ML partition running on BNNS — the CPU —
plus ORT MLAS `Add`/`Zero` kernels, with `__bzero` (536 samples) and
`memmove` (298) at the top of the busy stacks: Core ML through ORT runs
this graph on the CPU and copies the resident pool across the EP
boundary every step. The int8 run was skipped — `MatMulNBits` is not a
Core ML op anyway. **Dead end for this graph.**

### Metal, directly: an LM step off-ONNX

Core ML being dead, the step was measured on Metal itself. A standalone
benchmark (candle 0.9.2's `metal` backend — a Rust crate dispatching
Metal kernels, no ONNX) models the product's step exactly: 12 layers ×
512 hidden × 8 heads of 64, pre-norm GELU FFN 2048, tied 41,928-vocab
head; ~22 rows a call × 8 calls a keystroke at depths 27–34; a resident
KV pool of the pages layout's size (`[12, 4096, 8, 64]` per K and V ≈
192 MB fp32) in GPU buffers. Each call appends one position per row per
layer in place (`scatter_set` into each row's own page — what a fused
append kernel inside the step graph would do) and gathers the row's
history for attention. Per-stage timings force a device `synchronize`,
so stage sums overstate — the call total is the number to compare.

| leg | p50 / call | p95 | GFLOP/s |
| --- | --- | --- | --- |
| candle Metal fp32 | 76.1 ms | 110.5 ms | 38.9 |
| candle Metal **q8_0** | 30.9 ms | 93.3 ms | 54.6 |
| candle CPU fp32 (no Accelerate) | 94.1 ms | 538.5 ms | 15.7 |
| ORT int8 CPU — product `CharLm::advance`, 22 rows | 99.0 ms | 914.0 ms | — |

- **An int8/q8 matmul exists on this path**: `QMatMul` over candle's
  Metal kernels runs Q8_0 (also Q4_K/Q6_K/Q8_K) and is ~2.5× fp32 here.
- Dispatch dominates: the KV append alone is 24 `scatter_set`
  dispatches at ~0.3–0.45 ms each on this stack, and a call is ~250
  kernel dispatches total; a real integration fuses LN/GELU/softmax and
  holds one command buffer, paying the overhead once rather than per
  op.
- The VM's GPU is an "Apple Paravirtual device" — virtualised, not the
  M1's silicon — so these are sanity numbers, not the verdict. On the
  M1 the ORT int8 step is ~0.7–1 ms a row (~15–25 ms/call at 22 rows);
  the attached `metal-bench-aarch64-apple-darwin` binary runs the
  identical measurement there: `--mode candle --device metal [--q8]`
  for the Metal legs, `--mode ort --lm <dir>` for the CPU step.

#### The M1 runs it (98bdbba, defaults 22 rows × 8 calls × cap 4096)

| leg | p50 / call | attn | body |
| --- | --- | --- | --- |
| candle Metal q8_0 | 57.7 ms | 37.4 | 17.4 |
| candle Metal fp32 | 64.2 ms | 36.6 | 21.1 |
| candle CPU fp32 | 39.6 ms | 5.5 | 22.0 |
| ORT int8 `--mode ort` | 1,771 ms | broken | |

GPU utilisation ~0–8%, ~40 GFLOP/s: **Metal as built loses to the CPU**
— gather + qk + av is dispatch- and copy-bound, ~250 dispatches a call
at whatever the per-dispatch floor is. The ORT leg's 1,771 ms was the
bench's own bug, not ORT: `run_ort` hardcoded `cache_rows = 65,536`
pages = 3.2 GB of pool buffers — on an 8 GB mini the pool thrashed
swap; it now follows `--cap` (default 4,096 = 192 MB) and resets the
batch's depth each iteration so depths stay in the product's range.

Follow-ups this measurement named:

1. **Merging the ~8 calls of a keystroke: not possible — a hard data
   dependency.** They are not eight independent batches; they are the
   beam's sequential *positions*. `decode_many`'s `while workers.any(
   pending)` loop is one iteration per position: `survivors()` picks each
   worker's `width` candidates from the score list the previous `advance`
   produced, the survivors go on `steps` as `&worker.latest[parent]
   .state`, and ONE `transition.advance(&steps)` covers every live
   worker's rows at that position (~22 rows at the product shape).
   Position k's input rows are literally the previous call's output
   states — the call cannot be issued until those scores exist. What
   *is* mergeable is already merged: all workers share each call.
2. A fused paged-attention Metal kernel; the M1's per-dispatch floor
   is measured directly by the bench's new `--mode empty` leg (an
   empty compute dispatch, committed and waited, ×240 — on the VM's
   virtual GPU it is 0.36 ms/dispatch ⇒ ~90 ms of pure dispatch per
   ~250-dispatch call, which is most of why the Metal leg loses here).
3. The ORT leg fix above.
4. candle CPU with Q8_0 weights exists on the same path
   (`--device cpu --q8`); candle's CPU threading follows
   `RAYON_NUM_THREADS`.

### The M1 mismatch is not the reclaim fix

98bdbba's replay on the M1 fails the equality gate again — record 180,
the identical flip (`能量快` vs `能量会`) the M1 produced at 53d509c,
while the VM shows 0 mismatches even under 700K forced evictions. The
`PagePool` double-issue was a real bug, not this one. The machines
differ exactly where it matters: the M1 is `FEAT_DotProd=1,
FEAT_I8MM=0, FEAT_BF16=0, FEAT_SME=0` and the VM's M4 has all four —
the working hypothesis is that ORT's MLAS picks a different
`MatMulNBits` kernel per batch height on the DotProd-only path, so a
row's int8 GEMM result depends on which rows share its call. The cache
then makes it visible two ways: a hit returns scores gathered in a
foreign batch, and hits shrink the live miss batch.

Two instruments ship for the M1 to answer it directly:

- `cargo test --release -p ime-lm --no-run` produces the `charlm-*`
  test binary; `CHARLM_FIXTURES=<dir>/fixtures ./charlm-*
  int8_scores_do_not_depend_on_batchmates --exact` runs the
  solo-vs-128-row-batch bit assertion against the committed fixtures
  (the env override is what makes the binary portable — it bakes the
  build machine's manifest dir otherwise).
- `ime-cli replay --probe <eval-index>` replays one record with the
  LM's real `advance` calls recorded: at the first keystroke the
  incremental and baseline candidate lists differ it prints both
  lists, then re-scores every recorded call on a fresh `CharLm` —
  each input state rebuilt from its own history, scored alone and in
  the batch the call ran — so a batch-dependent kernel shows directly
  as `to_bits` diffs. A second lockstep pair runs the same record
  with the session cache at zero to split cache-restore causes from
  tower-reuse ones.

On the VM the probe is already evidence: record 180 *does* diverge
here too, but at keystroke 47 and only inside the top-8 ordering
(a rank flip the top-1 gate cannot see), and **the cache-at-zero
lockstep is identical** — the towers' incremental reuse is innocent;
the corruption enters only through the state cache.

#### Root cause, second: `materialise_pending` wrote the scratch in the wrong layout

The kernel hypothesis is disproven — `int8_scores_do_not_depend_on_batchmates`
shows 0 of 1.94 M bit-diffs on the VM, and the emitted-score tape puts the
corruption on the input side: `replay --probe 180`'s three-way rescore on a
fresh `CharLm` gives **0 of 385,152 live-vs-fresh diffs** against 212,636
recorded-vs-live, and the moved scores attribute to a scattered set of calls —
`moved-by-call (37, 1740), (51, 1305), (59, 78), (84, 2936), (111, 4908)…` —
not one bad call. Scores 1.36 M of 18.9 M shared (history, candidate) entries
moved by 1–2 nats; the first landed at call 37, roughly the fifth keystroke.

What those calls share is a `Pending` input. A state goes `Pending` when its
chain's claims fail — `free` empty *and* `reclaim` finds no stealable node —
and its history then rides the `state_*` scratch tensors the step declares
`[slots, layers, heads, T, head_dim]`. `materialise_pending` wrote each
position's `heads × head_dim` slice at `(layer * T + position) * heads *
head_dim` — the `[slots, L, T, H*D]` layout — so every pending input scored
attention over head/position-interleaved garbage. The bug is transient by
construction (the next call rewrites the scratch), which is the recorded-vs-
live pattern exactly, and symmetric when both sessions pend together, which
is why every earlier identical-lockstep run was blind to it. The M1/M4 split
is luck, not kernels: the same wrong inputs move a near-tie only where that
kernel's rounding lands.

`pending_states_score_like_resident_pages` forces the path: context nodes are
unstealable (`ch: None` skips the sweep), so forty 29-char contexts pin the
1,024-page floor shut and starts go Pending; advancing them materialises the
scratch, and the test asserts every produced score is bit-identical to the
roomy pool's. It fails on the old offsets (`context 33, beam 0`) and passes
on the fix — per-head `copy_span`s of `head_dim` elements landing each
`(layer, head, position)` at `((layer * heads + head) * T + position) *
head_dim`, with `heads`/`head_dim` read off the state-row shape. The tip
source offset was already right: state *outputs* are `[rows, L, H*D]` — one
position per row — so the read was never wrong; only the scratch write was.
`history_changes_scores` is the control that makes the fixture comparison
non-vacuous.

Pending is real at the shipping cap: the probe's corrupt calls cluster where
the pool is under churn, and `--lm-cache 64`'s 704 K evictions are a heavier
version of the same pressure. The M1's fix does not need to be machine-
specific — the wrong bytes were host-written before the kernel ever saw them.

### The M1 confirms it (e0635ca)

Product configuration on the Mac mini: kn-trigram + weight-only int8 LM at
lm-weight 1, int8 route A on 4 threads, `--incremental --lm-cache 4096
--reference-first`, eval3 dev, 40 records, 1,036 keystrokes:

| | median | p95 | max |
| --- | --- | --- | --- |
| per keystroke | 296.5 ms | 995.3 ms | 2,139.6 ms |

per keystroke: transition advance 308.3 ms — **bookkeeping 15.4 ms**, down
from 164 once the harness stopped thrashing and the per-step scratch
generation was cached — fill tower 26.2 ms, segmentation 0.2 ms; 180.3 LM
misses per keystroke, hit rate 80.1%. Peak RSS 2.15 GB;
`/usr/bin/time`: 2,029 s real / 1,739 s user / 307 s sys (the mini still
carried ~2.9 GB of unrelated swap). **0 mismatches**, and all 12 `charlm`
tests pass on the M1 — `pending_states_score_like_resident_pages` and
`int8_scores_do_not_depend_on_batchmates` included. The Pending fix is
confirmed end-to-end on DotProd.

What is left is raw LM compute: ~180 row-steps a keystroke at ~1.7 ms each
through ORT int8. The follow-ups, in the order measured:

1. **Merging the ~8 calls of a keystroke: not possible — a hard data
   dependency.** They are the beam's sequential *positions*, not eight
   independent batches: `decode_many` runs one iteration per lattice
   position, `survivors()` picks each worker's `width` candidates from the
   score list the previous `advance` produced, and position *k*'s input
   rows are literally the previous call's output states. What is
   mergeable is already merged — one `advance(&steps)` carries every live
   worker's rows (~22 at the product shape).
2. **candle CPU at the product shape, fp32 vs Q8_0 by thread count**
   (VM, 22 rows × 8 calls, cap 4096 — the M1 binary reports its own):

   | leg | mean ms / call | GFLOP/s |
   | --- | --- | --- |
   | ORT int8, 1 thread | 21.1 | — |
   | ORT int8, 2 threads | 16.7 | — |
   | ORT int8, 4 threads | 15.2 | — |
   | candle CPU fp32, 2 threads | 37.4 | 67.9 |
   | candle CPU fp32, 4 threads | 29.7 | 81.9 |
   | candle CPU Q8_0, 1 thread | 72.1 | 35.7 |
   | candle CPU Q8_0, 2 threads | 59.0 | 44.2 |
   | candle CPU Q8_0, 4 threads | 69.2 | 37.2 |

   On the VM, Q8_0 dequant is *slower* than the fp32 GEMM at every thread
   count and never reaches ORT int8 — `RAYON_NUM_THREADS` is the knob and
   the bench prints it. If the M1 disagrees, candle-q8 becomes the CPU
   backend; on this evidence it does not.
3. **Metal, directly.** `--mode empty` puts the VM's per-dispatch floor
   at 0.30 ms. `--mode fused` is the fused paged-attention kernel: one
   threadgroup per (row, head) × 64 threads, scores and softmax in
   threadgroup memory, all 12 layers inside one dispatch, `rows*heads`
   groups — **0.93 ms per call** on the virtual GPU, reading 33.5 MB of
   paged KV at ~36 GB/s. The unfused candle leg spent 37 ms on the same
   section; fused, the section costs a dispatch floor plus its bytes.
   The M1's real GPU prices both.
4. **Memory.** Two changes land here:

   - *The LM's two weight files were byte-identical* — the export already
     dedupes (`prefill.onnx`'s external entries repoint at
     `charlm.weights`, the manifest collapses to a shared table) when the
     tensor names coincide; this export predates it or its names differed,
     so `charlm-pages-int8/` was rewritten in place: prefill's entries
     repointed at `charlm.weights`, `prefill.weights` deleted, ~148 MB of
     mapped pages gone. Verified: loads, 0 mismatches.
   - *The trigram is memory-mappable.* `ProbTable` gains a `Mapped`
     storage (`Arc<Mmap>` + byte ranges — lookups slice straight into the
     page cache) and `NgramModel::open` detects the layout by magic: a
     flat `MGNM` file of 8-aligned little-endian arrays written by
     `ime-cli pack-ngram --model ngram.bin --out ngram.mmap` (~34 s, 759 MB
     vs 674 MB postcard — larger on disk, ~0 on the heap). Postcard files
     still load as before. `the_mapped_layout_scores_like_the_postcard_one`
     pins the two layouts bit-identical; a 20-record `decode` diff is
     byte-identical; a 10-record product-config replay with the mapped
     model gives 0 mismatches and peak RSS **2,719 MB vs 3,227 MB**
     (−508 MB of process RSS — the file pages, unlike the heap copy, are
     evictable under pressure; a `decode` call's peak alone drops 1.45 GB
     → 13 MB).

   Anonymous-resident drops ~650 MB; the single-session budget becomes
   roughly: trigram file-backed + 148 MB LM weights + 312 MB towers
   (mapped) + 192 MB pool + ~60–200 MB scratch + ~80–150 MB states —
   ~0.9–1.2 GB RSS on the harness, most of it evictable. The 1 GB target
   is close but the pool cap and scratch remain the honest levers.

### The M1 prices the levers (89678e6)

Replay on the Mac mini, product configuration with the mapped trigram and
the deduped LM dir (`--reference-first`, 40 dev records, 1,036
keystrokes): **0 mismatches**; median **267.4 ms**, p95 **915.6 ms**, max
2,602 ms. `advance` 282.4 s over 8,258 calls = **34.2 ms/call**;
186,812 misses (180/keystroke); bookkeeping 6.9 s total (**6.7 ms per
keystroke** — the reference-first split removed the contention entirely);
fill tower 25.3 s. `/usr/bin/time`: 1,844 s real; maximum RSS **2.91 GB**;
peak memory footprint **4.04 GB** — *up* from e0635ca's 2.15 GB on the
same machine, despite the mapped trigram and the deduped weights (see the
memory-regression section; `pack-ngram` itself: 3.5 s, 1.08 GB max RSS).

`metal-bench` on the M1, product shape (22 rows, 8 calls, cap 4096):

| leg | per call | keystroke (8 calls) |
| --- | --- | --- |
| candle metal fp32, unfused | 59.0 ms (attn 32.6, body 17.5, KV append 4.4, head 4.1) | 472 ms |
| candle cpu fp32, 1/2/4 threads | 64.4 / 43.0 / 37.4 | 515 / 344 / 299 |
| candle cpu q8, 1/2/4 threads | 73.5 / 53.5 / 58.2 | 588 / 428 / 465 |
| ORT int8, 1/4 threads | 26.1 / 15.9 | 209 / 127 |
| empty-kernel dispatch floor | 0.159 ms mean, 0.203 p95 | ~250 dispatches = 39.6 ms |
| fused paged-attention (12 layers, one dispatch) | 0.907 mean, 1.041 p95; 6.98e-10 vs CPU; 37 GB/s | 7.3 ms |

Two verdicts: **q8 is dead on the M1 too** — worse than fp32 at every
thread count, and ORT int8 stays the CPU backend. And the fused kernel is
**36x the unfused attention on real hardware** — the remaining candle
Metal time is dispatch count (~250 × 0.16 ms), not arithmetic: 2.61 GFLOP
a call is ~1 ms of M1 GPU. So the whole step goes to Metal in a handful
of dispatches; see the fused-step section.

### The whole step on Metal (this revision)

`ime-lm` has a `Backend::Metal`: one `MetalStep` whose kernels are the
whole transformer step — `lm_embed` (embed + input projection + the
position row), `lm_layer` ×12 (LN1 + QKV + fused paged attention + output
projection + LN2 + FFN, the fresh K/V landing in `next_*` for the host's
page write), `lm_head_proj` + `lm_head_mat` + `lm_head_logp` (full-vocab
logits, then log-softmax + gather restricted to the asked ids — the
softmax itself must see the whole row: the reference's logp is
`logits[c] − lse(vocab)`, and masking the lse to candidates shifts every
score by a per-row constant, ranking-identical but a useless diff
signal) — **16 dispatches per call, one command buffer**, weights resident
as `MTLBuffer`s, the page pool one buffer the gather reads in place. A
host write of `token`/`pages`/`mask`/`srow`/`depths`/`cand` (tens of KB)
is the only per-call traffic; `next_*` comes back as the produced state's
append.

Two weight encodings measure on the VM's virtual GPU (product shape,
22 rows × 8 calls, `--mode lm --metal-weights …`):

| encoding | p50 / call | max \|gpu − cpu-int8\| logp | argmax flips |
| --- | --- | --- | --- |
| int8 + per-out scale (`--metal-weights auto`) | 35.9 ms | 1.6–1.9e-1 | 0 |
| fp16 (`--metal-weights f16`, dequantized from the int8 export) | ~40 ms | 9.3e-2 deterministic across three clean runs; one 6.6e0 outlier earlier, never reproduced | 0 |

int8 wins and stays the default; the single 6.6e0 f16 outlier (one run
of four, three later runs landing on the same 9.3e-2) is flagged for the
M1 rather than declared — if it reappears on real hardware it is a
kernel-level bug, not noise.
The parity gate for any real number is the replay's own `0 mismatches`,
which the int8 backend meets end to end on this VM. Both runs below are
the 40-record product replay (`--incremental --lm-cache 4096`,
`--reference-first`, eval3 dev, 1,036 keystrokes):

| backend | median | p95 | max | `advance` / call | misses/ks | bookkeeping/ks | footprint finish | mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu, `lm-threads 4` | 139.7 ms | 358.9 | 611.6 | 13.65 ms (112.7 s) | 172.7 | 2.30 ms | 1,900 MB | 0 |
| metal | 181.8 ms | 358.3 | 679.8 | 16.25 ms (134.2 s) | 172.7 | 2.79 ms | 963 MB | 0 |
| cpu, `lm-threads 1` (product row, M1) | 267.4 ms | 915.6 | 2,602 | 34.2 ms | 180 | 6.7 ms | 4.04 GB | 0 |

The virtual GPU's Metal step (16.25 ms/call) already lands inside the
ORT int8 range on this VM — the M1's real GPU prices it; at the
fused-attention leg's 37 GB/s the step should sit well inside the ~3 ms
target at rows=22. The comparison rows span two machines (VM vs M1) —
the per-call column is the portable one.

#### The M1 verdict on fb578e0 (verbatim)

Replay `--backend metal --reference-first`, 40 dev records, int8 LM,
MGNM trigram, 1,036 keystrokes, **0 mismatches**: median **385.6 ms**,
p95 **846.9**, max 1,466.6. `advance` 37.6 ms/call over 8,258 calls =
300.1 ms/keystroke; 180.6 misses/keystroke; bookkeeping 7.1
ms/keystroke; fill 40.3 ms/keystroke. Memory: **max RSS 882 MB, peak
footprint 1,031 MB** (was 2.91 GB / 4.04 GB on the ORT CPU path);
footprint sat at 860–940 MB through the run — the <1 GB target is met
on the M1 too.

Latency by typed length (median / p95, ms):

| len | median | p95 |
| --- | --- | --- |
| 1–4 | 134.7 | 375.9 |
| 5–8 | 307.6 | 536.9 |
| 9–16 | 378.7 | 614.9 |
| 17–24 | 438.9 | 717.6 |
| 25–32 | 473.2 | 728.5 |
| 33–48 | 603.8 | 1,016.2 |
| 49–64 | 835.4 | 1,291.9 |

metal-bench `--mode lm`, rows=22, calls=8, on the M1:

| weights | mean / p50 / p95 per call | keystroke | max \|gpu−cpu\| logp | argmax flips |
| --- | --- | --- | --- | --- |
| int8 (auto) | 46.48 / 46.43 / 47.42 | 371.9 | 9.28e-2 | 0 |
| f16 | 53.61 / 53.47 / 55.66 | 428.9 | 9.27e-2 | 0 |

Same machine for reference: ORT int8 CPU 26.1 ms/call at 1 thread, 15.9
at 4; the fused paged-attention kernel alone 0.907 ms; the dispatch
floor 0.159 ms.

So the whole-step path is ~1.8× **slower** than ORT on one M1 CPU core:
2.61 GFLOP in 46 ms is ~56 GFLOP/s against a ~2.6 TFLOP/s fp32 part,
and the ~85 MB of int8 weights should stream in ~1.3 ms. It is the
matmul kernels — not dispatch (16 × 0.16 ≈ 2.5 ms), not attention
(0.9 ms) — that need rewriting as tiled `simdgroup_matrix` GEMMs; the
per-kernel profile leg (`--mode lm --profile`) splits them out.

#### The tiled-GEMM rewrite (this revision)

The matmuls now run as threadgroup-tiled `simdgroup_matrix` kernels
(`lm_mm`, still sixteen dispatches a call): each threadgroup owns a
32×64 output tile, eight simdgroups own 8 rows each and accumulate four
8×8 `simdgroup_float8x8` tiles; int8 weights dequantise into
threadgroup memory per 32-deep K-panel (so a weight element is read
once for all 32 rows — no per-row re-read), the int8 scale lands in the
epilogue, and the fused ops stay — LN as the A-staging prologue,
gelu/bias/residual/QKV-split as the epilogue (`flags` in one scalar
slot). `head_mat` is the same kernel: 22×512 @ 512×41928 logits with
the full-vocab `lse` kept. `LM_MM_NAIVE=1` swaps every `lm_mm` for a
per-element bisection kernel — it is what localised the two bugs below
(the tile machinery was innocent).

Two genuine kernel bugs hid in the rewrite, found by dumping every
buffer of one call (`LM_DUMP_DIR=<dir>` on `--mode lm`, `.bin` files +
numpy recompute):

- **LN stats under-summed.** The quad reduce read `s0 += shuffle_xor(s0,1) + shuffle_xor(s0,2)` — each lane adds itself and two of its three quad-mates, so every `mean`/`rstd` missed a quarter of the row. Layer-0 `next_k` was off ~0.2 system-wide (5–15% of each value) and the error compounded to ~4 by layer 11 → the 7.4 logp diff. Chaining the xor butterfly (`s0 += sh(s0,1); s0 += sh(s0,2)`) fixed L0 to 3e-6.
- **`q·k` covered 2 of 64 dims.** `lm_attn` gave each lane positions `{d2, d2+1}` mod 64 and dotted q·k over only that lane's two dims — a 2-dim partial with no cross-lane reduction, so all scores were wrong from the second layer up. Positions now split across lanes (`t = lane + 32*j`), each lane carrying the full 64-dim product.

VM (paravirtual GPU — timings sanity only, the M1 decides): `--mode lm`
rows=22 int8 **9.1e-2 max |gpu−cpu| logp, 0 argmax flips** at ~10 ms/call
(was 35.9 wrong); f16 **9.3e-2, 0 flips**, ~9.7 ms. The
`--reference-first` replay on six dev records: **0 mismatches**.
`--mode lm --profile` prints per-kernel-type GPU ms (counter sampling at
dispatch boundaries where the device offers a `timestamp` counter set,
one command buffer per dispatch as the wall-clock fallback — the 0.16 ms
floor shows inside each row). The VM's read: qkv 7.8 ms / attn 4.6 /
out 6.3 / ffn0 7.2 / ffn2 11.3 / head_mat 2.8 / head_logp 1.5 over
42.7 ms — the M1 rows decide whether 5 ms/call is reachable.

`--pause-at` now pauses once per run (a flag guards the check), not
once per record reaching the threshold.

A gather-kernel bug fixed on the way: `lm_head_logp` wrote each output
through `tid < count` under a 512-thread group, so any ask over 512
candidates (every early keystroke asks all ~7,322 emittable chars) left
`logp` slots at `+0.0` — and `+0.0` wins the argmax. The loop now strides
`i += HID`. It was invisible on every synthetic tape (they ask ≤48) and
only bit on real ones.

Review fixes folded into this revision (the independent review of
89678e6): the pool's scratch generation is per-`advance`-call now — a
rayon-parallel `advance` on one `&CharLm` can no longer share the
`state_*` tensors two callers memcpy into — and `reclaim`'s
`strong_count` check plus tombstone happen inside one lock order with
the memo/dedup acquisition paths, so a page cannot be re-issued to a
claim that raced the check; a concurrency test drives parallel advances
with `Pending` rows and asserts bit-equality with a serial run.
Scratch and pool buffers are zeroed on every backend so a masked tail
can never carry NaN into `probs @ values`; the `page_major=false`
manifest arm is gone (the layout fails to load, as no exporter emits
it); `Cached::order` is bounded by the entry count; and the
LSTM-only pooled-state test is renamed for what it observes.

### The RSS regression, resolved into regions

`vmmap` cannot see a process on this VM (`task_for_pid` is denied), so
the replay walks its own map instead: `mach_vm_region` over
`mach_task_self()` at `tables`/`opened`/`finished` (and at `--pause-at`),
per `(file | user tag)` resident/swapped/dirtied rows — `vmmap
-summary`-equivalent without leaving the process. Both backends, the
40-record product run (`--incremental --lm-cache 4096`, int8 everywhere,
`--reference-first`):

| at finish | CPU `--lm-threads 4` | Metal | backing |
| --- | --- | --- | --- |
| `ngram.mmap` | 540 MB | 540 MB | **file — clean, outside `phys_footprint`** |
| `charlm.weights` | 89 MB | 88 MB | file |
| `fill.weights` + `context.weights` | 195 MB | 195 MB | file |
| MALLOC large + small | **2,051 MB** | 742 MB | **anonymous** |
| tag 100 (IOKit) | — | 317 MB | anonymous: the MTL page pool + packed weights |
| footprint (`proc_pid_rusage`) | **1,900 MB** | **963 MB** | the honest target |
| peak RSS | 5,130 MB | 1,936 MB | RSS counts the clean pages |

`phys_footprint` counts the anonymous heap, not the clean file pages —
which is why the M1's 4.04 GB cannot be the mapped trigram. The
anonymous side is ORT: repacked weight copies plus the sessions' per-
call arenas (`session.disable_prepacking` removes ~300 MB of it but
slows the step 2.7×, so it stays on), the page pool's ~200 MB, and the
state/scratch working set. The `metal` backend removes the ORT heap
entirely — the LM sessions are gone, the pool is one `MTLBuffer` —
which is the whole ~1 GB delta: **1,900 → 963 MB**, inside the <1 GB
target on this VM; the M1's GPU accounting decides the rest.

Why the M1's RSS *rose* 2.15 → 2.91 GB with the mmap'd trigram: file
pages are clean but still count in `ru_maxrss`, and under the mini's
2.9 GB of swap pressure the same live anonymous pages plus compression
is the whole 4.04 GB footprint — `swapped_mb` in the breakdown reads it
directly on the next M1 run.

Thread accounting: the summary prints `thread counts
backend=… route_a_threads=4 lm_threads=N` — the product row keeps
`lm_threads=1`; a `--lm-threads 4` comparison row is recorded beside it
on the next M1 run.

### RSS accounting: one session, not two

The harness's interleaved default keeps two sessions resident
(measured + all-off baseline) — on the 8 GB mini that is what pushed
peak RSS to 3.0 GB and inflated bookkeeping to 164 ms/keystroke
against the VM's 42.4 under 2.7 GB of swap. `--reference-first` runs
the reference pass to completion first, drops its session, then runs
the timed incremental pass: peak RSS is one session's and the sys
share stops counting two sessions' pages. The equality gate is
unchanged — reference tops are recorded per record and compared the
same way.

### The product's RSS budget — where 1.48 GB goes

The M1's single-session peak at the shipping configuration is 1.48
GB, and the budget splits by backing store — file-backed pages are
shared and evictable under pressure, anonymous pages are not:

| component | resident cost | backing |
| --- | --- | --- |
| kn-trigram (`ngram.bin`, 657 MB on disk) | ~650 MB | **anonymous heap** — `fs::read` + postcard-deserialized into owned tables; not file-backed, not evictable, the single largest item |
| char-LM int8 weights | 2 × 148 MB mapped | `mmap` — evictable, but the export writes the *same bytes twice* (`charlm.weights` ≡ `prefill.weights`, identical MD5) as two separate files, so both map separately: ~296 MB mapped where 148 would do |
| route-A int8 towers | ~312 MB on disk | `mmap` — shared, evictable |
| page pool (`--lm-cache 4096`) | 192 MB | anonymous — 4,096 × 49,152 B, one `[12,8,64]` fp32 KV slice per node |
| pending scratch generation | ~60–200 MB | anonymous — `[slots≤129, 12, 8, ≤63, 64]` fp32, sized by pending rows × max depth; cached and reused across steps |
| beam states, probs, paths, ORT arenas | ~80–150 MB | anonymous |

Anonymous-resident ≈ 0.9–1.1 GB of the 1.48; the rest is hot
file-backed pages. On an 8 GB Mac the reducible items in order:

1. **The trigram (≈650 MB anonymous).** `NgramModel::from_bytes`
   copies the file into a deserialized heap structure; a memory-mapped
   or zero-copy table drops it to file-backed (or a smaller quantized
   table drops it outright). This is the difference between the
   product fitting in 1.5 GB and not.
2. **The duplicated LM weights file (~148 MB mapped).** Both graphs
   name separate byte-identical weight files; one shared file halves
   the mapped footprint.
3. The pool cap (`--lm-cache`) and the pending scratch bound — real
   money, but at 4096 the pool is already the stated 192 MB.

## Reproduce

```sh
cargo build --release -p ime-cli            # or --features gpu-coreml
ime-cli replay \
    --route-a ~/ml-ime-assets/route-a-export/fp32 \
    --emittable ~/ml-ime-assets/route-a-assets-v2/emittable.txt \
    --eval-set ~/ml-ime-assets/route-a-assets-v2/eval3.jsonl \
    --slice dev --records 100 [--incremental] [--backend coreml] \
    --out replay.json

# the full engine, on a box with the exports (resident-pages layout):
ime-cli replay \
    --route-a ~/ml-ime-assets/route-a-export/int8 --route-a-threads 4 \
    --lm /path/to/charlm-pages-int8 \
    [--model /path/to/ngram.mmap]          # `ime-cli pack-ngram` converts
                                          # a postcard ngram.bin once
    --emittable ~/ml-ime-assets/route-a-assets-v2/emittable.txt \
    --eval-set ~/ml-ime-assets/route-a-assets-v2/eval3.jsonl \
    --slice dev --records 40 --incremental \
    [--lm-threads N] [--lm-cache ROWS] [--max-paths N] \
    [--backend coreml] \
    --out replay.json

# the Metal microbenchmark (candle, no ONNX) — builds in the
# metal-bench/ crate beside the repo, not inside it:
metal-bench --mode candle --device metal [--q8] [--rows 22 --calls 8]
metal-bench --mode candle --device cpu [--q8]   # threads: RAYON_NUM_THREADS
metal-bench --mode ort --lm /path/to/charlm-pages-int8 [--rows 22]
metal-bench --mode empty                        # the per-dispatch floor
metal-bench --mode fused                        # the fused paged-attn kernel
# the Metal step in ime-lm, CPU-reference compared + per-kernel timing:
metal-bench --mode lm --lm /path/to/charlm-pages-int8 [--profile]
    [--metal-weights auto|f16] [--rows 22 --calls 8]
# debug legs: LM_DUMP_DIR=<dir> dumps one call's buffers as .bin;
#   LM_ONLY_OPS=n truncates the dispatch list; LM_MM_NAIVE=1 swaps the
#   tiled GEMM for the per-element bisection kernel

# the trigram, memory-mapped (postcard .bin files still load unchanged):
ime-cli pack-ngram --model ngram.bin --out ngram.mmap
```

The LM directory must be a `resident-pages` export (`mlime export
char-lm <ckpt> [--quantize int8]` on this revision) — the loader refuses
the removed `resident-candidates` layout. The binaries sent for the M1
run: `ime-cli-aarch64-apple-darwin`,
`ime-cli-aarch64-apple-darwin-coreml` and
`metal-bench-aarch64-apple-darwin`; the exports ship as
`charlm-resident-pages.tar.gz` (int8 and fp32).

#### The M1 verdict on 144c01a (verbatim)

metal-bench `--mode lm --calls 2 --iters 3` row sweep on the M1 —
**the tiled GEMM is wrong past one 32-row tile**:

| rows | ms/call | max \|gpu−cpu\| logp | argmax flips |
| --- | --- | --- | --- |
| 1 | 9.55 | 1.07e-2 | 0 |
| 3 / 8 / 9 / 16 | 12.0–13.7 | ≤7.17e-2 | 0 |
| 22 / 31 / 32 | 13.5–14.6 | ≤7.77e-2 | 0 |
| 33 | 21.1 | 3.99e32 | 10 |
| 40 | 19.5 | 1.48e28 | 74 |
| 64 | 21.4 | 7.92e28 | 169 |
| 65 | 26.5 | 2.07e1 | 176 |
| 100 | 35.0 | 1.29e1 | 375 |

Replay `--backend metal --reference-first`, 40 dev records: **40
mismatches**, nonsense tails (e.g. `zouzheqiaohehe`: expected
`走㙷㒅吖哦㔠㭘`, got `走着气啊噢㔠㔠`) — the reference itself (`cache_rows=0`)
advances >32 rows a call, so both sides carry the garbage.

Speed at rows=22: int8 **13.87 ms/call** (p95 15.46), f16 14.02 —
3.4× faster than fb578e0 and 1.9× faster than ORT int8 on one core.
Replay (indicative only until correct): median **132.0 ms**, p95 275.7,
max 489.2; `advance` 9.04 ms/call = 74.4 ms/keystroke; fill 25.3,
emission 26.3 ms/keystroke; peak footprint 1.10 GB, max RSS 1.34 GB.

Per-kernel profile (one call, serialised, 29.0 ms total): embed 0.34,
qkv 5.04, attn 2.95, out 3.80, ffn0 5.33, ffn2 6.76, head_proj 0.40,
head_mat 4.05, head_logp 0.38. The shape of the cost: rows=1 already
costs 9.5 ms and rows=32 14.6 ms — latency-bound, not arithmetic- or
bandwidth-bound (the ~85 MB of int8 weights stream in ~1.3 ms at M1
bandwidth).

#### The rows>32 fix and the fused layer tail (this revision)

Root cause: the tile split rows across `gr = sg * 8` — eight
simdgroups × 8 rows = 64 stripes inside a 32-row tile. Simdgroups 4–7
`simdgroup_load`ed past `tg_a`'s 1024 floats into `tg_b` (the staged
weight chunk) and the epilogue wrote `gm = mo + gr + i ≥ mo+32` into the
*next* tile's rows — guarded only by `gm < M`, so it fired exactly when
M > 32. The paravirtual GPU does show the same signature scaled down
(rows=33: 6.1e0 over 8 flips; the buffer dump put row 32's `next_k` off
6.1e0 at layer 0 with rows 0–31 clean), so the fix was proven on the
VM, not blind.

The fix re-decomposes the tile as 4 row-strips × 2 column-halves:
`tr = (sg & 3) * 8`, `tc = (sg >> 2) * 32`, four `simdgroup_float8x8`
accumulators a group, and a single-pass epilogue that stages the four
tiles back through `tg_b` and writes rows `mo + tr + i`, columns
`no + tc + r*8 + j`. Post-fix sweep on the VM (gate 0.5 — the bug
produced 6e0–4e32; the noise ceiling observed is ≤0.25): rows 1→
8.4e-3/0 flips, 31/32/33→8.2e-2/0, 64→1.5e-1/4, 65→2.5e-1/4, 100→
2.5e-1/12 — and `LM_MM_NAIVE=1` at rows=64 returns the identical
1.503e-1/4, so the residual diffs are int8-vs-fp32 accumulation noise,
not structure; f16 shows the same shape.

The regression test that would have caught it now exists twice:
`metal-bench --mode lm --sweep` runs rows {1, 31, 32, 33, 64, 65, 100}
under a 0.5 diff gate (the leg the M1 runs), and
`crates/ime-lm/tests/metal.rs` runs the same sweep under `cargo test`
on the VM — it fails on the pre-fix kernel.

Latency: ~9 ms of fixed cost ≈ 63 serial dispatch overheads at the
0.15–0.46 ms the M1 charges per encode — not occupancy: at rows ≤ 32
the GEMMs already launch 8–655 threadgroups against the M1's 8 GPU
cores (qkv N=1536→24, out/ffn2 N=512→8, ffn0 N=2048→32, head_mat
N=41928→655), and split-K for the small-M shape would multiply the
dispatch count or need atomics — the wrong direction. The step now runs
**28 dispatches**: `embed`, then per layer `qkv` (the tiled mm) plus
`tail` — `lm_tail`, one 8-row tile per threadgroup at 256 threads:
serial-head attention with the identical math to `lm_attn`, a barrier,
the out-projection + residual staged through threadgroup memory, LN2
stats via `simd_sum`, then the FFN over 64-wide u-blocks with per-
thread accumulators — and finally head-proj / head-mat / head-logp.
`LM_NO_TAIL=1` restores the unfused 63-dispatch list for bisection.
Each tail threadgroup reads its weight slice once per 8 rows; the
paravirtual GPU cannot price it, so the M1's `--mode lm` leg decides
what the dispatch saving is worth.

The fused path end-to-end on the VM (`--backend metal
--reference-first`, 40 dev records, 1,036 keystrokes): **0
mismatches**, median 138.7 ms, p95 288.1, max 530.1; `advance`
12.37 ms/call over 8,258 calls (unfused: 16.25); bookkeeping 2.0
ms/keystroke; hit rate 81.0%; footprint 967 MB, peak RSS 2,024 MB.
The VM's own `--mode lm` verdicts stay diff-identical to the unfused
path at every sweep row — 8.4e-3/0 at rows=1, 8.2e-2/0 at 31–33,
1.5e-1/4 at 64, 2.5e-1/4 at 65, 2.5e-1/12 at 100 — so the fused kernel
changes dispatch count, not numerics. The M1 prices both.

#### The M1 verdict on 181c196 (verbatim)

Correct: `--sweep` all 7 row counts within 0.5 (rows 1: 6.1e-2/0;
31–33: 9.28e-2/0; 64/65: 1.23e-1/32 flips; 100: 1.23e-1/96 flips);
replay `--backend metal --reference-first`, 40 dev records: **0
mismatches**.

Speed at rows=22:

| path | ms/call mean (p95) | rows=1 |
| --- | --- | --- |
| fused tail (default, 28 dispatches) int8 | 29.87 (30.25) | 13.16 |
| fused tail f16 | 38.51 (38.90) | — |
| `LM_NO_TAIL=1` (63 dispatches) int8 | **10.78 (11.65)** | 8.68 |
| 144c01a (63 dispatches, broken tiles) | 13.87 | 9.55 |

Profile, fused: embed 0.43, qkv 3.66, **tail 29.34 (82%)**, head_proj
0.28, head_mat 1.69, head_logp 0.35 (total 35.8). Profile,
`LM_NO_TAIL=1`: embed 0.35, qkv 4.55, attn 3.24, out 3.48, ffn0 4.41,
ffn2 6.01, head_proj 0.35, head_mat 2.66, head_logp 0.31 (total 25.3,
serialised).

Replay (fused default): median 242.8, p95 587.3, max 1083.2 ms;
`advance` 21.3 ms/call, 169.6 ms/keystroke; fill 38.0, emission 39.8
ms/keystroke; footprint 1.14 GB, max RSS 1.47 GB.

The tail kernel ran one 8-row tile per threadgroup — 3 threadgroups at
rows=22 against the M1's 8 GPU cores, a 10× parallelism loss traded
for 35 fewer dispatches. The M1's verdict: **fusion must keep each
GEMM's output-column tile count**, so the fused tail is out — the
63-dispatch list is the default and `lm_tail` survives only as the
`LM_TAIL=1` bisection leg.

#### The dispatch floor, attacked (this revision)

Default back to 63 dispatches — `LM_TAIL` flipped from opt-out
(`LM_NO_TAIL`) to opt-in — and the per-call wall is now split three
ways per `advance`: host encode (command buffer + per-op encode),
`commit` → GPU-scheduled, and scheduled → completed, printed by
`--mode lm` as `split per call`. On the VM (paravirtual, sanity only)
encode is ~0.07 ms of a ~14 ms call — nowhere near the
~0.14 ms-per-dispatch floor — so a reusable `MTLIndirectCommandBuffer`
has nothing to fix unless the M1's split disagrees (kernel scalar
arguments would also need a params buffer: ICB has no `setBytes`).

The encoder is now `MTLDispatchType.concurrent` with an explicit
`memoryBarrierWithResources:` over the eight mutable buffers the chain
passes through (`x`, `g_qkv`, `g_ff`, `g_y`, `logits`, `g_sc`,
`next_k`, `next_v`) between every consecutive op — the chain itself is
strictly serial, so what the concurrent encoder buys is each
dispatch's threadgroup scheduling overlapping the previous drain.
`LM_SERIAL_ENC=1` restores the serial encoder for bisection; the
sweep stays diff-identical under it on the VM. Elementwise work
already rides the GEMM prologues/epilogues (LN in, bias/gelu/residual
out), which is the only fusion the M1's verdict allows.

#### The M1 verdict on 9d42f8b (verbatim)

| leg | split (host-encode / commit→scheduled / gpu) | ms/call mean (p95) |
| --- | --- | --- |
| rows=22, concurrent encoder (default) | 0.065 / 0.077 / 10.669 | 10.84 (11.61) |
| rows=22, `LM_SERIAL_ENC=1` | — | 10.85 (11.69) |
| rows=1 | 0.080 / 0.083 / 8.594 | 8.71 (10.69) |

Sweep: all 7 row counts within 0.5 (33: 16.21 ms, 64: 16.93, 65:
21.39, 100: 25.35; flips 0/0/32/32/96 as before). 0 argmax flips at
rows 1 and 22.

So it is neither encoding nor submission — the concurrent encoder
changes nothing: the GPU itself spends 8.6 ms executing the 63 kernels
at rows=1. That is the kernels, not a dispatch floor. From the
181c196 no-tail profile: ffn2 is ~0.50 ms per dispatch for a
3072×768 int8 panel (2.36 MB) — ~5 GB/s against the M1's ~60 GB/s;
ffn0/qkv/out are similar. At M ≤ 32 these GEMMs launch only N/64
threadgroups (12 for out and ffn2, 36 for qkv, 48 for ffn0), and each
walks the whole K serially with a threadgroup barrier per K-panel: a
long dependent chain on a handful of threadgroups. Occupancy is
exactly the problem at this shape.

#### The M1 verdict on d4e902f (verbatim) — the first `lm_gv` is a large regression

| leg | gpu ms/call | mean (p95) |
| --- | --- | --- |
| rows=22 | 51.02 | 51.19 (51.56) |
| rows=1 | 7.93 | 8.05 (9.12) |

Sweep (all within 0.5): rows 1 7.77 ms; 31 70.86; 32 73.06; 33 16.02
(tiled kernel); 64 16.94; 65 21.40; 100 25.36.

Profile rows=22 (GB/s = weight bytes ÷ GPU ms): embed 0.29 (0.9), qkv
11.50 (0.8), attn 3.29, out 6.21 (0.5), ffn0 14.75 (0.9), ffn2 14.14
(0.9), head_proj 0.55 (1.9), head_mat 14.99 (1.4), head_logp 0.41;
total 66.1.

Profile rows=1: embed 0.22 (1.2), qkv 3.77 (2.5), attn 2.41, out 2.50
(1.3), ffn0 4.06 (3.1), ffn2 4.03 (3.1), head_proj 0.23 (4.6),
head_mat 2.23 (9.6), head_logp 0.25; total 19.7.

0.5-3 GB/s against ~60 GB/s: the kernel is transaction-bound — a lane
that owns one column and walks K reads one int8 per lane per step (32
bytes per simdgroup load), and 22 row accumulators per lane almost
certainly spills. Reference on the same M1: Accelerate `cblas_sgemm`
(AMX) at M=22 — 768×2304 0.122 ms, 768×768 0.027, 768×3072 0.238,
3072×768 0.196, 768×8000 0.80 — ~7-8 ms per call in fp32, so the GPU
must beat ~8 ms to be worth it.

#### The bandwidth rewrite (this revision)

Two pieces.

`--mode floor` — the in-encoder empty-dispatch floor: N ∈ {1, 16, 63}
dependent dispatches inside ONE command buffer and ONE encoder, the
serial encoder versus `MTLDispatchType.concurrent` with
`memoryBarrierWithResources:` between dispatches; scheduled→completed
per dispatch. The earlier floor priced one command buffer per
dispatch; this prices the per-dispatch cost when the whole step shares
an encoder.

`lm_gv` rewritten for bandwidth, int8 only. Every matmul weight table
is repacked at load into `[N, K]` (K contiguous per output column —
`nk_repacked`, a one-time CPU transpose, `wg_*` slabs alongside the
`[K, N]` ones; only when K is a multiple of 16 — all of ours are). A
threadgroup still covers 32 columns and splits K four ways across its
simdgroups, but now a lane owns one column and reads its K-slice in
16-byte `uint4` chunks (four `char4` words dequantised in registers),
then loops the M rows inside the chunk: one `float4` activation load
per row per four k's — broadcast across the simdgroup's lanes, which
walk the same (m, k) window — and a per-row fp32 accumulator in
registers. Each weight byte is fetched once and reused
for all M rows; the four K-partials still reduce through the 16 KB
`tg_part` and the shared epilogue (scale once, bias/gelu/residual/QKV
split) is unchanged. Dispatch is N/32 threadgroups of 128 threads:
all layers of qkv/out/ffn0/ffn2 plus `head_mat` run `lm_gv` at rows ≤
32; the fp32 head projection and non-int8 encodings stay tiled, picked
by `spec.wg.is_some()` at encode. `LM_NO_GV=1` still bisects; the old
per-byte `lm_gv` design is gone.

Verified on the VM: sweep all 7 row counts within 0.5 (gv at ≤32 —
identical 8.2e-2/0 diffs to the tiled path), the metal test passes,
`--profile` reports per-GEMM GB/s (weight bytes ÷ GPU ms; "—" for
attn/logp).

#### The M1 verdict on c302698 (verbatim) — the repacked `lm_gv` changed nothing

`--mode floor`, GPU ms per dispatch inside ONE encoder: n=1 serial
0.1493 / concurrent 0.1608; n=16 0.0146 / 0.0171; n=63 **0.0056 /
0.0058**. The in-encoder floor is ~6 µs, so 63 dispatches cost ~0.35
ms: dispatch count is not the problem.

`--mode lm`: rows=22 gpu 50.64 ms (mean 50.82, p95 51.17); rows=1 gpu
7.40 (mean 7.51, p95 8.49). Sweep all within 0.5 (1: 7.55, 31: 70.38,
32: 72.57, 33: 16.10, 64: 16.90, 65: 21.42, 100: 25.28).

Profile rows=22: qkv 10.82 (0.9 GB/s), attn 2.39, out 5.13 (0.6), ffn0
14.00 (0.9), ffn2 13.53 (0.9), head_mat 15.09 (1.4); rows=1: qkv 3.61
(2.6), out 2.54 (1.2), ffn0 3.92 (3.2), ffn2 3.86 (3.3), head_mat 1.99
(10.8).

The rewritten `lm_gv` measured the same as the one before it — the
repacked `[N, K]` path was dispatching (51 ms against the ~11 ms tiled
baseline proves it); the kernel is now bound on the activation side:
every 16 weight bytes come with `M` `float4` activation loads in the
inner loop, so per-lane a-traffic swamps the weight stream and the
22–32 accumulator registers almost certainly still spill.

The M1-verified kernel to replace it (standalone harness, fp16 weights
`[N, K]` K-contiguous, activations fp16 `[24, K]` padded, one
simdgroup per 24×32 output tile = 3×4 accumulators, K split 4 ways
across a threadgroup's simdgroups, NT=4, KS=4):

| K × N | ms | GFLOP/s | GB/s (fp16 W) |
| --- | --- | --- | --- |
| 768 × 2304 (qkv) | 0.0757 | 1029 | 46.8 |
| 768 × 768 (out) | 0.0369 | 704 | 32.0 |
| 768 × 3072 (ffn0) | 0.1004 | 1034 | 47.0 |
| 3072 × 768 (ffn2) | 0.1034 | 1004 | 45.6 |
| 768 × 8000 (head) | 0.2665 | 1014 | 46.1 |

Max abs error vs an fp32 CPU sum of the same fp16 inputs ≤ 5e-6.
Variants tried: KS=1 ~0.09–0.21 ms, KS=2 in between, NT=2/KS=8 ~10%
slower, NT=1 ~50% slower. Per layer ≈ 0.32 ms of GEMM → ~3.8 ms for 12
layers + 0.27 head, so ~5 ms per call with attention and the
elementwise work, vs Accelerate/AMX ~7–8 ms.

#### The staged small-M GEMM (this revision)

`lm_gv` is out; the M1-proven `simdgroup_matrix` kernel is in as the
M ≤ 32 path for qkv, out, ffn0, ffn2 and head_mat — the tiled `lm_mm`
keeps M > 32.

- `lm_stage` (one threadgroup per row, 32 tgs × 128 threads) packs each
  op's activations into the shared `a_stage` buffer as fp16 `[32, K]`
  rows — zero-padded past M — folding the op's LN in when `MM_LN`:
  per-simdgroup `simd_sum` partials of sum/sumsq land in threadgroup
  memory, thread 0 reduces them to mean/rstd, and the row writes
  `half((v − mean) · rstd · w + b)`. The staging is a GPU dispatch —
  host-side staging would read x mid-command-buffer and see stale data.
- `lm_sm` is the ported kernel with `ROWS8` fixed at 4 (one pipeline,
  no function constants): a threadgroup owns a 32-column output tile,
  its four simdgroups split K four ways into four `[32, 32]` partial
  planes in 16 KB of threadgroup memory, and a 128-thread epilogue sums
  them and applies the flags — `MM_SCALE` (new; set only under the
  int8 pack, `S[n]` per-out scale), `MM_BIAS`, `MM_GELU`, `MM_RES`, and
  the `MM_QKV` split (Q to the fused row, K/V strided into `next_*` by
  layer). All of `part`'s reduction layout, loads and epilogue order
  match the proven harness.
- Weights repack at load into `wh_*` slabs — fp16 `[N, K]` with K
  contiguous: under the int8 pack the int8 values are carried as fp16
  (the scale still folds in the epilogue); under f16 the table is
  transposed. Output rows pad to the 32-column tile and the epilogue
  guards `n < N`, which lets `head_mat`'s vocab (41,928 — not a
  multiple of 32) take the same kernel. The fp32 head projection stays
  on the tiled kernel.
- `lm_sm_i8` is the int8 A/B leg (`LM_SM_I8=1`): the same kernel over
  the `[N, K]` int8 repack, each simdgroup cooperatively dequantising a
  32-column × 32-deep panel into its threadgroup slot (two 16-byte
  `uint4` loads per lane → `half4` stores), then `simdgroup_load`ing
  the fp16 tiles from threadgroup memory. `LM_NO_SM=1` bisects back to
  the tiled kernel at every row count.
- Two bugs caught on the VM before shipping: the fp16 repack's
  `w_off` has to stride by 2 bytes per element, not the packed table's
  `w_elem` — layer ≥ 1 was reading weights half a slab early — and the
  `sm_i8` pick needs `rows <= 32` itself or the i8 kernel gets the
  tiled bindings at M > 32.
- `--profile` reports the repack's actual element size per op, so each
  GEMM's GB/s column prices the table the dispatch really streams.
- Dispatch count is now 113 per call at M ≤ 32 (embed + per layer
  stage/qkv/attn/stage/out/stage/ffn0/stage/ffn2 = 9×12 + stage +
  head_mat + head_proj + head_logp) — the stage kernels make the extra
  dispatches; the in-encoder floor of ~6 µs prices them at ~0.3 ms
  total.

VM verification (paravirtual GPU — sanity only): `--sweep` all 7 row
counts within 0.5 under both `lm_sm` and `LM_SM_I8=1` (fp16-level
residuals ~8e-2, 0 flips at rows ≤ 33); `cargo test` metal sweep
passes. The M1 legs decide fp16-vs-int8 and the ≤5 ms bar.

#### The M1 verdict on 7b7fd6a (verbatim) — the MMA kernel works

`--mode lm` (gpu / mean / p95 ms per call, max |gpu−cpu| logp, flips):

- rows=22 fp16: 8.028 / 8.214 / 8.746, 9.20e-2, 0
- rows=22 `LM_SM_I8=1`: 8.880 / 9.065 / 9.677, 9.20e-2, 0
- rows=1 fp16: 7.186 / 7.321 / 7.985; int8: 7.482 / 7.616 / 8.479
- Sweep all within 0.5: 1 7.34, 31 8.63, 32 8.62, 33 16.04 (tiled),
  64 16.96, 65 21.26, 100 25.41.
- Truncated step at rows=1 (`LM_ONLY_OPS=n`, gpu ms): n=1 0.484, 2
  0.542, 5 0.798, 10 1.190, 20 1.947, 40 3.585, 60 5.026, 80 7.111,
  100 7.174, 113 7.108 → ~80–90 µs per dispatch at rows=1, flat after
  op ~80. The step is latency-per-dispatch bound: the empty in-encoder
  floor is 6 µs; a real small kernel costs ~80 µs of launch + drain +
  memory latency.

**fp16 wins — the int8 sm leg is deleted.**

Replay `--backend metal --reference-first`, 40 dev records: **0
mismatches**; median 109.9 ms, p95 243.9, max 398.6. By typed length
(median/p95): 1–4 29.2/102.1, 5–8 84.7/164.7, 9–16 104.1/178.8, 17–24
118.7/192.0, 25–32 141.0/211.3, 33–48 192.5/281.3, 49–64 247.1/372.9,
65+ 259.2/285.7. Per keystroke: advance 55.4 ms (6.95 ms/call),
bookkeeping 3.1, fill 23.9, emission 24.8, decode 94.6 total, segment
0.16, context 0.2. Memory: max RSS 1.45 GB, peak footprint 1.31 GB
(was 1.03–1.14 GB; the fp16 repack is the +170 MB). The starting point
on this machine was median 267 / p95 916 ms.

#### The M1 verdict on 2823e03 (verbatim) — correct, no faster

- `--mode lm`: rows=22 gpu 8.290 (mean 8.464, p95 9.016), 9.41e-2,
  0 flips; rows=1 gpu 7.054 (mean 7.161, p95 7.822). Sweep all within
  0.5: 1 7.20, 31 8.97, 32 8.98, 33 13.35, 64 15.04, 65 18.44,
  100 22.61.
- Replay `--reference-first`, 40 dev records: **0 mismatches**; median
  115.0, p95 259.6, max 415.2 ms. By typed (median/p95): 1–4
  28.0/104.3, 5–8 85.1/177.2, 9–16 108.7/188.9, 17–24 122.7/209.0,
  25–32 150.6/228.4, 33–48 206.5/302.0, 49–64 257.3/408.9, 65+
  262.5/292.5. Per keystroke: advance 54.0 (6.77 ms/call), bookkeeping
  2.7, fill 25.3, emission 26.1, decode 99.8. Beam split totals:
  expand 41,094.6 ms (of which transition score 21,428.5), select
  3,466.8, finish 374.6, expanded 304,511,451 pairs = score calls →
  ~294k scored pairs and ~40 ms of expansion per keystroke, ~70 ns per
  score.
- Memory: max RSS 1.55 GB, peak footprint 1.21 GB; at "finished"
  resident 1,511 MB, swapped 291, footprint 1,209 — still over 1 GB.
- The step is compute-bound on padding: 113 → 64 dispatches changed
  nothing (rows=1 7.05 vs 7.19) and rows=1 ≈ rows=22 because every
  GEMM computes 32 padded rows (ROWS8=4): 32 × 2 × ~85M ≈ 5.4 GFLOP at
  ~1 TFLOP/s ≈ 5.4 ms floor.
- The beam expansion recomputes the whole sentence every keystroke:
  median grows 28 → 257 ms across typed bands while LM misses stay
  ~180/keystroke.

## What is next

- This revision's four items, in order: (a) `ROWS8 = ceil(M/8)` via
  function constants (pipelines 1..4, row-block stride ROWS8×8) so
  rows=22 pays 24 rows (−25%) and a GEMV kernel for M ≤ 2 (lane per
  column, 16-byte K chunks, broadcast activations — the user's harness
  measured 768×2304 at M=1 in 0.030 ms, ~59 GB/s); (b) incremental
  decode — keep each path's beam frontier per position, resume
  expansion from the longest unchanged prefix, bit-identical required
  plus a keystroke-by-keystroke equivalence test; (c)
  `Transition::score_many` — the trigram does one sorted merge /
  bigram-row lookup per beam, the LM reads its logp row by index;
  (d) resident memory under 1 GB — read the `finished` region
  breakdown and cut the largest.

- fp16 won the A/B (~9% at rows=22): `lm_sm_i8`, `LM_SM_I8`, and the
  whole `wg_*`/`nk_repacked` int8 `[N, K]` repack are deleted.
- **Dispatch count — Option C shipped**: `lm_stage` and the mid-encoder
  staging dispatches are gone. Producers write the raw fp16 x into
  `x16` plus per-row (Σx, Σx²) into an atomic stats buffer `st`
  (`MM_STA`); consumers apply LN algebraically in their epilogue —
  `LN(x)[k]·W[n,k]` summed = `rstd·(raw[n] − mean·c1[n]) + c2[n]` where
  `raw = Σ x·W'`, `W'[n,k] = lnw[k]·W[n,k]` folded into the repack,
  `c1[n] = Σ lnw[k]·W[n,k]`, `c2[n] = Σ lnb[k]·W[n,k]`, applied before
  `MM_SCALE` so int8 stays in int8-units (`cc` slabs `[L,2,N]` fp32 at
  buffer 15, `layer` param carries the stats slot on non-QKV ops).
  **64 dispatches/call** (1 embed + 12×5 + 3 head). VM rows=1 GPU
  3.49 ms vs ~7.1 at 113 dispatches — the M1 decides at ~85 µs each.
- **Decode breakdown in the replay summary**: `Breakdown` in beam.rs
  (`decode_many_stats`, `Stages::breakdown`); `Timed::score` counts
  `score_calls`/`score_ns`; replay prints "beam split" (expand /
  of-which-score / select / finish / expanded pairs). VM totals over
  the 40-record run: expand 36.8 ms/keystroke of which
  `transition.score` 20.6 (~294k pairs/keystroke), select 2.3,
  finish 0.1 — the unaccounted ~36 ms (M1) is expansion, mostly the
  per-pair transition score + the relax hashmap.
- **Two cuts applied**: (a) `emission.score` is beam-independent —
  hoisted to one pass over `allowed` per position (~16× fewer calls);
  (b) the guard maps `first_of`/`first_char` build only when
  `guarded`/`gap`-branch actually runs.
- **Memory**: the int8 `[K, N]` slabs (`w_qkv`/`w_out`/`w_ff0`/`w_ff2`)
  become 4-byte dummies whenever the small-M kernel owns every matmul
  — `sm = !naive && !no_sm && !tail && spec.wh.is_some()` at ALL M
  (grid `tgid.y` covers 32-row blocks) — and `advise_cold`
  `madvise(MADV_DONTNEED)`s the i8 mmap ranges after repack. VM
  footprint at `opened` 768 MB (was ~1,050 at 7b7fd6a).
- **Fill measured first**: `fill_stats` (calls/tokens) + per-call ms in
  the stage line. VM: 9.4–9.7 ms/call at ~75 avg tokens — ORT
  per-forward overhead dominates at that shape, and the call is
  already one batched `session.run` per keystroke (no batching
  headroom) → the Metal GEMM port is the follow-up the numbers point
  at (~64 dispatches ≈ 5.4 ms at M1 per-dispatch cost vs 23.9 ms ORT).
- The M1 re-legs: `--mode lm --sweep`, `--rows 22`/`--rows 1` with
  `--profile`, then the `--reference-first` replay with the beam split
  and the memory breakdown. Targets ≤4 ms/call rows=22, ≤3 ms rows=1,
  footprint back under 1 GB.
- The Rust-side levers that remain are the page-major scratch transpose
  (~30 ms a keystroke inside `advance`) and whatever the M1's own
  samply run names inside `bookkeeping` (157.7 ms there against 42.4
  here — a profile on the mini decides which part is parity and which
  is M1-specific).
- #84 wraps `Session` in the C ABI; the session is already generic over
  the engine for #88's end-to-end net.

### The post-2823e03 round (ROWS8, GEMV, `score_many`, incremental decode)

The M1 verdict on 2823e03 said the LM step is **compute-bound on padding**:
every GEMM computes 32 padded rows (`ROWS8` fixed at 4), so rows=1 costs
rows=32's ~5.4 GFLOP. This round ships the fixes in code; the M1 legs
decide each.

- **`ROWS8 = ceil(M/8)`** — `p_sm` is now `[ComputePipelineState; 4]`
  over the `SM_R8` function constant; the encode picks `r8 =
  rows.min(32).div_ceil(8)` per call (`p_sm[r8 - 1]`, grid
  `(ceil(N/32), ceil(M/(r8*8)), 1)`). Arrays stay at max `[4]` —
  function constants cannot size arrays — and loops bound by the
  constant. rows=22 now pads to 24 (−25% GEMM work).
- **GEMV for M ≤ 2** — `lm_smv`: one lane per output column, `uint4`
  (16-byte) weight loads walking K in `KS=4` slices across a
  threadgroup's simdgroups, `dot(float4, float4)` against the staged
  fp16 activations, `simd_sum` reduction, `live = n < N` gating with
  all lanes in the sum. rows≤2 selects `p_smv` (grid `ceil(N/32)`,
  tg-mem 1024 B). Expected rows=1 → ~1–2 ms on the M1.
- **`Transition::score_many`** — `score_many(state, weight, allowed,
  out)` with accumulate semantics (`out[i] += weight·score`), so
  `Both` composes alloc-free as two weighted passes — bit-identical at
  weight=1. The trigram keeps one sorted-merge `Row` cursor per
  context (bigram `partition_point` once per beam then adjacent-entry
  walk — ~22 cold mmap page touches → ~2); the char LM keeps the
  default per-candidate `score` (its `ids` map is not monotonic; the
  binary search over ~30–60 hot `scores.ids` entries already is "logp
  row by index"). `Timed::score_many` counts `score_calls +=
  allowed.len()`.
- **Incremental decode** — `decode_incremental` +
  `Incremental<T::State>` (a session's `frontier`): per-path
  `PathCache { signature, context, start, levels, guarded }` where
  `signature[k] = PosSig { allowed, emitted }` (emitted compared by
  `to_bits`) and `levels[k]` is the beam frontier surviving position
  `k` with states `compact`ed. Resume = longest matching signature
  prefix ∧ same context ∧ same `BeamOptions`; a resuming worker is
  seeded with `history = levels[..p]` candidates, `latest =
  levels[p-1]`, `guarded = guarded[p-1]`, `start = cached.start`, and
  the sink keeps the matched prefix before collecting fresh levels.
  `Session::refresh` calls it unconditionally; the frontier clears on
  `reset()` and at both early exits (`pieces.is_empty()`,
  undecodable).
- **Bit-identity test** — `incremental_decode_matches_a_fresh_decode_
  every_keystroke` decodes `zhonghuarenmingongheguo` keystroke by
  keystroke through one shared `Incremental` vs a fresh `decode_many`,
  asserting `chars`/`path`/`score`/`steps` bit-equal every keystroke
  and `resumed > 0`; `shifted_emissions_invalidate_every_position`
  bumps a salt each keystroke and asserts `resumed == 0` with the same
  bit-equality — the two tests pin both directions.
- **The M1-relevant finding (VM, 10-record replay)**:
  `resumed = 0` — under the fill tower the mechanism cannot fire.
  Miss attribution (`resume_blocked_*` in the beam split): emitted
  1454 : allowed 137 : context 0 — the tower's attention rescored
  earlier positions' emissions bitwise in ~91% of prefix ends, so the
  emitted half of the signature is the hard gate. Resumed frontiers
  would need candidate scores minus their emission terms, which
  changes the survivor SETS, not just the scores — no subset
  resumption stays bit-identical. The design is correct for
  position-local emissions (the `Uniform` leg resumes in the test)
  and dead under attention.
- **Dormancy** — once a call resumed nothing *and* an emit miss
  occurred, `Incremental.dormant` stops all collection: no sink, no
  signature probes, `paths` cleared. The frontier machinery then
  costs a branch per keystroke. Verified: emit misses stop at 22
  (dormant after ~3 keystrokes), `resumed=0`, still 0 mismatches.
- **Memory, VM 'finished' (10 records, Metal)**: footprint **829 MB**
  (was 1,183 pre-round); resident 1,679 / peak RSS 1,594. Regions:
  tag 100 (MTL buffers — fp16 repack ~170 + resident pool
  `cache_rows×L×H×4B×2` ≈ 300 at `--lm-cache 4096`) 394, MALLOC
  small 353, MALLOC large 153; file-backed outside `phys_footprint`:
  ngram.mmap 409, fill.weights 106, context.weights 89,
  charlm.weights 62. If the M1 lands >1 GB the next lever is
  `--lm-cache` (halving the pool ≈ −110 MB) — an arg, not a default
  to change silently.

## The post-5f6673e M1 verdict + the e2e pivot

### 5f6673e on the M1 (recorded verbatim)

- `--mode lm` gpu ms per call: rows=22 **7.247**, rows=8 **4.494**,
  rows=2 **4.840** (GEMV), rows=1 **4.371** (all 0 flips; max diff
  9.48e-2 / 7.17e-2 / 6.78e-2 / 6.39e-2). Sweep all within 0.5:
  1 4.41, 31 8.99, 32 9.00, 33 13.38, 64 15.03, 65 18.41, 100 22.71.
- Replay `--reference-first`, 40 dev records: **0 mismatches**;
  median **109.7**, p95 **237.9**, max **386.3** ms. By typed
  (median/p95): 1-4 28.4/99.3, 5-8 85.2/172.7, 9-16 106.2/189.3,
  17-24 117.3/200.7, 25-32 134.7/211.1, 33-48 181.7/283.2,
  49-64 228.3/330.6, 65+ 225.6/270.4. Per keystroke: advance
  **48.5** (6.09 ms/call), bookkeeping 2.8, fill **23.2**,
  emission 24.1, decode **94.7**. Beam split: expand 40,436
  (score 20,682), select 3,365, finish 1,480, expanded
  304,511,691, **resumed 0** (blocked: allowed 98, emit 82) —
  the finding confirmed on the real run: bidirectional emissions
  invalidate every prefix, nothing resumes.
- Memory: max RSS 1.53 GB, peak footprint **1.21 GB**; at
  "finished" resident 1,529, swapped 269, footprint 1,208.
  Largest resident regions: ngram.mmap 536 (clean file pages,
  outside phys_footprint), Metal tag ~398, MALLOC 299 + 242.

### Decisions

- **Incremental decode removed.** The M1 replay reproduced the VM
  finding — `resumed 0` with allowed/emit misses as the only blockers
  — under a bidirectional emission no frontier can ever be reused
  bit-identically. `decode_incremental`, `Incremental`/`PathCache`/
  `PosSig`/`CacheSink`, the resume counters, the session's frontier,
  and both tests are deleted, not dormanted.
- **Invest only toward the #88 e2e model** (route-A towers + a
  transformer reader decoder; no trigram, no score fusion):
  1. fill/context towers on the Metal fp16 mma GEMM path (today fill
     costs 23.2 ms/keystroke on ORT CPU at 4 threads);
  2. the LM step's small-rows floor — 4.4 ms at rows=1 ≈ ~65 µs × 64
     dispatches — since the e2e reader decoder is the same step;
  3. memory under 1 GB footprint — attribute the 299 + 242 MB
     MALLOC regions and cut what is not needed.
  Trigram-side expansion work (`score_many` merges, the relax map) is
  deliberately out — the trigram leaves the product.

### The fill/context towers on Metal (this round)

`Backend::Metal` now runs both towers on one fused command buffer per
forward, sharing the LM's `simdgroup_matrix` fp16 GEMM (`tw_mm`, the same
tile shape as `lm_sm`: 24- or 32-row tiles, K split four ways across a
threadgroup's simdgroups, partials through threadgroup memory). Embeds
are gathered on the host (word + span + position + bias) into the shared
scratch; LayerNorms run `tw_ln` with the residual fused; self-attention
is `tw_attn` (lane-per-query, 32-query softmax tile per threadgroup);
the fill tower's four cross blocks project the whole context K/V in one
mm (`cross_kv` repacked `[K, 4×1536]` interleaved k|v per block) then run
`tw_xattn` against the fp16 `ckv`; the head runs transform-gelu-LN-
decoder with the candidate mask and log-softmax fused (`tw_emask`,
`tw_lsm`). A `Mutex<Scratch>` owns every working buffer; `Scratch::ensure`
re-allocs only when a bound grows, so paths×width, ctx, vocab and the
attention-probs scratch are all tracked.

Three bugs the fused buffer hid and the ORT-intermediate bisection
(`graph.output` + `TW_DUMP` post-commit stage dumps) isolated:

- `tw_lsm` skipped `red[0]` in the final max reduce — rows whose
  row-max lived in simdgroup 0's columns diverged ±0.43 at the tail
  span positions. `for i in 1..8` → `for i in 0..8`.
- `cross_kv` was packed `[block][input_row][k|v]`; `nk_f16` wants
  `[input_row][all_outputs]` — every K row k was reading block-0 rows
  4k+b. Repacked as per-block `[K][k|v]` then `kv_all[k] = every
  block's row k` — verified against ORT `/Add_31`, `/Add_32` to 0.007.
- `sc.rscale` (the per-row `gate × has_context` buffer each cross
  out-projection reads) was host-written once per block *during
  encode*; in the fused command buffer all four writes land before any
  dispatch executes, so every block applied block 3's gate
  (0.19/−0.13/0.23/−0.22 differ). Fixed by giving every block its own
  `rscale` slice (`block*mp` offset bound at dispatch).

Verification (fp32 export, VM — timings there are sanity-only):

- `mdiff` CPU-vs-Metal on 6 lattice records (incl. all context
  records): every record ≤ **0.014** max |logp diff| after the three
  fixes (was 0.43/2.0/5.5/6.7 at the four stages).
- The fixture test `the_metal_towers_reproduce_the_recorded_table`
  (new, `fixtures/fp32/route-a-metal.json` generated by
  `python/scripts/routea_metal_map.py`) runs both ctx modes at
  `INT8_ATOL` — fp16 activations drift like the int8 leg's.
- Replay `--backend metal --reference-first`, 40 dev records:
  **1 mismatch** — record 477 `nanjiadianshizhenda`, last keystroke,
  `镇达` vs `镇大`. Path 0's `da` column scores `大` −0.8553 vs `达`
  −0.8551 — a **0.0002 margin**, 60× below the fp16 noise floor
  (towers diff ≤0.013 on that record). A bit-exact zero on ties needs
  fp32 throughout; every other record is bit-identical.
- Fill on Metal: 15.4 ms/call on the VM's paravirtual GPU (was ~9.5 on
  ORT CPU there); the M1 leg prices it against 23.2 ms ORT.
- Memory, Metal-only replay (no reference): peak footprint **611 MB**
  on the VM — the MALLOC 299 + 242 regions were the ORT reference
  sessions (`GraphSessions` opens lazily, `check` skipped under Metal),
  gone on the product path. Resident at finished: tag 100 (Metal
  buffers + fp16 repack) 476, fill.weights 474 + context.weights 406
  (clean file pages, outside footprint), MALLOC small+large 131.

### 6581419 on the M1 (recorded verbatim)

- Setup: the export dir carried no `route-a-metal.json`;
  `routea_metal_map.py` fails on the int8 export ("no encoder layers
  found"), works on fp32 → M1 runs used
  `--route-a data/route-a-export/fp32`.
- LM truncation at rows=1 (gpu ms): ops 1 0.519, 5 0.729, 10 0.965,
  20 1.448, 40 2.316, 64 4.548 → ~45 µs/op through op 40, then ~2.2 ms
  for the last 24 ops (the 41,928-vocab head ≈ 64 MB fp16/call is the
  likely bulk).
- Replay `--reference-first`: **0 mismatches**; median 106.2, p95
  224.6, max 328.9; advance 4.98 ms/call (39.7/keystroke), bookkeeping
  2.9, fill 21.6, emission 23.1, decode 93.8. Footprint 2.18 GB (the
  reference ORT sessions open).
- Product config (no reference): median 107.5, p95 226.1, max 426.5.
  By typed (median/p95): 1-4 36.8/126.6, 5-8 88.8/182.3,
  9-16 105.0/194.0, 17-24 110.3/205.1, 25-32 121.5/212.2,
  33-48 171.8/291.1, 49-64 203.1/356.5, 65+ 204.6/346.0. Stage split:
  fill 22,434 ms over 2,072 calls = **10.83 ms/call** at 75.4 avg
  tokens (ORT CPU ~11.6 — no real gain), advance 5.18 ms/call,
  **lm_bookkeeping 34,780 ms = 33.6 ms/keystroke (was 2.8-3.1)**.
  Memory: max RSS 2.07 GB, **peak footprint 2.32 GB**; at "finished"
  resident 1,981, footprint 1,749; `fill.weights` 474 MB resident
  (fp32 mmap stays after the fp16 repack).

### The fixes on top of 6581419 (this round, VM-verified)

1. **The export writes the map.** `mlime export route-a` calls
   `train.metal_map.write_metal_map(out_dir)` for both fp32 and int8
   exports — the fp32 map it emits is byte-identical to the committed
   fixture, and the int8 map carries `int8_weight`/`scale`/`zero_point`
   roles so the int8 export loads on Metal. `routea_metal_map.py` is
   deleted; the loader fails with "route-a-metal.json is missing —
   `mlime export route-a` writes it; re-export this directory".
2. **Bookkeeping regression found and fixed.** With the Metal towers,
   `materialise_pending_f32` was copying the whole pending chain each
   keystroke (33.6 ms/keystroke). The gather now writes
   `page_row[p]=u32::MAX; mask[p]=1` per pending position and the
   kernels test `pages[pg + t] != 0xffffffffu`, so only the tips
   materialise. VM bookkeeping: 1,982 → 977 ms over 40 records; with
   the round's full set, `lm_bookkeeping` 2,271 ms over 1,036
   keystrokes (~2.2 ms/ks — `bk_resolve` 2,003 ms is now the mass).
3. **Tower weight maps dropped.** `Backend::Metal` loads weights via
   `fs::read` into `Vec<u8>` — `Mmap` is `MAP_SHARED`, whose file pages
   cannot be dropped by madvise/msync at all, so there is no map to
   hold resident after the fp16 repack. The M1 'finished' table
   decides the <1 GB footprint.
4. **Fill calls explained + profiled.** 2,072 calls over 1,036
   keystrokes: the `--reference-first` equality baseline shares the
   engine's fill counter; the product path is exactly one call per
   keystroke (1,036 here). Per-kernel leg: `TW_ONLY_OPS` truncates the
   tower encoder like `LM_ONLY_OPS`, and the stage split prints
   `fill_ops` (106 dispatches at 75 tokens). VM fill: **7.69 ms/call**
   at 75.4 avg tokens (ORT ~9.5 on the same box); M1 numbers + per-GEMM
   TFLOP/s pending.
5. **Streaming head replaces the logits buffer.** The 41,928-vocab
   head was reading 64 MB of fp16 per call (~2.2 ms at rows=1 on the
   M1). New path keeps the logp exact: `lm_head_lse` streams the int8
   `[V, K]` table (21.5 MB — the same values the fp16 repack dequantises
   to, so the LSE is bitwise-equivalent-or-better), each of 328
   threadgroups covering 128 vocab rows with an online (max, sumexp)
   per hidden row merged to `part[ntg][M][2]`; `lm_head_cand` merges
   the partials to the exact LSE and re-dots only the call's candidate
   ids. `LM_HEAD_OLD=1` bisects back to the buffer + gather. The
   `[rows, V]` logits buffer is a 4-byte dummy under the streaming
   path. Bug caught by the sweep: at M > 32 the row-block index
   overran the `mx[32]`/`sm[32]` accumulators — the row block is now
   the outer loop with per-block partial writes.

Verification on the VM: `--mode lm` sweep all 7 row counts within 0.5
(max diff 6.4e-2 at rows=1; the >32 flips are the pre-existing
tie-breaks); replay `--backend metal --reference-first`, 40 dev
records, **0 mismatches** — record 477's 0.0002-margin pick now lands
on the same side as ORT.

### 095c0d5 on the M1 (recorded verbatim)

- LM step gpu ms: rows=22 streaming head **36.39** vs `LM_HEAD_OLD=1`
  **7.28**; rows=1 7.39 vs 4.51. Sweep with the streaming head: 1 7.17,
  31 52.37, 32 53.30, 33 59.09, 64 106.27, 65 113.29, 100 171.33 (all
  within 0.5). The streaming head is a 5x regression on the M1.
- Product-config replay (`LM_HEAD_OLD=1`, fp32 export for the Metal
  towers, no `--reference-first`): median 111.9, p95 249.6, max 416.4.
  By typed (median/p95): 1-4 43.1/140.1, 5-8 89.8/186.6,
  9-16 106.9/206.3, 17-24 111.7/216.4, 25-32 137.3/241.1,
  33-48 186.6/312.9, 49-64 228.9/383.1, 65+ 264.4/309.0. Stage split:
  fill 23,726 ms, **fill_calls 2,072** (two calls per keystroke also
  without the reference pass — the 1-call claim did not hold), 11.45
  ms/call, fill_ops 106; advance 45,135; lm_bookkeeping 10,631 (**10.3
  ms/keystroke**, was 2.8 before the Metal towers) with bk_marshal
  7,957, bk_resolve 2,448, bk_mat 7,801 over 224,158. Memory: max RSS
  1.94 GB, footprint 2.26 GB peak / 1,934 at "finished"; resident by
  region: Metal tag 1,249 MB, ngram.mmap 384 (clean), MALLOC 160 + 93.

Conclusion on the M1: the Metal towers give no speed (11.45 ms/call vs
ORT's ~11.6) and cost ~1 GB of footprint, and the streaming head is 5x
slower — both removed for landing (next section).

## The landing (this revision)

For landing, on the M1's verdict: the Metal towers give no speed (11.45
ms/call vs ORT's ~11.6) and cost ~1 GB of footprint; the streaming head
is 5x slower at the product shape (36.39 vs 7.28 ms at rows=22). Both
are removed outright — no dormant paths:

- **Metal towers removed.** `crates/ime-neural/src/metal.rs`, the
  `Backend::Metal` variant, the towers' `towers` field and every towers
  arm, the `Metal`/`MetalUnavailable` errors, the fixture test and the
  `route-a-metal.json` fixture are gone, and `--backend metal` maps the
  towers to `Backend::Cpu` (the LM keeps Metal). The export no longer
  writes the map — `metal_map.py` is deleted, `mlime export route-a`
  emits graphs + manifest only. The towers stay on ONNX Runtime.
- **Streaming head removed.** `lm_head_lse`/`lm_head_cand`, the
  `HeadLse`/`HeadCand` ops, the int8 `[V, K]` head repack, the `part`
  partials buffer and `LM_HEAD_OLD` are gone; the `[rows, V]`
  logits-matmul head (`MmHeadMat` + `HeadLogp`) is the only head.
- **Fill counting moved session-side.** `RouteA::fill_stats` counted
  forwards on the shared `RouteA`, so the replay's baseline session —
  which exists in both modes (as the `--reference-first` pass or the
  interleaved gate) — doubled it: 2,072 calls over 1,036 keystrokes.
  The counter is now `EngineStats::fill_calls`/`fill_tokens` in
  `Fused::scores`; the product path reports exactly **one fill call per
  keystroke** (1,036/1,036), 24.2 ms/call at 75.4 tokens on the VM.
- **Bookkeeping confirmed ≤3 ms/keystroke.** The regression lived in
  the interleaved gate session, not the product one: with the baseline
  alive between records, measured's cache and states pin most of the
  shared page pool, so the gate's own produced states go `Pending` and
  materialise every advance (bk_mat 262,130 — all of it inside the
  gate's passes; measured never pends). `--reference-first` — the
  baseline dropped before the measured pass — reads the product's own
  cost: **lm_bookkeeping 2,254 ms over 1,036 keystrokes ≈ 2.2 ms/ks**,
  bk_resolve 2,010 ms, bk_marshal 62 ms, bk_mat **0**. The sentinel
  gather (`page_row[p]=u32::MAX`) keeps only the tips materialising.
- **The replay's CPU-twin plumbing is gone** — the baseline session
  always decodes on the same `RouteA`; equality is structural, not
  backend-vs-backend.

### The product configuration

`ime-cli replay --route-a <export>/fp32 --route-a-threads 4 --lm
<charlm-pages-int8> --emittable <emittable.txt> --eval-set <eval3.jsonl>
--slice dev --records N --incremental --backend metal`: towers on ORT
(4 intra-op threads), the LM step on Metal (`lm_sm` fp16 simdgroup-
matrix GEMMs, ROWS8 = ceil(M/8), `lm_smv` GEMV at M ≤ 2, algebraic LN
epilogues, 64 dispatches), trigram + LM fused at the fused weight, the
char-prefix state cache at `--lm-cache 4096`, one fill call per
keystroke.

### What was tried and removed, with the M1 numbers

| tried | result on the M1 | fate |
| --- | --- | --- |
| fp16 `simdgroup_matrix` GEMM (`lm_sm`) | 8.03 ms/call at rows=22 vs tiled 51 / Accelerate ~7-8 | **kept** |
| `lm_smv` GEMV at M ≤ 2, ROWS8 = ceil(M/8) | rows=1 4.37 ms (was 7.19 padded to 32) | kept |
| algebraic-LN epilogues (64 dispatches) | rows=1 7.05 vs 7.19 — padding-bound, not dispatch-bound | kept (harmless) |
| int8 sm leg (`lm_sm_i8`) | 8.88 vs 8.03 at rows=22 | removed |
| `lm_gv` lane-per-column (byte loads, then 16-byte loads) | 51.02 / 50.64 ms at rows=22 — transaction-bound | removed |
| fused `lm_tail` layer tail | slower on the M1 (starved threadgroups) | removed |
| incremental decode (resume unchanged prefix) | resumed=0 — the fill tower rescoring moves scores bitwise | removed |
| streaming head (`lm_head_lse`/`lm_head_cand`) | 36.39 vs 7.28 ms at rows=22 — 5x regression | removed |
| Metal towers (the whole fill/context encoder) | 11.45 ms/call vs ORT ~11.6; +~1 GB footprint | removed |

### What bounds the keystroke now (M1, product configuration)

Per keystroke, roughly: the LM ~40-48 ms over ~8 calls at ~5-7 ms/call
(rows ~22; rows=1 is 4.4 — the per-dispatch latency floor, ~80 µs of
launch+drain+latency against a 6 µs empty-dispatch floor); the towers'
fill ~11.5 ms per call on ORT CPU; bookkeeping ~2-3 ms; and beam
expansion ~40 ms, of which the trigram's per-pair `score` is the bulk
(~20 ms over ~294k pairs at ~70 ns) — the trigram leaves with the #88
end-to-end model, and the towers + reader-decoder step it replaces are
the pieces measured above.

### 49ae1ff on the M1 (verbatim): the product configuration regresses, not the reference one

int8 export, towers on ORT CPU, LM on Metal. Sweep: all 7 within 0.5.

| run | mismatches | median / p95 / max ms | lm_bookkeeping | bk_mat (n) | max RSS / peak footprint |
|---|---|---|---|---|---|
| `--reference-first` | 0 | 106.9 / 230.3 / 400.2 | 2,874 ms = 2.8 ms/ks | 0.0 (0) | 1.79 GB / 1.17 GB |
| product (no reference) | 0 | 107.5 / 235.6 / 397.8 | **12,711 ms = 12.3 ms/ks** | **10,428 ms (300,141)** | 2.37 GB / **2.21 GB** |

Both: fill_calls 1,036 (one per keystroke, confirmed), fill ~23-25
ms/call on ORT at 75.4 tokens, advance ~48-49 s.

#### Root cause (verified on the VM with per-phase counters)

The materialise calls are the **gate session's**, not the product
session's — the earlier note's attribution was wrong in direction: the
measured session never produces a `Pending` row in either
configuration. `bk_mat_n` snapshotted at each record boundary stays
flat across every measured keystroke pass and jumps only across the
baseline's whole-record pass (record 51: +27,922 rows at pool live
3,730; record 158: at live 3,962 of 4,096 — it starves exactly when the
pool is near-full). Mechanism: the page pool (`SessionShape::cache_rows`
= `--lm-cache` = 4,096 position-pages) is sized for **one** session's
live-node working set. The interleaved gate session is a second client
of the same `CharLm`; measured's 4,096-entry char-prefix cache plus its
trie-resident nodes pin most of the pool when the gate's record pass
runs, so `claim()` finds no free page and `reclaim()` no dead leaf —
every produced state of that record goes `Pending` and each pending row
materialises its history from scratch on every later `advance`
(bk_marshal). Pending `LmState`s also retain the step's `Arc<Generation>`
tensors instead of a page reference — the ~1 GB extra footprint. A
single failed claim early in a record cascades: pending parents produce
pending children for the whole record, which is why the materialise
count comes in per-record bursts rather than a steady ~290.

#### The fix

The pool stays sized for one session — an earlier version of this note
doubled `cache_rows` to make room for the gate, which changed the
product's memory to accommodate a session the product never has. The
harness changed instead: **the interleaved gate is gone**. The replay's
two modes are now `--reference-first` (the baseline pass completes and
drops, then the measured session runs alone, tops compared after — a
one-off baseline replays a record only to localise a mismatch) and the
default, where the measured session is the pool's only client and the
report's `mismatches` is `null`. The measured session never pended in
either mode — the gate was the only second client — so with no gate the
materialise path and its ~1 GB of retained generations never trigger.

VM, same build, 40 dev records:

| run | mismatches | median / p95 ms | lm_bookkeeping | bk_mat (n) | peak footprint |
|---|---|---|---|---|---|
| product (no gate) | `null` | 86.7 / 178.9 | **380 ms = 0.37 ms/ks** | 0.0 (0) | 1,513 MB |
| `--reference-first` | 0 | 80.9 / 157.2 | 2,254 ms = 2.2 ms/ks | 0.0 (0) | 1,688 MB |

The bookkeeping gap between the rows is the shared `CharLm` counter
reading both sessions' resolves in the reference mode; the product
row's 0.37 ms/ks is the measured session alone. Product footprint now
reads one session's state — under the reference run's peak on the same
binary, and well below the gated builds' ~1.9-2.2 GB readings.

`a_lone_session_materialises_no_pending_rows` in `ime-lm/tests/charlm.rs`
locks the invariant: one session driven through a fixture sentence at
the product shape never materialises a pending row.

#### d7d9f62 on the M1 (verbatim): the fix confirmed

| run | mismatches | median / p95 / max ms | lm_bookkeeping | bk_mat (n) | max RSS / peak footprint |
|---|---|---|---|---|---|
| product | null | 109.1 / 236.4 / 377.5 | 330 ms = 0.32 ms/ks | 0 (0) | 1.74 GB / 1.05 GB |
| `--reference-first` | 0 | 105.8 / 233.5 / 380.5 | 2,854 ms | 0 (0) | 1.64 GB / 1.21 GB |

fill 22.7-24.1 ms/call (1,036 calls), advance ~48-49 s. The product run is
one session's memory picture — 1.05 GB peak, under the reference run's
1.21 GB and far under the interleaved gate's 2.21 GB; bookkeeping is
0.32 ms/keystroke measured-only (the reference row's 2,854 ms reads both
sessions' resolves on the shared counter).

#### The landing cleanup

The debug legs that decided these rounds are out of the library: `LM_TAIL`
(the fused per-layer tail kernel — the M1 measured it ~3x slower),
`LM_MM_NAIVE` (the per-element GEMM), `LM_NO_SM`, `LM_SERIAL_ENC` (one
encoder mode — the serial one; the M1 showed the concurrent encoder no
different), `IME_LM_CPU` (`--backend` already types the LM's backend), and
`LM_ONLY_OPS` (the dispatch truncation is now a typed API —
`CharLm::set_metal_step_limit` — driven by metal-bench's `--only-ops`).
`metal_step_matches_cpu_at_every_row_tile` is `#[ignore]`d and requires
`CHARLM_LM_DIR` — no silent pass without the export.
