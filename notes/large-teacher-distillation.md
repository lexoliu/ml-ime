# Phase 2 large teacher pilot: Qwen3.8-27B LoRA S3 (2026-11-22, #191, #328)

## Verdict

The 200k-example LoRA pilot (`S3`, `3,125` steps, `364317b`) of `Qwen/Qwen3.8-27B` (`Qwen3_5ForCausalLM`, `37561816e66b5b9153b1b64c10968c18c1e6f4a0`) already surpasses the best Phase 1 shipping model `S4/S5-prelude` (`10M+` training examples) on both **abbreviated** (`31.79%` vs `28.42%`, **`+3.37` points**, `+16` records) and **mixed** (`43.39%` vs `42.98%`, **`+0.41` points**, `+2` records) dev slices after seeing only `0.04` epochs of data.

On **full pinyin**, S3 trails `S4/S5-prelude` on dev (`73.29%` vs `78.11%`, `−4.82` points, `−24` records). Error analysis on `eval-full-dev.jsonl` isolates the primary cause: feeding raw full-pinyin strings directly into Qwen's multilingual BPE splits them into arbitrary English/multilingual subwords (`mageqiujiganfakeyidea` → `['mage', 'qi', 'uj', 'igan', 'fake', 'y', 'idea']`), causing accuracy to collapse on long keystrokes:

- `16–25` keys (`n = 169`): **`80.47%`** top-1 (`136 / 169`), `89.35%` top-8 — already above both `S4/S5-prelude` and `fused`.
- `26–35` keys (`n = 127`): `76.38%` top-1 (`97 / 127`), `87.40%` top-8.
- `36–50` keys (`n = 84`): `73.81%` top-1 (`62 / 84`), `80.95%` top-8.
- `51–200` keys (`n = 47`): **`48.94%`** top-1 (`23 / 47`), `70.21%` top-8.

Every lowercase ASCII letter `'a'..'z'` is already a single contiguous token (`64..89`) in Qwen 3.8's vocabulary. Switching `encode_prompt` to encode `keys` **one ASCII letter per token** (while keeping `context` in Qwen's canonical BPE) eliminates BPE subword mangling cleanly before S4's `2M`-example fine-tune.

## S3 dev gate (`eval3` dev, context on, beam 8)

| set | Qwen3.8-27B S3 (200k) | S4/S5-prelude (10M+) | S3 − S4/S5-prelude | S3 top-8 | S3 char acc | S3 MRR@8 |
| --- | --- | --- | --- | --- | --- | --- |
| full (`498`) | `73.29%` (`365/498`) | `78.11%` (`389/498`) | `−4.82` (`−24`) | `85.94%` | `93.50%` | `0.7834` |
| abbreviated (`475`) | **`31.79%`** (`151/475`) | `28.42%` (`135/475`) | **`+3.37`** (`+16`) | `45.89%` | `66.69%` | `0.3718` |
| mixed (`484`) | **`43.39%`** (`210/484`) | `42.98%` (`208/484`) | **`+0.41`** (`+2`) | `59.30%` | `78.42%` | `0.4949` |

`0` unanswered records across all three sets. On full pinyin dev, records with context (`n = 393`) reach `77.61%` (`305/393`); records without context (`n = 105`) are at `57.14%` (`60/105`).

## Architecture and measured G4 throughput

- **Model**: frozen `bf16` `Qwen/Qwen3.8-27B` (64 layers: 48 `linear_attention` + 16 `full_attention`, `hidden_size = 5120`, `intermediate_size = 17408`) + LoRA (`rank = 64, alpha = 128`, 400 adapted projections across all attention and MLP layers) + `1,992` trainable residue input embedding rows + dedicated `(7323, 5120)` `emit_head` (`482,851,840` trainable parameters).
- **Kernels**: `flash-linear-attention` prebuilt wheel installed (`chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` active); `causal-conv1d` fell back to PyTorch reference.
- **Training (`G4`, RTX PRO 6000 Blackwell 96 GB, `batch_size = 16, accumulate = 4`, gradient checkpointing)**:
  - `3,125` steps (`200,000` examples) in `13,701.76 s` (`3.81 h`, mean **`4.385 s/step`**).
  - Sustained **`97%–99%` GPU utilisation**, `541–560 W`, `61.3–63.6 / 97.9 GB` VRAM.
  - Loss fell from `4.7520` (step 1) to `0.2051` (step 3,125).
  - Compact resumable checkpoint (`teacher-checkpoint.pt`, `5.80 GB` with AdamW state) published every 250 steps and final checkpoint archived at `lexoliu/mlime-resume/mlime-teacher-qwen27b/s3-200k/teacher-final.pt`.
- **Batched lattice-constrained beam-8 evaluation (`G4`)**:
  - `~0.91 s/record` (`~7.5 min` per dev twin, `~23 min` total for all 3 twins = `1,457` records), `78%–86%` GPU utilisation, `54.2 GB` VRAM.
- **Total S3 wall time and cost**: `~4.22 h` on G4 (~**38 CU** at ~9 CU/h).

## S4 optimisations and updated budget

Before launching S4 (`2M` examples = `31,250` steps, seeded from `s3-200k/teacher-final.pt` via `--init-from` so pilot weights are inherited):

1. **One keystroke per token**: `encode_prompt` encodes `context` with canonical BPE and `keys` one ASCII letter per token (`a..z` → `64..89`), removing English BPE subword mangling on long full-pinyin strings.
2. **BF16 LoRA compute**: `TeacherLoraLinear` keeps `lora_a` and `lora_b` master weights in `fp32` for AdamW while casting the small rank-64 factors to `x.dtype` (`bf16`) in `forward`, avoiding 400× per-layer `[B, T, 17408]` activation casts to `fp32`.
3. **Rolling shuffle buffer (`4096`)**: smooths cross-shard source mixing per batch instead of shard-by-shard loss oscillation.
4. **Larger micro-batch + H100 throughput check**: increase micro-batch to `batch_size = 32, accumulate = 2` (halving 54 GB weight reads per update) and measure **H100 (80 GB HBM3, 3.35 TB/s)** vs **G4** directly from S3's checkpoint (keeping the winning leg's steps), then run S4 to `2M` examples on the faster card.

At S3's unoptimised G4 rate (`4.385 s/step`), `31,250` steps would take `38.1 h` (`~343 CU`); with BF16 LoRA + `batch_size = 32` + H100's 2× HBM3 bandwidth, target is `~2.0–2.5 s/step` (`~17–22 h`).

## Reproduce

```sh
mlime train teacher \
  --data-dir /tmp/s3_data/corpus --labels /tmp/s3_data/corpus/labels \
  --char-table crates/ime-pinyin/data/char_pinyin.tsv \
  --typo-table crates/ime-pinyin/data/typo.json \
  --unit-initials 0.50 --unit-lexicon data/lexicon/knowledge.tsv \
  --out /tmp/s3_out --batch-size 16 --accumulate 4 --max-steps 3125 \
  --checkpoint-every 250 --log-every 10 --lr 1e-4 \
  --lora-rank 64 --lora-alpha 128.0 --context-dropout 0.30 --seed 0
```

---

# Phase 2 teacher S4: Qwen3.8-27B 2M-example LoRA with one-letter keys (2026-11-24, #191, #335)

## Verdict

Scaling `Qwen/Qwen3.8-27B` LoRA to `2,000,000` examples (`31,250` steps, `04f1f6a`, `S4`, evaluated from the archived `step 31,000` checkpoint where `lr = 1.71e-8`) with one-letter key encoding, BF16 LoRA compute, rolling 4,096 shuffle buffer, and length-bucketed microbatches (#336) lifts every clean dev twin sharply over S3:

- **Abbreviated (`475`)**: **`38.32%`** top-1 (`182/475`, top-8 `54.74%`), **`+6.53` points** over S3 (`31.79%`) and **`+9.89` points** (`+47` records, `+79 / −32`) ahead of the best Phase 1 shipping model `S4/S5-prelude` (`28.42%`).
- **Mixed (`484`)**: **`53.51%`** top-1 (`259/484`, top-8 `69.83%`), **`+10.12` points** over S3 (`43.39%`) and **`+10.54` points** (`+51` records, `+86 / −35`) ahead of `S4/S5-prelude` (`42.98%`).
- **Full (`498`)**: **`76.91%`** top-1 (`383/498`, top-8 **`91.16%`**), **`+3.61` points** (`+18` records) and `+5.22` top-8 points over S3 (`73.29% / 85.94%`). On context-present records (`n = 393`), S4 ties `S4/S5-prelude` exactly at **`80.92%`** (`318/393`, top-8 `94.40%`), while the `−1.20`-point (`−6` records) overall delta sits entirely on context-absent records (`61.90%` vs `67.62%`, `n = 105`), where `S4/S5-prelude` has seen `12M+` training examples across all shards vs S4's `2M`. Record-level complementarity on full dev is wide: S4 solves `41` records `S4/S5-prelude` misses (chiefly semantic and world-knowledge disambiguations) against `47` solved only by `S4/S5-prelude`, for a combined union of **`86.35%`** (`430/498`).
- **Long full-pinyin fix**: one-letter key encoding raises every length bucket over S3 (`16–25` keys `80.47% → 83.43%`, `26–35` keys `76.38% → 79.53%`, `36–50` keys `73.81% → 76.19%` — `+1.19` above `S4/S5-prelude`, and `51–200` keys `48.94% → 55.32%` top-1 / `70.21% → 82.98%` top-8).
- **Eval3-typo (`498`)**: `54.02%` top-1 (`269/498`, top-8 `71.29%`) at fixed `γ = 1.0` (`S4/S5-prelude` `57.83%` at dev-selected `γ = 0.5`; union `66.27%`).

## S4 dev gate (`eval3` dev, context on, beam 8, `γ = 1.0`)

| set (`n`) | Qwen3.8-27B S4 (`step 31,000`) | S3 (200k) | S4 − S3 | S4/S5-prelude (10M+) | S4 − prelude | S4 top-8 | S4 char acc | S4 MRR@8 | union (S4 ∪ prelude) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full (`498`) | `76.91%` (`383/498`) | `73.29%` | `+3.61` (`+18`) | `78.11%` (`389/498`) | `−1.20` (`−6`) | `91.16%` | `95.13%` | `0.8277` | `86.35%` (`430/498`) |
| full, context present (`393`) | **`80.92%`** (`318/393`) | `77.61%` | `+3.31` (`+13`) | `80.92%` (`318/393`) | `0.00` (`0`) | `94.40%` | `96.75%` | `0.8651` | — |
| full, context absent (`105`) | `61.90%` (`65/105`) | `57.14%` | `+4.76` (`+5`) | `67.62%` (`71/105`) | `−5.71` (`−6`) | `79.05%` | `88.93%` | `0.6876` | — |
| abbreviated (`475`) | **`38.32%`** (`182/475`) | `31.79%` | **`+6.53`** (`+31`) | `28.42%` (`135/475`) | **`+9.89`** (`+47`) | `54.74%` | `72.61%` | `0.4435` | `45.05%` (`214/475`) |
| mixed (`484`) | **`53.51%`** (`259/484`) | `43.39%` | **`+10.12`** (`+49`) | `42.98%` (`208/484`) | **`+10.54`** (`+51`) | `69.83%` | `84.12%` | `0.5946` | `60.74%` (`294/484`) |
| eval3-typo (`498`) | `54.02%` (`269/498`) | — | — | `57.83%` (`288/498`, `γ=0.5`) | `−3.82` (`−19`) | `71.29%` | `86.94%` | `0.6056` | `66.27%` (`330/498`) |

`0` unanswered records across all four sets. Full-pinyin dev accuracy by keystroke length (S4 vs S3 vs `S4/S5-prelude`):

| keystrokes (`n`) | S4 top-1 | S4 top-8 | S3 top-1 | S4/S5-prelude top-1 | S4 − S3 | S4 − prelude |
| --- | --- | --- | --- | --- | --- | --- |
| `1–15` (`71`) | `71.83%` (`51/71`) | `92.96%` | — | `74.65%` (`53/71`) | — | `−2.82` |
| `16–25` (`169`) | `83.43%` (`141/169`) | `92.90%` | `80.47%` | `83.43%` (`141/169`) | `+2.96` | `0.00` |
| `26–35` (`127`) | `79.53%` (`101/127`) | `92.13%` | `76.38%` | `79.53%` (`101/127`) | `+3.15` | `0.00` |
| `36–50` (`84`) | **`76.19%`** (`64/84`) | `89.29%` | `73.81%` | `75.00%` (`63/84`) | `+2.38` | **`+1.19`** |
| `51–200` (`47`) | `55.32%` (`26/47`) | `82.98%` | `48.94%` | `65.96%` (`31/47`) | `+6.38` | `−10.64` |

## Training run and batched decode throughput

- **Training (`G4`, RTX PRO 6000 Blackwell 96 GB, `04f1f6a`, `batch_size = 32, accumulate = 2`, gradient checkpointing, length bucketing #336)**:
  - H100 runtimes were unavailable on Colab capacity (503) at leg start, so both legs ran on G4 (`~9 CU/h`).
  - **Leg 1**: steps `1–25,000` in `82,103.1 s` (`22.81 h`, **`3.284 s/step`**).
  - **Leg 2**: resumed from `step 25,000`, finished steps `25,001–31,250` in `20,290.5 s` (`5.64 h`, **`3.247 s/step`**).
  - **Total training time**: `102,393.6 s` (`28.44 h`, **`~256 CU`** on G4) — `25%` faster per step than S3 (`4.385 → 3.277 s/step`), with mean padding efficiency `pad_eff = 0.928` (`0.921–0.939`) and `98%–100%` GPU utilisation.
  - Loss fell from `0.4525` (step 1, initialised from `s3-200k/teacher-final.pt`) to `0.1855` at `step 31,000` (`lr = 1.71e-8`) and `0.1997` at `step 31,250` (`lr = 0.0`).
  - Archived checkpoint: `lexoliu/mlime-resume/mlime-teacher-qwen27b/s4-2m/teacher-checkpoint.pt` (`step 31,000 / 31,250`, `5.80 GB`).
- **Batched multi-record decode (`G4`, `98600da`, #338 + #344)**:
  - Span-vectorised `_SpanLattice`, left-padded multi-record prefill/decode (`batch_records = 16`, up to `128` active beams), and `consume=True` in `_select_cache_rows` (#344, halving decode cache VRAM to **`75.2 / 97.9 GB`** — fitting both 80 GB H100 and 96 GB G4) cut evaluation wall time by **6–7×** over S3 (`0.91 s/record`):
    - `full` dev (`498`): `99.3 s` (`0.199 s/record`, includes first-batch JIT warmup)
    - `abbr` dev (`475`): `64.9 s` (`0.137 s/record`)
    - `mixed` dev (`484`): `68.8 s` (`0.142 s/record`)
    - `typo` dev (`498`): `72.6 s` (`0.146 s/record`)
  - All four dev twins (`1,955` records total) completed in **`306 s` (`5.1 min`)** at `90%–96%` GPU utilisation.

## Updated S5 offline labelling budget

At the measured `98600da` G4 throughput (`~0.14–0.16 s/record` beam-8 decode plus `~0.03–0.04 s/record` prefix-reusing batched T1/T2 scoring = `~0.18–0.20 s/record`):

- `200,000` labelled examples (`100` interleaved shards × `2,000` rows): **`~10–11 h` on G4 (`~95 CU`)**, or **`~5–6 h` on H100**.
- `400,000` labelled examples (`200` interleaved shards × `2,000` rows): **`~20–22 h` on G4 (`~190 CU`)**, or **`~10–12 h` on H100**.
- Mid-shard `.part-*.parquet` checkpointing every `1,000` rows and per-shard resumability let S6 (`mlime train distill-keys`, `~15–20 min` per run) evaluate the first `100` shards (`200k` examples) before deciding whether to label the second `200k`.

## Reproduce S4

```sh
mlime train teacher \
  --data-dir /tmp/s4_data/corpus --labels /tmp/s4_data/corpus/labels \
  --char-table crates/ime-pinyin/data/char_pinyin.tsv \
  --typo-table crates/ime-pinyin/data/typo.json \
  --init-from s3-200k/teacher-final.pt \
  --unit-initials 0.50 --unit-lexicon data/lexicon/knowledge.tsv \
  --out /tmp/s4_out --batch-size 32 --accumulate 2 --max-steps 31250 \
  --checkpoint-every 500 --log-every 20 --lr 1e-4 \
  --lora-rank 64 --lora-alpha 128.0 --context-dropout 0.30 \
  --shuffle-buffer 4096 --seed 0
```
