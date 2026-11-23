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
