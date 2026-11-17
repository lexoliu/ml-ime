# The key-level engine's per-keystroke latency on the M1 (2026-11-17, #151)

## Verdict

The key-level engine meets #216 §8's latency item with room to spare: on the M1, replaying 498 eval3 dev records keystroke by keystroke, it answers in a median of 23.7 ms and a p95 of 54.0 ms, against §8's 109 ms and 236 ms. The fused engine, replayed on the same records in its product configuration, takes 130.6 ms and 270.8 ms. The key engine also holds a tenth of the fused engine's memory: a 128 MB footprint against 1.32 GB.

Both engines ran on the Mac mini (M1, 8 GB) with nothing else running on it. Each replay covered eval3 dev with context: 498 records and 14,625 keystrokes, one sentence and eight beams at a time.

- **Key engine.** S4's final (step 2,087) converted to Core ML and run on the CPU and the Neural Engine, at β 1, γ 1 and the sum combination, S4's choice on full pinyin. S5 has the same architecture and shapes, so these timings hold for it.
- **Fused engine.** The int8 route A towers on ONNX Runtime with 4 threads, the int8 paged character LM on Metal, the run3 trigram, and `--incremental`; `ime-cli replay`'s defaults are the product row's.

| engine | median | p95 | max | footprint | peak RSS |
|---|---|---|---|---|---|
| key-level | 23.7 ms | 54.0 ms | 120.9 ms | 128 MB | 298 MB |
| fused | 130.6 ms | 270.8 ms | 3,450 ms | 1.32 GB | 2.89 GB |

**By typed length** (median / p95, ms):

| keys typed | key-level | fused |
|---|---|---|
| 1–4 | 12 / 21 | 27 / 87 |
| 5–8 | 16 / 22 | 80 / 195 |
| 9–16 | 21 / 29 | 125 / 226 |
| 17–24 | 27 / 38 | 143 / 246 |
| 25–32 | 35 / 48 | 162 / 278 |
| 33–48 | 42 / 62 | 184 / 305 |
| 49–64 | 55 / 89 | 230 / 353 |
| 65+ | 65 / 102 | 299 / 560 |

**The equality gate.** The key engine's last-keystroke top-1 equals the Python constrained search over the same Core ML model on 497 of 498 records. The one difference is record 3095 (有 against 悠), where the reference's two candidates are 1.2e-7 apart.

**The fused engine against its record.** On the 40 records of the product row in `notes/keystroke-latency.md` the fused engine now replays at 119.9 / 240.9 ms against that record's 109.1 / 236.2. Its emission costs 23.8 ms a keystroke against 23.5, and its decode 98.7 against 95.1. The towers were exported again from route A v2's final weights (`mlime-e2e-init`), because the September exports have no reserved `<unk>` span row and today's loader refuses them (#109).

## Where the time goes

Key engine, mean milliseconds a keystroke:

| keys typed | total | keys call | miss tree calls | speculative tree call | host search | misses | rows queried |
|---|---|---|---|---|---|---|---|
| 1–8 | 15.2 | 6.2 | 4.2 | 2.1 | 2.7 | 1.8 | 14 |
| 9–24 | 24.4 | 6.0 | 8.6 | 2.8 | 6.7 | 3.6 | 43 |
| 25–48 | 39.2 | 7.3 | 14.3 | 3.8 | 13.3 | 5.8 | 90 |
| 49+ | 60.3 | 8.5 | 20.9 | 6.9 | 22.9 | 8.3 | 159 |

The keys call encodes the context and the keys at a fixed shape, so its cost stays near 6–8 ms at every length. What grows with length is the beam's work:

- **Tree calls.** Each keystroke issues one speculative call, made ahead for the rows the last search wanted, and then misses for the rows the speculation did not hold. 84.6% of queried rows were speculated, and a keystroke still needed a median of 3 miss calls. Of the 70,292 tree calls, 60,041 were in the 32-row bucket, 5,934 in the 64-row, 3,554 in the 128-row and 763 in the 256-row.
- **Host search.** The search's own bookkeeping on the CPU grows with the number of rows the beam queries.

At 49 or more keys, tree calls and host search are 44 of the 60 ms. They are the levers if longer input has to get faster: fewer miss round trips, by speculating deeper or batching the misses of one step into one call, and a cheaper host search.

No decoder state survives a keystroke, because the encoder reads all the keys in both directions. Each keystroke's cost is therefore that of a full re-encode plus the search above, and it is under 25 ms for the first 24 keys.

The fused engine spends 30 ms of its mean on the towers' emission and 109 ms on decode. Its decode is the LM's advance and the trigram's per-pair scoring inside the beam. 20 of its 14,625 keystrokes took over a second, almost all of it in decode. The key engine's slowest keystroke was 121 ms.

**Opening.** The model opens in 1.6 s and warms in 0.13 s once Core ML's compiled bundle is cached. The first open after a model changes compiles it and took 54 s here; a product process opens the model once.

## Reproduce

```sh
cargo build --release -p ime-cli
mlime export keys data/e2e-keys/s4/final/run/checkpoint-final.pt --out BUNDLE
mlime-coreml convert --bundle BUNDLE --work-dir data/e2e-keys/s4/coreml/work
mlime eval keys --coreml data/e2e-keys/s4/coreml/work/models --replay \
    --eval-set data/run3_pool/eval3.jsonl --slice dev --beta 1 --gamma 1 --context \
    --dump-dir data/e2e-keys/s4/coreml/dump-replay --out dev-replay.txt
ime-cli replay-keys --model data/e2e-keys/s4/coreml/work/models \
    --eval-set data/run3_pool/eval3.jsonl --slice dev --beta 1 --gamma 1 --combine sum \
    --reference data/e2e-keys/s4/coreml/dump-replay/dev-beta1-gamma1-sum.jsonl \
    --out keys-dev.json
mlime export route-a route-a-v2-final.pt --out route-a/int8 --quantize int8
ime-cli replay --route-a route-a/int8 --route-a-threads 4 \
    --lm data/char-lm-run/transformer-s1/char-lm-int8-pages --model data/run3/ngram.mmap \
    --emittable data/route-a-assets/emittable.txt --eval-set data/run3_pool/eval3.jsonl \
    --slice dev --incremental --backend metal --out fused-dev.json
```

`mlime eval keys --replay` writes the reference: the Python constrained search over the same Core ML functions, replayed keystroke by keystroke.
