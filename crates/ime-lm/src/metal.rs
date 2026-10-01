//! The Metal backend for a paged export's step graph — macOS only.
//!
//! Where the ORT path binds `charlm.onnx` and reads `next_state_*` back, this
//! backend runs the same arithmetic as hand-encoded Metal kernels: an
//! embedding + input-projection dispatch, per transformer block LN1 + QKV,
//! fused paged attention, output projection + residual, LN2 + FFN up + gelu,
//! FFN down + residual, then head-projection + tied head matmul +
//! log-softmax gather — one command buffer, 63 dispatches per `advance`
//! call. The resident page pool is a pair of `MTLBuffer`s the kernels only
//! read; a row's fresh key/value returns in `next_*`, and the host writes the
//! claimed page itself, exactly as the ORT path does.
//!
//! The product's per-row semantics match the export's: `pages`/`mask` mark the
//! history positions, `depths[row]` is the count, the new token sits at
//! position `depths[row]` (`input(embed[token]) + positions[depth]`), the
//! attention covers `depths + 1` entries with the row's own K/V at the tail,
//! and the candidate gather log-softmaxes the tied head over `candidates`.
//! `source_row` names a scratch slot — `skeys`/`svals` in `[slots, L, H, T,
//! D]` — where a pending state's history was materialised; `-1` reads the
//! page pool.
//!
//! Weights are resident `MTLBuffer`s in a uniform `[in, out]` view: the int8
//! export's `onnx::MatMul_*_int8` tensors already carry that layout with a
//! per-out-column fp32 scale, and the fp32 export's `model.*` weights
//! (`[out, in]`, torch's) are transposed at pack time. Activations and the
//! page pool stay fp32 end to end.
//!
//! The matmuls are the tiled `lm_mm` kernel — a threadgroup owns a
//! [32 x 64] output tile, stages the A and W panels in threadgroup memory,
//! and accumulates 8x8 tiles through simdgroup matrix multiply-accumulate,
//! so each weight byte is read once per tile instead of once per row. Five
//! dispatches per layer — LN1+QKV, the paged-attention kernel, out-proj +
//! residual, LN2+ff.0+gelu, ff.2+residual — plus embed, head-proj, the tied
//! head's full-vocab GEMM, and the candidate gather: 63 dispatches on one
//! command buffer.

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePassDescriptor,
    ComputePipelineState, CounterSampleBufferDescriptor, Device, FunctionConstantValues,
    MTLCounterSamplingPoint, MTLDataType, MTLResourceOptions, MTLSize, MTLStorageMode, NSRange,
};
use std::time::Instant;

use crate::{LmError, MetalWeights, WeightDtype, WeightTensor, WeightsFile, dim};

/// The transformer config the kernels are compiled for — checked against the
/// manifest's row shapes at open, since the source has them as #defines.
const HIDDEN: usize = 512;
/// The head count.
const HEADS: usize = 8;
/// `hidden / heads`.
const HEAD_DIM: usize = 64;
/// The FFN's intermediate width.
const FF: usize = 2048;

/// The step's whole kernel library, compiled once at open.
const STEP_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define HID 512u
#define NH 8u
#define HD 64u
#define NFF 2048u
#define LAYERS 12u
#define EPS 1e-5f
#define INV_SQRT_HD 0.125f   // 1/sqrt(64)

// STATS2 is the per-row slot stride (in floats) of the stats accumulator
// buffer: slot 0 = embed's x, slots 1..LAYERS = ffn2's x for the next
// layer's qkv, slots LAYERS+1..2*LAYERS = out's x for ffn0.
constant uint STATS2 = (2u * LAYERS + 1u) * 2u;

// Weight element at [i*out + o] — [in, out] row-major so a warp's consecutive
// outputs read consecutive bytes. Q: 0 = f16, 1 = i8 + per-out scale, 2 = f32.
template<int Q>
inline float ldw(device const void* w, device const float* s, ulong i, ulong o, ulong out) {
    if (Q == 1) {
        return float(((device const char*)w)[i * out + o]) * s[o];
    }
    if (Q == 2) {
        return ((device const float*)w)[i * out + o];
    }
    return float(((device const half*)w)[i * out + o]);
}

// erf via Abramowitz & Stegun 7.1.28 (|eps| <= 1.2e-7) — Metal has no erf.
inline float erf_as(float x) {
    const float sign = signbit(x) ? -1.0f : 1.0f;
    const float a = min(fabs(x), 8.0f);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float y = 1.0f
        - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
            - 0.284496736f) * t + 0.254829592f) * t * exp(-a * a);
    return sign * y;
}

// The row's x: input(embed[token]) + positions[depth] — the position added
// after the projection, matching `embed_at`. One group per row; the embedding
// column and position row live in threadgroup memory.
template<int QE, int QI>
kernel void lm_embed(
    device float*       x       [[buffer(0)]],   // [rows, HID]
    device const void*  w_emb   [[buffer(1)]],   // [HID, V] qt
    device const float* s_emb   [[buffer(2)]],   // [V]
    device const float* pos_tab [[buffer(3)]],   // [*, HID]
    device const uint*  token   [[buffer(4)]],   // [rows]
    device const uint*  depths  [[buffer(5)]],   // [rows] — the new position
    device const void*  w_in    [[buffer(6)]],   // [HID, HID] qt
    device const float* s_in    [[buffer(7)]],   // [HID]
    device const float* b_in    [[buffer(8)]],   // [HID]
    constant uint&      vocab   [[buffer(9)]],
    device half*        x16     [[buffer(10)]],  // [rows, HID] fp16 x
    device atomic_float* stats  [[buffer(11)]],  // [rows, STATS2] sums
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_x[HID];
    threadgroup float tg_red[2 * (HID / 32)];
    const uint row = tgid;
    tg_x[tid] = ldw<QE>(w_emb, s_emb, tid, token[row], vocab);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = b_in[tid] + pos_tab[(ulong)depths[row] * HID + tid];
    for (uint i = 0; i < HID; ++i) {
        acc += tg_x[i] * ldw<QI>(w_in, s_in, i, tid, HID);
    }
    x[row * HID + tid] = acc;
    x16[(ulong)row * HID + tid] = half(acc);
    // Slot 0's Σx/Σx² come from this row — the threadgroup owns it whole,
    // so the reduce is in-register; the later slots are cleared here so
    // the layer producers' atomicAdds start from zero each call.
    for (uint s = 2u + tid; s < STATS2; s += HID) {
        atomic_store_explicit(
            stats + (ulong)row * STATS2 + s, 0.0f, memory_order_relaxed);
    }
    const float s0 = simd_sum(acc);
    const float s1 = simd_sum(acc * acc);
    if (lane == 0) {
        tg_red[2 * sg] = s0;
        tg_red[2 * sg + 1] = s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t0 = 0.0f;
        float t1 = 0.0f;
        for (uint i = 0; i < HID / 32; ++i) {
            t0 += tg_red[2 * i];
            t1 += tg_red[2 * i + 1];
        }
        atomic_store_explicit(
            stats + (ulong)row * STATS2, t0, memory_order_relaxed);
        atomic_store_explicit(
            stats + (ulong)row * STATS2 + 1u, t1, memory_order_relaxed);
    }
}

// ---- tiled GEMM -------------------------------------------------------------
// One [MM_MT x MM_NT] output tile per threadgroup: A ([M,K], lda-strided) and
// W (the packed [in,out] table, ldn-strided) stage into threadgroup memory in
// MM_KB-deep k-chunks, so every weight element is read once per tile and
// consumed by all MM_MT rows through 8x8 simdgroup matrix multiply-accumulate.
// flags:
//   MM_LN   layer-norm A first — ln packs [w|b] over K, row stats computed
//           on the fly per tile row (recomputed per column-tile — cheap: the
//           A panel is at most 32 x K floats).
//   MM_GELU erf-gelu epilogue, applied after bias+scale
//   MM_RES  the store is `res[m,n] += acc` — res and out alias.
//   MM_QKV  split the tile's columns: < HID -> out[m,n], < 2*HID -> out2 at
//           (row*LAYERS + layer)*HID + n-HID (next_k), the rest -> out3
//           (next_v); nothing lands in `out` above HID.
//   MM_BIAS add bias[n]
// Q: 0 = f16 weights, 1 = i8 + a per-out scale applied in the epilogue so
// the accumulate is a pure int-dot sum, 2 = f32.
constant uint MM_MT = 32u;
constant uint MM_NT = 64u;
constant uint MM_KB = 32u;
constant uint MM_LN = 1u;
constant uint MM_GELU = 2u;
constant uint MM_RES = 4u;
constant uint MM_QKV = 8u;
constant uint MM_BIAS = 16u;
constant uint MM_SCALE = 32u;
// MM_H16 — the epilogue also writes the output as fp16 into `out_h`, the
// [rows, N] half-word staging buffer the consuming `lm_sm` reads as A.
// MM_STA — the epilogue atomicAdds per-row (Σv, Σv²) into stats[m][slot],
// `layer` carrying the slot index (it is free on non-QKV ops); the
// consumer's algebraic LN reads the sums.
constant uint MM_H16 = 64u;
constant uint MM_STA = 128u;

template<int Q>
kernel void lm_mm(
    device const float* A    [[buffer(0)]],   // [M, K] f32, lda-strided
    device const float* ln   [[buffer(1)]],   // [2K] ln w||b, when MM_LN
    device const void*  W    [[buffer(2)]],   // [K, N] packed weights
    device const float* S    [[buffer(3)]],   // [N] per-out scale, when Q==1
    device const float* bias [[buffer(4)]],   // [N], when MM_BIAS
    device float*       res  [[buffer(5)]],   // [M, N], when MM_RES
    device float*       out  [[buffer(6)]],   // [M, N]
    device float*       out2 [[buffer(7)]],   // next_k, when MM_QKV
    device float*       out3 [[buffer(8)]],   // next_v, when MM_QKV
    constant uint&      flags [[buffer(9)]],
    constant uint&      K     [[buffer(10)]],
    constant uint&      N     [[buffer(11)]],
    constant uint&      M     [[buffer(12)]],
    constant uint&      lda   [[buffer(13)]],
    constant uint&      ldn   [[buffer(14)]],
    constant uint&      layer [[buffer(15)]],
    device half*        out_h [[buffer(16)]],  // [M, N] fp16, when MM_H16
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tpos.x;
    const uint mo = tgid.y * MM_MT;
    const uint no = tgid.x * MM_NT;
    threadgroup float tg_a[MM_MT * MM_KB];
    threadgroup float tg_b[MM_KB * MM_NT];
    threadgroup float tg_s[MM_NT];
    threadgroup float tg_mean[MM_MT];
    threadgroup float tg_rstd[MM_MT];
    threadgroup float tg_zero[64];

    // The tile's one-off stages: the per-out scale, a zero matrix, and the
    // LN row stats — all before the first barrier.
    if (Q == 1 && tid < MM_NT / 4) {
        const uint e = tid * 4;
        const uint gn = no + e;
        float4 sv = float4(0.0f);
        if (gn + 4 <= N) {
            sv = *(device const float4*)(S + gn);
        } else {
            for (uint j = 0; j < 4 && gn + j < N; ++j) { sv[j] = S[gn + j]; }
        }
        *(threadgroup float4*)(tg_s + e) = sv;
    }
    if (tid < 64) { tg_zero[tid] = 0.0f; }
    if (flags & MM_LN) {
        // Four threads per tile row reduce sum/sumsq over the lda-strided A.
        const uint rr = tid / 4;
        const uint lane4 = tid & 3;
        if (rr < MM_MT) {
            const uint gm = mo + rr;
            float s0 = 0.0f;
            float s1 = 0.0f;
            if (gm < M) {
                device const float* p = A + (ulong)gm * lda;
                for (uint k = lane4 * 4; k < K; k += 16) {
                    if (k + 4 <= K && (lda & 3u) == 0u) {
                        const float4 v = *(device const float4*)(p + k);
                        s0 += v.x + v.y + v.z + v.w;
                        s1 += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                    } else {
                        for (uint j = 0; j < 4 && k + j < K; ++j) {
                            const float xv = p[k + j];
                            s0 += xv;
                            s1 += xv * xv;
                        }
                    }
                }
            }
            // The xor butterfly: after xor-1 each lane holds its pair sum,
            // xor-2 combines pairs — the quad's full sum on every lane.
            s0 += simd_shuffle_xor(s0, 1);
            s0 += simd_shuffle_xor(s0, 2);
            s1 += simd_shuffle_xor(s1, 1);
            s1 += simd_shuffle_xor(s1, 2);
            if (lane4 == 0) {
                const float mean = s0 / (float)K;
                tg_mean[rr] = mean;
                tg_rstd[rr] = rsqrt(max(s1 / (float)K - mean * mean, 0.0f) + EPS);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 z;
    simdgroup_load(z, tg_zero, 8);
    simdgroup_float8x8 c[4];
    for (uint r = 0; r < 4; ++r) { c[r] = z; }
    // The 32x64 tile splits 4 row-strips x 2 column-halves across the 8
    // simdgroups: tr is this simdgroup's 8 rows, tc its 32 columns.
    const uint tr = (sg & 3u) * 8;
    const uint tc = (sg >> 2) * 32;
    for (uint k0 = 0; k0 < K; k0 += MM_KB) {
        // Stage the A chunk: 32x32 floats = one vec4 per thread.
        {
            const uint i = tid / 8;
            const uint e = tid % 8;
            const uint gm = mo + i;
            const uint k = k0 + e * 4;
            float4 v = float4(0.0f);
            if (gm < M && k < K) {
                device const float* p = A + (ulong)gm * lda + k;
                if (k + 4 <= K && (lda & 3u) == 0u) {
                    v = *(device const float4*)p;
                } else {
                    for (uint j = 0; j < 4 && k + j < K; ++j) { v[j] = p[j]; }
                }
            }
            if (flags & MM_LN) {
                v = (v - tg_mean[i]) * tg_rstd[i];
                if (k + 4 <= K) {
                    const float4 wv = *(device const float4*)(ln + k);
                    const float4 bv = *(device const float4*)(ln + K + k);
                    v = v * wv + bv;
                } else {
                    for (uint j = 0; j < 4 && k + j < K; ++j) {
                        v[j] = v[j] * ln[k + j] + ln[K + k + j];
                    }
                }
            }
            *(threadgroup float4*)(tg_a + i * MM_KB + e * 4) = v;
        }
        // Stage the W chunk: 32x64 elements = two vec4s per thread.
        for (uint u = tid; u < MM_KB * MM_NT / 4; u += 256) {
            const uint i = u / 16;
            const uint e = u % 16;
            const uint k = k0 + i;
            const uint gn = no + e * 4;
            float4 v = float4(0.0f);
            if (k < K) {
                if (Q == 1) {
                    device const char* w8 =
                        (device const char*)W + (ulong)k * ldn + gn;
                    if (gn + 4 <= N) {
                        const char4 qv = *(device const char4*)w8;
                        v = float4((float)qv.x, (float)qv.y, (float)qv.z,
                                   (float)qv.w);
                    } else {
                        for (uint j = 0; j < 4 && gn + j < N; ++j) {
                            v[j] = (float)w8[j];
                        }
                    }
                } else if (Q == 0) {
                    device const half* wh =
                        (device const half*)W + (ulong)k * ldn + gn;
                    if (gn + 4 <= N && (ldn & 3u) == 0u) {
                        v = float4(*(device const half4*)wh);
                    } else {
                        for (uint j = 0; j < 4 && gn + j < N; ++j) {
                            v[j] = (float)wh[j];
                        }
                    }
                } else {
                    device const float* wf =
                        (device const float*)W + (ulong)k * ldn + gn;
                    if (gn + 4 <= N && (ldn & 3u) == 0u) {
                        v = *(device const float4*)wf;
                    } else {
                        for (uint j = 0; j < 4 && gn + j < N; ++j) {
                            v[j] = wf[j];
                        }
                    }
                }
            }
            *(threadgroup float4*)(tg_b + i * MM_NT + e * 4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Accumulate the chunk: 4 sub-steps of 8x8x8 simdgroup mma each.
        for (uint ks = 0; ks < MM_KB; ks += 8) {
            simdgroup_float8x8 a;
            simdgroup_load(a, tg_a + tr * MM_KB + ks, MM_KB);
            for (uint r = 0; r < 4; ++r) {
                simdgroup_float8x8 b;
                simdgroup_load(b, tg_b + ks * MM_NT + tc + r * 8, MM_NT);
                simdgroup_multiply_accumulate(c[r], a, b, c[r]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: the simdgroup's 8x32 result lands in the (now free) tg_b;
    // each lane then writes its pair of elements per staged tile after
    // bias/scale/gelu/residual.
    for (uint r = 0; r < 4; ++r) {
        simdgroup_store(c[r], tg_b + sg * 256 + r * 64, 8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < 4; ++r) {
        const uint gn0 = no + tc + r * 8;
        for (uint e = 0; e < 2; ++e) {
            const uint flat = lane * 2 + e;
            const uint i = flat / 8;
            const uint j = flat % 8;
            const uint gm = mo + tr + i;
            const uint gn = gn0 + j;
            if (gm >= M || gn >= N) { continue; }
            float v = tg_b[sg * 256 + r * 64 + flat];
            if (Q == 1) { v *= tg_s[gn - no]; }
            if (flags & MM_BIAS) { v += bias[gn]; }
            if (flags & MM_GELU) {
                v = 0.5f * v * (1.0f + erf_as(v * 0.70710678f));
            }
            if (flags & MM_RES) { v += res[(ulong)gm * N + gn]; }
            if (flags & MM_QKV) {
                if (gn < HID) {
                    out[(ulong)gm * N + gn] = v;
                } else if (gn < 2 * HID) {
                    out2[((ulong)gm * LAYERS + layer) * HID + gn - HID] = v;
                } else {
                    out3[((ulong)gm * LAYERS + layer) * HID + gn - 2 * HID] = v;
                }
            } else {
                out[(ulong)gm * N + gn] = v;
            }
            if (flags & MM_H16) { out_h[(ulong)gm * N + gn] = half(v); }
        }
    }
}

// ---- paged attention --------------------------------------------------------
// One threadgroup per row, one simdgroup per head: a lane holds the head's
// dims lane*2/lane*2+1, scores its share of the depth+1 positions (t == depth
// is the row's fresh key, read from the QKV kernel's next_k/next_v output),
// and the softmax's normalised probs share through `sc`. Masked positions
// score -INFINITY so their p is 0 and the AV pass never touches their pages.
kernel void lm_attn(
    device float*       qkv    [[buffer(0)]],   // [rows, 3*HID]: q in, av out
    device const float* keys   [[buffer(1)]],   // [pages, L, NH*HD]
    device const float* values [[buffer(2)]],
    device const float* skeys  [[buffer(3)]],   // [slots, L, NH, T, HD]
    device const float* svals  [[buffer(4)]],
    device const uint*  pages  [[buffer(5)]],   // [rows, width]
    device const uchar* mask   [[buffer(6)]],   // [rows, width]
    device const int*   srow   [[buffer(7)]],   // [rows]: -1 pool, else slot
    device const uint*  depths [[buffer(8)]],   // [rows] history length
    device const float* next_k [[buffer(9)]],   // [rows, L, NH*HD] fresh K
    device const float* next_v [[buffer(10)]],  // [rows, L, NH*HD] fresh V
    device float*       sc     [[buffer(11)]],  // [rows, NH, width+1]
    constant uint&      width  [[buffer(12)]],
    constant uint&      layer  [[buffer(13)]],
    device half*        a16    [[buffer(14)]],  // [rows, HID] fp16 av
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgid.x;
    const uint head = sg;
    const uint depth = depths[row];
    const uint pg = row * width;
    const uint scb = (row * NH + head) * (width + 1);
    const int src = srow[row];
    const ulong kvbase = ((ulong)row * LAYERS + layer) * HID + head * HD;
    const uint d2 = lane * 2;
    device float* qp = qkv + (ulong)row * 3 * HID + head * HD;

    // Positions split across the lanes (t = lane + 32*j); each lane carries
    // the full 64-dim q·k for the positions it owns, since a dot product
    // cannot split across lanes without a per-position reduction.
    float m = -INFINITY;
    for (uint t = lane; t <= depth; t += 32) {
        float s = -INFINITY;
        if (t == depth) {
            device const float* k = next_k + kvbase;
            float acc = 0.0f;
            for (uint d = 0; d < HD; ++d) { acc += qp[d] * k[d]; }
            s = acc * INV_SQRT_HD;
        } else if (mask[pg + t] != 0) {
            device const float* k;
            if (src < 0 || pages[pg + t] != 0xffffffffu) {
                k = keys
                    + (((ulong)pages[pg + t] * LAYERS + layer) * NH + head)
                        * HD;
            } else {
                k = skeys
                    + ((((ulong)src * LAYERS + layer) * NH + head) * width
                        + t) * HD;
            }
            float acc = 0.0f;
            for (uint d = 0; d < HD; ++d) { acc += qp[d] * k[d]; }
            s = acc * INV_SQRT_HD;
        }
        sc[scb + t] = s;
        m = max(m, s);
    }
    m = simd_max(m);
    float tot = 0.0f;
    for (uint t = lane; t <= depth; t += 32) {
        const float p = exp(sc[scb + t] - m);
        sc[scb + t] = p;
        tot += p;
    }
    tot = simd_sum(tot);
    const float inv = 1.0f / tot;
    for (uint t = lane; t <= depth; t += 32) {
        sc[scb + t] *= inv;
    }
    // The AV pass reads every lane's share — the device-memory barrier makes
    // the normalised probs visible across the simdgroup.
    simdgroup_barrier(mem_flags::mem_device);
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (uint t = 0; t <= depth; ++t) {
        const float p = sc[scb + t];
        if (p == 0.0f) { continue; }
        device const float* v;
        if (t == depth) {
            v = next_v + kvbase;
        } else if (src < 0 || pages[pg + t] != 0xffffffffu) {
            v = values
                + (((ulong)pages[pg + t] * LAYERS + layer) * NH + head) * HD;
        } else {
            v = svals
                + ((((ulong)src * LAYERS + layer) * NH + head) * width + t) * HD;
        }
        a0 += p * v[d2];
        a1 += p * v[d2 + 1];
    }
    qp[d2] = a0;
    qp[d2 + 1] = a1;
    // The same av lands fp16 for the out-projection's staged A — the row
    // is threadgroup-owned so no cross-tile coordination is needed.
    a16[(ulong)row * HID + head * HD + d2] = half(a0);
    a16[(ulong)row * HID + head * HD + d2 + 1u] = half(a1);
}

// log-softmax over the vocab, then gather the row's K candidates.
kernel void lm_head_logp(
    device const float* logits [[buffer(0)]],   // [rows, V]
    device const uint*  cand   [[buffer(1)]],   // [rows, K]
    device float*       out    [[buffer(2)]],   // [rows, K]
    constant uint&      vocab  [[buffer(3)]],
    constant uint&      count  [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    const uint row = tgid;
    threadgroup float tg_red[HID];
    float m = -INFINITY;
    for (uint v = tid; v < vocab; v += HID) {
        m = max(m, logits[row * vocab + v]);
    }
    tg_red[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = HID / 2; s > 0; s >>= 1) {
        if (tid < s) { tg_red[tid] = max(tg_red[tid], tg_red[tid + s]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mx = tg_red[0];
    float sum = 0.0f;
    for (uint v = tid; v < vocab; v += HID) {
        sum += exp(logits[row * vocab + v] - mx);
    }
    tg_red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = HID / 2; s > 0; s >>= 1) {
        if (tid < s) { tg_red[tid] += tg_red[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float lse = mx + log(tg_red[0]);
    for (uint i = tid; i < count; i += HID) {
        const uint c = cand[row * count + i];
        out[row * count + i] = logits[row * vocab + c] - lse;
    }
}

template [[host_name("lm_embed_i8")]] kernel void lm_embed<1, 1>(
    device float*, device const void*, device const float*, device const float*,
    device const uint*, device const uint*, device const void*,
    device const float*, device const float*, constant uint&,
    device half*, device atomic_float*, uint, uint, uint, uint);
template [[host_name("lm_embed_f16")]] kernel void lm_embed<0, 0>(
    device float*, device const void*, device const float*, device const float*,
    device const uint*, device const uint*, device const void*,
    device const float*, device const float*, constant uint&,
    device half*, device atomic_float*, uint, uint, uint, uint);
template [[host_name("lm_embed_f32")]] kernel void lm_embed<2, 2>(
    device float*, device const void*, device const float*, device const float*,
    device const uint*, device const uint*, device const void*,
    device const float*, device const float*, constant uint&,
    device half*, device atomic_float*, uint, uint, uint, uint);
template [[host_name("lm_mm_i8")]] kernel void lm_mm<1>(
    device const float*, device const float*, device const void*,
    device const float*, device const float*, device float*,
    device float*, device float*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, device half*,
    uint2, uint2, uint, uint);
template [[host_name("lm_mm_f16")]] kernel void lm_mm<0>(
    device const float*, device const float*, device const void*,
    device const float*, device const float*, device float*,
    device float*, device float*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, device half*,
    uint2, uint2, uint, uint);
template [[host_name("lm_mm_f32")]] kernel void lm_mm<2>(
    device const float*, device const float*, device const void*,
    device const float*, device const float*, device float*,
    device float*, device float*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, device half*,
    uint2, uint2, uint, uint);

// ---- staged-input GEMM ------------------------------------------------------
// `lm_sm` runs C[M, N] = A * W^T over an [N, K] fp16 weight table: one
// simdgroup per 32-column output tile accumulating SM_R8x4 simdgroup
// matrices (one pipeline per SM_R8 in 1..4 — the tile is SM_R8*8 rows),
// K split four ways across the threadgroup's simdgroups, the partials
// reduced through threadgroup memory — the M1-measured shape (one 8x8
// MAC per lane step, each weight half-word fetched once for all rows).
// The grid's y carries SM_R8*8-row blocks, so any M dispatches it.
// A is the producer's fp16 staging buffer (x16/a16/u16), never a separate
// stage dispatch: embed/out/ffn2 write x16, attn writes a16, ffn0 writes
// u16, head_proj writes x16. Layer-norm is applied algebraically — with
// the LN weight folded into the repacked table (W'[n,k] = lnw[k]*W[n,k])
// and c1[n] = sum_k lnw[k]*W[n,k], c2[n] = sum_k lnb[k]*W[n,k] precomputed
// at load, the consumer's epilogue computes rstd*(raw - mean*c1[n]) + c2[n]
// off the row sums the x producers atomicAdd into `stats`. Same epilogue
// family as lm_mm.
constant uint SM_NT = 4u;
constant uint SM_KS = 4u;
// Row fragments per tile (1..4 -> 8..32 rows), one pipeline per value:
// padding every call to 32 rows costs 32x the arithmetic at rows=1 and
// ~45% extra at rows=22, so the encode side picks SM_R8 = ceil(M/8).
constant uint SM_R8 [[function_constant(0)]];

kernel void lm_sm(
    device const half*  A     [[buffer(0)]],   // [M32, K] fp16, staged
    device const half*  W     [[buffer(1)]],   // [N, K] fp16 weights
    device const float* S     [[buffer(2)]],   // [N] scale, when MM_SCALE
    device const float* bias  [[buffer(3)]],
    device float*       res   [[buffer(4)]],
    device float*       out   [[buffer(5)]],
    device float*       out2  [[buffer(6)]],
    device float*       out3  [[buffer(7)]],
    constant uint&      flags [[buffer(8)]],
    constant uint&      K     [[buffer(9)]],
    constant uint&      N     [[buffer(10)]],
    constant uint&      M     [[buffer(11)]],
    constant uint&      layer [[buffer(12)]],  // stats slot on LN/STA ops
    device half*        out_h [[buffer(13)]],  // [M, N] fp16, when MM_H16
    device atomic_float* stats [[buffer(14)]], // [rows, STATS2]
    device const float* cc    [[buffer(15)]],  // [2N] c1|c2, when MM_LN
    threadgroup float*  part  [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgi  [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint col0 = tgid.x * 8u * SM_NT;
    const uint row0 = tgid.y * SM_R8 * 8u;
    const uint kspan = K / SM_KS;
    const uint k0 = sgi * kspan;
    // Row fragments beyond SM_R8 stay zero and are never stored — a
    // function constant cannot size an array, so the declaration keeps
    // the maximum width while every loop bounds at SM_R8.
    simdgroup_float8x8 acc[4][SM_NT];
    for (uint r = 0; r < SM_R8; ++r) {
        for (uint c = 0; c < SM_NT; ++c) {
            acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    for (uint k = k0; k < k0 + kspan; k += 8u) {
        simdgroup_half8x8 a[4];
        for (uint r = 0; r < SM_R8; ++r) {
            simdgroup_load(a[r], A + (ulong)(row0 + r * 8u) * K + k, K);
        }
        for (uint c = 0; c < SM_NT; ++c) {
            simdgroup_half8x8 w;
            simdgroup_load(
                w, W + (ulong)(col0 + c * 8u) * K + k, K, ulong2(0, 0), true);
            for (uint r = 0; r < SM_R8; ++r) {
                simdgroup_multiply_accumulate(acc[r][c], a[r], w, acc[r][c]);
            }
        }
    }
    // Stage the four K-partials — SM_R8 x 4 KB of threadgroup memory.
    const uint tile = SM_R8 * 8u * 8u * SM_NT;
    for (uint r = 0; r < SM_R8; ++r) {
        for (uint c = 0; c < SM_NT; ++c) {
            simdgroup_store(
                acc[r][c], part + sgi * tile + r * 8u * 8u * SM_NT + c * 8u,
                8u * SM_NT);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The 128 threads cover the tile's (row, column) pairs — thread
    // (sgi, lane) owns column `lane` of rows sgi + 4*j — the four
    // partials sum, then the epilogue: algebraic LN off the row's stats,
    // scale, bias, gelu, residual, the QKV split, plus the producer-side
    // writes (fp16 into out_h, row sums into stats).
    float psum[8];
    float psq[8];
    for (uint j = 0; j < 8u; ++j) { psum[j] = 0.0f; psq[j] = 0.0f; }
    uint j = 0u;
    for (uint i = sgi * 32u + lane; i < tile; i += SM_KS * 32u, ++j) {
        float v = part[i] + part[tile + i] + part[2u * tile + i]
                + part[3u * tile + i];
        const uint m = row0 + i / (8u * SM_NT);
        const uint n = col0 + i % (8u * SM_NT);
        if (m >= M || n >= N) { continue; }
        if (flags & MM_LN) {
            // LN(x)*W without touching x: stats[m] holds (Σx, Σx²) the
            // producer atomicAdd'd; c1/c2 fold the LN weights at pack.
            const float sx = atomic_load_explicit(
                stats + (ulong)m * STATS2 + layer * 2u,
                memory_order_relaxed);
            const float ss = atomic_load_explicit(
                stats + (ulong)m * STATS2 + layer * 2u + 1u,
                memory_order_relaxed);
            const float mean = sx / (float)K;
            const float rstd =
                rsqrt(max(ss / (float)K - mean * mean, 0.0f) + EPS);
            v = rstd * (v - mean * cc[n]) + cc[N + n];
        }
        if (flags & MM_SCALE) { v *= S[n]; }
        if (flags & MM_BIAS) { v += bias[n]; }
        if (flags & MM_GELU) {
            v = 0.5f * v * (1.0f + erf_as(v * 0.70710678f));
        }
        if (flags & MM_RES) { v += res[(ulong)m * N + n]; }
        if (flags & MM_STA) {
            psum[j] += v;
            psq[j] += v * v;
        }
        if (flags & MM_QKV) {
            if (n < HID) {
                out[(ulong)m * N + n] = v;
            } else if (n < 2 * HID) {
                out2[((ulong)m * LAYERS + layer) * HID + n - HID] = v;
            } else {
                out3[((ulong)m * LAYERS + layer) * HID + n - 2 * HID] = v;
            }
        } else {
            out[(ulong)m * N + n] = v;
        }
        if (flags & MM_H16) { out_h[(ulong)m * N + n] = half(v); }
    }
    if (flags & MM_STA) {
        // The row sums land on lane 0 — every lane of the simdgroup held
        // one column of the same row — then one atomicAdd per row.
        for (uint r = 0; r < SM_R8 * 2u; ++r) {
            const float sx = simd_sum(psum[r]);
            const float ss = simd_sum(psq[r]);
            const uint m = row0 + sgi + r * 4u;
            if (lane == 0u && m < M) {
                atomic_fetch_add_explicit(
                    stats + (ulong)m * STATS2 + layer * 2u, sx,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(
                    stats + (ulong)m * STATS2 + layer * 2u + 1u, ss,
                    memory_order_relaxed);
            }
        }
    }
}

// ---- M <= 2 GEMV -------------------------------------------------------------
// `lm_smv` is the M <= 2 sibling of `lm_sm`: padding one live row out to
// an 8-row fragment still pays 8x the arithmetic, so a lane owns one
// output column and walks its K-slice in 16-byte loads (the fp16 [N, K]
// table is K-contiguous), broadcast activation rows in registers, K
// split SM_KS ways across the threadgroup's simdgroups with the same
// partials reduce and the same epilogue family as `lm_sm`.
kernel void lm_smv(
    device const half*  A     [[buffer(0)]],   // [32, K] fp16, staged
    device const half*  W     [[buffer(1)]],   // [N, K] fp16 weights
    device const float* S     [[buffer(2)]],   // [N] scale, when MM_SCALE
    device const float* bias  [[buffer(3)]],
    device float*       res   [[buffer(4)]],
    device float*       out   [[buffer(5)]],
    device float*       out2  [[buffer(6)]],
    device float*       out3  [[buffer(7)]],
    constant uint&      flags [[buffer(8)]],
    constant uint&      K     [[buffer(9)]],
    constant uint&      N     [[buffer(10)]],
    constant uint&      M     [[buffer(11)]],
    constant uint&      layer [[buffer(12)]],
    device half*        out_h [[buffer(13)]],
    device atomic_float* stats [[buffer(14)]],
    device const float* cc    [[buffer(15)]],
    threadgroup float*  part  [[threadgroup(0)]], // [M, SM_KS * 32] partials
    uint tgid [[threadgroup_position_in_grid]],
    uint sgi  [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint n = tgid * 32u + lane;
    const uint kspan = K / SM_KS;
    const uint k0 = sgi * kspan;
    float acc[2] = {0.0f, 0.0f};
    if (n < N) {
        for (uint k = k0; k < k0 + kspan; k += 8u) {
            const uint4 wb =
                *(device const uint4*)(W + (ulong)n * K + k);
            const float4 wl = float4(as_type<half4>(wb.xy));
            const float4 wh = float4(as_type<half4>(wb.zw));
            const uint4 xb = *(device const uint4*)(A + k);
            const float4 al = float4(as_type<half4>(xb.xy));
            const float4 ah = float4(as_type<half4>(xb.zw));
            acc[0] += dot(wl, al) + dot(wh, ah);
            if (M > 1u) {
                const uint4 xb1 = *(device const uint4*)(A + K + k);
                const float4 bl = float4(as_type<half4>(xb1.xy));
                const float4 bh = float4(as_type<half4>(xb1.zw));
                acc[1] += dot(wl, bl) + dot(wh, bh);
            }
        }
    }
    part[sgi * 32u + lane] = acc[0];
    part[SM_KS * 32u + sgi * 32u + lane] = acc[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi != 0u) { return; }
    float sums[2];
    for (uint m = 0; m < M; ++m) {
        sums[m] = part[m * SM_KS * 32u + lane]
                + part[m * SM_KS * 32u + 32u + lane]
                + part[m * SM_KS * 32u + 64u + lane]
                + part[m * SM_KS * 32u + 96u + lane];
    }
    // Every lane of simdgroup 0 stays in the loop so the stats'
    // `simd_sum` reduce covers the tile's whole 32-column span; lanes
    // past N carry a zero through the epilogue and skip the stores.
    const bool live = n < N;
    for (uint m = 0; m < M; ++m) {
        float v = 0.0f;
        if (live) {
            v = sums[m];
            if (flags & MM_LN) {
                const float sx = atomic_load_explicit(
                    stats + (ulong)m * STATS2 + layer * 2u,
                    memory_order_relaxed);
                const float ss = atomic_load_explicit(
                    stats + (ulong)m * STATS2 + layer * 2u + 1u,
                    memory_order_relaxed);
                const float mean = sx / (float)K;
                const float rstd =
                    rsqrt(max(ss / (float)K - mean * mean, 0.0f) + EPS);
                v = rstd * (v - mean * cc[n]) + cc[N + n];
            }
            if (flags & MM_SCALE) { v *= S[n]; }
            if (flags & MM_BIAS) { v += bias[n]; }
            if (flags & MM_GELU) {
                v = 0.5f * v * (1.0f + erf_as(v * 0.70710678f));
            }
            if (flags & MM_RES) { v += res[(ulong)m * N + n]; }
        }
        if (flags & MM_STA) {
            const float sx = simd_sum(v);
            const float ss = simd_sum(v * v);
            if (lane == 0u) {
                atomic_fetch_add_explicit(
                    stats + (ulong)m * STATS2 + layer * 2u, sx,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(
                    stats + (ulong)m * STATS2 + layer * 2u + 1u, ss,
                    memory_order_relaxed);
            }
        }
        if (!live) { continue; }
        if (flags & MM_QKV) {
            if (n < HID) {
                out[(ulong)m * N + n] = v;
            } else if (n < 2 * HID) {
                out2[((ulong)m * LAYERS + layer) * HID + n - HID] = v;
            } else {
                out3[((ulong)m * LAYERS + layer) * HID + n - 2 * HID] = v;
            }
        } else {
            out[(ulong)m * N + n] = v;
        }
        if (flags & MM_H16) { out_h[(ulong)m * N + n] = half(v); }
    }
}

"#;

/// One dispatch in the step's fixed sequence — the list is built once at
/// open: embed, then per layer qkv / attn / out / ffn0 / ffn2, then the
/// head-projection, the tied head, and the candidate gather.
#[derive(Debug, Clone, Copy)]
enum Op {
    /// `lm_embed` — the token's embed + input projection + position.
    Embed,
    /// `lm_mm` — LN1 + the QKV projection, the K/V halves split into `next_*`.
    MmQkv(u32),
    /// `lm_attn` — the row's paged attention over pool/scratch/fresh keys.
    Attn(u32),
    /// `lm_mm` — the attention out-projection + residual.
    MmOut(u32),
    /// `lm_mm` — LN2 + ff.0 + gelu.
    MmFfn0(u32),
    /// `lm_mm` — ff.2 + residual.
    MmFfn2(u32),
    /// `lm_mm<f32>` — the final norm + projection into `g_y` (fp32 weights).
    MmHeadProj,
    /// `lm_mm` — the tied head's full-vocab logits.
    MmHeadMat,
    /// `lm_head_logp` — log-softmax + the candidate gather.
    HeadLogp,
}

impl Op {
    /// The profile row's label.
    fn name(self) -> String {
        match self {
            Op::Embed => "embed".to_owned(),
            Op::MmQkv(l) | Op::Attn(l) | Op::MmOut(l) | Op::MmFfn0(l) | Op::MmFfn2(l) => {
                let kind = match self {
                    Op::MmQkv(_) => "qkv",
                    Op::Attn(_) => "attn",
                    Op::MmOut(_) => "out",
                    Op::MmFfn0(_) => "ffn0",
                    _ => "ffn2",
                };
                format!("l{l:02}.{kind}")
            }
            Op::MmHeadProj => "head_proj".to_owned(),
            Op::MmHeadMat => "head_mat".to_owned(),
            Op::HeadLogp => "head_logp".to_owned(),
        }
    }
}

/// The epilogue flags `lm_mm` takes.
const MM_LN: u32 = 1;
/// As `MM_LN`.
const MM_GELU: u32 = 2;
/// As `MM_LN`.
const MM_RES: u32 = 4;
/// As `MM_LN`.
const MM_QKV: u32 = 8;
/// As `MM_LN`.
const MM_BIAS: u32 = 16;
/// `lm_sm` — multiply the reduced partial by `S[n]`; set only for ops
/// whose scale buffer is real (the int8 pack).
const MM_SCALE: u32 = 32;
/// Write the epilogue's fp16 output into `out_h` — the staged A a
/// consuming `lm_sm` reads.
const MM_H16: u32 = 64;
/// Accumulate per-row (Σv, Σv²) into `stats` at the slot `layer` names —
/// the x-producers (out, ffn2) feed the consumers' algebraic LN.
const MM_STA: u32 = 128;
/// The stats buffer's per-row float stride: slots 0..2*layers+1, two
/// floats each — must match the kernel's `STATS2`.
const STATS2: usize = (2 * 12 + 1) * 2;

/// Everything an `lm_mm` dispatch binds: the pipeline, the nine buffer
/// arguments (with byte offsets for the slabbed weights), the flags, and the
/// K/N/lda/ldn shape scalars.
struct MmSpec<'a> {
    /// The pipeline — `p_mm` or `p_mm_f32`.
    pipe: &'a ComputePipelineState,
    /// `wh` — the `[N, K]` fp16 slab `lm_sm` binds instead of `w` (LN
    /// weight folded in for the LN-consuming ops), or `None` when this op
    /// has no repack and stays on the tiled kernel.
    wh: Option<&'a Buffer>,
    /// `a16` — the fp16 `[rows, K]` staging buffer `lm_sm` reads as A:
    /// `x16` for the residual consumers, `a16` for the out projection,
    /// `u16` for ffn2.
    a16: &'a Buffer,
    /// `cc` — the `[2N]` per-layer (c1, c2) slab the algebraic LN reads,
    /// at `cc_off` — or the dummy when the op has no LN.
    cc: &'a Buffer,
    /// As `ln_off`.
    cc_off: u64,
    /// `out_h` — the fp16 `[rows, N]` buffer the epilogue writes under
    /// `MM_H16` for the op's own consumer; the dummy otherwise.
    out_h: &'a Buffer,
    /// A — the `[M, K]` input, `lda`-strided.
    a: &'a Buffer,
    /// `ln` — `[2K]` w||b — and its byte offset.
    ln: &'a Buffer,
    /// As `ln`.
    ln_off: u64,
    /// `W` — the packed `[K, N]` slab — and its byte offset.
    w: &'a Buffer,
    /// As `ln_off`.
    w_off: u64,
    /// `S` — the `[N]` scale — and its byte offset.
    s: &'a Buffer,
    /// As `ln_off`.
    s_off: u64,
    /// `bias` and its byte offset.
    bias: &'a Buffer,
    /// As `ln_off`.
    bias_off: u64,
    /// `res`.
    res: &'a Buffer,
    /// `out`.
    out: &'a Buffer,
    /// `out2`.
    out2: &'a Buffer,
    /// `out3`.
    out3: &'a Buffer,
    /// `flags`.
    flags: u32,
    /// `K`.
    k: u32,
    /// `N`.
    n: u32,
    /// `lda`/`ldn`/`layer` ride along.
    lda: u32,
    /// As `lda`.
    ldn: u32,
    /// As `lda`.
    layer: u32,
}

/// The per-call dimensions every encoder shares.
#[derive(Debug, Clone, Copy)]
struct StepCtx {
    /// The batch's row count.
    rows: usize,
    /// The candidate count per row.
    count: usize,
    /// The history axis — the page row's width.
    width: usize,
}

/// How the step's matmul weights are packed — picked once for the whole model
/// from the export's own tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightKind {
    /// fp16 — packed at load: int8 entries dequantize, fp32 entries cast;
    /// no scale buffers reach the kernel.
    F16,
    /// `onnx::MatMul_*_int8` `[in, out]` i8 + a per-out fp32 scale.
    Int8,
    /// `model.*` fp32, transposed to `[in, out]` at pack.
    Float32,
}

/// One tensor's bytes inside the weights map, bounds-checked.
fn tensor_bytes<'a>(
    map: &'a [u8],
    table: &WeightsFile,
    tensor: &WeightTensor,
) -> Result<&'a [u8], LmError> {
    let mismatched = |reason: String| LmError::Weights {
        path: table.file.clone().into(),
        reason,
    };
    let start = usize::try_from(tensor.offset)
        .map_err(|_| mismatched(format!("the offset of {} overflows usize", tensor.name)))?;
    let end = usize::try_from(tensor.length)
        .ok()
        .and_then(|length| start.checked_add(length))
        .ok_or_else(|| mismatched(format!("the extent of {} overflows usize", tensor.name)))?;
    map.get(start..end).ok_or_else(|| {
        mismatched(format!(
            "the extent of {} runs past {} bytes",
            tensor.name,
            map.len()
        ))
    })
}

/// The tensors named *name*, or an error listing what was found instead.
fn tensor_named<'t>(table: &'t WeightsFile, name: &str) -> Result<&'t WeightTensor, LmError> {
    table
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| LmError::Weights {
            path: table.file.clone().into(),
            reason: format!("the table has no {name}"),
        })
}

/// A `Buffer` holding *bytes* on the shared heap.
fn buffer_of(device: &Device, bytes: &[u8]) -> Buffer {
    device.new_buffer_with_data(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// The `[N, K]` fp16 repack `lm_sm` binds: a simdgroup loads each 8x8
/// weight tile transposed straight from device memory. *w* is the
/// `[inner, out]` table — int8 values carried as fp16 (the scale still
/// folds in the epilogue) under the int8 pack, fp16 transposed under
/// the fp16 pack. With `ln` — the consuming LN's `(weight, bias)` — the
/// weight multiplies into the table (`W'[n,k] = lnw[k] * W[n,k]`) and the
/// constants the algebraic epilogue needs come back as the second tuple
/// element: `c1[n] = Σk lnw[k]·W[n,k]` then `c2[n] = Σk lnb[k]·W[n,k]`,
/// `[2*out]` fp32. `None` when the shape can't fill the fixed 32-column,
/// 8-deep tiles — the op then keeps the tiled kernel.
fn nk_repacked_f16(
    device: &Device,
    w: &Buffer,
    inner: usize,
    out: usize,
    i8_source: bool,
    ln: Option<(&[f32], &[f32])>,
) -> (Option<Buffer>, Option<Vec<f32>>) {
    if !inner.is_multiple_of(32) {
        return (None, None);
    }
    // Output rows pad to the 32-column tile; the kernel's epilogue
    // guards `n < N` so the pad never lands in the output.
    let out_p = out.next_multiple_of(32);
    let buf = device.new_buffer(
        u64::try_from(inner * out_p * 2).expect("a repacked slab's size fits u64"),
        MTLResourceOptions::StorageModeShared,
    );
    let dst = buf.contents().cast::<u16>();
    for n in out..out_p {
        for k in 0..inner {
            // Safety: dst owns `inner * out_p` fp16 elements.
            unsafe { *dst.add(n * inner + k) = 0 };
        }
    }
    let mut cc = ln.map(|_| vec![0.0f32; 2 * out]);
    let lnw = ln.map(|pair| pair.0);
    let lnb = ln.map(|pair| pair.1);
    if i8_source {
        let src = w.contents().cast::<u8>();
        for n in 0..out {
            let mut c1 = 0.0f32;
            let mut c2 = 0.0f32;
            for k in 0..inner {
                // Safety: src owns `inner * out` int8 elements — the
                // [inner, out] table; dst its padded fp16 transpose.
                let v = f32::from(unsafe { *src.add(k * out + n) }.cast_signed());
                if let (Some(w_ln), Some(b_ln)) = (lnw, lnb) {
                    c1 += w_ln[k] * v;
                    c2 += b_ln[k] * v;
                    let folded = w_ln[k] * v;
                    unsafe {
                        *dst.add(n * inner + k) = half::f16::from_f32(folded).to_bits();
                    }
                } else {
                    unsafe {
                        *dst.add(n * inner + k) = half::f16::from_f32(v).to_bits();
                    }
                }
            }
            if let Some(cc) = cc.as_mut() {
                cc[n] = c1;
                cc[out + n] = c2;
            }
        }
    } else {
        let src = w.contents().cast::<u16>();
        for n in 0..out {
            let mut c1 = 0.0f32;
            let mut c2 = 0.0f32;
            for k in 0..inner {
                // Safety: both buffers own `inner * out` fp16 elements.
                let v = f32::from(half::f16::from_bits(unsafe { *src.add(k * out + n) }));
                let folded = if let (Some(w_ln), Some(b_ln)) = (lnw, lnb) {
                    c1 += w_ln[k] * v;
                    c2 += b_ln[k] * v;
                    w_ln[k] * v
                } else {
                    v
                };
                unsafe {
                    *dst.add(n * inner + k) = half::f16::from_f32(folded).to_bits();
                }
            }
            if let Some(cc) = cc.as_mut() {
                cc[n] = c1;
                cc[out + n] = c2;
            }
        }
    }
    (Some(buf), cc)
}

/// Advise a tensor's mmap range out of the resident set — its bytes were
/// read once into buffers and repacks at open and never again. The range
/// snaps to whole pages inside `[offset, offset + length)`: a table shorter
/// than a page shares one with live data and stays resident.
fn advise_cold(map: &[u8], offset: u64, length: u64) {
    const PAGE: u64 = 4096;
    let lo = offset.next_multiple_of(PAGE);
    let hi = offset.saturating_add(length) / PAGE * PAGE;
    if hi <= lo {
        return;
    }
    let Ok(start) = usize::try_from(lo) else {
        return;
    };
    let Ok(len) = usize::try_from(hi - lo) else {
        return;
    };
    if start.saturating_add(len) > map.len() {
        return;
    }
    // Safety: [start, start + len) lies inside the map and stays mapped —
    // MADV_DONTNEED drops the resident pages only.
    unsafe {
        libc::madvise(
            map.as_ptr()
                .wrapping_add(start)
                .cast_mut()
                .cast::<libc::c_void>(),
            len,
            libc::MADV_DONTNEED,
        );
    }
}

/// A `Buffer` holding *values*, fp32 — biases, norms, positions and the
/// fp32-path weights.
fn buffer_of_f32(device: &Device, values: &[f32]) -> Buffer {
    buffer_of(
        device,
        // Safety: f32 is plain data — 4-byte LE on every target this builds on.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) },
    )
}

/// *src* `[out, in]` fp32 transposed into `[in, out]` — the layout `ldw`
/// strides so consecutive output columns read consecutively.
fn transposed_f32(src: &[f32], out: usize, inner: usize) -> Vec<f32> {
    let mut dst = vec![0f32; out * inner];
    for o in 0..out {
        for i in 0..inner {
            dst[i * out + o] = src[o * inner + i];
        }
    }
    dst
}

/// *src* `[out, in]` fp32 transposed and cast into `[in, out]` fp16 bytes.
fn transposed_f16(src: &[f32], out: usize, inner: usize) -> Vec<u8> {
    let mut dst = Vec::with_capacity(out * inner * 2);
    for i in 0..inner {
        for o in 0..out {
            dst.extend_from_slice(&half::f16::from_f32(src[o * inner + i]).to_le_bytes());
        }
    }
    dst
}

/// An int8 `[in, out]` tensor dequantized by its per-out scale into fp16
/// bytes — the same values, in the packed format `ldw<0>` reads.
fn f16_of_i8(w: &[u8], scale: &[f32], inner: usize, out: usize) -> Vec<u8> {
    let mut dst = Vec::with_capacity(inner * out * 2);
    for i in 0..inner {
        for o in 0..out {
            dst.extend_from_slice(
                &half::f16::from_f32(f32::from(w[i * out + o].cast_signed()) * scale[o])
                    .to_le_bytes(),
            );
        }
    }
    dst
}

/// The byte slice's fp32 contents — the weights file is little-endian and so
/// is every supported host.
fn f32s(bytes: &[u8]) -> &[f32] {
    bytemuck::cast_slice(bytes)
}

/// The resident pool's two `MTLBuffer`s — `keys` and `values`, `[pages, L,
/// H*D]` fp32 shared-memory — plus the slice accessors the host-side page
/// writes and pending materialisations run through. `Clone`d handles share
/// the one allocation: the `PagePool` owns one for `write_page`, the
/// `MetalStep` clones one to bind into its kernels.
#[derive(Debug, Clone)]
pub struct PoolBuffers {
    /// `keys[page, layer, head*head_dim + dim]` as a flat f32 run.
    keys: Buffer,
    /// `values[page, ..]` — same layout.
    values: Buffer,
}

/// A `step`'s wall-clock split — the three suspects behind the ~0.14 ms
/// per-dispatch floor: host encode, `commit` to GPU-scheduled, and
/// scheduled to completed.
#[derive(Debug, Default, Clone, Copy)]
pub struct StepTiming {
    /// Host nanoseconds in the encode loop — command buffer creation,
    /// per-op `encode_op`, `end_encoding`.
    pub encode_ns: u64,
    /// `commit()` to `wait_until_scheduled()` — submission latency.
    pub submit_ns: u64,
    /// `wait_until_scheduled()` to `wait_until_completed()` — the GPU's span.
    pub gpu_ns: u64,
}

impl StepTiming {
    /// Milliseconds for a nanosecond field.
    #[must_use]
    pub fn encode_ms(&self) -> f64 {
        f64::from(u32::try_from(self.encode_ns).unwrap_or(u32::MAX)) / 1e6
    }

    /// [`Self::encode_ms`] for `submit_ns`.
    #[must_use]
    pub fn submit_ms(&self) -> f64 {
        f64::from(u32::try_from(self.submit_ns).unwrap_or(u32::MAX)) / 1e6
    }

    /// [`Self::encode_ms`] for `gpu_ns`.
    #[must_use]
    pub fn gpu_ms(&self) -> f64 {
        f64::from(u32::try_from(self.gpu_ns).unwrap_or(u32::MAX)) / 1e6
    }
}

impl PoolBuffers {
    /// A zeroed pool of *pages* pages on the system device, `[pages, layers,
    /// hidden]` fp32 per tensor.
    ///
    /// # Errors
    ///
    /// If no Metal device answers.
    pub fn new(pages: usize, layers: usize, hidden: usize) -> Result<Self, LmError> {
        let device = Device::system_default().ok_or(LmError::MetalUnavailable)?;
        let bytes = pages * layers * hidden * 4;
        let alloc = |len: usize| -> Buffer {
            let size = u64::try_from(len).expect("a pool size fits u64");
            let buffer = device.new_buffer(size, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, len);
            }
            buffer
        };
        Ok(Self {
            keys: alloc(bytes),
            values: alloc(bytes),
        })
    }

    /// Buffer *index*'s contents as a flat f32 slice — the layout the pool's
    /// `slice_at`/`write_page` index math is written against.
    ///
    /// # Safety
    ///
    /// `contents()` hands the shared-mode buffer's CPU pointer; the slice is
    /// `&mut` because Metal shares the mapping — every writer here serialises
    /// behind the pool's own locks and the `advance` lockstep, so the mutable
    /// alias never overlaps a concurrent host read. The GPU kernels read the
    /// same bytes inside a committed command buffer, after the host writes
    /// land.
    #[expect(
        clippy::mut_from_ref,
        reason = "a shared-mode MTLBuffer exposes one CPU mapping; the pool's locks give the write alias"
    )]
    pub fn slice_mut(&self, index: usize) -> &mut [f32] {
        let buffer = if index == 0 { &self.keys } else { &self.values };
        // Safety: shared-mode buffer; see the type-level note.
        unsafe {
            std::slice::from_raw_parts_mut(
                buffer.contents().cast::<f32>(),
                usize::try_from(buffer.length()).expect("a buffer length fits usize") / 4,
            )
        }
    }
}

/// The buffers the step needs: weights packed for its dtype, the resident
/// pool, the scratch generation the pending rows materialise into, the host
/// inputs as shared buffers, and the scratch/output rows.
pub struct MetalStep {
    /// The GPU this step encodes on.
    device: Device,
    /// Its queue: one command buffer per call.
    queue: CommandQueue,
    /// The embed/in-projection kernel for the weights' dtype.
    p_embed: ComputePipelineState,
    /// The tiled GEMM for the weights' dtype — every matmul in the step.
    p_mm: ComputePipelineState,
    /// `lm_mm<f32>` — the head projection, whose weight stays fp32.
    p_mm_f32: ComputePipelineState,
    /// `lm_sm` — the staged simdgroup-matrix kernel over fp16 `[N, K]`
    /// weights at any M (the grid's y carries 32-row blocks); its A is
    /// whatever the producing op already wrote fp16.
    /// `lm_sm` specialised on the tile's row count — `p_sm[r - 1]` computes
    /// r*8 rows per threadgroup.
    p_sm: [ComputePipelineState; 4],
    /// The M <= 2 GEMV sibling — a 32-row tile would pay 16-32x the live
    /// arithmetic, so one lane per output column walks the K-slices.
    p_smv: ComputePipelineState,
    /// The paged-attention kernel — one threadgroup per row, one simdgroup
    /// per head.
    p_attn: ComputePipelineState,
    /// The log-softmax gather over `candidates`.
    p_head_logp: ComputePipelineState,
    /// The pool the kernels read — `keys`/`values` `[pages, L, H*D]` f32.
    /// Clones of the `PagePool`'s own buffers, which own the allocation.
    pool_keys: Buffer,
    /// As `pool_keys`.
    pool_values: Buffer,
    /// The scratch generation for pending rows, grown on demand —
    /// `skeys`/`svals` `[slots, L, H, tmax, D]` f32.
    skeys: Buffer,
    /// As `skeys`.
    svals: Buffer,
    /// The scratch generation's allocated shape: slots × width.
    scratch_slots: usize,
    /// Its T axis.
    scratch_width: usize,
    /// Per-call host inputs, grown on demand.
    token: Buffer,
    /// As `token`.
    pages: Buffer,
    /// As `token`.
    mask: Buffer,
    /// As `token`.
    srow: Buffer,
    /// As `token`.
    depths: Buffer,
    /// As `token`.
    cand: Buffer,
    /// `(rows, width, count)` the input buffers were sized for.
    input_shape: (usize, usize, usize),
    /// The step's intermediates — `x`, the LN output, the fused QKV row, the
    /// FFN row — plus the per-row scores tile, the tied logits, the fresh
    /// `next_*` K/V, and the gathered log-probs.
    x: Buffer,
    /// As `x`.
    g_qkv: Buffer,
    /// As `x`.
    g_ff: Buffer,
    /// As `x`.
    g_y: Buffer,
    /// As `x`.
    logits: Buffer,
    /// The `(row, head, t)` score tile `[rows, NH, width+1]`.
    g_sc: Buffer,
    /// The fresh K/V outputs `[rows, L, H*D]`.
    next_k: Buffer,
    /// As `next_k`.
    next_v: Buffer,
    /// The gathered log-probs `[rows, count]`.
    logp: Buffer,
    /// `(rows, width, count)` the scratch/output buffers were sized for.
    scratch_shape: (usize, usize, usize),
    /// Embedding + tied head table `[H, V]` in the weights' dtype.
    w_embed: Buffer,
    /// `embed_f16` — the same table as fp16 `[V, H]` for `lm_sm` and the
    /// `lm_mm` head matmul.
    embed_f16: Option<Buffer>,
    /// Its per-out scale, or the one-element dummy when the dtype is f32.
    s_embed: Buffer,
    /// `input.weight` `[H, H]` (transposed `[in, out]`), same dtype.
    w_in: Buffer,
    /// As `s_embed`.
    s_in: Buffer,
    /// `input.bias` `[H]` fp32.
    b_in: Buffer,
    /// `positions.weight` `[T, H]` fp32.
    pos: Buffer,
    /// `qkv.weight` `[L*H, 3*H]` in the weights' dtype — the four-byte
    /// dummy when `lm_sm` owns every matmul and the packed slabs were
    /// never built.
    w_qkv: Buffer,
    /// `wh_qkv` — the fp16 `[3*H, H]` repack `lm_sm` binds, LN1's weight
    /// folded in.
    wh_qkv: Option<Buffer>,
    /// Per-out scale `[L*3*H]` or the dummy.
    s_qkv: Buffer,
    /// `qkv.bias` `[L*3*H]` fp32.
    b_qkv: Buffer,
    /// `out.weight` `[L*H, H]`.
    w_out: Buffer,
    /// As `wh_qkv`, repacked `[H, H]` — the out projection's input is not
    /// LN'd, so nothing folds.
    wh_out: Option<Buffer>,
    /// As `s_qkv`.
    s_out: Buffer,
    /// `out.bias` `[L*H]` fp32.
    b_out: Buffer,
    /// `ff.0.weight` `[L*H, NFF]`.
    w_ff0: Buffer,
    /// As `wh_qkv`, repacked `[NFF, H]` with LN2's weight folded in.
    wh_ff0: Option<Buffer>,
    /// As `s_qkv`.
    s_ff0: Buffer,
    /// `ff.0.bias` `[L*NFF]` fp32.
    b_ff0: Buffer,
    /// `ff.2.weight` `[L*NFF, H]`.
    w_ff2: Buffer,
    /// As `wh_qkv`, repacked `[H, NFF]` — ffn2's input is ffn0's gelu,
    /// not LN'd.
    wh_ff2: Option<Buffer>,
    /// As `s_qkv`.
    s_ff2: Buffer,
    /// `ff.2.bias` `[L*H]` fp32.
    b_ff2: Buffer,
    /// `norm1/norm2` weights and biases interleaved `[L, 4*H]` fp32.
    ln_wb: Buffer,
    /// `norm.weight || norm.bias` `[2*H]` fp32.
    ln_f: Buffer,
    /// `project.weight` transposed `[H, H]` fp32.
    w_head: Buffer,
    /// `project.bias` `[H]` fp32.
    b_head: Buffer,
    /// The weight element size in bytes — int8 1, fp16 2, fp32 4 — for the
    /// per-layer byte offsets into each slab.
    w_elem: u64,
    /// A four-byte buffer bound wherever a kernel argument is unused.
    dummy: Buffer,
    /// The step's dispatch list, built once at open.
    ops: Vec<Op>,
    /// `cc_qkv` — the algebraic-LN constants `[L][2*3H]`: c1 then c2 for
    /// each layer's qkv input — or the dummy when no repack exists.
    cc_qkv: Buffer,
    /// As `cc_qkv`, `[L][2*NFF]` for ffn0's LN2 input.
    cc_ff0: Buffer,
    /// The dispatch-list truncation `set_step_limit` set — metal-bench's
    /// `--only-ops` measurement leg.
    step_limit: Option<usize>,
    /// `x16` — the residual stream as raw fp16 `[rows32, H]`, written by
    /// the embed, out-projection, and ffn2 ops (and `head_proj`'s `g_y` for
    /// `head_mat`), read by `lm_sm`'s A operand.
    x16: Buffer,
    /// `a16` — attention's av output as fp16 `[rows32, H]`, read by the
    /// out projection's A.
    a16: Buffer,
    /// `u16` — ffn0's gelu output as fp16 `[rows32, NFF]`, read by ffn2's
    /// A.
    u16: Buffer,
    /// `st` — the row-sum accumulators `[rows, 2L+1]` float2 slots: embed
    /// zeroes and fills slot 0, out atomicAdds LAYERS+1+l, ffn2 l+1, and
    /// the LN consumers read them — `metal_atomic` types in the kernels.
    st: Buffer,
    /// `set_profiling` — when on, `step` records per-dispatch GPU ms.
    profiling: bool,
    /// The `(op, ms, weight-bytes)` rows the last profiled `step` collected.
    last_profile: Vec<(String, f64, u64)>,
    /// The last `step`'s host/GPU wall split — [`Self::timing`].
    last_timing: StepTiming,
    /// `layers`/`heads`/`head_dim`/`vocab`/`hidden`/`ff` of the manifest —
    /// asserted equal to the kernel's compile-time values.
    layers: usize,
    /// As `layers`.
    hidden: usize,
    /// As `layers`.
    vocab: usize,
    /// `ff` — the feed-forward width the manifest recorded.
    feedforward: usize,
}

impl MetalStep {
    /// Compile the kernels, pack *table*'s tensors into buffers for them,
    /// and allocate the resident pool of *pages* pages — the step's own
    /// handle, with [`Self::pool_buffers`] handing the `PagePool` its clone.
    ///
    /// # Errors
    ///
    /// If no Metal device answers, the kernels do not compile, the manifest's
    /// row shapes are not the transformer's `12×512×8×64`, or a tensor the
    /// table names is missing, mistyped or misshaped.
    #[expect(
        clippy::too_many_lines,
        reason = "one pass wires ~60 named buffers into the step; splitting it adds indirection, not clarity"
    )]
    pub fn new(
        state_rows: &[Vec<i64>],
        pages: usize,
        table: &WeightsFile,
        map: &[u8],
        weights: MetalWeights,
    ) -> Result<Self, LmError> {
        type LayerWeights<'t> = (
            Buffer,
            Buffer,
            Buffer,
            Buffer,
            Buffer,
            usize,
            Vec<(
                &'t WeightTensor,
                &'t WeightTensor,
                &'t WeightTensor,
                &'t WeightTensor,
            )>,
        );
        let mismatched = |reason: String| LmError::Weights {
            path: table.file.clone().into(),
            reason,
        };
        let row = state_rows
            .first()
            .ok_or_else(|| mismatched("the manifest names no state tensors".to_owned()))?;
        let layers = usize::try_from(row[1])
            .map_err(|_| mismatched("the state rows' layer axis is negative".to_owned()))?;
        let heads = usize::try_from(row[2])
            .map_err(|_| mismatched("the state rows' head axis is negative".to_owned()))?;
        let head_dim = usize::try_from(row[3])
            .map_err(|_| mismatched("the state rows' dim axis is negative".to_owned()))?;
        let hidden = heads * head_dim;
        if !(layers == 12 && heads == HEADS && head_dim == HEAD_DIM && hidden == HIDDEN) {
            return Err(mismatched(format!(
                "the kernels are compiled for 12×{HIDDEN}×{HEADS}×{HEAD_DIM}; the manifest gives {layers}×{hidden}×{heads}×{head_dim}"
            )));
        }
        let device = Device::system_default().ok_or(LmError::MetalUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(STEP_SOURCE, &options)
            .map_err(LmError::Metal)?;
        let i8_layout = table
            .tensors
            .iter()
            .any(|tensor| tensor.dtype == WeightDtype::Int8);
        let kind = match weights {
            MetalWeights::Auto if i8_layout => WeightKind::Int8,
            MetalWeights::Auto => WeightKind::Float32,
            MetalWeights::F16 => WeightKind::F16,
        };
        let named = |name: &str| -> Result<&WeightTensor, LmError> { tensor_named(table, name) };
        let suffix = match kind {
            WeightKind::F16 => "f16",
            WeightKind::Int8 => "i8",
            WeightKind::Float32 => "f32",
        };
        let function = |name: &str| -> Result<metal::Function, LmError> {
            library
                .get_function(name, None)
                .map_err(|e| LmError::Metal(format!("{name}: {e}")))
        };
        let pipeline = |name: &str| -> Result<ComputePipelineState, LmError> {
            let function = function(name)?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(LmError::Metal)
        };
        let dummy = buffer_of_f32(&device, &[0.0f32]);
        // The per-matmul pair: the weight buffer in the step's dtype and its
        // scale, or the dummy when the dtype has no scale.
        let pair = |w: &[u8], scale: Option<&[u8]>| -> (Buffer, Buffer) {
            (
                buffer_of(&device, w),
                scale.map_or_else(|| dummy.clone(), |s| buffer_of(&device, s)),
            )
        };
        // A matmul tensor's `[in, out]` bytes: int8 already carries them;
        // fp32 comes as torch's `[out, in]` and transposes at pack; fp16
        // converts either to the same packed `[in, out]` layout.
        let matrix = |tensor: &WeightTensor,
                      inner: usize,
                      out: usize|
         -> Result<(Buffer, Buffer), LmError> {
            match kind {
                WeightKind::Int8 => {
                    if tensor.dtype != WeightDtype::Int8 {
                        return Err(mismatched(format!("{} is not int8", tensor.name)));
                    }
                    if tensor.shape != vec![dim(inner), dim(out)] {
                        return Err(mismatched(format!(
                            "{} is {:?}, not [{inner}, {out}]",
                            tensor.name, tensor.shape
                        )));
                    }
                    let scale = tensor_named(table, &tensor.name.replace("_int8", "_scale"))?;
                    Ok(pair(
                        tensor_bytes(map, table, tensor)?,
                        Some(tensor_bytes(map, table, scale)?),
                    ))
                }
                WeightKind::F16 => match tensor.dtype {
                    WeightDtype::Int8 => {
                        if tensor.shape != vec![dim(inner), dim(out)] {
                            return Err(mismatched(format!(
                                "{} is {:?}, not [{inner}, {out}]",
                                tensor.name, tensor.shape
                            )));
                        }
                        let scale = tensor_named(table, &tensor.name.replace("_int8", "_scale"))?;
                        Ok((
                            buffer_of(
                                &device,
                                &f16_of_i8(
                                    tensor_bytes(map, table, tensor)?,
                                    f32s(tensor_bytes(map, table, scale)?),
                                    inner,
                                    out,
                                ),
                            ),
                            dummy.clone(),
                        ))
                    }
                    WeightDtype::Float32 => {
                        if tensor.shape != vec![dim(out), dim(inner)] {
                            return Err(mismatched(format!(
                                "{} is {:?}, not [{out}, {inner}]",
                                tensor.name, tensor.shape
                            )));
                        }
                        Ok((
                            buffer_of(
                                &device,
                                &transposed_f16(
                                    f32s(tensor_bytes(map, table, tensor)?),
                                    out,
                                    inner,
                                ),
                            ),
                            dummy.clone(),
                        ))
                    }
                    dtype => Err(mismatched(format!(
                        "{} is {dtype:?}; fp16 packs int8 or fp32",
                        tensor.name
                    ))),
                },
                WeightKind::Float32 => {
                    if tensor.dtype != WeightDtype::Float32 {
                        return Err(mismatched(format!("{} is not float32", tensor.name)));
                    }
                    if tensor.shape != vec![dim(out), dim(inner)] {
                        return Err(mismatched(format!(
                            "{} is {:?}, not [{out}, {inner}]",
                            tensor.name, tensor.shape
                        )));
                    }
                    Ok((
                        buffer_of_f32(
                            &device,
                            &transposed_f32(f32s(tensor_bytes(map, table, tensor)?), out, inner),
                        ),
                        dummy.clone(),
                    ))
                }
            }
        };
        // The int8 export names its matmuls `onnx::MatMul_<id>_int8`: the
        // first by id is the input projection, each block then adds its four
        // in order (qkv, out, ff.0, ff.2), and the last is the tied table.
        // `cold` collects the int8 tables' map ranges — they are read into
        // buffers and repacks below, and the mmap pages are advised cold
        // before the struct returns so they stop occupying the resident set.
        let mut cold: Vec<(u64, u64)> = Vec::new();
        let (w_embed, s_embed, w_in, s_in, b_in, vocab, by_layer): LayerWeights<'_> = if i8_layout {
            let mut mats: Vec<&WeightTensor> = table
                .tensors
                .iter()
                .filter(|t| t.dtype == WeightDtype::Int8)
                .collect();
            mats.sort_by_key(|t| {
                t.name
                    .strip_prefix("onnx::MatMul_")
                    .and_then(|n| n.strip_suffix("_int8"))
                    .and_then(|n| n.parse::<u64>().ok())
                    .unwrap_or(u64::MAX)
            });
            if mats.len() != 1 + 4 * layers + 1 {
                return Err(mismatched(format!(
                    "expected {} int8 matmuls, found {}",
                    2 + 4 * layers,
                    mats.len()
                )));
            }
            cold.extend(mats.iter().map(|t| (t.offset, t.length)));
            let embed = mats[1 + 4 * layers];
            if embed.shape.len() != 2 || embed.shape[0] != dim(hidden) {
                return Err(mismatched(format!(
                    "the tied embed is {:?}, not [{hidden}, vocab]",
                    embed.shape
                )));
            }
            let vocab = usize::try_from(embed.shape[1])
                .map_err(|_| mismatched("the embed's vocab axis is negative".to_owned()))?;
            let (w_embed, s_embed) = matrix(embed, hidden, vocab)?;
            let (w_in, s_in) = matrix(mats[0], hidden, hidden)?;
            let bias = f32s(tensor_bytes(map, table, named("model.input.bias")?)?);
            if bias.len() != hidden {
                return Err(mismatched("model.input.bias is not [hidden]".to_owned()));
            }
            let mut by_layer = Vec::with_capacity(layers);
            for block in 0..layers {
                by_layer.push((
                    mats[1 + 4 * block],
                    mats[2 + 4 * block],
                    mats[3 + 4 * block],
                    mats[4 + 4 * block],
                ));
            }
            Ok::<_, LmError>((
                w_embed,
                s_embed,
                w_in,
                s_in,
                buffer_of_f32(&device, bias),
                vocab,
                by_layer,
            ))
        } else {
            let embed = named("model.embed.weight")?;
            if embed.shape.len() != 2 || embed.shape[1] != dim(hidden) {
                return Err(mismatched(format!(
                    "model.embed.weight is {:?}, not [vocab, {hidden}]",
                    embed.shape
                )));
            }
            let vocab = usize::try_from(embed.shape[0])
                .map_err(|_| mismatched("the embed's vocab axis is negative".to_owned()))?;
            let (w_embed, _) = matrix(embed, hidden, vocab)?;
            let (w_in, _) = matrix(named("model.input.weight")?, hidden, hidden)?;
            let bias = f32s(tensor_bytes(map, table, named("model.input.bias")?)?);
            if bias.len() != hidden {
                return Err(mismatched("model.input.bias is not [hidden]".to_owned()));
            }
            let mut by_layer = Vec::with_capacity(layers);
            for block in 0..layers {
                by_layer.push((
                    named(&format!("model.blocks.{block}.qkv.weight"))?,
                    named(&format!("model.blocks.{block}.out.weight"))?,
                    named(&format!("model.blocks.{block}.ff.0.weight"))?,
                    named(&format!("model.blocks.{block}.ff.2.weight"))?,
                ));
            }
            Ok::<_, LmError>((
                w_embed,
                dummy.clone(),
                w_in,
                dummy.clone(),
                buffer_of_f32(&device, bias),
                vocab,
                by_layer,
            ))
        }?;
        // The weight stack's axes follow the source layout, not the packed
        // kind: int8 tensors are `[in, out]`, fp32 torch's `[out, in]`.
        let ff0 = by_layer[0].2;
        let feedforward = match ff0.dtype {
            WeightDtype::Int8 => usize::try_from(ff0.shape[1])
                .map_err(|_| mismatched("the ff.0 out axis is negative".to_owned()))?,
            _ => usize::try_from(ff0.shape[0])
                .map_err(|_| mismatched("the ff.0 out axis is negative".to_owned()))?,
        };
        if feedforward != FF {
            return Err(mismatched(format!(
                "the kernels are compiled for feed-forward {FF}; the manifest gives {feedforward}"
            )));
        }
        // The per-layer packs: every weight stacks its layers contiguously,
        // the kernel's `(layer * HID + i)` stride reaching into them.
        // `lm_sm` binds the fp16 [N, K] pack — built whenever the table
        // is int8 (values carried as fp16, scale folded later) or f16
        // (transposed); fp32 keeps the tiled kernel.
        let sm_pack = i8_layout || kind == WeightKind::F16;
        let mut qkv_w = Vec::new();
        let mut qkv_f16 = Vec::new();
        let mut qkv_s = Vec::new();
        let mut qkv_b = Vec::new();
        let mut out_w = Vec::new();
        let mut out_f16 = Vec::new();
        let mut out_s = Vec::new();
        let mut out_b = Vec::new();
        let mut ff0_w = Vec::new();
        let mut ff0_f16 = Vec::new();
        let mut ff0_s = Vec::new();
        let mut ff0_b = Vec::new();
        let mut ff2_w = Vec::new();
        let mut ff2_f16 = Vec::new();
        let mut ff2_s = Vec::new();
        let mut ff2_b = Vec::new();
        let mut ln = Vec::with_capacity(layers * 4 * hidden);
        // The algebraic-LN constants, one `[2N]` c1|c2 row per layer per
        // op family — collected as the repacks fold the LN weights in.
        let mut cc_qkv = Vec::new();
        let mut cc_ff0 = Vec::new();
        for (block, (qkv, out, ff0, ff2)) in by_layer.iter().enumerate() {
            // The norms come first — the repacks fold their weights into
            // the consuming op's table.
            for name in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"] {
                ln.extend_from_slice(f32s(tensor_bytes(
                    map,
                    table,
                    named(&format!("model.blocks.{block}.{name}"))?,
                )?));
            }
            let lb = block * 4 * hidden;
            let (w, s) = matrix(qkv, hidden, 3 * hidden)?;
            let (w_h, c) = if sm_pack {
                nk_repacked_f16(
                    &device,
                    &w,
                    hidden,
                    3 * hidden,
                    i8_layout,
                    Some((&ln[lb..lb + hidden], &ln[lb + hidden..lb + 2 * hidden])),
                )
            } else {
                (None, None)
            };
            qkv_f16.push(w_h);
            if let Some(c) = c {
                cc_qkv.extend_from_slice(&c);
            }
            qkv_w.push(w);
            qkv_s.push(s);
            let (w, s) = matrix(out, hidden, hidden)?;
            out_f16.push(if sm_pack {
                nk_repacked_f16(&device, &w, hidden, hidden, i8_layout, None).0
            } else {
                None
            });
            out_w.push(w);
            out_s.push(s);
            let (w, s) = matrix(ff0, hidden, feedforward)?;
            let (w_h, c) = if sm_pack {
                nk_repacked_f16(
                    &device,
                    &w,
                    hidden,
                    feedforward,
                    i8_layout,
                    Some((
                        &ln[lb + 2 * hidden..lb + 3 * hidden],
                        &ln[lb + 3 * hidden..lb + 4 * hidden],
                    )),
                )
            } else {
                (None, None)
            };
            ff0_f16.push(w_h);
            if let Some(c) = c {
                cc_ff0.extend_from_slice(&c);
            }
            ff0_w.push(w);
            ff0_s.push(s);
            let (w, s) = matrix(ff2, feedforward, hidden)?;
            ff2_f16.push(if sm_pack {
                nk_repacked_f16(&device, &w, feedforward, hidden, i8_layout, None).0
            } else {
                None
            });
            ff2_w.push(w);
            ff2_s.push(s);
            for (name, dst) in [
                ("qkv.bias", &mut qkv_b),
                ("out.bias", &mut out_b),
                ("ff.0.bias", &mut ff0_b),
                ("ff.2.bias", &mut ff2_b),
            ] {
                dst.push(buffer_of_f32(
                    &device,
                    f32s(tensor_bytes(
                        map,
                        table,
                        named(&format!("model.blocks.{block}.{name}"))?,
                    )?),
                ));
            }
        }
        // Single per-family buffers: copy each layer's bytes into one slab so
        // the kernel's `layer` stride reads a contiguous [L, ..] table.
        let concat = |bufs: &[Buffer]| -> Buffer {
            let size = usize::try_from(bufs[0].length()).expect("a buffer length fits usize");
            let total = u64::try_from(size * bufs.len()).expect("a slab size fits u64");
            let dst = device.new_buffer(total, MTLResourceOptions::StorageModeShared);
            for (index, buf) in bufs.iter().enumerate() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.contents().cast::<u8>(),
                        dst.contents().cast::<u8>().add(index * size),
                        size,
                    );
                }
            }
            dst
        };
        // `None` as soon as one layer has no repack — the op then stays
        // on the tiled kernel.
        let concat_opt = |bufs: &[Option<Buffer>]| -> Option<Buffer> {
            let mut slabs = Vec::with_capacity(bufs.len());
            for buf in bufs {
                slabs.push(buf.clone()?);
            }
            Some(concat(&slabs))
        };
        // The `lm_sm` tables: fp16 `[N, K]` per family — and the embed's
        // tied table for the head.
        let embed_f16 = if sm_pack {
            nk_repacked_f16(&device, &w_embed, hidden, vocab, i8_layout, None).0
        } else {
            None
        };
        let wh_qkv = concat_opt(&qkv_f16);
        let wh_out = concat_opt(&out_f16);
        let wh_ff0 = concat_opt(&ff0_f16);
        let wh_ff2 = concat_opt(&ff2_f16);
        // The packed `[K, N]` slabs survive only when a repack failed —
        // otherwise `lm_sm` owns every matmul and a four-byte dummy stands in.
        let keep_w = !sm_pack
            || embed_f16.is_none()
            || wh_qkv.is_none()
            || wh_out.is_none()
            || wh_ff0.is_none()
            || wh_ff2.is_none();
        let slab =
            |bufs: &[Buffer]| -> Buffer { if keep_w { concat(bufs) } else { dummy.clone() } };
        let cc_slab = |flat: &[f32]| -> Buffer {
            if flat.is_empty() {
                dummy.clone()
            } else {
                buffer_of_f32(&device, flat)
            }
        };
        // The int8 tables' pages in the weights map are dead once the
        // buffers and repacks above hold their own copies.
        for &(off, len) in &cold {
            advise_cold(map, off, len);
        }
        let pos = named("model.positions.weight")?;
        if pos.shape.len() != 2 || pos.shape[1] != dim(hidden) {
            return Err(mismatched(format!(
                "model.positions.weight is {:?}, not [T, {hidden}]",
                pos.shape
            )));
        }
        let head_w = named("model.project.weight")?;
        if head_w.shape != vec![dim(hidden), dim(hidden)] {
            return Err(mismatched(format!(
                "model.project.weight is {:?}, not [{hidden}, {hidden}]",
                head_w.shape
            )));
        }
        let ln_f_w = f32s(tensor_bytes(map, table, named("model.norm.weight")?)?);
        let ln_f_b = f32s(tensor_bytes(map, table, named("model.norm.bias")?)?);
        if ln_f_w.len() != hidden || ln_f_b.len() != hidden {
            return Err(mismatched("the final norm is not [hidden]".to_owned()));
        }
        let mut ln_f = Vec::with_capacity(2 * hidden);
        ln_f.extend_from_slice(ln_f_w);
        ln_f.extend_from_slice(ln_f_b);
        let pool = PoolBuffers::new(pages, layers, hidden)?;
        // The fixed dispatch list: embed, then each layer's qkv / attn /
        // out / ffn0 / ffn2 — 64 dispatches (LN staging happens inside the
        // producers' epilogues, not as its own op).
        let mut ops = Vec::with_capacity(4 + layers * 5);
        ops.push(Op::Embed);
        for layer in 0..layers {
            let l = u32::try_from(layer).expect("a layer index fits u32");
            ops.push(Op::MmQkv(l));
            ops.push(Op::Attn(l));
            ops.push(Op::MmOut(l));
            ops.push(Op::MmFfn0(l));
            ops.push(Op::MmFfn2(l));
        }
        ops.push(Op::MmHeadProj);
        ops.push(Op::MmHeadMat);
        ops.push(Op::HeadLogp);
        let p_head_logp = pipeline("lm_head_logp")?;
        Ok(Self {
            device: device.clone(),
            queue: device.new_command_queue(),
            p_embed: pipeline(&format!("lm_embed_{suffix}"))?,
            p_mm: pipeline(&format!("lm_mm_{suffix}"))?,
            p_mm_f32: pipeline("lm_mm_f32")?,
            p_sm: {
                let mut pipes = Vec::with_capacity(4);
                for r8 in 1u32..=4 {
                    let constants = FunctionConstantValues::new();
                    constants.set_constant_value_at_index(
                        std::ptr::from_ref(&r8).cast(),
                        MTLDataType::UInt,
                        0,
                    );
                    let function = library
                        .get_function("lm_sm", Some(constants))
                        .map_err(|e| LmError::Metal(format!("lm_sm rows={}: {e}", r8 * 8)))?;
                    pipes.push(
                        device
                            .new_compute_pipeline_state_with_function(&function)
                            .map_err(LmError::Metal)?,
                    );
                }
                let [a, b, c, d] =
                    <[ComputePipelineState; 4]>::try_from(pipes).expect("four row-width pipelines");
                [a, b, c, d]
            },
            p_smv: pipeline("lm_smv")?,
            p_attn: pipeline("lm_attn")?,
            p_head_logp,
            pool_keys: pool.keys.clone(),
            pool_values: pool.values.clone(),
            skeys: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            svals: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            scratch_slots: 0,
            scratch_width: 0,
            token: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            pages: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            mask: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            srow: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            depths: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            cand: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            input_shape: (0, 0, 0),
            x: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            g_qkv: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            g_ff: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            g_y: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            logits: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            g_sc: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            next_k: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            next_v: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            logp: device.new_buffer(4, MTLResourceOptions::StorageModeShared),
            scratch_shape: (0, 0, 0),
            embed_f16,
            w_embed,
            s_embed,
            w_in,
            s_in,
            b_in,
            pos: buffer_of_f32(&device, f32s(tensor_bytes(map, table, pos)?)),
            w_qkv: slab(&qkv_w),
            wh_qkv,
            s_qkv: concat(&qkv_s),
            b_qkv: concat(&qkv_b),
            w_out: slab(&out_w),
            wh_out,
            s_out: concat(&out_s),
            b_out: concat(&out_b),
            w_ff0: slab(&ff0_w),
            wh_ff0,
            s_ff0: concat(&ff0_s),
            b_ff0: concat(&ff0_b),
            w_ff2: slab(&ff2_w),
            wh_ff2,
            s_ff2: concat(&ff2_s),
            b_ff2: concat(&ff2_b),
            ln_wb: buffer_of_f32(&device, &ln),
            ln_f: buffer_of_f32(&device, &ln_f),
            w_head: buffer_of_f32(
                &device,
                &transposed_f32(f32s(tensor_bytes(map, table, head_w)?), hidden, hidden),
            ),
            b_head: buffer_of_f32(
                &device,
                f32s(tensor_bytes(map, table, named("model.project.bias")?)?),
            ),
            w_elem: match kind {
                WeightKind::Int8 => 1,
                WeightKind::F16 => 2,
                WeightKind::Float32 => 4,
            },
            dummy: dummy.clone(),
            ops,
            cc_qkv: cc_slab(&cc_qkv),
            cc_ff0: cc_slab(&cc_ff0),
            step_limit: None,
            x16: device.new_buffer(
                32 * hidden as u64 * 2,
                MTLResourceOptions::StorageModeShared,
            ),
            a16: device.new_buffer(
                32 * hidden as u64 * 2,
                MTLResourceOptions::StorageModeShared,
            ),
            u16: device.new_buffer(
                32 * feedforward as u64 * 2,
                MTLResourceOptions::StorageModeShared,
            ),
            st: device.new_buffer(
                32 * STATS2 as u64 * 4,
                MTLResourceOptions::StorageModeShared,
            ),
            profiling: false,
            last_profile: Vec::new(),
            last_timing: StepTiming::default(),
            layers,
            hidden,
            vocab,
            feedforward,
        })
    }
}

impl MetalStep {
    /// The `PoolBuffers` handle the `PagePool` carries — clones of the step's
    /// own pool buffers, sharing the one allocation.
    pub fn pool_buffers(&self) -> PoolBuffers {
        PoolBuffers {
            keys: self.pool_keys.clone(),
            values: self.pool_values.clone(),
        }
    }

    /// The pending scratch's `[slots, L, H, T, D]` pair as flat f32 slices —
    /// the buffers `materialise_pending` writes before `step` runs.
    ///
    /// # Safety
    ///
    /// The caller materialises, then encodes: the GPU only reads the slice
    /// inside the committed command buffer of `step`, so no alias overlaps.
    pub fn scratch_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        // Safety: shared-mode buffers; the only readers are the kernels of the
        // next `step` call, which the encode order separates from this write.
        unsafe {
            (
                std::slice::from_raw_parts_mut(
                    self.skeys.contents().cast::<f32>(),
                    usize::try_from(self.skeys.length()).expect("a buffer length fits usize") / 4,
                ),
                std::slice::from_raw_parts_mut(
                    self.svals.contents().cast::<f32>(),
                    usize::try_from(self.svals.length()).expect("a buffer length fits usize") / 4,
                ),
            )
        }
    }

    /// The gathered log-probs `[rows, count]` after `step` completes.
    pub fn logp(&self, rows: usize, count: usize) -> &[f32] {
        // Safety: `step`'s command buffer has completed; the slice covers the
        // rows*count floats the kernel wrote.
        unsafe { std::slice::from_raw_parts(self.logp.contents().cast::<f32>(), rows * count) }
    }

    /// Every buffer of the last `step` — inputs and intermediates alike —
    /// as raw bytes, for the debug dump that bisects a wrong call against a
    /// reference. The caller decodes each by its element type.
    pub fn dump(&self) -> Vec<(String, Vec<u8>)> {
        let words = |buffer: &Buffer| -> Vec<u8> {
            // Safety: shared-mode buffers, and every caller runs `dump`
            // after `step`'s buffer completed.
            unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().cast::<u8>(),
                    usize::try_from(buffer.length()).expect("a buffer length fits usize"),
                )
            }
            .to_vec()
        };
        let mut out: Vec<(String, Vec<u8>)> = [
            ("x", &self.x),
            ("x16", &self.x16),
            ("a16", &self.a16),
            ("u16", &self.u16),
            ("st", &self.st),
            ("g_y", &self.g_y),
            ("logits", &self.logits),
            ("next_k", &self.next_k),
            ("next_v", &self.next_v),
            ("logp", &self.logp),
            ("token", &self.token),
            ("pages", &self.pages),
            ("srow", &self.srow),
            ("depths", &self.depths),
            ("cand", &self.cand),
            ("mask", &self.mask),
            ("g_qkv", &self.g_qkv),
            ("g_sc", &self.g_sc),
            ("skeys", &self.skeys),
        ]
        .into_iter()
        .map(|(name, buffer)| (name.to_owned(), words(buffer)))
        .collect();
        // The pool prefix up to the highest page the call's `pages` named —
        // the resident K/V a reference needs, without the untouched tail.
        let page_bytes = self.layers * self.hidden * 4;
        let hi = words(&self.pages)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .max()
            .map_or(0, |m| (m as usize + 1) * page_bytes);
        for (name, buffer) in [
            ("pool_keys", &self.pool_keys),
            ("pool_values", &self.pool_values),
        ] {
            let mut bytes = words(buffer);
            bytes.truncate(hi);
            out.push((name.to_owned(), bytes));
        }
        out
    }

    /// The fresh `next_state_*` slices `[rows, L, H*D]` after `step`.
    pub fn next_k(&self, rows: usize) -> &[f32] {
        // Safety: as `logp` — `[rows, L, H*D]` flat.
        unsafe {
            std::slice::from_raw_parts(
                self.next_k.contents().cast::<f32>(),
                rows * self.layers * self.hidden,
            )
        }
    }

    /// As `next_k`.
    pub fn next_v(&self, rows: usize) -> &[f32] {
        // Safety: as `next_k`.
        unsafe {
            std::slice::from_raw_parts(
                self.next_v.contents().cast::<f32>(),
                rows * self.layers * self.hidden,
            )
        }
    }

    /// One step: *token*/*pages*/*mask*/*srow*/*depths* index the rows, *cand*
    /// the `[rows, count]` candidate ids, *width* the history-axis length and
    /// *slots* the scratch's first axis. Fills the input buffers, encodes the
    /// sixteen dispatches on one command buffer, and waits for it.
    /// One `advance` on the GPU: upload the inputs, encode the op list, wait.
    #[expect(
        clippy::too_many_arguments,
        reason = "the step marshals the host's six input arrays"
    )]
    pub fn step(
        &mut self,
        token: &[u32],
        pages: &[u32],
        mask: &[u8],
        srow: &[i32],
        depths: &[u32],
        cand: &[u32],
        width: usize,
        count: usize,
        slots: usize,
    ) {
        self.step_inner(token, pages, mask, srow, depths, cand, width, count, slots);
    }

    #[allow(clippy::too_many_arguments)]
    fn step_inner(
        &mut self,
        token: &[u32],
        pages: &[u32],
        mask: &[u8],
        srow: &[i32],
        depths: &[u32],
        cand: &[u32],
        count: usize,
        width: usize,
        slots: usize,
    ) {
        let rows = token.len();
        self.ensure(rows, width, count, slots);
        self.upload_inputs(token, pages, mask, srow, depths, cand);
        let ctx = StepCtx { rows, count, width };
        let ops = match self.step_limit {
            Some(n) => &self.ops[..self.ops.len().min(n)],
            None => &self.ops[..],
        };
        if self.profiling {
            self.run_profiled(&ctx);
            return;
        }
        let cpu_start = Instant::now();
        let buffer = self.queue.new_command_buffer();
        // The serial encoder — the M1 measured no gain from the concurrent
        // one, and the in-order dispatch chain needs no barriers.
        let enc = buffer.new_compute_command_encoder();
        for op in ops {
            self.encode_op(enc, *op, &ctx);
        }
        enc.end_encoding();
        let encode_ns = cpu_start.elapsed().as_nanos();
        buffer.commit();
        let submit_start = Instant::now();
        buffer.wait_until_scheduled();
        let submit_ns = submit_start.elapsed().as_nanos();
        buffer.wait_until_completed();
        self.last_timing = StepTiming {
            encode_ns: u64::try_from(encode_ns).unwrap_or(u64::MAX),
            submit_ns: u64::try_from(submit_ns).unwrap_or(u64::MAX),
            gpu_ns: u64::try_from(submit_start.elapsed().as_nanos()).unwrap_or(u64::MAX),
        };
    }

    /// The per-call input buffers: contiguous host arrays, one shared-mode
    /// copy each.
    fn upload_inputs(
        &mut self,
        token: &[u32],
        pages: &[u32],
        mask: &[u8],
        srow: &[i32],
        depths: &[u32],
        cand: &[u32],
    ) {
        let fill = |buffer: &Buffer, bytes: &[u8]| unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().cast::<u8>(),
                bytes.len(),
            );
        };
        fill(&self.token, unsafe {
            std::slice::from_raw_parts(token.as_ptr().cast::<u8>(), token.len() * 4)
        });
        fill(&self.pages, unsafe {
            std::slice::from_raw_parts(pages.as_ptr().cast::<u8>(), pages.len() * 4)
        });
        fill(&self.mask, mask);
        fill(&self.srow, unsafe {
            std::slice::from_raw_parts(srow.as_ptr().cast::<u8>(), srow.len() * 4)
        });
        fill(&self.depths, unsafe {
            std::slice::from_raw_parts(depths.as_ptr().cast::<u8>(), depths.len() * 4)
        });
        fill(&self.cand, unsafe {
            std::slice::from_raw_parts(cand.as_ptr().cast::<u8>(), cand.len() * 4)
        });
    }

    /// The `MmSpec` for an op — every field the `lm_mm` ABI takes.
    #[expect(
        clippy::too_many_lines,
        reason = "nine op arms each spell out nine buffer-offset bindings — splitting them hides the layout"
    )]
    fn mm_spec(&self, op: Op) -> MmSpec<'_> {
        let hidden = u64::try_from(self.hidden).expect("the hidden size fits u64");
        let hidden32 = u32::try_from(self.hidden).expect("the hidden size fits u32");
        let ff = u64::try_from(self.feedforward).expect("the ff width fits u64");
        let ff32 = u32::try_from(self.feedforward).expect("the ff width fits u32");
        let esz = self.w_elem;
        let sc = if self.w_elem == 1 { MM_SCALE } else { 0 };
        let layer = match op {
            Op::MmQkv(l) | Op::MmOut(l) | Op::MmFfn0(l) | Op::MmFfn2(l) => u64::from(l),
            _ => 0,
        };
        match op {
            Op::MmQkv(l) => MmSpec {
                pipe: &self.p_mm,
                wh: self.wh_qkv.as_ref(),
                a16: &self.x16,
                cc: &self.cc_qkv,
                cc_off: layer * 2 * 3 * hidden * 4,
                out_h: &self.dummy,
                a: &self.x,
                ln: &self.ln_wb,
                ln_off: layer * 4 * hidden * 4,
                w: &self.w_qkv,
                w_off: layer * hidden * 3 * hidden * esz,
                s: &self.s_qkv,
                s_off: layer * 3 * hidden * 4,
                bias: &self.b_qkv,
                bias_off: layer * 3 * hidden * 4,
                res: &self.dummy,
                out: &self.g_qkv,
                out2: &self.next_k,
                out3: &self.next_v,
                flags: MM_LN | MM_BIAS | MM_QKV | sc,
                k: hidden32,
                n: 3 * hidden32,
                lda: hidden32,
                ldn: 3 * hidden32,
                layer: l,
            },
            Op::MmOut(l) => MmSpec {
                pipe: &self.p_mm,
                wh: self.wh_out.as_ref(),
                a16: &self.a16,
                cc: &self.dummy,
                cc_off: 0,
                out_h: &self.x16,
                a: &self.g_qkv,
                ln: &self.dummy,
                ln_off: 0,
                w: &self.w_out,
                w_off: layer * hidden * hidden * esz,
                s: &self.s_out,
                s_off: layer * hidden * 4,
                bias: &self.b_out,
                bias_off: layer * hidden * 4,
                res: &self.x,
                out: &self.x,
                out2: &self.dummy,
                out3: &self.dummy,
                // H16 stages x for ffn0's read; STA drops the row sums at
                // slot LAYERS+1+l — carried in `layer`, free off QKV.
                flags: MM_BIAS | MM_RES | MM_H16 | MM_STA | sc,
                k: hidden32,
                n: hidden32,
                lda: 3 * hidden32,
                ldn: hidden32,
                layer: 12 + 1 + l,
            },
            Op::MmFfn0(l) => MmSpec {
                pipe: &self.p_mm,
                wh: self.wh_ff0.as_ref(),
                a16: &self.x16,
                cc: &self.cc_ff0,
                cc_off: layer * 2 * ff * 4,
                out_h: &self.u16,
                a: &self.x,
                ln: &self.ln_wb,
                ln_off: (layer * 4 + 2) * hidden * 4,
                w: &self.w_ff0,
                w_off: layer * hidden * ff * esz,
                s: &self.s_ff0,
                s_off: layer * ff * 4,
                bias: &self.b_ff0,
                bias_off: layer * ff * 4,
                res: &self.dummy,
                out: &self.g_ff,
                out2: &self.dummy,
                out3: &self.dummy,
                // The gelu lands fp16 in u16 for ffn2's A.
                flags: MM_LN | MM_BIAS | MM_GELU | MM_H16 | sc,
                k: hidden32,
                n: ff32,
                lda: hidden32,
                ldn: ff32,
                // Its LN reads the sums out-proj l dropped at 13+l.
                layer: 12 + 1 + l,
            },
            Op::MmFfn2(l) => MmSpec {
                pipe: &self.p_mm,
                wh: self.wh_ff2.as_ref(),
                a16: &self.u16,
                cc: &self.dummy,
                cc_off: 0,
                out_h: &self.x16,
                a: &self.g_ff,
                ln: &self.dummy,
                ln_off: 0,
                w: &self.w_ff2,
                w_off: layer * ff * hidden * esz,
                s: &self.s_ff2,
                s_off: layer * hidden * 4,
                bias: &self.b_ff2,
                bias_off: layer * hidden * 4,
                res: &self.x,
                out: &self.x,
                out2: &self.dummy,
                out3: &self.dummy,
                // x16 feeds the next layer's qkv and this row's head_proj;
                // the sums land at slot l+1 for qkv(l+1).
                flags: MM_BIAS | MM_RES | MM_H16 | MM_STA | sc,
                k: ff32,
                n: hidden32,
                lda: ff32,
                ldn: hidden32,
                layer: l + 1,
            },
            Op::MmHeadProj => MmSpec {
                pipe: &self.p_mm_f32,
                wh: None,
                a16: &self.x16,
                cc: &self.dummy,
                cc_off: 0,
                out_h: &self.x16,
                a: &self.x,
                ln: &self.ln_f,
                ln_off: 0,
                w: &self.w_head,
                w_off: 0,
                s: &self.dummy,
                s_off: 0,
                bias: &self.b_head,
                bias_off: 0,
                res: &self.dummy,
                out: &self.g_y,
                out2: &self.dummy,
                out3: &self.dummy,
                // fp16 g_y into x16 for head_mat's A — the tiled f32
                // epilogue writes it the same way.
                flags: MM_LN | MM_BIAS | MM_H16,
                k: hidden32,
                n: hidden32,
                lda: hidden32,
                ldn: hidden32,
                layer: 0,
            },
            Op::MmHeadMat => MmSpec {
                pipe: &self.p_mm,
                wh: self.embed_f16.as_ref(),
                a16: &self.x16,
                cc: &self.dummy,
                cc_off: 0,
                out_h: &self.dummy,
                a: &self.g_y,
                ln: &self.dummy,
                ln_off: 0,
                w: &self.w_embed,
                w_off: 0,
                s: &self.s_embed,
                s_off: 0,
                bias: &self.dummy,
                bias_off: 0,
                res: &self.dummy,
                out: &self.logits,
                out2: &self.dummy,
                out3: &self.dummy,
                flags: sc,
                k: hidden32,
                n: u32::try_from(self.vocab).expect("the vocab fits u32"),
                lda: hidden32,
                ldn: u32::try_from(self.vocab).expect("the vocab fits u32"),
                layer: 0,
            },
            _ => unreachable!("only the Mm ops reach mm_spec"),
        }
    }

    /// Encode the `lm_mm`/`lm_sm` dispatch *spec* describes — the buffer
    /// and scalar bindings, then the grid the picked kernel wants.
    fn encode_mm(&self, enc: &ComputeCommandEncoderRef, spec: &MmSpec<'_>, ctx: &StepCtx) {
        // `lm_sm` at every M when the op repacked its table to fp16
        // [N, K]: its A operand is the fp16 buffer the producing op
        // already wrote — no stage dispatch exists. The grid's y carries
        // the 32-row blocks. The tiled `lm_mm` covers any op without a
        // repack (the fp32 head projection) — its epilogue still writes
        // `out_h` under MM_H16, but the algebraic LN stays `lm_sm`-only.
        let sm = spec.wh.is_some();
        let m32 = u32::try_from(ctx.rows).expect("a row count fits u32");
        if sm {
            // `w_off` strides the packed table by the source elem size;
            // the fp16 repack strides at 2 bytes per element.
            enc.set_buffer(0, Some(spec.a16), 0);
            enc.set_buffer(
                1,
                Some(spec.wh.expect("sm implies the fp16 repack")),
                spec.w_off / self.w_elem * 2,
            );
            enc.set_buffer(2, Some(spec.s), spec.s_off);
            enc.set_buffer(3, Some(spec.bias), spec.bias_off);
            enc.set_buffer(4, Some(spec.res), 0);
            enc.set_buffer(5, Some(spec.out), 0);
            enc.set_buffer(6, Some(spec.out2), 0);
            enc.set_buffer(7, Some(spec.out3), 0);
            enc.set_bytes(8, 4, std::ptr::from_ref(&spec.flags).cast());
            enc.set_bytes(9, 4, std::ptr::from_ref(&spec.k).cast());
            enc.set_bytes(10, 4, std::ptr::from_ref(&spec.n).cast());
            enc.set_bytes(11, 4, std::ptr::from_ref(&m32).cast());
            enc.set_bytes(12, 4, std::ptr::from_ref(&spec.layer).cast());
            enc.set_buffer(13, Some(spec.out_h), 0);
            enc.set_buffer(14, Some(&self.st), 0);
            enc.set_buffer(15, Some(spec.cc), spec.cc_off);
            if ctx.rows <= 2 {
                // The GEMV leg: one lane per output column, the tile
                // shape would pad one row to eight.
                enc.set_compute_pipeline_state(&self.p_smv);
                enc.set_threadgroup_memory_length(0, 4 * 32 * 2 * 4);
                enc.dispatch_thread_groups(
                    MTLSize::new(u64::from(spec.n).div_ceil(32), 1, 1),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                // The row-fragment pipeline covering `rows` with the
                // least padding (r*8 rows per threadgroup).
                let r8 = u64::try_from(ctx.rows.min(32).div_ceil(8))
                    .expect("a row-fragment count fits u64");
                enc.set_compute_pipeline_state(
                    &self.p_sm[usize::try_from(r8 - 1).expect("r8 in 1..=4")],
                );
                enc.set_threadgroup_memory_length(0, 4 * r8 * 8 * 8 * 4 * 4);
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        u64::from(spec.n).div_ceil(32),
                        u64::from(m32).div_ceil(r8 * 8),
                        1,
                    ),
                    MTLSize::new(128, 1, 1),
                );
            }
            return;
        }
        enc.set_compute_pipeline_state(spec.pipe);
        enc.set_buffer(0, Some(spec.a), 0);
        enc.set_buffer(1, Some(spec.ln), spec.ln_off);
        enc.set_buffer(2, Some(spec.w), spec.w_off);
        enc.set_buffer(3, Some(spec.s), spec.s_off);
        enc.set_buffer(4, Some(spec.bias), spec.bias_off);
        enc.set_buffer(5, Some(spec.res), 0);
        enc.set_buffer(6, Some(spec.out), 0);
        enc.set_buffer(7, Some(spec.out2), 0);
        enc.set_buffer(8, Some(spec.out3), 0);
        enc.set_bytes(9, 4, std::ptr::from_ref(&spec.flags).cast());
        enc.set_bytes(10, 4, std::ptr::from_ref(&spec.k).cast());
        enc.set_bytes(11, 4, std::ptr::from_ref(&spec.n).cast());
        enc.set_bytes(12, 4, std::ptr::from_ref(&m32).cast());
        enc.set_bytes(13, 4, std::ptr::from_ref(&spec.lda).cast());
        enc.set_bytes(14, 4, std::ptr::from_ref(&spec.ldn).cast());
        enc.set_bytes(15, 4, std::ptr::from_ref(&spec.layer).cast());
        // `out_h` rides buffer 16 — the tiled kernel writes the fp16
        // staging under MM_H16 (only head_proj's g_y needs it there).
        enc.set_buffer(16, Some(spec.out_h), 0);
        let n64 = u64::from(spec.n);
        let m64 = u64::try_from(ctx.rows).expect("a row count fits u64");
        enc.dispatch_thread_groups(
            MTLSize::new(n64.div_ceil(64), m64.div_ceil(32), 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Encode the op: pipeline, buffers, scalars, dispatch.
    fn encode_op(&self, enc: &ComputeCommandEncoderRef, op: Op, ctx: &StepCtx) {
        let rows64 = u64::try_from(ctx.rows).expect("a row count fits u64");
        let hidden64 = u64::try_from(self.hidden).expect("the hidden size fits u64");
        match op {
            Op::Embed => {
                enc.set_compute_pipeline_state(&self.p_embed);
                enc.set_buffer(0, Some(&self.x), 0);
                enc.set_buffer(1, Some(&self.w_embed), 0);
                enc.set_buffer(2, Some(&self.s_embed), 0);
                enc.set_buffer(3, Some(&self.pos), 0);
                enc.set_buffer(4, Some(&self.token), 0);
                enc.set_buffer(5, Some(&self.depths), 0);
                enc.set_buffer(6, Some(&self.w_in), 0);
                enc.set_buffer(7, Some(&self.s_in), 0);
                enc.set_buffer(8, Some(&self.b_in), 0);
                let vocab = u32::try_from(self.vocab).expect("the vocab fits u32");
                enc.set_bytes(9, 4, std::ptr::from_ref(&vocab).cast());
                enc.set_buffer(10, Some(&self.x16), 0);
                enc.set_buffer(11, Some(&self.st), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(rows64, 1, 1),
                    MTLSize::new(hidden64, 1, 1),
                );
            }
            Op::Attn(layer) => {
                enc.set_compute_pipeline_state(&self.p_attn);
                enc.set_buffer(0, Some(&self.g_qkv), 0);
                enc.set_buffer(1, Some(&self.pool_keys), 0);
                enc.set_buffer(2, Some(&self.pool_values), 0);
                enc.set_buffer(3, Some(&self.skeys), 0);
                enc.set_buffer(4, Some(&self.svals), 0);
                enc.set_buffer(5, Some(&self.pages), 0);
                enc.set_buffer(6, Some(&self.mask), 0);
                enc.set_buffer(7, Some(&self.srow), 0);
                enc.set_buffer(8, Some(&self.depths), 0);
                enc.set_buffer(9, Some(&self.next_k), 0);
                enc.set_buffer(10, Some(&self.next_v), 0);
                enc.set_buffer(11, Some(&self.g_sc), 0);
                let width32 = u32::try_from(ctx.width).expect("a history width fits u32");
                enc.set_bytes(12, 4, std::ptr::from_ref(&width32).cast());
                enc.set_bytes(13, 4, std::ptr::from_ref(&layer).cast());
                enc.set_buffer(14, Some(&self.a16), 0);
                enc.dispatch_thread_groups(MTLSize::new(rows64, 1, 1), MTLSize::new(256, 1, 1));
            }
            Op::HeadLogp => {
                enc.set_compute_pipeline_state(&self.p_head_logp);
                enc.set_buffer(0, Some(&self.logits), 0);
                enc.set_buffer(1, Some(&self.cand), 0);
                enc.set_buffer(2, Some(&self.logp), 0);
                let vocab = u32::try_from(self.vocab).expect("the vocab fits u32");
                let count32 = u32::try_from(ctx.count).expect("a candidate count fits u32");
                enc.set_bytes(3, 4, std::ptr::from_ref(&vocab).cast());
                enc.set_bytes(4, 4, std::ptr::from_ref(&count32).cast());
                enc.dispatch_thread_groups(
                    MTLSize::new(rows64, 1, 1),
                    MTLSize::new(hidden64, 1, 1),
                );
            }
            op @ (Op::MmQkv(_)
            | Op::MmOut(_)
            | Op::MmFfn0(_)
            | Op::MmFfn2(_)
            | Op::MmHeadProj
            | Op::MmHeadMat) => {
                let spec = self.mm_spec(op);
                self.encode_mm(enc, &spec, ctx);
            }
        }
    }

    /// The profiled `step`: the same ops, each timed on the GPU. With a
    /// device that samples counters at dispatch boundaries (the real Macs)
    /// all dispatches stay in one encoder inside a counter-instrumented
    /// pass; the paravirtual fallback runs one command buffer per dispatch
    /// and reports commit-to-completion wall time (≈0.16 ms of it is the
    /// dispatch floor).
    #[expect(
        clippy::cast_precision_loss,
        reason = "the counters are ns deltas — f64 ms keeps three decimals"
    )]
    fn run_profiled(&mut self, ctx: &StepCtx) {
        let limit = self
            .step_limit
            .unwrap_or(self.ops.len())
            .min(self.ops.len());
        if self
            .device
            .supports_counter_sampling(MTLCounterSamplingPoint::AtDispatchBoundary)
            && let Some(set) = self
                .device
                .counter_sets()
                .iter()
                .find(|s| s.name() == "timestamp")
        {
            let desc = CounterSampleBufferDescriptor::new();
            desc.set_counter_set(set);
            let count = u64::try_from(limit + 1).expect("a dispatch count fits u64");
            desc.set_sample_count(count);
            desc.set_storage_mode(MTLStorageMode::Shared);
            if let Ok(csb) = self.device.new_counter_sample_buffer_with_descriptor(&desc) {
                let pass = ComputePassDescriptor::new();
                if let Some(att) = pass.sample_buffer_attachments().object_at(0) {
                    att.set_sample_buffer(&csb);
                    att.set_start_of_encoder_sample_index(0);
                    att.set_end_of_encoder_sample_index(count);
                    let buffer = self.queue.new_command_buffer();
                    let enc = buffer.compute_command_encoder_with_descriptor(pass);
                    enc.sample_counters_in_buffer(&csb, 0, true);
                    for (i, op) in self.ops.iter().take(limit).enumerate() {
                        self.encode_op(enc, *op, ctx);
                        enc.sample_counters_in_buffer(
                            &csb,
                            u64::try_from(i + 1).expect("a sample index fits u64"),
                            true,
                        );
                    }
                    enc.end_encoding();
                    buffer.commit();
                    buffer.wait_until_completed();
                    let ns = csb.resolve_counter_range(NSRange {
                        location: 0,
                        length: count,
                    });
                    self.last_profile = self
                        .ops
                        .iter()
                        .take(limit)
                        .enumerate()
                        .map(|(i, op)| {
                            let hi = ns.get(i + 1).copied().unwrap_or(0);
                            let lo = ns.get(i).copied().unwrap_or(0);
                            (
                                op.name(),
                                hi.saturating_sub(lo) as f64 / 1e6,
                                self.op_weight_bytes(*op, ctx),
                            )
                        })
                        .collect();
                    return;
                }
            }
        }
        // Wall-clock fallback — one command buffer per dispatch.
        let mut rows_out = Vec::with_capacity(limit);
        for op in self.ops.iter().take(limit) {
            let buffer = self.queue.new_command_buffer();
            let enc = buffer.new_compute_command_encoder();
            self.encode_op(enc, *op, ctx);
            enc.end_encoding();
            let tick = Instant::now();
            buffer.commit();
            buffer.wait_until_completed();
            rows_out.push((
                op.name(),
                tick.elapsed().as_secs_f64() * 1e3,
                self.op_weight_bytes(*op, ctx),
            ));
        }
        self.last_profile = rows_out;
    }

    /// The weight bytes one dispatch of *op* streams at *ctx* — the
    /// denominator of an effective-GB/s read; attention and the gathers
    /// move activations, not weights, and report 0.
    fn op_weight_bytes(&self, op: Op, _ctx: &StepCtx) -> u64 {
        let h = u64::try_from(self.hidden).expect("the hidden size fits u64");
        let f = u64::try_from(self.feedforward).expect("the ff width fits u64");
        let v = u64::try_from(self.vocab).expect("the vocab fits u64");
        // The fp16 [N, K] repack is the weight table whenever it exists.
        let repack =
            |f16_tab: &Option<Buffer>| -> u64 { f16_tab.as_ref().map_or(self.w_elem, |_| 2) };
        let e = match op {
            Op::MmQkv(_) => repack(&self.wh_qkv),
            Op::MmOut(_) => repack(&self.wh_out),
            Op::MmFfn0(_) => repack(&self.wh_ff0),
            Op::MmFfn2(_) => repack(&self.wh_ff2),
            Op::MmHeadMat => repack(&self.embed_f16),
            _ => self.w_elem,
        };
        match op {
            Op::Embed | Op::MmOut(_) => h * h * e,
            Op::MmQkv(_) => h * 3 * h * e,
            Op::MmFfn0(_) | Op::MmFfn2(_) => h * f * e,
            Op::MmHeadProj => h * h * 4,
            Op::MmHeadMat => h * v * e,
            Op::Attn(_) | Op::HeadLogp => 0,
        }
    }

    /// Turn per-dispatch GPU timing on or off — the rows land in
    /// [`Self::take_profile`].
    pub fn set_profiling(&mut self, on: bool) {
        self.profiling = on;
    }

    /// Truncate the step's dispatch list at *limit* ops — the measurement
    /// bisection `metal-bench`'s `--only-ops` drives. `None` runs the
    /// whole step.
    pub fn set_step_limit(&mut self, limit: Option<usize>) {
        self.step_limit = limit;
    }

    /// The `(op, ms, weight-bytes)` rows the last profiled `step` collected
    /// — empty before the first one.
    pub fn take_profile(&mut self) -> Vec<(String, f64, u64)> {
        std::mem::take(&mut self.last_profile)
    }

    /// The last `step`'s host/GPU wall split — zeros before the first one.
    pub fn timing(&self) -> StepTiming {
        self.last_timing
    }

    /// Grow the per-call buffers when *rows*/*width*/*count*/*slots* outgrow
    /// them; a steady keystroke's shapes stay constant, so growth is a
    /// one-off. Called before `scratch_mut`'s slices are handed out so they
    /// see the sized allocation.
    pub fn ensure(&mut self, rows: usize, width: usize, count: usize, slots: usize) {
        let zero = |device: &Device, bytes: usize| -> Buffer {
            let buffer = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes);
            }
            buffer
        };
        if self.input_shape.0 < rows || self.input_shape.1 < width || self.input_shape.2 < count {
            let r = rows.max(self.input_shape.0.max(64));
            let w = width.max(self.input_shape.1.max(64));
            let c = count.max(self.input_shape.2.max(64));
            self.token = zero(&self.device, r * 4);
            self.depths = zero(&self.device, r * 4);
            self.srow = zero(&self.device, r * 4);
            self.pages = zero(&self.device, r * w * 4);
            self.mask = zero(&self.device, r * w);
            self.cand = zero(&self.device, r * c * 4);
            self.input_shape = (r, w, c);
        }
        let need_scratch = self.scratch_shape.0 < rows
            || self.scratch_shape.1 < width
            || self.scratch_shape.2 < count
            || self.scratch_slots < slots
            || self.scratch_width < width;
        if !need_scratch {
            return;
        }
        let r = rows.max(self.scratch_shape.0.max(64));
        let w = width.max(self.scratch_shape.1.max(64));
        let c = count.max(self.scratch_shape.2.max(64));
        let s = slots.max(self.scratch_slots.max(4));
        self.x = zero(&self.device, r * self.hidden * 4);
        self.g_qkv = zero(&self.device, r * 3 * self.hidden * 4);
        self.g_ff = zero(&self.device, r * self.feedforward * 4);
        self.g_y = zero(&self.device, r * self.hidden * 4);
        self.logits = zero(&self.device, r * self.vocab * 4);
        self.g_sc = zero(&self.device, r * HEADS * (w + 1) * 4);
        self.next_k = zero(&self.device, r * self.layers * self.hidden * 4);
        self.next_v = zero(&self.device, r * self.layers * self.hidden * 4);
        self.logp = zero(&self.device, r * c * 4);
        // The fp16 staging pads to `lm_sm`'s 32-row block; `st`'s slots
        // index `m < M` so the row count needs no padding.
        let r32 = r.div_ceil(32) * 32;
        self.x16 = zero(&self.device, r32 * self.hidden * 2);
        self.a16 = zero(&self.device, r32 * self.hidden * 2);
        self.u16 = zero(&self.device, r32 * self.feedforward * 2);
        self.st = zero(&self.device, r * STATS2 * 4);
        let slot_bytes = s * self.layers * HEADS * w * HEAD_DIM * 4;
        self.skeys = zero(&self.device, slot_bytes);
        self.svals = zero(&self.device, slot_bytes);
        self.scratch_slots = s;
        self.scratch_width = w;
        self.scratch_shape = (r, w, c);
    }
}
