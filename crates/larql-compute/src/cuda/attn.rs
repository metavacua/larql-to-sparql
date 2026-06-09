//! CUDA fused decode-time attention.
//!
//! Phase `cuda-fused-attention`: correctness-first fused helpers for
//! the attention path. The legacy `decode_attention` helper is kept as
//! a cuBLAS + softmax reference path; the fused helpers below cover
//! RMSNorm+QKV projection and decode-time QK norm + RoPE + KV append +
//! softmax + V aggregation in a single custom launch.

use std::sync::OnceLock;

use cudarc::cublas::{
    sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
    Gemm, GemmConfig,
};
use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda-oxide")]
use cudarc::nvrtc::Ptx;
use cudarc::nvrtc::{compile_ptx, compile_ptx_with_opts, CompileOptions};

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

/// CUDA C source for a row-per-block scaled-softmax with optional
/// causal mask + softcap. Written to be deterministic, easy to read,
/// and obviously-correct rather than tuned. seq_len up to ~4096
/// supported via the strided loop.
const SOFTMAX_SRC: &str = r#"
// NVRTC compiles without the standard headers, so we provide
// IEEE-754 inf/-inf bit patterns directly.
#define POS_INF (__int_as_float(0x7f800000))
#define NEG_INF (__int_as_float(0xff800000))

extern "C" __global__ void scaled_softmax(
    float *x,
    int n_rows,
    int n_cols,
    float scale,
    float softcap,    // 0 -> no softcap
    int causal        // nonzero -> apply causal mask
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    float *r = x + (size_t)row * n_cols;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float smem[];

    // ── Pass 1: pre-process + max ─────────────────────────────────
    float my_max = NEG_INF;
    for (int j = tid; j < n_cols; j += bdim) {
        float v = r[j] * scale;
        if (softcap > 0.f) {
            v = softcap * tanhf(v / softcap);
        }
        if (causal && j > row) {
            v = NEG_INF;
        }
        r[j] = v;
        if (v > my_max) my_max = v;
    }
    smem[tid] = my_max;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = smem[tid], b = smem[tid + s];
            smem[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    float row_max = smem[0];

    // ── Pass 2: exp + sum ─────────────────────────────────────────
    float my_sum = 0.f;
    for (int j = tid; j < n_cols; j += bdim) {
        float e = __expf(r[j] - row_max);
        r[j] = e;
        my_sum += e;
    }
    smem[tid] = my_sum;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];

    // ── Pass 3: normalise ─────────────────────────────────────────
    float inv = 1.f / row_sum;
    for (int j = tid; j < n_cols; j += bdim) {
        r[j] *= inv;
    }
}
"#;

const QKV_RMS_PROJ_SRC: &str = r#"
extern "C" __global__ void qkv_rms_proj_f32(
    const float* x,
    const float* norm_w,
    const float* wq,
    const float* wk,
    const float* wv,
    float* q,
    float* k,
    float* v,
    int hidden,
    int q_dim,
    int kv_dim,
    float eps,
    float norm_offset
) {
    int row = blockIdx.x;
    int total_rows = q_dim + kv_dim + kv_dim;
    if (row >= total_rows) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];

    float ss = 0.f;
    for (int d = tid; d < hidden; d += bdim) {
        float xv = x[d];
        ss += xv * xv;
    }
    smem[tid] = ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(smem[0] / (float)hidden + eps);

    const float* w;
    float* out;
    int local_row;
    if (row < q_dim) {
        w = wq;
        out = q;
        local_row = row;
    } else if (row < q_dim + kv_dim) {
        w = wk;
        out = k;
        local_row = row - q_dim;
    } else {
        w = wv;
        out = v;
        local_row = row - q_dim - kv_dim;
    }

    float acc = 0.f;
    const float* w_row = w + (size_t)local_row * hidden;
    for (int d = tid; d < hidden; d += bdim) {
        float xn = x[d] * inv_rms * (norm_w[d] + norm_offset);
        acc += xn * w_row[d];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[local_row] = smem[0];
}
"#;

const FUSED_DECODE_ATTN_SRC: &str = r#"
#define NEG_INF (__int_as_float(0xff800000))

// `cuda-attn-wmma-f16kv` Phase 1: K/V cache is now f16. Reads convert
// to f32 via `cvt.f32.f16`; writes convert from f32 via
// `cvt.rn.f16.f32`. Q/K_new/V_new/out stay in f32 — the kernel's
// internal compute is still f32, only the K/V slab's storage and
// HBM bandwidth are halved.
__device__ float ld_kvcache(const unsigned short* p) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[0]));
    return f;
}
__device__ void st_kvcache(unsigned short* p, float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    p[0] = h;
}

extern "C" __global__ void fused_decode_attention_f32(
    const float* q,
    const float* k_new,
    const float* v_new,
    unsigned short* k_cache,
    unsigned short* v_cache,
    const float* q_norm,
    const float* k_norm,
    float* out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* pos_dev,
    int max_seq,
    int rotary_dim,
    float rope_base,
    float eps,
    float qk_norm_offset,
    float attn_scale,
    float softcap,
    int use_qk_norm,
    int d_split
) {
    // cuda-decode-cuda-graph: `pos` is read from device memory so the
    // captured graph can be replayed after writing a new value into
    // `*pos_dev` instead of re-launching with a different immediate
    // kernel arg (graphs bake in immediate args at capture time).
    int pos = *pos_dev;
    int qh = blockIdx.x;
    // cuda-attn-grid-split: each q_head's output is split across
    // `d_split` blocks (blockIdx.y). Each block computes a slice
    // `[d_start, d_end)` of `out[qh, :]`. The Q/K reductions and the
    // softmax-of-scores are recomputed redundantly in each chunk
    // (the per-block work doesn't depend on `d`), but the
    // additional grid parallelism gets us from 8 → 8*d_split blocks
    // — closer to the RTX 4090's 128 SMs. K/V cache writes are
    // gated to `dchunk == 0` to avoid duplicate writes.
    int dchunk = blockIdx.y;
    if (qh >= num_q_heads || dchunk >= d_split || pos >= max_seq) return;
    int d_per_chunk = head_dim / d_split;
    int d_start = dchunk * d_per_chunk;
    int d_end   = d_start + d_per_chunk;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];
    float* scores = smem;
    float* scratch = smem + max_seq;
    // Pre-rotated Q vector. cuda-attn-rope-hoist Phase 1: computed
    // once per attn call (depends only on `pos`, not `j`), then read
    // by every iteration of the score loop. Saves ~n_ctx redundant
    // (cosf, sinf, powf) triples per (head, d).
    float* q_rot = smem + max_seq + bdim;

    int group = max(1, num_q_heads / max(1, num_kv_heads));
    int kvh = min(num_kv_heads - 1, qh / group);
    const float* q_head = q + (size_t)qh * head_dim;
    const float* k_head = k_new + (size_t)kvh * head_dim;
    const float* v_head = v_new + (size_t)kvh * head_dim;

    float q_ss = 0.f;
    float k_ss = 0.f;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        float kv = k_head[d];
        q_ss += qv * qv;
        k_ss += kv * kv;
    }
    scratch[tid] = q_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float q_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    scratch[tid] = k_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float k_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    // ── Pre-rotate Q once per attention call ────────────────────────
    // Q's RoPE rotation depends only on `pos`, not on `j`. The score
    // loop below previously recomputed it per `(j, d)` — with
    // n_ctx ≈ 25 active threads and head_dim = 256 that's ~6 400
    // redundant trig triples per Q head per layer call. Hoist:
    //
    // `rotary_dim == 0` now means **skip RoPE entirely** (no rotation,
    // pass Q/K through unchanged). Previously upgraded to head_dim
    // (which silently rotated EVERY dim — wrong for callers that
    // already RoPE'd on host, e.g. Qwen 3.6's qwen35_gqa_decode_step).
    int rdim_pre = min(rotary_dim, head_dim);
    int hdim_pre = (rdim_pre > 0) ? rdim_pre / 2 : 1;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        if (use_qk_norm) qv *= q_inv * (q_norm[d] + qk_norm_offset);
        if (d < rdim_pre) {
            int pair = d % hdim_pre;
            bool imag = d >= hdim_pre;
            float re = q_head[pair];
            float im = q_head[pair + hdim_pre];
            if (use_qk_norm) {
                re *= q_inv * (q_norm[pair]            + qk_norm_offset);
                im *= q_inv * (q_norm[pair + hdim_pre] + qk_norm_offset);
            }
            float freq  = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim_pre);
            float angle = (float)pos * freq;
            float c = __cosf(angle);
            float s = __sinf(angle);
            qv = imag ? (re * s + im * c) : (re * c - im * s);
        }
        q_rot[d] = qv;
    }
    __syncthreads();

    // Append the current K/V once per KV head. Other Q heads sharing
    // the same KV head compute against k_new/v_new directly for pos.
    // cuda-attn-grid-split: gate K/V cache writes to dchunk == 0 so
    // multiple chunks for the same q_head don't double-write.
    if ((qh % group) == 0 && dchunk == 0) {
        for (int d = tid; d < head_dim; d += bdim) {
            float kv = k_head[d];
            if (use_qk_norm) kv *= k_inv * (k_norm[d] + qk_norm_offset);
            // Compute the rotated current key directly so the append
            // path does not need an extra per-block scratch vector.
            float k_rot;
            // See rdim_pre above — `rotary_dim == 0` skips RoPE.
            int rdim = min(rotary_dim, head_dim);
            int hdim = (rdim > 0) ? rdim / 2 : 1;
            if (d < rdim) {
                int pair = d % hdim;
                bool imag = d >= hdim;
                float re = k_head[pair];
                float im = k_head[pair + hdim];
                if (use_qk_norm) {
                    re *= k_inv * (k_norm[pair] + qk_norm_offset);
                    im *= k_inv * (k_norm[pair + hdim] + qk_norm_offset);
                }
                float freq = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim);
                float angle = (float)pos * freq;
                float c = __cosf(angle);
                float s = __sinf(angle);
                k_rot = imag ? (re * s + im * c) : (re * c - im * s);
            } else {
                k_rot = kv;
            }
            size_t idx = ((size_t)pos * num_kv_heads + kvh) * head_dim + d;
            st_kvcache(k_cache + idx, k_rot);
            st_kvcache(v_cache + idx, v_head[d]);
        }
    }
    __syncthreads();

    int n_ctx = pos + 1;
    for (int j = tid; j < n_ctx; j += bdim) {
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++) {
            // Q is pre-rotated above (cuda-attn-rope-hoist).
            float qv = q_rot[d];

            float kv;
            if (j == pos) {
                // See rdim_pre above — `rotary_dim == 0` skips RoPE.
                int rdim = min(rotary_dim, head_dim);
                if (d < rdim) {
                    int hdim = rdim / 2;
                    int pair = d % hdim;
                    bool imag = d >= hdim;
                    float re = k_head[pair];
                    float im = k_head[pair + hdim];
                    if (use_qk_norm) {
                        re *= k_inv * (k_norm[pair] + qk_norm_offset);
                        im *= k_inv * (k_norm[pair + hdim] + qk_norm_offset);
                    }
                    float freq = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim);
                    float angle = (float)pos * freq;
                    float c = __cosf(angle);
                    float s = __sinf(angle);
                    kv = imag ? (re * s + im * c) : (re * c - im * s);
                } else {
                    kv = k_head[d];
                    if (use_qk_norm) kv *= k_inv * (k_norm[d] + qk_norm_offset);
                }
            } else {
                kv = ld_kvcache(k_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + d);
            }
            dot += qv * kv;
        }
        float logit = dot * attn_scale;
        if (softcap > 0.f) logit = softcap * tanhf(logit / softcap);
        scores[j] = logit;
    }
    for (int j = tid + n_ctx; j < max_seq; j += bdim) {
        scores[j] = NEG_INF;
    }
    __syncthreads();

    float my_max = NEG_INF;
    for (int j = tid; j < n_ctx; j += bdim) {
        float s = scores[j];
        if (s > my_max) my_max = s;
    }
    scratch[tid] = my_max;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    float row_max = scratch[0];

    float my_sum = 0.f;
    for (int j = tid; j < n_ctx; j += bdim) {
        float e = __expf(scores[j] - row_max);
        scores[j] = e;
        my_sum += e;
    }
    scratch[tid] = my_sum;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.f / scratch[0];

    // cuda-attn-grid-split: each block writes only its `[d_start, d_end)`
    // slice of `out[qh, :]`. With d_split == 1 this collapses to the
    // legacy full-head_dim loop.
    for (int d = tid + d_start; d < d_end; d += bdim) {
        float acc = 0.f;
        for (int j = 0; j < n_ctx; j++) {
            float prob = scores[j] * inv_sum;
            float vv = (j == pos)
                ? v_head[d]
                : ld_kvcache(v_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + d);
            acc += prob * vv;
        }
        out[(size_t)qh * head_dim + d] = acc;
    }
}
"#;

/// FlashAttention v1-style tiled decode-attention kernel.
///
/// Same semantics as `fused_decode_attention_f32` above, but the
/// score/softmax/output loop streams the K cache in TILE_K-sized
/// tiles instead of materialising `scores[0..n_ctx]` in shmem. Per
/// thread: running `(m, l, acc_d)` state (FA v1 online softmax).
/// One thread owns one output dim across the tile loop.
///
/// Unlocks `n_ctx > 24K` per launch — the
/// `fused_decode_attention_f32` kernel's shmem grows linearly with
/// n_ctx, so 24K is the practical cap on Ada (96 KB dynamic shmem
/// budget). The tiled kernel has a fixed shmem footprint
/// independent of n_ctx: TILE_K + bdim + head_dim slots ≈ 6 KB at
/// TILE_K=1024 / bdim=256 / head_dim=128.
///
/// Tradeoff: per-tile rescale of `(l, acc_d)` adds ~2 multiplies +
/// 1 exp per tile per thread. At n_ctx ≤ 24K (32-tile sweep) that
/// overhead is negligible vs the dominant per-token Q·K compute.
/// Smaller n_ctx workloads can still use the non-tiled kernel — the
/// dispatch chooses tiled when the non-tiled shmem would exceed the
/// 96 KB opt-in budget.
const FUSED_DECODE_ATTN_TILED_SRC: &str = r#"
#define NEG_INF (__int_as_float(0xff800000))

__device__ float ld_kvcache(const unsigned short* p) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[0]));
    return f;
}
__device__ void st_kvcache(unsigned short* p, float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    p[0] = h;
}

// Tile size in K-cache rows. Must satisfy:
//   - TILE_K >= blockDim.x (so each thread covers ≥ 1 score per tile)
//   - shmem(TILE_K + bdim + head_dim) * 4 ≤ 96 KB opt-in
// At bdim=256 / head_dim=128: max TILE_K ≈ 24K, but we pick 1024 to
// keep the inner score loop short and the rescale frequent enough
// that no single tile dominates wall time.
#define TILE_K 1024

extern "C" __global__ void fused_decode_attention_tiled_f32(
    const float* q,
    const float* k_new,
    const float* v_new,
    unsigned short* k_cache,
    unsigned short* v_cache,
    const float* q_norm,
    const float* k_norm,
    float* out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* pos_dev,
    int max_seq,
    int rotary_dim,
    float rope_base,
    float eps,
    float qk_norm_offset,
    float attn_scale,
    float softcap,
    int use_qk_norm,
    int d_split
) {
    int pos = *pos_dev;
    int qh = blockIdx.x;
    int dchunk = blockIdx.y;
    if (qh >= num_q_heads || dchunk >= d_split || pos >= max_seq) return;
    int d_per_chunk = head_dim / d_split;
    int d_start = dchunk * d_per_chunk;
    int d_end   = d_start + d_per_chunk;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];
    float* tile_scores = smem;                    // [TILE_K]
    float* scratch     = smem + TILE_K;           // [bdim]
    float* q_rot       = smem + TILE_K + bdim;    // [head_dim]

    int group = max(1, num_q_heads / max(1, num_kv_heads));
    int kvh = min(num_kv_heads - 1, qh / group);
    const float* q_head = q + (size_t)qh * head_dim;
    const float* k_head = k_new + (size_t)kvh * head_dim;
    const float* v_head = v_new + (size_t)kvh * head_dim;

    // ── 1. QK-norm: rsqrt(mean(q^2)+eps), rsqrt(mean(k^2)+eps) ─────
    float q_ss = 0.f;
    float k_ss = 0.f;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        float kv = k_head[d];
        q_ss += qv * qv;
        k_ss += kv * kv;
    }
    scratch[tid] = q_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float q_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    scratch[tid] = k_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float k_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    // ── 2. Pre-rotate Q once ──────────────────────────────────────
    int rdim_pre = min(rotary_dim, head_dim);
    int hdim_pre = (rdim_pre > 0) ? rdim_pre / 2 : 1;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        if (use_qk_norm) qv *= q_inv * (q_norm[d] + qk_norm_offset);
        if (d < rdim_pre) {
            int pair = d % hdim_pre;
            bool imag = d >= hdim_pre;
            float re = q_head[pair];
            float im = q_head[pair + hdim_pre];
            if (use_qk_norm) {
                re *= q_inv * (q_norm[pair]            + qk_norm_offset);
                im *= q_inv * (q_norm[pair + hdim_pre] + qk_norm_offset);
            }
            float freq  = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim_pre);
            float angle = (float)pos * freq;
            float c = __cosf(angle);
            float s = __sinf(angle);
            qv = imag ? (re * s + im * c) : (re * c - im * s);
        }
        q_rot[d] = qv;
    }
    __syncthreads();

    // ── 3. Write current K/V to cache (gated to one Q head + dchunk 0)
    if ((qh % group) == 0 && dchunk == 0) {
        for (int d = tid; d < head_dim; d += bdim) {
            float kv = k_head[d];
            if (use_qk_norm) kv *= k_inv * (k_norm[d] + qk_norm_offset);
            float k_rot;
            int rdim = min(rotary_dim, head_dim);
            int hdim = (rdim > 0) ? rdim / 2 : 1;
            if (d < rdim) {
                int pair = d % hdim;
                bool imag = d >= hdim;
                float re = k_head[pair];
                float im = k_head[pair + hdim];
                if (use_qk_norm) {
                    re *= k_inv * (k_norm[pair] + qk_norm_offset);
                    im *= k_inv * (k_norm[pair + hdim] + qk_norm_offset);
                }
                float freq = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim);
                float angle = (float)pos * freq;
                float c = __cosf(angle);
                float s = __sinf(angle);
                k_rot = imag ? (re * s + im * c) : (re * c - im * s);
            } else {
                k_rot = kv;
            }
            size_t idx = ((size_t)pos * num_kv_heads + kvh) * head_dim + d;
            st_kvcache(k_cache + idx, k_rot);
            st_kvcache(v_cache + idx, v_head[d]);
        }
    }
    __syncthreads();

    // ── 4. Tiled streaming softmax + V output ─────────────────────
    //
    // CRITICAL: all threads must reach every `__syncthreads()` in
    // lockstep. The tile loop is therefore the OUTERMOST construct;
    // the d-loop (per-thread output-dim assignment) lives inside.
    // Threads beyond d_per_chunk participate in the shared score +
    // reduction phases but skip the per-d acc accumulation.
    //
    // Per-thread state (all threads):
    //   m_state — running max of (s · attn_scale) (uniform across threads
    //             via per-tile scratch broadcast)
    //   l_state — running sum of exp(s - m_state) (likewise uniform)
    //   my_acc  — per-thread output for this thread's assigned d
    //   my_d    — this thread's output dim, or -1 if none
    //
    // For each tile of K positions [t_start .. t_end):
    //   1. All threads cooperatively score the tile into tile_scores[]
    //   2. Reduce tile_max
    //   3. new_m = max(m_state, tile_max); rescale = exp(m_state - new_m)
    //   4. Overwrite tile_scores[j] := exp(tile_scores[j] - new_m)
    //   5. Reduce tile_l = sum(tile_scores) over the tile
    //   6. l_state = l_state * rescale + tile_l
    //   7. Threads with my_d ≥ 0: my_acc = my_acc * rescale +
    //      sum_j tile_scores[j] * V[j][my_d]
    //   8. m_state = new_m
    // Final: out[my_d] = my_acc / l_state (only threads with my_d ≥ 0)

    int n_ctx = pos + 1;
    int num_tiles = (n_ctx + TILE_K - 1) / TILE_K;

    int my_d_candidate = tid + d_start;
    int my_d = (my_d_candidate < d_end) ? my_d_candidate : -1;

    float m_state = NEG_INF;
    float l_state = 0.f;
    float my_acc = 0.f;

    for (int t = 0; t < num_tiles; t++) {
        int t_start = t * TILE_K;
        int t_end   = min(t_start + TILE_K, n_ctx);
        int tile_size = t_end - t_start;

        // 1. Score this tile. All threads cooperate.
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            int j = t_start + j_local;
            float dot = 0.f;
            for (int di = 0; di < head_dim; di++) {
                float qv = q_rot[di];
                float kv;
                if (j == pos) {
                    // Re-rotate K at `pos` on-the-fly to match the
                    // non-tiled kernel's `j == pos` branch.
                    int rdim = min(rotary_dim, head_dim);
                    if (di < rdim) {
                        int hdim = rdim / 2;
                        int pair = di % hdim;
                        bool imag = di >= hdim;
                        float re = k_head[pair];
                        float im = k_head[pair + hdim];
                        if (use_qk_norm) {
                            re *= k_inv * (k_norm[pair] + qk_norm_offset);
                            im *= k_inv * (k_norm[pair + hdim] + qk_norm_offset);
                        }
                        float freq = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim);
                        float angle = (float)pos * freq;
                        float c = __cosf(angle);
                        float s = __sinf(angle);
                        kv = imag ? (re * s + im * c) : (re * c - im * s);
                    } else {
                        kv = k_head[di];
                        if (use_qk_norm) kv *= k_inv * (k_norm[di] + qk_norm_offset);
                    }
                } else {
                    kv = ld_kvcache(k_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + di);
                }
                dot += qv * kv;
            }
            float logit = dot * attn_scale;
            if (softcap > 0.f) logit = softcap * tanhf(logit / softcap);
            tile_scores[j_local] = logit;
        }
        __syncthreads();

        // 2. Reduce tile_max.
        float my_tile_max = NEG_INF;
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            my_tile_max = fmaxf(my_tile_max, tile_scores[j_local]);
        }
        scratch[tid] = my_tile_max;
        __syncthreads();
        for (int stride = bdim / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float tile_max = scratch[0];

        // 3. Update running max + rescale factor.
        float new_m = fmaxf(m_state, tile_max);
        float rescale = (m_state == NEG_INF) ? 0.f : __expf(m_state - new_m);

        // 4. Convert tile_scores to exp(score - new_m). Reuses the
        //    same shmem region — saves a second pass later.
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            tile_scores[j_local] = __expf(tile_scores[j_local] - new_m);
        }
        __syncthreads();

        // 5. Reduce sum(exp(score-new_m)) for this tile.
        float my_tile_l = 0.f;
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            my_tile_l += tile_scores[j_local];
        }
        scratch[tid] = my_tile_l;
        __syncthreads();
        for (int stride = bdim / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            __syncthreads();
        }
        float tile_l_total = scratch[0];

        // 6. Update running l.
        l_state = l_state * rescale + tile_l_total;

        // 7. Update this thread's acc (only if my_d is valid).
        if (my_d >= 0) {
            my_acc *= rescale;
            for (int j_local = 0; j_local < tile_size; j_local++) {
                int j = t_start + j_local;
                float vv = (j == pos)
                    ? v_head[my_d]
                    : ld_kvcache(v_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + my_d);
                my_acc += tile_scores[j_local] * vv;
            }
        }

        // 8. Advance running max.
        m_state = new_m;

        // Sync before the next tile reuses tile_scores / scratch.
        __syncthreads();
    }

    if (my_d >= 0) {
        out[(size_t)qh * head_dim + my_d] = my_acc / l_state;
    }
}
"#;

/// Batched-prefill K/V cache writer. One CUDA block per
/// `(seq_pos, kv_head)` pair; rotates K with RoPE (and optional
/// QK-norm) and writes to `k_cache[base_pos + sp, kvh, :]`. V is a
/// raw copy. Runs as kernel 1 of the two-kernel batched prefill
/// attention dispatch (`cuda-prefill-batched-attention`).
const KV_CACHE_WRITE_SEQ_SRC: &str = r#"
// `cuda-attn-wmma-f16kv` Phase 1: K/V cache is f16. Write-side
// helper converts f32 → f16 via `cvt.rn.f16.f32`.
__device__ void st_kv_seq(unsigned short* p, float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    p[0] = h;
}

extern "C" __global__ void kv_cache_write_seq_f32(
    const float* k_seq,
    const float* v_seq,
    unsigned short* k_cache,
    unsigned short* v_cache,
    const float* k_norm,
    int num_kv_heads,
    int head_dim,
    const int* base_pos_dev,
    int seq_len,
    int max_seq,
    int rotary_dim,
    float rope_base,
    float eps,
    float qk_norm_offset,
    int use_qk_norm
) {
    int sp  = blockIdx.x;
    int kvh = blockIdx.y;
    if (sp >= seq_len || kvh >= num_kv_heads) return;
    int pos = (*base_pos_dev) + sp;
    if (pos >= max_seq) return;

    int tid  = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float scratch[];

    const float* k_head = k_seq + ((size_t)sp * num_kv_heads + kvh) * head_dim;
    const float* v_head = v_seq + ((size_t)sp * num_kv_heads + kvh) * head_dim;

    float k_ss = 0.f;
    for (int d = tid; d < head_dim; d += bdim) {
        float kv = k_head[d];
        k_ss += kv * kv;
    }
    scratch[tid] = k_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float k_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    // See FUSED_DECODE_ATTN_SRC: `rotary_dim == 0` means skip RoPE.
    int rdim = min(rotary_dim, head_dim);
    int hdim = (rdim > 0) ? rdim / 2 : 1;
    for (int d = tid; d < head_dim; d += bdim) {
        float k_rot;
        if (d < rdim) {
            int pair = d % hdim;
            bool imag = d >= hdim;
            float re = k_head[pair];
            float im = k_head[pair + hdim];
            if (use_qk_norm) {
                re *= k_inv * (k_norm[pair]        + qk_norm_offset);
                im *= k_inv * (k_norm[pair + hdim] + qk_norm_offset);
            }
            float freq  = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim);
            float angle = (float)pos * freq;
            float c = __cosf(angle);
            float s = __sinf(angle);
            k_rot = imag ? (re * s + im * c) : (re * c - im * s);
        } else {
            float kv = k_head[d];
            if (use_qk_norm) kv *= k_inv * (k_norm[d] + qk_norm_offset);
            k_rot = kv;
        }
        size_t idx = ((size_t)pos * num_kv_heads + kvh) * head_dim + d;
        st_kv_seq(k_cache + idx, k_rot);
        st_kv_seq(v_cache + idx, v_head[d]);
    }
}
"#;

/// Batched-prefill attention kernel. `grid_dim = (num_q_heads,
/// seq_len, 1)`, one block per `(qh, sp)` pair. Reads K/V from the
/// cache (written by `kv_cache_write_seq_f32`) over positions
/// `[0, base_pos + sp]` causally. Q-RoPE pre-rotation hoisted into
/// shared memory (same trick as `cuda-attn-rope-hoist`).
const FUSED_PREFILL_ATTN_SRC: &str = r#"
#define NEG_INF (__int_as_float(0xff800000))

// `cuda-attn-wmma-f16kv` Phase 1: K/V cache is f16 — read via cvt.
__device__ float ld_kvc_pf(const unsigned short* p) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[0]));
    return f;
}

extern "C" __global__ void fused_prefill_attention_f32(
    const float* q_seq,        // [seq_len, num_q_heads, head_dim]
    const unsigned short* k_cache, // [max_seq, num_kv_heads, head_dim] (f16 since cuda-attn-wmma-f16kv)
    const unsigned short* v_cache, // same shape
    const float* q_norm,
    float* out_seq,            // [seq_len, num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* base_pos_dev,
    int seq_len,
    int max_seq,
    int rotary_dim,
    float rope_base,
    float eps,
    float qk_norm_offset,
    float attn_scale,
    float softcap,
    int use_qk_norm
) {
    int qh = blockIdx.x;
    int sp = blockIdx.y;
    if (qh >= num_q_heads || sp >= seq_len) return;
    int pos = (*base_pos_dev) + sp;
    if (pos >= max_seq) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];
    float* scores  = smem;
    float* scratch = smem + max_seq;
    float* q_rot   = smem + max_seq + bdim;

    int group = max(1, num_q_heads / max(1, num_kv_heads));
    int kvh = min(num_kv_heads - 1, qh / group);
    const float* q_head = q_seq + (size_t)(sp * num_q_heads + qh) * head_dim;

    float q_ss = 0.f;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        q_ss += qv * qv;
    }
    scratch[tid] = q_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float q_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    // Pre-rotate Q once (depends only on `pos`, not on `j`).
    // See FUSED_DECODE_ATTN_SRC: `rotary_dim == 0` means skip RoPE.
    int rdim_pre = min(rotary_dim, head_dim);
    int hdim_pre = (rdim_pre > 0) ? rdim_pre / 2 : 1;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        if (use_qk_norm) qv *= q_inv * (q_norm[d] + qk_norm_offset);
        if (d < rdim_pre) {
            int pair = d % hdim_pre;
            bool imag = d >= hdim_pre;
            float re = q_head[pair];
            float im = q_head[pair + hdim_pre];
            if (use_qk_norm) {
                re *= q_inv * (q_norm[pair]            + qk_norm_offset);
                im *= q_inv * (q_norm[pair + hdim_pre] + qk_norm_offset);
            }
            float freq  = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim_pre);
            float angle = (float)pos * freq;
            float c = __cosf(angle);
            float s = __sinf(angle);
            qv = imag ? (re * s + im * c) : (re * c - im * s);
        }
        q_rot[d] = qv;
    }
    __syncthreads();

    int n_ctx = pos + 1;

    for (int j = tid; j < n_ctx; j += bdim) {
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++) {
            float qv = q_rot[d];
            float kv = ld_kvc_pf(k_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + d);
            dot += qv * kv;
        }
        float logit = dot * attn_scale;
        if (softcap > 0.f) logit = softcap * tanhf(logit / softcap);
        scores[j] = logit;
    }
    for (int j = tid + n_ctx; j < max_seq; j += bdim) {
        scores[j] = NEG_INF;
    }
    __syncthreads();

    float my_max = NEG_INF;
    for (int j = tid; j < n_ctx; j += bdim) {
        float s = scores[j];
        if (s > my_max) my_max = s;
    }
    scratch[tid] = my_max;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    float row_max = scratch[0];

    float my_sum = 0.f;
    for (int j = tid; j < n_ctx; j += bdim) {
        float e = __expf(scores[j] - row_max);
        scores[j] = e;
        my_sum += e;
    }
    scratch[tid] = my_sum;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.f / scratch[0];

    for (int d = tid; d < head_dim; d += bdim) {
        float acc = 0.f;
        for (int j = 0; j < n_ctx; j++) {
            float prob = scores[j] * inv_sum;
            float vv   = ld_kvc_pf(v_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + d);
            acc += prob * vv;
        }
        out_seq[(size_t)(sp * num_q_heads + qh) * head_dim + d] = acc;
    }
}
"#;

/// FlashAttention v1-style tiled variant of `fused_prefill_attention_f32`.
/// Same FA-v1 pattern as the decode tiled kernel (#252) but adapted
/// for the prefill grid (`grid = (num_q_heads, seq_len, 1)`, one
/// block per (qh, sp), no K-norm, no `j == pos` special case since
/// the K/V cache write happens in a separate kernel). Unlocks
/// per-block n_ctx > ~24K — i.e. batched prefill at base_pos +
/// seq_len > 24K. Fixed ~6 KB shmem regardless of n_ctx.
const FUSED_PREFILL_ATTN_TILED_SRC: &str = r#"
#define NEG_INF (__int_as_float(0xff800000))

__device__ float ld_kvc_pf(const unsigned short* p) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[0]));
    return f;
}

#define TILE_K 1024

extern "C" __global__ void fused_prefill_attention_tiled_f32(
    const float* q_seq,
    const unsigned short* k_cache,
    const unsigned short* v_cache,
    const float* q_norm,
    float* out_seq,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* base_pos_dev,
    int seq_len,
    int max_seq,
    int rotary_dim,
    float rope_base,
    float eps,
    float qk_norm_offset,
    float attn_scale,
    float softcap,
    int use_qk_norm
) {
    int qh = blockIdx.x;
    int sp = blockIdx.y;
    if (qh >= num_q_heads || sp >= seq_len) return;
    int pos = (*base_pos_dev) + sp;
    if (pos >= max_seq) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];
    float* tile_scores = smem;                    // [TILE_K]
    float* scratch     = smem + TILE_K;           // [bdim]
    float* q_rot       = smem + TILE_K + bdim;    // [head_dim]

    int group = max(1, num_q_heads / max(1, num_kv_heads));
    int kvh = min(num_kv_heads - 1, qh / group);
    const float* q_head = q_seq + (size_t)(sp * num_q_heads + qh) * head_dim;

    // ── 1. Q-norm (no K-norm — host pre-rotated K already) ────────
    float q_ss = 0.f;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        q_ss += qv * qv;
    }
    scratch[tid] = q_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float q_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    // ── 2. Pre-rotate Q once ──────────────────────────────────────
    int rdim_pre = min(rotary_dim, head_dim);
    int hdim_pre = (rdim_pre > 0) ? rdim_pre / 2 : 1;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        if (use_qk_norm) qv *= q_inv * (q_norm[d] + qk_norm_offset);
        if (d < rdim_pre) {
            int pair = d % hdim_pre;
            bool imag = d >= hdim_pre;
            float re = q_head[pair];
            float im = q_head[pair + hdim_pre];
            if (use_qk_norm) {
                re *= q_inv * (q_norm[pair]            + qk_norm_offset);
                im *= q_inv * (q_norm[pair + hdim_pre] + qk_norm_offset);
            }
            float freq  = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim_pre);
            float angle = (float)pos * freq;
            float c = __cosf(angle);
            float s = __sinf(angle);
            qv = imag ? (re * s + im * c) : (re * c - im * s);
        }
        q_rot[d] = qv;
    }
    __syncthreads();

    // ── 3. Tiled streaming softmax + V output (FA v1) ─────────────
    int n_ctx = pos + 1;
    int num_tiles = (n_ctx + TILE_K - 1) / TILE_K;

    // Each thread owns one output dim if tid < head_dim. Threads
    // beyond head_dim participate in shared score + reductions but
    // skip acc accumulation. (head_dim=128, bdim=256 — half the
    // threads are output-active.)
    int my_d = (tid < head_dim) ? tid : -1;

    float m_state = NEG_INF;
    float l_state = 0.f;
    float my_acc = 0.f;

    for (int t = 0; t < num_tiles; t++) {
        int t_start = t * TILE_K;
        int t_end   = min(t_start + TILE_K, n_ctx);
        int tile_size = t_end - t_start;

        // 1. Score this tile (all threads cooperate).
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            int j = t_start + j_local;
            float dot = 0.f;
            for (int di = 0; di < head_dim; di++) {
                float qv = q_rot[di];
                float kv = ld_kvc_pf(k_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + di);
                dot += qv * kv;
            }
            float logit = dot * attn_scale;
            if (softcap > 0.f) logit = softcap * tanhf(logit / softcap);
            tile_scores[j_local] = logit;
        }
        __syncthreads();

        // 2. Reduce tile_max.
        float my_tile_max = NEG_INF;
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            my_tile_max = fmaxf(my_tile_max, tile_scores[j_local]);
        }
        scratch[tid] = my_tile_max;
        __syncthreads();
        for (int stride = bdim / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float tile_max = scratch[0];

        // 3. Update running max + rescale factor.
        float new_m = fmaxf(m_state, tile_max);
        float rescale = (m_state == NEG_INF) ? 0.f : __expf(m_state - new_m);

        // 4. Convert tile_scores to exp(score - new_m).
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            tile_scores[j_local] = __expf(tile_scores[j_local] - new_m);
        }
        __syncthreads();

        // 5. Reduce tile_l = sum(exp(score - new_m)).
        float my_tile_l = 0.f;
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            my_tile_l += tile_scores[j_local];
        }
        scratch[tid] = my_tile_l;
        __syncthreads();
        for (int stride = bdim / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            __syncthreads();
        }
        float tile_l_total = scratch[0];

        // 6. Update running l.
        l_state = l_state * rescale + tile_l_total;

        // 7. Update this thread's acc (only if my_d is valid).
        if (my_d >= 0) {
            my_acc *= rescale;
            for (int j_local = 0; j_local < tile_size; j_local++) {
                int j = t_start + j_local;
                float vv = ld_kvc_pf(v_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + my_d);
                my_acc += tile_scores[j_local] * vv;
            }
        }

        // 8. Advance running max.
        m_state = new_m;

        __syncthreads();
    }

    if (my_d >= 0) {
        out_seq[(size_t)(sp * num_q_heads + qh) * head_dim + my_d] = my_acc / l_state;
    }
}
"#;

/// `cuda-spec-branching-tree` T2.2: tree-mask variant of the
/// batched-prefill attention kernel. Identical to
/// `fused_prefill_attention_f32` except the per-position loop filters
/// `j` against a per-tree-node ancestor bitset, so siblings in a
/// branching draft tree don't leak attention across each other.
///
/// Masking rule:
/// - `j ∈ [0, base_pos)` → prior history, fully causal (no mask, as
///   before).
/// - `j ∈ [base_pos, base_pos + sp]` → in-tree position. Allowed iff
///   bit `(j - base_pos)` of `ancestors_dev[sp]` is set.
///
/// For a linear chain, `ancestors_dev[sp] = (1 << (sp+1)) - 1`, so
/// every in-tree position up to and including `sp` is allowed → the
/// kernel reduces bit-exactly to the causal-only kernel.
const FUSED_PREFILL_ATTN_TREE_SRC: &str = r#"
#define NEG_INF (__int_as_float(0xff800000))

__device__ float ld_kvc_pft(const unsigned short* p) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[0]));
    return f;
}

extern "C" __global__ void fused_prefill_attention_tree_mask_f32(
    const float* q_seq,
    const unsigned short* k_cache,
    const unsigned short* v_cache,
    const float* q_norm,
    float* out_seq,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* base_pos_dev,
    const unsigned long long* ancestors_dev, // [seq_len] u64 bitsets
    int seq_len,
    int max_seq,
    int rotary_dim,
    float rope_base,
    float eps,
    float qk_norm_offset,
    float attn_scale,
    float softcap,
    int use_qk_norm
) {
    int qh = blockIdx.x;
    int sp = blockIdx.y;
    if (qh >= num_q_heads || sp >= seq_len) return;
    int base_pos = (*base_pos_dev);
    int pos = base_pos + sp;
    if (pos >= max_seq) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];
    float* scores  = smem;
    float* scratch = smem + max_seq;
    float* q_rot   = smem + max_seq + bdim;

    int group = max(1, num_q_heads / max(1, num_kv_heads));
    int kvh = min(num_kv_heads - 1, qh / group);
    const float* q_head = q_seq + (size_t)(sp * num_q_heads + qh) * head_dim;

    float q_ss = 0.f;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        q_ss += qv * qv;
    }
    scratch[tid] = q_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float q_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    // See FUSED_DECODE_ATTN_SRC: `rotary_dim == 0` means skip RoPE.
    int rdim_pre = min(rotary_dim, head_dim);
    int hdim_pre = (rdim_pre > 0) ? rdim_pre / 2 : 1;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        if (use_qk_norm) qv *= q_inv * (q_norm[d] + qk_norm_offset);
        if (d < rdim_pre) {
            int pair = d % hdim_pre;
            bool imag = d >= hdim_pre;
            float re = q_head[pair];
            float im = q_head[pair + hdim_pre];
            if (use_qk_norm) {
                re *= q_inv * (q_norm[pair]            + qk_norm_offset);
                im *= q_inv * (q_norm[pair + hdim_pre] + qk_norm_offset);
            }
            float freq  = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim_pre);
            float angle = (float)pos * freq;
            float c = __cosf(angle);
            float s = __sinf(angle);
            qv = imag ? (re * s + im * c) : (re * c - im * s);
        }
        q_rot[d] = qv;
    }
    __syncthreads();

    int n_ctx = pos + 1;
    unsigned long long anc = ancestors_dev[sp];

    for (int j = tid; j < n_ctx; j += bdim) {
        // Tree-mask: in-tree positions filtered by ancestor bitset.
        // History positions (j < base_pos) are always allowed.
        bool allow = true;
        if (j >= base_pos) {
            int bit = j - base_pos;
            allow = ((anc >> bit) & 1ULL) != 0ULL;
        }
        if (!allow) {
            scores[j] = NEG_INF;
            continue;
        }
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++) {
            float qv = q_rot[d];
            float kv = ld_kvc_pft(k_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + d);
            dot += qv * kv;
        }
        float logit = dot * attn_scale;
        if (softcap > 0.f) logit = softcap * tanhf(logit / softcap);
        scores[j] = logit;
    }
    for (int j = tid + n_ctx; j < max_seq; j += bdim) {
        scores[j] = NEG_INF;
    }
    __syncthreads();

    float my_max = NEG_INF;
    for (int j = tid; j < n_ctx; j += bdim) {
        float s = scores[j];
        if (s > my_max) my_max = s;
    }
    scratch[tid] = my_max;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    float row_max = scratch[0];

    float my_sum = 0.f;
    for (int j = tid; j < n_ctx; j += bdim) {
        float e = __expf(scores[j] - row_max);
        scores[j] = e;
        my_sum += e;
    }
    scratch[tid] = my_sum;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.f / scratch[0];

    for (int d = tid; d < head_dim; d += bdim) {
        float acc = 0.f;
        for (int j = 0; j < n_ctx; j++) {
            float prob = scores[j] * inv_sum;
            float vv   = ld_kvc_pft(v_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + d);
            acc += prob * vv;
        }
        out_seq[(size_t)(sp * num_q_heads + qh) * head_dim + d] = acc;
    }
}
"#;

/// FlashAttention v1-style tiled variant of
/// `fused_prefill_attention_tree_mask_f32`. Same FA-v1 pattern as
/// `FUSED_PREFILL_ATTN_TILED_SRC` (#253) with the additional
/// per-position ancestor-mask check that gates score computation
/// for in-tree positions j ∈ [base_pos, base_pos + sp].
///
/// Used by `fused_prefill_attention_tree_seq_device_into_pos_dev`
/// when per-block n_ctx > 16384 (the same threshold as the linear
/// prefill dispatch).
const FUSED_PREFILL_ATTN_TREE_TILED_SRC: &str = r#"
#define NEG_INF (__int_as_float(0xff800000))

__device__ float ld_kvc_pft(const unsigned short* p) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[0]));
    return f;
}

#define TILE_K 1024

extern "C" __global__ void fused_prefill_attention_tree_mask_tiled_f32(
    const float* q_seq,
    const unsigned short* k_cache,
    const unsigned short* v_cache,
    const float* q_norm,
    float* out_seq,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* base_pos_dev,
    const unsigned long long* ancestors_dev,
    int seq_len,
    int max_seq,
    int rotary_dim,
    float rope_base,
    float eps,
    float qk_norm_offset,
    float attn_scale,
    float softcap,
    int use_qk_norm
) {
    int qh = blockIdx.x;
    int sp = blockIdx.y;
    if (qh >= num_q_heads || sp >= seq_len) return;
    int base_pos = (*base_pos_dev);
    int pos = base_pos + sp;
    if (pos >= max_seq) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];
    float* tile_scores = smem;                    // [TILE_K]
    float* scratch     = smem + TILE_K;           // [bdim]
    float* q_rot       = smem + TILE_K + bdim;    // [head_dim]

    int group = max(1, num_q_heads / max(1, num_kv_heads));
    int kvh = min(num_kv_heads - 1, qh / group);
    const float* q_head = q_seq + (size_t)(sp * num_q_heads + qh) * head_dim;

    // ── 1. Q-norm ──────────────────────────────────────────────────
    float q_ss = 0.f;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        q_ss += qv * qv;
    }
    scratch[tid] = q_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float q_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    // ── 2. Pre-rotate Q ───────────────────────────────────────────
    int rdim_pre = min(rotary_dim, head_dim);
    int hdim_pre = (rdim_pre > 0) ? rdim_pre / 2 : 1;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        if (use_qk_norm) qv *= q_inv * (q_norm[d] + qk_norm_offset);
        if (d < rdim_pre) {
            int pair = d % hdim_pre;
            bool imag = d >= hdim_pre;
            float re = q_head[pair];
            float im = q_head[pair + hdim_pre];
            if (use_qk_norm) {
                re *= q_inv * (q_norm[pair]            + qk_norm_offset);
                im *= q_inv * (q_norm[pair + hdim_pre] + qk_norm_offset);
            }
            float freq  = 1.0f / __powf(rope_base, (float)(2 * pair) / (float)rdim_pre);
            float angle = (float)pos * freq;
            float c = __cosf(angle);
            float s = __sinf(angle);
            qv = imag ? (re * s + im * c) : (re * c - im * s);
        }
        q_rot[d] = qv;
    }
    __syncthreads();

    // ── 3. Tiled streaming softmax + V output (FA v1 + tree mask) ──
    int n_ctx = pos + 1;
    int num_tiles = (n_ctx + TILE_K - 1) / TILE_K;
    unsigned long long anc = ancestors_dev[sp];

    int my_d = (tid < head_dim) ? tid : -1;

    float m_state = NEG_INF;
    float l_state = 0.f;
    float my_acc = 0.f;

    for (int t = 0; t < num_tiles; t++) {
        int t_start = t * TILE_K;
        int t_end   = min(t_start + TILE_K, n_ctx);
        int tile_size = t_end - t_start;

        // 1. Score (with tree-mask) this tile.
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            int j = t_start + j_local;
            // History positions (j < base_pos) are always allowed.
            // In-tree positions filtered by ancestor bitset.
            bool allow = true;
            if (j >= base_pos) {
                int bit = j - base_pos;
                allow = ((anc >> bit) & 1ULL) != 0ULL;
            }
            if (!allow) {
                tile_scores[j_local] = NEG_INF;
                continue;
            }
            float dot = 0.f;
            for (int di = 0; di < head_dim; di++) {
                float qv = q_rot[di];
                float kv = ld_kvc_pft(k_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + di);
                dot += qv * kv;
            }
            float logit = dot * attn_scale;
            if (softcap > 0.f) logit = softcap * tanhf(logit / softcap);
            tile_scores[j_local] = logit;
        }
        __syncthreads();

        // 2. Reduce tile_max.
        float my_tile_max = NEG_INF;
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            my_tile_max = fmaxf(my_tile_max, tile_scores[j_local]);
        }
        scratch[tid] = my_tile_max;
        __syncthreads();
        for (int stride = bdim / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float tile_max = scratch[0];

        // 3. Update running max + rescale factor. If the whole tile
        //    was masked (tile_max == NEG_INF), skip the rescale (no
        //    new contribution) — m_state and acc stay as-is.
        if (tile_max == NEG_INF) {
            __syncthreads();
            continue;
        }
        float new_m = fmaxf(m_state, tile_max);
        float rescale = (m_state == NEG_INF) ? 0.f : __expf(m_state - new_m);

        // 4. Convert tile_scores to exp(score - new_m). NEG_INF
        //    scores produce 0 — safe to include in tile_l sum.
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            tile_scores[j_local] = __expf(tile_scores[j_local] - new_m);
        }
        __syncthreads();

        // 5. Reduce tile_l.
        float my_tile_l = 0.f;
        for (int j_local = tid; j_local < tile_size; j_local += bdim) {
            my_tile_l += tile_scores[j_local];
        }
        scratch[tid] = my_tile_l;
        __syncthreads();
        for (int stride = bdim / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            __syncthreads();
        }
        float tile_l_total = scratch[0];

        // 6. Update running l.
        l_state = l_state * rescale + tile_l_total;

        // 7. Update this thread's acc.
        if (my_d >= 0) {
            my_acc *= rescale;
            for (int j_local = 0; j_local < tile_size; j_local++) {
                int j = t_start + j_local;
                float vv = ld_kvc_pft(v_cache + ((size_t)j * num_kv_heads + kvh) * head_dim + my_d);
                my_acc += tile_scores[j_local] * vv;
            }
        }

        m_state = new_m;
        __syncthreads();
    }

    if (my_d >= 0) {
        out_seq[(size_t)(sp * num_q_heads + qh) * head_dim + my_d] = my_acc / l_state;
    }
}
"#;

/// Lazily-loaded softmax module + function. cudarc's `CudaContext` is
/// `Send + Sync`; `OnceLock` gives us thread-safe one-time init.
static SOFTMAX_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static QKV_RMS_PROJ_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static FUSED_DECODE_ATTN_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> =
    OnceLock::new();
static FUSED_DECODE_ATTN_TILED_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> =
    OnceLock::new();
static KV_CACHE_WRITE_SEQ_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> =
    OnceLock::new();
static FUSED_PREFILL_ATTN_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> =
    OnceLock::new();
static FUSED_PREFILL_ATTN_TILED_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> =
    OnceLock::new();
static FUSED_PREFILL_ATTN_TREE_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> =
    OnceLock::new();
static FUSED_PREFILL_ATTN_TREE_TILED_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> =
    OnceLock::new();

fn softmax_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = SOFTMAX_FUNC.get() {
        return Ok(f);
    }
    #[cfg(feature = "cuda-oxide")]
    let module = {
        let ptx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("larql_compute.ptx");
        drv.ctx
            .load_module(Ptx::from_file(ptx))
            .map_err(|e| CudaInitError::DriverMissing(format!("load cuda-oxide softmax: {e:?}")))?
    };
    #[cfg(feature = "cuda-oxide")]
    let func = module.load_function("scaled_softmax_oxide").map_err(|e| {
        CudaInitError::DriverMissing(format!("load cuda-oxide softmax function: {e:?}"))
    })?;
    #[cfg(not(feature = "cuda-oxide"))]
    let (module, func) = {
        // First call: compile PTX and load the function.
        let ptx = compile_ptx(SOFTMAX_SRC)
            .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile softmax: {e:?}")))?;
        let module = drv
            .ctx
            .load_module(ptx)
            .map_err(|e| CudaInitError::DriverMissing(format!("load module: {e:?}")))?;
        let func = module
            .load_function("scaled_softmax")
            .map_err(|e| CudaInitError::DriverMissing(format!("load function: {e:?}")))?;
        (module, func)
    };
    let _ = SOFTMAX_FUNC.set((module, func));
    let (_, f) = SOFTMAX_FUNC.get().unwrap();
    Ok(f)
}

fn qkv_rms_proj_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = QKV_RMS_PROJ_FUNC.get() {
        return Ok(f);
    }
    let ptx = compile_ptx(QKV_RMS_PROJ_SRC)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile qkv_rms_proj: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load qkv module: {e:?}")))?;
    let func = module
        .load_function("qkv_rms_proj_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load qkv function: {e:?}")))?;
    let _ = QKV_RMS_PROJ_FUNC.set((module, func));
    let (_, f) = QKV_RMS_PROJ_FUNC.get().unwrap();
    Ok(f)
}

/// `cuda-attn-grid-split`: choose how many `head_dim` chunks to split
/// each q_head's output across. With `d_split = 1` the kernel runs as
/// before (1 block per q_head); with `d_split > 1` the grid grows to
/// `(num_q_heads, d_split, 1)` and the per-block output loop only
/// covers `head_dim / d_split` elements.
///
/// `LARQL_CUDA_ATTN_DSPLIT=N` (1, 2, 4, 8, 16) overrides the default;
/// `=0` is treated as 1 (no split). `head_dim` must be divisible by
/// the chosen value or we fall back to 1.
pub(crate) fn choose_attn_d_split(num_q_heads: usize, head_dim: usize) -> i32 {
    let chosen: i32 = std::env::var("LARQL_CUDA_ATTN_DSPLIT")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|n| matches!(*n, 0 | 1 | 2 | 4 | 8 | 16))
        .unwrap_or_else(|| {
            // Heuristic: target ≥ 32 blocks per kernel call so we
            // get a few blocks per SM-quarter. RTX 4090 has 128 SMs;
            // 32 blocks ≈ 1 block per 4 SMs, leaving room for the
            // mmvq kernels' tail to overlap on the other SMs.
            let target_blocks: usize = 32;
            let needed = target_blocks.div_ceil(num_q_heads.max(1));
            // Snap to the largest power of 2 ≤ needed (and ≤ 16).
            let mut k = 1;
            while k * 2 <= needed.min(16) {
                k *= 2;
            }
            k as i32
        });
    let chosen = if chosen == 0 { 1 } else { chosen };
    if chosen <= 1 || head_dim % (chosen as usize) != 0 {
        1
    } else {
        chosen
    }
}

/// Opt into the architecture's larger dynamic shared-memory budget
/// for the attention kernels. Default per-block shmem cap is ~48 KB
/// across CCs, but Ampere (164 KB), Ada (100 KB), and Hopper (228 KB)
/// all support more with this opt-in. Set 96 KB — fits on every
/// SM 7.0+ architecture and lets the post-PR-#246 n_ctx-sized shmem
/// scale to ~24K-token prefills in a single launch.
///
/// 96 KB → max n_ctx ≈ (96·1024 / 4 − 256 − 128) ≈ 24,184 slots,
/// which covers the 16K/24K targets the long-context bench wants.
/// Past that the kernel returns `cudaErrorInvalidValue` on launch
/// and the caller falls back; production users running 32K+ will
/// need a tiled-scores rework.
const MAX_DYNAMIC_SHMEM_BYTES: i32 = 96 * 1024;

fn opt_in_dynamic_shmem(func: &CudaFunction, kernel: &str) -> Result<(), CudaInitError> {
    func.set_attribute(
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        MAX_DYNAMIC_SHMEM_BYTES,
    )
    .map_err(|e| {
        CudaInitError::DriverMissing(format!(
            "set MAX_DYNAMIC_SHARED_SIZE_BYTES on {kernel}: {e:?}"
        ))
    })
}

fn fused_decode_attention_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = FUSED_DECODE_ATTN_FUNC.get() {
        return Ok(f);
    }
    // --use_fast_math: swap cosf/sinf/expf/tanhf for the SFU-fast
    // __cosf/__sinf/__expf/__tanhf intrinsics. RoPE and softmax both
    // benefit; numerical drift stays inside the existing 1e-3 parity
    // bound. ~3× faster trig on the SFU pipeline.
    let opts = CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(FUSED_DECODE_ATTN_SRC, opts).map_err(|e| {
        CudaInitError::DriverMissing(format!("nvrtc compile fused_decode_attention: {e:?}"))
    })?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load fused attention module: {e:?}")))?;
    let func = module
        .load_function("fused_decode_attention_f32")
        .map_err(|e| {
            CudaInitError::DriverMissing(format!("load fused attention function: {e:?}"))
        })?;
    opt_in_dynamic_shmem(&func, "fused_decode_attention_f32")?;
    let _ = FUSED_DECODE_ATTN_FUNC.set((module, func));
    let (_, f) = FUSED_DECODE_ATTN_FUNC.get().unwrap();
    Ok(f)
}

/// FlashAttention v1-style tiled variant of
/// [`fused_decode_attention_function`]. Same kernel ABI; streams
/// the K cache in TILE_K (1024) chunks instead of materialising
/// `scores[0..n_ctx]` in shmem. Used when n_ctx > ~24K would
/// overflow the 96 KB dynamic-shmem opt-in budget of the non-tiled
/// kernel.
fn fused_decode_attention_tiled_function(
    drv: &Driver,
) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = FUSED_DECODE_ATTN_TILED_FUNC.get() {
        return Ok(f);
    }
    let opts = CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(FUSED_DECODE_ATTN_TILED_SRC, opts).map_err(|e| {
        CudaInitError::DriverMissing(format!("nvrtc compile fused_decode_attention_tiled: {e:?}"))
    })?;
    let module = drv.ctx.load_module(ptx).map_err(|e| {
        CudaInitError::DriverMissing(format!("load fused_decode_attention_tiled module: {e:?}"))
    })?;
    let func = module
        .load_function("fused_decode_attention_tiled_f32")
        .map_err(|e| {
            CudaInitError::DriverMissing(format!(
                "load fused_decode_attention_tiled function: {e:?}"
            ))
        })?;
    opt_in_dynamic_shmem(&func, "fused_decode_attention_tiled_f32")?;
    let _ = FUSED_DECODE_ATTN_TILED_FUNC.set((module, func));
    let (_, f) = FUSED_DECODE_ATTN_TILED_FUNC.get().unwrap();
    Ok(f)
}

/// Tile size used by the tiled-scores kernel. Must match the
/// `#define TILE_K` in `FUSED_DECODE_ATTN_TILED_SRC` — host uses it
/// to size the shmem allocation.
const FUSED_DECODE_ATTN_TILED_TILE_K: usize = 1024;

/// `n_ctx` (= pos+1) above which the tiled kernel becomes
/// preferable to the non-tiled kernel. The non-tiled kernel stores
/// `n_ctx` f32 scores in shmem; at `MAX_DYNAMIC_SHMEM_BYTES = 96
/// KB` with scratch + q_rot overhead, the practical ceiling is
/// ~24K. Use 16K as a conservative dispatch threshold: below it,
/// the non-tiled kernel is faster (simpler control flow, no
/// per-tile rescale), and at/above it we route to tiled.
const NON_TILED_DECODE_ATTN_N_CTX_MAX: usize = 16384;

/// Optional per-call attention knobs. The kernel folds these in.
#[derive(Clone, Copy, Debug)]
pub struct AttentionOpts {
    pub causal: bool,
    pub softcap: f32, // 0.0 → no softcap
}

#[derive(Clone, Copy, Debug)]
pub struct QkvProjDims {
    pub hidden: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
}

#[derive(Clone, Debug)]
pub struct QkvProjOutput {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct FusedDecodeAttentionOpts {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub pos: usize,
    pub max_seq: usize,
    pub rotary_dim: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub qk_norm_offset: f32,
    pub attn_scale: f32,
    pub softcap: f32,
}

#[derive(Clone, Debug)]
pub struct FusedDecodeAttentionOutput {
    pub out: Vec<f32>,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

/// Device-resident output of [`fused_decode_attention_device`].
/// `cuda-decode-device-resident` Phase 1.
///
/// `out` stays on the GPU so the next projection (wo) can run
/// without an extra round trip. `k_cache` / `v_cache` come back to
/// the host because Phase 1 still stores the KV cache as
/// `Vec<f32>`. Phase 3 swaps those for `CudaSlice<f32>` and drops
/// the dtoh.
pub struct FusedDecodeAttentionDeviceOutput {
    pub out: CudaSlice<f32>,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn qkv_rms_proj(
    backend: &CudaBackend,
    x: &[f32],
    norm_weight: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    dims: QkvProjDims,
    eps: f32,
    norm_offset: f32,
) -> Result<QkvProjOutput, CudaInitError> {
    assert_eq!(x.len(), dims.hidden);
    assert_eq!(norm_weight.len(), dims.hidden);
    assert_eq!(wq.len(), dims.q_dim * dims.hidden);
    assert_eq!(wk.len(), dims.kv_dim * dims.hidden);
    assert_eq!(wv.len(), dims.kv_dim * dims.hidden);

    let drv = backend.driver();
    let func = qkv_rms_proj_function(drv)?;
    let x_dev = drv.device_buf_from(x)?;
    let norm_dev = drv.device_buf_from(norm_weight)?;
    let wq_dev = drv.device_buf_from(wq)?;
    let wk_dev = drv.device_buf_from(wk)?;
    let wv_dev = drv.device_buf_from(wv)?;
    let mut q_dev = drv.device_alloc_uninit(dims.q_dim)?;
    let mut k_dev = drv.device_alloc_uninit(dims.kv_dim)?;
    let mut v_dev = drv.device_alloc_uninit(dims.kv_dim)?;

    let block_dim: u32 = 256;
    let total_rows = dims.q_dim + dims.kv_dim + dims.kv_dim;
    let cfg = LaunchConfig {
        grid_dim: (total_rows as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: block_dim * std::mem::size_of::<f32>() as u32,
    };
    let hidden_i = dims.hidden as i32;
    let q_dim_i = dims.q_dim as i32;
    let kv_dim_i = dims.kv_dim as i32;

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(&x_dev)
            .arg(&norm_dev)
            .arg(&wq_dev)
            .arg(&wk_dev)
            .arg(&wv_dev)
            .arg(&mut q_dev)
            .arg(&mut k_dev)
            .arg(&mut v_dev)
            .arg(&hidden_i)
            .arg(&q_dim_i)
            .arg(&kv_dim_i)
            .arg(&eps)
            .arg(&norm_offset)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch qkv_rms_proj: {e:?}")))?;
    }

    drv.sync()?;
    Ok(QkvProjOutput {
        q: drv.to_host(&q_dev)?,
        k: drv.to_host(&k_dev)?,
        v: drv.to_host(&v_dev)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn fused_decode_attention(
    backend: &CudaBackend,
    q: &[f32],
    k_new: &[f32],
    v_new: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    opts: FusedDecodeAttentionOpts,
) -> Result<FusedDecodeAttentionOutput, CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    assert_eq!(q.len(), q_dim);
    assert_eq!(k_new.len(), kv_dim);
    assert_eq!(v_new.len(), kv_dim);
    assert_eq!(k_cache.len(), cache_len);
    assert_eq!(v_cache.len(), cache_len);
    assert!(opts.pos < opts.max_seq);

    let use_qk_norm = q_norm.is_some() && k_norm.is_some();
    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };

    let drv = backend.driver();
    let func = fused_decode_attention_function(drv)?;
    let q_dev = drv.device_buf_from(q)?;
    let k_new_dev = drv.device_buf_from(k_new)?;
    let v_new_dev = drv.device_buf_from(v_new)?;
    // `cuda-attn-wmma-f16kv` Phase 1: kernel expects f16 K/V cache.
    // Convert host-side f32 → f16 once and htod the f16 buffer.
    let k_cache_h: Vec<half::f16> = k_cache.iter().map(|&v| half::f16::from_f32(v)).collect();
    let v_cache_h: Vec<half::f16> = v_cache.iter().map(|&v| half::f16::from_f32(v)).collect();
    let mut k_cache_dev = drv
        .stream
        .clone_htod(&k_cache_h)
        .map_err(|e| CudaInitError::DriverMissing(format!("htod k_cache f16: {e:?}")))?;
    let mut v_cache_dev = drv
        .stream
        .clone_htod(&v_cache_h)
        .map_err(|e| CudaInitError::DriverMissing(format!("htod v_cache f16: {e:?}")))?;
    let q_norm_dev = drv.device_buf_from(q_norm)?;
    let k_norm_dev = drv.device_buf_from(k_norm)?;
    let mut out_dev = drv.device_alloc_uninit(q_dim)?;
    // cuda-decode-cuda-graph: pos is read from device memory.
    let pos_dev = drv
        .stream
        .clone_htod(&[opts.pos as i32])
        .map_err(|e| CudaInitError::DriverMissing(format!("htod pos: {e:?}")))?;

    let block_dim: u32 = 256;
    let d_split_i = choose_attn_d_split(opts.num_q_heads, opts.head_dim);
    let cfg = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, d_split_i as u32, 1),
        block_dim: (block_dim, 1, 1),
        // cuda-decode-shmem-fix: shmem holds `n_ctx = pos+1` score
        // slots, not the slab capacity. Allocating `max_seq` slots
        // forces ~80 KB shmem at max_seq=20000 and drops RTX 4090
        // occupancy from ~3 blocks/SM to 1 block/SM — a 3-4×
        // wall-time hit that doesn't track actual cached context.
        // The kernel arg `max_seq` (set below) is updated to match
        // so the kernel's scratch / q_rot offsets and the
        // (now-redundant) NEG_INF tail-fill loop stay consistent.
        shared_mem_bytes: ((opts.pos + 1 + block_dim as usize + opts.head_dim)
            * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    // cuda-decode-shmem-fix: kernel `max_seq` arg drives the shmem
    // offset for scratch/q_rot and bounds the NEG_INF tail-fill
    // loop. Both should track the *current* n_ctx (= pos+1), not the
    // slab capacity — see the shared_mem_bytes comment above.
    let max_seq_i = (opts.pos + 1) as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(&q_dev)
            .arg(&k_new_dev)
            .arg(&v_new_dev)
            .arg(&mut k_cache_dev)
            .arg(&mut v_cache_dev)
            .arg(&q_norm_dev)
            .arg(&k_norm_dev)
            .arg(&mut out_dev)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(&pos_dev)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .arg(&d_split_i)
            .launch(cfg)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!("launch fused_decode_attention: {e:?}"))
            })?;
    }

    drv.sync()?;
    // `cuda-attn-wmma-f16kv` Phase 1: dtoh f16 → host, convert to f32
    // for the public `FusedDecodeAttentionOutput { k_cache: Vec<f32>,
    // v_cache: Vec<f32> }` contract.
    let k_cache_f16: Vec<half::f16> = drv
        .stream
        .clone_dtoh(&k_cache_dev)
        .map_err(|e| CudaInitError::DriverMissing(format!("dtoh k_cache f16: {e:?}")))?;
    let v_cache_f16: Vec<half::f16> = drv
        .stream
        .clone_dtoh(&v_cache_dev)
        .map_err(|e| CudaInitError::DriverMissing(format!("dtoh v_cache f16: {e:?}")))?;
    Ok(FusedDecodeAttentionOutput {
        out: drv.to_host(&out_dev)?,
        k_cache: k_cache_f16.iter().map(|v| v.to_f32()).collect(),
        v_cache: v_cache_f16.iter().map(|v| v.to_f32()).collect(),
    })
}

/// `cuda-decode-device-resident` Phase 3: full device-resident
/// fused decode attention. Q/K-new/V-new are `CudaSlice<f32>`
/// (Phase 1) **and** the K/V cache is now `&mut CudaSlice<f32>`
/// — the kernel reads prior tokens from it and writes the new row
/// in place at `pos`, with zero PCIe traffic. Returns just the
/// attention output as a device-resident slice.
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_attention_device_kv(
    backend: &CudaBackend,
    q_dev: &CudaSlice<f32>,
    k_new_dev: &CudaSlice<f32>,
    v_new_dev: &CudaSlice<f32>,
    k_cache_dev: &mut CudaSlice<half::f16>,
    v_cache_dev: &mut CudaSlice<half::f16>,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    opts: FusedDecodeAttentionOpts,
) -> Result<CudaSlice<f32>, CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    assert_eq!(q_dev.len(), q_dim);
    assert_eq!(k_new_dev.len(), kv_dim);
    assert_eq!(v_new_dev.len(), kv_dim);
    assert_eq!(k_cache_dev.len(), cache_len);
    assert_eq!(v_cache_dev.len(), cache_len);
    assert!(opts.pos < opts.max_seq);

    let use_qk_norm = q_norm.is_some() && k_norm.is_some();
    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };

    let drv = backend.driver();
    // cuda-decode-tiled-dispatch: route to the FA-v1 tiled kernel
    // when n_ctx > 16K. The non-tiled kernel's shmem is linear in
    // n_ctx (~24K max at 96 KB opt-in); the tiled kernel uses a
    // fixed 6 KB regardless of n_ctx. 16K threshold leaves the
    // non-tiled path on its fastest operating point (it has simpler
    // control flow than the tiled streaming pattern).
    let n_ctx = opts.pos + 1;
    let use_tiled = n_ctx > NON_TILED_DECODE_ATTN_N_CTX_MAX;
    let func = if use_tiled {
        fused_decode_attention_tiled_function(drv)?
    } else {
        fused_decode_attention_function(drv)?
    };
    let q_norm_dev = drv.device_buf_from(q_norm)?;
    let k_norm_dev = drv.device_buf_from(k_norm)?;
    let mut out_dev = drv.device_alloc_uninit(q_dim)?;
    // cuda-decode-cuda-graph: pos is read from device memory.
    let pos_dev = drv
        .stream
        .clone_htod(&[opts.pos as i32])
        .map_err(|e| CudaInitError::DriverMissing(format!("htod pos: {e:?}")))?;

    let block_dim: u32 = 256;
    let d_split_i = choose_attn_d_split(opts.num_q_heads, opts.head_dim);
    // Tiled kernel: fixed shmem footprint (TILE_K + bdim + head_dim).
    // Non-tiled kernel: shmem grows with n_ctx (n_ctx + bdim + head_dim).
    // See PR #246 for the non-tiled shmem-by-n_ctx fix.
    let shmem_slots = if use_tiled {
        FUSED_DECODE_ATTN_TILED_TILE_K + block_dim as usize + opts.head_dim
    } else {
        n_ctx + block_dim as usize + opts.head_dim
    };
    let cfg = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, d_split_i as u32, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (shmem_slots * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    // Non-tiled kernel uses max_seq for the shmem layout offset
    // (PR #246 made it track n_ctx). The tiled kernel doesn't use
    // max_seq for shmem layout (its scratch/q_rot offsets are at
    // fixed TILE_K + bdim), but the bounds check `pos >= max_seq`
    // still uses it — pass the slab capacity so the check matches
    // the cache slab's actual capacity.
    let max_seq_i = if use_tiled {
        opts.max_seq as i32
    } else {
        n_ctx as i32
    };
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(q_dev)
            .arg(k_new_dev)
            .arg(v_new_dev)
            .arg(k_cache_dev)
            .arg(v_cache_dev)
            .arg(&q_norm_dev)
            .arg(&k_norm_dev)
            .arg(&mut out_dev)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(&pos_dev)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .arg(&d_split_i)
            .launch(cfg)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!(
                    "launch fused_decode_attention_device_kv: {e:?}"
                ))
            })?;
    }

    // Phase 3: no sync, no dtoh of K/V cache slabs. The kernel
    // wrote the new row into k_cache_dev/v_cache_dev at `pos`;
    // those buffers are persistent across calls so subsequent
    // tokens read them without any PCIe traffic.
    Ok(out_dev)
}

/// `cuda-decode-cuda-graph` variant of
/// [`fused_decode_attention_device_kv`]. Differs in three ways:
///
/// * `pos_dev`, `q_norm_dev`, `k_norm_dev` come pre-allocated from
///   `DecodeScratch` — no per-call htod / alloc that would create
///   spurious memory nodes inside the captured graph.
/// * `out_dev` is supplied by the caller (`scratch.attn_out`) instead
///   of being freshly allocated, so the captured kernel uses a
///   stable destination pointer across replays.
/// * Returns `()` — the result lives in `out_dev`.
///
/// The caller MUST `htod_into_slice(&[pos_i], pos_dev, 0)` before
/// each replay and ensure `q_norm_dev` / `k_norm_dev` already hold
/// the per-layer norm weights (or zeros if `use_qk_norm == 0`).
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_attention_device_kv_into(
    backend: &CudaBackend,
    q_dev: &CudaSlice<f32>,
    k_new_dev: &CudaSlice<f32>,
    v_new_dev: &CudaSlice<f32>,
    k_cache_dev: &mut CudaSlice<half::f16>,
    v_cache_dev: &mut CudaSlice<half::f16>,
    q_norm_dev: &CudaSlice<f32>,
    k_norm_dev: &CudaSlice<f32>,
    pos_dev: &CudaSlice<i32>,
    out_dev: &mut CudaSlice<f32>,
    use_qk_norm: bool,
    opts: FusedDecodeAttentionOpts,
) -> Result<(), CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    debug_assert_eq!(q_dev.len(), q_dim);
    debug_assert_eq!(k_new_dev.len(), kv_dim);
    debug_assert_eq!(v_new_dev.len(), kv_dim);
    debug_assert_eq!(k_cache_dev.len(), cache_len);
    debug_assert_eq!(v_cache_dev.len(), cache_len);
    debug_assert_eq!(out_dev.len(), q_dim);
    debug_assert_eq!(pos_dev.len(), 1);
    debug_assert_eq!(q_norm_dev.len(), opts.head_dim);
    debug_assert_eq!(k_norm_dev.len(), opts.head_dim);

    let drv = backend.driver();
    let func = fused_decode_attention_function(drv)?;

    let block_dim: u32 = 256;
    let d_split_i = choose_attn_d_split(opts.num_q_heads, opts.head_dim);
    let cfg = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, d_split_i as u32, 1),
        block_dim: (block_dim, 1, 1),
        // cuda-decode-shmem-fix: shmem holds `n_ctx = pos+1` score
        // slots, not the slab capacity. Allocating `max_seq` slots
        // forces ~80 KB shmem at max_seq=20000 and drops RTX 4090
        // occupancy from ~3 blocks/SM to 1 block/SM — a 3-4×
        // wall-time hit that doesn't track actual cached context.
        // The kernel arg `max_seq` (set below) is updated to match
        // so the kernel's scratch / q_rot offsets and the
        // (now-redundant) NEG_INF tail-fill loop stay consistent.
        shared_mem_bytes: ((opts.pos + 1 + block_dim as usize + opts.head_dim)
            * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    // cuda-decode-shmem-fix: kernel `max_seq` arg drives the shmem
    // offset for scratch/q_rot and bounds the NEG_INF tail-fill
    // loop. Both should track the *current* n_ctx (= pos+1), not the
    // slab capacity — see the shared_mem_bytes comment above.
    let max_seq_i = (opts.pos + 1) as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(q_dev)
            .arg(k_new_dev)
            .arg(v_new_dev)
            .arg(k_cache_dev)
            .arg(v_cache_dev)
            .arg(q_norm_dev)
            .arg(k_norm_dev)
            .arg(out_dev)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(pos_dev)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .arg(&d_split_i)
            .launch(cfg)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!(
                    "launch fused_decode_attention_device_kv_into: {e:?}"
                ))
            })?;
    }
    Ok(())
}

/// Device-resident variant of [`fused_decode_attention`]. Q / K-new /
/// V-new come in as `CudaSlice<f32>` (already produced by
/// `q4k_matvec_device` etc.) and the attention output stays on the
/// GPU. Phase 1 still pulls the K/V cache back to host because
/// `CudaKvCache` is `Vec<f32>`-backed; that goes away in Phase 3.
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_attention_device(
    backend: &CudaBackend,
    q_dev: &CudaSlice<f32>,
    k_new_dev: &CudaSlice<f32>,
    v_new_dev: &CudaSlice<f32>,
    k_cache: &[f32],
    v_cache: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    opts: FusedDecodeAttentionOpts,
) -> Result<FusedDecodeAttentionDeviceOutput, CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    assert_eq!(q_dev.len(), q_dim);
    assert_eq!(k_new_dev.len(), kv_dim);
    assert_eq!(v_new_dev.len(), kv_dim);
    assert_eq!(k_cache.len(), cache_len);
    assert_eq!(v_cache.len(), cache_len);
    assert!(opts.pos < opts.max_seq);

    let use_qk_norm = q_norm.is_some() && k_norm.is_some();
    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };

    let drv = backend.driver();
    let func = fused_decode_attention_function(drv)?;
    // `cuda-attn-wmma-f16kv` Phase 1: kernel expects f16 K/V cache.
    let k_cache_h: Vec<half::f16> = k_cache.iter().map(|&v| half::f16::from_f32(v)).collect();
    let v_cache_h: Vec<half::f16> = v_cache.iter().map(|&v| half::f16::from_f32(v)).collect();
    let mut k_cache_dev = drv
        .stream
        .clone_htod(&k_cache_h)
        .map_err(|e| CudaInitError::DriverMissing(format!("htod k_cache f16: {e:?}")))?;
    let mut v_cache_dev = drv
        .stream
        .clone_htod(&v_cache_h)
        .map_err(|e| CudaInitError::DriverMissing(format!("htod v_cache f16: {e:?}")))?;
    let q_norm_dev = drv.device_buf_from(q_norm)?;
    let k_norm_dev = drv.device_buf_from(k_norm)?;
    let mut out_dev = drv.device_alloc_uninit(q_dim)?;
    // cuda-decode-cuda-graph: pos is read from device memory.
    let pos_dev = drv
        .stream
        .clone_htod(&[opts.pos as i32])
        .map_err(|e| CudaInitError::DriverMissing(format!("htod pos: {e:?}")))?;

    let block_dim: u32 = 256;
    let d_split_i = choose_attn_d_split(opts.num_q_heads, opts.head_dim);
    let cfg = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, d_split_i as u32, 1),
        block_dim: (block_dim, 1, 1),
        // cuda-decode-shmem-fix: shmem holds `n_ctx = pos+1` score
        // slots, not the slab capacity. Allocating `max_seq` slots
        // forces ~80 KB shmem at max_seq=20000 and drops RTX 4090
        // occupancy from ~3 blocks/SM to 1 block/SM — a 3-4×
        // wall-time hit that doesn't track actual cached context.
        // The kernel arg `max_seq` (set below) is updated to match
        // so the kernel's scratch / q_rot offsets and the
        // (now-redundant) NEG_INF tail-fill loop stay consistent.
        shared_mem_bytes: ((opts.pos + 1 + block_dim as usize + opts.head_dim)
            * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    // cuda-decode-shmem-fix: kernel `max_seq` arg drives the shmem
    // offset for scratch/q_rot and bounds the NEG_INF tail-fill
    // loop. Both should track the *current* n_ctx (= pos+1), not the
    // slab capacity — see the shared_mem_bytes comment above.
    let max_seq_i = (opts.pos + 1) as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(q_dev)
            .arg(k_new_dev)
            .arg(v_new_dev)
            .arg(&mut k_cache_dev)
            .arg(&mut v_cache_dev)
            .arg(&q_norm_dev)
            .arg(&k_norm_dev)
            .arg(&mut out_dev)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(&pos_dev)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .arg(&d_split_i)
            .launch(cfg)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!("launch fused_decode_attention_device: {e:?}"))
            })?;
    }

    drv.sync()?;
    // `cuda-attn-wmma-f16kv` Phase 1: cache slabs are f16; convert.
    let k_cache_f16: Vec<half::f16> = drv
        .stream
        .clone_dtoh(&k_cache_dev)
        .map_err(|e| CudaInitError::DriverMissing(format!("dtoh k_cache f16: {e:?}")))?;
    let v_cache_f16: Vec<half::f16> = drv
        .stream
        .clone_dtoh(&v_cache_dev)
        .map_err(|e| CudaInitError::DriverMissing(format!("dtoh v_cache f16: {e:?}")))?;
    Ok(FusedDecodeAttentionDeviceOutput {
        out: out_dev,
        k_cache: k_cache_f16.iter().map(|v| v.to_f32()).collect(),
        v_cache: v_cache_f16.iter().map(|v| v.to_f32()).collect(),
    })
}

impl Default for AttentionOpts {
    fn default() -> Self {
        AttentionOpts {
            causal: false,
            softcap: 0.0,
        }
    }
}

/// In-place row-wise softmax on a `[n_rows, n_cols]` row-major device
/// buffer. `scale` is applied before the row max (so `1/sqrt(d)` etc).
pub(crate) fn softmax_inplace(
    drv: &Driver,
    x_dev: &mut cudarc::driver::CudaSlice<f32>,
    n_rows: usize,
    n_cols: usize,
    scale: f32,
    opts: AttentionOpts,
) -> Result<(), CudaInitError> {
    let func = softmax_function(drv)?;
    #[cfg(not(feature = "cuda-oxide"))]
    // 1024 threads = max blockDim on every supported arch.
    let block_dim: u32 = 1024;
    #[cfg(feature = "cuda-oxide")]
    // cuda-oxide softmax is a correctness-first row-serial pilot kernel.
    let block_dim: u32 = 1;
    let grid_dim: u32 = n_rows as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        #[cfg(not(feature = "cuda-oxide"))]
        shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
        #[cfg(feature = "cuda-oxide")]
        shared_mem_bytes: 0,
    };
    let n_rows_i = n_rows as i32;
    let n_cols_i = n_cols as i32;
    let causal_i: i32 = if opts.causal { 1 } else { 0 };
    let softcap_f = opts.softcap;
    #[cfg(feature = "cuda-oxide")]
    let len = x_dev.len();
    // SAFETY: The kernel writes at most `n_rows * n_cols` f32 values
    // starting at the slice base; the buffer length matches the shape
    // (caller guarantees).
    unsafe {
        let mut builder = drv.stream.launch_builder(func);
        builder.arg(x_dev);
        #[cfg(feature = "cuda-oxide")]
        builder.arg(&len);
        builder
            .arg(&n_rows_i)
            .arg(&n_cols_i)
            .arg(&scale)
            .arg(&softcap_f)
            .arg(&causal_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch softmax: {e:?}")))?;
    }
    Ok(())
}

/// Single-head decode-time attention: `out = softmax((Q @ K^T) * scale, mask) @ V`.
///
/// Inputs are row-major contiguous slices:
///   Q: `[n_q, head_dim]`
///   K: `[n_kv, head_dim]`
///   V: `[n_kv, head_dim]`
/// Output: `[n_q, head_dim]`.
///
/// One synchronous host roundtrip total.
pub fn decode_attention(
    backend: &CudaBackend,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    opts: AttentionOpts,
) -> Result<Vec<f32>, CudaInitError> {
    let drv = backend.driver();
    debug_assert_eq!(q.len(), n_q * head_dim);
    debug_assert_eq!(k.len(), n_kv * head_dim);
    debug_assert_eq!(v.len(), n_kv * head_dim);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // ── 1. attn_logits = Q @ K^T  [n_q × n_kv] row-major ─────────────
    // Reuse the same row-major-via-column-major identity from
    // cuda::matmul: passing K as the first cuBLAS arg with op=T.
    let q_dev = drv.device_buf_from(q)?;
    let k_dev = drv.device_buf_from(k)?;
    let v_dev = drv.device_buf_from(v)?;

    let mut logits_dev = drv.device_alloc_uninit(n_q * n_kv)?;
    let cfg_qk = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n_kv as i32,
        n: n_q as i32,
        k: head_dim as i32,
        alpha: 1.0_f32,
        lda: head_dim as i32,
        ldb: head_dim as i32,
        beta: 0.0_f32,
        ldc: n_kv as i32,
    };
    // SAFETY: dimensions / leading-dims match buffer lengths.
    unsafe {
        drv.blas
            .gemm(cfg_qk, &k_dev, &q_dev, &mut logits_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("gemm QK^T: {e:?}")))?;
    }

    // ── 2. softmax(logits, scale, mask) in place ────────────────────
    softmax_inplace(drv, &mut logits_dev, n_q, n_kv, scale, opts)?;

    // ── 3. out = attn @ V  [n_q × head_dim] row-major ───────────────
    // Same row-major identity:
    //   row-major attn (n_q, n_kv)  ≡ col-major (n_kv, n_q)
    //   row-major V    (n_kv, head_dim) ≡ col-major (head_dim, n_kv)
    //   want col-major out (head_dim, n_q) = V^T_cm × attn_cm
    // cuBLAS: transa=N, transb=N, M=head_dim, N=n_q, K=n_kv,
    //         lda=head_dim, ldb=n_kv, ldc=head_dim.
    let mut out_dev = drv.device_alloc_uninit(n_q * head_dim)?;
    let cfg_av = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: head_dim as i32,
        n: n_q as i32,
        k: n_kv as i32,
        alpha: 1.0_f32,
        lda: head_dim as i32,
        ldb: n_kv as i32,
        beta: 0.0_f32,
        ldc: head_dim as i32,
    };
    unsafe {
        drv.blas
            .gemm(cfg_av, &v_dev, &logits_dev, &mut out_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("gemm attn@V: {e:?}")))?;
    }

    drv.sync()?;
    drv.to_host(&out_dev)
}

fn kv_cache_write_seq_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = KV_CACHE_WRITE_SEQ_FUNC.get() {
        return Ok(f);
    }
    let opts = CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KV_CACHE_WRITE_SEQ_SRC, opts).map_err(|e| {
        CudaInitError::DriverMissing(format!("nvrtc compile kv_cache_write_seq: {e:?}"))
    })?;
    let module = drv.ctx.load_module(ptx).map_err(|e| {
        CudaInitError::DriverMissing(format!("load kv_cache_write_seq module: {e:?}"))
    })?;
    let func = module
        .load_function("kv_cache_write_seq_f32")
        .map_err(|e| {
            CudaInitError::DriverMissing(format!("load kv_cache_write_seq function: {e:?}"))
        })?;
    let _ = KV_CACHE_WRITE_SEQ_FUNC.set((module, func));
    let (_, f) = KV_CACHE_WRITE_SEQ_FUNC.get().unwrap();
    Ok(f)
}

fn fused_prefill_attention_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = FUSED_PREFILL_ATTN_FUNC.get() {
        return Ok(f);
    }
    let opts = CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(FUSED_PREFILL_ATTN_SRC, opts).map_err(|e| {
        CudaInitError::DriverMissing(format!("nvrtc compile fused_prefill_attention: {e:?}"))
    })?;
    let module = drv.ctx.load_module(ptx).map_err(|e| {
        CudaInitError::DriverMissing(format!("load fused_prefill_attention module: {e:?}"))
    })?;
    let func = module
        .load_function("fused_prefill_attention_f32")
        .map_err(|e| {
            CudaInitError::DriverMissing(format!("load fused_prefill_attention function: {e:?}"))
        })?;
    opt_in_dynamic_shmem(&func, "fused_prefill_attention_f32")?;
    let _ = FUSED_PREFILL_ATTN_FUNC.set((module, func));
    let (_, f) = FUSED_PREFILL_ATTN_FUNC.get().unwrap();
    Ok(f)
}

/// FlashAttention v1-style tiled variant of
/// [`fused_prefill_attention_function`]. Same kernel ABI; streams K
/// cache in TILE_K (1024) chunks instead of materialising
/// `scores[0..n_ctx]` in shmem. Used when per-block n_ctx
/// (= base_pos + sp + 1, max at sp = seq_len-1) would overflow the
/// 96 KB opt-in budget of the non-tiled prefill kernel.
fn fused_prefill_attention_tiled_function(
    drv: &Driver,
) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = FUSED_PREFILL_ATTN_TILED_FUNC.get() {
        return Ok(f);
    }
    let opts = CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(FUSED_PREFILL_ATTN_TILED_SRC, opts).map_err(|e| {
        CudaInitError::DriverMissing(format!(
            "nvrtc compile fused_prefill_attention_tiled: {e:?}"
        ))
    })?;
    let module = drv.ctx.load_module(ptx).map_err(|e| {
        CudaInitError::DriverMissing(format!("load fused_prefill_attention_tiled module: {e:?}"))
    })?;
    let func = module
        .load_function("fused_prefill_attention_tiled_f32")
        .map_err(|e| {
            CudaInitError::DriverMissing(format!(
                "load fused_prefill_attention_tiled function: {e:?}"
            ))
        })?;
    opt_in_dynamic_shmem(&func, "fused_prefill_attention_tiled_f32")?;
    let _ = FUSED_PREFILL_ATTN_TILED_FUNC.set((module, func));
    let (_, f) = FUSED_PREFILL_ATTN_TILED_FUNC.get().unwrap();
    Ok(f)
}

/// TILE_K matching the prefill tiled kernel's `#define TILE_K`.
/// Host uses it for shmem allocation.
const FUSED_PREFILL_ATTN_TILED_TILE_K: usize = 1024;

/// Threshold above which `fused_prefill_attention_launch_pos_dev`
/// routes to the tiled prefill kernel. Same 16K as the decode-step
/// dispatch — leaves the non-tiled path on its fastest operating
/// point (simpler control flow, no per-tile rescale) for sub-16K
/// per-block n_ctx.
const NON_TILED_PREFILL_ATTN_N_CTX_MAX: i32 = 16384;

fn fused_prefill_attention_tree_function(
    drv: &Driver,
) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = FUSED_PREFILL_ATTN_TREE_FUNC.get() {
        return Ok(f);
    }
    let opts = CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(FUSED_PREFILL_ATTN_TREE_SRC, opts).map_err(|e| {
        CudaInitError::DriverMissing(format!("nvrtc compile fused_prefill_attn_tree: {e:?}"))
    })?;
    let module = drv.ctx.load_module(ptx).map_err(|e| {
        CudaInitError::DriverMissing(format!("load fused_prefill_attn_tree module: {e:?}"))
    })?;
    let func = module
        .load_function("fused_prefill_attention_tree_mask_f32")
        .map_err(|e| {
            CudaInitError::DriverMissing(format!("load fused_prefill_attn_tree function: {e:?}"))
        })?;
    opt_in_dynamic_shmem(&func, "fused_prefill_attention_tree_mask_f32")?;
    let _ = FUSED_PREFILL_ATTN_TREE_FUNC.set((module, func));
    let (_, f) = FUSED_PREFILL_ATTN_TREE_FUNC.get().unwrap();
    Ok(f)
}

/// FlashAttention v1-style tiled variant of
/// [`fused_prefill_attention_tree_function`]. Same ABI; streams K
/// cache in TILE_K=1024 chunks instead of materialising
/// `scores[0..n_ctx]` in shmem. Used by the tree-mask prefill
/// dispatch when per-block n_ctx > 16384.
fn fused_prefill_attention_tree_tiled_function(
    drv: &Driver,
) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = FUSED_PREFILL_ATTN_TREE_TILED_FUNC.get() {
        return Ok(f);
    }
    let opts = CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(FUSED_PREFILL_ATTN_TREE_TILED_SRC, opts).map_err(|e| {
        CudaInitError::DriverMissing(format!(
            "nvrtc compile fused_prefill_attn_tree_tiled: {e:?}"
        ))
    })?;
    let module = drv.ctx.load_module(ptx).map_err(|e| {
        CudaInitError::DriverMissing(format!("load fused_prefill_attn_tree_tiled module: {e:?}"))
    })?;
    let func = module
        .load_function("fused_prefill_attention_tree_mask_tiled_f32")
        .map_err(|e| {
            CudaInitError::DriverMissing(format!(
                "load fused_prefill_attn_tree_tiled function: {e:?}"
            ))
        })?;
    opt_in_dynamic_shmem(&func, "fused_prefill_attention_tree_mask_tiled_f32")?;
    let _ = FUSED_PREFILL_ATTN_TREE_TILED_FUNC.set((module, func));
    let (_, f) = FUSED_PREFILL_ATTN_TREE_TILED_FUNC.get().unwrap();
    Ok(f)
}

/// Pre-allocated-output variant: writes the attention output into the
/// caller-supplied `out_seq` (must have at least `seq_len * q_dim`
/// elements). Same contract / kernel launches as
/// [`fused_prefill_attention_seq_device`] otherwise.
///
/// `cuda-spec-cuda-graph` Phase A companion — lets the spec batched
/// forward keep its attention output in a scratch buffer instead of
/// alloc+memcpy-dtod per layer.
#[allow(clippy::too_many_arguments)]
pub fn fused_prefill_attention_seq_device_into(
    backend: &CudaBackend,
    q_seq: &CudaSlice<f32>,
    k_seq: &CudaSlice<f32>,
    v_seq: &CudaSlice<f32>,
    k_cache: &mut CudaSlice<half::f16>,
    v_cache: &mut CudaSlice<half::f16>,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    out_seq: &mut CudaSlice<f32>,
    base_pos: usize,
    seq_len: usize,
    opts: FusedDecodeAttentionOpts,
) -> Result<(), CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    if q_seq.len() < seq_len * q_dim {
        return Err(CudaInitError::DriverMissing(format!(
            "q_seq.len={} < seq_len*q_dim={}*{}",
            q_seq.len(),
            seq_len,
            q_dim,
        )));
    }
    if k_seq.len() < seq_len * kv_dim || v_seq.len() < seq_len * kv_dim {
        return Err(CudaInitError::DriverMissing(format!(
            "k_seq/v_seq.len mismatch for seq_len*kv_dim={}*{}",
            seq_len, kv_dim,
        )));
    }
    if out_seq.len() < seq_len * q_dim {
        return Err(CudaInitError::DriverMissing(format!(
            "out_seq.len={} < seq_len*q_dim={}*{}",
            out_seq.len(),
            seq_len,
            q_dim,
        )));
    }
    if k_cache.len() != cache_len || v_cache.len() != cache_len {
        return Err(CudaInitError::DriverMissing(format!(
            "k_cache/v_cache.len mismatch for max_seq*num_kv_heads*head_dim={}",
            cache_len,
        )));
    }
    if base_pos + seq_len > opts.max_seq {
        return Err(CudaInitError::DriverMissing(format!(
            "base_pos+seq_len={}>{}=max_seq",
            base_pos + seq_len,
            opts.max_seq,
        )));
    }

    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };
    let use_qk_norm = !q_norm.iter().all(|&v| v == 0.0) || !k_norm.iter().all(|&v| v == 0.0);

    let drv = backend.driver();
    let func_kv = kv_cache_write_seq_function(drv)?;
    let func_attn = fused_prefill_attention_function(drv)?;
    // q_norm / k_norm: cache via arc_norm_device_buf (keyed by host
    // pointer). On the spec hot path this collapses to a one-time
    // upload per (layer, q_norm or k_norm host slice).
    let q_norm_arc = backend.arc_norm_device_buf(q_norm)?;
    let k_norm_arc = backend.arc_norm_device_buf(k_norm)?;

    let block_dim_kv: u32 = 256;
    let cfg_kv = LaunchConfig {
        grid_dim: (seq_len as u32, opts.num_kv_heads as u32, 1),
        block_dim: (block_dim_kv, 1, 1),
        shared_mem_bytes: (block_dim_kv as usize * std::mem::size_of::<f32>()) as u32,
    };
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    let base_pos_i = base_pos as i32;
    let base_pos_dev = drv
        .stream
        .clone_htod(&[base_pos_i])
        .map_err(|e| CudaInitError::DriverMissing(format!("clone_htod base_pos: {e:?}")))?;
    let seq_len_i = seq_len as i32;
    // cuda-prefill-shmem-fix: kernel `max_seq` arg drives shmem
    // offset for scratch/q_rot and bounds the NEG_INF tail-fill
    // loop. Use the actual max n_ctx (= base_pos + seq_len) so the
    // launch helper's shmem allocation tracks real work, not slab
    // capacity. The kernel's `pos >= max_seq` bounds check stays
    // safe because `pos = base_pos + sp` and `sp < seq_len`.
    let max_seq_i = (base_pos + seq_len) as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    fused_prefill_attention_launch_pos_dev(
        drv,
        func_kv,
        func_attn,
        q_seq,
        k_seq,
        v_seq,
        k_cache,
        v_cache,
        &k_norm_arc,
        &q_norm_arc,
        out_seq,
        &base_pos_dev,
        seq_len,
        opts,
        num_kv_heads_i,
        head_dim_i,
        seq_len_i,
        max_seq_i,
        rotary_dim_i,
        use_qk_norm_i,
        cfg_kv,
    )?;

    Ok(())
}

/// `cuda-spec-cuda-graph` Phase B: variant of
/// [`fused_prefill_attention_seq_device_into`] that takes `base_pos`
/// as a pre-existing device-side i32 slot. Lets a captured CUDA Graph
/// be replayed at a different cache position via `memcpy_htod` into
/// the slot, without re-launching the wrapper.
///
/// Required for the spec batched forward's graph-capture path. The
/// caller (typically a `SpecDecodeScratch`) owns `base_pos_dev` and
/// writes the current position into it before each invocation.
#[allow(clippy::too_many_arguments)]
pub fn fused_prefill_attention_seq_device_into_pos_dev(
    backend: &CudaBackend,
    q_seq: &CudaSlice<f32>,
    k_seq: &CudaSlice<f32>,
    v_seq: &CudaSlice<f32>,
    k_cache: &mut CudaSlice<half::f16>,
    v_cache: &mut CudaSlice<half::f16>,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    out_seq: &mut CudaSlice<f32>,
    base_pos_dev: &CudaSlice<i32>,
    seq_len: usize,
    opts: FusedDecodeAttentionOpts,
) -> Result<(), CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    if q_seq.len() < seq_len * q_dim
        || k_seq.len() < seq_len * kv_dim
        || v_seq.len() < seq_len * kv_dim
        || out_seq.len() < seq_len * q_dim
        || k_cache.len() != cache_len
        || v_cache.len() != cache_len
    {
        return Err(CudaInitError::DriverMissing(
            "fused_prefill_attention_seq_device_into_pos_dev: shape mismatch".into(),
        ));
    }

    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };
    let use_qk_norm = !q_norm.iter().all(|&v| v == 0.0) || !k_norm.iter().all(|&v| v == 0.0);

    let drv = backend.driver();
    let func_kv = kv_cache_write_seq_function(drv)?;
    let func_attn = fused_prefill_attention_function(drv)?;
    let q_norm_arc = backend.arc_norm_device_buf(q_norm)?;
    let k_norm_arc = backend.arc_norm_device_buf(k_norm)?;

    let block_dim_kv: u32 = 256;
    let cfg_kv = LaunchConfig {
        grid_dim: (seq_len as u32, opts.num_kv_heads as u32, 1),
        block_dim: (block_dim_kv, 1, 1),
        shared_mem_bytes: (block_dim_kv as usize * std::mem::size_of::<f32>()) as u32,
    };
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    let seq_len_i = seq_len as i32;
    // cuda-prefill-shmem-fix: same as the non-pos-dev variant — the
    // launch helper sizes shmem from `max_seq_i`. Use the actual max
    // n_ctx (= opts.pos + seq_len, where opts.pos is the host-side
    // base_pos hint set by the caller in decode.rs) so shmem tracks
    // real work, not the slab capacity. PR #246 fixed this for the
    // non-spec-decode path; this is the spec-decode follow-up.
    let max_seq_i = (opts.pos + seq_len) as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    fused_prefill_attention_launch_pos_dev(
        drv,
        func_kv,
        func_attn,
        q_seq,
        k_seq,
        v_seq,
        k_cache,
        v_cache,
        &k_norm_arc,
        &q_norm_arc,
        out_seq,
        base_pos_dev,
        seq_len,
        opts,
        num_kv_heads_i,
        head_dim_i,
        seq_len_i,
        max_seq_i,
        rotary_dim_i,
        use_qk_norm_i,
        cfg_kv,
    )?;

    Ok(())
}

/// `cuda-spec-branching-tree` T2.3: tree-mask variant of
/// [`fused_prefill_attention_seq_device_into_pos_dev`]. Adds an
/// `ancestors_dev` device-resident u64 array — one bitset per tree
/// node — that masks attention to ancestor-only positions within the
/// tree slice of the cache.
///
/// Contract:
/// - `ancestors_dev.len() >= seq_len`. Bit `k` of `ancestors_dev[sp]`
///   is set iff position `k` (in tree-local coords) is an ancestor of
///   tree node `sp` (inclusive of self). Construct via
///   `DraftTree::ancestor_bitsets()`.
/// - K/V cache writes happen at `base_pos + tree_index` linearly — the
///   existing `kv_cache_write_seq_f32` kernel is reused unchanged
///   (Strategy A from the design doc).
/// - Linear-chain trees (bitset `(1 << (k+1)) - 1`) SHALL produce
///   bit-identical output to
///   [`fused_prefill_attention_seq_device_into_pos_dev`].
#[allow(clippy::too_many_arguments)]
pub fn fused_prefill_attention_tree_seq_device_into_pos_dev(
    backend: &CudaBackend,
    q_seq: &CudaSlice<f32>,
    k_seq: &CudaSlice<f32>,
    v_seq: &CudaSlice<f32>,
    k_cache: &mut CudaSlice<half::f16>,
    v_cache: &mut CudaSlice<half::f16>,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    out_seq: &mut CudaSlice<f32>,
    base_pos_dev: &CudaSlice<i32>,
    ancestors_dev: &CudaSlice<u64>,
    seq_len: usize,
    opts: FusedDecodeAttentionOpts,
) -> Result<(), CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    if q_seq.len() < seq_len * q_dim
        || k_seq.len() < seq_len * kv_dim
        || v_seq.len() < seq_len * kv_dim
        || out_seq.len() < seq_len * q_dim
        || k_cache.len() != cache_len
        || v_cache.len() != cache_len
        || ancestors_dev.len() < seq_len
    {
        return Err(CudaInitError::DriverMissing(
            "fused_prefill_attention_tree_seq_device_into_pos_dev: shape mismatch".into(),
        ));
    }
    if seq_len > 64 {
        // Ancestor bitsets are u64 — one bit per tree position. A
        // tree larger than 64 nodes cannot be represented; caller
        // SHOULD have capped tree size via SpecConfig::tree_nodes().
        return Err(CudaInitError::DriverMissing(
            "fused_prefill_attention_tree_seq_device_into_pos_dev: seq_len > 64".into(),
        ));
    }

    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };
    let use_qk_norm = !q_norm.iter().all(|&v| v == 0.0) || !k_norm.iter().all(|&v| v == 0.0);

    let drv = backend.driver();
    let func_kv = kv_cache_write_seq_function(drv)?;
    // cuda-prefill-tree-tiled-dispatch: route to the FA-v1 tree-mask
    // tiled kernel when per-block n_ctx > 16384. Same threshold as
    // the linear prefill dispatch (see fused_prefill_attention_launch_pos_dev).
    let max_seq_i = (opts.pos + seq_len) as i32;
    let use_tiled = max_seq_i > NON_TILED_PREFILL_ATTN_N_CTX_MAX;
    let func_attn = if use_tiled {
        fused_prefill_attention_tree_tiled_function(drv)?
    } else {
        fused_prefill_attention_tree_function(drv)?
    };
    let q_norm_arc = backend.arc_norm_device_buf(q_norm)?;
    let k_norm_arc = backend.arc_norm_device_buf(k_norm)?;

    let block_dim_kv: u32 = 256;
    let cfg_kv = LaunchConfig {
        grid_dim: (seq_len as u32, opts.num_kv_heads as u32, 1),
        block_dim: (block_dim_kv, 1, 1),
        shared_mem_bytes: (block_dim_kv as usize * std::mem::size_of::<f32>()) as u32,
    };
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    let seq_len_i = seq_len as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    // Reuse the existing K/V write kernel — Strategy A from the
    // design doc: K/V at base_pos + tree_index linearly, ancestor
    // mask filters reads.
    unsafe {
        drv.stream
            .launch_builder(func_kv)
            .arg(k_seq)
            .arg(v_seq)
            .arg(&mut *k_cache)
            .arg(&mut *v_cache)
            .arg(&*k_norm_arc)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(base_pos_dev)
            .arg(&seq_len_i)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&use_qk_norm_i)
            .launch(cfg_kv)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!(
                    "launch kv_cache_write_seq_pos_dev (tree): {e:?}"
                ))
            })?;
    }

    let block_dim_attn: u32 = 256;
    // Tiled kernel: fixed (TILE_K + bdim + head_dim) shmem.
    // Non-tiled kernel: (n_ctx + bdim + head_dim) per PR #246's
    // shmem-by-n_ctx fix.
    let shmem_slots = if use_tiled {
        FUSED_PREFILL_ATTN_TILED_TILE_K + block_dim_attn as usize + opts.head_dim
    } else {
        max_seq_i as usize + block_dim_attn as usize + opts.head_dim
    };
    let cfg_attn = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, seq_len as u32, 1),
        block_dim: (block_dim_attn, 1, 1),
        shared_mem_bytes: (shmem_slots * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;

    unsafe {
        drv.stream
            .launch_builder(func_attn)
            .arg(q_seq)
            .arg(&*k_cache)
            .arg(&*v_cache)
            .arg(&*q_norm_arc)
            .arg(out_seq)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(base_pos_dev)
            .arg(ancestors_dev)
            .arg(&seq_len_i)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .launch(cfg_attn)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!(
                    "launch fused_prefill_attention_tree_mask_pos_dev: {e:?}"
                ))
            })?;
    }

    Ok(())
}

/// Shared kernel-launch helper for the `_pos_dev` family. Avoids
/// duplicating the launch builder code between the alloc-base_pos
/// wrapper and the device-pointer wrapper.
#[allow(clippy::too_many_arguments)]
fn fused_prefill_attention_launch_pos_dev(
    drv: &Driver,
    func_kv: &CudaFunction,
    func_attn: &CudaFunction,
    q_seq: &CudaSlice<f32>,
    k_seq: &CudaSlice<f32>,
    v_seq: &CudaSlice<f32>,
    k_cache: &mut CudaSlice<half::f16>,
    v_cache: &mut CudaSlice<half::f16>,
    k_norm_dev: &CudaSlice<f32>,
    q_norm_dev: &CudaSlice<f32>,
    out_seq: &mut CudaSlice<f32>,
    base_pos_dev: &CudaSlice<i32>,
    seq_len: usize,
    opts: FusedDecodeAttentionOpts,
    num_kv_heads_i: i32,
    head_dim_i: i32,
    seq_len_i: i32,
    max_seq_i: i32,
    rotary_dim_i: i32,
    use_qk_norm_i: i32,
    cfg_kv: LaunchConfig,
) -> Result<(), CudaInitError> {
    unsafe {
        drv.stream
            .launch_builder(func_kv)
            .arg(k_seq)
            .arg(v_seq)
            .arg(&mut *k_cache)
            .arg(&mut *v_cache)
            .arg(k_norm_dev)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(base_pos_dev)
            .arg(&seq_len_i)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&use_qk_norm_i)
            .launch(cfg_kv)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!("launch kv_cache_write_seq_pos_dev: {e:?}"))
            })?;
    }

    // cuda-prefill-tiled-dispatch: route to the FA-v1 tiled kernel
    // when per-block n_ctx (= base_pos + sp + 1, max at sp = seq_len-1)
    // exceeds the non-tiled kernel's shmem budget. The non-tiled
    // path uses n_ctx-sized shmem (~24K max at 96 KB opt-in); the
    // tiled path has a fixed ~6 KB footprint. Same 16K threshold as
    // the decode-attn dispatch.
    let use_tiled = max_seq_i > NON_TILED_PREFILL_ATTN_N_CTX_MAX;
    let func_attn_dispatched = if use_tiled {
        fused_prefill_attention_tiled_function(drv)?
    } else {
        func_attn
    };

    let block_dim_attn: u32 = 256;
    let shmem_slots = if use_tiled {
        FUSED_PREFILL_ATTN_TILED_TILE_K + block_dim_attn as usize + opts.head_dim
    } else {
        max_seq_i as usize + block_dim_attn as usize + opts.head_dim
    };
    let cfg_attn = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, seq_len as u32, 1),
        block_dim: (block_dim_attn, 1, 1),
        // Tiled: fixed (TILE_K + bdim + head_dim) regardless of n_ctx.
        // Non-tiled: (n_ctx + bdim + head_dim) — PR #246 shmem-by-n_ctx fix.
        shared_mem_bytes: (shmem_slots * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;

    unsafe {
        drv.stream
            .launch_builder(func_attn_dispatched)
            .arg(q_seq)
            .arg(&*k_cache)
            .arg(&*v_cache)
            .arg(q_norm_dev)
            .arg(out_seq)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(base_pos_dev)
            .arg(&seq_len_i)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .launch(cfg_attn)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!(
                    "launch fused_prefill_attention_pos_dev: {e:?}"
                ))
            })?;
    }

    Ok(())
}

pub fn fused_prefill_attention_seq_device(
    backend: &CudaBackend,
    q_seq: &CudaSlice<f32>,
    k_seq: &CudaSlice<f32>,
    v_seq: &CudaSlice<f32>,
    k_cache: &mut CudaSlice<half::f16>,
    v_cache: &mut CudaSlice<half::f16>,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    base_pos: usize,
    seq_len: usize,
    opts: FusedDecodeAttentionOpts,
) -> Result<CudaSlice<f32>, CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    if q_seq.len() != seq_len * q_dim {
        return Err(CudaInitError::DriverMissing(format!(
            "q_seq.len={} != seq_len*q_dim={}*{}",
            q_seq.len(),
            seq_len,
            q_dim,
        )));
    }
    if k_seq.len() != seq_len * kv_dim || v_seq.len() != seq_len * kv_dim {
        return Err(CudaInitError::DriverMissing(format!(
            "k_seq/v_seq.len mismatch for seq_len*kv_dim={}*{}",
            seq_len, kv_dim,
        )));
    }
    if k_cache.len() != cache_len || v_cache.len() != cache_len {
        return Err(CudaInitError::DriverMissing(format!(
            "k_cache/v_cache.len mismatch for max_seq*num_kv_heads*head_dim={}",
            cache_len,
        )));
    }
    if base_pos + seq_len > opts.max_seq {
        return Err(CudaInitError::DriverMissing(format!(
            "base_pos+seq_len={}>{}=max_seq",
            base_pos + seq_len,
            opts.max_seq,
        )));
    }

    let use_qk_norm = q_norm.is_some() && k_norm.is_some();
    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };

    let drv = backend.driver();
    let func_kv = kv_cache_write_seq_function(drv)?;
    // cuda-prefill-tiled-dispatch (legacy): same threshold + tiled
    // routing as the non-pos_dev variant — see #253. Caller has
    // base_pos host-side, so max_seq_i = base_pos + seq_len is
    // computed below before this dispatch needs it... actually
    // it's computed at line ~3392. Re-compute here for the dispatch.
    let max_seq_i_for_dispatch = (base_pos + seq_len) as i32;
    let use_tiled = max_seq_i_for_dispatch > NON_TILED_PREFILL_ATTN_N_CTX_MAX;
    let func_attn = if use_tiled {
        fused_prefill_attention_tiled_function(drv)?
    } else {
        fused_prefill_attention_function(drv)?
    };
    let q_norm_dev = drv.device_buf_from(q_norm)?;
    let k_norm_dev = drv.device_buf_from(k_norm)?;
    let mut out_seq = drv.device_alloc_uninit(seq_len * q_dim)?;

    let block_dim_kv: u32 = 256;
    let cfg_kv = LaunchConfig {
        grid_dim: (seq_len as u32, opts.num_kv_heads as u32, 1),
        block_dim: (block_dim_kv, 1, 1),
        shared_mem_bytes: (block_dim_kv as usize * std::mem::size_of::<f32>()) as u32,
    };
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    let base_pos_i = base_pos as i32;
    // Allocate a small device int slot for base_pos. The kernel now
    // reads pos from `const int*` so a captured CUDA Graph can replay
    // at a different position via `memcpy_htod`. For non-captured
    // calls this is a one-time htod of 4 bytes per launch.
    let base_pos_dev = drv
        .stream
        .clone_htod(&[base_pos_i])
        .map_err(|e| CudaInitError::DriverMissing(format!("clone_htod base_pos: {e:?}")))?;
    let seq_len_i = seq_len as i32;
    // cuda-prefill-shmem-fix (legacy variant): drive shmem and the
    // kernel `max_seq` arg from the actual n_ctx (base_pos + seq_len)
    // rather than slab capacity. Same fix as PR #246's non-pos-dev
    // variant — legacy variant takes base_pos as a direct usize arg.
    let max_seq_i = (base_pos + seq_len) as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    unsafe {
        drv.stream
            .launch_builder(func_kv)
            .arg(k_seq)
            .arg(v_seq)
            .arg(&mut *k_cache)
            .arg(&mut *v_cache)
            .arg(&k_norm_dev)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(&base_pos_dev)
            .arg(&seq_len_i)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&use_qk_norm_i)
            .launch(cfg_kv)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!("launch kv_cache_write_seq: {e:?}"))
            })?;
    }

    let block_dim_attn: u32 = 256;
    // Tiled: fixed (TILE_K + bdim + head_dim).
    // Non-tiled: (n_ctx + bdim + head_dim) — PR #246 shmem-by-n_ctx fix.
    let shmem_slots = if use_tiled {
        FUSED_PREFILL_ATTN_TILED_TILE_K + block_dim_attn as usize + opts.head_dim
    } else {
        max_seq_i as usize + block_dim_attn as usize + opts.head_dim
    };
    let cfg_attn = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, seq_len as u32, 1),
        block_dim: (block_dim_attn, 1, 1),
        shared_mem_bytes: (shmem_slots * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;

    unsafe {
        drv.stream
            .launch_builder(func_attn)
            .arg(q_seq)
            .arg(&*k_cache)
            .arg(&*v_cache)
            .arg(&q_norm_dev)
            .arg(&mut out_seq)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(&base_pos_dev)
            .arg(&seq_len_i)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .launch(cfg_attn)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!("launch fused_prefill_attention: {e:?}"))
            })?;
    }

    Ok(out_seq)
}

#[cfg(all(test, any(feature = "cuda", feature = "cuda-oxide")))]
mod tree_mask_tests {
    //! `cuda-spec-branching-tree` T2.5+T2.6 — parity tests for the
    //! tree-mask batched-prefill attention kernel.
    //!
    //! Strategy: GPU-vs-GPU comparison against the existing causal
    //! `fused_prefill_attention_seq_device_into_pos_dev`. Re-using
    //! the production causal kernel as the oracle avoids re-deriving
    //! the full kernel arithmetic (RMS-Q, RoPE, GQA, softmax) in pure
    //! Rust, while still exactly exercising the tree-mask contract.
    //!
    //! Gated on `LARQL_CUDA_AVAILABLE=1` like the other CUDA tests.

    use super::{
        fused_prefill_attention_seq_device_into_pos_dev,
        fused_prefill_attention_tree_seq_device_into_pos_dev, FusedDecodeAttentionOpts,
    };
    use crate::cuda::CudaBackend;
    use half::f16;

    fn gpu_or_skip() -> Option<CudaBackend> {
        if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
            return None;
        }
        CudaBackend::new().ok()
    }

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFF_FFFF) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    fn make_opts(
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> FusedDecodeAttentionOpts {
        FusedDecodeAttentionOpts {
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos: 0,
            max_seq,
            rotary_dim: head_dim,
            rope_base: 10_000.0,
            eps: 1e-6,
            qk_norm_offset: 0.0,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            softcap: 0.0,
        }
    }

    /// Build f16 KV-cache device buffer, seeded with history at slot 0
    /// and zeros elsewhere.
    fn seed_kv_cache(
        backend: &CudaBackend,
        history_f32: &[f32],
        max_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> (
        cudarc::driver::CudaSlice<f16>,
        cudarc::driver::CudaSlice<f16>,
    ) {
        let cache_len = max_seq * num_kv_heads * head_dim;
        let mut host_k = vec![f16::from_f32(0.0); cache_len];
        let mut host_v = vec![f16::from_f32(0.0); cache_len];
        for (i, &v) in history_f32.iter().enumerate() {
            host_k[i] = f16::from_f32(v);
            host_v[i] = f16::from_f32(v);
        }
        let drv = backend.driver();
        let k_dev = drv.stream.clone_htod(&host_k).expect("memcpy k_cache seed");
        let v_dev = drv.stream.clone_htod(&host_v).expect("memcpy v_cache seed");
        (k_dev, v_dev)
    }

    #[test]
    fn tree_mask_linear_chain_bit_exact_vs_causal() {
        // T2.6: when ancestor bitsets are `(1 << (k+1)) - 1`
        // (= causal mask), the tree-mask kernel SHALL produce
        // bit-identical output to the existing causal kernel.
        let Some(backend) = gpu_or_skip() else { return };

        let num_q_heads = 8;
        let num_kv_heads = 2;
        let head_dim = 64;
        let max_seq = 128;
        let seq_len = 4;
        let base_pos: i32 = 16;

        let opts = make_opts(num_q_heads, num_kv_heads, head_dim, max_seq);
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_host = synth(seq_len * q_dim, 0xA);
        let k_host = synth(seq_len * kv_dim, 0xB);
        let v_host = synth(seq_len * kv_dim, 0xC);
        let history = synth(base_pos as usize * kv_dim, 0xD);
        let base_pos_host = [base_pos];

        let drv = backend.driver();
        let q_dev = drv.stream.clone_htod(&q_host).expect("q");
        let k_dev = drv.stream.clone_htod(&k_host).expect("k");
        let v_dev = drv.stream.clone_htod(&v_host).expect("v");
        let base_pos_dev = drv.stream.clone_htod(&base_pos_host).expect("base_pos");

        // Causal-only baseline.
        let (mut k_cache_a, mut v_cache_a) =
            seed_kv_cache(&backend, &history, max_seq, num_kv_heads, head_dim);
        let mut out_a = drv
            .stream
            .clone_htod(&vec![0.0_f32; seq_len * q_dim])
            .expect("out_a");
        fused_prefill_attention_seq_device_into_pos_dev(
            &backend,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut k_cache_a,
            &mut v_cache_a,
            None,
            None,
            &mut out_a,
            &base_pos_dev,
            seq_len,
            opts,
        )
        .expect("causal kernel");

        // Tree-mask with linear-chain bitsets.
        let (mut k_cache_b, mut v_cache_b) =
            seed_kv_cache(&backend, &history, max_seq, num_kv_heads, head_dim);
        let mut out_b = drv
            .stream
            .clone_htod(&vec![0.0_f32; seq_len * q_dim])
            .expect("out_b");
        let ancestors: Vec<u64> = (0..seq_len).map(|k| (1u64 << (k + 1)) - 1).collect();
        let ancestors_dev = drv.stream.clone_htod(&ancestors).expect("ancestors");
        fused_prefill_attention_tree_seq_device_into_pos_dev(
            &backend,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut k_cache_b,
            &mut v_cache_b,
            None,
            None,
            &mut out_b,
            &base_pos_dev,
            &ancestors_dev,
            seq_len,
            opts,
        )
        .expect("tree kernel");

        drv.sync().expect("sync");
        let host_a = drv.to_host(&out_a).expect("dtoh a");
        let host_b = drv.to_host(&out_b).expect("dtoh b");
        assert_eq!(host_a.len(), host_b.len());
        for (i, (a, b)) in host_a.iter().zip(host_b.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "linear-chain tree-mask must be bit-exact at idx {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn tree_mask_sibling_no_leak() {
        // T2.5 scenario "sibling branches don't leak attention".
        //
        //          0 (root)
        //         /        \
        //        1          2
        //
        // ancestors[2] = {0, 2} — excludes node 1. Changing node 1's
        // K/V (chain-B's slot) MUST NOT change node 2's output.
        let Some(backend) = gpu_or_skip() else { return };

        let num_q_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 32;
        let max_seq = 16;
        let seq_len = 3;
        let base_pos: i32 = 4;

        let opts = make_opts(num_q_heads, num_kv_heads, head_dim, max_seq);
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_host = synth(seq_len * q_dim, 0xF1);
        let k_host = synth(seq_len * kv_dim, 0xF2);
        let v_host = synth(seq_len * kv_dim, 0xF3);
        let history = synth(base_pos as usize * kv_dim, 0xF4);
        let base_pos_host = [base_pos];

        let ancestors: Vec<u64> = vec![0b001, 0b011, 0b101];

        let drv = backend.driver();
        let q_dev = drv.stream.clone_htod(&q_host).expect("q");
        let base_pos_dev = drv.stream.clone_htod(&base_pos_host).expect("base_pos");
        let ancestors_dev = drv.stream.clone_htod(&ancestors).expect("ancestors");

        let run = |poison_node_1: bool| -> Vec<f32> {
            let mut k_local = k_host.clone();
            let mut v_local = v_host.clone();
            if poison_node_1 {
                // Replace node 1's K/V row with a distinctive pattern.
                // Node 1's K/V occupies indices [kv_dim, 2*kv_dim).
                for s in k_local[kv_dim..2 * kv_dim].iter_mut() {
                    *s = 5.0;
                }
                for s in v_local[kv_dim..2 * kv_dim].iter_mut() {
                    *s = 7.0;
                }
            }
            let k_dev = drv.stream.clone_htod(&k_local).expect("k");
            let v_dev = drv.stream.clone_htod(&v_local).expect("v");
            let (mut k_cache, mut v_cache) =
                seed_kv_cache(&backend, &history, max_seq, num_kv_heads, head_dim);
            let mut out = drv
                .stream
                .clone_htod(&vec![0.0_f32; seq_len * q_dim])
                .expect("out");
            fused_prefill_attention_tree_seq_device_into_pos_dev(
                &backend,
                &q_dev,
                &k_dev,
                &v_dev,
                &mut k_cache,
                &mut v_cache,
                None,
                None,
                &mut out,
                &base_pos_dev,
                &ancestors_dev,
                seq_len,
                opts,
            )
            .expect("tree kernel");
            drv.sync().expect("sync");
            drv.to_host(&out).expect("dtoh")
        };

        let out_clean = run(false);
        let out_poisoned = run(true);

        // Node 2 occupies output rows [2 * q_dim, 3 * q_dim).
        let start = 2 * q_dim;
        let end = 3 * q_dim;
        for i in start..end {
            assert_eq!(
                out_clean[i].to_bits(),
                out_poisoned[i].to_bits(),
                "sibling leak at output idx {i} (node 2 row): \
                 clean {} vs poisoned {} — node 2 must not see node 1's K/V",
                out_clean[i],
                out_poisoned[i],
            );
        }
    }
}
