//! Format-aware single-vector matvec dispatch.
//!
//! One entry point, `encode`, that routes to the right shader based on the
//! weight's quantization format:
//!
//! | format          | shader (preferred)   | input type | input buffer used |
//! |-----------------|----------------------|------------|--------------------|
//! | `Q4_K`, `Q4_KF` | `q4kf_proj`          | f32        | `f32_in` + offset  |
//! | `Q6_K`          | `q6k_matvec`         | f32        | `f32_in` + offset  |
//! | `Q4_0`, `Q8_0`  | `q4_matvec`          | Q8 + scales| `q8_in` + `q8s_in` |
//!
//! The same dispatch is used by two callers in the Metal pipeline:
//!
//! 1. **Per-projection QKV / O fallback** (`full_pipeline.rs`, `decode.rs`).
//!    Gemma 4 mixed-quant vindexes (Q4_K Q/K/O + Q6_K V) can't use the
//!    fused `q4kf_qkv_proj` shader and fall back to three separate calls
//!    through this helper.
//!
//! 2. **FFN gate/up/down** with format-aware routing (Gemma 4 ships Q4_K
//!    gate/up + Q6_K down). The same `encode` function handles all three.
//!
//! All dispatches are single-vector: one input row × N output rows. For
//! multi-position prefill the caller loops over positions, passing
//! `f32_in_off` / `out_off` in bytes.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

use crate::metal::kernel::KernelHandle;

/// Single-vector matvec dispatch for kernels whose threadgroup geometry
/// travels with their `KernelHandle`. Avoids duplicating the 8-line
/// dispatch pattern across each `QuantFormat` arm.
#[allow(clippy::too_many_arguments)]
fn dispatch_kh(
    enc: &ComputeCommandEncoderRef,
    kh: &KernelHandle,
    w_buf: &Buffer,
    f32_in: &Buffer,
    f32_in_off: u64,
    out_buf: &Buffer,
    out_off: u64,
    n: u32,
    k: u32,
) {
    let num_tgs = (n as u64).div_ceil(kh.rows_per_tg);
    enc.set_compute_pipeline_state(&kh.state);
    enc.set_buffer(0, Some(w_buf), 0);
    enc.set_buffer(1, Some(f32_in), f32_in_off);
    enc.set_buffer(2, Some(out_buf), out_off);
    enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(kh.threads_per_tg, 1, 1),
    );
}

/// Metal shader pipelines this stage may dispatch, in one bundle.
///
/// Not every caller has every pipeline (e.g. the legacy benchmark path
/// passes `None` for `q4kf_proj`). The dispatcher falls back to
/// `q4k_matvec_fallback` when the preferred shader is absent.
///
/// All fields are now `&KernelHandle` so geometry travels with the
/// pipeline — the bug class where a different pipeline (e.g. `q4k_proj`)
/// was passed in the matvec slot and the dispatch used the WRONG
/// `ROWS_PER_TG` from the shader module is now caught at compile time.
pub struct Pipelines<'a> {
    /// Preferred shader for `Q4_K` / `Q4_KF` — 144-byte GGUF llama.cpp-exact.
    pub q4kf_proj: Option<&'a ComputePipelineState>,
    /// Fallback for `Q4_K` if `q4kf_proj` is unavailable.
    pub q4k_matvec_fallback: &'a KernelHandle,
    pub q6k_matvec: &'a KernelHandle,
    pub q4_matvec: &'a KernelHandle,
    /// Q4_K matmul (gemm) — amortises dequant across `seq_len` positions
    /// in a single dispatch. When present and the call-site has
    /// `seq_len > 1`, the dispatcher prefers this over `seq_len`
    /// independent matvec calls. `None` falls back to per-position matvec
    /// (e.g. legacy benchmarks that don't bind the matmul pipeline).
    pub q4k_matmul: Option<&'a KernelHandle>,
}

/// Encode a single-vector matvec `out[N] = W[N×K] · x[K]` onto `enc`.
///
/// * `w_buf` is the quantised weight buffer for the full `N` rows.
/// * `f32_in` / `f32_in_off` supply a `K`-float vector (used for Q4_K /
///   Q4_KF / Q6_K which consume f32 directly).
/// * `q8_in` / `q8_in_off` / `q8s_in` / `q8s_in_off` supply the Q8-quantised
///   version (used for Q4_0 / Q8_0). For Q4_K / Q4_KF / Q6_K these can
///   point anywhere — they're not read.
/// * `out_buf` / `out_off` is the `N`-float output slot.
///
/// Does not call `end_encoding` — the caller owns the encoder lifecycle.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    format: crate::QuantFormat,
    w_buf: &Buffer,
    f32_in: &Buffer,
    f32_in_off: u64,
    q8_in: &Buffer,
    q8_in_off: u64,
    q8s_in: &Buffer,
    q8s_in_off: u64,
    out_buf: &Buffer,
    out_off: u64,
    pipes: &Pipelines<'_>,
    num_rows: usize,
    hidden: usize,
) {
    let n = num_rows as u32;
    let k = hidden as u32;
    match format {
        crate::QuantFormat::Q4_KF => {
            // Q4_KF: dispatch the llama.cpp-exact pre-baked-scale shader.
            // Falls back to the canonical Q4_K matvec if the Q4_KF pipeline
            // wasn't compiled into this backend.
            if let Some(q4kf_proj_pipe) = pipes.q4kf_proj {
                use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                let num_tgs = (num_rows as u64).div_ceil(q4kf::ROWS_PER_TG);
                enc.set_compute_pipeline_state(q4kf_proj_pipe);
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(f32_in), f32_in_off);
                enc.set_buffer(2, Some(out_buf), out_off);
                enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_tgs, 1, 1),
                    MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
                );
            } else {
                dispatch_kh(
                    enc,
                    pipes.q4k_matvec_fallback,
                    w_buf,
                    f32_in,
                    f32_in_off,
                    out_buf,
                    out_off,
                    n,
                    k,
                );
            }
        }
        crate::QuantFormat::Q4_K => {
            // Q4_K weights must dispatch the Q4_K kernel (8 rows/TG, 256
            // threads). Routing them through the Q4_KF kernel both
            // misinterprets the format (Q4_KF uses pre-baked half-scales)
            // and gets the threadgroup geometry wrong (4 rows / 64 threads),
            // leaving ~75% of output rows unwritten.
            if crate::options::env_flag(crate::options::ENV_DBG_QM) {
                eprintln!(
                    "[quant_matvec] Q4_K path — kh.rows_per_tg={} kh.threads_per_tg={} n={} k={}",
                    pipes.q4k_matvec_fallback.rows_per_tg,
                    pipes.q4k_matvec_fallback.threads_per_tg,
                    n,
                    k
                );
            }
            dispatch_kh(
                enc,
                pipes.q4k_matvec_fallback,
                w_buf,
                f32_in,
                f32_in_off,
                out_buf,
                out_off,
                n,
                k,
            );
        }
        crate::QuantFormat::Q6_K => {
            let kh = pipes.q6k_matvec;
            let num_tgs = (num_rows as u64).div_ceil(kh.rows_per_tg);
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(f32_in), f32_in_off);
            enc.set_buffer(2, Some(out_buf), out_off);
            enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        }
        crate::QuantFormat::Q4_0 => {
            // Q4_0 matvec expects Q8 input + Q8 scales (per-32 f16-scaled
            // blocks). Geometry travels with the kernel handle.
            let kernel = pipes.q4_matvec;
            let num_tgs = (num_rows as u64).div_ceil(kernel.rows_per_tg);
            enc.set_compute_pipeline_state(&kernel.state);
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(q8_in), q8_in_off);
            enc.set_buffer(2, Some(q8s_in), q8s_in_off);
            enc.set_buffer(3, Some(out_buf), out_off);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &k as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(kernel.threads_per_tg, 1, 1),
            );
        }
        crate::QuantFormat::Q8_0 => {
            // Q8_0 weights are NOT a Q4_0 kernel input — Q8_0 blocks are
            // 34 bytes per 32 values while Q4_0 is 18. Pre-2026-05-09
            // this branch shared the Q4_0 arm above, which read the
            // wrong byte stride and produced garbage. Production decode
            // routes Q8_0 weights through `quant.q8_matvec_pipeline` and
            // `attention.q8_qkv_proj_pipeline` directly (see
            // `metal/decode/encode_qkv.rs::encode_q4_0_norm_and_qkv`),
            // not through this generic dispatcher. If a future caller
            // wants Q8_0 inside the FFN-style `qmv::encode` path,
            // extend `Pipelines` with `q8_matvec` and dispatch it here.
            panic!(
                "metal::stages::quant_matvec::encode: Q8_0 not yet routed \
                 through this generic dispatcher. Use \
                 `quant.q8_matvec_pipeline` directly, or extend the \
                 `Pipelines` struct to carry a Q8 kernel."
            );
        }
        crate::QuantFormat::BF16 | crate::QuantFormat::F16 | crate::QuantFormat::F32 => {
            // Not dispatchable via this Q4 shader path — caller should use
            // a float matvec or dequantize before calling.
        }
    }
}
