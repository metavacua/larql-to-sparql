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

use std::ffi::c_void;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

/// Metal shader pipelines this stage may dispatch, in one bundle.
///
/// Not every caller has every pipeline (e.g. the legacy benchmark path
/// passes `None` for `q4kf_proj`). The dispatcher falls back to
/// `q4k_matvec_fallback` when the preferred shader is absent.
pub struct Pipelines<'a> {
    /// Preferred shader for `Q4_K` / `Q4_KF` — 144-byte GGUF llama.cpp-exact.
    pub q4kf_proj: Option<&'a ComputePipelineState>,
    /// Fallback for `Q4_K` if `q4kf_proj` is unavailable.
    pub q4k_matvec_fallback: &'a ComputePipelineState,
    pub q6k_matvec: &'a ComputePipelineState,
    pub q4_matvec: &'a ComputePipelineState,
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
        crate::QuantFormat::Q4_K | crate::QuantFormat::Q4_KF => {
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
                use crate::metal::shaders::q4k_matvec as q4k;
                let num_tgs = (num_rows as u64).div_ceil(q4k::ROWS_PER_TG);
                enc.set_compute_pipeline_state(pipes.q4k_matvec_fallback);
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(f32_in), f32_in_off);
                enc.set_buffer(2, Some(out_buf), out_off);
                enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_tgs, 1, 1),
                    MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                );
            }
        }
        crate::QuantFormat::Q6_K => {
            use crate::metal::shaders::q6k_matvec as q6k;
            let num_tgs = (num_rows as u64).div_ceil(q6k::ROWS_PER_TG);
            enc.set_compute_pipeline_state(pipes.q6k_matvec);
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(f32_in), f32_in_off);
            enc.set_buffer(2, Some(out_buf), out_off);
            enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
            );
        }
        crate::QuantFormat::Q4_0 | crate::QuantFormat::Q8_0 => {
            // Q4_0 matvec expects Q8 input + Q8 scales (per-32 f16-scaled blocks).
            use crate::metal::shaders::q4_matvec as q4mv;
            let num_tgs = (num_rows as u64).div_ceil(q4mv::ROWS_PER_TG);
            enc.set_compute_pipeline_state(pipes.q4_matvec);
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(q8_in), q8_in_off);
            enc.set_buffer(2, Some(q8s_in), q8s_in_off);
            enc.set_buffer(3, Some(out_buf), out_off);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &k as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(q4mv::THREADS_PER_TG, 1, 1),
            );
        }
    }
}
