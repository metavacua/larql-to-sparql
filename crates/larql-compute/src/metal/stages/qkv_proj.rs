//! Q + K + V projections — one call per position.
//!
//! Three code paths depending on the weight format + mix:
//!
//! - **Fused f32-input** (`encode_fused_f32`): all three projections share
//!   the same format (Q4_K or Q4_KF) and we dispatch the llama.cpp-exact
//!   `q4kf_qkv_proj` shader in one go. Fastest path.
//! - **Per-projection f32-input** (`encode_per_proj`): mixed formats
//!   (e.g. Gemma 4 Q4_K Q/K + Q6_K V). Three separate shader dispatches.
//! - **Fused Q8-input** (`encode_fused_q8`): `Q8_0` attention layers use
//!   `q8_qkv_proj` with pre-quantised Q8 input from `input_norm::encode_q8`.
//!
//! All paths are per-position single-vector dispatches. Multi-position
//! prefill is achieved by looping over positions with buffer offsets.

use std::ffi::c_void;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

use super::quant_matvec;

/// Per-projection format + weight tuple used by the mixed-format path.
pub struct Proj<'a> {
    pub format: crate::QuantFormat,
    pub w_buf: &'a Buffer,
    pub out_buf: &'a Buffer,
    pub out_off: u64,
    pub rows: usize,
}

/// Fused Q4_K / Q4_KF QKV — all three projections same format.
///
/// Dispatches `q4kf_qkv_proj` (preferred, 144-byte GGUF) or its legacy
/// 148-byte fallback if only that's available. Writes Q / K / V outputs
/// at their respective byte offsets.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_f32(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    wq_buf: &Buffer,
    wk_buf: &Buffer,
    wv_buf: &Buffer,
    f32_in: &Buffer,
    f32_in_off: u64,
    q_out: &Buffer, q_off: u64,
    k_out: &Buffer, k_off: u64,
    v_out: &Buffer, v_off: u64,
    q_rows: usize, kv_rows: usize, hidden: usize,
) {
    use crate::metal::shaders::q4kf_qkv_proj as q4kf_qkv;
    let total_rows = (q_rows + kv_rows + kv_rows) as u32;
    let q_rows_val = q_rows as u32;
    let k_rows_val = kv_rows as u32;
    let v_rows_val = kv_rows as u32;
    let k_val = hidden as u32;
    let num_tgs = (total_rows as u64).div_ceil(q4kf_qkv::ROWS_PER_TG);
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(wq_buf), 0);
    enc.set_buffer(1, Some(wk_buf), 0);
    enc.set_buffer(2, Some(wv_buf), 0);
    enc.set_buffer(3, Some(f32_in), f32_in_off);
    enc.set_buffer(4, Some(q_out), q_off);
    enc.set_buffer(5, Some(k_out), k_off);
    enc.set_buffer(6, Some(v_out), v_off);
    enc.set_bytes(7, 4, &q_rows_val as *const u32 as *const c_void);
    enc.set_bytes(8, 4, &k_rows_val as *const u32 as *const c_void);
    enc.set_bytes(9, 4, &v_rows_val as *const u32 as *const c_void);
    enc.set_bytes(10, 4, &k_val as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(q4kf_qkv::THREADS_PER_TG, 1, 1),
    );
}

/// Per-projection f32-input QKV — mixed formats (Gemma 4 Q4_K + Q6_K).
///
/// One dispatch per projection, each through
/// [`super::quant_matvec::encode`] which picks the right shader by format.
/// The Q8 buffer parameters are only read for Q4_0 / Q8_0 projections;
/// callers with pure f32-input formats can pass any valid buffer + 0 offset.
#[allow(clippy::too_many_arguments)]
pub fn encode_per_proj(
    enc: &ComputeCommandEncoderRef,
    pipes: &quant_matvec::Pipelines<'_>,
    f32_in: &Buffer,
    f32_in_off: u64,
    q8_in: &Buffer,
    q8_in_off: u64,
    q8s_in: &Buffer,
    q8s_in_off: u64,
    projections: [Proj<'_>; 3],
    hidden: usize,
) {
    for p in projections {
        quant_matvec::encode(
            enc, p.format, p.w_buf,
            f32_in, f32_in_off,
            q8_in, q8_in_off, q8s_in, q8s_in_off,
            p.out_buf, p.out_off,
            pipes,
            p.rows, hidden,
        );
    }
}

/// Fused Q8-input QKV — for Q8_0 attention.
///
/// Input comes from `input_norm::encode_q8`. Weights are Q8 int8 + per-row
/// f32 scale buffers. `q8_qkv_proj` writes all three outputs in one dispatch.
#[allow(clippy::too_many_arguments)]
pub fn encode_fused_q8(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    wq_buf: &Buffer, wq_scale: &Buffer,
    wk_buf: &Buffer, wk_scale: &Buffer,
    wv_buf: &Buffer, wv_scale: &Buffer,
    q8_in: &Buffer, q8_in_off: u64,
    q8s_in: &Buffer, q8s_in_off: u64,
    q_out: &Buffer, q_off: u64,
    k_out: &Buffer, k_off: u64,
    v_out: &Buffer, v_off: u64,
    q_rows: usize, kv_rows: usize, hidden: usize,
) {
    let q_rows_val = q_rows as u32;
    let k_rows_val = kv_rows as u32;
    let v_rows_val = kv_rows as u32;
    let k_val = hidden as u32;
    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(wq_buf), 0);
    enc.set_buffer(1, Some(wk_buf), 0);
    enc.set_buffer(2, Some(wv_buf), 0);
    enc.set_buffer(3, Some(q8_in), q8_in_off);
    enc.set_buffer(4, Some(wq_scale), 0);
    enc.set_buffer(5, Some(wk_scale), 0);
    enc.set_buffer(6, Some(wv_scale), 0);
    enc.set_buffer(7, Some(q8s_in), q8s_in_off);
    enc.set_buffer(8, Some(q_out), q_off);
    enc.set_buffer(9, Some(k_out), k_off);
    enc.set_buffer(10, Some(v_out), v_off);
    enc.set_bytes(11, 4, &q_rows_val as *const u32 as *const c_void);
    enc.set_bytes(12, 4, &k_rows_val as *const u32 as *const c_void);
    enc.set_bytes(13, 4, &v_rows_val as *const u32 as *const c_void);
    enc.set_bytes(14, 4, &k_val as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(total_rows, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}
