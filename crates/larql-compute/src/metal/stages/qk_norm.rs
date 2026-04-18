//! QK-norm and V-norm — per-head RMS norm applied inside attention.
//!
//! All three variants use the same `qk_norm` shader (one TG per head,
//! cooperative simdgroup reduction). They differ only in:
//!   - Whose buffer they target (Q vs K vs V)
//!   - Which weight they multiply (learned q_norm / k_norm / all-ones)
//!   - The norm offset (Gemma 2/3 stores `weight - 1` → offset 1.0;
//!     Gemma 4 stores raw → offset 0.0; V-norm is parameter-free →
//!     offset 0.0, weight = 1.0)

use std::ffi::c_void;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

/// Compute the threadgroup width for a `head_dim`-long cooperative reduction.
/// Rounds up to a power of two, capped at 512 (shader limit).
fn tg_width(head_dim: usize) -> u64 {
    let mut tg: u64 = 1;
    while (tg as usize) < head_dim && tg < 512 { tg <<= 1; }
    tg
}

/// Per-head RMS norm on Q and K (pre-RoPE, Gemma 3 / Gemma 4).
///
/// One shader dispatch per head per position. Writes back to the same Q/K
/// buffers (in-place). Returns `true` on success so the caller can tell
/// `fused_attention` to skip its internal QK-norm (otherwise double-norm).
/// Returns `false` if the pipeline or weights are absent — the caller
/// should then fall back to the shader's internal normalisation.
#[allow(clippy::too_many_arguments)]
pub fn encode_qk_norm(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    q_buf: &Buffer, q_w_buf: &Buffer,
    k_buf: &Buffer, k_w_buf: &Buffer,
    seq_len: usize,
    num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
    eps: f32, qk_norm_offset: f32,
) {
    let hd_val = head_dim as u32;
    let nq_val = num_q_heads as u32;
    let nkv_val = num_kv_heads as u32;
    let tg_w = tg_width(head_dim);

    for pos in 0..seq_len {
        let q_buf_off = (pos * num_q_heads * head_dim * 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(q_buf), q_buf_off);
        enc.set_buffer(1, Some(q_buf), q_buf_off);
        enc.set_buffer(2, Some(q_w_buf), 0);
        enc.set_bytes(3, 4, &hd_val as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nq_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
        enc.set_bytes(6, 4, &qk_norm_offset as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_q_heads as u64, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );

        let k_buf_off = (pos * num_kv_heads * head_dim * 4) as u64;
        enc.set_buffer(0, Some(k_buf), k_buf_off);
        enc.set_buffer(1, Some(k_buf), k_buf_off);
        enc.set_buffer(2, Some(k_w_buf), 0);
        enc.set_bytes(4, 4, &nkv_val as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_kv_heads as u64, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );
    }
}

/// Parameter-free per-head RMS norm on V (Gemma 4).
///
/// Weight is implicitly 1.0 (shader still takes a weight buffer — the
/// caller stages an all-ones vector of length `head_dim`). Offset is 0.
pub fn encode_v_norm(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    v_buf: &Buffer, ones_buf: &Buffer,
    seq_len: usize,
    num_kv_heads: usize, head_dim: usize,
    eps: f32,
) {
    let hd_val = head_dim as u32;
    let nkv_val = num_kv_heads as u32;
    let zero_off: f32 = 0.0;
    let tg_w = tg_width(head_dim);

    for pos in 0..seq_len {
        let v_buf_off = (pos * num_kv_heads * head_dim * 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(v_buf), v_buf_off);
        enc.set_buffer(1, Some(v_buf), v_buf_off);
        enc.set_buffer(2, Some(ones_buf), 0);
        enc.set_bytes(3, 4, &hd_val as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nkv_val as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
        enc.set_bytes(6, 4, &zero_off as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_kv_heads as u64, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );
    }
}
