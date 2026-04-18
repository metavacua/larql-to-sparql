//! Per-layer residual scalar — Gemma 4's learned stabiliser.
//!
//! Multiplies the layer's final residual (`h_bufs[l + 1]`) by a per-layer
//! scalar typically in the range 0.02–0.8. Without this the residual
//! magnitude explodes across layers because Gemma 4's post-attention norm
//! weights can reach ~100. Mirrors `apply_layer_scalar` on the CPU path
//! and Step 8 of `decode_token`.
//!
//! Scoped to positions 0..seq_len for multi-position prefill; decode
//! calls with seq_len = 1.
//!
//! Caller owns the encoder lifecycle.

use std::ffi::c_void;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

/// If `scalar` is non-zero, scale the f32 residual at each position by `scalar`.
///
/// * `h_buf` is the residual buffer holding `seq_len × hidden` f32s starting
///   at byte 0, one `hidden`-sized slice per position.
/// * `pipeline` must be the pipeline for the `scale_vector` shader.
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    h_buf: &Buffer,
    seq_len: usize,
    hidden: usize,
    scalar: f32,
) {
    if scalar == 0.0 { return; }
    let hidden_val = hidden as u32;
    for pos in 0..seq_len {
        let h_off = (pos * hidden * 4) as u64;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(h_buf), h_off);
        enc.set_buffer(1, Some(h_buf), h_off);
        enc.set_bytes(2, 4, &hidden_val as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &scalar as *const f32 as *const c_void);
        // `scale_vector` uses `thread_position_in_grid` — one thread per
        // element, not a single 256-thread threadgroup.
        enc.dispatch_threads(
            MTLSize::new(hidden as u64, 1, 1),
            MTLSize::new(256.min(hidden as u64), 1, 1),
        );
    }
}
