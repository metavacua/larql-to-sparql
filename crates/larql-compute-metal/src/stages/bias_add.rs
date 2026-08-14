//! Attention projection bias add — GPT-OSS's Q/K/V/O biases.
//!
//! Element-wise `out[i] += bias[i]` over one projection's output row,
//! dispatched only when the layer carries the bias (the CPU authority is
//! `forward::add_bias`, applied at the same four points). The bias must
//! land before anything downstream reads the projection: before
//! QK-norm/RoPE for Q/K, before the KV-cache append for V, before the
//! residual add for O.
//!
//! Caller owns the encoder lifecycle. `out_off` lets prefill dispatch per
//! position into a `[seq, dim]` buffer.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

/// Add `n` bias elements onto `out` at byte offset `out_off`.
///
/// `pipeline` must be the `bias_add` shader's pipeline
/// ([`crate::kernels::attention::AttentionKernels::bias_add_pipeline`]).
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    out: &Buffer,
    out_off: u64,
    bias: &Buffer,
    n: usize,
) {
    let n_val = n as u32;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(out), out_off);
    enc.set_buffer(1, Some(bias), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const c_void);
    enc.dispatch_threads(
        MTLSize::new(n as u64, 1, 1),
        MTLSize::new(crate::kernels::DISPATCH_TG_MAX_THREADS.min(n as u64), 1, 1),
    );
}
