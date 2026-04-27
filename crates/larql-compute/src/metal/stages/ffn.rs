//! Feed-forward block — gate+up → activation → down.
//!
//! Two variants depending on `FfnType`:
//!
//! - **Gated** (Llama / Gemma / Qwen / most modern): `out = down(act(gate) ⊙ up)`
//!   with activation = SiLU or GELU-tanh. Dispatched as
//!   `gate_matvec + up_matvec + geglu + down_matvec`.
//!
//! - **Standard** (StarCoder2): `out = down(act(up))`. Dispatched as
//!   `up_matvec + activation + down_matvec`. No gate.
//!
//! All matvecs are format-aware (`stages::quant_matvec`). Activation is a
//! single multi-position dispatch over `seq_len * inter` elementwise
//! threads.

use std::ffi::c_void;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

use super::quant_matvec;

/// Activation variant for this layer.
#[derive(Clone, Copy)]
pub enum Activation {
    SiLU,
    GeluTanh,
}

/// Gated FFN (Llama / Gemma / Qwen): `down(act(gate) * up)`.
#[allow(clippy::too_many_arguments)]
pub fn encode_gated(
    enc: &ComputeCommandEncoderRef,
    pipes: &quant_matvec::Pipelines<'_>,
    geglu_silu_pipeline: &ComputePipelineState,
    geglu_gelu_tanh_pipeline: &ComputePipelineState,
    gate_format: crate::QuantFormat,
    up_format: crate::QuantFormat,
    down_format: crate::QuantFormat,
    activation: Activation,
    gate_buf: &Buffer, up_buf: &Buffer, down_buf: &Buffer,
    ffn_norm_out: &Buffer, // f32 input for Q4_K / Q6_K / Q4_KF
    ffn_q8_in: &Buffer,    // Q8 input for Q4_0 / Q8_0
    ffn_q8s_in: &Buffer,
    gate_scratch: &Buffer, // holds per-position `inter` floats
    up_scratch: &Buffer,
    act_scratch: &Buffer,
    down_out: &Buffer,
    seq_len: usize,
    inter: usize, hidden: usize,
    h_stride_bytes: u64,       // hidden * 4
    inter_stride_bytes: u64,   // inter * 4
    q8_stride_bytes: u64,      // Q8 input bytes per pos
    q8s_stride_bytes: u64,     // Q8 scales bytes per pos
) {
    // Gate+up per position.
    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;
        quant_matvec::encode(
            enc, gate_format, gate_buf,
            ffn_norm_out, h_off,
            ffn_q8_in, q8_off, ffn_q8s_in, q8s_off,
            gate_scratch, inter_off,
            pipes,
            inter, hidden,
        );
        quant_matvec::encode(
            enc, up_format, up_buf,
            ffn_norm_out, h_off,
            ffn_q8_in, q8_off, ffn_q8s_in, q8s_off,
            up_scratch, inter_off,
            pipes,
            inter, hidden,
        );
    }

    // Multi-position elementwise GEGLU.
    {
        let total_inter = (seq_len * inter) as u64;
        let total_inter_val = (seq_len * inter) as u32;
        let geglu_pipe = match activation {
            Activation::GeluTanh => geglu_gelu_tanh_pipeline,
            Activation::SiLU => geglu_silu_pipeline,
        };
        enc.set_compute_pipeline_state(geglu_pipe);
        enc.set_buffer(0, Some(gate_scratch), 0);
        enc.set_buffer(1, Some(up_scratch), 0);
        enc.set_buffer(2, Some(act_scratch), 0);
        enc.set_bytes(3, 4, &total_inter_val as *const u32 as *const c_void);
        enc.dispatch_threads(MTLSize::new(total_inter, 1, 1), MTLSize::new(256, 1, 1));
    }

    // Down projection per position. Q4_K / Q4_KF / Q6_K take f32 input
    // (no Q8 staging). Q4_0 / Q8_0 here fall through the generic path —
    // today no production vindex uses those formats for down.
    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;
        quant_matvec::encode(
            enc, down_format, down_buf,
            act_scratch, inter_off,
            ffn_q8_in, q8_off, ffn_q8s_in, q8s_off,
            down_out, h_off,
            pipes,
            hidden, inter,
        );
    }
}

/// Standard FFN (StarCoder2): `down(act(up))`. No gate.
#[allow(clippy::too_many_arguments)]
pub fn encode_standard(
    enc: &ComputeCommandEncoderRef,
    pipes: &quant_matvec::Pipelines<'_>,
    silu_pipeline: &ComputePipelineState,
    gelu_tanh_pipeline: &ComputePipelineState,
    up_format: crate::QuantFormat,
    down_format: crate::QuantFormat,
    activation: Activation,
    up_buf: &Buffer, down_buf: &Buffer,
    ffn_norm_out: &Buffer,
    ffn_q8_in: &Buffer,
    ffn_q8s_in: &Buffer,
    up_scratch: &Buffer,
    act_scratch: &Buffer,
    down_out: &Buffer,
    seq_len: usize,
    inter: usize, hidden: usize,
    h_stride_bytes: u64,
    inter_stride_bytes: u64,
    q8_stride_bytes: u64,
    q8s_stride_bytes: u64,
) {
    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;
        quant_matvec::encode(
            enc, up_format, up_buf,
            ffn_norm_out, h_off,
            ffn_q8_in, q8_off, ffn_q8s_in, q8s_off,
            up_scratch, inter_off,
            pipes,
            inter, hidden,
        );
    }

    {
        let total_inter = (seq_len * inter) as u64;
        let total_inter_val = (seq_len * inter) as u32;
        let act_pipe = match activation {
            Activation::GeluTanh => gelu_tanh_pipeline,
            Activation::SiLU => silu_pipeline,
        };
        enc.set_compute_pipeline_state(act_pipe);
        enc.set_buffer(0, Some(up_scratch), 0);
        enc.set_buffer(1, Some(act_scratch), 0);
        enc.set_bytes(2, 4, &total_inter_val as *const u32 as *const c_void);
        enc.dispatch_threads(MTLSize::new(total_inter, 1, 1), MTLSize::new(256, 1, 1));
    }

    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride_bytes;
        let q8s_off = pos as u64 * q8s_stride_bytes;
        quant_matvec::encode(
            enc, down_format, down_buf,
            act_scratch, inter_off,
            ffn_q8_in, q8_off, ffn_q8s_in, q8s_off,
            down_out, h_off,
            pipes,
            hidden, inter,
        );
    }
}
