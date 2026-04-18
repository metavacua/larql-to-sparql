//! Full pipeline: ALL Q4 (attention + FFN) in ONE Metal command buffer.
//!
//! Correct inference path with norms and residual connections:
//!   Per layer:
//!     1. rms_norm(h, input_norm) → h_norm
//!     2. Q4 Q/K/V projections from h_norm
//!     3. Fused attention (RoPE + GQA + softcap)
//!     4. Q4 O projection
//!     5. Post-attn norm (if post_norms) + residual_add(h, o_out) → h
//!     6. rms_norm(h, post_attn_norm) → h_ffn
//!     7. Q4 gate/up → GEGLU → Q4 down
//!     8. Post-FFN norm (if post_norms) + residual_add(h, ffn_out) → h
//!     9. Q8 quantize h → next layer

use std::ffi::c_void;
use metal::*;

use crate::metal::buffers::BufferCache;
use crate::metal::shaders::q4_matvec as q4mv_shader;
use super::q4_common::Q4Pipelines;

/// Weights for one transformer layer — ALL Q4 + norm weights.
/// Matches `crate::FullPipelineLayer` but with borrowed Metal-friendly data.
pub struct LayerWeights<'a> {
    pub wq_q4: &'a [u8],
    pub wk_q4: &'a [u8],
    pub wv_q4: &'a [u8],
    pub wo_q4: &'a [u8],
    pub gate_q4: &'a [u8],
    pub up_q4: &'a [u8],
    pub down_t_q4: &'a [u8],
}

#[allow(dead_code, clippy::too_many_arguments)]
fn encode_q4_matvec(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buf_q4: &Buffer,
    buf_q8: &Buffer,
    buf_q8s: &Buffer,
    buf_out: &Buffer,
    num_rows: usize,
    hidden: usize,
) {
    let n_val = num_rows as u32;
    let k_val = hidden as u32;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buf_q4), 0);
    enc.set_buffer(1, Some(buf_q8), 0);
    enc.set_buffer(2, Some(buf_q8s), 0);
    enc.set_buffer(3, Some(buf_out), 0);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
    let num_tgs = (num_rows as u64).div_ceil(q4mv_shader::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(q4mv_shader::THREADS_PER_TG, 1, 1),
    );
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn encode_q8_matvec(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buf_w8: &Buffer,     // Q8 weight int8 values
    buf_q8: &Buffer,     // Q8 input int8 values
    buf_w8s: &Buffer,    // Q8 weight per-block scales
    buf_q8s: &Buffer,    // Q8 input per-block scales
    buf_out: &Buffer,
    num_rows: usize,
    hidden: usize,
) {
    let n_val = num_rows as u32;
    let k_val = hidden as u32;
    let rows_per_tg = 8u64;
    let num_tgs = (num_rows as u64).div_ceil(rows_per_tg);
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buf_w8), 0);
    enc.set_buffer(1, Some(buf_q8), 0);
    enc.set_buffer(2, Some(buf_w8s), 0);
    enc.set_buffer(3, Some(buf_q8s), 0);
    enc.set_buffer(4, Some(buf_out), 0);
    enc.set_bytes(5, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &k_val as *const u32 as *const c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

#[allow(clippy::too_many_arguments)]
pub fn encode_rms_norm(
    enc: &ComputeCommandEncoderRef,
    rms_pipeline: &ComputePipelineState,
    buf_x: &Buffer,
    buf_weight: &Buffer,
    buf_out: &Buffer,
    len: usize,
    eps: f32,
    offset: f32,
) {
    let len_val = len as u32;
    enc.set_compute_pipeline_state(rms_pipeline);
    enc.set_buffer(0, Some(buf_x), 0);
    enc.set_buffer(1, Some(buf_weight), 0);
    enc.set_buffer(2, Some(buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const c_void);
    // Single threadgroup — cooperative SIMD reduction requires all threads in one TG.
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(len as u64), 1, 1));
}

pub fn encode_residual_add(
    enc: &ComputeCommandEncoderRef,
    add_pipeline: &ComputePipelineState,
    buf_a: &Buffer,
    buf_b: &Buffer,
    buf_out: &Buffer,
    len: usize,
) {
    let len_val = len as u32;
    enc.set_compute_pipeline_state(add_pipeline);
    enc.set_buffer(0, Some(buf_a), 0);
    enc.set_buffer(1, Some(buf_b), 0);
    enc.set_buffer(2, Some(buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const c_void);
    enc.dispatch_threads(MTLSize::new(len as u64, 1, 1), MTLSize::new(256.min(len as u64), 1, 1));
}

/// Q4_0 matvec with explicit input/output offsets (bytes).
/// Same as `encode_q4_matvec` but lets the caller point at a specific row of
/// a multi-position staging buffer — used in prefill (`seq_len > 1`) where
/// each position's Q8 input and output live at `pos * stride` byte offsets.
#[allow(clippy::too_many_arguments)]
fn encode_q4_matvec_offset(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buf_q4: &Buffer,
    buf_q8: &Buffer,
    q8_off: u64,
    buf_q8s: &Buffer,
    q8s_off: u64,
    buf_out: &Buffer,
    out_off: u64,
    num_rows: usize,
    hidden: usize,
) {
    let n_val = num_rows as u32;
    let k_val = hidden as u32;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buf_q4), 0);
    enc.set_buffer(1, Some(buf_q8), q8_off);
    enc.set_buffer(2, Some(buf_q8s), q8s_off);
    enc.set_buffer(3, Some(buf_out), out_off);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
    let num_tgs = (num_rows as u64).div_ceil(q4mv_shader::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tgs, 1, 1),
        MTLSize::new(q4mv_shader::THREADS_PER_TG, 1, 1),
    );
}

/// Format-dispatched quant matvec with explicit input/output byte offsets.
/// Mirrors `encode_quant_matvec` but takes `in_off` / `out_off` byte offsets
/// so a single backing buffer can hold `seq_len` rows addressed by position.
/// Q4_K / Q6_K / Q4_KF read f32 input at `in_off`; Q4_0 / Q8_0 read Q8 input.
#[allow(dead_code, clippy::too_many_arguments)]
fn encode_quant_matvec_offset(
    enc: &ComputeCommandEncoderRef,
    format: crate::QuantFormat,
    q4_pipeline: &ComputePipelineState,
    q8_pipeline: &ComputePipelineState,
    q4k_pipeline: &ComputePipelineState,
    q6k_pipeline: &ComputePipelineState,
    buf_w: &Buffer,
    buf_input: &Buffer,
    in_off: u64,
    _buf_scales: &Buffer,
    buf_input_scales: &Buffer,
    buf_out: &Buffer,
    out_off: u64,
    num_rows: usize,
    hidden: usize,
) {
    match format {
        crate::QuantFormat::Q4_K | crate::QuantFormat::Q4_KF => {
            use crate::metal::shaders::q4k_matvec as q4k;
            let n = num_rows as u32;
            let k = hidden as u32;
            let tgs = (num_rows as u64).div_ceil(q4k::ROWS_PER_TG);
            enc.set_compute_pipeline_state(q4k_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), in_off);
            enc.set_buffer(2, Some(buf_out), out_off);
            enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
        }
        crate::QuantFormat::Q6_K => {
            use crate::metal::shaders::q6k_matvec as q6k;
            let n = num_rows as u32;
            let k = hidden as u32;
            let tgs = (num_rows as u64).div_ceil(q6k::ROWS_PER_TG);
            enc.set_compute_pipeline_state(q6k_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), in_off);
            enc.set_buffer(2, Some(buf_out), out_off);
            enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(q6k::THREADS_PER_TG, 1, 1));
        }
        crate::QuantFormat::Q4_0 => {
            // Q4_0 with Q8 input + (weight) scales + input scales.
            let n_val = num_rows as u32;
            let k_val = hidden as u32;
            enc.set_compute_pipeline_state(q4_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), in_off);
            enc.set_buffer(2, Some(buf_input_scales), 0);
            enc.set_buffer(3, Some(buf_out), out_off);
            enc.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &k_val as *const u32 as *const c_void);
            let num_tgs = (num_rows as u64).div_ceil(q4mv_shader::ROWS_PER_TG);
            enc.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(q4mv_shader::THREADS_PER_TG, 1, 1),
            );
        }
        crate::QuantFormat::Q8_0 => {
            let n = num_rows as u32;
            let k = hidden as u32;
            let rows_per_tg = 8u64;
            let num_tgs = (num_rows as u64).div_ceil(rows_per_tg);
            enc.set_compute_pipeline_state(q8_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), in_off);
            enc.set_buffer(2, Some(_buf_scales), 0);
            enc.set_buffer(3, Some(buf_input_scales), 0);
            enc.set_buffer(4, Some(buf_out), out_off);
            enc.set_bytes(5, 4, &n as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &k as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_tgs, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
    }
}

/// Format-aware FFN matvec (gate / up): N rows × K cols. Reads either f32
/// input (Q4_K / Q4_KF / Q6_K) or Q8 input (Q4_0 / Q8_0) and writes f32 output.
/// The `f32_in` buffer holds one hidden-sized f32 vector per position starting
/// at `f32_in_off` bytes; `q8_in` / `q8s_in` hold the Q8 version at their own
/// offsets. All matvecs are single-vector dispatches.
#[allow(clippy::too_many_arguments)]
fn dispatch_ffn_matvec(
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
    q4k_pipeline: &ComputePipelineState,
    q6k_pipeline: &ComputePipelineState,
    q4kf_proj_pipeline: Option<&ComputePipelineState>,
    q4_matvec_pipeline: &ComputePipelineState,
    num_rows: usize,
    hidden: usize,
) {
    let n = num_rows as u32;
    let k = hidden as u32;
    match format {
        crate::QuantFormat::Q4_K | crate::QuantFormat::Q4_KF => {
            // Prefer `q4kf_proj` (144-byte GGUF, llama.cpp-exact inner loop).
            if let Some(q4kf_proj_pipe) = q4kf_proj_pipeline {
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
                enc.set_compute_pipeline_state(q4k_pipeline);
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
            enc.set_compute_pipeline_state(q6k_pipeline);
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
            enc.set_compute_pipeline_state(q4_matvec_pipeline);
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

/// Dispatch a matvec based on the weight's quantization format.
/// Q4_K/Q6_K take f32 input. Q8_0/Q4_0 take Q8 input.
#[allow(dead_code, clippy::too_many_arguments)]
fn encode_quant_matvec(
    enc: &ComputeCommandEncoderRef,
    format: crate::QuantFormat,
    q4_pipeline: &ComputePipelineState,
    q8_pipeline: &ComputePipelineState,
    q4k_pipeline: &ComputePipelineState,
    q6k_pipeline: &ComputePipelineState,
    buf_w: &Buffer,
    buf_input: &Buffer,        // f32 for Q4_K/Q6_K, Q8 int8 for Q4_0/Q8_0
    buf_scales: &Buffer,       // Q8 weight scales (Q8_0 only) or input scales
    buf_input_scales: &Buffer, // Q8 input scales (Q8_0 only)
    buf_out: &Buffer,
    num_rows: usize,
    hidden: usize,
) {
    match format {
        crate::QuantFormat::Q4_K => {
            let n = num_rows as u32;
            let k = hidden as u32;
            let tgs = (num_rows as u64).div_ceil(4); // Q4_K: 4 rows per TG
            enc.set_compute_pipeline_state(q4k_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), 0);  // f32 input
            enc.set_buffer(2, Some(buf_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(128, 1, 1));
        }
        crate::QuantFormat::Q6_K => {
            let n = num_rows as u32;
            let k = hidden as u32;
            let tgs = (num_rows as u64).div_ceil(4);
            enc.set_compute_pipeline_state(q6k_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), 0);
            enc.set_buffer(2, Some(buf_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(128, 1, 1));
        }
        crate::QuantFormat::Q4_KF => {
            // Q4_KF: same as Q4_K but data layout is different (pre-baked scales)
            // Uses the same q4k_matvec pipeline (standalone) as fallback
            // In practice, Q4_KF goes through the fused QKV path, not here
            let n = num_rows as u32;
            let k = hidden as u32;
            let tgs = (num_rows as u64).div_ceil(4);
            enc.set_compute_pipeline_state(q4k_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), 0);
            enc.set_buffer(2, Some(buf_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(128, 1, 1));
        }
        crate::QuantFormat::Q4_0 => {
            encode_q4_matvec(enc, q4_pipeline, buf_w, buf_input, buf_scales, buf_out, num_rows, hidden);
        }
        crate::QuantFormat::Q8_0 => {
            encode_q8_matvec(enc, q8_pipeline, buf_w, buf_input, buf_scales, buf_input_scales, buf_out, num_rows, hidden);
        }
    }
}

/// Run all layers in ONE Metal command buffer with correct norms and residuals.
///
/// Multi-position aware: processes `seq_len >= 1` tokens through every stage.
/// For `seq_len == 1` this is the decode path; for `seq_len > 1` it is the
/// prefill path and populates the KV cache for subsequent decode.
///
/// Architecture coverage:
/// - Pre-norm (Llama / Mistral / Qwen): `has_post_norms = false`, `use_qk_norm = false`
/// - Post-norm + QK-norm (Gemma 3 / Gemma 4): `has_post_norms = true`, `use_qk_norm = true`
/// - Gated FFN (default) + Standard FFN (StarCoder2)
/// - SiLU + GELU-tanh activations
/// - Q4_K / Q6_K / Q4_KF / Q8_0 attention weights (Q4_K/Q6_K/Q4_KF take f32 input;
///   Q8_0 takes Q8 input via fused norm+Q8 shader)
///
/// QK-norm ordering: when `use_qk_norm` is true and `qk_norm_pipeline` is
/// supplied, QK-norm is applied **before** RoPE (matching `decode_token` and
/// the Gemma 3/4 reference implementations). `fused_attention` is then called
/// with `use_qk_norm = 0` to avoid a second normalisation.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_pipeline(
    queue: &CommandQueue,
    bufs: &BufferCache,
    q4: &Q4Pipelines,
    geglu_pipeline: &ComputePipelineState,
    geglu_gelu_tanh_pipeline: &ComputePipelineState,
    silu_pipeline: &ComputePipelineState,
    gelu_tanh_pipeline: &ComputePipelineState,
    q8_quant_pipeline: &ComputePipelineState,
    fused_attn_pipeline: Option<&ComputePipelineState>,
    q8_matvec_pipeline: &ComputePipelineState,
    q8_qkv_proj_pipeline: &ComputePipelineState,
    q4k_matvec_pipeline: &ComputePipelineState,
    q6k_matvec_pipeline: &ComputePipelineState,
    rms_norm_pipeline: &ComputePipelineState,
    residual_add_pipeline: &ComputePipelineState,
    rms_norm_q8_pipeline: &ComputePipelineState,
    residual_norm_q8_pipeline: &ComputePipelineState,
    q4k_qkv_proj_pipeline: Option<&ComputePipelineState>,
    q4kf_qkv_proj_pipeline: Option<&ComputePipelineState>,
    q4kf_proj_pipeline: Option<&ComputePipelineState>,
    rope_at_pos_pipeline: Option<&ComputePipelineState>,
    qk_norm_pipeline: Option<&ComputePipelineState>,
    scale_vector_pipeline: Option<&ComputePipelineState>,
    mut kv_cache: Option<&mut super::kv_cache::KVCache>,
    layers: &[crate::FullPipelineLayer],
    x: &[f32],
    hidden: usize,
    inter: usize,
    q_dim: usize,
    kv_dim: usize,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    _head_dim: usize,
    _rope_base: f32, // global fallback; per-layer layers[l].rope_base used in loop
    use_qk_norm: bool,
    softcap: f32,
) -> Vec<f32> {
    let num_layers = layers.len();
    let hidden_val = hidden as u32;
    let inter_val = inter as u32;
    let _n_blocks = (hidden / 32) as u32;

    // Pre-cache Q8 attention weight buffers (higher precision for Q/K dot products)
    let wq_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wq.data)).collect();
    let wq_scale_bufs: Vec<_> = layers.iter().map(|l| bufs.transient_from_f32(l.wq.scales.unwrap_or(&[]))).collect();
    let wk_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wk.data)).collect();
    let wk_scale_bufs: Vec<_> = layers.iter().map(|l| bufs.transient_from_f32(l.wk.scales.unwrap_or(&[]))).collect();
    let wv_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wv.data)).collect();
    let wv_scale_bufs: Vec<_> = layers.iter().map(|l| bufs.transient_from_f32(l.wv.scales.unwrap_or(&[]))).collect();
    let wo_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wo.data)).collect();
    let wo_scale_bufs: Vec<_> = layers.iter().map(|l| bufs.transient_from_f32(l.wo.scales.unwrap_or(&[]))).collect();
    // Q4 FFN weight buffers
    let gate_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.gate.data)).collect();
    let up_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.up.data)).collect();
    let down_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.down.data)).collect();

    // Norm weight buffers
    let input_norm_bufs: Vec<_> = layers.iter().map(|l| bufs.transient_from_f32(l.input_norm)).collect();
    let post_attn_norm_bufs: Vec<_> = layers.iter().map(|l| bufs.transient_from_f32(l.post_attn_norm)).collect();
    let pre_ffn_norm_bufs: Vec<Option<_>> = layers.iter().map(|l| {
        l.pre_ffn_norm.map(|n| bufs.transient_from_f32(n))
    }).collect();
    let post_ffn_norm_bufs: Vec<Option<_>> = layers.iter().map(|l| {
        l.post_ffn_norm.map(|n| bufs.transient_from_f32(n))
    }).collect();

    // Initial hidden state as f32 buffer
    let mut h_bufs = Vec::with_capacity(num_layers + 1);
    h_bufs.push(bufs.transient_from_f32(x));

    // Pre-allocate all intermediate buffers
    let mut norm_outs = Vec::with_capacity(num_layers);
    let mut q_outs = Vec::with_capacity(num_layers);
    let mut k_outs = Vec::with_capacity(num_layers);
    let mut v_outs = Vec::with_capacity(num_layers);
    let mut attn_outs = Vec::with_capacity(num_layers);
    let mut o_outs = Vec::with_capacity(num_layers);
    let mut h_post_attns = Vec::with_capacity(num_layers);
    let mut ffn_norm_outs = Vec::with_capacity(num_layers);
    let mut gate_outs = Vec::with_capacity(num_layers);
    let mut up_outs = Vec::with_capacity(num_layers);
    let mut act_bufs_vec = Vec::with_capacity(num_layers);
    let mut down_outs = Vec::with_capacity(num_layers);

    let mut q8_bufs = Vec::with_capacity(num_layers);
    let mut q8s_bufs = Vec::with_capacity(num_layers);
    let mut ffn_q8_bufs = Vec::with_capacity(num_layers);
    let mut ffn_q8s_bufs = Vec::with_capacity(num_layers);

    // All per-position buffers are scaled by seq_len. Single-position
    // (seq_len == 1, decode) is the existing fast path; multi-position
    // (seq_len > 1, prefill) is the fix for the previous undersized-buffer
    // crash — every downstream stage (RoPE, fused attention, KV cache copy)
    // already assumes seq_len-many rows.
    //
    // Gemma 4 uses different Q/KV dims per layer (sliding head_dim=256 vs
    // global head_dim=512), so each per-layer intermediate buffer is sized
    // from that layer's own `layer.num_q_heads * layer.head_dim`, not the
    // function-level `q_dim` / `kv_dim` (which only reflect one variant).
    // Gemma 3 / Llama / Mistral all have constant head_dim so this reduces
    // to the same allocation as before.
    //
    // The Q8 staging buffers (`q8_bufs` / `q8s_bufs`) are shared between
    // the Q8 attention-input path (hidden floats → Q8 hidden bytes) and the
    // O-projection input path (layer_q_dim floats → Q8 bytes). Sized at
    // max(hidden, max_layer_q_dim) per position so both writers fit with offsets.
    let max_layer_q_dim = layers.iter()
        .map(|l| l.num_q_heads * l.head_dim)
        .max().unwrap_or(q_dim);
    let q8_row_max = hidden.max(max_layer_q_dim);
    let q8s_row_bytes = ((q8_row_max + 31) / 32) * 4;
    for l in 0..num_layers {
        let lq = layers[l].num_q_heads * layers[l].head_dim;
        let lkv = layers[l].num_kv_heads * layers[l].head_dim;
        norm_outs.push(bufs.output((seq_len * hidden * 4) as u64));
        q_outs.push(bufs.output((seq_len * lq * 4) as u64));
        k_outs.push(bufs.output((seq_len * lkv * 4) as u64));
        v_outs.push(bufs.output((seq_len * lkv * 4) as u64));
        attn_outs.push(bufs.output((seq_len * lq * 4) as u64));
        o_outs.push(bufs.output((seq_len * hidden * 4) as u64));
        h_post_attns.push(bufs.output((seq_len * hidden * 4) as u64));
        ffn_norm_outs.push(bufs.output((seq_len * hidden * 4) as u64));
        gate_outs.push(bufs.output((seq_len * inter * 4) as u64));
        up_outs.push(bufs.output((seq_len * inter * 4) as u64));
        act_bufs_vec.push(bufs.output((seq_len * inter * 4) as u64));
        down_outs.push(bufs.output((seq_len * hidden * 4) as u64));
        h_bufs.push(bufs.output((seq_len * hidden * 4) as u64));
        q8_bufs.push(bufs.output((seq_len * q8_row_max) as u64));
        q8s_bufs.push(bufs.output((seq_len * q8s_row_bytes) as u64));
        ffn_q8_bufs.push(bufs.output((seq_len * hidden) as u64));
        ffn_q8s_bufs.push(bufs.output((seq_len * ((hidden + 31) / 32) * 4) as u64));
    }

    let mut cmd = queue.new_command_buffer();
    let dump_path = std::env::var("LARQL_METAL_DUMP_LAYERS").ok();
    // Dump h_embed (input to layer 0) before any compute — lets us
    // verify CPU and Metal start from the same point.
    if let Some(ref dir) = dump_path {
        let ptr = h_bufs[0].contents() as *const f32;
        if !ptr.is_null() {
            let s = unsafe { std::slice::from_raw_parts(ptr, seq_len * hidden) };
            let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
            let path = format!("{dir}/metal_h_embed.f32");
            let _ = std::fs::write(&path, &bytes);
        }
    }

    for l in 0..num_layers {
        let eps = layers[l].eps;
        let layer_rope_base = layers[l].rope_base;
        let layer_head_dim = layers[l].head_dim;
        let layer_num_q_heads = layers[l].num_q_heads;
        let layer_num_kv_heads = layers[l].num_kv_heads;
        let layer_q_dim = layer_num_q_heads * layer_head_dim;
        let layer_kv_dim = layer_num_kv_heads * layer_head_dim;
        let layer_attn_scale = layers[l].attn_scale;
        let norm_offset = layers[l].norm_offset;
        let has_post_norms = layers[l].has_post_norms;

        // ── 1+3. Input norm + Q/K/V projections (format-aware) ──
        let attn_format = layers[l].wq.format;
        let uses_f32_input = attn_format == crate::QuantFormat::Q4_K || attn_format == crate::QuantFormat::Q6_K || attn_format == crate::QuantFormat::Q4_KF;

        // Per-position offsets (bytes). `layer_q_dim` / `layer_kv_dim` are the
        // **this layer's** actual dimensions — Gemma 4 alternates between
        // sliding (head_dim=256) and global (head_dim=512) layers so these
        // differ per layer. Offsets into the per-layer allocated buffers use
        // the per-layer dims; the function-level `q_dim` / `kv_dim` are only
        // used as fallback stride for the caller's Q8 staging bucket.
        let h_off = |p: usize| (p * hidden * 4) as u64;
        let q_off = |p: usize| (p * layer_q_dim * 4) as u64;
        let kv_off = |p: usize| (p * layer_kv_dim * 4) as u64;
        let inter_off = |p: usize| (p * inter * 4) as u64;
        let q8_off = |p: usize| (p * q8_row_max) as u64;
        let q8s_off = |p: usize| (p * q8s_row_bytes) as u64;
        let ffn_q8_off = |p: usize| (p * hidden) as u64;
        let ffn_q8s_off = |p: usize| (p * ((hidden + 31) / 32) * 4) as u64;

        if uses_f32_input {
            // Q4_K / Q6_K / Q4_KF path: per-position RMS norm → per-position QKV matvec.
            // The underlying shaders (rms_norm + q4k_qkv_proj / q4k_matvec) are all
            // single-vector dispatches, so we loop over positions with buffer offsets.
            for pos in 0..seq_len {
                // Step 1: input RMS-norm for position `pos`.
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(rms_norm_pipeline);
                    enc.set_buffer(0, Some(&h_bufs[l]), h_off(pos));
                    enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                    enc.set_buffer(2, Some(&norm_outs[l]), h_off(pos));
                    enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                    enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(256.min(hidden as u64), 1, 1),
                    );
                    enc.end_encoding();
                }

                // Step 2: Q+K+V projection for position `pos`.
                //
                // Shader selection:
                // - Q4_K / Q4_KF with all three same format → fused QKV shader
                //   (`q4kf_qkv_proj` — 144-byte GGUF block layout, matches
                //   everything our extractor / llama.cpp writes today).
                // - Q6_K anywhere (Gemma 4 Ollama-style V) or mixed formats →
                //   three separate per-projection matvecs.
                //
                // NOTE: the legacy `q4k_qkv_proj` shader uses a 148-byte block
                // layout that no vindex actually contains, so we don't dispatch
                // through it here. The 144-byte `q4kf_qkv_proj` shader is the
                // correct path for modern Q4_K data.
                let all_same_format = layers[l].wq.format == layers[l].wk.format
                    && layers[l].wk.format == layers[l].wv.format;
                let fuseable = all_same_format
                    && matches!(layers[l].wq.format,
                        crate::QuantFormat::Q4_K | crate::QuantFormat::Q4_KF);
                // Prefer q4kf (144-byte GGUF) over q4k (148-byte legacy).
                let fused_pipe = q4kf_qkv_proj_pipeline.or(q4k_qkv_proj_pipeline);
                if let Some(fused_pipeline) = fused_pipe.filter(|_| fuseable) {
                    use crate::metal::shaders::q4kf_qkv_proj as q4kf_qkv;
                    let total_rows = (layer_q_dim + layer_kv_dim + layer_kv_dim) as u32;
                    let q_rows_val = layer_q_dim as u32;
                    let k_rows_val = layer_kv_dim as u32;
                    let v_rows_val = layer_kv_dim as u32;
                    let k_val = hidden as u32;
                    let num_tgs = (total_rows as u64).div_ceil(q4kf_qkv::ROWS_PER_TG);
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(fused_pipeline);
                    enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                    enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                    enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                    enc.set_buffer(3, Some(&norm_outs[l]), h_off(pos));
                    enc.set_buffer(4, Some(&q_outs[l]), q_off(pos));
                    enc.set_buffer(5, Some(&k_outs[l]), kv_off(pos));
                    enc.set_buffer(6, Some(&v_outs[l]), kv_off(pos));
                    enc.set_bytes(7, 4, &q_rows_val as *const u32 as *const c_void);
                    enc.set_bytes(8, 4, &k_rows_val as *const u32 as *const c_void);
                    enc.set_bytes(9, 4, &v_rows_val as *const u32 as *const c_void);
                    enc.set_bytes(10, 4, &k_val as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(q4kf_qkv::THREADS_PER_TG, 1, 1),
                    );
                    enc.end_encoding();
                } else {
                    // Per-projection matvec — handles mixed Q4_K/Q6_K (Ollama
                    // strategy used by Gemma 4 Q4K vindexes) and isolates
                    // formats so each uses its own dedicated shader.
                    // Mirrors `metal::decode::decode_token`'s `encode_single_proj`
                    // with buffer offsets for multi-position.
                    let k_val = hidden as u32;
                    for (fmt, w_buf, out_buf, out_off_v, rows) in [
                        (layers[l].wq.format, &wq_bufs[l], &q_outs[l], q_off(pos), layer_q_dim),
                        (layers[l].wk.format, &wk_bufs[l], &k_outs[l], kv_off(pos), layer_kv_dim),
                        (layers[l].wv.format, &wv_bufs[l], &v_outs[l], kv_off(pos), layer_kv_dim),
                    ] {
                        let enc = cmd.new_compute_command_encoder();
                        match fmt {
                            crate::QuantFormat::Q6_K => {
                                use crate::metal::shaders::q6k_matvec as q6k;
                                let n = rows as u32;
                                let num_tgs = (rows as u64).div_ceil(q6k::ROWS_PER_TG);
                                enc.set_compute_pipeline_state(q6k_matvec_pipeline);
                                enc.set_buffer(0, Some(w_buf), 0);
                                enc.set_buffer(1, Some(&norm_outs[l]), h_off(pos));
                                enc.set_buffer(2, Some(out_buf), out_off_v);
                                enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
                                enc.set_bytes(4, 4, &k_val as *const u32 as *const c_void);
                                enc.dispatch_thread_groups(
                                    MTLSize::new(num_tgs, 1, 1),
                                    MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
                                );
                            }
                            _ => {
                                // Q4_K / Q4_KF / Q4_0 single-projection matvec.
                                // Prefer the 144-byte GGUF-layout `q4kf_proj`
                                // shader when available — that's the format every
                                // freshly-extracted vindex uses. The legacy
                                // `q4k_matvec` shader expects a 148-byte layout
                                // that no current extractor emits, so fall back
                                // to it only if q4kf_proj is missing.
                                if let Some(q4kf_proj_pipe) = q4kf_proj_pipeline {
                                    use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                                    let n = rows as u32;
                                    let num_tgs = (rows as u64).div_ceil(q4kf::ROWS_PER_TG);
                                    enc.set_compute_pipeline_state(q4kf_proj_pipe);
                                    enc.set_buffer(0, Some(w_buf), 0);
                                    enc.set_buffer(1, Some(&norm_outs[l]), h_off(pos));
                                    enc.set_buffer(2, Some(out_buf), out_off_v);
                                    enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
                                    enc.set_bytes(4, 4, &k_val as *const u32 as *const c_void);
                                    enc.dispatch_thread_groups(
                                        MTLSize::new(num_tgs, 1, 1),
                                        MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
                                    );
                                } else {
                                    use crate::metal::shaders::q4k_matvec as q4k;
                                    let n = rows as u32;
                                    let num_tgs = (rows as u64).div_ceil(q4k::ROWS_PER_TG);
                                    enc.set_compute_pipeline_state(q4k_matvec_pipeline);
                                    enc.set_buffer(0, Some(w_buf), 0);
                                    enc.set_buffer(1, Some(&norm_outs[l]), h_off(pos));
                                    enc.set_buffer(2, Some(out_buf), out_off_v);
                                    enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
                                    enc.set_bytes(4, 4, &k_val as *const u32 as *const c_void);
                                    enc.dispatch_thread_groups(
                                        MTLSize::new(num_tgs, 1, 1),
                                        MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                                    );
                                }
                            }
                        }
                        enc.end_encoding();
                    }
                }
            }
        } else {
            // Q8_0 attention path: per-position fused (rms_norm + Q8 quantize),
            // then per-position Q8 QKV projection. The Q8 staging buffers hold
            // one `hidden` row per position via `q8_off` / `q8s_off`.
            for pos in 0..seq_len {
                // Fused rms_norm → Q8 quantize for position `pos`.
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(rms_norm_q8_pipeline);
                enc.set_buffer(0, Some(&h_bufs[l]), h_off(pos));
                enc.set_buffer(1, Some(&input_norm_bufs[l]), 0);
                enc.set_buffer(2, Some(&q8_bufs[l]), q8_off(pos));
                enc.set_buffer(3, Some(&q8s_bufs[l]), q8s_off(pos));
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                enc.end_encoding();

                let q_rows_val = layer_q_dim as u32;
                let k_rows_val = layer_kv_dim as u32;
                let v_rows_val = layer_kv_dim as u32;
                let k_val = hidden as u32;
                let total_rows = layer_q_dim + layer_kv_dim + layer_kv_dim;

                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(q8_qkv_proj_pipeline);
                enc.set_buffer(0, Some(&wq_bufs[l]), 0);
                enc.set_buffer(1, Some(&wk_bufs[l]), 0);
                enc.set_buffer(2, Some(&wv_bufs[l]), 0);
                enc.set_buffer(3, Some(&q8_bufs[l]), q8_off(pos));
                enc.set_buffer(4, Some(&wq_scale_bufs[l]), 0);
                enc.set_buffer(5, Some(&wk_scale_bufs[l]), 0);
                enc.set_buffer(6, Some(&wv_scale_bufs[l]), 0);
                enc.set_buffer(7, Some(&q8s_bufs[l]), q8s_off(pos));
                enc.set_buffer(8, Some(&q_outs[l]), q_off(pos));
                enc.set_buffer(9, Some(&k_outs[l]), kv_off(pos));
                enc.set_buffer(10, Some(&v_outs[l]), kv_off(pos));
                enc.set_bytes(11, 4, &q_rows_val as *const u32 as *const c_void);
                enc.set_bytes(12, 4, &k_rows_val as *const u32 as *const c_void);
                enc.set_bytes(13, 4, &v_rows_val as *const u32 as *const c_void);
                enc.set_bytes(14, 4, &k_val as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(total_rows as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
            }
        }

        // ── 3 (pre). Optional parameter-free V-norm (Gemma 4). ──
        // Normalises V per-head before it feeds into attention. Matches
        // `decode_token`'s Step 3. Only applied when the architecture
        // requests it (`layer.has_v_norm`).
        if layers[l].has_v_norm {
            if let Some(qk_norm_pipe) = qk_norm_pipeline {
                // Re-use qk_norm shader: it already RMS-norms per head with
                // a weight vector. V-norm is parameter-free (weight = 1,
                // offset = 0), so we stage an all-ones weight buffer once.
                let ones: Vec<f32> = vec![1.0; layer_head_dim];
                let ones_buf = bufs.transient_from_f32(&ones);
                let hd_val = layer_head_dim as u32;
                let nkv_val = layer_num_kv_heads as u32;
                let zero_off: f32 = 0.0;
                let mut tg_w: u64 = 1;
                while (tg_w as usize) < layer_head_dim && tg_w < 512 { tg_w <<= 1; }

                let enc = cmd.new_compute_command_encoder();
                for pos in 0..seq_len {
                    let v_buf_off = (pos * layer_num_kv_heads * layer_head_dim * 4) as u64;
                    enc.set_compute_pipeline_state(qk_norm_pipe);
                    enc.set_buffer(0, Some(&v_outs[l]), v_buf_off);
                    enc.set_buffer(1, Some(&v_outs[l]), v_buf_off);
                    enc.set_buffer(2, Some(&ones_buf), 0);
                    enc.set_bytes(3, 4, &hd_val as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &nkv_val as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
                    enc.set_bytes(6, 4, &zero_off as *const f32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(layer_num_kv_heads as u64, 1, 1),
                        MTLSize::new(tg_w, 1, 1),
                    );
                }
                enc.end_encoding();
            }
        }

        // Stage dump: Q just after QKV projection, before QK-norm.
        if dump_path.is_some() && l == 0 {
            cmd.commit();
            cmd.wait_until_completed();
            let ptr = q_outs[l].contents() as *const f32;
            if !ptr.is_null() {
                let n = seq_len * layer_q_dim;
                let s = unsafe { std::slice::from_raw_parts(ptr, n) };
                let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(
                    format!("{}/metal_L0_q_out_raw.f32", dump_path.as_ref().unwrap()),
                    &bytes,
                );
            }
            cmd = queue.new_command_buffer();
        }

        // ── 3a. QK-norm on Q and K (pre-RoPE). Gemma 3 / Gemma 4. ──
        //
        // Applied BEFORE RoPE to match the reference implementation and
        // `decode_token` in `metal/decode.rs`. When we do this here, we tell
        // `fused_attention` below to skip its internal QK-norm
        // (`use_qk_norm = 0`) so we don't double-normalise.
        let applied_prerope_qk_norm = if use_qk_norm {
            if let (Some(qk_norm_pipe), Some(q_w_slice), Some(k_w_slice)) =
                (qk_norm_pipeline, layers[l].q_norm_weight, layers[l].k_norm_weight)
            {
                let q_w_buf = bufs.transient_from_f32(q_w_slice);
                let k_w_buf = bufs.transient_from_f32(k_w_slice);
                let hd_val = layer_head_dim as u32;
                let nq_val = layer_num_q_heads as u32;
                let nkv_val = layer_num_kv_heads as u32;
                let qk_off = layers[l].qk_norm_offset;
                // Threadgroup width = next power-of-two up to 512 covering head_dim.
                let mut tg_w: u64 = 1;
                while (tg_w as usize) < layer_head_dim && tg_w < 512 { tg_w <<= 1; }

                let enc = cmd.new_compute_command_encoder();
                for pos in 0..seq_len {
                    // Q heads: grid (num_q_heads, 1, 1), buffer base at
                    // position `pos` offsets by `pos * num_q_heads * head_dim` f32s.
                    let q_buf_off = (pos * layer_num_q_heads * layer_head_dim * 4) as u64;
                    enc.set_compute_pipeline_state(qk_norm_pipe);
                    enc.set_buffer(0, Some(&q_outs[l]), q_buf_off);
                    enc.set_buffer(1, Some(&q_outs[l]), q_buf_off);
                    enc.set_buffer(2, Some(&q_w_buf), 0);
                    enc.set_bytes(3, 4, &hd_val as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &nq_val as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
                    enc.set_bytes(6, 4, &qk_off as *const f32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(layer_num_q_heads as u64, 1, 1),
                        MTLSize::new(tg_w, 1, 1),
                    );

                    // K heads
                    let k_buf_off = (pos * layer_num_kv_heads * layer_head_dim * 4) as u64;
                    enc.set_buffer(0, Some(&k_outs[l]), k_buf_off);
                    enc.set_buffer(1, Some(&k_outs[l]), k_buf_off);
                    enc.set_buffer(2, Some(&k_w_buf), 0);
                    enc.set_bytes(4, 4, &nkv_val as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(layer_num_kv_heads as u64, 1, 1),
                        MTLSize::new(tg_w, 1, 1),
                    );
                }
                enc.end_encoding();
                true
            } else {
                // use_qk_norm requested but pipeline or weights missing —
                // fall back to fused_attention's internal QK-norm (legacy path).
                false
            }
        } else {
            false
        };

        // Stage dump: Q after QK-norm, before RoPE.
        if dump_path.is_some() && l == 0 {
            cmd.commit();
            cmd.wait_until_completed();
            let ptr = q_outs[l].contents() as *const f32;
            if !ptr.is_null() {
                let n = seq_len * layer_q_dim;
                let s = unsafe { std::slice::from_raw_parts(ptr, n) };
                let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(
                    format!("{}/metal_L0_q_out_after_qk_norm.f32", dump_path.as_ref().unwrap()),
                    &bytes,
                );
            }
            cmd = queue.new_command_buffer();
        }

        // ── 3b. Apply RoPE separately when populating KV cache ──
        // When kv_cache is provided, apply RoPE to Q and K via rope_at_pos per head
        // per position, write K/V to cache, then run fused_attention with skip_rope=1.
        // When kv_cache is None, let fused_attention handle RoPE internally (skip_rope=0).
        let use_separate_rope = kv_cache.is_some() && rope_at_pos_pipeline.is_some();

        if use_separate_rope {
            let rope_pipeline = rope_at_pos_pipeline.unwrap();
            let hd = layer_head_dim as u32;
            // Gemma 4 global layers use partial rotation: `rotary_dim < head_dim`.
            // Pass `layer.rotary_dim` (0 means "full head_dim" per the shader).
            let rdim_val = layers[l].rotary_dim as u32;
            let rdim_effective = if layers[l].rotary_dim == 0 {
                layer_head_dim
            } else {
                layers[l].rotary_dim
            };
            let hdim = (rdim_effective / 2) as u64;

            let enc = cmd.new_compute_command_encoder();
            for pos in 0..seq_len {
                let pos_val = pos as u32;
                for qh in 0..layer_num_q_heads {
                    let offset = (pos * layer_num_q_heads * layer_head_dim + qh * layer_head_dim) as u64 * 4;
                    enc.set_compute_pipeline_state(rope_pipeline);
                    enc.set_buffer(0, Some(&q_outs[l]), offset);
                    enc.set_bytes(1, 4, &hd as *const u32 as *const c_void);
                    enc.set_bytes(2, 4, &layer_rope_base as *const f32 as *const c_void);
                    enc.set_bytes(3, 4, &pos_val as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &rdim_val as *const u32 as *const c_void);
                    enc.dispatch_threads(MTLSize::new(hdim, 1, 1), MTLSize::new(hdim.min(256), 1, 1));
                }
                for kvh in 0..layer_num_kv_heads {
                    let offset = (pos * layer_num_kv_heads * layer_head_dim + kvh * layer_head_dim) as u64 * 4;
                    enc.set_compute_pipeline_state(rope_pipeline);
                    enc.set_buffer(0, Some(&k_outs[l]), offset);
                    enc.set_bytes(1, 4, &hd as *const u32 as *const c_void);
                    enc.set_bytes(2, 4, &layer_rope_base as *const f32 as *const c_void);
                    enc.set_bytes(3, 4, &pos_val as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &rdim_val as *const u32 as *const c_void);
                    enc.dispatch_threads(MTLSize::new(hdim, 1, 1), MTLSize::new(hdim.min(256), 1, 1));
                }
            }
            enc.end_encoding();
        }

        // ── 4. Fused attention (RoPE + GQA + softcap). Already multi-position. ──
        // Dispatches one threadgroup per (head, position); the shader caps at
        // `seq_len` positions and applies a causal mask per row.
        if let Some(fused_pipeline) = fused_attn_pipeline {
            let seq_val = seq_len as u32;
            let hd_val = layer_head_dim as u32;
            let nq_val = layer_num_q_heads as u32;
            let nkv_val = layer_num_kv_heads as u32;
            let scale_val = layer_attn_scale;
            // If we pre-applied QK-norm above, tell fused_attention to skip it.
            // Otherwise fall back to the shader's internal normalisation (legacy).
            let qknorm_val = if use_qk_norm && !applied_prerope_qk_norm { 1u32 } else { 0u32 };

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(fused_pipeline);
            enc.set_buffer(0, Some(&q_outs[l]), 0);
            enc.set_buffer(1, Some(&k_outs[l]), 0);
            enc.set_buffer(2, Some(&v_outs[l]), 0);
            enc.set_buffer(3, Some(&attn_outs[l]), 0);
            enc.set_bytes(4, 4, &seq_val as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &hd_val as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &nq_val as *const u32 as *const c_void);
            enc.set_bytes(7, 4, &nkv_val as *const u32 as *const c_void);
            enc.set_bytes(8, 4, &scale_val as *const f32 as *const c_void);
            enc.set_bytes(9, 4, &layer_rope_base as *const f32 as *const c_void);
            enc.set_bytes(10, 4, &qknorm_val as *const u32 as *const c_void);
            enc.set_bytes(11, 4, &softcap as *const f32 as *const c_void);
            let skip_rope_val = if use_separate_rope { 1u32 } else { 0u32 };
            enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const c_void);
            // Pass per-layer rotary_dim (0 = full head_dim). Gemma 4 global
            // layers use partial rotation (rotary_dim = head_dim / 4) — this
            // must match what the caller pre-applied in separate RoPE.
            let rotary_dim_val = layers[l].rotary_dim as u32;
            enc.set_bytes(13, 4, &rotary_dim_val as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(layer_num_q_heads as u64, seq_len as u64, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }

        // ── 5. O projection. Per position. ──
        //
        // Format-aware: Q4_K / Q4_KF / Q6_K weights go through the f32-input
        // `q4kf_proj` / `q6k_matvec` shaders directly (no Q8 quantize step);
        // Q8_0 / Q4_0 weights go through the Q8-input path (quantize attn_out
        // first, then Q8/Q4 matvec). Matches decode.rs layer O projection.
        let o_format = layers[l].wo.format;
        let o_is_q4k = matches!(o_format,
            crate::QuantFormat::Q4_K | crate::QuantFormat::Q4_KF | crate::QuantFormat::Q6_K);
        for pos in 0..seq_len {
            if o_is_q4k {
                let o_rows = hidden as u32;
                let o_k = layer_q_dim as u32;
                let enc = cmd.new_compute_command_encoder();
                match o_format {
                    crate::QuantFormat::Q6_K => {
                        use crate::metal::shaders::q6k_matvec as q6k;
                        let num_tgs = (hidden as u64).div_ceil(q6k::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(q6k_matvec_pipeline);
                        enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                        enc.set_buffer(1, Some(&attn_outs[l]), q_off(pos));
                        enc.set_buffer(2, Some(&o_outs[l]), h_off(pos));
                        enc.set_bytes(3, 4, &o_rows as *const u32 as *const c_void);
                        enc.set_bytes(4, 4, &o_k as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(num_tgs, 1, 1),
                            MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
                        );
                    }
                    _ => {
                        // Q4_K / Q4_KF → q4kf_proj (144-byte GGUF layout).
                        let q4kf_proj_pipe = q4kf_proj_pipeline.expect(
                            "q4kf_proj pipeline required for Q4_K O projection");
                        use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                        let num_tgs = (hidden as u64).div_ceil(q4kf::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(q4kf_proj_pipe);
                        enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                        enc.set_buffer(1, Some(&attn_outs[l]), q_off(pos));
                        enc.set_buffer(2, Some(&o_outs[l]), h_off(pos));
                        enc.set_bytes(3, 4, &o_rows as *const u32 as *const c_void);
                        enc.set_bytes(4, 4, &o_k as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(num_tgs, 1, 1),
                            MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
                        );
                    }
                }
                enc.end_encoding();
            } else {
                // Q8_0 / Q4_0 path: quantize attn_out → Q8, then Q8 matvec.
                let attn_dim_val = layer_q_dim as u32;
                let attn_blocks = (layer_q_dim as u64).div_ceil(32);
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(q8_quant_pipeline);
                    enc.set_buffer(0, Some(&attn_outs[l]), q_off(pos));
                    enc.set_buffer(1, Some(&q8_bufs[l]), q8_off(pos));
                    enc.set_buffer(2, Some(&q8s_bufs[l]), q8s_off(pos));
                    enc.set_bytes(3, 4, &attn_dim_val as *const u32 as *const c_void);
                    enc.dispatch_threads(
                        MTLSize::new(attn_blocks, 1, 1),
                        MTLSize::new(256.min(attn_blocks), 1, 1),
                    );
                    enc.end_encoding();
                }
                {
                    let o_rows = hidden as u32;
                    let o_k = layer_q_dim as u32;
                    let o_tgs = (hidden as u64).div_ceil(8);
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(q8_matvec_pipeline);
                    enc.set_buffer(0, Some(&wo_bufs[l]), 0);
                    enc.set_buffer(1, Some(&q8_bufs[l]), q8_off(pos));
                    enc.set_buffer(2, Some(&wo_scale_bufs[l]), 0);
                    enc.set_buffer(3, Some(&q8s_bufs[l]), q8s_off(pos));
                    enc.set_buffer(4, Some(&o_outs[l]), h_off(pos));
                    enc.set_bytes(5, 4, &o_rows as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &o_k as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(o_tgs, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.end_encoding();
                }
            }
        }

        // ── 6. Post-attention residual + pre-FFN norm. ──
        //
        // Two output representations are needed here:
        //   (a) ffn_norm_outs[l]  — f32 per position; consumed by Q4_K / Q4_KF /
        //                            Q6_K FFN which expect f32 input.
        //   (b) ffn_q8_bufs[l] + ffn_q8s_bufs[l] — Q8 + scales per position;
        //       consumed only by Q4_0 / Q8_0 FFN.
        // We always produce (a) via residual_add + rms_norm. We only produce
        // (b) when the FFN actually needs it, to avoid the extra dispatch.
        //
        // `h_post_attns[l]` holds the post-residual f32 hidden state for the
        // final residual add at the end of this layer (step 10).
        let ffn_format = layers[l].gate.format;
        let ffn_needs_q8 = matches!(ffn_format,
            crate::QuantFormat::Q4_0 | crate::QuantFormat::Q8_0);


        for pos in 0..seq_len {
            // First build h_post_attns[l][pos] = h + O (or h + norm(O) for post-norm).
            if has_post_norms {
                // Post-norm: norm(O) first, then residual add.
                let normed = bufs.output((hidden * 4) as u64);
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(rms_norm_pipeline);
                    enc.set_buffer(0, Some(&o_outs[l]), h_off(pos));
                    enc.set_buffer(1, Some(&post_attn_norm_bufs[l]), 0);
                    enc.set_buffer(2, Some(&normed), 0);
                    enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                    enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                    enc.end_encoding();
                }
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(residual_add_pipeline);
                enc.set_buffer(0, Some(&h_bufs[l]), h_off(pos));
                enc.set_buffer(1, Some(&normed), 0);
                enc.set_buffer(2, Some(&h_post_attns[l]), h_off(pos));
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                enc.end_encoding();
            } else {
                // Pre-norm: residual add first (h + O), then norm happens below.
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(residual_add_pipeline);
                enc.set_buffer(0, Some(&h_bufs[l]), h_off(pos));
                enc.set_buffer(1, Some(&o_outs[l]), h_off(pos));
                enc.set_buffer(2, Some(&h_post_attns[l]), h_off(pos));
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                enc.end_encoding();
            }

            // Pre-FFN rms_norm on h_post_attns → ffn_norm_outs (f32).
            let pre_ffn_weight = if has_post_norms {
                pre_ffn_norm_bufs[l].as_ref().unwrap_or(&post_attn_norm_bufs[l])
            } else {
                &post_attn_norm_bufs[l]
            };
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(rms_norm_pipeline);
                enc.set_buffer(0, Some(&h_post_attns[l]), h_off(pos));
                enc.set_buffer(1, Some(pre_ffn_weight), 0);
                enc.set_buffer(2, Some(&ffn_norm_outs[l]), h_off(pos));
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                enc.end_encoding();
            }

            // If FFN is Q8-input, additionally Q8-quantise ffn_norm_outs.
            if ffn_needs_q8 {
                let blocks = (hidden as u64).div_ceil(32);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(q8_quant_pipeline);
                enc.set_buffer(0, Some(&ffn_norm_outs[l]), h_off(pos));
                enc.set_buffer(1, Some(&ffn_q8_bufs[l]), ffn_q8_off(pos));
                enc.set_buffer(2, Some(&ffn_q8s_bufs[l]), ffn_q8s_off(pos));
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.dispatch_threads(
                    MTLSize::new(blocks, 1, 1),
                    MTLSize::new(256.min(blocks), 1, 1),
                );
                enc.end_encoding();
            }
        }

        // ── 7-9. FFN: gate+up → GEGLU → down. Format-aware per position. ──
        //
        // Gate/up format drives the input representation:
        //   Q4_K / Q4_KF / Q6_K → f32 input (ffn_norm_outs), direct matvec
        //   Q4_0 / Q8_0         → Q8 input (ffn_q8_bufs + ffn_q8s_bufs)
        //
        // The down projection matches Gemma 4's Ollama strategy (Q4_K gate/up
        // with Q6_K down): each format dispatched to its dedicated shader.
        let gate_format = layers[l].gate.format;
        let up_format = layers[l].up.format;
        let down_format = layers[l].down.format;

        for pos in 0..seq_len {
            // Gate projection (only if gated)
            if layers[l].ffn_type != crate::FfnType::Standard {
                let enc = cmd.new_compute_command_encoder();
                dispatch_ffn_matvec(
                    enc, gate_format,
                    &gate_bufs[l], &ffn_norm_outs[l], h_off(pos),
                    &ffn_q8_bufs[l], ffn_q8_off(pos),
                    &ffn_q8s_bufs[l], ffn_q8s_off(pos),
                    &gate_outs[l], inter_off(pos),
                    q4k_matvec_pipeline, q6k_matvec_pipeline,
                    q4kf_proj_pipeline, &q4.matvec,
                    inter, hidden,
                );
                enc.end_encoding();
            }
            // Up projection
            {
                let enc = cmd.new_compute_command_encoder();
                dispatch_ffn_matvec(
                    enc, up_format,
                    &up_bufs[l], &ffn_norm_outs[l], h_off(pos),
                    &ffn_q8_bufs[l], ffn_q8_off(pos),
                    &ffn_q8s_bufs[l], ffn_q8s_off(pos),
                    &up_outs[l], inter_off(pos),
                    q4k_matvec_pipeline, q6k_matvec_pipeline,
                    q4kf_proj_pipeline, &q4.matvec,
                    inter, hidden,
                );
                enc.end_encoding();
            }
        }

        // Activation — multi-position pointwise dispatch over `seq_len * inter`.
        {
            let total_inter = (seq_len * inter) as u64;
            let total_inter_val = (seq_len * inter) as u32;
            if layers[l].ffn_type == crate::FfnType::Standard {
                let activation_pipe = match layers[l].activation {
                    crate::Activation::GeluTanh => gelu_tanh_pipeline,
                    _ => silu_pipeline,
                };
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(activation_pipe);
                enc.set_buffer(0, Some(&up_outs[l]), 0);
                enc.set_buffer(1, Some(&act_bufs_vec[l]), 0);
                enc.set_bytes(2, 4, &total_inter_val as *const u32 as *const c_void);
                enc.dispatch_threads(MTLSize::new(total_inter, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
            } else {
                let geglu_pipe = match layers[l].activation {
                    crate::Activation::GeluTanh => geglu_gelu_tanh_pipeline,
                    _ => geglu_pipeline,
                };
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(geglu_pipe);
                enc.set_buffer(0, Some(&gate_outs[l]), 0);
                enc.set_buffer(1, Some(&up_outs[l]), 0);
                enc.set_buffer(2, Some(&act_bufs_vec[l]), 0);
                enc.set_bytes(3, 4, &total_inter_val as *const u32 as *const c_void);
                enc.dispatch_threads(MTLSize::new(total_inter, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }
        }

        // Down projection — per position, format-aware. Reads act_bufs_vec
        // (f32) of size `inter`, writes down_outs (f32) of size `hidden`.
        // For Q4_0 / Q8_0 down we'd need to Q8-quantise act_bufs_vec first;
        // current production vindexes use Q4_K / Q6_K so that branch is a
        // fallback that goes through Q4_0 f32-matvec (non-fast but correct).
        for pos in 0..seq_len {
            let enc = cmd.new_compute_command_encoder();
            match down_format {
                crate::QuantFormat::Q6_K => {
                    use crate::metal::shaders::q6k_matvec as q6k;
                    let n = hidden as u32;
                    let k = inter as u32;
                    let num_tgs = (hidden as u64).div_ceil(q6k::ROWS_PER_TG);
                    enc.set_compute_pipeline_state(q6k_matvec_pipeline);
                    enc.set_buffer(0, Some(&down_bufs[l]), 0);
                    enc.set_buffer(1, Some(&act_bufs_vec[l]), inter_off(pos));
                    enc.set_buffer(2, Some(&down_outs[l]), h_off(pos));
                    enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_tgs, 1, 1),
                        MTLSize::new(q6k::THREADS_PER_TG, 1, 1),
                    );
                }
                crate::QuantFormat::Q4_K | crate::QuantFormat::Q4_KF => {
                    if let Some(q4kf_proj_pipe) = q4kf_proj_pipeline {
                        use crate::metal::shaders::q4kf_qkv_proj as q4kf;
                        let n = hidden as u32;
                        let k = inter as u32;
                        let num_tgs = (hidden as u64).div_ceil(q4kf::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(q4kf_proj_pipe);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_bufs_vec[l]), inter_off(pos));
                        enc.set_buffer(2, Some(&down_outs[l]), h_off(pos));
                        enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
                        enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(num_tgs, 1, 1),
                            MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
                        );
                    } else {
                        use crate::metal::shaders::q4k_matvec as q4k;
                        let n = hidden as u32;
                        let k = inter as u32;
                        let num_tgs = (hidden as u64).div_ceil(q4k::ROWS_PER_TG);
                        enc.set_compute_pipeline_state(q4k_matvec_pipeline);
                        enc.set_buffer(0, Some(&down_bufs[l]), 0);
                        enc.set_buffer(1, Some(&act_bufs_vec[l]), inter_off(pos));
                        enc.set_buffer(2, Some(&down_outs[l]), h_off(pos));
                        enc.set_bytes(3, 4, &n as *const u32 as *const c_void);
                        enc.set_bytes(4, 4, &k as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(num_tgs, 1, 1),
                            MTLSize::new(q4k::THREADS_PER_TG, 1, 1),
                        );
                    }
                }
                _ => {
                    // Q4_0 / Q8_0 fallback via f32 input matvec.
                    enc.set_compute_pipeline_state(&q4.f32_matvec);
                    enc.set_buffer(0, Some(&down_bufs[l]), 0);
                    enc.set_buffer(1, Some(&act_bufs_vec[l]), inter_off(pos));
                    enc.set_buffer(2, Some(&down_outs[l]), h_off(pos));
                    enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &inter_val as *const u32 as *const c_void);
                    enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
                }
            }
            enc.end_encoding();
        }

        // ── 10. Post-FFN: optional norm, then residual add → h for next layer. ──
        for pos in 0..seq_len {
            if has_post_norms {
                if let Some(ref post_ffn_buf) = post_ffn_norm_bufs[l] {
                    let normed = bufs.output((hidden * 4) as u64);
                    {
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(rms_norm_pipeline);
                        enc.set_buffer(0, Some(&down_outs[l]), h_off(pos));
                        enc.set_buffer(1, Some(post_ffn_buf), 0);
                        enc.set_buffer(2, Some(&normed), 0);
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                        enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                        enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
                        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                        enc.end_encoding();
                    }
                    {
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(residual_add_pipeline);
                        enc.set_buffer(0, Some(&h_post_attns[l]), h_off(pos));
                        enc.set_buffer(1, Some(&normed), 0);
                        enc.set_buffer(2, Some(&h_bufs[l + 1]), h_off(pos));
                        enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                        enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
                        enc.end_encoding();
                    }
                    continue;
                }
            }
            // Pre-norm or post-norm-without-post_ffn_norm: plain residual.
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(residual_add_pipeline);
            enc.set_buffer(0, Some(&h_post_attns[l]), h_off(pos));
            enc.set_buffer(1, Some(&down_outs[l]), h_off(pos));
            enc.set_buffer(2, Some(&h_bufs[l + 1]), h_off(pos));
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
            enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256.min(hidden as u64), 1, 1));
            enc.end_encoding();
        }

        // ── 11. Per-layer residual scalar (Gemma 4). ──
        //
        // Gemma 4 multiplies the layer's residual output by a learned scalar
        // (`layer_scalar`, typically ~0.05) to stabilise the residual stream.
        // Without this the residual magnitude explodes across layers since
        // Gemma 4's post_attn_norm weights can reach ~100. Matches
        // `decode_token`'s Step 8 and `apply_layer_scalar` on CPU.
        if let Some(scale_pipe) = scale_vector_pipeline.filter(|_| layers[l].layer_scalar != 0.0) {
            let scalar = layers[l].layer_scalar;
            let enc = cmd.new_compute_command_encoder();
            for pos in 0..seq_len {
                enc.set_compute_pipeline_state(scale_pipe);
                enc.set_buffer(0, Some(&h_bufs[l + 1]), h_off(pos));
                enc.set_buffer(1, Some(&h_bufs[l + 1]), h_off(pos));
                enc.set_bytes(2, 4, &hidden_val as *const u32 as *const c_void);
                enc.set_bytes(3, 4, &scalar as *const f32 as *const c_void);
                // `scale_vector` uses `thread_position_in_grid` — need one
                // thread per element, not a single 256-thread threadgroup.
                enc.dispatch_threads(
                    MTLSize::new(hidden as u64, 1, 1),
                    MTLSize::new(256.min(hidden as u64), 1, 1),
                );
            }
            enc.end_encoding();
        }

        // Optional per-layer residual dump (LARQL_METAL_DUMP_LAYERS=<dir>).
        // Commits the buffer up to this layer, reads h_bufs[l+1], writes to
        // `{dir}/metal_layer_{l}.f32` as raw little-endian floats. Enables
        // diffing against the CPU reference layer-by-layer to bisect the
        // first layer where the Metal compute path diverges from CPU.
        if let Some(ref dir) = dump_path {
            cmd.commit();
            cmd.wait_until_completed();
            let write_f32 = |name: &str, buf: &metal::Buffer, n: usize| {
                let ptr = buf.contents() as *const f32;
                if ptr.is_null() { return; }
                let s = unsafe { std::slice::from_raw_parts(ptr, n) };
                let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
                let path = format!("{dir}/metal_layer_{l:02}_{name}.f32");
                if let Err(e) = std::fs::write(&path, &bytes) {
                    eprintln!("[dump] failed to write {path}: {e}");
                }
            };
            // End-of-layer residual (matches CPU dump exactly).
            write_f32("h_out", &h_bufs[l + 1], seq_len * hidden);
            // Per-stage snapshots for layer 0 only (noise budget): these
            // let us bisect which shader stage first diverges from CPU.
            if l == 0 {
                write_f32("norm_out",     &norm_outs[l],     seq_len * hidden);
                write_f32("q_out",        &q_outs[l],        seq_len * layer_q_dim);
                write_f32("k_out",        &k_outs[l],        seq_len * layer_kv_dim);
                write_f32("v_out",        &v_outs[l],        seq_len * layer_kv_dim);
                write_f32("attn_out",     &attn_outs[l],     seq_len * layer_q_dim);
                write_f32("o_out",        &o_outs[l],        seq_len * hidden);
                write_f32("h_post_attn",  &h_post_attns[l],  seq_len * hidden);
                write_f32("ffn_norm_out", &ffn_norm_outs[l], seq_len * hidden);
                write_f32("gate_out",     &gate_outs[l],     seq_len * inter);
                write_f32("up_out",       &up_outs[l],       seq_len * inter);
                write_f32("act_buf",      &act_bufs_vec[l],  seq_len * inter);
                write_f32("down_out",     &down_outs[l],     seq_len * hidden);
            }
            cmd = queue.new_command_buffer();
        }
    }

    cmd.commit();
    cmd.wait_until_completed();

    // Populate KV cache from GPU-computed RoPE'd K and V (post-commit, buffers readable)
    if let Some(ref mut kv) = kv_cache {
        for l in 0..num_layers {
            let lhd = layers[l].head_dim;
            let lnkv = layers[l].num_kv_heads;
            while kv.layers.len() <= l {
                kv.layers.push(super::kv_cache::LayerKVCache::new(
                    bufs, 4096, lnkv, lhd));
            }
            let total_kv = seq_len * lnkv * lhd;
            let k_src = k_outs[l].contents() as *const f32;
            let v_src = v_outs[l].contents() as *const f32;
            let k_dst = kv.layers[l].k_cache.contents() as *mut f32;
            let v_dst = kv.layers[l].v_cache.contents() as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(k_src, k_dst, total_kv);
                std::ptr::copy_nonoverlapping(v_src, v_dst, total_kv);
            }
            kv.layers[l].current_len = seq_len;
        }
    }

    // Read final hidden state — `seq_len * hidden` floats, caller reshapes
    // to [seq_len, hidden] (see `layer_graph::generate`).
    crate::metal::buffers::read_buffer_f32(&h_bufs[num_layers], seq_len * hidden)
}
