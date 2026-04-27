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
#[allow(dead_code, clippy::too_many_arguments)]
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

/// Format-aware single-vector matvec, used by both FFN gate/up/down and
/// the QKV per-projection fallback. Thin wrapper around
/// [`crate::metal::stages::quant_matvec::encode`] kept to preserve the
/// old local-helper name while the refactor to `stages/` proceeds.
#[allow(dead_code, clippy::too_many_arguments)]
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
    use crate::metal::stages::quant_matvec;
    let pipes = quant_matvec::Pipelines {
        q4kf_proj: q4kf_proj_pipeline,
        q4k_matvec_fallback: q4k_pipeline,
        q6k_matvec: q6k_pipeline,
        q4_matvec: q4_matvec_pipeline,
    };
    quant_matvec::encode(
        enc, format, w_buf,
        f32_in, f32_in_off,
        q8_in, q8_in_off, q8s_in, q8s_in_off,
        out_buf, out_off,
        &pipes,
        num_rows, hidden,
    );
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
            use crate::metal::shaders::q4k_matvec as q4k;
            let n = num_rows as u32;
            let k = hidden as u32;
            let tgs = (num_rows as u64).div_ceil(q4k::ROWS_PER_TG);
            enc.set_compute_pipeline_state(q4k_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), 0);
            enc.set_buffer(2, Some(buf_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
        }
        crate::QuantFormat::Q6_K => {
            use crate::metal::shaders::q6k_matvec as q6k;
            let n = num_rows as u32;
            let k = hidden as u32;
            let tgs = (num_rows as u64).div_ceil(q6k::ROWS_PER_TG);
            enc.set_compute_pipeline_state(q6k_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), 0);
            enc.set_buffer(2, Some(buf_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(q6k::THREADS_PER_TG, 1, 1));
        }
        crate::QuantFormat::Q4_KF => {
            use crate::metal::shaders::q4k_matvec as q4k;
            let n = num_rows as u32;
            let k = hidden as u32;
            let tgs = (num_rows as u64).div_ceil(q4k::ROWS_PER_TG);
            enc.set_compute_pipeline_state(q4k_pipeline);
            enc.set_buffer(0, Some(buf_w), 0);
            enc.set_buffer(1, Some(buf_input), 0);
            enc.set_buffer(2, Some(buf_out), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(q4k::THREADS_PER_TG, 1, 1));
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
    _q8_matvec_pipeline: &ComputePipelineState,
    q8_qkv_proj_pipeline: &ComputePipelineState,
    q4k_matvec_pipeline: &ComputePipelineState,
    q6k_matvec_pipeline: &ComputePipelineState,
    rms_norm_pipeline: &ComputePipelineState,
    residual_add_pipeline: &ComputePipelineState,
    rms_norm_q8_pipeline: &ComputePipelineState,
    _residual_norm_q8_pipeline: &ComputePipelineState,
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
    _kv_dim: usize,
    seq_len: usize,
    _num_q_heads: usize,
    _num_kv_heads: usize,
    _head_dim: usize,
    _rope_base: f32, // global fallback; per-layer layers[l].rope_base used in loop
    use_qk_norm: bool,
    softcap: f32,
) -> Vec<f32> {
    let num_layers = layers.len();
    let _hidden_val = hidden as u32;
    let _inter_val = inter as u32;
    let _n_blocks = (hidden / 32) as u32;

    // Pre-cache Q8 attention weight buffers (higher precision for Q/K dot products)
    // Stable across calls → cache by slice identity (skips per-token Metal-buffer
    // allocation for ~68+ norm/scale handles on 34-layer Gemma 3 4B).
    let wq_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wq.data)).collect();
    let wq_scale_bufs: Vec<_> = layers.iter().map(|l| bufs.get_f32(l.wq.scales.unwrap_or(&[]))).collect();
    let wk_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wk.data)).collect();
    let wk_scale_bufs: Vec<_> = layers.iter().map(|l| bufs.get_f32(l.wk.scales.unwrap_or(&[]))).collect();
    let wv_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wv.data)).collect();
    let wv_scale_bufs: Vec<_> = layers.iter().map(|l| bufs.get_f32(l.wv.scales.unwrap_or(&[]))).collect();
    let wo_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wo.data)).collect();
    let _wo_scale_bufs: Vec<_> = layers.iter().map(|l| bufs.get_f32(l.wo.scales.unwrap_or(&[]))).collect();
    // Q4 FFN weight buffers
    let gate_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.gate.data)).collect();
    let up_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.up.data)).collect();
    let down_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.down.data)).collect();

    // Norm weight buffers — also stable; cache.
    let input_norm_bufs: Vec<_> = layers.iter().map(|l| bufs.get_f32(l.input_norm)).collect();
    let post_attn_norm_bufs: Vec<_> = layers.iter().map(|l| bufs.get_f32(l.post_attn_norm)).collect();
    let pre_ffn_norm_bufs: Vec<Option<_>> = layers.iter().map(|l| {
        l.pre_ffn_norm.map(|n| bufs.get_f32(n))
    }).collect();
    let post_ffn_norm_bufs: Vec<Option<_>> = layers.iter().map(|l| {
        l.post_ffn_norm.map(|n| bufs.get_f32(n))
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
    let q8s_row_bytes = q8_row_max.div_ceil(32) * 4;
    for layer in layers.iter().take(num_layers) {
        let lq = layer.num_q_heads * layer.head_dim;
        let lkv = layer.num_kv_heads * layer.head_dim;
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
        ffn_q8s_bufs.push(bufs.output((seq_len * hidden.div_ceil(32) * 4) as u64));
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
        let _inter_off = |p: usize| (p * inter * 4) as u64;
        let q8_off = |p: usize| (p * q8_row_max) as u64;
        let q8s_off = |p: usize| (p * q8s_row_bytes) as u64;
        let _ffn_q8_off = |p: usize| (p * hidden) as u64;
        let _ffn_q8s_off = |p: usize| (p * hidden.div_ceil(32) * 4) as u64;

        // Stage 1+2: input norm + Q/K/V projection, format-aware, per position.
        use crate::metal::stages::{input_norm, qkv_proj, quant_matvec};
        let all_same_format = layers[l].wq.format == layers[l].wk.format
            && layers[l].wk.format == layers[l].wv.format;
        let fused_qkv_pipe = q4kf_qkv_proj_pipeline.or(q4k_qkv_proj_pipeline)
            .filter(|_| all_same_format
                && matches!(layers[l].wq.format,
                    crate::QuantFormat::Q4_K | crate::QuantFormat::Q4_KF));
        let qm_pipes = quant_matvec::Pipelines {
            q4kf_proj: q4kf_proj_pipeline,
            q4k_matvec_fallback: q4k_matvec_pipeline,
            q6k_matvec: q6k_matvec_pipeline,
            q4_matvec: &q4.matvec,
        };

        if uses_f32_input {
            // Q4_K / Q6_K / Q4_KF: f32 norm output, then either fused or
            // per-projection QKV matvec.
            for pos in 0..seq_len {
                let enc = cmd.new_compute_command_encoder();
                input_norm::encode_f32(
                    enc, rms_norm_pipeline,
                    &h_bufs[l], h_off(pos),
                    &input_norm_bufs[l],
                    &norm_outs[l], h_off(pos),
                    hidden, eps, norm_offset,
                );
                if let Some(fused_pipeline) = fused_qkv_pipe {
                    qkv_proj::encode_fused_f32(
                        enc, fused_pipeline,
                        &wq_bufs[l], &wk_bufs[l], &wv_bufs[l],
                        &norm_outs[l], h_off(pos),
                        &q_outs[l], q_off(pos),
                        &k_outs[l], kv_off(pos),
                        &v_outs[l], kv_off(pos),
                        layer_q_dim, layer_kv_dim, hidden,
                    );
                } else {
                    qkv_proj::encode_per_proj(
                        enc, &qm_pipes,
                        &norm_outs[l], h_off(pos),
                        // Q8 input unused for f32-input formats — pass the
                        // norm-out buffer as a harmless placeholder.
                        &norm_outs[l], 0, &norm_outs[l], 0,
                        [
                            qkv_proj::Proj { format: layers[l].wq.format, w_buf: &wq_bufs[l], out_buf: &q_outs[l], out_off: q_off(pos),  rows: layer_q_dim },
                            qkv_proj::Proj { format: layers[l].wk.format, w_buf: &wk_bufs[l], out_buf: &k_outs[l], out_off: kv_off(pos), rows: layer_kv_dim },
                            qkv_proj::Proj { format: layers[l].wv.format, w_buf: &wv_bufs[l], out_buf: &v_outs[l], out_off: kv_off(pos), rows: layer_kv_dim },
                        ],
                        hidden,
                    );
                }
                enc.end_encoding();
            }
        } else {
            // Q8_0: fused rms_norm+Q8-quantise, then fused Q8 QKV projection.
            for pos in 0..seq_len {
                let enc = cmd.new_compute_command_encoder();
                input_norm::encode_q8(
                    enc, rms_norm_q8_pipeline,
                    &h_bufs[l], h_off(pos),
                    &input_norm_bufs[l],
                    &q8_bufs[l], q8_off(pos),
                    &q8s_bufs[l], q8s_off(pos),
                    hidden, eps, norm_offset,
                );
                qkv_proj::encode_fused_q8(
                    enc, q8_qkv_proj_pipeline,
                    &wq_bufs[l], &wq_scale_bufs[l],
                    &wk_bufs[l], &wk_scale_bufs[l],
                    &wv_bufs[l], &wv_scale_bufs[l],
                    &q8_bufs[l], q8_off(pos),
                    &q8s_bufs[l], q8s_off(pos),
                    &q_outs[l], q_off(pos),
                    &k_outs[l], kv_off(pos),
                    &v_outs[l], kv_off(pos),
                    layer_q_dim, layer_kv_dim, hidden,
                );
                enc.end_encoding();
            }
        }

        // ── 3 (pre). Optional parameter-free V-norm (Gemma 4). ──
        if layers[l].has_v_norm {
            if let Some(qk_norm_pipe) = qk_norm_pipeline {
                let ones: Vec<f32> = vec![1.0; layer_head_dim];
                let ones_buf = bufs.transient_from_f32(&ones);
                let enc = cmd.new_compute_command_encoder();
                crate::metal::stages::qk_norm::encode_v_norm(
                    enc, qk_norm_pipe,
                    &v_outs[l], &ones_buf,
                    seq_len, layer_num_kv_heads, layer_head_dim, eps,
                );
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
        let applied_prerope_qk_norm = if use_qk_norm {
            if let (Some(qk_norm_pipe), Some(q_w_slice), Some(k_w_slice)) =
                (qk_norm_pipeline, layers[l].q_norm_weight, layers[l].k_norm_weight)
            {
                let q_w_buf = bufs.get_f32(q_w_slice);
                let k_w_buf = bufs.get_f32(k_w_slice);
                let enc = cmd.new_compute_command_encoder();
                crate::metal::stages::qk_norm::encode_qk_norm(
                    enc, qk_norm_pipe,
                    &q_outs[l], &q_w_buf,
                    &k_outs[l], &k_w_buf,
                    seq_len, layer_num_q_heads, layer_num_kv_heads, layer_head_dim,
                    eps, layers[l].qk_norm_offset,
                );
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
        let use_separate_rope = kv_cache.is_some() && rope_at_pos_pipeline.is_some();
        if use_separate_rope {
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::rope::encode(
                enc, rope_at_pos_pipeline.unwrap(),
                &q_outs[l], &k_outs[l],
                seq_len, layer_num_q_heads, layer_num_kv_heads, layer_head_dim,
                layers[l].rotary_dim, layer_rope_base,
            );
            enc.end_encoding();
        }

        // ── 4. Fused attention (RoPE + GQA + softcap, multi-position). ──
        if let Some(fused_pipeline) = fused_attn_pipeline {
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::attention::encode(
                enc, fused_pipeline,
                &q_outs[l], &k_outs[l], &v_outs[l], &attn_outs[l],
                seq_len, layer_num_q_heads, layer_num_kv_heads, layer_head_dim,
                layer_attn_scale, layer_rope_base,
                crate::metal::stages::attention::Flags {
                    // Caller pre-applied QK-norm: tell shader to skip its internal
                    // normalisation so we don't double-normalise.
                    use_qk_norm: use_qk_norm && !applied_prerope_qk_norm,
                    skip_rope: use_separate_rope,
                    softcap,
                    rotary_dim: layers[l].rotary_dim as u32,
                },
            );
            enc.end_encoding();
        }

        // ── 5. O projection. Per position. ──
        for pos in 0..seq_len {
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::o_proj::encode(
                enc, &qm_pipes, q8_quant_pipeline,
                layers[l].wo.format,
                &wo_bufs[l],
                &attn_outs[l], q_off(pos),
                &q8_bufs[l], q8_off(pos),
                &q8s_bufs[l], q8s_off(pos),
                &o_outs[l], h_off(pos),
                layer_q_dim, hidden,
            );
            enc.end_encoding();
        }

        // ── 6. Post-attention residual + pre-FFN norm (+ optional Q8 quant). ──
        //
        // Two output representations are needed here:
        //   (a) ffn_norm_outs[l]  — f32 per position; consumed by Q4_K / Q4_KF /
        //                            Q6_K FFN which expect f32 input.
        //   (b) ffn_q8_bufs[l] + ffn_q8s_bufs[l] — Q8 + scales per position;
        //       consumed only by Q4_0 / Q8_0 FFN.
        // `h_post_attns[l]` holds the post-residual f32 hidden state for the
        // final residual add at the end of this layer (step 10).
        let ffn_format = layers[l].gate.format;
        let ffn_needs_q8 = matches!(ffn_format,
            crate::QuantFormat::Q4_0 | crate::QuantFormat::Q8_0);
        let pre_ffn_weight_buf: &metal::Buffer = if has_post_norms {
            pre_ffn_norm_bufs[l].as_ref().unwrap_or(&post_attn_norm_bufs[l])
        } else {
            &post_attn_norm_bufs[l]
        };
        {
            let mut scratch = |bytes: u64| bufs.output(bytes);
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::residual::encode_post_attn(
                enc, rms_norm_pipeline, residual_add_pipeline, q8_quant_pipeline,
                &mut scratch,
                &h_bufs[l], &o_outs[l], &h_post_attns[l], &ffn_norm_outs[l],
                &post_attn_norm_bufs[l], pre_ffn_weight_buf,
                &ffn_q8_bufs[l], &ffn_q8s_bufs[l],
                seq_len, hidden, eps, norm_offset,
                has_post_norms, ffn_needs_q8,
                (hidden * 4) as u64,
                hidden as u64,
                (hidden.div_ceil(32) * 4) as u64,
            );
            enc.end_encoding();
        }

        // ── 7-9. FFN: gate+up → activation → down. Format-aware per position. ──
        {
            use crate::metal::stages::ffn;
            let act = match layers[l].activation {
                crate::Activation::GeluTanh => ffn::Activation::GeluTanh,
                _ => ffn::Activation::SiLU,
            };
            let h_stride = (hidden * 4) as u64;
            let inter_stride = (inter * 4) as u64;
            let q8_stride = hidden as u64;
            let q8s_stride = (hidden.div_ceil(32) * 4) as u64;

            let enc = cmd.new_compute_command_encoder();
            if layers[l].ffn_type == crate::FfnType::Standard {
                ffn::encode_standard(
                    enc, &qm_pipes, silu_pipeline, gelu_tanh_pipeline,
                    layers[l].up.format, layers[l].down.format, act,
                    &up_bufs[l], &down_bufs[l],
                    &ffn_norm_outs[l], &ffn_q8_bufs[l], &ffn_q8s_bufs[l],
                    &up_outs[l], &act_bufs_vec[l], &down_outs[l],
                    seq_len, inter, hidden,
                    h_stride, inter_stride, q8_stride, q8s_stride,
                );
            } else {
                ffn::encode_gated(
                    enc, &qm_pipes, geglu_pipeline, geglu_gelu_tanh_pipeline,
                    layers[l].gate.format, layers[l].up.format, layers[l].down.format, act,
                    &gate_bufs[l], &up_bufs[l], &down_bufs[l],
                    &ffn_norm_outs[l], &ffn_q8_bufs[l], &ffn_q8s_bufs[l],
                    &gate_outs[l], &up_outs[l], &act_bufs_vec[l], &down_outs[l],
                    seq_len, inter, hidden,
                    h_stride, inter_stride, q8_stride, q8s_stride,
                );
            }
            enc.end_encoding();
        }

        // ── 10. Post-FFN: optional norm, then residual add → h for next layer. ──
        {
            let mut scratch = |bytes: u64| bufs.output(bytes);
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::residual::encode_post_ffn(
                enc, rms_norm_pipeline, residual_add_pipeline,
                &mut scratch,
                &down_outs[l], &h_post_attns[l], &h_bufs[l + 1],
                post_ffn_norm_bufs[l].as_ref(),
                seq_len, hidden, eps, norm_offset,
                has_post_norms,
                (hidden * 4) as u64,
            );
            enc.end_encoding();
        }

        // ── 11. Per-layer residual scalar (Gemma 4). ──
        if let Some(scale_pipe) = scale_vector_pipeline {
            let enc = cmd.new_compute_command_encoder();
            crate::metal::stages::layer_scalar::encode(
                enc, scale_pipe, &h_bufs[l + 1], seq_len, hidden, layers[l].layer_scalar,
            );
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
