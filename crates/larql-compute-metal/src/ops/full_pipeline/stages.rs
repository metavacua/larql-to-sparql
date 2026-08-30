//! Per-stage encoders extracted from the `dispatch_full_pipeline`
//! per-layer body.
//!
//! Each stage takes a context bundle so the function signatures stay
//! readable instead of carrying 20+ parameters. Behaviour mirrors the
//! inline code byte-for-byte — pure organisation, no logic change.

use metal::{CommandBufferRef, ComputePipelineState};

use super::buffers::LayerBuffers;
use crate::stages::{input_norm, qkv_proj, quant_matvec};
use larql_compute::FullPipelineLayer;

/// Per-layer geometry + offsets needed by the input-norm + QKV stage.
pub(super) struct LayerCtx {
    pub eps: f32,
    pub norm_offset: f32,
    pub layer_q_dim: usize,
    pub layer_kv_dim: usize,
    pub q8_row_max: usize,
    pub q8s_row_bytes: usize,
}

/// Pipeline references the input-norm + QKV stage may dispatch.
/// All matvec-side fields are bare `ComputePipelineState`s mirroring
/// the existing `dispatch_full_pipeline` signature; only `q4_matvec`
/// flows through the format-aware quant_matvec stage helper which
/// expects a [`crate::kernels::KernelHandle`].
#[allow(dead_code)]
pub(super) struct InputNormQkvPipes<'a> {
    pub rms_norm: &'a ComputePipelineState,
    pub rms_norm_q8: &'a ComputePipelineState,
    pub q8_qkv_proj: &'a ComputePipelineState,
    pub q4kf_qkv_proj: Option<&'a ComputePipelineState>,
    pub q4k_qkv_proj: Option<&'a ComputePipelineState>,
    /// Mixed Q4_K Q/K + Q6_K V fused kernel (the production Gemma 3/4
    /// shape). Absent → the mixed route degrades to per-projection, as
    /// it always silently did before the QKV plan landed.
    pub q4k_q6k_qkv_proj: Option<&'a crate::kernels::KernelHandle>,
    /// Element-wise `out += bias` for attention projection biases
    /// (GPT-OSS). Absent only on legacy callers; a layer that carries a
    /// bias refuses loudly rather than silently dropping it.
    pub bias_add: Option<&'a ComputePipelineState>,
    pub qm_pipes: quant_matvec::Pipelines<'a>,
}

/// Stage 5 — O projection (+ optional O bias), per position in one
/// encoder. (q4k_matmul ship-log: wiring the gemm here for seq_len>1
/// prefill was a kernel-isolated 3.8× that did NOT translate end-to-end
/// — within-noise short, ~10% regression long; the matvec was already
/// bandwidth-near-peak and the matmul's `[seq_len × q_dim]` X working
/// set thrashes L1. Reverted 2026-04-28.)
///
/// K comes from the store's own row width, not the logical `layer_q_dim`
/// — O rows may be padded to the quant block (audit-F17 class; the
/// padded columns dequantise to zero, and `buffers.rs` allocates the
/// zeroed read slack). The O bias, when the layer carries one, joins
/// before the residual add — the same point the CPU reference applies it.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_o_proj_stage(
    cmd: &CommandBufferRef,
    layer: &FullPipelineLayer<'_>,
    layer_idx: usize,
    seq_len: usize,
    hidden: usize,
    layer_q_dim: usize,
    qm_pipes: &quant_matvec::Pipelines<'_>,
    q8_quant_pipeline: &ComputePipelineState,
    bias_add_pipeline: Option<&ComputePipelineState>,
    lb: &LayerBuffers,
) {
    let l = layer_idx;
    let h_off = |p: usize| (p * hidden * 4) as u64;
    let q_off = |p: usize| (p * layer_q_dim * 4) as u64;
    let q8_off = |p: usize| (p * lb.q8_row_max) as u64;
    let q8s_off = |p: usize| (p * lb.q8s_row_bytes) as u64;
    let o_k_store = layer.wo.stored_cols(hidden, layer_q_dim);
    let enc = cmd.new_compute_command_encoder();
    for pos in 0..seq_len {
        crate::stages::o_proj::encode(
            enc,
            qm_pipes,
            q8_quant_pipeline,
            layer.wo.format(),
            &lb.wo[l],
            &lb.attn_out[l],
            q_off(pos),
            &lb.q8[l],
            q8_off(pos),
            &lb.q8s[l],
            q8s_off(pos),
            &lb.o_out[l],
            h_off(pos),
            o_k_store,
            hidden,
        );
    }
    if let Some(ob) = &lb.attn_o_bias[l] {
        let bias_pipe = bias_add_pipeline.expect(
            "layer carries an O-projection bias but no bias_add pipeline \
             was supplied — refusing to silently drop it",
        );
        for pos in 0..seq_len {
            crate::stages::bias_add::encode(enc, bias_pipe, &lb.o_out[l], h_off(pos), ob, hidden);
        }
    }
    enc.end_encoding();
}

/// Stage 1+3 — input norm followed by Q/K/V projection. Format-aware
/// per layer (Q4_K family takes f32 input through a fused or
/// per-projection shader; Q4_0 fuses the norm with Q8 quant then
/// dispatches per-projection Q4_0 matvec; Q8_0 uses the fused-Q8-QKV
/// shader).
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_input_norm_and_qkv(
    cmd: &CommandBufferRef,
    layer: &FullPipelineLayer<'_>,
    layer_idx: usize,
    seq_len: usize,
    hidden: usize,
    ctx: &LayerCtx,
    pipes: &InputNormQkvPipes<'_>,
    lb: &LayerBuffers,
) {
    let l = layer_idx;
    // The QKV plan — kernel route + required input encoding — from the
    // full (wq, wk, wv) triple. This replaces two single-operand gates
    // (`uses_f32_input` keyed on wq alone; `all_same_format` + a
    // two-arm format match) and the hand-written Q8_0 triple check
    // below, all of which could disagree with each other and with the
    // decode path's route table (capability audit, slice 1). It also
    // makes the mixed Q4_K/Q6_K-V kernel reachable at prefill for the
    // first time — a Gemma 3/4 layer previously degraded to three
    // per-projection dispatches here.
    let plan = qkv_proj::plan_qkv(layer.wq.format(), layer.wk.format(), layer.wv.format());

    // The store's own row width — writers pad rows to the quant block, so
    // hidden%256≠0 models (GPT-OSS: 2880 → 3072) store wider rows than
    // `hidden`. The f32-input QKV kernels derive their superblock count
    // as `K / 256`, so they must run at the stored width; the padded
    // weight columns dequantise to exactly zero, making this exact
    // (`buffers.rs` allocates and zeroes the read slack). Mirrors the
    // decode path in `decode/encode_qkv.rs`.
    let k_store = layer.wq.stored_cols(ctx.layer_q_dim, hidden);
    for (name, w, rows) in [
        ("wk", &layer.wk, ctx.layer_kv_dim),
        ("wv", &layer.wv, ctx.layer_kv_dim),
    ] {
        assert_eq!(
            w.stored_cols(rows, hidden),
            k_store,
            "QKV row stores disagree on their padded width ({name}); \
             refusing to guess a shared K"
        );
    }

    let h_off = |p: usize| (p * hidden * 4) as u64;
    let q_off = |p: usize| (p * ctx.layer_q_dim * 4) as u64;
    let kv_off = |p: usize| (p * ctx.layer_kv_dim * 4) as u64;
    let q8_off = |p: usize| (p * ctx.q8_row_max) as u64;
    let q8s_off = |p: usize| (p * ctx.q8s_row_bytes) as u64;

    // Fused-kernel availability per route. Encoded as a
    // (pipeline, kernel) pair so the dispatcher can't use a pipeline
    // with another kernel's TG geometry — desync silently leaves rows
    // unwritten behind the kernels' row guards. A route whose pipeline
    // is absent degrades to per-projection.
    let fused_qkv_pipe: Option<(&ComputePipelineState, qkv_proj::FusedQkvKernel)> = match plan.route
    {
        qkv_proj::QkvFormatRoute::UniformQ4Kf => pipes
            .q4kf_qkv_proj
            .map(|p| (p, qkv_proj::FusedQkvKernel::Q4kf))
            .or_else(|| {
                pipes
                    .q4k_qkv_proj
                    .map(|p| (p, qkv_proj::FusedQkvKernel::Q4k))
            }),
        qkv_proj::QkvFormatRoute::UniformQ4K => pipes
            .q4k_qkv_proj
            .map(|p| (p, qkv_proj::FusedQkvKernel::Q4k)),
        qkv_proj::QkvFormatRoute::MixedQ4kQ6kV => pipes
            .q4k_q6k_qkv_proj
            .map(|kh| (&kh.state, qkv_proj::FusedQkvKernel::Q4kQ6kV)),
        qkv_proj::QkvFormatRoute::FusedQ8 | qkv_proj::QkvFormatRoute::PerProjection => None,
    };

    // Encoder coalescing: hoist `cmd.new_compute_command_encoder()` and
    // `enc.end_encoding()` out of the per-position loop so we pay one
    // encoder-create + end_encoding per layer per stage instead of
    // `seq_len` of them. The per-position dispatches inside don't touch
    // encoder lifecycle (only set_pipeline_state / set_buffer / dispatch),
    // so they run back-to-back on the GPU. Saves ~5 µs × seq_len per layer
    // on prefill — see ROADMAP P0 "Prefill: per-position matvec → matmul"
    // entry, 2026-04-27.
    if plan.input == qkv_proj::QkvInputEncoding::F32 {
        // Q4_K / Q6_K / Q4_KF: f32 norm output, then either fused or
        // per-projection QKV matvec.
        let enc = cmd.new_compute_command_encoder();
        for pos in 0..seq_len {
            input_norm::encode_f32(
                enc,
                pipes.rms_norm,
                &lb.h[l],
                h_off(pos),
                &lb.input_norm[l],
                &lb.norm_out[l],
                h_off(pos),
                hidden,
                ctx.eps,
                ctx.norm_offset,
            );
            if let Some((fused_pipeline, fused_kernel)) = fused_qkv_pipe {
                qkv_proj::encode_fused_f32(
                    enc,
                    fused_pipeline,
                    fused_kernel,
                    &lb.wq[l],
                    &lb.wk[l],
                    &lb.wv[l],
                    &lb.norm_out[l],
                    h_off(pos),
                    &lb.q_out[l],
                    q_off(pos),
                    &lb.k_out[l],
                    kv_off(pos),
                    &lb.v_out[l],
                    kv_off(pos),
                    ctx.layer_q_dim,
                    ctx.layer_kv_dim,
                    k_store,
                );
            } else {
                let pos_qoff = q_off(pos);
                let pos_kvoff = kv_off(pos);
                qkv_proj::encode_per_proj(
                    enc,
                    &pipes.qm_pipes,
                    &lb.norm_out[l],
                    h_off(pos),
                    // Q8 input unused for f32-input formats — placeholder.
                    &lb.norm_out[l],
                    0,
                    &lb.norm_out[l],
                    0,
                    [
                        qkv_proj::Proj {
                            format: layer.wq.format(),
                            w_buf: &lb.wq[l],
                            out_buf: &lb.q_out[l],
                            out_off: pos_qoff,
                            rows: ctx.layer_q_dim,
                        },
                        qkv_proj::Proj {
                            format: layer.wk.format(),
                            w_buf: &lb.wk[l],
                            out_buf: &lb.k_out[l],
                            out_off: pos_kvoff,
                            rows: ctx.layer_kv_dim,
                        },
                        qkv_proj::Proj {
                            format: layer.wv.format(),
                            w_buf: &lb.wv[l],
                            out_buf: &lb.v_out[l],
                            out_off: pos_kvoff,
                            rows: ctx.layer_kv_dim,
                        },
                    ],
                    k_store,
                );
            }
        }
        enc.end_encoding();
    } else {
        // Legacy Q8-input formats: first fuse rms_norm+Q8-quantise, then
        // route by weight layout. Q4_0 weights stay packed Q4_0 and must go
        // through the Q4_0 matvec helper; Q8_0 weights use the fused Q8 QKV
        // shader with separate per-row weight scales.
        let enc = cmd.new_compute_command_encoder();
        for pos in 0..seq_len {
            input_norm::encode_q8(
                enc,
                pipes.rms_norm_q8,
                &lb.h[l],
                h_off(pos),
                &lb.input_norm[l],
                &lb.q8[l],
                q8_off(pos),
                &lb.q8s[l],
                q8s_off(pos),
                hidden,
                ctx.eps,
                ctx.norm_offset,
            );
            if plan.route == qkv_proj::QkvFormatRoute::FusedQ8 {
                qkv_proj::encode_fused_q8(
                    enc,
                    pipes.q8_qkv_proj,
                    &lb.wq[l],
                    lb.wq_scale[l]
                        .as_ref()
                        .expect("q8 qkv path requires external scales"),
                    &lb.wk[l],
                    lb.wk_scale[l]
                        .as_ref()
                        .expect("q8 qkv path requires external scales"),
                    &lb.wv[l],
                    lb.wv_scale[l]
                        .as_ref()
                        .expect("q8 qkv path requires external scales"),
                    &lb.q8[l],
                    q8_off(pos),
                    &lb.q8s[l],
                    q8s_off(pos),
                    &lb.q_out[l],
                    q_off(pos),
                    &lb.k_out[l],
                    kv_off(pos),
                    &lb.v_out[l],
                    kv_off(pos),
                    ctx.layer_q_dim,
                    ctx.layer_kv_dim,
                    hidden,
                );
            } else {
                let pos_qoff = q_off(pos);
                let pos_kvoff = kv_off(pos);
                qkv_proj::encode_per_proj(
                    enc,
                    &pipes.qm_pipes,
                    &lb.h[l],
                    h_off(pos),
                    &lb.q8[l],
                    q8_off(pos),
                    &lb.q8s[l],
                    q8s_off(pos),
                    [
                        qkv_proj::Proj {
                            format: layer.wq.format(),
                            w_buf: &lb.wq[l],
                            out_buf: &lb.q_out[l],
                            out_off: pos_qoff,
                            rows: ctx.layer_q_dim,
                        },
                        qkv_proj::Proj {
                            format: layer.wk.format(),
                            w_buf: &lb.wk[l],
                            out_buf: &lb.k_out[l],
                            out_off: pos_kvoff,
                            rows: ctx.layer_kv_dim,
                        },
                        qkv_proj::Proj {
                            format: layer.wv.format(),
                            w_buf: &lb.wv[l],
                            out_buf: &lb.v_out[l],
                            out_off: pos_kvoff,
                            rows: ctx.layer_kv_dim,
                        },
                    ],
                    hidden,
                );
            }
        }
        enc.end_encoding();
    }

    // Attention projection biases (GPT-OSS: Q/K/V all carry one) join
    // right after the projections, so QK-norm/RoPE and the KV-cache copy
    // downstream read the biased values — the same points the CPU
    // reference (`forward::add_bias`) applies them. Dispatched per
    // position, only when the layer has the bias.
    let bias_sites: [(_, _, _, &dyn Fn(usize) -> u64, _); 3] = [
        (
            layer.attn_q_bias,
            &lb.attn_q_bias[l],
            &lb.q_out[l],
            &q_off,
            ctx.layer_q_dim,
        ),
        (
            layer.attn_k_bias,
            &lb.attn_k_bias[l],
            &lb.k_out[l],
            &kv_off,
            ctx.layer_kv_dim,
        ),
        (
            layer.attn_v_bias,
            &lb.attn_v_bias[l],
            &lb.v_out[l],
            &kv_off,
            ctx.layer_kv_dim,
        ),
    ];
    if bias_sites.iter().any(|(s, ..)| s.is_some()) {
        let bias_pipe = pipes.bias_add.expect(
            "layer carries attention projection biases but no bias_add pipeline was \
             supplied — refusing to silently drop them",
        );
        let enc = cmd.new_compute_command_encoder();
        for (slice, buf, out, off, n) in bias_sites {
            let (Some(s), Some(b)) = (slice, buf) else {
                continue;
            };
            assert_eq!(
                s.len(),
                n,
                "attention projection bias has {} entries but the projection is \
                 {n} wide — the extracted tensor does not match this model",
                s.len()
            );
            for pos in 0..seq_len {
                crate::stages::bias_add::encode(enc, bias_pipe, out, off(pos), b, n);
            }
        }
        enc.end_encoding();
    }
}
