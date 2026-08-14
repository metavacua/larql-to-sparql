//! Step 1 of the decode pipeline: input norm + fused Q/K/V projection.
//!
//! Two top-level paths gated on `uses_kquant`:
//!   - **Q4_K family** (Q4_K, Q6_K, Q4_KF) — RMS or LayerNorm into f32,
//!     then a fused QKV shader keyed on the (wq.fmt, wk.fmt, wv.fmt)
//!     triplet:
//!       * uniform Q4_K / Q4_KF → `q4k_qkv_proj` / `q4kf_qkv_proj`
//!       * Q4_K Q/K + Q6_K V (Gemma 3 / 4 Ollama convention) →
//!         `q4k_q6k_qkv_proj`
//!       * anything else → per-projection fallback through `quant_matvec`
//!   - **Q4_0** (legacy Q8 input) — fused norm+Q8 quantize, then
//!     per-projection Q4_0 matvec.
//!   - **Q8_0** — fused norm+Q8 quantize, then `q8_qkv_proj`.
//!
//! Used to live inline in `decode_token_with_moe_fn`. Pulled out here
//! so the hot decode function stays scannable.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::MetalBackend;
use larql_compute::FullPipelineLayer;

/// Buffer references the QKV step reads or writes.
pub(super) struct QkvBufs<'a> {
    // Input
    pub h_in: &'a metal::Buffer,
    // Per-layer weights + scales
    pub input_norm: &'a metal::Buffer,
    pub input_norm_bias: Option<&'a [f32]>,
    pub wq: &'a metal::Buffer,
    pub wk: &'a metal::Buffer,
    pub wv: &'a metal::Buffer,
    pub wq_scales: Option<&'a metal::Buffer>, // present only for external-scale formats
    pub wk_scales: Option<&'a metal::Buffer>,
    pub wv_scales: Option<&'a metal::Buffer>,
    // Outputs
    pub norm_out: &'a metal::Buffer,
    pub q_out: &'a metal::Buffer,
    pub k_out: &'a metal::Buffer,
    pub v_out: &'a metal::Buffer,
    // Scratch (Q4_0 path only)
    pub ffn_q8: &'a metal::Buffer,
    pub ffn_q8s: &'a metal::Buffer,
}

#[derive(Copy, Clone)]
pub(super) struct QkvDims {
    pub hidden: usize,
    pub layer_q_dim: usize,
    pub layer_kv_dim: usize,
    pub eps: f32,
    pub norm_offset: f32,
}

impl MetalBackend {
    /// Encode input norm + fused QKV projection. `uses_kquant` selects the
    /// top-level path; the layer's per-projection formats select the
    /// inner shader. Behaviour mirrors the inline form previously in
    /// `decode/mod.rs` byte-for-byte.
    ///
    /// **M2 migration note (2026-05-09)**: this function reads the
    /// layer through structured views (`weights().attention.{wq,wk,wv}`,
    /// `norms().{norm_type, ...}`) instead of touching `layer.wq` /
    /// `layer.norm_type` directly. The inner helpers (`encode_q4k_qkv`,
    /// `encode_q4k_input_norm`, etc.) still take `&FullPipelineLayer`
    /// — migrating them is the next step. See modularity tracker M2 in
    /// ROADMAP.md for the full migration plan.
    pub(super) fn encode_input_norm_and_qkv(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: QkvBufs<'_>,
        dims: QkvDims,
        input_already_normed: bool,
        // True when `attn_fused_will_fire` says the fused attention
        // kernel will apply the Q/K/V projection biases itself — the
        // separate bias_add dispatches below must then be SKIPPED or the
        // biases apply twice. Same shared authority both sites consult.
        qkv_bias_deferred: bool,
    ) {
        // The QKV plan (kernel route + input encoding) from the full
        // (wq, wk, wv) triple — the same authority the prefill and
        // hybrid paths consult. Replaces the caller-supplied
        // `uses_kquant` boolean, which was keyed on wq alone
        // (capability audit, slice 1). Unroutable triples (floats,
        // BitNet, mixed input encodings) refuse here, before any
        // dispatch is encoded.
        let plan = crate::stages::qkv_proj::plan_qkv(
            layer.wq.format(),
            layer.wk.format(),
            layer.wv.format(),
        );
        if plan.input == crate::stages::qkv_proj::QkvInputEncoding::F32 {
            // Default path (since 2026-05-09): separate `rms_norm` dispatch
            // + non-fused `q4k_q6k_qkv_proj`. The fused alternative
            // (`q4k_q6k_qkv_proj_normed`) saves 1 dispatch/layer (~0.24
            // ms/tok) but rereads H+norm_w 3× per TG, dropping the kernel
            // from 287 → 199 GB/s — the bandwidth cost (~1.4 ms/tok in
            // the per-kernel diag) exceeds the dispatch saving. Measured
            // end-to-end on Gemma 3 4B: +1.6 tok/s, −0.30 ms/tok GPU fwd
            // by defusing. `LARQL_QKV_FUSED=1` opts back in.
            // Cached at startup; see `metal::flags::DecodeFlags`.
            let use_fused = self.decode_flags.qkv_fused;

            // Pull structured views once at the top — replaces the
            // direct `layer.wq.format()` / `layer.norm_type` /
            // `layer.input_norm_bias` reads scattered through the
            // function body. The compiler optimises the view methods
            // to plain field copies, so this is zero-cost.
            let weights = layer.weights().attention;
            let norms = layer.norms();

            // Route descriptor — replaces the inline `(q, k, v)`
            // boolean conjunction. The normed-QKV opt-in fires only on
            // the mixed-Q4K-Q6K-V route today, but reading it through
            // the descriptor means a future "uniform Q4_K with normed
            // kernel" variant (or any other (q, k, v) triple supported
            // by `q4k_q6k_qkv_proj_normed`) lands as one match arm
            // here, not a new boolean.
            use crate::stages::qkv_proj::{pick_qkv_route, QkvFormatRoute};
            let route = pick_qkv_route(
                weights.wq.format(),
                weights.wk.format(),
                weights.wv.format(),
            );
            let mixed_q4k_q6k_v = matches!(route, QkvFormatRoute::MixedQ4kQ6kV);
            // The norm-fused kernel derives its projection width from the
            // norm width, so it cannot serve padded row stores (stored
            // width > hidden, e.g. GPT-OSS 2880 → 3072); those take the
            // separate-norm chain, whose QKV dispatch runs at the store's
            // own width.
            let store_is_padded =
                weights.wq.stored_cols(dims.layer_q_dim, dims.hidden) != dims.hidden;
            if mixed_q4k_q6k_v
                && use_fused
                && !store_is_padded
                && norms.norm_type == larql_compute::NormType::RmsNorm
                && norms.input_norm_bias.is_none()
            {
                // Fused norm+QKV path always recomputes norm internally;
                // `input_already_normed` is ignored here.
                self.encode_normed_q4k_q6k_qkv(enc, layer, &bufs, dims);
            } else {
                // D-RMS-FUSE Phase 1: the previous layer's
                // `encode_post_ffn_residual` may have pre-written
                // `bufs.norm_out` via `residual_norm_store` using THIS
                // layer's input_norm weight. When `input_already_normed`
                // is true, skip the redundant rms_norm dispatch.
                if !input_already_normed {
                    self.encode_q4k_input_norm(enc, layer, &bufs, dims);
                }
                self.encode_q4k_qkv(enc, layer, &bufs, dims);
            }
        } else {
            // D-RMS-FUSE Phase 1 doesn't apply to the Q4_0 path yet —
            // run the standard norm+qkv chain.
            let _ = input_already_normed;
            self.encode_q4_0_norm_and_qkv(enc, layer, &bufs, dims);
        }

        // Attention projection biases (GPT-OSS: Q/K/V all carry one) join
        // right after the projections, so QK-norm/RoPE and the KV-cache
        // append downstream read the biased values — the same points the
        // CPU reference (`forward::add_bias`) applies them. Dispatched
        // only when present; bias-free layers encode nothing extra.
        if qkv_bias_deferred {
            return;
        }
        for (bias, out, n) in [
            (layer.attn_q_bias, bufs.q_out, dims.layer_q_dim),
            (layer.attn_k_bias, bufs.k_out, dims.layer_kv_dim),
            (layer.attn_v_bias, bufs.v_out, dims.layer_kv_dim),
        ] {
            if let Some(b) = bias {
                assert_eq!(
                    b.len(),
                    n,
                    "attention projection bias has {} entries but the projection \
                     is {n} wide — the extracted tensor does not match this model",
                    b.len()
                );
                let b_buf = self.bufs.get_f32(b);
                crate::stages::bias_add::encode(
                    enc,
                    &self.attention.bias_add_pipeline,
                    out,
                    0,
                    &b_buf,
                    n,
                );
            }
        }
    }

    // ── Q4_K family: norm → f32, then fused QKV shader ───────────────────────

    fn encode_q4k_input_norm(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &QkvBufs<'_>,
        dims: QkvDims,
    ) {
        use crate::ops::full_pipeline::encode_rms_norm;
        let QkvDims {
            hidden,
            eps,
            norm_offset,
            ..
        } = dims;

        // M2: read through the structured norm view. `norms.norm_type`
        // is the only `layer.*` field this function consumes; the rest
        // of the inputs are already pre-extracted into `bufs` and `dims`.
        let norms = layer.norms();
        if norms.norm_type == larql_compute::NormType::LayerNorm {
            let len_val = hidden as u32;
            if let Some(bias) = bufs.input_norm_bias {
                let bias_buf = self.bufs.get_f32(bias);
                enc.set_compute_pipeline_state(&self.norms.layer_norm_pipeline);
                enc.set_buffer(0, Some(bufs.h_in), 0);
                enc.set_buffer(1, Some(bufs.input_norm), 0);
                enc.set_buffer(2, Some(&bias_buf), 0);
                enc.set_buffer(3, Some(bufs.norm_out), 0);
                enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            } else {
                enc.set_compute_pipeline_state(&self.norms.layer_norm_no_bias_pipeline);
                enc.set_buffer(0, Some(bufs.h_in), 0);
                enc.set_buffer(1, Some(bufs.input_norm), 0);
                enc.set_buffer(2, Some(bufs.norm_out), 0);
                enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            }
            enc.dispatch_threads(
                MTLSize::new(hidden as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                    1,
                    1,
                ),
            );
        } else {
            encode_rms_norm(
                enc,
                &self.norms.rms_norm_pipeline,
                bufs.h_in,
                bufs.input_norm,
                bufs.norm_out,
                hidden,
                eps,
                norm_offset,
            );
        }
    }

    fn encode_q4k_qkv(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &QkvBufs<'_>,
        dims: QkvDims,
    ) {
        let QkvDims {
            hidden,
            layer_q_dim,
            layer_kv_dim,
            ..
        } = dims;

        // M2: read attention weights through the structured view rather
        // than touching `layer.wq` / `layer.wk` / `layer.wv` directly.
        let attn_weights = layer.weights().attention;

        // The store's own row width, derived from the byte count — writers
        // pad rows to the quant block, so hidden%256≠0 models (GPT-OSS:
        // 2880 → 3072) store wider rows than `hidden`. Every kernel below
        // derives its superblock count as `K / 256`; handing it the logical
        // width truncates the tail and desynchronises the row stride
        // (audit F17). The padded weight columns dequantise to exactly
        // zero, so running at the stored width is exact provided the input
        // buffer has that many readable floats (the setup allocates the
        // slack and zeroes it once).
        let k_store = attn_weights.wq.stored_cols(layer_q_dim, hidden);
        for (name, w, rows) in [
            ("wk", &attn_weights.wk, layer_kv_dim),
            ("wv", &attn_weights.wv, layer_kv_dim),
        ] {
            assert_eq!(
                w.stored_cols(rows, hidden),
                k_store,
                "QKV row stores disagree on their padded width ({name}); \
                 refusing to guess a shared K"
            );
        }

        // Format-route descriptor — single source of truth for how a
        // `(q, k, v)` triple maps to a fused QKV pipeline. See
        // `metal::stages::qkv_proj::pick_qkv_route` for the table.
        use crate::stages::qkv_proj::{pick_qkv_route, QkvFormatRoute};
        let route = pick_qkv_route(
            attn_weights.wq.format(),
            attn_weights.wk.format(),
            attn_weights.wv.format(),
        );

        match route {
            // Q8_0 triples carry the Q8 input encoding, so the plan
            // sends them down the Q8 branch of the entry function —
            // this f32-input helper can never see the route.
            QkvFormatRoute::FusedQ8 => {
                unreachable!("FusedQ8 is served by the Q8-input branch")
            }
            QkvFormatRoute::UniformQ4K | QkvFormatRoute::UniformQ4Kf => {
                use crate::stages::qkv_proj::FusedQkvKernel;
                let (fused_pipe, fused_kernel) = match route {
                    QkvFormatRoute::UniformQ4Kf => {
                        (&self.attention.q4kf_qkv_proj_pipeline, FusedQkvKernel::Q4kf)
                    }
                    QkvFormatRoute::UniformQ4K => {
                        (&self.attention.q4k_qkv_proj_pipeline, FusedQkvKernel::Q4k)
                    }
                    _ => unreachable!("outer match restricts to Uniform*"),
                };
                crate::stages::qkv_proj::encode_fused_f32(
                    enc,
                    &fused_pipe.state,
                    fused_kernel,
                    bufs.wq,
                    bufs.wk,
                    bufs.wv,
                    bufs.norm_out,
                    0,
                    bufs.q_out,
                    0,
                    bufs.k_out,
                    0,
                    bufs.v_out,
                    0,
                    layer_q_dim,
                    layer_kv_dim,
                    k_store,
                );
            }
            QkvFormatRoute::MixedQ4kQ6kV => {
                // Geometry travels with the bound `KernelHandle` (mirrors the
                // decode_hybrid Q4_K geometry-fix pattern).
                //
                // Same superblock-truncation hazard as `encode_fused_f32`'s
                // assert (audit F17): this kernel derives its superblock
                // count as `K / 256`, so a misaligned K silently drops the
                // tail. The stored width satisfies this by construction;
                // an unpadded misaligned store must refuse, not truncate.
                assert!(
                    k_store.is_multiple_of(256),
                    "mixed Q4K/Q6K-V QKV kernel requires K % 256 == 0; got {k_store}"
                );
                let kh = &self.attention.q4k_q6k_qkv_proj_pipeline;
                let total_rows = (layer_q_dim + layer_kv_dim + layer_kv_dim) as u64;
                let num_tgs = total_rows.div_ceil(kh.rows_per_tg);
                let q_rows_u = layer_q_dim as u32;
                let k_rows_u = layer_kv_dim as u32;
                let v_rows_u = layer_kv_dim as u32;
                let k_u = k_store as u32;
                enc.set_compute_pipeline_state(&kh.state);
                enc.set_buffer(0, Some(bufs.wq), 0);
                enc.set_buffer(1, Some(bufs.wk), 0);
                enc.set_buffer(2, Some(bufs.wv), 0);
                enc.set_buffer(3, Some(bufs.norm_out), 0);
                enc.set_buffer(4, Some(bufs.q_out), 0);
                enc.set_buffer(5, Some(bufs.k_out), 0);
                enc.set_buffer(6, Some(bufs.v_out), 0);
                enc.set_bytes(7, 4, &q_rows_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &k_rows_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(9, 4, &v_rows_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(10, 4, &k_u as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_tgs, 1, 1),
                    MTLSize::new(kh.threads_per_tg, 1, 1),
                );
            }
            QkvFormatRoute::PerProjection => {
                // Mixed-but-unsupported (e.g. Q4_KF + Q6_K, or Q4_0 legacy):
                // per-projection dispatch through the format-aware helper.
                use crate::stages::qkv_proj::{self, Proj};
                use crate::stages::quant_matvec::Pipelines;
                let pipes = Pipelines {
                    q4kf_proj: Some(&self.attention.q4kf_proj_pipeline.state),
                    q4k_matvec_fallback: &self.quant.q4k_matvec_pipeline,
                    q6k_matvec: &self.quant.q6k_matvec_pipeline,
                    q4_matvec: &self.q4.matvec,
                    // Decode is seq=1; matmul amortisation has nothing to amortise.
                    q4k_matmul: None,
                };
                qkv_proj::encode_per_proj(
                    enc,
                    &pipes,
                    bufs.norm_out,
                    0,
                    // Q8 bufs unused for f32-input formats — pass norm as a
                    // harmless placeholder.
                    bufs.norm_out,
                    0,
                    bufs.norm_out,
                    0,
                    [
                        Proj {
                            format: attn_weights.wq.format(),
                            w_buf: bufs.wq,
                            out_buf: bufs.q_out,
                            out_off: 0,
                            rows: layer_q_dim,
                        },
                        Proj {
                            format: attn_weights.wk.format(),
                            w_buf: bufs.wk,
                            out_buf: bufs.k_out,
                            out_off: 0,
                            rows: layer_kv_dim,
                        },
                        Proj {
                            format: attn_weights.wv.format(),
                            w_buf: bufs.wv,
                            out_buf: bufs.v_out,
                            out_off: 0,
                            rows: layer_kv_dim,
                        },
                    ],
                    k_store,
                );
            }
        }
    }

    // ── Q4_0 / Q8_0 legacy: norm+Q8 → QKV ────────────────────────────────────

    fn encode_q4_0_norm_and_qkv(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &QkvBufs<'_>,
        dims: QkvDims,
    ) {
        let QkvDims {
            hidden,
            layer_q_dim,
            layer_kv_dim,
            eps,
            norm_offset,
        } = dims;
        let hidden_val = hidden as u32;

        // Fused norm + Q8 quantize (in-place into the FFN scratch
        // buffers — they're re-quantised before the FFN dispatch).
        enc.set_compute_pipeline_state(&self.norms.rms_norm_q8_pipeline);
        enc.set_buffer(0, Some(bufs.h_in), 0);
        enc.set_buffer(1, Some(bufs.input_norm), 0);
        enc.set_buffer(2, Some(bufs.ffn_q8), 0);
        enc.set_buffer(3, Some(bufs.ffn_q8s), 0);
        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                1,
                1,
            ),
        );

        // M2: read the per-projection format triple once, through the
        // structured weights view. Route from the plan's table — this
        // was the second copy of a hand-written Q8_0 triple check
        // (prefill held the other) that the route table now owns.
        let attn_weights = layer.weights().attention;
        let route = crate::stages::qkv_proj::pick_qkv_route(
            attn_weights.wq.format(),
            attn_weights.wk.format(),
            attn_weights.wv.format(),
        );
        if route == crate::stages::qkv_proj::QkvFormatRoute::FusedQ8 {
            let total_rows = (layer_q_dim + layer_kv_dim + layer_kv_dim) as u32;
            let q_rows = layer_q_dim as u32;
            let k_rows = layer_kv_dim as u32;
            let v_rows = layer_kv_dim as u32;
            let k_val = hidden as u32;
            // Pull dispatch geometry from the bound `KernelHandle` —
            // same fix class as the decode_hybrid Q4_K geometry bug.
            // q8_qkv_proj is currently 8 rows/TG, 256 threads; if a
            // future bump changes the variant, the dispatch follows.
            assert!(
                hidden <= crate::shaders::q8_attn_proj::MAX_K,
                "q8_qkv_proj stages its input in threadgroup memory capped at \
                 K = {}; hidden {hidden} would corrupt it (audit F13)",
                crate::shaders::q8_attn_proj::MAX_K,
            );
            let kh = &self.attention.q8_qkv_proj_pipeline;
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(bufs.wq), 0);
            enc.set_buffer(1, Some(bufs.wk), 0);
            enc.set_buffer(2, Some(bufs.wv), 0);
            enc.set_buffer(3, Some(bufs.ffn_q8), 0);
            enc.set_buffer(
                4,
                Some(
                    bufs.wq_scales
                        .expect("legacy scale path requires an external-scale format"),
                ),
                0,
            );
            enc.set_buffer(
                5,
                Some(
                    bufs.wk_scales
                        .expect("legacy scale path requires an external-scale format"),
                ),
                0,
            );
            enc.set_buffer(
                6,
                Some(
                    bufs.wv_scales
                        .expect("legacy scale path requires an external-scale format"),
                ),
                0,
            );
            enc.set_buffer(7, Some(bufs.ffn_q8s), 0);
            enc.set_buffer(8, Some(bufs.q_out), 0);
            enc.set_buffer(9, Some(bufs.k_out), 0);
            enc.set_buffer(10, Some(bufs.v_out), 0);
            enc.set_bytes(11, 4, &q_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(12, 4, &k_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(13, 4, &v_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(14, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((total_rows as u64).div_ceil(kh.rows_per_tg), 1, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        } else {
            use crate::stages::qkv_proj::{self, Proj};
            use crate::stages::quant_matvec::Pipelines;
            let pipes = Pipelines {
                q4kf_proj: Some(&self.attention.q4kf_proj_pipeline.state),
                q4k_matvec_fallback: &self.quant.q4k_matvec_pipeline,
                q6k_matvec: &self.quant.q6k_matvec_pipeline,
                q4_matvec: &self.q4.matvec,
                q4k_matmul: None,
            };
            qkv_proj::encode_per_proj(
                enc,
                &pipes,
                bufs.h_in,
                0,
                bufs.ffn_q8,
                0,
                bufs.ffn_q8s,
                0,
                [
                    Proj {
                        format: attn_weights.wq.format(),
                        w_buf: bufs.wq,
                        out_buf: bufs.q_out,
                        out_off: 0,
                        rows: layer_q_dim,
                    },
                    Proj {
                        format: attn_weights.wk.format(),
                        w_buf: bufs.wk,
                        out_buf: bufs.k_out,
                        out_off: 0,
                        rows: layer_kv_dim,
                    },
                    Proj {
                        format: attn_weights.wv.format(),
                        w_buf: bufs.wv,
                        out_buf: bufs.v_out,
                        out_off: 0,
                        rows: layer_kv_dim,
                    },
                ],
                hidden,
            );
        }
    }

    // ── Fused RMS norm + Q4K/Q6K QKV (Gemma 3/4 production path) ─────────────

    /// Fused dispatch: cooperatively reduces ||h||² within each TG, then runs
    /// the Q4_K+Q6_K mixed QKV matvec with inline normalization.
    /// Replaces `encode_q4k_input_norm` + `encode_q4k_qkv` (saves 1 dispatch).
    fn encode_normed_q4k_q6k_qkv(
        &self,
        enc: &ComputeCommandEncoderRef,
        _layer: &FullPipelineLayer,
        bufs: &QkvBufs<'_>,
        dims: QkvDims,
    ) {
        use crate::shaders::q4k_q6k_qkv_proj as sh;
        let QkvDims {
            hidden,
            layer_q_dim,
            layer_kv_dim,
            eps,
            norm_offset,
        } = dims;
        let total_rows = (layer_q_dim + layer_kv_dim + layer_kv_dim) as u64;
        let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
        let q_u = layer_q_dim as u32;
        let k_u = layer_kv_dim as u32;
        let v_u = layer_kv_dim as u32;
        let hidden_u = hidden as u32;

        enc.set_compute_pipeline_state(&self.attention.q4k_q6k_qkv_proj_normed_pipeline.state);
        enc.set_buffer(0, Some(bufs.wq), 0);
        enc.set_buffer(1, Some(bufs.wk), 0);
        enc.set_buffer(2, Some(bufs.wv), 0);
        enc.set_buffer(3, Some(bufs.h_in), 0);
        enc.set_buffer(4, Some(bufs.input_norm), 0);
        enc.set_buffer(5, Some(bufs.q_out), 0);
        enc.set_buffer(6, Some(bufs.k_out), 0);
        enc.set_buffer(7, Some(bufs.v_out), 0);
        enc.set_bytes(8, 4, &q_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(9, 4, &k_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(10, 4, &v_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(11, 4, &hidden_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(12, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(13, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(sh::THREADS_PER_TG, 1, 1),
        );
    }
}
