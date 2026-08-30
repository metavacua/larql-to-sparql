//! The gate+up and down phases as separate encodes, for the split MoE path.
//!
//! Split out of `encode_ffn.rs`; [`super`] holds the shared types.

#[allow(unused_imports)]
use super::*;
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;
use metal::{ComputeCommandEncoderRef, MTLSize};

impl MetalBackend {
    /// Encode the gate+up dispatch only. Writes to `bufs.gate_out_scratch`
    /// and `bufs.up_out`; does NOT encode activation or down.
    pub(crate) fn encode_ffn_gate_up_phase(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        dims: FfnDims,
    ) {
        let FfnDims { hidden, inter, .. } = dims;
        let inter_val = inter as u32;
        let hidden_val = hidden as u32;
        // Same per-operand route as `encode_ffn_step`. The previous
        // family-boolean split hardcoded the 8sg Q4_K kernel (ignoring
        // every variant flag, so split-mode profiled a kernel
        // production might not run) and sent a Q6_K gate to the Q4_K
        // pipelines (capability audit, slice 2).
        use larql_compute::QuantFormat;
        let route_fmt = validate_ffn_formats(layer);

        if route_fmt == QuantFormat::Q4_KF {
            use crate::shaders::q4kf_ffn_gate_up as q4kf_gu;
            use crate::shaders::q4kf_qkv_proj as q4kf;
            if layer.is_gated() {
                let n = (inter as u64).div_ceil(q4kf_gu::ROWS_PER_TG);
                enc.set_compute_pipeline_state(&self.ffn.q4kf_ffn_gate_up_pipeline.state);
                enc.set_buffer(0, Some(bufs.gate_w), 0);
                enc.set_buffer(1, Some(bufs.up_w), 0);
                enc.set_buffer(2, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
                enc.set_buffer(4, Some(bufs.up_out), 0);
                enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(n * 2, 1, 1),
                    MTLSize::new(q4kf_gu::THREADS_PER_TG, 1, 1),
                );
            } else {
                let n = (inter as u64).div_ceil(q4kf::ROWS_PER_TG);
                enc.set_compute_pipeline_state(&self.attention.q4kf_proj_pipeline.state);
                enc.set_buffer(0, Some(bufs.up_w), 0);
                enc.set_buffer(1, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(2, Some(bufs.up_out), 0);
                enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(n, 1, 1),
                    MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
                );
            }
        } else if route_fmt == QuantFormat::Q6_K {
            // Separated per-format gate/up via quant_matvec — mirrors
            // `encode_q6k_ffn`'s first half.
            use crate::stages::quant_matvec::{self as qmv, Pipelines};
            let pipes = Pipelines {
                q4kf_proj: Some(&self.attention.q4kf_proj_pipeline.state),
                q4k_matvec_fallback: &self.quant.q4k_matvec_pipeline,
                q6k_matvec: &self.quant.q6k_matvec_pipeline,
                q4_matvec: &self.q4.matvec,
                q4k_matmul: None,
            };
            if layer.is_gated() {
                qmv::encode(
                    enc,
                    QuantFormat::Q6_K,
                    bufs.gate_w,
                    bufs.ffn_norm_out,
                    0,
                    bufs.ffn_norm_out,
                    0,
                    bufs.ffn_norm_out,
                    0,
                    bufs.gate_out_scratch,
                    0,
                    &pipes,
                    inter,
                    hidden,
                );
            }
            qmv::encode(
                enc,
                QuantFormat::Q6_K,
                bufs.up_w,
                bufs.ffn_norm_out,
                0,
                bufs.ffn_norm_out,
                0,
                bufs.ffn_norm_out,
                0,
                bufs.up_out,
                0,
                &pipes,
                inter,
                hidden,
            );
        } else if route_fmt == QuantFormat::Q4_K {
            if layer.is_gated() {
                let (pipeline, rows, tgs) = self.q4k_gate_up_selection();
                let n = (inter as u64).div_ceil(rows);
                enc.set_compute_pipeline_state(pipeline);
                enc.set_buffer(0, Some(bufs.gate_w), 0);
                enc.set_buffer(1, Some(bufs.up_w), 0);
                enc.set_buffer(2, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
                enc.set_buffer(4, Some(bufs.up_out), 0);
                enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n * 2, 1, 1), MTLSize::new(tgs, 1, 1));
            } else {
                let rpt = self.quant.q4k_matvec_pipeline.rows_per_tg;
                let tpt = self.quant.q4k_matvec_pipeline.threads_per_tg;
                let n = (inter as u64).div_ceil(rpt);
                enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
                enc.set_buffer(0, Some(bufs.up_w), 0);
                enc.set_buffer(1, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(2, Some(bufs.up_out), 0);
                enc.set_bytes(3, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n, 1, 1), MTLSize::new(tpt, 1, 1));
            }
        } else {
            // Q4_0 path
            let kernel = &self.q4.matvec;
            let n = (inter as u64).div_ceil(kernel.rows_per_tg);
            let tg = MTLSize::new(kernel.threads_per_tg, 1, 1);
            if layer.is_gated() {
                enc.set_compute_pipeline_state(&kernel.state);
                enc.set_buffer(0, Some(bufs.gate_w), 0);
                enc.set_buffer(1, Some(bufs.ffn_q8), 0);
                enc.set_buffer(2, Some(bufs.ffn_q8s), 0);
                enc.set_buffer(3, Some(bufs.gate_out_scratch), 0);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n, 1, 1), tg);
                enc.set_buffer(0, Some(bufs.up_w), 0);
                enc.set_buffer(3, Some(bufs.up_out), 0);
                enc.dispatch_thread_groups(MTLSize::new(n, 1, 1), tg);
            } else {
                enc.set_compute_pipeline_state(&kernel.state);
                enc.set_buffer(0, Some(bufs.up_w), 0);
                enc.set_buffer(1, Some(bufs.ffn_q8), 0);
                enc.set_buffer(2, Some(bufs.ffn_q8s), 0);
                enc.set_buffer(3, Some(bufs.up_out), 0);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(n, 1, 1), tg);
            }
        }
    }

    /// Encode the activation (GEGLU/SiLU) + down dispatch only. Reads from
    /// `bufs.gate_out_scratch` / `bufs.up_out` written by `encode_ffn_gate_up_phase`.
    pub(crate) fn encode_ffn_down_phase(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        dims: FfnDims,
    ) {
        let FfnDims {
            hidden,
            inter,
            inter_padded,
        } = dims;
        let inter_val = inter as u32;
        let inter_padded_val = inter_padded as u32;
        let hidden_val = hidden as u32;
        // Same per-operand route and the SAME fused-down guards as
        // `encode_q4k_ffn`. This phase previously fired the fused
        // Q4_K GEGLU+down on `down == Q4_K` alone — no
        // MAX_FUSED_GEGLU_DOWN_INTER ceiling, no LARQL_FUSED_DOWN
        // opt-out — so split-mode routed Gemma 4 31B through the very
        // kernel that guard exists to avoid, and measurements were not
        // the production configuration (capability audit, slice 2).
        use larql_compute::QuantFormat;
        let route_fmt = validate_ffn_formats(layer);

        if route_fmt == QuantFormat::Q4_KF || route_fmt == QuantFormat::Q6_K {
            // Both route their down per its own format via qmv; only
            // the gate/up side differed, and that phase has run.
            if layer.is_gated() {
                self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);
            } else {
                self.encode_activation(
                    enc,
                    layer,
                    bufs.up_out,
                    bufs.act_buf,
                    inter_val,
                    inter as u64,
                );
            }
            self.encode_qmv_down(enc, layer, bufs, hidden, inter);
        } else if route_fmt == QuantFormat::Q4_K {
            if layer.is_gated() {
                // Hard-disabled — see the integration-NaN note in
                // `encode_q4k_ffn`'s fused-q6k arm. Original gate:
                //   decode_flags.fused_q6k_down && down == Q6_K
                //     && activation == GeluTanh
                let use_fused_q6k = false;
                if layer.down.format() == larql_compute::QuantFormat::Q4_K
                    && inter_padded <= MAX_FUSED_GEGLU_DOWN_INTER
                    && self.decode_flags.fused_down
                {
                    self.encode_q4k_fused_geglu_down(
                        enc,
                        layer,
                        bufs,
                        hidden,
                        inter_padded,
                        hidden_val,
                        inter_padded_val,
                    );
                } else if use_fused_q6k {
                    let kh = &self.ffn.q6k_geglu_gelu_tanh_down_pipeline;
                    let n_tgs = (hidden as u64).div_ceil(kh.rows_per_tg);
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(bufs.down_w), 0);
                    enc.set_buffer(1, Some(bufs.gate_out_scratch), 0);
                    enc.set_buffer(2, Some(bufs.up_out), 0);
                    enc.set_buffer(3, Some(bufs.down_out), 0);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &inter_val as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        metal::MTLSize::new(n_tgs, 1, 1),
                        metal::MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                } else {
                    self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);
                    self.encode_qmv_down(enc, layer, bufs, hidden, inter_padded);
                }
            } else {
                self.encode_activation(
                    enc,
                    layer,
                    bufs.up_out,
                    bufs.act_buf,
                    inter_val,
                    inter as u64,
                );
                // Down per its own format (was hardcoded q4k_matvec).
                self.encode_qmv_down(enc, layer, bufs, hidden, inter_padded);
            }
        } else {
            // Q4_0
            if layer.is_gated() {
                self.encode_geglu(enc, layer, bufs, inter_val, inter as u64);
            } else {
                self.encode_activation(
                    enc,
                    layer,
                    bufs.up_out,
                    bufs.act_buf,
                    inter_val,
                    inter as u64,
                );
            }
            enc.set_compute_pipeline_state(&self.q4.f32_matvec);
            enc.set_buffer(0, Some(bufs.down_w), 0);
            enc.set_buffer(1, Some(bufs.act_buf), 0);
            enc.set_buffer(2, Some(bufs.down_out), 0);
            enc.set_bytes(3, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &inter_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
    }

    pub(super) fn encode_activation(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        in_buf: &metal::Buffer,
        out_buf: &metal::Buffer,
        inter_val: u32,
        inter_threads: u64,
    ) {
        crate::stages::ffn::assert_metal_activation_supported(
            layer.activation,
            "metal::decode::encode_activation",
        );
        let pipe = match layer.activation {
            larql_compute::Activation::Silu => &self.ffn.silu_pipeline,
            larql_compute::Activation::GeluTanh => &self.ffn.gelu_tanh_pipeline,
            larql_compute::Activation::GeluExact | larql_compute::Activation::ReLU => {
                unreachable!()
            }
        };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(in_buf), 0);
        enc.set_buffer(1, Some(out_buf), 0);
        enc.set_bytes(2, 4, &inter_val as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new(inter_threads, 1, 1), MTLSize::new(256, 1, 1));
    }

    pub(super) fn encode_qmv_down(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: &FfnBufs<'_>,
        hidden: usize,
        inter: usize,
    ) {
        use crate::stages::quant_matvec::{self as qmv, Pipelines};
        let pipes = Pipelines {
            q4kf_proj: Some(&self.attention.q4kf_proj_pipeline.state),
            q4k_matvec_fallback: &self.quant.q4k_matvec_pipeline,
            q6k_matvec: &self.quant.q6k_matvec_pipeline,
            q4_matvec: &self.q4.matvec,
            q4k_matmul: None,
        };
        qmv::encode(
            enc,
            layer.down.format(),
            bufs.down_w,
            bufs.act_buf,
            0,
            bufs.act_buf,
            0,
            bufs.act_buf,
            0,
            bufs.down_out,
            0,
            &pipes,
            hidden,
            inter,
        );
    }
}
