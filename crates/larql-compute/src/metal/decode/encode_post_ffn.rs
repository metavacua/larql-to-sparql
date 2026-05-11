//! Step 7: post-FFN residual + optional post-FFN norm.
//!
//! Three shapes covered, all behaviourally identical to the previously-inlined
//! versions (one in the dense branch, one inside the MoE-deferred FFN path):
//!
//! 1. `has_post_norms == false` — straight residual add `h_post_attn + down_out → new_h`.
//! 2. `has_post_norms && layer.post_ffn_norm.is_none()` — same straight residual
//!    add (post_ffn norm slot wasn't populated for this layer).
//! 3. `has_post_norms && layer.post_ffn_norm.is_some()` — RMS-norm `down_out` against
//!    `post_ffn_norm`, then residual-add against `h_post_attn` into `new_h`.
//!    When `use_fused == true`, dispatches the single fused
//!    `post_ffn_norm_residual_add` kernel (default-on for the dense path); when
//!    `use_fused == false`, falls back to the unfused `rms_norm` +
//!    `residual_add` two-dispatch chain (used by the MoE-deferred FFN path,
//!    matching prior behaviour exactly).
//!
//! `LARQL_FUSED_POST_FFN_NORM=0` is honoured only via the `use_fused` arg the
//! caller passes — the env-var resolution stays in the decode loop so this
//! helper has zero env-var I/O on the hot path.

use crate::metal::ops::full_pipeline::{encode_residual_add, encode_rms_norm};
use crate::metal::MetalBackend;
use crate::FullPipelineLayer;
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

pub(super) struct PostFfnBufs<'a> {
    pub down_out: &'a Buffer,
    pub h_post_attn: &'a Buffer,
    pub new_h: &'a Buffer,
    /// Scratch for the unfused chain. Unused when `use_fused == true`.
    pub normed_scratch: &'a Buffer,
}

/// D-RMS-FUSE Phase 1 hint: when present + `LARQL_FUSED_PRELAYER_NORM=1`,
/// the non-post-norms branch dispatches `residual_norm_store` instead of
/// plain `residual_add`, fusing the next layer's input rms_norm into the
/// same kernel call. The next layer's `encode_q4k_input_norm` then skips
/// its own dispatch (the data is already in the shared `norm_f32_buf`).
pub(super) struct PreLayerNormFusion<'a> {
    /// Next layer's `input_norm` weight slice.
    pub next_input_norm: &'a [f32],
    /// Shared `norm_f32_buf` (= next layer's `bufs.norm_out`) — written by
    /// the fused `residual_norm_store` dispatch.
    pub next_norm_out: &'a Buffer,
}

impl MetalBackend {
    pub(super) fn encode_post_ffn_residual(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: PostFfnBufs<'_>,
        hidden: usize,
        use_fused: bool,
        prelayer_fusion: Option<&PreLayerNormFusion<'_>>,
    ) {
        // M2: read norm-related layer fields through the structured view.
        // `post_ffn_norm` is the only weight slice that doesn't have a
        // pre-extracted buffer in `bufs` — keep that as a direct field
        // access on the layer.
        let norms_view = layer.norms();

        // D-RMS-FUSE Phase 1: on the non-post-norms path (Llama / Mistral /
        // Qwen / etc.), if the caller passed in next-layer info AND the
        // env var is on, dispatch `residual_norm_store` to fuse the
        // residual-add with the next layer's input rms_norm in one kernel.
        // Saves 1 dispatch per layer × num_layers (~7 µs each).
        if let Some(fusion) = prelayer_fusion
            .filter(|_| !norms_view.has_post_norms && self.decode_flags.fused_prelayer_norm)
        {
            let next_input_norm_buf = self.bufs.get_f32(fusion.next_input_norm);
            let hidden_val = hidden as u32;
            let eps = norms_view.eps;
            let norm_offset = norms_view.norm_offset;
            enc.set_compute_pipeline_state(&self.norms.residual_norm_store_pipeline);
            enc.set_buffer(0, Some(bufs.h_post_attn), 0); // a (residual base)
            enc.set_buffer(1, Some(bufs.down_out), 0); // b (FFN output)
            enc.set_buffer(2, Some(&next_input_norm_buf), 0); // weight = next layer's input_norm
            enc.set_buffer(3, Some(fusion.next_norm_out), 0); // norm_out (next layer's normed input)
            enc.set_buffer(4, Some(bufs.new_h), 0); // sum_out (raw new_h for residual)
            enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(
                    crate::metal::kernel::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                    1,
                    1,
                ),
            );
            return;
        }

        if norms_view.has_post_norms {
            if let Some(post_ffn) = layer.post_ffn_norm {
                let post_ffn_buf = self.bufs.get_f32(post_ffn);
                if use_fused {
                    let hidden_val = hidden as u32;
                    let eps = norms_view.eps;
                    let norm_offset = norms_view.norm_offset;
                    enc.set_compute_pipeline_state(&self.norms.post_ffn_norm_residual_add_pipeline);
                    enc.set_buffer(0, Some(bufs.down_out), 0);
                    enc.set_buffer(1, Some(bufs.h_post_attn), 0);
                    enc.set_buffer(2, Some(&post_ffn_buf), 0);
                    enc.set_buffer(3, Some(bufs.new_h), 0);
                    enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(
                            crate::metal::kernel::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                            1,
                            1,
                        ),
                    );
                } else {
                    encode_rms_norm(
                        enc,
                        &self.norms.rms_norm_pipeline,
                        bufs.down_out,
                        &post_ffn_buf,
                        bufs.normed_scratch,
                        hidden,
                        norms_view.eps,
                        norms_view.norm_offset,
                    );
                    encode_residual_add(
                        enc,
                        &self.norms.residual_add_pipeline,
                        bufs.h_post_attn,
                        bufs.normed_scratch,
                        bufs.new_h,
                        hidden,
                    );
                }
            } else {
                encode_residual_add(
                    enc,
                    &self.norms.residual_add_pipeline,
                    bufs.h_post_attn,
                    bufs.down_out,
                    bufs.new_h,
                    hidden,
                );
            }
        } else {
            encode_residual_add(
                enc,
                &self.norms.residual_add_pipeline,
                bufs.h_post_attn,
                bufs.down_out,
                bufs.new_h,
                hidden,
            );
        }
    }
}
