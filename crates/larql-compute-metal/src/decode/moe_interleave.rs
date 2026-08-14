//! MoE interleave tail for decode.
//!
//! Hybrid MoE layers need a command-buffer split: attention produces
//! `h_post_attn`, the expert path runs on CPU or remotely, and the dense FFN
//! may be encoded on a second GPU command buffer so it overlaps the remote
//! expert round trip. This module owns that tail so `decode/mod.rs` can keep
//! the per-layer happy path readable.

use metal::{Buffer, CommandBuffer, ComputeCommandEncoder};

use super::{diag, encode_ffn, encode_post_ffn, gpu_timing, moe_combine};
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;

pub(super) struct MoeInterleaveCtx<'a> {
    pub layer_idx: usize,
    pub num_layers: usize,
    pub hidden: usize,
    pub inter: usize,
    pub inter_padded: usize,
    pub defer_ffn_for_split: bool,
    pub stage_timing_split: bool,
    pub layer_in_snapshot: Option<&'a [f32]>,
    pub dump_l0_dir: Option<&'a str>,
}

pub(super) struct MoeInterleaveBufs<'a> {
    pub gate_w: &'a Buffer,
    pub up_w: &'a Buffer,
    pub down_w: &'a Buffer,
    pub h_post_attn: &'a Buffer,
    pub ffn_norm_out: &'a Buffer,
    pub ffn_q8: &'a Buffer,
    pub ffn_q8s: &'a Buffer,
    pub gate_out_scratch: &'a Buffer,
    pub up_out: &'a Buffer,
    pub act_buf: &'a Buffer,
    pub down_out: &'a Buffer,
    pub normed_scratch: &'a Buffer,
    pub new_h: &'a Buffer,
}

pub(super) struct MoeCommandState<'a> {
    pub cmd: &'a mut CommandBuffer,
    pub enc: &'a mut ComputeCommandEncoder,
    pub encoder_ended: &'a mut bool,
    pub gpu_time: &'a mut gpu_timing::TokenGpuTime,
    pub residual_dump: &'a mut diag::ResidualDump,
}

/// Context for the merged-command-buffer MoE path: routing on CPU, expert
/// dispatches + weighted combine encoded into the SAME command buffer the
/// next layer's attention will ride — one wait per layer instead of two.
/// Built by `decode_token_q4k_moe` when the backend's expert scratch is
/// live; layers that miss the preconditions fall back to the callback arm
/// unchanged.
pub struct InlineMoeCtx<'a> {
    pub(crate) scratch: &'a crate::moe_dispatch::MoeScratch,
    pub(crate) eps: f32,
}

impl<'a> InlineMoeCtx<'a> {
    pub(crate) fn new(scratch: &'a crate::moe_dispatch::MoeScratch, eps: f32) -> Self {
        Self { scratch, eps }
    }
}

impl MetalBackend {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn handle_moe_interleave(
        &self,
        layer: &FullPipelineLayer<'_>,
        ctx: MoeInterleaveCtx<'_>,
        bufs: MoeInterleaveBufs<'_>,
        state: MoeCommandState<'_>,
        moe_fn: &mut Option<&mut dyn FnMut(usize, &[f32]) -> Vec<f32>>,
        moe_collect_fn: &mut Option<&mut dyn FnMut(usize) -> Vec<f32>>,
        inline_moe: Option<&InlineMoeCtx<'_>>,
    ) {
        // Proceed when this is a hybrid-MoE layer (layer.moe is Some) OR when
        // the entire FFN is remote (ffn_is_remote), which also routes through
        // the moe_fn callback path instead of running a local GPU FFN.
        if layer.moe.is_none() && !layer.ffn_is_remote {
            return;
        }
        // Borrow the MoE weights if present (used only in the local-expert
        // fallback branch — never reached when moe_fn is Some or ffn_is_remote).
        let moe_ref = layer.moe.as_ref();

        state.enc.end_encoding();
        state.cmd.commit();
        state.cmd.wait_until_completed();
        // In split mode the cb we just waited contains ONLY attention
        // (steps 1-5). In non-split mode it normally contains attention +
        // dense FFN; but when stage_timing_split was active, attention was
        // already committed at its own boundary so this cb contains only FFN
        // + post-residual.
        let cb_stage = if ctx.defer_ffn_for_split {
            gpu_timing::DecodeStage::Attention
        } else if ctx.stage_timing_split {
            gpu_timing::DecodeStage::DenseFfn
        } else {
            gpu_timing::DecodeStage::Other
        };
        state.gpu_time.record_stage(state.cmd, cb_stage);
        *state.encoder_ended = true;

        // MoE and dense FFN run on the SAME input (`h_post_attn`, the
        // post-attention residual). Dense FFN output is already in `new_h`.
        let attn_ptr = bufs.h_post_attn.contents() as *const f32;
        let attn_slice = unsafe { std::slice::from_raw_parts(attn_ptr, ctx.hidden) };

        // ── Merged-CB fast path ──────────────────────────────────────────
        // Route on CPU, then encode this layer's experts AND the weighted
        // combine into a FRESH command buffer that stays OPEN — the next
        // layer's attention rides it, so the per-layer wait count halves.
        // Every precondition miss falls through to the callback arm below,
        // byte-for-byte unchanged.
        if let Some(ictx) = inline_moe {
            if self.try_inline_zero_copy_moe(
                layer,
                &ctx,
                &bufs,
                ictx,
                attn_slice,
                &mut *state.cmd,
                &mut *state.enc,
                &mut *state.encoder_ended,
            ) {
                return;
            }
        }
        let moe_out = if ctx.defer_ffn_for_split {
            // Split path: fire MoE NOW, then encode dense FFN + post-FFN
            // residual on a fresh cb so GPU runs while the remote trip is in
            // flight. Pure-MoE layers have no dense branch to overlap —
            // fire and collect with no GPU work in between.
            let fire = moe_fn.as_deref_mut().expect("split_mode implies moe_fn");
            fire(ctx.layer_idx, attn_slice);

            *state.cmd = self.queue.new_command_buffer().to_owned();
            if layer.has_dense_ffn() {
                let ffn_enc = state.cmd.new_compute_command_encoder();

                self.encode_ffn_step(
                    ffn_enc,
                    layer,
                    encode_ffn::FfnBufs {
                        gate_w: bufs.gate_w,
                        up_w: bufs.up_w,
                        down_w: bufs.down_w,
                        ffn_norm_out: bufs.ffn_norm_out,
                        ffn_q8: bufs.ffn_q8,
                        ffn_q8s: bufs.ffn_q8s,
                        gate_out_scratch: bufs.gate_out_scratch,
                        up_out: bufs.up_out,
                        act_buf: bufs.act_buf,
                        down_out: bufs.down_out,
                    },
                    encode_ffn::FfnDims {
                        hidden: ctx.hidden,
                        inter: ctx.inter,
                        inter_padded: ctx.inter_padded,
                    },
                );

                // Always unfused here: this preserves the previous split-MoE path.
                // D-RMS-FUSE Phase 1 not applied: split-MoE path commits per-layer
                // boundaries that don't match the cross-layer fusion pattern.
                self.encode_post_ffn_residual(
                    ffn_enc,
                    layer,
                    encode_post_ffn::PostFfnBufs {
                        down_out: bufs.down_out,
                        h_post_attn: bufs.h_post_attn,
                        new_h: bufs.new_h,
                        normed_scratch: bufs.normed_scratch,
                    },
                    ctx.hidden,
                    false,
                    None,
                );
                ffn_enc.end_encoding();
            }
            state.cmd.commit();

            let collect = moe_collect_fn
                .as_deref_mut()
                .expect("split_mode implies moe_collect_fn");
            let result = collect(ctx.layer_idx);
            state.cmd.wait_until_completed();
            state
                .gpu_time
                .record_stage(state.cmd, gpu_timing::DecodeStage::DenseFfn);
            result
        } else if let Some(ref mut f) = moe_fn {
            f(ctx.layer_idx, attn_slice)
        } else {
            // Local expert fallback — only reachable when moe_fn is None and
            // ffn_is_remote is false (otherwise we'd have taken a branch above).
            let moe = moe_ref.expect("cpu_moe_forward requires moe weights");
            larql_compute::cpu::ops::moe::cpu_moe_forward(
                attn_slice,
                moe,
                layer.norm_offset,
                layer.eps,
            )
        };

        // Accumulate the FFN contribution into the output buffer.
        //
        // Dense hybrid MoE path: new_h = (h_post_attn + dense_ffn) + moe_out.
        //   The GPU has already written `h_post_attn + dense_ffn` into new_h,
        //   so we add moe_out in-place.
        //
        // Remote-FFN path (ffn_is_remote) and pure-MoE layers (no dense
        // branch extracted): new_h = h_post_attn + moe_out. The GPU did
        // NOT run a local FFN, so new_h is uninitialised for this layer;
        // set it directly rather than accumulating into garbage.
        let h_ptr = bufs.new_h.contents() as *mut f32;
        if layer.ffn_is_remote || !layer.has_dense_ffn() {
            // attn_ptr was already computed above (h_post_attn contents).
            unsafe {
                for (i, v) in moe_out.iter().enumerate() {
                    *h_ptr.add(i) = *attn_ptr.add(i) + v;
                }
            }
        } else {
            // Hybrid MoE: new_h already holds (h_post_attn + dense_ffn),
            // add the expert contribution.
            unsafe {
                for (i, v) in moe_out.iter().enumerate() {
                    *h_ptr.add(i) += v;
                }
            }
        }

        if ctx.layer_idx == 0 {
            if let Some(dir) = ctx.dump_l0_dir {
                diag::dump_l0_moe_intermediates(
                    dir,
                    bufs.h_post_attn,
                    bufs.ffn_norm_out,
                    bufs.gate_out_scratch,
                    bufs.up_out,
                    bufs.act_buf,
                    bufs.down_out,
                    bufs.new_h,
                    &moe_out,
                    ctx.hidden,
                    ctx.inter,
                );
            }
        }

        moe_combine::apply_outer_combine(layer, bufs.new_h, bufs.h_post_attn, ctx.hidden);

        if let Some(layer_in) = ctx.layer_in_snapshot {
            let ha = super::super::buffers::read_buffer_f32(bufs.h_post_attn, ctx.hidden);
            let lo = super::super::buffers::read_buffer_f32(bufs.new_h, ctx.hidden);
            state
                .residual_dump
                .record_layer(ctx.layer_idx, layer_in, &ha, &lo);
        }

        if ctx.layer_idx + 1 < ctx.num_layers {
            *state.cmd = self.queue.new_command_buffer().to_owned();
            *state.enc = state.cmd.new_compute_command_encoder().to_owned();
            *state.encoder_ended = false;
        }
    }

    /// The merged-CB arm's gate + execution. Returns `true` when the
    /// layer's experts and combine were encoded into a fresh, still-open
    /// command buffer (assigned into `cmd`/`enc`); `false` leaves all
    /// state untouched for the callback arm.
    ///
    /// Preconditions (any miss → `false`):
    /// - pure-MoE layer (no dense branch — the combine kernel writes
    ///   `new_h = h_post_attn + Σ`, which would drop a dense contribution),
    /// - identity combine class: `MoePostExpertNormPolicy::None`, no
    ///   combined-output norm, layer_scalar ∈ {0, 1} (both no-ops in
    ///   `apply_outer_combine`, which this path skips),
    /// - no diagnostic captures for this layer (they read `new_h` on the
    ///   CPU before the deferred commit would produce it),
    /// - every selected expert's bytes resolve into ONE registered region
    ///   per projection with u32-expressible offsets.
    #[allow(clippy::too_many_arguments)]
    fn try_inline_zero_copy_moe(
        &self,
        layer: &FullPipelineLayer<'_>,
        ctx: &MoeInterleaveCtx<'_>,
        bufs: &MoeInterleaveBufs<'_>,
        ictx: &InlineMoeCtx<'_>,
        h_post_attn: &[f32],
        cmd: &mut CommandBuffer,
        enc: &mut ComputeCommandEncoder,
        encoder_ended: &mut bool,
    ) -> bool {
        use larql_compute::cpu::ops::moe::{
            moe_expert_input, moe_route_from_router_input, moe_router_input,
        };
        let Some(moe) = layer.moe.as_ref() else {
            return false;
        };
        if layer.ffn_is_remote
            || ctx.defer_ffn_for_split
            || ctx.stage_timing_split
            || layer.has_dense_ffn()
            || !matches!(
                moe.routing_policy.post_expert_norm,
                larql_compute::MoePostExpertNormPolicy::None
            )
            || layer.moe_combined_output_norm
            || !(layer.layer_scalar == 0.0 || layer.layer_scalar == 1.0)
            || ctx.layer_in_snapshot.is_some()
            || ctx.dump_l0_dir.is_some()
        {
            return false;
        }
        // Same biased-Gated refusal as the staged dispatcher: a Gated
        // layer with expert biases has no kernel; refusing INTO the
        // callback arm keeps that path's loud assert authoritative.
        let biased_gated_servable =
            matches!(moe.gate_rule, larql_compute::MoeGateRule::ClampedGlu { .. })
                || (moe.experts_gate_up_bias.is_empty() && moe.experts_down_bias.is_empty());
        if !biased_gated_servable {
            return false;
        }
        let scratch = ictx.scratch;
        if moe.top_k != scratch.top_k
            || moe.intermediate_size != scratch.inter
            || ctx.hidden != scratch.hidden
            || moe.expert_data_format != scratch.format
            || moe.gate_up_cols(ctx.hidden) != scratch.weight_cols
        {
            return false;
        }

        // CPU routing — identical helpers to the staged dispatcher.
        let expert_input = moe_expert_input(h_post_attn, moe, 0.0, ictx.eps);
        let router_in = moe_router_input(h_post_attn, &expert_input, moe, 0.0, ictx.eps);
        let (expert_indices, expert_weights) = moe_route_from_router_input(&router_in, moe);

        let Some(resolved) =
            self.resolve_selected_experts(scratch, &expert_indices, &expert_weights, |ei| {
                Some((
                    moe.experts_gate_up.get(ei).copied()?,
                    moe.experts_down.get(ei).copied()?,
                ))
            })
        else {
            return false;
        };

        // Fresh command buffer; encode experts + combine; leave it OPEN —
        // the caller's loop encodes the next layer's attention into it and
        // the next interleave (or the token epilogue) commits it.
        *cmd = self.queue.new_command_buffer().to_owned();
        *enc = cmd.new_compute_command_encoder().to_owned();
        self.encode_experts_and_combine_zero_copy(
            enc,
            &expert_input,
            moe,
            scratch,
            &resolved,
            bufs.h_post_attn,
            bufs.new_h,
        );
        *encoder_ended = false;
        true
    }
}
