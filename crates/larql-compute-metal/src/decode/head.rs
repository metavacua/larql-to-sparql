//! The decode-time LM head, encoded into the decode command buffer.
//!
//! TOKEN-B1 rung 2. Rung 1 fused the lm_head Q4_K matvec with the partial
//! top-K reduction, which removed the full-vocab readback and the CPU scan
//! but left the *boundary*: the decode command buffer committed and waited,
//! the hidden state came back to the host, the host applied the final norm,
//! and a second command buffer carried the head. Two submissions and two
//! waits per token.
//!
//! ```text
//! rung 1                              rung 2
//!   decode CB                           decode CB
//!     transformer                         transformer
//!   commit + wait                         final norm
//!   read hidden      (host)               lm_head matvec
//!   final norm       (host)               partial top-K
//!   lm_head CB                          commit + wait
//!   commit + wait                       read partials
//! ```
//!
//! What crosses the host boundary afterwards is `num_tgs × K_TOPK`
//! (val, idx) pairs, never the hidden state and never the logits.
//!
//! The head refuses rather than approximates. Every precondition it cannot
//! honour — a norm it does not implement, a store whose geometry it cannot
//! explain, a `top_k` wider than one partial row — returns `None`, and the
//! caller runs the unfused path, which stays the reference this is pinned
//! against (see `head/tests.rs`).
//!
//! # Which decode paths it can actually ride
//!
//! The head needs the decode's encoder still OPEN at the end of the token,
//! because that is what "rides the same command buffer" means. Two paths
//! differ here, and the difference is not a bug in either:
//!
//! - **GPU-route / merged-CB** (`LARQL_GPU_ROUTE=1`, the production
//!   gpt-oss path): the MoE layer encodes into the still-open buffer and
//!   leaves it open, so the head rides it and the token costs one wait.
//! - **Legacy MoE interleave**: `handle_moe_interleave` commits at each
//!   layer boundary and only re-opens an encoder when another layer
//!   follows, so after the final layer the buffer is already committed.
//!   The head then encodes nothing and returns `None`, and the caller runs
//!   the unfused path — correct, just not faster.
//!
//! So a `None` here is not necessarily a malformed plan; on the legacy
//! path it is the expected answer. Anything reading a missing speedup as a
//! broken head should check which arm ran first.

use larql_compute::DecodeHeadPlan;
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use crate::MetalBackend;

/// Q4_K super-block geometry, from the store's own definition — the head
/// must not invent a stride the writer did not use.
use larql_models::quant::ggml::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};

/// A caller's request to run the head inside the decode command buffer,
/// plus the slot its result lands in.
///
/// An out-slot rather than a changed return type: `decode_token_with_moe_split_fn`
/// has several callers that want the hidden state, and widening its return
/// for one of them would make every other call site describe a decision it
/// does not make.
pub struct HeadRequest<'a, 'p> {
    /// What to run. See [`DecodeHeadPlan`].
    pub plan: &'a DecodeHeadPlan<'p>,
    /// Filled with the reduced top-K when the head actually ran. Left
    /// `None` when any precondition refused, which is the caller's signal
    /// to use the returned hidden state and run the unfused head.
    pub out: &'a mut Option<Vec<(u32, f32)>>,
}

/// The GPU-side results of an encoded head, read after the single commit.
pub(super) struct HeadBuffers {
    partial_vals: Buffer,
    partial_idxs: Buffer,
    /// Held only so it can be returned to the pool — nothing reads it on
    /// the host. Without this the head allocates a fresh full-vocab
    /// buffer (804 KB on gpt-oss) on every single token, because
    /// `BufferCache::output` can only hand back what something recycled.
    scores: Buffer,
    norm_out: Buffer,
    topk_tgs: usize,
    top_k: usize,
}

impl HeadBuffers {
    /// Reduce the partials to the caller's `top_k`, then return every
    /// scratch buffer to the pool.
    ///
    /// Must be called after the command buffer has been waited on — that
    /// is the `ScratchGuard` invariant, and it holds here because the only
    /// call site sits below the final `wait_until_completed`.
    pub(super) fn reduce_and_recycle(self, bufs: &crate::buffers::BufferCache) -> Vec<(u32, f32)> {
        let hits = MetalBackend::reduce_topk_partial(
            &self.partial_vals,
            &self.partial_idxs,
            self.topk_tgs,
            self.top_k,
        );
        bufs.recycle(self.partial_vals);
        bufs.recycle(self.partial_idxs);
        bufs.recycle(self.scores);
        bufs.recycle(self.norm_out);
        hits
    }
}

impl MetalBackend {
    /// Encode final norm → lm_head Q4_K matvec → partial top-K onto `enc`,
    /// reading the decode's own `h_buf`. `None` when the plan states
    /// something this head does not implement, leaving `enc` untouched.
    ///
    /// Called with the decode's encoder still open, so it adds no
    /// submission of its own — that is the entire point of the rung.
    pub(super) fn encode_decode_head(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_buf: &Buffer,
        hidden: usize,
        plan: &DecodeHeadPlan<'_>,
    ) -> Option<HeadBuffers> {
        // ── Preconditions, each a refusal rather than a fallback ──
        //
        // Only RMS norm is implemented here. LayerNorm's mean-subtraction
        // and optional bias are a different kernel with a different
        // binding table; silently applying RMS to a LayerNorm model would
        // produce a plausible-looking distribution over the wrong logits.
        if plan.norm_type != larql_compute::NormType::RmsNorm {
            return None;
        }
        if plan.final_norm_weight.len() != hidden || hidden == 0 {
            return None;
        }
        if plan.vocab == 0 || plan.cols < hidden {
            return None;
        }
        // The partial reduction emits one row of `K_TOPK` per threadgroup;
        // a wider request cannot be served from it.
        if plan.top_k == 0 || plan.top_k > crate::shaders::f32_gemv::K_TOPK {
            return None;
        }
        // The store must divide into whole super-blocks at exactly the
        // stated row width. A store this function cannot explain is not
        // read at a guessed stride — same contract as `q4k_row_query`,
        // which is the CPU-side authority on this geometry.
        if !plan.cols.is_multiple_of(Q4_K_BLOCK_ELEMS) {
            return None;
        }
        let row_bytes = plan.cols / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES;
        if plan.lm_head_q4k.len() != plan.vocab.checked_mul(row_bytes)? {
            return None;
        }

        // ── Final norm: h_buf[hidden] → norm_out[cols] ──
        //
        // `norm_out` is `cols` wide because the matvec reads a padded row.
        // The norm kernel writes exactly `[0, hidden)`, so the tail carries
        // whatever the pooled scratch last held — zero it, or the padding
        // contributes garbage to every logit. Only the tail needs clearing;
        // the prefix is fully overwritten. For a 2880-hidden model padded
        // to 3072 that is 192 floats.
        let norm_w = self.bufs.get_f32(plan.final_norm_weight);
        let norm_out = self.bufs.output((plan.cols * 4) as u64);
        if plan.cols > hidden {
            unsafe {
                let tail = (norm_out.contents() as *mut f32).add(hidden);
                std::ptr::write_bytes(tail, 0, plan.cols - hidden);
            }
        }
        crate::ops::full_pipeline::encode_rms_norm(
            enc,
            &self.norms.rms_norm_pipeline,
            h_buf,
            &norm_w,
            &norm_out,
            hidden,
            plan.norm_eps,
            plan.norm_offset,
        );

        // ── lm_head matvec: scores[vocab] = W[vocab, cols] · norm_out ──
        //
        // Geometry comes from the bound pipeline rather than a constant,
        // for the reason `q4k_matvec` documents: a ROWS_PER_TG that
        // disagrees with the shader silently drops or double-counts rows.
        let w = self.bufs.get_bytes(plan.lm_head_q4k);
        let scores = self.bufs.output((plan.vocab * 4) as u64);
        let n = plan.vocab as u32;
        let k = plan.cols as u32;
        let rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let num_tgs = (plan.vocab as u64).div_ceil(rows_per_tg);

        enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
        enc.set_buffer(0, Some(&w), 0);
        enc.set_buffer(1, Some(&norm_out), 0);
        enc.set_buffer(2, Some(&scores), 0);
        enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );

        // ── Partial top-K over the scores, still GPU-side ──
        let (partial_vals, partial_idxs, topk_tgs) =
            self.encode_topk_partial(enc, &scores, plan.vocab);

        Some(HeadBuffers {
            partial_vals,
            partial_idxs,
            scores,
            norm_out,
            topk_tgs,
            top_k: plan.top_k,
        })
    }
}

#[cfg(test)]
mod tests;
