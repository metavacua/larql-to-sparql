//! Split-scale MXFP4 expert dispatch — the arm that binds two streams.
//!
//! Every other grouped-expert format in this backend keeps its scales inside
//! the weight stream and selects a fused half by adding a byte offset. MXFP4
//! stored natively does neither: the e8m0 exponents are a stream of their own
//! with their own placement, and the fused gate/up rows may be arranged the
//! way the checkpoint ships them rather than the way larql's extraction path
//! rewrites them. Both facts need their own binding, so the encoding lives
//! here rather than as two more arms in `moe_zero_copy`'s match.
//!
//! ## What the kernel is told, and why each thing is told rather than derived
//!
//! | binding | why not derived |
//! |---|---|
//! | `s_offsets` | the exponent region's placement is the container writer's, not `payload/16` |
//! | `ROWBASE`/`ROWSTRIDE` | a byte offset can only express contiguous halves |
//!
//! Both derivations were available and both would have been silent when
//! wrong, which is the reason they are bindings instead.

use metal::*;
use std::ffi::c_void;

use super::ResolvedExpert;
use crate::kernels::quant::ExpertScaleBinding;
use crate::moe_dispatch::MoeScratch;
use crate::shaders::mxfp4_grouped_experts::{ROW_BASE_IDENTITY, ROW_STRIDE_IDENTITY};
use crate::MetalBackend;
use larql_compute::MoeLayerWeights;
use larql_models::quant::mxfp4::FusedHalf;

/// The two halves of a fused gate/up region, in output order.
const FUSED_HALVES: [FusedHalf; 2] = [FusedHalf::Gate, FusedHalf::Up];

/// Every selected expert shares one input vector, so the kernel's per-slot
/// input stride is zero. (The down projection uses its own non-zero stride —
/// each slot reads the activation its own gate/up produced.)
const XSTRIDE_SHARED: u32 = 0;

impl MetalBackend {
    /// Gate and up for every selected expert, one 2-D dispatch per half.
    ///
    /// # Panics
    /// If the configured MXFP4 arm is an interleaved-scale one. Those arms
    /// keep the shared inline-scale arity, so they carry neither a scale
    /// offset table nor a row walk — they cannot serve a stored bank, and
    /// dispatching one here would read exponents out of the activation
    /// buffer.
    pub(super) fn encode_mxfp4_gate_up(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
        resolved: &[ResolvedExpert],
    ) {
        let (kh, binding) = self.quant.grouped_experts_for(scratch.format);
        assert_eq!(
            binding,
            ExpertScaleBinding::SplitE8M0,
            "moe_zero_copy_mxfp4: the selected MXFP4 arm binds its scales \
             inline, so it can express neither the container's scale \
             placement nor a {:?} row arrangement; select the split arm",
            moe.fused_row_layout,
        );

        let inter = scratch.inter;
        let n_rows = inter as u32;
        let k_cols = scratch.weight_cols as u32;
        let row_tiles = (inter as u64).div_ceil(kh.rows_per_tg);

        let payload_base = &resolved[0].gate_up.0;
        let scale_base = &Self::scale_base(resolved, |r| r.gate_up_scales.as_ref(), "gate_up");
        let offsets: Vec<u32> = resolved.iter().map(|r| r.gate_up.1 as u32).collect();
        let s_offsets: Vec<u32> = Self::scale_offsets(resolved, |r| r.gate_up_scales.as_ref());

        for half in FUSED_HALVES {
            // The half is selected by WHICH ROWS the kernel walks, not by
            // shifting the base offset — that is the whole point of the row
            // walk, and it is what lets one offset table serve both halves.
            let (row_base, row_stride) = moe.fused_row_layout.row_walk(half, inter);
            let (row_base, row_stride) = (row_base as u32, row_stride as u32);
            let out_buf = match half {
                FusedHalf::Gate => &scratch.g_out,
                FusedHalf::Up => &scratch.u_out,
            };
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(payload_base), 0);
            set_u32_table(enc, 1, &offsets);
            enc.set_buffer(2, Some(scale_base), 0);
            set_u32_table(enc, 3, &s_offsets);
            enc.set_buffer(4, Some(&scratch.x_buf), 0);
            enc.set_buffer(5, Some(out_buf), 0);
            set_u32(enc, 6, &n_rows);
            set_u32(enc, 7, &k_cols);
            set_u32(enc, 8, &XSTRIDE_SHARED);
            set_u32(enc, 9, &row_base);
            set_u32(enc, 10, &row_stride);
            enc.dispatch_thread_groups(
                MTLSize::new(row_tiles, resolved.len() as u64, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        }
    }

    /// The down projection for every selected expert, one 2-D dispatch.
    ///
    /// Down is not a fused operand — there are no halves to choose between —
    /// so its row walk is the identity. It still needs `s_offsets`: the
    /// exponent placement question is about the container, not about fusion.
    ///
    /// # Panics
    /// As [`Self::encode_mxfp4_gate_up`].
    pub(super) fn encode_mxfp4_down(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        scratch: &MoeScratch,
        resolved: &[ResolvedExpert],
    ) {
        let (kh, binding) = self.quant.grouped_experts_for(scratch.format);
        assert_eq!(
            binding,
            ExpertScaleBinding::SplitE8M0,
            "moe_zero_copy_mxfp4: the selected MXFP4 arm binds its scales \
             inline and cannot address the container's exponent stream",
        );

        let n_out = scratch.hidden as u32;
        // Each slot contracts over its own activation row, written strided to
        // `inter_padded` by the activation stage.
        let k_in = scratch.inter_padded as u32;
        let xstride_own = scratch.inter_padded as u32;
        let row_tiles = (scratch.hidden as u64).div_ceil(kh.rows_per_tg);

        let payload_base = &resolved[0].down.0;
        let scale_base = &Self::scale_base(resolved, |r| r.down_scales.as_ref(), "down");
        let offsets: Vec<u32> = resolved.iter().map(|r| r.down.1 as u32).collect();
        let s_offsets: Vec<u32> = Self::scale_offsets(resolved, |r| r.down_scales.as_ref());

        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(payload_base), 0);
        set_u32_table(enc, 1, &offsets);
        enc.set_buffer(2, Some(scale_base), 0);
        set_u32_table(enc, 3, &s_offsets);
        enc.set_buffer(4, Some(&scratch.act_buf), 0);
        enc.set_buffer(5, Some(&scratch.expert_outs), 0);
        set_u32(enc, 6, &n_out);
        set_u32(enc, 7, &k_in);
        set_u32(enc, 8, &xstride_own);
        set_u32(enc, 9, &ROW_BASE_IDENTITY);
        set_u32(enc, 10, &ROW_STRIDE_IDENTITY);
        enc.dispatch_thread_groups(
            MTLSize::new(row_tiles, resolved.len() as u64, 1),
            MTLSize::new(kh.threads_per_tg, 1, 1),
        );
    }

    /// The one buffer every selected expert's exponents live in.
    ///
    /// # Panics
    /// If any expert lacks a scale binding, or if they do not share a base.
    /// The grouped kernel addresses through a single base plus an offset
    /// table, so a second base has no representation here — and a bank that
    /// reached this arm without exponents is a split-scale format with no
    /// scales, which is what the resolution step exists to prevent.
    fn scale_base<'r>(
        resolved: &'r [ResolvedExpert],
        pick: impl Fn(&'r ResolvedExpert) -> Option<&'r (Buffer, u64)>,
        what: &str,
    ) -> Buffer {
        let first = pick(&resolved[0])
            .unwrap_or_else(|| {
                panic!(
                    "moe_zero_copy_mxfp4: expert {} has no {what} exponent \
                     stream, but MXFP4 is a split-scale format",
                    resolved[0].expert_id
                )
            })
            .0
            .clone();
        for r in resolved {
            let b = pick(r).unwrap_or_else(|| {
                panic!(
                    "moe_zero_copy_mxfp4: expert {} has no {what} exponent \
                     stream, but MXFP4 is a split-scale format",
                    r.expert_id
                )
            });
            assert_eq!(
                b.0.gpu_address(),
                first.gpu_address(),
                "moe_zero_copy_mxfp4: {what} exponents span more than one \
                 buffer; the grouped kernel addresses through one base"
            );
        }
        first
    }

    /// Byte offsets of each selected expert's exponent stream within
    /// [`Self::scale_base`]'s buffer.
    fn scale_offsets<'r>(
        resolved: &'r [ResolvedExpert],
        pick: impl Fn(&'r ResolvedExpert) -> Option<&'r (Buffer, u64)>,
    ) -> Vec<u32> {
        resolved
            .iter()
            .map(|r| {
                let off = pick(r).expect("scale_base already proved presence").1;
                u32::try_from(off).expect(
                    "moe_zero_copy_mxfp4: exponent offset exceeds 4 GiB, which \
                     the u32 offset table cannot carry",
                )
            })
            .collect()
    }
}

/// Width of one binding-table entry, taken from the type the shader declares
/// (`constant uint&` / `device const uint*`) rather than spelled as 4.
const U32_BYTES: u64 = std::mem::size_of::<u32>() as u64;

/// Bind a `u32` as an inline constant.
fn set_u32(enc: &metal::ComputeCommandEncoderRef, slot: u64, v: &u32) {
    enc.set_bytes(slot, U32_BYTES, v as *const u32 as *const c_void);
}

/// Bind a `u32` table as an inline constant — the offset tables are
/// per-dispatch and must not enter the address-keyed weight cache.
fn set_u32_table(enc: &metal::ComputeCommandEncoderRef, slot: u64, v: &[u32]) {
    // Every table this helper binds (payload offsets, e8m0 offsets) is
    // route-dependent — the witness must see it or the descriptor path's
    // "zero" claim has a blind spot (caught by rung G's positive control).
    crate::route_witness::bump(&crate::route_witness::OFFSET_BINDS);
    enc.set_bytes(
        slot,
        v.len() as u64 * U32_BYTES,
        v.as_ptr() as *const c_void,
    );
}

#[cfg(test)]
mod tests;
