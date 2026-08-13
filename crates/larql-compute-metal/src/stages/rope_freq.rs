//! RoPE frequency bindings, shared by every rope-bearing Metal kernel.
//!
//! Four kernels rotate Q/K and all four bind the same pair: a `constant
//! float*` of per-pair inverse frequencies and a `constant float&` amplitude
//! on `cos`/`sin`.
//!
//! - `rope_at_pos` / `rope_apply` / `rope_at_pos_batched` /
//!   `rope_at_pos_batched_qk` — [`super::rope`] and the decode encoders
//! - `qk_norm_rope_fused` — decode, bound by `decode::encode_attn`
//! - `attn_fused` — decode, bound by `decode::encode_attn`
//! - `fused_attention` — prefill, bound by [`super::attention::encode`]
//!
//! **Why the kernels no longer see `rope_base`.** Each used to derive
//! `1/base^(2d/rdim)` in-shader, which silently discarded every scaling
//! family a checkpoint might declare — llama3 bands, YaRN's ramp and
//! amplitude, and Gemma 3's linear position divisor — while Metal *prefill*
//! honoured all three because it ropes on the host. The same model therefore
//! encoded its prompt under one rule and generated under another, on
//! `gemma-3-4b-it` among others. See `docs/k3-funnel.md` §4.10.
//!
//! The frequencies are now computed once by
//! `larql_compute::attention::rope::rope_freq_plan` — the same function the
//! CPU path calls — and this module owns the binding convention so the two
//! `set_bytes` calls exist in exactly one place.

use metal::ComputeCommandEncoderRef;
use std::ffi::c_void;

use larql_compute::attention::rope::RopeFreqPlan;

use super::SCALAR_BYTES;

/// Largest frequency table `set_bytes` may carry, in floats.
///
/// Metal's `setBytes:length:atIndex:` is documented for small inline data
/// (4 KB). A table is `rotary_dim / 2` entries and the widest head in the
/// support table is 512, so 256 floats (1 KB) is the realistic ceiling; this
/// bound exists so an unexpected geometry fails loudly rather than silently
/// truncating the rotation.
pub(crate) const MAX_INV_FREQ_ENTRIES: usize = 1024;

/// Rotary pairs a layer rotates, from the same fields the kernels read.
///
/// `rotary_dim == 0` means the full `head_dim`, matching
/// `FullPipelineLayer::rotary_dim` and every rope kernel's own
/// `(rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim)`.
pub(crate) fn expected_pairs(head_dim: usize, rotary_dim: usize) -> usize {
    let rdim = if rotary_dim == 0 {
        head_dim
    } else {
        rotary_dim.min(head_dim)
    };
    rdim.max(2) / 2
}

/// Bind a layer's `(inv_freq, amplitude)` pair.
///
/// Both indices are explicit because, unlike the sinks pair, these two are
/// *not* adjacent: the frequency table took over the old `rope_base` slot in
/// place, while the amplitude was appended at the end of each signature so no
/// other buffer index had to move.
///
/// # Panics
///
/// If the table is empty or implausibly large — either means the plan does
/// not describe this layer's geometry, and rotating against the wrong table
/// produces a plausible-looking wrong forward pass rather than an error.
pub(crate) fn bind(
    enc: &ComputeCommandEncoderRef,
    inv_freq_index: u64,
    amplitude_index: u64,
    plan: &RopeFreqPlan,
    head_dim: usize,
    rotary_dim: usize,
) {
    let inv_freq = plan.inv_freq_f32();
    let expected = expected_pairs(head_dim, rotary_dim);
    assert_eq!(
        inv_freq.len(),
        expected,
        "RoPE frequency table has {} entries but this layer rotates {expected} pairs \
         (head_dim {head_dim}, rotary_dim {rotary_dim}) — the plan does not describe \
         this layer. A shorter table makes the kernel rotate against the wrong \
         frequencies, which looks like a plausible forward pass rather than an error.",
        inv_freq.len(),
    );
    assert!(
        inv_freq.len() <= MAX_INV_FREQ_ENTRIES,
        "RoPE frequency table has {} entries, above the {MAX_INV_FREQ_ENTRIES} \
         `set_bytes` ceiling",
        inv_freq.len(),
    );
    enc.set_bytes(
        inv_freq_index,
        std::mem::size_of_val(&inv_freq[..]) as u64,
        inv_freq.as_ptr() as *const c_void,
    );
    let amplitude = plan.amplitude as f32;
    enc.set_bytes(
        amplitude_index,
        SCALAR_BYTES,
        &amplitude as *const f32 as *const c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::attention::rope::{rope_freq_plan, RopeFreqScaling};
    use larql_models::YarnRopeScaling;

    /// Gemma 3's global layers divide position by 8. The kernels never see
    /// that divisor, so it has to be inside the table they *do* see —
    /// otherwise Metal decode ropes at 8× the position prefill used, which is
    /// the §4.10 defect.
    #[test]
    fn linear_divisor_is_folded_into_the_table() {
        let plain = rope_freq_plan(256, 1.0, 1_000_000.0, 1.0, RopeFreqScaling::None);
        let divided = rope_freq_plan(256, 1.0, 1_000_000.0, 8.0, RopeFreqScaling::None);
        for (p, d) in plain.inv_freq.iter().zip(&divided.inv_freq) {
            assert!(
                (p / 8.0 - d).abs() < 1e-18,
                "divisor must scale every entry: {p} / 8 != {d}"
            );
        }
    }

    /// YaRN is the only family with an amplitude, and a table alone cannot
    /// express it — so the pair has to travel together.
    #[test]
    fn yarn_carries_an_amplitude_the_table_cannot_express() {
        let yarn = RopeFreqScaling::Yarn(YarnRopeScaling {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 4096.0,
            truncate: false,
            mscale: None,
            mscale_all_dim: None,
        });
        let plan = rope_freq_plan(64, 1.0, 150_000.0, 1.0, yarn);
        assert!((plan.amplitude - 1.346_573_590_279_972_7).abs() < 1e-12);
        let unscaled = rope_freq_plan(64, 1.0, 150_000.0, 1.0, RopeFreqScaling::None);
        assert_eq!(unscaled.amplitude, 1.0);
    }

    /// The bind-time geometry check must agree with the kernels' own rule,
    /// or it would reject correct plans (or accept wrong ones).
    #[test]
    fn expected_pairs_matches_the_kernel_rule() {
        assert_eq!(expected_pairs(256, 0), 128); // 0 = full head_dim
        assert_eq!(expected_pairs(256, 64), 32); // partial rotation
        assert_eq!(expected_pairs(256, 512), 128); // clamped to head_dim
        assert_eq!(expected_pairs(0, 0), 1); // degenerate, matches max(2)/2
    }

    /// The plan builder and the bind-time check must agree for every
    /// geometry, or a correct plan fails to bind.
    #[test]
    fn plan_width_agrees_with_expected_pairs() {
        for (hd, rot) in [(256usize, 0usize), (256, 64), (64, 0), (128, 32)] {
            let frac = if rot == 0 {
                1.0
            } else {
                rot as f64 / hd as f64
            };
            let plan = rope_freq_plan(hd, frac, 10_000.0, 1.0, RopeFreqScaling::None);
            assert_eq!(
                plan.inv_freq.len(),
                expected_pairs(hd, rot),
                "head_dim {hd}, rotary_dim {rot}"
            );
        }
    }

    /// Table width follows the rotary width, not the head width — a partial
    /// rotation (Gemma 4 global at 25 %) must produce a shorter table.
    #[test]
    fn table_width_follows_rotary_width() {
        let full = rope_freq_plan(256, 1.0, 10_000.0, 1.0, RopeFreqScaling::None);
        let quarter = rope_freq_plan(256, 0.25, 10_000.0, 1.0, RopeFreqScaling::None);
        assert_eq!(full.inv_freq.len(), 128);
        assert_eq!(quarter.inv_freq.len(), 32);
        assert!(full.inv_freq.len() <= MAX_INV_FREQ_ENTRIES);
    }
}
