//! Attention-sink bindings, shared by every sink-aware Metal kernel.
//!
//! Three kernels take sinks and all three bind them the same way: a
//! `constant float*` of per-head sink logits, immediately followed by a
//! `constant uint&` flag saying whether to read it.
//!
//! - `fused_attention`         — prefill, bound by [`super::attention::encode`]
//! - `attn_fused`              — decode, bound by `decode::encode_attn`
//! - `kv_append_attend_fused`  — decode, bound by `decode::encode_attn`
//!
//! This module owns that convention so the placeholder, the length
//! check, and the pair of `set_bytes` calls exist in exactly one place.
//! The CPU-side maths this mirrors lives in `larql_compute::attention::softmax`.

use metal::ComputeCommandEncoderRef;
use std::ffi::c_void;

use super::SCALAR_BYTES;

/// Placeholder bound to the sinks slot when the architecture has none.
///
/// Metal has no null buffer binding, so the slot always carries a real
/// allocation and [`HAS_SINKS`]/[`NO_SINKS`] decides whether the shader
/// reads it.
const NO_SINKS_PLACEHOLDER: [f32; 1] = [0.0];

/// `has_sinks` value telling the shader to read the sinks slot.
const HAS_SINKS: u32 = 1;

/// `has_sinks` value telling the shader the slot is a placeholder.
const NO_SINKS: u32 = 0;

/// Slots offset from the sinks pointer to its `has_sinks` flag. Every
/// sink-aware kernel signature declares the two consecutively.
const FLAG_SLOT_OFFSET: u64 = 1;

/// Resolve the `(sinks, has_sinks)` pair a sink-aware kernel binds.
///
/// # Panics
///
/// If the sink vector's length differs from the query-head count — the
/// tensor does not describe this model, and padding it would produce a
/// plausible-looking wrong forward pass.
fn resolve(sinks: Option<&[f32]>, num_q_heads: usize) -> (&[f32], u32) {
    match sinks {
        Some(s) => {
            assert_eq!(
                s.len(),
                num_q_heads,
                "attention-sink vector has {} entries but the layer has {num_q_heads} \
                 query heads — the extracted tensor does not match this model",
                s.len()
            );
            (s, HAS_SINKS)
        }
        None => (&NO_SINKS_PLACEHOLDER, NO_SINKS),
    }
}

/// Bind the sink pair into `enc`, starting at `sinks_index`.
///
/// The flag goes into `sinks_index + 1` — see [`FLAG_SLOT_OFFSET`].
/// Architectures without sinks pass `None` and get the placeholder plus
/// a zero flag, which every sink-aware shader treats as an ordinary
/// softmax.
pub(crate) fn bind(
    enc: &ComputeCommandEncoderRef,
    sinks_index: u64,
    sinks: Option<&[f32]>,
    num_q_heads: usize,
) {
    let (vals, has_sinks) = resolve(sinks, num_q_heads);
    enc.set_bytes(
        sinks_index,
        std::mem::size_of_val(vals) as u64,
        vals.as_ptr() as *const c_void,
    );
    enc.set_bytes(
        sinks_index + FLAG_SLOT_OFFSET,
        SCALAR_BYTES,
        &has_sinks as *const u32 as *const c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_sinks_bind_the_placeholder_and_a_zero_flag() {
        let (vals, flag) = resolve(None, 64);
        assert_eq!(flag, NO_SINKS);
        assert_eq!(vals, &NO_SINKS_PLACEHOLDER);
    }

    #[test]
    fn the_placeholder_is_never_empty() {
        // A zero-length `set_bytes` is invalid in Metal, so the
        // no-sinks path must still hand over a real allocation.
        let (vals, _) = resolve(None, 8);
        assert!(!vals.is_empty());
    }

    #[test]
    fn present_sinks_pass_through_with_the_flag_set() {
        let sinks = [-2.45f32, 0.0, 2.45, 8.19];
        let (vals, flag) = resolve(Some(&sinks), 4);
        assert_eq!(flag, HAS_SINKS);
        assert_eq!(vals, &sinks);
    }

    #[test]
    #[should_panic(expected = "does not match this model")]
    fn a_short_sink_vector_panics() {
        resolve(Some(&[0.0, 1.0]), 64);
    }

    #[test]
    #[should_panic(expected = "does not match this model")]
    fn a_long_sink_vector_panics() {
        resolve(Some(&[0.0; 128]), 64);
    }

    #[test]
    fn the_flag_slot_follows_the_pointer_slot() {
        // Pins the convention the three kernel signatures share; a
        // shader that separates the two would break silently otherwise.
        assert_eq!(FLAG_SLOT_OFFSET, 1);
    }
}
