//! HF `layer_types` selector values and the per-layer attention-span rule.
//!
//! A hybrid model interleaves full-attention and sliding-window layers, and
//! modern checkpoints state that interleave explicitly in `config.json` rather
//! than implying it through a stride. `openai/gpt-oss-20b` ships 24 entries
//! alternating `sliding_attention` / `full_attention` at `sliding_window: 128`.
//!
//! **That list was parsed and validated but never consulted**, so every
//! architecture without a hand-written override ran full attention on every
//! layer — the [§4.7.8] shape a fourth time: a config fact with one
//! behavioural default silently answering for everyone. Resolving it here, off
//! the config, means a family that declares `layer_types` is served correctly
//! with no per-architecture code.
//!
//! The defect is invisible below the window: at 85 tokens a 128-wide sliding
//! layer and a full-attention layer compute the identical thing, which is why
//! a Gate-B layer diff taken over a short fixture passed while a 510-token
//! forward was still wrong.
//!
//! [§4.7.8]: ../../../../docs/k3-funnel.md

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
/// A layer attending only to the last `sliding_window` positions.
pub const LAYER_TYPE_SLIDING_ATTENTION: &str = "sliding_attention";

/// A layer attending to the whole prefix.
pub const LAYER_TYPE_FULL_ATTENTION: &str = "full_attention";

/// Whether `layer_types` marks `layer` as sliding-window.
///
/// Returns `None` when the config declares no `layer_types`, leaving the
/// caller's own rule in charge — a stride-based family (Gemma 3) or the
/// all-full default. Out-of-range indices are `None` for the same reason:
/// a list too short to answer must not answer.
pub fn is_sliding_from_layer_types(
    layer_types: Option<&Vec<String>>,
    layer: usize,
) -> Option<bool> {
    let entry = layer_types?.get(layer)?;
    Some(entry.eq_ignore_ascii_case(LAYER_TYPE_SLIDING_ATTENTION))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpt_oss_layer_types() -> Vec<String> {
        (0..24)
            .map(|i| {
                if i % 2 == 0 {
                    LAYER_TYPE_SLIDING_ATTENTION.to_string()
                } else {
                    LAYER_TYPE_FULL_ATTENTION.to_string()
                }
            })
            .collect()
    }

    /// GPT-OSS's actual interleave: even layers slide, odd layers do not.
    #[test]
    fn gpt_oss_interleave_is_read_from_the_list() {
        let lt = gpt_oss_layer_types();
        assert_eq!(is_sliding_from_layer_types(Some(&lt), 0), Some(true));
        assert_eq!(is_sliding_from_layer_types(Some(&lt), 1), Some(false));
        assert_eq!(is_sliding_from_layer_types(Some(&lt), 22), Some(true));
        assert_eq!(is_sliding_from_layer_types(Some(&lt), 23), Some(false));
    }

    /// Half the layers must slide — a rule that answered `false` everywhere
    /// would also pass a test that only checked layer 1.
    #[test]
    fn gpt_oss_interleave_is_half_sliding() {
        let lt = gpt_oss_layer_types();
        let sliding = (0..lt.len())
            .filter(|i| is_sliding_from_layer_types(Some(&lt), *i) == Some(true))
            .count();
        assert_eq!(sliding, 12, "expected 12 of 24 sliding, got {sliding}");
    }

    #[test]
    fn absent_layer_types_defers_to_the_caller() {
        assert_eq!(is_sliding_from_layer_types(None, 0), None);
    }

    /// A short list must not answer for layers it does not cover.
    #[test]
    fn out_of_range_layer_defers_to_the_caller() {
        let lt = vec![LAYER_TYPE_SLIDING_ATTENTION.to_string()];
        assert_eq!(is_sliding_from_layer_types(Some(&lt), 0), Some(true));
        assert_eq!(is_sliding_from_layer_types(Some(&lt), 1), None);
    }

    /// Matching is case-insensitive, as everywhere else config strings are
    /// compared in this crate.
    #[test]
    fn matching_ignores_case() {
        let lt = vec!["Sliding_Attention".to_string()];
        assert_eq!(is_sliding_from_layer_types(Some(&lt), 0), Some(true));
    }

    /// Anything that is not the sliding marker is full attention, including
    /// a value this crate has never seen.
    #[test]
    fn unknown_entry_is_treated_as_full_attention() {
        let lt = vec!["chunked_attention".to_string()];
        assert_eq!(is_sliding_from_layer_types(Some(&lt), 0), Some(false));
    }
}
