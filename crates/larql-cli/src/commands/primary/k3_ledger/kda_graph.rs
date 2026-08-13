//! Which KDA projections share one input, and can a rotation be folded into them?
//!
//! Pure. A shared-input rotation bundle is only cheap if several large matrices
//! genuinely read the *same* tensor: you rotate the input once and fold `R^T`
//! into each weight offline, so `(W R^T)(R h) = W h` and nothing downstream ever
//! sees the rotation. The economics are set by how many matrices share the ride.
//!
//! The premise "all five KDA projections consume the same hidden state" is
//! **wrong by one**, and wrong in the expensive direction.
//!
//! ## Traced against `modeling_kimi_linear.py` (the class K3 actually imports)
//!
//! `KimiDecoderLayer` — both the plain path and `_forward_attn_residual`, which
//! is the one K3 takes — computes `input_layernorm(hidden_states)` and passes
//! that single tensor into `self_attn`. AttnRes mixes block residuals *before*
//! the norm, so it does not change the module's input contract.
//!
//! Inside `KimiDeltaAttention::forward`, six tensors read that hidden state:
//!
//! | tensor | shape | note |
//! |---|---|---|
//! | `q_proj` | `[12288, 7168]` | full rank |
//! | `k_proj` | `[12288, 7168]` | full rank |
//! | `v_proj` | `[12288, 7168]` | full rank |
//! | `g_proj` | `[12288, 7168]` | full rank — `use_full_rank_gate: true` |
//! | `f_a_proj` | `[128, 7168]` | tiny; decay gate, feeds `f_b_proj` |
//! | `b_proj` | `[96, 7168]` | tiny; `beta` |
//!
//! **`o_proj` is not among them.** It reads `o_norm(kernel_output, g)` — a
//! `FusedRMSNormGated` over `head_dim` with a learned per-channel gain and a
//! sigmoid gate. A rotation does not commute with that, so `o_proj` cannot join
//! the bundle; rotating it needs its own runtime transform amortised over one
//! matrix instead of four.
//!
//! So the bundle is **4 of the 5 full-rank matrices, 80% of KDA attention
//! bytes** — attractive, but not the 5-way amortisation the proposal assumed.
//!
//! ## The trap
//!
//! `f_a_proj` and `b_proj` read the same tensor. Rotate `h` and fold `R^T` into
//! only `q/k/v/g` and the decay gate and `beta` silently receive `R h` — a
//! plausible wrong number, not a crash. Both are small enough that folding is
//! free; **forgetting them is the whole cost.** Same failure shape as the
//! grouped-expert `XSTRIDE` bug.
//!
//! Output-side folding is unavailable too: `q/k/v` pass through a depthwise
//! `ShortConvolution` with `silu` after projection, which no per-channel
//! transform commutes with.

use serde::Serialize;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// K3 KDA geometry (`linear_attn_config`), used to derive expected shapes.
pub const HEAD_DIM: u64 = 128;
pub const NUM_HEADS: u64 = 96;
pub const HIDDEN_SIZE: u64 = 7168;
/// `head_dim * num_heads` — the width every full-rank KDA projection emits.
pub const PROJECTION_SIZE: u64 = HEAD_DIM * NUM_HEADS;

/// How a tensor relates to a shared input rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RotationRole {
    /// Reads the post-`input_layernorm` hidden state. Fold `R^T` into it.
    SharedInput,
    /// Reads another projection's output. Unaffected by the input rotation.
    IntermediateInput,
    /// Reads the gated-normed kernel output. Cannot join the bundle.
    GatedOutput,
    /// Not a projection — norm gain, conv kernel, bias or scalar.
    NotAProjection,
}

impl RotationRole {
    /// Must this tensor be rewritten when the shared input is rotated?
    ///
    /// True for every `SharedInput` tensor including the tiny ones. Omitting
    /// `f_a_proj` or `b_proj` is the trap this predicate exists to close.
    pub fn needs_fold(self) -> bool {
        self == Self::SharedInput
    }
}

/// Classify a KDA tensor by its name suffix (the part after `self_attn.`).
pub fn role(suffix: &str) -> RotationRole {
    let base = suffix.strip_suffix(".weight").unwrap_or(suffix);
    match base {
        "q_proj" | "k_proj" | "v_proj" | "g_proj" | "f_a_proj" | "b_proj" => {
            RotationRole::SharedInput
        }
        // g_a/g_b are the low-rank gate arm, dead while use_full_rank_gate holds.
        "f_b_proj" | "g_b_proj" => RotationRole::IntermediateInput,
        "g_a_proj" => RotationRole::SharedInput,
        "o_proj" => RotationRole::GatedOutput,
        _ => RotationRole::NotAProjection,
    }
}

/// One traced tensor.
#[derive(Debug, Clone, Serialize)]
pub struct KdaTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub role: RotationRole,
}

impl KdaTensor {
    pub fn params(&self) -> u64 {
        self.shape.iter().product()
    }
    /// A projection wide enough for the rotation economics to care about.
    pub fn is_full_rank(&self) -> bool {
        self.shape.len() == 2
            && self.shape.contains(&PROJECTION_SIZE)
            && self.shape.contains(&HIDDEN_SIZE)
    }
}

/// What one shared rotation would cover in a KDA layer.
#[derive(Debug, Clone, Serialize)]
pub struct BundleCoverage {
    pub bundled_params: u64,
    pub standalone_params: u64,
    pub bundled_full_rank: usize,
    pub total_full_rank: usize,
    /// Tensors that must be rewritten but are easy to miss (the small ones).
    pub easy_to_miss: Vec<String>,
}

impl BundleCoverage {
    /// Fraction of full-rank projection bytes the one rotation amortises over.
    pub fn byte_coverage(&self) -> f64 {
        let total = self.bundled_params + self.standalone_params;
        if total == 0 {
            return 0.0;
        }
        self.bundled_params as f64 / total as f64
    }
}

/// Summarise a layer's traced tensors into rotation-bundle economics.
pub fn coverage(tensors: &[KdaTensor]) -> BundleCoverage {
    let full: Vec<_> = tensors.iter().filter(|t| t.is_full_rank()).collect();
    BundleCoverage {
        bundled_params: full
            .iter()
            .filter(|t| t.role.needs_fold())
            .map(|t| t.params())
            .sum(),
        standalone_params: full
            .iter()
            .filter(|t| !t.role.needs_fold())
            .map(|t| t.params())
            .sum(),
        bundled_full_rank: full.iter().filter(|t| t.role.needs_fold()).count(),
        total_full_rank: full.len(),
        easy_to_miss: tensors
            .iter()
            .filter(|t| t.role.needs_fold() && !t.is_full_rank())
            .map(|t| t.name.clone())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, shape: &[u64]) -> KdaTensor {
        KdaTensor {
            name: name.into(),
            shape: shape.to_vec(),
            role: role(name),
        }
    }

    /// The real layer-49 tensor table, read from shard 50's safetensors header.
    fn k3_layer() -> Vec<KdaTensor> {
        vec![
            t("q_proj.weight", &[12288, 7168]),
            t("k_proj.weight", &[12288, 7168]),
            t("v_proj.weight", &[12288, 7168]),
            t("g_proj.weight", &[12288, 7168]),
            t("o_proj.weight", &[7168, 12288]),
            t("f_a_proj.weight", &[128, 7168]),
            t("f_b_proj.weight", &[12288, 128]),
            t("b_proj.weight", &[96, 7168]),
            t("o_norm.weight", &[128]),
            t("q_conv1d.weight", &[12288, 1, 4]),
        ]
    }

    #[test]
    fn four_of_the_five_full_rank_projections_share_one_input() {
        // The proposal assumed five. It is four, and the missing one is o_proj.
        let c = coverage(&k3_layer());
        assert_eq!(c.total_full_rank, 5);
        assert_eq!(c.bundled_full_rank, 4);
        assert_eq!(role("o_proj.weight"), RotationRole::GatedOutput);
    }

    #[test]
    fn the_bundle_covers_eighty_percent_of_projection_bytes() {
        let c = coverage(&k3_layer());
        assert!(
            (c.byte_coverage() - 0.8).abs() < 1e-9,
            "{}",
            c.byte_coverage()
        );
    }

    #[test]
    fn the_small_shared_input_readers_are_reported_as_easy_to_miss() {
        // f_a_proj and b_proj must be folded too. Missing them corrupts the
        // decay gate and beta with a plausible wrong number.
        let c = coverage(&k3_layer());
        assert_eq!(c.easy_to_miss.len(), 2);
        assert!(c.easy_to_miss.iter().any(|n| n.starts_with("f_a_proj")));
        assert!(c.easy_to_miss.iter().any(|n| n.starts_with("b_proj")));
    }

    #[test]
    fn every_shared_input_tensor_needs_a_fold_and_nothing_else_does() {
        for name in ["q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"] {
            assert!(role(name).needs_fold(), "{name} reads the hidden state");
        }
        for name in [
            "o_proj", "f_b_proj", "o_norm", "q_conv1d", "A_log", "dt_bias",
        ] {
            assert!(!role(name).needs_fold(), "{name} does not");
        }
    }

    #[test]
    fn the_low_rank_gate_arm_is_classified_even_though_k3_does_not_use_it() {
        // use_full_rank_gate: true, so g_a/g_b are absent from K3's checkpoint —
        // but a sibling checkpoint without that flag must still classify.
        assert_eq!(role("g_a_proj.weight"), RotationRole::SharedInput);
        assert_eq!(role("g_b_proj.weight"), RotationRole::IntermediateInput);
    }

    #[test]
    fn names_classify_with_or_without_the_weight_suffix() {
        assert_eq!(role("q_proj"), role("q_proj.weight"));
    }

    #[test]
    fn the_conv_and_norm_weights_are_not_projections() {
        assert!(!t("q_conv1d.weight", &[12288, 1, 4]).is_full_rank());
        assert!(!t("o_norm.weight", &[128]).is_full_rank());
    }

    #[test]
    fn projection_size_is_head_dim_times_num_heads() {
        assert_eq!(PROJECTION_SIZE, 12288);
    }

    #[test]
    fn an_empty_layer_does_not_divide_by_zero() {
        assert_eq!(coverage(&[]).byte_coverage(), 0.0);
    }
}
