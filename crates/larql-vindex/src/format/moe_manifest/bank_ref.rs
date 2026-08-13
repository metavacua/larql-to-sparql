//! Manifest references to expert banks (format spec §8.1).
//!
//! A bank reference binds a storage location and an expert count to a
//! programme. It is the **only** place that binding exists — LYRW files carry
//! no programme identity (§6.2), so "the binary says one thing, the manifest
//! says another" is unrepresentable rather than merely discouraged.

use serde::{Deserialize, Serialize};

use super::programme::Programme;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// An expert's own operand dimensions.
///
/// For a latent bank these are the latent width, not the residual width — the
/// distinction that makes K3's 3584/3072/3584 correct where 7168 would be
/// wrong (§6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertDims {
    pub input: u32,
    pub intermediate: u32,
    pub output: u32,
}

/// One bank of experts, named by the manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BankRef {
    /// Expert count in this bank. A dense layer is the `1` case.
    pub experts: u32,
    /// Programme name as written in the manifest; resolved via [`Programme`].
    pub programme: String,
    /// Storage stem — `routed/layer_012`, extended with `.seg{N}` per segment.
    pub storage: String,
    #[serde(default)]
    pub expert_dims: Option<ExpertDims>,
}

impl BankRef {
    /// Resolve the declared programme, or `None` if this binary does not know
    /// it. An unknown programme is a clean refusal, never a guess (§V2-0).
    pub fn resolve_programme(&self) -> Option<Programme> {
        Programme::from_name(&self.programme)
    }

    /// Whether this bank's experts operate in a latent space and therefore
    /// require the layer's `routed_input` / `routed_output` transforms.
    pub fn needs_latent_transforms(&self) -> bool {
        self.resolve_programme()
            .is_some_and(|p| p.operates_in_latent_space())
    }

    /// Walkable features this bank contributes (§15.1): every expert's
    /// intermediate rows, across the whole bank, with no router involved.
    ///
    /// `None` when dims were not declared — the count is unknowable rather
    /// than zero, and reporting zero would read as "nothing to browse".
    pub fn walkable_feature_count(&self) -> Option<u64> {
        self.expert_dims
            .map(|d| u64::from(self.experts) * u64::from(d.intermediate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k3_routed() -> BankRef {
        BankRef {
            experts: 896,
            programme: "latent-moe-v1".into(),
            storage: "routed/layer_012".into(),
            expert_dims: Some(ExpertDims {
                input: 3_584,
                intermediate: 3_072,
                output: 3_584,
            }),
        }
    }

    fn shared() -> BankRef {
        BankRef {
            experts: 2,
            programme: "gated-mlp-v1".into(),
            storage: "shared/layer_012".into(),
            expert_dims: None,
        }
    }

    #[test]
    fn a_bank_round_trips_through_json() {
        let json = serde_json::to_string(&k3_routed()).unwrap();
        assert_eq!(serde_json::from_str::<BankRef>(&json).unwrap(), k3_routed());
    }

    #[test]
    fn absent_dims_deserialise_to_none() {
        let json = r#"{"experts": 2, "programme": "gated-mlp-v1", "storage": "shared/layer_0"}"#;
        let b: BankRef = serde_json::from_str(json).unwrap();
        assert_eq!(b.expert_dims, None);
    }

    #[test]
    fn a_registered_programme_resolves() {
        assert_eq!(
            k3_routed().resolve_programme(),
            Some(Programme::LatentMoeV1)
        );
    }

    #[test]
    fn an_unregistered_programme_resolves_to_none_not_a_default() {
        let b = BankRef {
            programme: "mixtral-expert-v9".into(),
            ..shared()
        };
        assert_eq!(b.resolve_programme(), None);
        // And must not be silently treated as latent or as anything else.
        assert!(!b.needs_latent_transforms());
    }

    #[test]
    fn only_a_latent_bank_needs_the_transforms() {
        assert!(k3_routed().needs_latent_transforms());
        assert!(!shared().needs_latent_transforms());
    }

    #[test]
    fn latent_dims_are_the_experts_own_width_not_the_residual_width() {
        // The §6.2 trap: K3's latent bank is 3584, never 7168.
        let dims = k3_routed().expert_dims.unwrap();
        assert_eq!(dims.input, 3_584);
        assert_eq!(dims.output, 3_584);
    }

    #[test]
    fn walkable_features_span_the_whole_bank() {
        assert_eq!(k3_routed().walkable_feature_count(), Some(2_752_512));
    }

    #[test]
    fn undeclared_dims_make_the_feature_count_unknown_not_zero() {
        // Zero would read as "nothing to browse", which is a different claim.
        assert_eq!(shared().walkable_feature_count(), None);
    }

    #[test]
    fn a_dense_layer_is_the_single_expert_case() {
        let dense = BankRef {
            experts: 1,
            programme: "gated-mlp-v1".into(),
            storage: "dense/layer_0".into(),
            expert_dims: Some(ExpertDims {
                input: 2_304,
                intermediate: 9_216,
                output: 2_304,
            }),
        };
        assert_eq!(dense.walkable_feature_count(), Some(9_216));
        assert!(!dense.needs_latent_transforms());
    }
}
