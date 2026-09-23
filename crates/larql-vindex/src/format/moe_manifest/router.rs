//! Router descriptor (format spec §8.1, §8.2).
//!
//! The manifest *names* routing semantics; the adapter implements them. The
//! vocabulary here is sized by the conformance envelope, which spans four
//! genuinely different balancing mechanisms:
//!
//! | model | scoring | selection | post-processing |
//! | ----- | ------- | --------- | --------------- |
//! | Direct / Gemma | softmax | top-2/4 | — |
//! | Kimi-Linear-48B | sigmoid | top-8 of 256, grouped | renormalise, scale 2.446 |
//! | Inkling-Small | sigmoid | top-6 of 256 | gate bias, norm-after-top-k, route scale 8.0, global scale, **shared-expert sink** |
//! | K3 | — | top-16 of 896 | quantile-balanced |
//!
//! Every one of those is a combination of the same handful of knobs, which is
//! the claim this type exists to make falsifiable: if a real model needs a
//! knob that is not here, the manifest vocabulary was under-specified and that
//! is an ABI finding, not a post-freeze patch.

use serde::{Deserialize, Serialize};

/// How raw router logits become scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreActivation {
    Softmax,
    Sigmoid,
}

/// How experts are picked from the scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Selection {
    /// Plain top-k over all experts.
    TopK { k: usize },
    /// Top-k within each of `groups` partitions. Kimi-Linear declares one
    /// group, which is degenerate but present in the released schema — the
    /// manifest records what the model says rather than simplifying it away.
    GroupedTopK { k: usize, groups: usize },
}

impl Selection {
    /// Experts activated per token.
    pub const fn experts_per_token(&self) -> usize {
        match self {
            Self::TopK { k } => *k,
            Self::GroupedTopK { k, groups } => *k * *groups,
        }
    }
}

/// Post-selection score handling. Order matters and is fixed: bias is applied
/// to scores **before** selection; normalisation and scaling apply **after**.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RouterPostProcessing {
    /// Renormalise the selected scores to sum to 1. OLMoE deliberately does
    /// *not* — it keeps raw softmax probabilities — so this is opt-in.
    pub renormalise: bool,
    /// Normalise after top-k rather than before (Inkling's `norm_after_topk`).
    pub norm_after_top_k: bool,
    /// Per-expert additive bias applied to scores before selection.
    pub gate_bias: Option<String>,
    /// Constant multiplier on the routed branch's contribution.
    pub route_scale: Option<f64>,
    /// K3's balanced assignment. Named, not described — the adapter owns it.
    pub quantile_balanced: bool,
}

/// A layer's routing description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Router {
    /// Manifest key of the router weight matrix.
    pub scores: String,
    pub activation: ScoreActivation,
    pub selection: Selection,
    #[serde(default)]
    pub post: RouterPostProcessing,
    /// Whether shared experts are scored by the router and participate in
    /// normalisation (Inkling's `shared_expert_sink`).
    ///
    /// This is not cosmetic: with a sink, the shared experts' weights come out
    /// of the same normalised budget as the routed ones, so a reduction that
    /// ignores it silently over-weights the routed branch.
    #[serde(default)]
    pub shared_expert_sink: bool,
}

impl Router {
    /// Whether the router's own output needs no further weighting — true only
    /// for a plain softmax top-k with nothing applied after it.
    pub fn is_plain_softmax_top_k(&self) -> bool {
        self.activation == ScoreActivation::Softmax
            && matches!(self.selection, Selection::TopK { .. })
            && self.post == RouterPostProcessing::default()
            && !self.shared_expert_sink
    }

    /// Names of manifest-addressed tensors this router needs beyond `scores`.
    pub fn auxiliary_tensors(&self) -> Vec<&str> {
        self.post.gate_bias.as_deref().into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemma_style() -> Router {
        Router {
            scores: "layers.0.router.weight".into(),
            activation: ScoreActivation::Softmax,
            selection: Selection::TopK { k: 2 },
            post: RouterPostProcessing::default(),
            shared_expert_sink: false,
        }
    }

    fn kimi_linear_style() -> Router {
        Router {
            scores: "layers.1.router.weight".into(),
            activation: ScoreActivation::Sigmoid,
            selection: Selection::GroupedTopK { k: 8, groups: 1 },
            post: RouterPostProcessing {
                renormalise: true,
                route_scale: Some(2.446),
                ..Default::default()
            },
            shared_expert_sink: false,
        }
    }

    fn inkling_style() -> Router {
        Router {
            scores: "layers.3.router.weight".into(),
            activation: ScoreActivation::Sigmoid,
            selection: Selection::TopK { k: 6 },
            post: RouterPostProcessing {
                norm_after_top_k: true,
                gate_bias: Some("layers.3.router.bias".into()),
                route_scale: Some(8.0),
                ..Default::default()
            },
            shared_expert_sink: true,
        }
    }

    fn k3_style() -> Router {
        Router {
            scores: "layers.12.router.weight".into(),
            activation: ScoreActivation::Softmax,
            selection: Selection::TopK { k: 16 },
            post: RouterPostProcessing {
                quantile_balanced: true,
                ..Default::default()
            },
            shared_expert_sink: false,
        }
    }

    #[test]
    fn every_envelope_router_round_trips_through_json() {
        for r in [
            gemma_style(),
            kimi_linear_style(),
            inkling_style(),
            k3_style(),
        ] {
            let json = serde_json::to_string(&r).unwrap();
            let back: Router = serde_json::from_str(&json).unwrap();
            assert_eq!(back, r, "{json}");
        }
    }

    #[test]
    fn plain_top_k_counts_its_own_k() {
        assert_eq!(Selection::TopK { k: 16 }.experts_per_token(), 16);
    }

    #[test]
    fn grouped_top_k_counts_k_per_group() {
        assert_eq!(
            Selection::GroupedTopK { k: 8, groups: 4 }.experts_per_token(),
            32
        );
    }

    #[test]
    fn a_single_group_is_equivalent_to_plain_top_k_in_count() {
        // Kimi-Linear's degenerate case: recorded as grouped, counts as top-8.
        assert_eq!(
            Selection::GroupedTopK { k: 8, groups: 1 }.experts_per_token(),
            Selection::TopK { k: 8 }.experts_per_token()
        );
    }

    #[test]
    fn only_the_bare_softmax_router_is_plain() {
        assert!(gemma_style().is_plain_softmax_top_k());
        assert!(!kimi_linear_style().is_plain_softmax_top_k());
        assert!(!inkling_style().is_plain_softmax_top_k());
        // K3 is softmax top-k but quantile-balanced, so it is not plain.
        assert!(!k3_style().is_plain_softmax_top_k());
    }

    #[test]
    fn a_sink_router_is_never_plain_even_with_softmax_top_k() {
        // The sink changes the reduction's weight budget; treating it as plain
        // would silently over-weight the routed branch.
        let sink = Router {
            shared_expert_sink: true,
            ..gemma_style()
        };
        assert!(!sink.is_plain_softmax_top_k());
    }

    #[test]
    fn gate_bias_is_reported_as_an_auxiliary_tensor() {
        assert_eq!(
            inkling_style().auxiliary_tensors(),
            vec!["layers.3.router.bias"]
        );
    }

    #[test]
    fn a_router_without_bias_needs_no_auxiliary_tensors() {
        assert!(gemma_style().auxiliary_tensors().is_empty());
    }

    #[test]
    fn post_processing_defaults_to_doing_nothing() {
        // OLMoE keeps raw softmax probabilities; renormalising by default
        // would silently change every model that does not ask for it.
        let d = RouterPostProcessing::default();
        assert!(!d.renormalise);
        assert!(!d.norm_after_top_k);
        assert!(!d.quantile_balanced);
        assert_eq!(d.gate_bias, None);
        assert_eq!(d.route_scale, None);
    }

    #[test]
    fn an_absent_post_block_deserialises_to_the_default() {
        let json = r#"{
            "scores": "r.weight",
            "activation": "softmax",
            "selection": {"kind": "top_k", "k": 2}
        }"#;
        let r: Router = serde_json::from_str(json).unwrap();
        assert_eq!(r.post, RouterPostProcessing::default());
        assert!(!r.shared_expert_sink);
    }

    #[test]
    fn selection_is_tagged_so_the_two_kinds_cannot_be_confused() {
        let json = serde_json::to_string(&Selection::GroupedTopK { k: 8, groups: 1 }).unwrap();
        assert!(json.contains("grouped_top_k"), "{json}");
    }
}
