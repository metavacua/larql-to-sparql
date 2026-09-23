//! Colocated tests for `layer` — per-layer manifest well-formedness.
//!
//! Each test corrupts one field of a valid layer and requires the matching
//! named defect. The checks here are the ones the manifest can fail on its
//! own, before any weight byte is read; operand-absence needs the LYRW files
//! and is tested with the capability check.

use super::bank_ref::{BankRef, ExpertDims};
use super::layer::{Combine, InputSpace, LayerDefect, MoeLayer, Reduction, Transforms};
use super::router::{Router, RouterPostProcessing, ScoreActivation, Selection};

const LAYER: u32 = 12;
const ROUTED_EXPERTS: u32 = 128;

fn router(k: usize) -> Router {
    Router {
        scores: format!("layers.{LAYER}.router.weight"),
        activation: ScoreActivation::Softmax,
        selection: Selection::TopK { k },
        post: RouterPostProcessing::default(),
        shared_expert_sink: false,
    }
}

fn routed(programme: &str) -> BankRef {
    BankRef {
        experts: ROUTED_EXPERTS,
        programme: programme.into(),
        storage: format!("routed/layer_{LAYER:03}"),
        expert_dims: Some(ExpertDims {
            input: 2_816,
            intermediate: 704,
            output: 2_816,
        }),
    }
}

/// A conventional residual-space MoE layer — the Gemma/GPT-OSS shape.
fn residual_layer() -> MoeLayer {
    MoeLayer {
        layer: LAYER,
        input_space: InputSpace::Residual,
        router: router(8),
        transforms: None,
        routed_bank: routed("gated-mlp-v1"),
        shared_bank: None,
        reduction: Reduction::GateWeightedSum,
        routed_output_norm: None,
        combine: Combine::ResidualAdd,
    }
}

/// A K3-shaped latent layer with shared experts and transforms.
fn latent_layer() -> MoeLayer {
    MoeLayer {
        layer: LAYER,
        input_space: InputSpace::Latent,
        router: router(16),
        transforms: Some(Transforms {
            routed_input: format!("layers.{LAYER}.routed_expert_down_proj"),
            routed_output: format!("layers.{LAYER}.routed_expert_up_proj"),
        }),
        routed_bank: BankRef {
            experts: 896,
            programme: "latent-moe-v1".into(),
            storage: format!("routed/layer_{LAYER:03}"),
            expert_dims: Some(ExpertDims {
                input: 3_584,
                intermediate: 3_072,
                output: 3_584,
            }),
        },
        shared_bank: Some(BankRef {
            experts: 2,
            programme: "gated-mlp-v1".into(),
            storage: format!("shared/layer_{LAYER:03}"),
            expert_dims: Some(ExpertDims {
                input: 3_584,
                intermediate: 3_072,
                output: 3_584,
            }),
        }),
        reduction: Reduction::GateWeightedSum,
        routed_output_norm: Some(format!("layers.{LAYER}.routed_out_norm")),
        combine: Combine::ResidualAdd,
    }
}

#[test]
fn a_conventional_residual_layer_is_well_formed() {
    assert_eq!(residual_layer().defects(), vec![]);
    assert!(residual_layer().is_well_formed());
}

#[test]
fn a_latent_layer_with_transforms_is_well_formed() {
    assert_eq!(latent_layer().defects(), vec![]);
}

#[test]
fn both_layer_shapes_round_trip_through_json() {
    for l in [residual_layer(), latent_layer()] {
        let json = serde_json::to_string(&l).unwrap();
        assert_eq!(
            serde_json::from_str::<MoeLayer>(&json).unwrap(),
            l,
            "{json}"
        );
    }
}

#[test]
fn an_unknown_programme_is_named_with_its_storage() {
    let mut l = residual_layer();
    l.routed_bank.programme = "mixtral-expert-v9".into();
    assert!(l.defects().iter().any(|d| matches!(
        d,
        LayerDefect::UnknownProgramme { programme, .. } if programme == "mixtral-expert-v9"
    )));
}

#[test]
fn an_unknown_programme_does_not_also_report_a_space_contradiction() {
    // One root cause, one defect — for the checks that DEPEND on the
    // programme. A misleading space contradiction sends the reader chasing
    // input_space when the real problem is the name.
    //
    // Asserted by absence of the dependent defect, deliberately not by a total
    // count: a count assertion would forbid reporting *independent* defects
    // alongside, which is the over-suppression this scoping exists to avoid.
    let mut l = residual_layer();
    l.routed_bank.programme = "not-a-programme".into();
    assert!(!l
        .defects()
        .iter()
        .any(|d| matches!(d, LayerDefect::InputSpaceContradictsProgramme { .. })));
}

#[test]
fn an_unknown_programme_does_not_conceal_independent_storage_defects() {
    // A typo'd programme name must not hide a second, real corruption sitting
    // behind it — storage shape is not a function of programme identity.
    let mut l = residual_layer();
    l.routed_bank.programme = "not-a-programme".into();
    l.routed_bank.storage = "  ".into();
    l.routed_bank.experts = 0;

    let defects = l.defects();
    assert!(defects
        .iter()
        .any(|d| matches!(d, LayerDefect::UnknownProgramme { .. })));
    assert!(defects.contains(&LayerDefect::BankHasNoStorage {
        layer: LAYER,
        which: "routed"
    }));
    assert!(defects.contains(&LayerDefect::BankHasNoExperts {
        layer: LAYER,
        which: "routed"
    }));
}

#[test]
fn a_shared_bank_does_not_require_a_sink() {
    // The asymmetry that must stay pinned: a sink needs a shared bank, but
    // always-active shared experts OUTSIDE sink normalisation are valid —
    // that is Kimi-Linear's arrangement. `shared_expert_sink` must never
    // become shorthand for "this layer has shared experts".
    let mut l = latent_layer();
    l.router.shared_expert_sink = false;
    assert!(l.shared_bank.is_some());
    assert!(l.is_well_formed(), "{:?}", l.defects());
}

#[test]
fn an_empty_bank_storage_reference_is_refused() {
    let mut l = residual_layer();
    l.routed_bank.storage = String::new();
    assert!(l.defects().contains(&LayerDefect::BankHasNoStorage {
        layer: LAYER,
        which: "routed"
    }));
}

#[test]
fn two_banks_sharing_one_storage_reference_are_refused() {
    let mut l = latent_layer();
    l.shared_bank.as_mut().unwrap().storage = l.routed_bank.storage.clone();
    assert!(l
        .defects()
        .iter()
        .any(|d| matches!(d, LayerDefect::DuplicateBankStorage { .. })));
}

#[test]
fn two_empty_storage_references_report_emptiness_not_duplication() {
    // Both blank is one defect class, not two — reporting "duplicate storage
    // ''" on top would be noise pointing at the wrong fix.
    let mut l = latent_layer();
    l.routed_bank.storage = String::new();
    l.shared_bank.as_mut().unwrap().storage = String::new();
    assert!(!l
        .defects()
        .iter()
        .any(|d| matches!(d, LayerDefect::DuplicateBankStorage { .. })));
}

#[test]
fn a_zero_width_bank_is_refused() {
    let mut l = residual_layer();
    l.routed_bank.expert_dims.as_mut().unwrap().intermediate = 0;
    assert!(l
        .defects()
        .iter()
        .any(|d| matches!(d, LayerDefect::BankHasZeroDimension { .. })));
}

#[test]
fn an_undeclared_dims_bank_is_not_treated_as_zero_width() {
    // Absent dims mean unknown, not zero — reporting a zero-width defect
    // would be inventing a fact the manifest never stated.
    let mut l = residual_layer();
    l.routed_bank.expert_dims = None;
    assert!(l.is_well_formed(), "{:?}", l.defects());
}

#[test]
fn a_latent_bank_without_transforms_is_refused() {
    let mut l = latent_layer();
    l.transforms = None;
    assert!(l.defects().iter().any(|d| matches!(
        d,
        LayerDefect::LatentBankWithoutTransforms {
            which: "routed",
            ..
        }
    )));
}

#[test]
fn a_latent_programme_in_a_residual_layer_is_refused() {
    let mut l = latent_layer();
    l.input_space = InputSpace::Residual;
    assert!(l
        .defects()
        .iter()
        .any(|d| matches!(d, LayerDefect::InputSpaceContradictsProgramme { .. })));
}

#[test]
fn a_residual_programme_in_a_latent_layer_is_refused() {
    let mut l = residual_layer();
    l.input_space = InputSpace::Latent;
    assert!(l
        .defects()
        .iter()
        .any(|d| matches!(d, LayerDefect::InputSpaceContradictsProgramme { .. })));
}

#[test]
fn an_empty_router_scores_name_is_refused() {
    let mut l = residual_layer();
    l.router.scores = "   ".into();
    assert!(l
        .defects()
        .contains(&LayerDefect::RouterHasNoScores { layer: LAYER }));
}

#[test]
fn selecting_more_experts_than_the_bank_holds_is_refused() {
    let mut l = residual_layer();
    l.router.selection = Selection::TopK {
        k: ROUTED_EXPERTS as usize + 1,
    };
    assert!(l.defects().contains(&LayerDefect::SelectionExceedsBank {
        layer: LAYER,
        selected: ROUTED_EXPERTS as usize + 1,
        available: ROUTED_EXPERTS,
    }));
}

#[test]
fn selecting_exactly_the_whole_bank_is_allowed() {
    // Dense-equivalent routing is degenerate but not malformed.
    let mut l = residual_layer();
    l.router.selection = Selection::TopK {
        k: ROUTED_EXPERTS as usize,
    };
    assert!(l.is_well_formed());
}

#[test]
fn grouped_selection_is_counted_across_groups() {
    let mut l = residual_layer();
    l.router.selection = Selection::GroupedTopK {
        k: ROUTED_EXPERTS as usize,
        groups: 2,
    };
    assert!(l
        .defects()
        .iter()
        .any(|d| matches!(d, LayerDefect::SelectionExceedsBank { .. })));
}

#[test]
fn a_sink_router_without_a_shared_bank_is_refused() {
    // The sink scores shared experts; with no shared bank there is nothing to
    // score, and the reduction's weight budget would be silently wrong.
    let mut l = residual_layer();
    l.router.shared_expert_sink = true;
    assert!(l
        .defects()
        .contains(&LayerDefect::SinkWithoutSharedBank { layer: LAYER }));
}

#[test]
fn a_sink_router_with_a_shared_bank_is_well_formed() {
    let mut l = latent_layer();
    l.router.shared_expert_sink = true;
    assert!(l.is_well_formed());
}

#[test]
fn banks_lists_routed_first_then_shared() {
    let l = latent_layer();
    let banks = l.banks();
    assert_eq!(banks.len(), 2);
    assert!(banks[0].storage.starts_with("routed/"));
    assert!(banks[1].storage.starts_with("shared/"));
}

#[test]
fn a_layer_without_a_shared_bank_lists_one_bank() {
    assert_eq!(residual_layer().banks().len(), 1);
}

#[test]
fn walkable_features_sum_across_banks() {
    // 896 x 3072 routed + 2 x 3072 shared
    assert_eq!(
        latent_layer().walkable_feature_count(),
        Some(896 * 3_072 + 2 * 3_072)
    );
}

#[test]
fn one_undeclared_bank_makes_the_layer_count_unknown() {
    let mut l = latent_layer();
    l.shared_bank.as_mut().unwrap().expert_dims = None;
    assert_eq!(l.walkable_feature_count(), None);
}

#[test]
fn an_absent_shared_bank_deserialises_to_none() {
    let json = serde_json::to_string(&residual_layer()).unwrap();
    let back: MoeLayer = serde_json::from_str(&json).unwrap();
    assert_eq!(back.shared_bank, None);
    assert_eq!(back.transforms, None);
}
