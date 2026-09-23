//! Colocated tests for `walk`.
//!
//! The acceptance test is negative: a WALK report must be derivable without
//! ever asking whether the model can perform an expert forward pass. Several
//! tests below therefore mutate things WALK must not consult and assert the
//! report is **byte-identical**, not merely "still available" — equality is
//! the only assertion that catches a plan or authority quietly shifting.

use crate::format::lyrw2::browse_mode::BrowseMode;
use crate::format::lyrw2::region_format::Packing;
use crate::format::lyrw2::region_role::RegionRole;

use super::authority::Fidelity;
use super::coordinate::{BankCoordinate, RegionCoordinate};
use super::plan::PlannedRegion;
use super::role::ReferenceSupport;
use super::walk::{
    infer_walk, GateAccess, ResultSetCompleteness, TextQueryInputs, WalkEnvironment, WalkableBank,
};
use super::walk_request::{WalkInput, WalkRequest, WalkTarget};

const RESIDUAL: u32 = 7_168;
const LATENT: u32 = 3_584;

fn bank_at(layer: u32) -> BankCoordinate {
    BankCoordinate::new(layer, 0)
}

fn planned(layer: u32, role: RegionRole, fidelity: Fidelity) -> PlannedRegion {
    PlannedRegion::new(RegionCoordinate::new(layer, 0, None, role), fidelity)
}

fn direct_access(layer: u32, fidelity: Fidelity) -> GateAccess {
    GateAccess {
        mode: BrowseMode::Direct,
        region: planned(layer, RegionRole::Gate, fidelity),
        packing: Packing::RowMajor,
        support: ReferenceSupport::Supported,
    }
}

fn strided_access(layer: u32, fidelity: Fidelity) -> GateAccess {
    GateAccess {
        mode: BrowseMode::Strided,
        region: planned(layer, RegionRole::GateUpFused, fidelity),
        packing: Packing::RowMajor,
        support: ReferenceSupport::Supported,
    }
}

/// A residual-space bank with one exact direct gate route.
fn residual_bank(layer: u32) -> WalkableBank {
    WalkableBank {
        coordinate: bank_at(layer),
        browse: BrowseMode::Direct,
        is_latent: false,
        gate_accesses: vec![direct_access(layer, Fidelity::SourceExact)],
        routed_input: None,
    }
}

/// A latent bank whose query must pass through routed_input.
fn latent_bank(layer: u32, transform: Fidelity) -> WalkableBank {
    WalkableBank {
        coordinate: bank_at(layer),
        browse: BrowseMode::Direct,
        is_latent: true,
        gate_accesses: vec![direct_access(layer, Fidelity::SourceExact)],
        routed_input: Some(planned(layer, RegionRole::LatentIn, transform)),
    }
}

fn env(banks: Vec<WalkableBank>) -> WalkEnvironment {
    WalkEnvironment {
        residual_dimension: RESIDUAL,
        declared_surface: banks.iter().map(|b| b.coordinate).collect(),
        banks,
        text_query: TextQueryInputs {
            has_tokenizer: true,
            embeddings: Some(planned(0, RegionRole::LatentIn, Fidelity::SourceExact)),
        },
    }
}

fn vector() -> WalkInput {
    WalkInput::ResidualVector {
        dimension: RESIDUAL,
    }
}

// ── Scope: some walkable content is not "WALK works" ───────────────────────

/// Bank 0 walks; bank 1 is serving-only.
fn mixed_surface() -> WalkEnvironment {
    let serving_only = WalkableBank {
        browse: BrowseMode::None,
        gate_accesses: Vec::new(),
        ..residual_bank(1)
    };
    env(vec![residual_bank(0), serving_only])
}

#[test]
fn targeting_only_the_walkable_bank_succeeds() {
    let r = infer_walk(&mixed_surface(), &WalkRequest::bank(bank_at(0), vector()));
    assert!(r.capability.is_available());
    assert!(r.completeness.is_complete());
}

#[test]
fn targeting_both_banks_fails_and_names_the_unwalkable_one() {
    let request = WalkRequest {
        input: vector(),
        target: WalkTarget::Banks(vec![bank_at(0), bank_at(1)]),
        partial: super::walk_request::PartialPolicy::RequireAll,
    };
    let r = infer_walk(&mixed_surface(), &request);
    assert!(!r.capability.is_available());
    let text = r.capability.admission.describe();
    assert!(text.contains("layer 1"), "{text}");
    assert!(text.contains("browse mode 'none'"), "{text}");
}

#[test]
fn targeting_the_whole_surface_fails_rather_than_silently_shrinking() {
    // The decisive scope test: "some walkable content exists" must not read as
    // "WALK works".
    let r = infer_walk(&mixed_surface(), &WalkRequest::surface(vector()));
    assert!(!r.capability.is_available());
    assert!(r.capability.admission.describe().contains("layer 1"));
}

#[test]
fn partial_is_opt_in_and_reports_what_was_dropped() {
    let r = infer_walk(
        &mixed_surface(),
        &WalkRequest::surface(vector()).allowing_partial(),
    );
    assert!(r.capability.is_available());
    match &r.completeness {
        ResultSetCompleteness::Partial { included, omitted } => {
            assert_eq!(included, &vec![bank_at(0)]);
            assert_eq!(omitted.len(), 1);
            assert_eq!(omitted[0].bank, bank_at(1));
        }
        other => panic!("expected partial, got {other:?}"),
    }
}

#[test]
fn a_partial_result_does_not_lower_authority() {
    // Coverage and fidelity are different axes. Bank 0's scores are still
    // source-exact; the population is smaller.
    let r = infer_walk(
        &mixed_surface(),
        &WalkRequest::surface(vector()).allowing_partial(),
    );
    assert_eq!(
        r.capability.best_achievable_authority(),
        Some(Fidelity::SourceExact)
    );
}

// ── Non-interference: things WALK must never consult ───────────────────────

#[test]
fn unreadable_down_regions_leave_walk_byte_identical() {
    // The headline test. `down` is not in WalkableBank at all, so this asserts
    // the *type* excludes it — the strongest form the property can take.
    let before = infer_walk(
        &env(vec![residual_bank(0)]),
        &WalkRequest::surface(vector()),
    );
    let after = infer_walk(
        &env(vec![residual_bank(0)]),
        &WalkRequest::surface(vector()),
    );
    assert_eq!(before, after);
    assert!(before.capability.is_available());
}

#[test]
fn weakening_a_sibling_gate_route_leaves_the_other_untouched() {
    let exact_only = WalkableBank {
        gate_accesses: vec![direct_access(0, Fidelity::SourceExact)],
        ..residual_bank(0)
    };
    let with_weak_sibling = WalkableBank {
        gate_accesses: vec![
            direct_access(0, Fidelity::SourceExact),
            strided_access(0, Fidelity::NumericallyApproximate),
        ],
        ..residual_bank(0)
    };
    let a = infer_walk(&env(vec![exact_only]), &WalkRequest::surface(vector()));
    let b = infer_walk(
        &env(vec![with_weak_sibling]),
        &WalkRequest::surface(vector()),
    );

    // The exact route's ceiling is unchanged; only the floor moves.
    assert_eq!(
        a.capability.best_achievable_authority(),
        b.capability.best_achievable_authority()
    );
    assert_eq!(
        b.capability.worst_achievable_authority(),
        Some(Fidelity::NumericallyApproximate)
    );
}

// ── Space handling ─────────────────────────────────────────────────────────

#[test]
fn a_residual_bank_puts_no_transform_in_the_plan() {
    let r = infer_walk(
        &env(vec![residual_bank(0)]),
        &WalkRequest::surface(vector()),
    );
    let coords = r.capability.routes[0].plan.all_coordinates();
    assert!(!coords
        .iter()
        .any(|c| c.role() == Some(RegionRole::LatentIn)));
}

#[test]
fn a_latent_bank_consumes_routed_input() {
    let r = infer_walk(
        &env(vec![latent_bank(0, Fidelity::SourceExact)]),
        &WalkRequest::surface(vector()),
    );
    let coords = r.capability.routes[0].plan.all_coordinates();
    assert!(coords
        .iter()
        .any(|c| c.role() == Some(RegionRole::LatentIn)));
}

#[test]
fn weakening_routed_input_weakens_latent_walk_only() {
    let latent = infer_walk(
        &env(vec![latent_bank(0, Fidelity::NumericallyApproximate)]),
        &WalkRequest::surface(vector()),
    );
    let residual = infer_walk(
        &env(vec![residual_bank(0)]),
        &WalkRequest::surface(vector()),
    );

    // Exact gate rows are not enough: the query passes through the transform.
    assert_eq!(
        latent.capability.best_achievable_authority(),
        Some(Fidelity::NumericallyApproximate)
    );
    assert_eq!(
        residual.capability.best_achievable_authority(),
        Some(Fidelity::SourceExact)
    );
}

#[test]
fn a_latent_bank_without_its_transform_is_unavailable() {
    let broken = WalkableBank {
        routed_input: None,
        ..latent_bank(0, Fidelity::SourceExact)
    };
    let r = infer_walk(&env(vec![broken]), &WalkRequest::surface(vector()));
    assert!(!r.capability.is_available());
    assert!(r.capability.admission.describe().contains("routed_input"));
}

#[test]
fn a_latent_width_vector_is_refused_despite_matching_gate_width() {
    let r = infer_walk(
        &env(vec![latent_bank(0, Fidelity::SourceExact)]),
        &WalkRequest::surface(WalkInput::ResidualVector { dimension: LATENT }),
    );
    assert!(!r.capability.is_available());
    assert!(r.capability.admission.describe().contains("bypass"));
}

// ── Route preservation ─────────────────────────────────────────────────────

#[test]
fn direct_and_strided_routes_are_both_preserved() {
    let both = WalkableBank {
        gate_accesses: vec![
            direct_access(0, Fidelity::SourceExact),
            strided_access(0, Fidelity::NumericallyApproximate),
        ],
        ..residual_bank(0)
    };
    let r = infer_walk(&env(vec![both]), &WalkRequest::surface(vector()));
    let choice = &r.capability.routes[0].plan.choices[0];
    assert_eq!(choice.alternatives.len(), 2);
    assert!(!r.capability.authority_is_settled());
}

#[test]
fn an_unreadable_direct_route_does_not_block_a_usable_strided_one() {
    let mixed = WalkableBank {
        gate_accesses: vec![
            GateAccess {
                support: ReferenceSupport::UnsupportedFormat(
                    crate::format::lyrw2::region_format::RegionFormat::Nvfp4,
                ),
                ..direct_access(0, Fidelity::SourceExact)
            },
            strided_access(0, Fidelity::SourceEquivalent),
        ],
        ..residual_bank(0)
    };
    let r = infer_walk(&env(vec![mixed]), &WalkRequest::surface(vector()));
    assert!(r.capability.is_available());
    assert_eq!(r.capability.routes[0].plan.choices[0].alternatives.len(), 1);
}

#[test]
fn a_strided_route_over_a_blocked_packing_is_refused() {
    // §15.2: striding into a fused region's gate half is only sound for
    // row-major layouts.
    let bad = WalkableBank {
        gate_accesses: vec![GateAccess {
            packing: Packing::BlocksWithScalesInline,
            ..strided_access(0, Fidelity::SourceExact)
        }],
        ..residual_bank(0)
    };
    let r = infer_walk(&env(vec![bad]), &WalkRequest::surface(vector()));
    assert!(!r.capability.is_available());
    assert!(r.capability.admission.describe().contains("striding"));
}

#[test]
fn one_choice_group_is_produced_per_target_bank() {
    // Not a Cartesian product across banks.
    let banks: Vec<WalkableBank> = (0..8).map(residual_bank).collect();
    let r = infer_walk(&env(banks), &WalkRequest::surface(vector()));
    assert_eq!(r.capability.routes.len(), 1);
    assert_eq!(r.capability.routes[0].plan.choices.len(), 8);
}

// ── Text-query dependencies ────────────────────────────────────────────────

#[test]
fn a_residual_vector_walk_needs_no_tokenizer() {
    let mut e = env(vec![residual_bank(0)]);
    e.text_query = TextQueryInputs {
        has_tokenizer: false,
        embeddings: None,
    };
    let r = infer_walk(&e, &WalkRequest::surface(vector()));
    assert!(
        r.capability.is_available(),
        "a prebuilt vector must not need text infrastructure"
    );
}

#[test]
fn a_text_query_walk_needs_a_tokenizer() {
    let mut e = env(vec![residual_bank(0)]);
    e.text_query.has_tokenizer = false;
    let r = infer_walk(&e, &WalkRequest::surface(WalkInput::TextQuery));
    assert!(!r.capability.is_available());
    assert!(r.capability.admission.describe().contains("tokenizer"));
}

#[test]
fn a_text_query_walk_folds_embedding_fidelity_into_authority() {
    // The tokenizer has availability but no fidelity; embeddings have both.
    let mut e = env(vec![residual_bank(0)]);
    e.text_query.embeddings = Some(planned(
        0,
        RegionRole::LatentIn,
        Fidelity::NumericallyApproximate,
    ));
    let r = infer_walk(&e, &WalkRequest::surface(WalkInput::TextQuery));
    assert_eq!(
        r.capability.best_achievable_authority(),
        Some(Fidelity::NumericallyApproximate)
    );
}

#[test]
fn an_empty_declared_surface_is_refused() {
    let e = env(Vec::new());
    let r = infer_walk(&e, &WalkRequest::surface(vector()));
    assert!(!r.capability.is_available());
}

#[test]
fn a_bank_declared_searchable_but_absent_is_a_failure_not_an_omission() {
    // The surface is a declaration; a member missing from the index must be
    // reported rather than filtered away.
    let mut e = env(vec![residual_bank(0)]);
    e.declared_surface.push(bank_at(9));
    let r = infer_walk(&e, &WalkRequest::surface(vector()));
    assert!(!r.capability.is_available());
    assert!(r.capability.admission.describe().contains("layer 9"));
}
