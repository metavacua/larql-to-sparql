//! Colocated tests for `plan` — route-scoped authority and choice groups.
//!
//! Two properties carry the design: an operation admitting several routes of
//! different fidelity must not collapse to one number, and a plan with many
//! independent choices must not be enumerated.

use crate::format::lyrw2::region_role::RegionRole;
use crate::format::moe_manifest::programme::Programme;

use super::authority::Fidelity;
use super::coordinate::{BankCoordinate, RegionCoordinate};
use super::operation::{OperationCapability, OperationFailure};
use super::plan::{
    OperationPlan, PlanChoice, PlannedRegion, QualifiedAlternative, QualifiedOperationRoute,
};

fn region(layer: u32, role: RegionRole, fidelity: Fidelity) -> PlannedRegion {
    PlannedRegion::new(RegionCoordinate::new(layer, 0, None, role), fidelity)
}

/// A route reading one exact gate region — the direct-browse shape.
fn direct_gate_route() -> QualifiedOperationRoute {
    QualifiedOperationRoute::new(OperationPlan::fixed_regions(vec![region(
        0,
        RegionRole::Gate,
        Fidelity::SourceExact,
    )]))
}

/// A route striding an approximate fused region — same operation, worse bytes.
fn strided_fused_route() -> QualifiedOperationRoute {
    QualifiedOperationRoute::new(OperationPlan::fixed_regions(vec![region(
        0,
        RegionRole::GateUpFused,
        Fidelity::NumericallyApproximate,
    )]))
}

fn alternative(roles: &'static [RegionRole], fidelity: Fidelity) -> QualifiedAlternative {
    QualifiedAlternative {
        alternative: roles,
        regions: roles.iter().map(|r| region(0, *r, fidelity)).collect(),
        components: Vec::new(),
    }
}

#[test]
fn a_route_with_no_choices_has_a_settled_authority() {
    let r = direct_gate_route();
    assert!(r.authority_is_settled());
    assert_eq!(r.best_achievable_authority().level, Fidelity::SourceExact);
    assert_eq!(r.worst_achievable_authority().level, Fidelity::SourceExact);
}

#[test]
fn two_routes_of_different_fidelity_are_both_preserved() {
    // The case a single operation-level authority cannot express: reporting
    // source-exact risks binding the approximate route; reporting approximate
    // understates the exact one.
    let c = OperationCapability::available(vec![direct_gate_route(), strided_fused_route()]);
    assert_eq!(c.routes.len(), 2);
    assert_eq!(c.best_achievable_authority(), Some(Fidelity::SourceExact));
    assert_eq!(
        c.worst_achievable_authority(),
        Some(Fidelity::NumericallyApproximate)
    );
}

#[test]
fn an_operation_with_differing_routes_reports_an_unsettled_authority() {
    let c = OperationCapability::available(vec![direct_gate_route(), strided_fused_route()]);
    assert!(
        !c.authority_is_settled(),
        "binding can still change the outcome"
    );
}

#[test]
fn an_operation_whose_routes_agree_reports_a_settled_authority() {
    let c = OperationCapability::available(vec![direct_gate_route(), direct_gate_route()]);
    assert!(c.authority_is_settled());
}

#[test]
fn an_unavailable_operation_has_no_routes_and_therefore_no_authority() {
    // The fail-closed invariant, restated in the new shape: no routes means no
    // fidelity, so a contradictory selection cannot acquire a weak-but-valid
    // one by default.
    let c = OperationCapability::unavailable(vec![OperationFailure::InvalidSelection {
        detail: "disjoint segments".into(),
    }]);
    assert!(c.routes.is_empty());
    assert_eq!(c.best_achievable_authority(), None);
    assert_eq!(c.worst_achievable_authority(), None);
    assert!(!c.authority_is_settled());
}

#[test]
fn routes_exist_exactly_when_the_operation_admits() {
    assert!(OperationCapability::available(vec![direct_gate_route()]).is_well_formed());
    assert!(
        OperationCapability::unavailable(vec![OperationFailure::NoExecutableRoute { layer: 0 }])
            .is_well_formed()
    );
    // An available operation with no routes is malformed by construction.
    let malformed = OperationCapability::available(Vec::new());
    assert!(!malformed.is_well_formed());
}

#[test]
fn a_degraded_operation_still_carries_its_routes() {
    use super::operation::Degradation;
    let c = OperationCapability::degraded(
        vec![direct_gate_route()],
        vec![Degradation::MissingQueryMetadata { what: "labels" }],
    );
    assert!(c.is_available());
    assert!(c.is_well_formed());
    assert_eq!(c.best_achievable_authority(), Some(Fidelity::SourceExact));
}

// ── Choice groups ──────────────────────────────────────────────────────────

const FUSED: &[RegionRole] = &[RegionRole::GateUpFused, RegionRole::Down];
const DECOMPOSED: &[RegionRole] = &[RegionRole::Gate, RegionRole::Up, RegionRole::Down];

fn two_way_choice(layer: u32) -> PlanChoice {
    PlanChoice {
        bank: BankCoordinate::new(layer, 0),
        alternatives: vec![
            alternative(FUSED, Fidelity::NumericallyApproximate),
            alternative(DECOMPOSED, Fidelity::SourceExact),
        ],
    }
}

#[test]
fn a_choice_group_reports_a_range_rather_than_a_value() {
    // Binding has not chosen, so the honest statement is an interval.
    let plan = OperationPlan {
        fixed_components: Vec::new(),
        choices: vec![two_way_choice(0)],
    };
    assert!(plan.has_open_choices());
    assert_eq!(
        plan.best_achievable_authority().level,
        Fidelity::SourceExact
    );
    assert_eq!(
        plan.worst_achievable_authority().level,
        Fidelity::NumericallyApproximate
    );
}

#[test]
fn thirty_independent_choices_do_not_enumerate() {
    // The property the representation exists for: 2^30 whole-model routes are
    // never materialised. Thirty choice groups stay thirty.
    let plan = OperationPlan {
        fixed_components: Vec::new(),
        choices: (0..30).map(two_way_choice).collect(),
    };
    assert_eq!(plan.choices.len(), 30);
    assert_eq!(
        plan.best_achievable_authority().level,
        Fidelity::SourceExact
    );
    assert_eq!(
        plan.worst_achievable_authority().level,
        Fidelity::NumericallyApproximate
    );
}

#[test]
fn a_fixed_region_drags_the_ceiling_down_across_every_choice() {
    // Weakest-link still applies: no binding can beat the fixed regions.
    let plan = OperationPlan {
        fixed_components: vec![
            region(0, RegionRole::LatentIn, Fidelity::NumericallyApproximate).as_component(),
        ],
        choices: vec![PlanChoice {
            bank: BankCoordinate::new(0, 0),
            alternatives: vec![alternative(FUSED, Fidelity::SourceExact)],
        }],
    };
    assert_eq!(
        plan.best_achievable_authority().level,
        Fidelity::NumericallyApproximate
    );
}

#[test]
fn a_single_alternative_choice_is_not_an_open_choice() {
    let plan = OperationPlan {
        fixed_components: Vec::new(),
        choices: vec![PlanChoice {
            bank: BankCoordinate::new(0, 0),
            alternatives: vec![alternative(FUSED, Fidelity::SourceEquivalent)],
        }],
    };
    assert!(!plan.has_open_choices());
    assert_eq!(
        plan.best_achievable_authority().level,
        plan.worst_achievable_authority().level
    );
}

#[test]
fn the_best_alternative_is_the_one_whose_weakest_region_is_strongest() {
    // Not "the one with an exact region somewhere" — weakest-link decides.
    let mixed = QualifiedAlternative {
        alternative: DECOMPOSED,
        regions: vec![
            region(0, RegionRole::Gate, Fidelity::SourceExact),
            region(0, RegionRole::Up, Fidelity::StructurallyApproximate),
            region(0, RegionRole::Down, Fidelity::SourceExact),
        ],
        components: Vec::new(),
    };
    let uniform = alternative(FUSED, Fidelity::NumericallyApproximate);
    let choice = PlanChoice {
        bank: BankCoordinate::new(0, 0),
        alternatives: vec![mixed, uniform.clone()],
    };
    assert_eq!(
        choice.best_alternative().unwrap().alternative,
        uniform.alternative,
        "the uniformly-approximate route beats one with a weak link"
    );
}

#[test]
fn a_plan_lists_every_coordinate_it_could_read() {
    let plan = OperationPlan {
        fixed_components: vec![
            region(0, RegionRole::LatentIn, Fidelity::SourceExact).as_component()
        ],
        choices: vec![two_way_choice(0)],
    };
    let coords = plan.all_coordinates();
    // latent_in + fused + down + gate + up, deduplicated across alternatives.
    assert_eq!(coords.len(), 5, "{coords:?}");
}

#[test]
fn an_empty_plan_is_empty_and_settled() {
    let plan = OperationPlan::default();
    assert!(plan.is_empty());
    assert!(!plan.has_open_choices());
    // Nothing selected floors at analysis-only rather than defaulting high.
    assert_eq!(
        plan.best_achievable_authority().level,
        Fidelity::AnalysisOnly
    );
}

#[test]
fn programme_alternatives_survive_into_the_plan_unchanged() {
    // The alternative identity is carried through so kernel binding can match
    // a kernel to the layout it actually supports.
    let choice = two_way_choice(0);
    let declared = Programme::GatedMlpV1.role_alternatives();
    assert_eq!(choice.alternatives[0].alternative, declared[0]);
    assert_eq!(choice.alternatives[1].alternative, declared[1]);
}
