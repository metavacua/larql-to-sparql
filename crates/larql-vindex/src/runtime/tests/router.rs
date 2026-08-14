//! Colocated tests for `router`.

use larql_compute::MoeTopKWeightPolicy;

use super::support::{ascending, vector};
use crate::runtime::error::ExecutionError;
use crate::runtime::router::{BoundExpertScaling, BoundRouter, RouterKernel, SelectedExpert};

const POPULATION: u32 = 4;
const HIDDEN: u32 = 3;
const TOP_K: usize = 2;
const ROUTER: &str = "router";
const SCALE: &str = "router_scale";

fn router(per_expert_scale: Option<&[f32]>) -> BoundRouter<'static> {
    BoundRouter {
        weight: ascending(ROUTER, POPULATION, HIDDEN),
        top_k: TOP_K,
        selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
        scaling: match per_expert_scale {
            Some(v) => BoundExpertScaling::PerExpert {
                scales: vector(SCALE, v),
            },
            None => BoundExpertScaling::None,
        },
        kernel: RouterKernel::default(),
    }
}

#[test]
fn a_router_reports_the_population_and_width_it_scores() {
    let r = router(None);
    assert_eq!(r.population(), POPULATION as usize);
    assert_eq!(r.hidden_dim(), HIDDEN as usize);
}

#[test]
fn a_router_validates_against_the_bank_it_routes_into() {
    router(None)
        .validate(POPULATION as usize, HIDDEN as usize)
        .unwrap();
}

#[test]
fn a_router_addressing_a_different_population_is_refused() {
    let err = router(None)
        .validate(POPULATION as usize + 1, HIDDEN as usize)
        .unwrap_err();
    assert!(matches!(err, ExecutionError::DimensionMismatch { .. }));
}

// ── The scale is routing semantics, so a shard carries all of it ───────────

#[test]
fn a_scale_must_cover_the_addressable_population_not_the_resident_shard() {
    // Expert 90's learned scale is part of what routing *means*, whether or
    // not expert 90 lives here. Truncating the vector to the resident subset
    // would make two shards of one model weight the same expert differently.
    let addressable = POPULATION as usize;
    let shard_size = 2;
    let truncated = router(Some(&vec![1.0f32; shard_size]));
    assert!(
        truncated.validate(addressable, HIDDEN as usize).is_err(),
        "a scale sized to the shard must be refused"
    );
    let full = router(Some(&vec![1.0f32; addressable]));
    full.validate(addressable, HIDDEN as usize).unwrap();
}

#[test]
fn a_non_finite_scale_is_refused_at_bind_time() {
    // It would multiply a valid routing weight into a NaN that propagates
    // through the reduction into the residual stream, surfacing as an
    // inexplicable token rather than a bad operand.
    let mut scales = vec![1.0f32; POPULATION as usize];
    scales[2] = f32::NAN;
    let err = router(Some(&scales))
        .validate(POPULATION as usize, HIDDEN as usize)
        .unwrap_err();
    let ExecutionError::NonFiniteExpertScale { expert, .. } = err else {
        panic!("expected a non-finite scale refusal, got {err}");
    };
    assert_eq!(expert, 2);
}

#[test]
fn an_infinite_scale_is_refused_too() {
    let mut scales = vec![1.0f32; POPULATION as usize];
    scales[0] = f32::INFINITY;
    assert!(matches!(
        router(Some(&scales))
            .validate(POPULATION as usize, HIDDEN as usize)
            .unwrap_err(),
        ExecutionError::NonFiniteExpertScale { expert: 0, .. }
    ));
}

#[test]
fn scaling_reports_the_incumbent_policy_it_corresponds_to() {
    use larql_compute::MoeExpertScalePolicy;
    assert_eq!(router(None).scaling.policy(), MoeExpertScalePolicy::None);
    assert_eq!(
        router(Some(&vec![1.0f32; POPULATION as usize]))
            .scaling
            .policy(),
        MoeExpertScalePolicy::PerExpert
    );
}

#[test]
fn a_per_expert_scale_must_cover_the_whole_population() {
    // A short scale vector would silently leave the tail of the population
    // unscaled rather than fail.
    let short = router(Some(&[1.0, 1.0]));
    assert!(short
        .validate(POPULATION as usize, HIDDEN as usize)
        .is_err());

    let full = router(Some(&[1.0, 1.0, 1.0, 1.0]));
    full.validate(POPULATION as usize, HIDDEN as usize).unwrap();
}

#[test]
fn a_router_without_a_scale_validates_without_one() {
    assert!(router(None).scaling.scales().is_none());
    router(None)
        .validate(POPULATION as usize, HIDDEN as usize)
        .unwrap();
}

#[test]
fn a_router_describes_its_depth_and_population() {
    let text = router(None).describe();
    assert!(text.contains(&format!("top-{TOP_K}")), "{text}");
    assert!(text.contains(&POPULATION.to_string()), "{text}");
}

#[test]
fn a_selected_expert_keeps_both_its_weights() {
    // Two selections can share a final weight after renormalisation while
    // having scored differently; the raw score is where that stays visible.
    let s = SelectedExpert {
        expert_id: 3,
        weight: 0.5,
        raw_score: 0.2,
    };
    assert_eq!(s.expert_id, 3);
    assert_ne!(s.weight, s.raw_score);
    assert_eq!(s, s);
}

#[test]
fn each_router_kernel_has_a_distinct_name_and_the_reference_is_the_default() {
    // The names reach operator-facing refusals — `KernelOperandUnsuitable`
    // reports which kernel declined an operand — so two kernels sharing one
    // would make a report unactionable.
    assert_eq!(RouterKernel::default(), RouterKernel::Reference);
    assert_ne!(
        RouterKernel::Reference.name(),
        RouterKernel::Incumbent.name()
    );
}

#[test]
fn each_scaling_policy_has_a_distinct_name() {
    assert_eq!(BoundExpertScaling::None.name(), "none");
    assert_ne!(
        router(Some(&[1.0, 1.0, 1.0, 1.0])).scaling.name(),
        BoundExpertScaling::None.name()
    );
}
