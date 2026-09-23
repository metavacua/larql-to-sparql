//! Colocated tests for the synthetic layer builder.
//!
//! The builder feeds a bench, an allocation guard and a residency probe. If it
//! is wrong they all measure something other than what they claim, and none of
//! them would notice — a bench does not check its own answers.

use crate::runtime::execute::{execute, execute_traced};
use crate::runtime::fixtures::synthetic::{SyntheticLayer, SyntheticShape};
use crate::runtime::inputs::MoeInputs;

fn shape() -> SyntheticShape {
    SyntheticShape {
        population: 5,
        hidden: 4,
        intermediate: 3,
        top_k: 2,
    }
}

// ── The byte image ─────────────────────────────────────────────────────────

#[test]
fn the_image_is_exactly_the_size_the_shape_predicts() {
    let layer = SyntheticLayer::build(shape());
    assert_eq!(layer.bytes.len(), shape().total_bytes());
}

#[test]
fn every_extent_lies_inside_the_image() {
    let layer = SyntheticLayer::build(shape());
    assert!(layer.router.end <= layer.bytes.len());
    for extents in &layer.experts {
        assert!(extents.gate_up.end <= layer.bytes.len());
        assert!(extents.down.end <= layer.bytes.len());
    }
}

#[test]
fn extents_tile_the_image_without_gaps_or_overlap() {
    // A gap would waste pages the residency probe then attributes to routing;
    // an overlap would make two experts share weights.
    let layer = SyntheticLayer::build(shape());
    let mut cursor = layer.router.end;
    assert_eq!(layer.router.start, 0);
    for extents in &layer.experts {
        assert_eq!(extents.gate_up.start, cursor);
        assert_eq!(extents.down.start, extents.gate_up.end);
        cursor = extents.down.end;
    }
    assert_eq!(cursor, layer.bytes.len());
}

#[test]
fn each_experts_regions_are_adjacent() {
    // Expert-major locality is the property the residency probe measures.
    let layer = SyntheticLayer::build(shape());
    for extents in &layer.experts {
        assert_eq!(extents.gate_up.end, extents.down.start);
    }
}

#[test]
fn the_population_gets_one_set_of_extents_each() {
    assert_eq!(
        SyntheticLayer::build(shape()).experts.len(),
        shape().population
    );
}

// ── Shape arithmetic ───────────────────────────────────────────────────────

#[test]
fn a_fused_projection_is_two_halves_wide() {
    let s = shape();
    assert_eq!(s.projection_elements(), 2 * s.intermediate * s.hidden);
}

#[test]
fn selected_bytes_counts_the_router_plus_top_k_experts() {
    let s = shape();
    assert_eq!(
        s.selected_bytes(),
        s.router_bytes() + s.top_k * s.expert_bytes()
    );
    assert!(
        s.selected_bytes() < s.total_bytes(),
        "routing must be sparse"
    );
}

#[test]
fn selected_bytes_never_exceeds_the_population() {
    let s = SyntheticShape {
        top_k: 99,
        ..shape()
    };
    assert_eq!(s.selected_bytes(), s.total_bytes());
}

// ── Binding and execution ──────────────────────────────────────────────────

#[test]
fn a_synthetic_layer_binds_to_a_valid_operation() {
    let layer = SyntheticLayer::build(shape());
    layer.bind(&layer.bytes).validate().unwrap();
}

#[test]
fn a_synthetic_layer_executes_to_finite_values() {
    let layer = SyntheticLayer::build(shape());
    let out = execute(
        &layer.bind(&layer.bytes),
        MoeInputs::shared(&layer.residual()),
    )
    .unwrap();
    assert_eq!(out.len(), shape().hidden);
    assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    assert!(
        out.iter().any(|v| *v != 0.0),
        "a zero delta would be vacuous"
    );
}

#[test]
fn binding_over_a_copy_of_the_image_gives_the_same_answer() {
    // The property the residency probe depends on: identical content bound
    // over different backing memory executes identically.
    let layer = SyntheticLayer::build(shape());
    let copy = layer.bytes.clone();
    assert_eq!(
        execute(
            &layer.bind(&layer.bytes),
            MoeInputs::shared(&layer.residual())
        )
        .unwrap(),
        execute(&layer.bind(&copy), MoeInputs::shared(&layer.residual())).unwrap()
    );
}

#[test]
fn routing_selects_exactly_top_k_experts() {
    let layer = SyntheticLayer::build(shape());
    let (_, trace) = execute_traced(
        &layer.bind(&layer.bytes),
        MoeInputs::shared(&layer.residual()),
    )
    .unwrap();
    assert_eq!(trace.selection.len(), shape().top_k);
    assert_eq!(trace.router_scores.len(), shape().population);
}

#[test]
fn different_populations_produce_different_images() {
    let small = SyntheticLayer::build(shape());
    let large = SyntheticLayer::build(SyntheticShape {
        population: 9,
        ..shape()
    });
    assert!(large.bytes.len() > small.bytes.len());
    assert_eq!(large.experts.len(), 9);
}

#[test]
fn the_weight_generator_is_deterministic_and_varies_by_seed() {
    use crate::runtime::fixtures::synthetic::weight;
    assert_eq!(weight(1, 7), weight(1, 7));
    assert_ne!(weight(1, 7), weight(2, 7));
    assert_ne!(weight(1, 7), weight(1, 8));
    assert!((-0.5..0.5).contains(&weight(3, 11)));
}
