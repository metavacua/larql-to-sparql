//! Fixture A — one direct routed-MoE layer against an independent oracle.
//!
//! Checkpoint by checkpoint, not final-vector-only. A single equality
//! assertion on the output says an execution is wrong and stops there; these
//! say *where*:
//!
//! ```text
//! router_scores    scoring, or the router operand
//! selection        top-k depth, ordering, or a weight policy
//! expert_outputs   region interpretation, activation, or the gated product
//! reduced          the combine
//! residual_delta   the transform stages
//! ```

use crate::runtime::execute::{execute, execute_traced};
use crate::runtime::fixtures::direct_moe::{
    input, up_at, DirectMoeFixture, HIDDEN, INTERMEDIATE, POPULATION, TOP_K,
};
use crate::runtime::fixtures::direct_moe_oracle::oracle;
use crate::runtime::inputs::MoeInputs;
use crate::runtime::projection::{BoundProjection, ProjectionArrangement};

/// Reference and oracle run the same arithmetic in the same order, so they
/// agree to well within this. It is a float-comparison guard, not a tolerance
/// budget — loosening it to make a test pass would be hiding a real defect.
const EPSILON: f32 = 1e-6;

fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < EPSILON, "{what}[{i}]: {a} vs expected {e}");
    }
}

// ── The bound operation is well-formed ─────────────────────────────────────

#[test]
fn both_arrangements_validate() {
    let fixture = DirectMoeFixture::new();
    for arrangement in ProjectionArrangement::ALL {
        let op = fixture.operation(arrangement);
        op.validate()
            .unwrap_or_else(|e| panic!("{}: {e}", arrangement.name()));
    }
}

#[test]
fn the_operation_describes_itself_as_direct_not_latent() {
    let fixture = DirectMoeFixture::new();
    let op = fixture.operation(ProjectionArrangement::Decomposed);
    assert!(!op.is_latent());
    assert_eq!(op.bank_input_dim(), HIDDEN, "no routed-input transform");
}

// ── Checkpoint agreement with the oracle ───────────────────────────────────

#[test]
fn router_scores_match_the_oracle() {
    let fixture = DirectMoeFixture::new();
    let (_, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let expected = oracle(&input());
    assert_eq!(trace.router_scores.len(), POPULATION);
    assert_close(
        &trace.router_scores,
        &expected.router_scores,
        "router_scores",
    );
}

#[test]
fn router_scores_are_a_distribution_over_the_whole_population() {
    // Softmax runs before selection, so all four probabilities must be present
    // and sum to one — a softmax applied after top-k would also "work" and
    // would change every weight.
    let fixture = DirectMoeFixture::new();
    let (_, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let total: f32 = trace.router_scores.iter().sum();
    assert!((total - 1.0).abs() < EPSILON, "scores sum to {total}");
}

#[test]
fn selected_expert_ids_and_order_match_the_oracle() {
    let fixture = DirectMoeFixture::new();
    let (_, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let expected: Vec<u32> = oracle(&input())
        .selected
        .iter()
        .map(|(id, _)| *id)
        .collect();
    assert_eq!(trace.selected_ids(), expected);
    assert_eq!(trace.selection.len(), TOP_K);
}

#[test]
fn selection_is_ordered_by_descending_score() {
    // Order is part of the contract, not an accident of the sort: the
    // reduction is commutative but the trace and any downstream consumer that
    // takes "the top expert" are not.
    let fixture = DirectMoeFixture::new();
    let (_, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let raw: Vec<f32> = trace.selection.iter().map(|s| s.raw_score).collect();
    assert!(raw.windows(2).all(|w| w[0] >= w[1]), "{raw:?}");
}

#[test]
fn gate_weights_match_the_oracle() {
    let fixture = DirectMoeFixture::new();
    let (_, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let expected: Vec<f32> = oracle(&input()).selected.iter().map(|(_, w)| *w).collect();
    assert_close(&trace.gate_weights(), &expected, "gate_weights");
}

#[test]
fn renormalisation_makes_the_selected_weights_sum_to_one() {
    let fixture = DirectMoeFixture::new();
    let (_, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let total: f32 = trace.gate_weights().iter().sum();
    assert!((total - 1.0).abs() < EPSILON, "weights sum to {total}");
}

#[test]
fn raw_scores_are_kept_alongside_renormalised_weights() {
    // Renormalisation can make two different routings produce the same final
    // weights; the raw score is where that disagreement is still visible.
    let fixture = DirectMoeFixture::new();
    let (_, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let raw_total: f32 = trace.selection.iter().map(|s| s.raw_score).sum();
    assert!(
        raw_total < 1.0,
        "top-{TOP_K} of {POPULATION} is not all of it"
    );
}

#[test]
fn per_expert_outputs_match_the_oracle() {
    let fixture = DirectMoeFixture::new();
    let (_, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let expected = oracle(&input());
    assert_eq!(trace.expert_outputs.len(), TOP_K);
    for (id, values) in &expected.expert_outputs {
        let actual = trace
            .expert_output(*id)
            .unwrap_or_else(|| panic!("expert {id} was not recorded"));
        assert_eq!(actual.len(), HIDDEN);
        assert_close(actual, values, &format!("expert {id} output"));
    }
}

#[test]
fn the_weighted_reduction_matches_the_oracle() {
    let fixture = DirectMoeFixture::new();
    let (_, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    assert_close(&trace.reduced, &oracle(&input()).reduced, "reduced");
}

#[test]
fn the_residual_delta_matches_the_oracle() {
    let fixture = DirectMoeFixture::new();
    let output = execute(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    assert_close(&output, &oracle(&input()).reduced, "residual delta");
}

#[test]
fn a_direct_moe_delta_equals_its_reduction() {
    // No transforms bound, so the two checkpoints must coincide. If they ever
    // differ, a stage is being applied that nothing bound.
    let fixture = DirectMoeFixture::new();
    let (output, trace) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    assert_eq!(trace.reduced, trace.residual_delta);
    assert_eq!(output, trace.residual_delta);
}

// ── The recipe, not the arrangement ────────────────────────────────────────

#[test]
fn fused_and_decomposed_storage_produce_identical_output() {
    // The headline property: same semantics, two physical layouts, one
    // executor, byte-identical result.
    let fixture = DirectMoeFixture::new();
    let decomposed = execute(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let fused = execute(
        &fixture.operation(ProjectionArrangement::Fused),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    assert_eq!(decomposed, fused, "arrangement changed the answer");
}

#[test]
fn fused_and_decomposed_agree_at_every_checkpoint() {
    // Equal outputs could in principle survive compensating errors upstream.
    let fixture = DirectMoeFixture::new();
    let (_, a) = execute_traced(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    let (_, b) = execute_traced(
        &fixture.operation(ProjectionArrangement::Fused),
        MoeInputs::shared(&input()),
    )
    .unwrap();
    assert_eq!(a, b);
}

#[test]
fn the_fused_up_half_is_the_second_block_not_every_other_row() {
    // The discriminating test for the concatenated-vs-interleaved bug. Reading
    // the fused region interleaved would return unit 0's up row where unit 1's
    // belongs — same shape, wrong values, no error anywhere downstream.
    let fixture = DirectMoeFixture::new();
    let op = fixture.operation(ProjectionArrangement::Fused);
    let expert = 2usize;
    let unit = 1usize;
    let projection = &op.banks[0].experts[expert].projection;
    assert!(matches!(projection, BoundProjection::Fused { .. }));

    let mut row = vec![0.0f32; HIDDEN];
    projection.up_row_into(unit, &mut row).unwrap();

    let correct: Vec<f32> = (0..HIDDEN).map(|h| up_at(expert, unit, h)).collect();
    let interleaved: Vec<f32> = (0..HIDDEN).map(|h| up_at(expert, 0, h)).collect();
    assert_close(&row, &correct, "fused up row");
    assert_ne!(
        correct, interleaved,
        "the fixture must discriminate the bug"
    );
}

#[test]
fn both_arrangements_report_the_same_dimensions() {
    let fixture = DirectMoeFixture::new();
    for arrangement in ProjectionArrangement::ALL {
        let op = fixture.operation(arrangement);
        let projection = &op.banks[0].experts[0].projection;
        assert_eq!(
            projection.intermediate_dim(),
            INTERMEDIATE,
            "{}",
            arrangement.name()
        );
        assert_eq!(projection.hidden_dim(), HIDDEN, "{}", arrangement.name());
    }
}

// ── The traced and untraced paths are one implementation ───────────────────

#[test]
fn tracing_does_not_change_the_result() {
    let fixture = DirectMoeFixture::new();
    let op = fixture.operation(ProjectionArrangement::Decomposed);
    let plain = execute(&op, MoeInputs::shared(&input())).unwrap();
    let (traced, _) = execute_traced(&op, MoeInputs::shared(&input())).unwrap();
    assert_eq!(plain, traced);
}

#[test]
fn execution_is_deterministic_across_runs() {
    let fixture = DirectMoeFixture::new();
    let op = fixture.operation(ProjectionArrangement::Fused);
    let first = execute(&op, MoeInputs::shared(&input())).unwrap();
    let second = execute(&op, MoeInputs::shared(&input())).unwrap();
    assert_eq!(first, second);
}

// ── Refusals ───────────────────────────────────────────────────────────────

#[test]
fn a_residual_of_the_wrong_width_is_refused_naming_both_widths() {
    let fixture = DirectMoeFixture::new();
    let err = execute(
        &fixture.operation(ProjectionArrangement::Decomposed),
        MoeInputs::shared(&[1.0, 2.0]),
    )
    .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("residual"), "{text}");
    assert!(text.contains(&HIDDEN.to_string()), "{text}");
}
