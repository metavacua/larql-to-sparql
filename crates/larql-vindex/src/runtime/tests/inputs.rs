//! Colocated tests for `inputs` — routing on a vector the experts never see.
//!
//! Gemma applies a router-specific norm, scale and scalar on top of the expert
//! input, so its router scores a different vector. Before this existed the
//! operation would have scored on the expert input, selected different experts,
//! and produced a wrong answer with no shape error anywhere.

use super::operation::direct;
use crate::runtime::execute::{execute, execute_traced};
use crate::runtime::inputs::MoeInputs;

const RESIDUAL_WIDTH: usize = 6;

fn bank_input() -> Vec<f32> {
    vec![0.4, -0.3, 0.9, 0.1, -0.7, 0.2]
}

/// A router input that is emphatically not the bank input — the shape a
/// learned router norm and scale produce.
fn router_input() -> Vec<f32> {
    vec![-0.9, 0.8, -0.2, 0.6, 0.5, -0.4]
}

// ── Construction ───────────────────────────────────────────────────────────

#[test]
fn shared_inputs_point_both_reads_at_one_vector() {
    let x = bank_input();
    let inputs = MoeInputs::shared(&x);
    assert_eq!(inputs.bank, inputs.router);
    assert!(inputs.is_shared());
}

#[test]
fn split_inputs_keep_the_two_apart() {
    let bank = bank_input();
    let router = router_input();
    let inputs = MoeInputs::split(&bank, &router);
    assert_eq!(inputs.bank, bank.as_slice());
    assert_eq!(inputs.router, router.as_slice());
    assert!(!inputs.is_shared());
}

#[test]
fn sharing_is_decided_by_value_not_by_pointer() {
    // Two separately-computed vectors that happen to be equal are shared for
    // every purpose that matters, and a model whose router norm is the
    // identity produces exactly that.
    let bank = bank_input();
    let copy = bank_input();
    assert!(MoeInputs::split(&bank, &copy).is_shared());
}

#[test]
fn inputs_describe_which_shape_they_are() {
    let bank = bank_input();
    let router = router_input();
    assert!(MoeInputs::shared(&bank).describe().contains("shared"));
    assert!(MoeInputs::split(&bank, &router)
        .describe()
        .contains("split"));
}

// ── The behaviour the split exists for ─────────────────────────────────────

/// An operation whose router selects expert `e` by input component `e`.
///
/// `direct()`'s router uses ascending weights, which makes its selection
/// almost input-insensitive — the high-index rows dominate whatever it is
/// given. A first version of the test below used it and saw scores move while
/// the selection did not, which proves nothing about routing. A one-hot router
/// makes the selection readable by hand: top-k over the input's own largest
/// components.
fn selection_follows_input() -> crate::runtime::operation::BoundMoeOperation<'static> {
    use super::support::matrix;
    let mut op = direct();
    let population = op.banks[0].experts.len();
    let hidden = op.residual_dim;
    let mut weights = vec![0.0f32; population * hidden];
    for (e, row) in weights.chunks_mut(hidden).enumerate() {
        row[e] = 1.0;
    }
    op.router.weight = matrix("router", population as u32, hidden as u32, &weights);
    op
}

#[test]
fn the_router_input_decides_the_selection() {
    // Same experts, same bank input, different router input. If routing read
    // the bank input, these would be identical — which is precisely the bug a
    // single-input model would have shipped.
    let op = selection_follows_input();
    let bank = bank_input();
    let router = router_input();

    let (_, shared) = execute_traced(&op, MoeInputs::shared(&bank)).unwrap();
    let (_, split) = execute_traced(&op, MoeInputs::split(&bank, &router)).unwrap();

    assert_ne!(
        shared.router_scores, split.router_scores,
        "the router must score the vector it was given"
    );
    // bank   = [0.4, -0.3, 0.9, 0.1, ...] → components 2 then 0
    // router = [-0.9, 0.8, -0.2, 0.6, ...] → components 1 then 3
    assert_eq!(shared.selected_ids(), vec![2, 0]);
    assert_eq!(split.selected_ids(), vec![1, 3]);
}

#[test]
fn the_bank_input_decides_the_expert_arithmetic() {
    // The mirror: hold the router input fixed, change what the experts see.
    // Selection must not move; the outputs must.
    let op = direct();
    let router = router_input();
    let bank_a = bank_input();
    let bank_b: Vec<f32> = bank_a.iter().map(|v| v * 0.5).collect();

    let (out_a, trace_a) = execute_traced(&op, MoeInputs::split(&bank_a, &router)).unwrap();
    let (out_b, trace_b) = execute_traced(&op, MoeInputs::split(&bank_b, &router)).unwrap();

    assert_eq!(
        trace_a.selected_ids(),
        trace_b.selected_ids(),
        "selection is a function of the router input alone"
    );
    assert_eq!(trace_a.gate_weights(), trace_b.gate_weights());
    assert_ne!(out_a, out_b, "the experts read the bank input");
}

#[test]
fn a_shared_input_is_exactly_a_split_of_one_vector_with_itself() {
    // No special-casing: the shared path must not be a different code path.
    let op = direct();
    let x = bank_input();
    assert_eq!(
        execute(&op, MoeInputs::shared(&x)).unwrap(),
        execute(&op, MoeInputs::split(&x, &x)).unwrap()
    );
}

#[test]
fn the_recorded_router_scores_come_from_the_router_input() {
    // Verifiable independently: scoring the router input through a second
    // execution as a shared input must reproduce the same scores.
    let op = direct();
    let bank = bank_input();
    let router = router_input();

    let (_, split) = execute_traced(&op, MoeInputs::split(&bank, &router)).unwrap();
    let (_, router_as_shared) = execute_traced(&op, MoeInputs::shared(&router)).unwrap();
    assert_eq!(split.router_scores, router_as_shared.router_scores);
    assert_eq!(split.selected_ids(), router_as_shared.selected_ids());
}

// ── Widths ─────────────────────────────────────────────────────────────────

#[test]
fn the_bank_input_width_is_the_one_checked_against_the_residual_dim() {
    let op = direct();
    let router = router_input();
    let short = vec![0.1f32; RESIDUAL_WIDTH - 1];
    assert!(execute(&op, MoeInputs::split(&short, &router)).is_err());
}
