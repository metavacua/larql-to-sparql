//! The tie-breaking contract, pinned before Gemma parity.
//!
//! Exact ties in router scores are rare and entirely possible — quantised
//! weights and small populations both make them more likely, and a tie is
//! exactly the case where two correct-looking implementations diverge.
//!
//! The contract:
//!
//! ```text
//! score descending, then expert id ascending
//! total order over all finite scores
//! non-finite scores refused, never ordered
//! identical inputs → identical selection, byte for byte
//! ```
//!
//! These tests exist so that a Gemma divergence at a tie is diagnosed as a
//! selection-policy difference in minutes rather than mistaken for a weight
//! decoding fault.

use super::operation::direct;
use super::support::matrix;
use crate::runtime::error::ExecutionError;
use crate::runtime::execute::execute_traced;
use crate::runtime::inputs::MoeInputs;
use crate::runtime::kernels::{top_k, top_k_with_margin};

const ROUTER: &str = "router";

// ── The three cases named in the contract ──────────────────────────────────

#[test]
fn a_tie_inside_the_selected_top_k_orders_by_expert_id() {
    // Both experts are selected either way; the question is the order they
    // appear in, which the trace and any "top expert" consumer depend on.
    let picked = top_k(&[0.9, 0.5, 0.5, 0.1], 3);
    assert_eq!(
        picked.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        vec![0, 1, 2],
        "tied experts 1 and 2 must appear in id order"
    );
}

#[test]
fn a_tie_across_the_top_k_boundary_selects_the_lower_expert_id() {
    // The consequential case: one of the tied pair runs and the other does
    // not, so the policy changes the output rather than just its ordering.
    let (picked, margin) = top_k_with_margin(&[0.9, 0.5, 0.5, 0.1], 2);
    assert_eq!(
        picked.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(margin, Some(0.0), "the boundary was decided by a tie");
}

#[test]
fn the_same_tied_routing_repeated_gives_a_byte_identical_selection() {
    let scores = [0.25f32, 0.25, 0.25, 0.25];
    let first = top_k(&scores, 2);
    for _ in 0..64 {
        assert_eq!(top_k(&scores, 2), first);
    }
    assert_eq!(
        first.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        vec![0, 1]
    );
}

// ── The margin, as parity triage ───────────────────────────────────────────

#[test]
fn a_clear_boundary_reports_the_gap_that_decided_it() {
    let (_, margin) = top_k_with_margin(&[0.9, 0.7, 0.2, 0.1], 2);
    let margin = margin.expect("a boundary exists");
    assert!((margin - 0.5).abs() < 1e-6, "0.7 - 0.2 = {margin}");
}

#[test]
fn there_is_no_margin_when_the_selection_covers_the_population() {
    assert_eq!(top_k_with_margin(&[0.6, 0.4], 2).1, None);
    assert_eq!(top_k_with_margin(&[0.6, 0.4], 9).1, None);
}

#[test]
fn there_is_no_margin_when_nothing_is_selected() {
    assert_eq!(top_k_with_margin(&[0.6, 0.4], 0).1, None);
}

#[test]
fn a_tie_decided_selection_is_flagged_in_the_trace() {
    // A router whose rows are all identical scores every expert equally, so
    // the boundary is a pure tie and top-k is decided entirely by policy.
    let mut op = direct();
    let population = op.banks[0].experts.len();
    let hidden = op.residual_dim;
    let uniform = vec![0.25f32; population * hidden];
    op.router.weight = matrix(ROUTER, population as u32, hidden as u32, &uniform);

    let (_, trace) = execute_traced(&op, MoeInputs::shared(&vec![0.1f32; hidden])).unwrap();
    assert!(
        trace.selection_was_decided_by_a_tie(),
        "margin was {:?}",
        trace.selection_margin
    );
    assert_eq!(trace.selected_ids(), vec![0, 1], "lowest ids win the tie");
}

#[test]
fn an_ordinary_selection_is_not_flagged_as_a_tie() {
    let (_, trace) = execute_traced(
        &direct(),
        MoeInputs::shared(&[0.4, -0.3, 0.9, 0.1, -0.7, 0.2]),
    )
    .unwrap();
    assert!(!trace.selection_was_decided_by_a_tie());
    assert!(trace.selection_margin.unwrap() > 0.0);
}

// ── A total order, not a fallback ──────────────────────────────────────────

#[test]
fn ordering_is_total_over_signed_zeroes() {
    // `partial_cmp` calls +0.0 and -0.0 equal, so the id tie-break decides;
    // `total_cmp` orders them. Either way the result must be deterministic,
    // which a fallback-to-Equal sort does not guarantee.
    let first = top_k(&[0.0, -0.0, 0.0], 2);
    for _ in 0..32 {
        assert_eq!(top_k(&[0.0, -0.0, 0.0], 2), first);
    }
}

#[test]
fn every_permutation_of_a_fully_tied_population_selects_the_same_ids() {
    // Position in the input must not influence selection when scores are equal.
    for population in 2..8usize {
        let scores = vec![0.5f32; population];
        let picked: Vec<usize> = top_k(&scores, 2).iter().map(|(i, _)| *i).collect();
        assert_eq!(picked, vec![0, 1], "population {population}");
    }
}

// ── Non-finite scores are refused, never ordered ───────────────────────────

#[test]
fn a_nan_router_score_is_refused_naming_the_expert() {
    // A NaN would otherwise take whatever position the total order gives it
    // and be selected or dropped by sort mechanics.
    let mut op = direct();
    let population = op.banks[0].experts.len();
    let hidden = op.residual_dim;
    let mut values = vec![0.25f32; population * hidden];
    values[hidden] = f32::NAN; // expert 1's first weight
    op.router.weight = matrix(ROUTER, population as u32, hidden as u32, &values);

    let err = execute_traced(&op, MoeInputs::shared(&vec![0.5f32; hidden])).unwrap_err();
    let ExecutionError::NonFiniteRouterScore { expert, value } = err else {
        panic!("expected a non-finite refusal, got {err}");
    };
    assert_eq!(expert, 1);
    assert!(value.is_nan());
}

#[test]
fn an_infinite_router_weight_is_refused_rather_than_dominating_the_softmax() {
    let mut op = direct();
    let population = op.banks[0].experts.len();
    let hidden = op.residual_dim;
    let mut values = vec![0.25f32; population * hidden];
    values[0] = f32::INFINITY;
    op.router.weight = matrix(ROUTER, population as u32, hidden as u32, &values);

    // softmax(inf) yields NaN once the max-shift subtracts inf from inf.
    let err = execute_traced(&op, MoeInputs::shared(&vec![0.5f32; hidden])).unwrap_err();
    assert!(
        matches!(err, ExecutionError::NonFiniteRouterScore { .. }),
        "{err}"
    );
}

#[test]
fn finite_scores_are_never_refused() {
    let (_, trace) = execute_traced(
        &direct(),
        MoeInputs::shared(&[0.4, -0.3, 0.9, 0.1, -0.7, 0.2]),
    )
    .unwrap();
    assert!(trace.router_scores.iter().all(|s| s.is_finite()));
}
