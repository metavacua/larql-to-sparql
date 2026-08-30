//! What happens when routing selects an expert this bank does not hold.
//!
//! Relaxing `validate` to permit shards created an execution requirement:
//!
//! > If routing selects an expert absent from the bound local bank, execution
//! > must produce an explicit missing-expert result — not skip it, not
//! > renormalise around it, not substitute another expert.
//!
//! Each of those three would produce a token quietly missing part of its FFN
//! contribution, and it would read as entirely reasonable output. The request
//! has to survive to the caller so that placement can satisfy it; remote
//! fetching will later answer exactly this without touching router semantics.

use super::operation::direct;
use super::support::matrix;
use crate::runtime::execute::{execute, execute_traced};
use crate::runtime::inputs::MoeInputs;
use crate::runtime::operation::BoundMoeOperation;
use crate::runtime::ExecutionError;

const ROUTER: &str = "router";

/// A router that selects expert `e` by input component `e`, so the test
/// controls the selection by choosing the input.
fn addressable_router(op: &mut BoundMoeOperation<'static>) {
    let population = op.router.population();
    let hidden = op.residual_dim;
    let mut weights = vec![0.0f32; population * hidden];
    for (e, row) in weights.chunks_mut(hidden).enumerate() {
        row[e] = 1.0;
    }
    op.router.weight = matrix(ROUTER, population as u32, hidden as u32, &weights);
}

/// A shard holding experts 0 and 1 only, out of a population of 4.
fn shard() -> BoundMoeOperation<'static> {
    let mut op = direct();
    addressable_router(&mut op);
    op.banks[0].experts.truncate(2);
    op.banks[0].experts[0].expert_id = 0;
    op.banks[0].experts[1].expert_id = 1;
    op.router.top_k = 1;
    op
}

// ── The shard is legitimate ────────────────────────────────────────────────

#[test]
fn a_shard_validates_and_reports_that_it_is_partial() {
    let op = shard();
    op.validate().expect("a shard is a valid operation");
    assert!(!op.holds_full_population());
    assert_eq!(op.banks[0].population(), 2);
    assert_eq!(op.router.population(), 4);
}

#[test]
fn a_shard_serves_a_selection_it_holds() {
    // Component 1 largest → expert 1, which this shard has.
    let op = shard();
    let input = vec![0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
    let (out, trace) = execute_traced(&op, MoeInputs::shared(&input)).unwrap();
    assert_eq!(trace.selected_ids(), vec![1]);
    assert!(out.iter().any(|v| *v != 0.0));
}

// ── The requirement ────────────────────────────────────────────────────────

#[test]
fn a_selection_the_shard_lacks_is_reported_not_skipped() {
    // Component 3 largest → expert 3, which this shard does not hold.
    let op = shard();
    let input = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
    let err = execute(&op, MoeInputs::shared(&input)).unwrap_err();
    let ExecutionError::SelectedExpertNotResident {
        expert,
        bank,
        resident,
        population,
    } = &err
    else {
        panic!("expected a placement result, got {err}");
    };
    assert_eq!(*expert, 3, "the report must name the selected expert");
    assert_eq!((*resident, *population), (2, 4));
    assert!(
        bank.contains("bank"),
        "must name the bank coordinate: {bank}"
    );
}

#[test]
fn the_report_names_the_layer_and_bank_it_looked_in() {
    // A whole-model run has many banks; "expert 3 is missing" without saying
    // from where sends the reader hunting.
    let mut op = shard();
    op.banks[0].bank = crate::format::capability::coordinate::BankCoordinate::new(17, 2);
    let input = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
    let text = execute(&op, MoeInputs::shared(&input))
        .unwrap_err()
        .to_string();
    assert!(text.contains("layer 17"), "{text}");
    assert!(text.contains("bank 2"), "{text}");
    assert!(text.contains('3'), "{text}");
}

#[test]
fn a_missing_expert_is_distinct_from_an_unaddressable_one() {
    // Two different repairs: fetch the operand, versus fix the index. The
    // error types must not collapse, or an operator cannot tell which.
    let op = shard();
    let input = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
    let missing = execute(&op, MoeInputs::shared(&input)).unwrap_err();
    assert!(matches!(
        missing,
        ExecutionError::SelectedExpertNotResident { .. }
    ));

    let mut broken = shard();
    broken.banks[0].experts[0].expert_id = 99;
    let unaddressable = broken.validate().unwrap_err();
    assert!(matches!(
        unaddressable,
        ExecutionError::ExpertOutOfRange { .. }
    ));
}

#[test]
fn nothing_is_reduced_when_a_selected_expert_is_absent() {
    // The failure mode being excluded: producing a partial delta from whatever
    // experts happened to be resident. There is no partial answer here.
    let op = shard();
    let input = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
    assert!(execute(&op, MoeInputs::shared(&input)).is_err());
}

#[test]
fn a_partly_resident_selection_still_refuses_rather_than_contributing_what_it_has() {
    // top-2 where one expert is resident and one is not. Renormalising over
    // the survivor would produce a confident, wrong, correctly-shaped token.
    let mut op = shard();
    op.router.top_k = 2;
    let input = vec![0.0, 5.0, 0.0, 4.0, 0.0, 0.0]; // experts 1 (held) then 3 (not)
    let err = execute(&op, MoeInputs::shared(&input)).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::SelectedExpertNotResident { expert: 3, .. }
    ));
}

#[test]
fn a_bank_can_be_asked_whether_it_holds_an_expert_without_failing() {
    // Placement needs to ask before executing; asking must not be an error.
    let op = shard();
    assert!(op.banks[0].holds(1));
    assert!(!op.banks[0].holds(3));
}

#[test]
fn routing_is_unchanged_by_what_the_shard_happens_to_hold() {
    // The selection is a property of the router and its input alone. If a
    // shard could influence it, two shards of one model would route
    // differently and the model would depend on its own sharding.
    let full = {
        let mut op = direct();
        addressable_router(&mut op);
        op.router.top_k = 1;
        op
    };
    let input = vec![0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
    let (_, full_trace) = execute_traced(&full, MoeInputs::shared(&input)).unwrap();
    let (_, shard_trace) = execute_traced(&shard(), MoeInputs::shared(&input)).unwrap();
    assert_eq!(full_trace.selected_ids(), shard_trace.selected_ids());
    assert_eq!(full_trace.router_scores, shard_trace.router_scores);
}
