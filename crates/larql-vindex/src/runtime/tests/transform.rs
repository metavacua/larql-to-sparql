//! Colocated tests for `transform` — the latent stages.
//!
//! A direct MoE binds no transforms, so nothing in fixture A exercises this
//! file. It is the machinery Mini-K3 needs (`residual → routed_input → bank →
//! routed_output → residual`), and leaving it untested until then would mean
//! discovering its faults mixed in with K3 router semantics.

use super::support::{ascending, matrix};
use crate::runtime::error::ExecutionError;
use crate::runtime::transform::{apply_stage, BoundTransform, TransformStage};

const ROUTED_IN: &str = "latent_in";
const ROUTED_OUT: &str = "latent_out";

/// `[out, in]` = [2, 3]: rows [1,2,3] and [4,5,6].
fn down_project() -> BoundTransform<'static> {
    BoundTransform {
        stage: TransformStage::RoutedInput,
        weight: ascending(ROUTED_IN, 2, 3),
    }
}

/// `[out, in]` = [3, 2], projecting back to the residual width.
fn up_project() -> BoundTransform<'static> {
    BoundTransform {
        stage: TransformStage::RoutedOutput,
        weight: ascending(ROUTED_OUT, 3, 2),
    }
}

// ── One transform ──────────────────────────────────────────────────────────

#[test]
fn a_transform_contracts_the_input_axis() {
    // [1,2,3]·[1,1,1] = 6; [4,5,6]·[1,1,1] = 15.
    let out = down_project().apply(&[1.0, 1.0, 1.0]).unwrap();
    assert_eq!(out, vec![6.0, 15.0]);
}

#[test]
fn a_transform_reports_the_widths_it_maps_between() {
    let t = down_project();
    assert_eq!(t.in_dim(), 3);
    assert_eq!(t.out_dim(), 2);
}

#[test]
fn an_input_of_the_wrong_width_is_refused_naming_both() {
    let err = down_project().apply(&[1.0, 1.0]).unwrap_err();
    let ExecutionError::DimensionMismatch {
        expected, found, ..
    } = err
    else {
        panic!("expected a dimension mismatch");
    };
    assert_eq!((expected, found), (3, 2));
}

#[test]
fn a_transform_describes_its_stage_and_widths() {
    let text = down_project().describe();
    assert!(text.contains(TransformStage::RoutedInput.name()), "{text}");
    assert!(text.contains("3→2"), "{text}");
}

#[test]
fn stages_name_the_two_sides_of_the_bank() {
    assert_eq!(TransformStage::RoutedInput.name(), "routed_input");
    assert_eq!(TransformStage::RoutedOutput.name(), "routed_output");
    assert_ne!(TransformStage::RoutedInput, TransformStage::RoutedOutput);
}

// ── Stage application ──────────────────────────────────────────────────────

#[test]
fn an_empty_transform_list_is_the_identity() {
    // The direct-MoE case: no stage bound means the residual passes through
    // unchanged, with no branch anywhere that could skip a bound projection.
    let input = vec![1.0, -2.0, 3.0];
    let out = apply_stage(&[], TransformStage::RoutedInput, &input).unwrap();
    assert_eq!(out, input);
}

#[test]
fn only_transforms_bound_to_the_requested_stage_are_applied() {
    let transforms = [down_project(), up_project()];
    let out = apply_stage(&transforms, TransformStage::RoutedInput, &[1.0, 1.0, 1.0]).unwrap();
    assert_eq!(out, vec![6.0, 15.0], "the output stage must not run here");
}

#[test]
fn the_output_stage_projects_back_to_the_residual_width() {
    let transforms = [down_project(), up_project()];
    let out = apply_stage(&transforms, TransformStage::RoutedOutput, &[1.0, 1.0]).unwrap();
    assert_eq!(out, vec![3.0, 7.0, 11.0]);
}

#[test]
fn chained_transforms_apply_in_declaration_order() {
    // [2,3] then [3,2]: order matters, and the reversed pair would not even
    // typecheck dimensionally — which is the point.
    let first = BoundTransform {
        stage: TransformStage::RoutedInput,
        weight: matrix(ROUTED_IN, 2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    };
    let second = BoundTransform {
        stage: TransformStage::RoutedInput,
        weight: matrix(ROUTED_IN, 1, 2, &[1.0, 1.0]),
    };
    let out = apply_stage(
        &[first, second],
        TransformStage::RoutedInput,
        &[5.0, 7.0, 9.0],
    )
    .unwrap();
    assert_eq!(out, vec![12.0], "5 + 7");
}

#[test]
fn a_stage_mismatch_in_a_chain_is_refused() {
    let mismatched = BoundTransform {
        stage: TransformStage::RoutedInput,
        weight: ascending(ROUTED_IN, 2, 5),
    };
    assert!(apply_stage(
        &[down_project(), mismatched],
        TransformStage::RoutedInput,
        &[1.0, 1.0, 1.0]
    )
    .is_err());
}
