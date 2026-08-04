//! Colocated tests for `reduction`.

use crate::runtime::axis::Axis;
use crate::runtime::error::ExecutionError;
use crate::runtime::reduction::BoundReduction;

#[test]
fn a_weighted_sum_accumulates_scaled_contributions() {
    let mut acc = vec![0.0f32; 3];
    BoundReduction::WeightedSum
        .accumulate(&mut acc, &[1.0, 2.0, 3.0], 0.5)
        .unwrap();
    BoundReduction::WeightedSum
        .accumulate(&mut acc, &[2.0, 2.0, 2.0], 0.25)
        .unwrap();
    assert_eq!(acc, vec![1.0, 1.5, 2.0]);
}

#[test]
fn accumulation_adds_rather_than_replaces() {
    // The bug this guards: a second expert overwriting the first's
    // contribution produces output that looks entirely plausible.
    let mut acc = vec![1.0f32; 2];
    BoundReduction::WeightedSum
        .accumulate(&mut acc, &[1.0, 1.0], 1.0)
        .unwrap();
    assert_eq!(acc, vec![2.0, 2.0]);
}

#[test]
fn a_zero_weight_contributes_nothing() {
    let mut acc = vec![5.0f32; 2];
    BoundReduction::WeightedSum
        .accumulate(&mut acc, &[9.0, 9.0], 0.0)
        .unwrap();
    assert_eq!(acc, vec![5.0, 5.0]);
}

#[test]
fn a_width_disagreement_is_refused_naming_the_output_axis() {
    let mut acc = vec![0.0f32; 3];
    let err = BoundReduction::WeightedSum
        .accumulate(&mut acc, &[1.0, 2.0], 1.0)
        .unwrap_err();
    let ExecutionError::DimensionMismatch {
        axis,
        expected,
        found,
        operand,
    } = err
    else {
        panic!("expected a dimension mismatch");
    };
    assert_eq!(axis, Axis::OutputWidth);
    assert_eq!((expected, found), (3, 2));
    assert!(
        operand.contains(BoundReduction::WeightedSum.name()),
        "{operand}"
    );
}

#[test]
fn the_reduction_names_itself() {
    assert_eq!(BoundReduction::WeightedSum.name(), "weighted_sum");
}
