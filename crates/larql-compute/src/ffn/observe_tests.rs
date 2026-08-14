//! Tests for [`super::observe`] — split out to keep the module within
//! the one-concept-per-file size budget.

use super::observe::*;
use ndarray::Array2;

#[test]
fn unobserved_is_absent_with_the_named_reason() {
    let obs = FfnActivations::unobserved();
    assert!(obs.is_absent());
    assert_eq!(obs.absent_reason(), Some(REASON_UNOBSERVED_BACKEND));
    assert_eq!(obs.into_dense(), None);
}

#[test]
fn dense_round_trips_through_into_dense() {
    let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let obs = FfnActivations::Dense(a.clone());
    assert!(!obs.is_absent());
    assert_eq!(obs.absent_reason(), None);
    assert_eq!(obs.into_dense(), Some(a));
}

#[test]
fn sparse_records_and_densifies_only_what_was_computed() {
    let mut s = SparseActivations::new(2, 4);
    assert_eq!(s.seq_len(), 2);
    assert_eq!(s.num_features(), 4);
    s.record(0, 1, 0.5);
    s.record(1, 3, -2.0);
    assert_eq!(
        s.position(0),
        &[SparseEntry {
            feature: 1,
            activation: 0.5,
            gate_score: None,
            up_score: None,
        }]
    );
    assert_eq!(s.position(1)[0].feature, 3);
    assert_eq!(s.position(1)[0].activation, -2.0);
    let dense = FfnActivations::Sparse(s).into_dense().unwrap();
    assert_eq!(dense.shape(), &[2, 4]);
    assert_eq!(dense[[0, 1]], 0.5);
    assert_eq!(dense[[1, 3]], -2.0);
    // Every unrecorded slot is zero — the honest zero of "this
    // feature contributed nothing to the executed output".
    assert_eq!(dense.iter().filter(|v| **v != 0.0).count(), 2);
}

/// Item-17 seam: scored records carry the projection scores the
/// executed kernel used; plain records stay honestly unscored.
#[test]
fn record_scored_carries_the_kernel_scores() {
    let mut s = SparseActivations::new(1, 4);
    s.record_scored(0, 2, 0.75, Some(1.5), Some(0.5));
    s.record(0, 3, -0.25);
    let entries = s.position(0);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].gate_score, Some(1.5));
    assert_eq!(entries[0].up_score, Some(0.5));
    assert_eq!(entries[0].activation, 0.75);
    assert_eq!(entries[1].gate_score, None);
    assert_eq!(entries[1].up_score, None);
    // Scores don't perturb densification.
    let dense = s.to_dense();
    assert_eq!(dense[[0, 2]], 0.75);
    assert_eq!(dense[[0, 3]], -0.25);
}

/// Item-17 seam: per-position kernel labels name which sub-path
/// computed each position (walk positions can route differently).
#[test]
fn kernel_labels_are_per_position() {
    let mut s = SparseActivations::new(2, 2);
    assert_eq!(s.kernel(0), None);
    s.set_kernel(0, "sparse:serial");
    assert_eq!(s.kernel(0), Some("sparse:serial"));
    assert_eq!(s.kernel(1), None, "labels must not leak across positions");
}

#[test]
fn sparse_duplicate_records_accumulate() {
    let mut s = SparseActivations::new(1, 2);
    s.record(0, 0, 1.0);
    s.record(0, 0, 0.25);
    assert_eq!(s.to_dense()[[0, 0]], 1.25);
}

#[test]
#[should_panic(expected = "feature 5 >= num_features 2")]
fn sparse_record_panics_on_out_of_range_feature() {
    SparseActivations::new(1, 2).record(0, 5, 1.0);
}

#[test]
fn sparse_equality_is_structural() {
    let mut a = SparseActivations::new(1, 3);
    a.record(0, 2, 1.0);
    let mut b = SparseActivations::new(1, 3);
    b.record(0, 2, 1.0);
    assert_eq!(FfnActivations::Sparse(a), FfnActivations::Sparse(b.clone()));
    b.record(0, 1, 4.0);
    assert!(FfnActivations::Sparse(b) != FfnActivations::unobserved());
}

/// Scores and kernel labels participate in structural equality — two
/// observations that executed differently must not compare equal.
#[test]
fn scores_and_kernels_participate_in_equality() {
    let mut a = SparseActivations::new(1, 3);
    a.record_scored(0, 2, 1.0, Some(0.5), None);
    let mut b = SparseActivations::new(1, 3);
    b.record(0, 2, 1.0);
    assert_ne!(a, b, "scored vs unscored entries must differ");

    let mut c = SparseActivations::new(1, 3);
    c.record_scored(0, 2, 1.0, Some(0.5), None);
    assert_eq!(a, c);
    c.set_kernel(0, "sparse:gather_q4k");
    assert_ne!(a, c, "kernel labels must differ");
}
