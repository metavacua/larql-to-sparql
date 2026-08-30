//! Colocated tests for `projection` — the fused/decomposed equivalence.
//!
//! The row-mapping tests here are the unit-level statement of what fixture A
//! asserts end to end. Both are worth having: the fixture proves the whole
//! layer agrees, these say precisely which row a given call returns.

use crate::format::lyrw2::region_role::RegionRole;

use super::support::{matrix, TEST_VARIANT};
use crate::runtime::error::ExecutionError;
use crate::runtime::projection::{BoundProjection, ProjectionArrangement};

const INTERMEDIATE: u32 = 2;
const HIDDEN: u32 = 3;

/// Gate rows [1,2,3] and [4,5,6]; up rows [7,8,9] and [10,11,12].
const GATE: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const UP: [f32; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

fn decomposed() -> BoundProjection<'static> {
    BoundProjection::Decomposed {
        gate: matrix(&RegionRole::Gate.name(), INTERMEDIATE, HIDDEN, &GATE),
        up: matrix(&RegionRole::Up.name(), INTERMEDIATE, HIDDEN, &UP),
    }
}

fn fused() -> BoundProjection<'static> {
    let mut values = GATE.to_vec();
    values.extend_from_slice(&UP);
    BoundProjection::Fused {
        gate_up: matrix(
            &RegionRole::GateUpFused.name(),
            INTERMEDIATE * 2,
            HIDDEN,
            &values,
        ),
    }
}

fn row(projection: &BoundProjection<'_>, unit: usize, gate: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; HIDDEN as usize];
    if gate {
        projection.gate_row_into(unit, &mut out).unwrap();
    } else {
        projection.up_row_into(unit, &mut out).unwrap();
    }
    out
}

// ── Both arrangements yield the same rows ──────────────────────────────────

#[test]
fn decomposed_and_fused_return_identical_gate_rows() {
    for unit in 0..INTERMEDIATE as usize {
        assert_eq!(row(&decomposed(), unit, true), row(&fused(), unit, true));
    }
}

#[test]
fn decomposed_and_fused_return_identical_up_rows() {
    for unit in 0..INTERMEDIATE as usize {
        assert_eq!(row(&decomposed(), unit, false), row(&fused(), unit, false));
    }
}

#[test]
fn the_fused_up_half_starts_after_every_gate_row() {
    // Unit 1's up row is region row 3, not row 3-by-coincidence: with
    // INTERMEDIATE = 2 an interleaved reading would also land on row 3 for
    // unit 1, so this checks unit 0 too, where the two disagree.
    assert_eq!(row(&fused(), 0, false), vec![7.0, 8.0, 9.0]);
    assert_eq!(row(&fused(), 1, false), vec![10.0, 11.0, 12.0]);
}

#[test]
fn gate_rows_are_the_first_half() {
    assert_eq!(row(&fused(), 0, true), vec![1.0, 2.0, 3.0]);
    assert_eq!(row(&fused(), 1, true), vec![4.0, 5.0, 6.0]);
}

// ── Dimensions ─────────────────────────────────────────────────────────────

#[test]
fn both_arrangements_report_the_same_dimensions() {
    for projection in [decomposed(), fused()] {
        assert_eq!(projection.intermediate_dim(), INTERMEDIATE as usize);
        assert_eq!(projection.hidden_dim(), HIDDEN as usize);
    }
}

#[test]
fn a_fused_region_halves_its_row_count() {
    // The property that makes an odd row count detectable.
    let BoundProjection::Fused { gate_up } = fused() else {
        panic!("expected a fused projection");
    };
    assert_eq!(gate_up.rows(), (INTERMEDIATE * 2) as usize);
}

// ── Validation ─────────────────────────────────────────────────────────────

#[test]
fn both_arrangements_validate_against_the_layer_shape() {
    for projection in [decomposed(), fused()] {
        projection
            .validate(INTERMEDIATE as usize, HIDDEN as usize)
            .unwrap();
    }
}

#[test]
fn a_decomposed_region_bound_as_fused_is_refused_by_row_count() {
    // The clearest signal that the arrangement was misread: a region with
    // `intermediate` rows claiming to hold `2 * intermediate`.
    let wrong = BoundProjection::Fused {
        gate_up: matrix(&RegionRole::GateUpFused.name(), INTERMEDIATE, HIDDEN, &GATE),
    };
    let err = wrong
        .validate(INTERMEDIATE as usize, HIDDEN as usize)
        .unwrap_err();
    let ExecutionError::DimensionMismatch {
        expected, found, ..
    } = err
    else {
        panic!("expected a dimension mismatch");
    };
    assert_eq!((expected, found), (4, 2));
}

#[test]
fn a_decomposed_pair_that_disagrees_on_width_is_refused() {
    let wrong = BoundProjection::Decomposed {
        gate: matrix(&RegionRole::Gate.name(), INTERMEDIATE, HIDDEN, &GATE),
        up: matrix(
            &RegionRole::Up.name(),
            INTERMEDIATE,
            2,
            &[1.0, 2.0, 3.0, 4.0],
        ),
    };
    assert!(wrong
        .validate(INTERMEDIATE as usize, HIDDEN as usize)
        .is_err());
}

#[test]
fn a_row_past_the_intermediate_width_is_refused() {
    let mut out = vec![0.0f32; HIDDEN as usize];
    assert!(fused()
        .up_row_into(INTERMEDIATE as usize, &mut out)
        .is_err());
    assert!(decomposed()
        .gate_row_into(INTERMEDIATE as usize, &mut out)
        .is_err());
}

// ── Naming comes from the role registry ────────────────────────────────────

#[test]
fn arrangements_name_themselves_from_the_roles_they_store() {
    assert_eq!(
        ProjectionArrangement::Decomposed.name(),
        format!("{}+{}", RegionRole::Gate.name(), RegionRole::Up.name())
    );
    assert_eq!(
        ProjectionArrangement::Fused.name(),
        RegionRole::GateUpFused.name()
    );
}

#[test]
fn every_arrangement_declares_the_roles_it_needs() {
    assert_eq!(
        ProjectionArrangement::Decomposed.roles(),
        vec![RegionRole::Gate, RegionRole::Up]
    );
    assert_eq!(
        ProjectionArrangement::Fused.roles(),
        vec![RegionRole::GateUpFused]
    );
    assert_eq!(ProjectionArrangement::ALL.len(), 2);
}

#[test]
fn a_projection_reports_which_arrangement_it_is() {
    assert_eq!(
        decomposed().arrangement(),
        ProjectionArrangement::Decomposed
    );
    assert_eq!(fused().arrangement(), ProjectionArrangement::Fused);
}

#[test]
fn describing_a_projection_names_its_arrangement_and_operands() {
    let text = decomposed().describe();
    assert!(
        text.contains(&ProjectionArrangement::Decomposed.name()),
        "{text}"
    );
    assert!(text.contains(&RegionRole::Gate.name()), "{text}");
    assert!(text.contains(TEST_VARIANT), "{text}");

    let fused_text = fused().describe();
    assert!(
        fused_text.contains(&RegionRole::GateUpFused.name()),
        "{fused_text}"
    );
}
