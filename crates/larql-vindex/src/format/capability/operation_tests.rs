//! Colocated tests for `operation` — admission verdicts and their diagnoses.
//!
//! Route-scoped authority is exercised in `plan_tests`; these cover the
//! admission verdict itself and the requirement that an operator can triage
//! from the failure text alone.

use crate::format::lyrw2::region_role::RegionRole;

use super::authority::Fidelity;
use super::coordinate::RegionCoordinate;
use super::operation::{Degradation, OperationAdmission, OperationCapability, OperationFailure};
use super::plan::{OperationPlan, PlannedRegion, QualifiedOperationRoute};

fn coordinate() -> RegionCoordinate {
    RegionCoordinate::new(3, 0, Some(1), RegionRole::Down)
}

fn route() -> QualifiedOperationRoute {
    QualifiedOperationRoute::new(OperationPlan::fixed_regions(vec![PlannedRegion::new(
        coordinate(),
        Fidelity::SourceExact,
    )]))
}

#[test]
fn available_and_degraded_both_admit() {
    assert!(OperationCapability::available(vec![route()]).is_available());
    assert!(OperationCapability::degraded(
        vec![route()],
        vec![Degradation::MissingQueryMetadata { what: "labels" }]
    )
    .is_available());
}

#[test]
fn unavailable_does_not_admit() {
    let c =
        OperationCapability::unavailable(vec![OperationFailure::NoExecutableRoute { layer: 3 }]);
    assert!(!c.is_available());
    assert!(c.routes.is_empty());
}

#[test]
fn an_invalid_selection_is_distinguishable_from_ordinary_unavailability() {
    let invalid = OperationCapability::unavailable(vec![OperationFailure::InvalidSelection {
        detail: "gate covers 0, down covers 1".into(),
    }]);
    let ordinary = OperationCapability::unavailable(vec![OperationFailure::MissingDocumentInput {
        what: "tokenizer",
    }]);
    assert!(invalid.admission.is_invalid_selection());
    assert!(!ordinary.admission.is_invalid_selection());
}

#[test]
fn an_available_operation_is_never_an_invalid_selection() {
    assert!(!OperationAdmission::Available.is_invalid_selection());
    assert!(!OperationAdmission::Degraded {
        reasons: Vec::new()
    }
    .is_invalid_selection());
}

#[test]
fn a_missing_contract_is_distinct_from_missing_bytes() {
    // Remote execution cannot be inferred from local absence; the two must
    // never read alike.
    let contract = OperationFailure::MissingContract {
        what: "routed wire contract",
    };
    let bytes = OperationFailure::RequiredRegionUnusable {
        coordinate: coordinate(),
        cause: "absent".into(),
    };
    assert!(contract.describe().contains("cannot be inferred"));
    assert!(!bytes.describe().contains("cannot be inferred"));
}

#[test]
fn failures_carry_exact_coordinates() {
    let f = OperationFailure::RequiredRegionUnusable {
        coordinate: coordinate(),
        cause: "absent from every selected segment".into(),
    };
    let s = f.describe();
    for part in ["layer 3", "bank 0", "role down", "segment 1"] {
        assert!(s.contains(part), "{s} missing {part}");
    }
}

#[test]
fn every_failure_kind_renders_distinguishably() {
    let rendered: Vec<String> = [
        OperationFailure::RequiredRegionUnusable {
            coordinate: coordinate(),
            cause: "absent".into(),
        },
        OperationFailure::InvalidSelection {
            detail: "disjoint segments".into(),
        },
        OperationFailure::MissingDocumentInput { what: "embeddings" },
        OperationFailure::MissingContract {
            what: "routed wire",
        },
        OperationFailure::NoExecutableRoute { layer: 7 },
    ]
    .iter()
    .map(|f| f.describe())
    .collect();
    for i in 0..rendered.len() {
        for j in (i + 1)..rendered.len() {
            assert_ne!(rendered[i], rendered[j], "{i} vs {j}");
        }
    }
    assert!(rendered[1].contains("invalid selection"));
    assert!(rendered[4].contains("layer 7"));
}

#[test]
fn missing_query_metadata_says_correctness_is_unchanged() {
    // §15.3: absent query metadata downgrades richness, never correctness. The
    // wording matters — an operator must not read it as "results may be wrong".
    let d = Degradation::MissingQueryMetadata {
        what: "feature_labels.json",
    };
    assert!(
        d.describe().contains("unchanged correctness"),
        "{}",
        d.describe()
    );
}

#[test]
fn a_non_browsable_bank_degrades_rather_than_failing() {
    let d = Degradation::BankNotBrowsable {
        bank_id: 4,
        reason: "fused region cannot be strided".into(),
    };
    let s = d.describe();
    assert!(s.contains("bank 4"), "{s}");
    assert!(s.contains("cannot be strided"), "{s}");
}

#[test]
fn admission_descriptions_state_the_verdict_first() {
    assert!(OperationAdmission::Available
        .describe()
        .starts_with("available"));
    assert!(OperationAdmission::Degraded {
        reasons: vec![Degradation::MissingQueryMetadata { what: "labels" }],
    }
    .describe()
    .starts_with("available, degraded"));
    assert!(OperationAdmission::Unavailable {
        reasons: vec![OperationFailure::NoExecutableRoute { layer: 0 }],
    }
    .describe()
    .starts_with("unavailable"));
}
