//! Tests for `verdict` — the classification a sweep reports per layer.
//!
//! Two properties carry the weight, and both are about what a *headline number*
//! ends up meaning:
//!
//! - a residency refusal must never read as a parity failure, or the sweep
//!   measures slice coverage instead of execution;
//! - an observed agreement must never read as a contract, or a
//!   schedule-dependent result becomes a promise nobody can keep.

use crate::runtime::axis::Axis;
use crate::runtime::error::{ExecutionError, OperandUnsuitability};
use crate::runtime::verdict::{RefusalKind, Verdict};

const OPERAND: &str = "gate_up_fused::test";
const KERNEL: &str = "incumbent_q4k_q8k";

/// One error of every variant, so the classification sweep below cannot narrow
/// silently when a variant is added.
fn every_error() -> Vec<ExecutionError> {
    vec![
        ExecutionError::UnsupportedFormat {
            format: "q4_k".into(),
            operand: OPERAND.into(),
        },
        ExecutionError::UnsupportedView {
            view: "transpose".into(),
            operand: OPERAND.into(),
        },
        ExecutionError::DimensionMismatch {
            operand: OPERAND.into(),
            axis: Axis::Rows,
            expected: 2,
            found: 3,
        },
        ExecutionError::ShortRegion {
            operand: OPERAND.into(),
            needed: 8,
            found: 4,
        },
        ExecutionError::RowOutOfRange {
            operand: OPERAND.into(),
            row: 9,
            rows: 2,
        },
        ExecutionError::NotAMatrix {
            operand: OPERAND.into(),
            found: "vector".into(),
        },
        ExecutionError::NonFiniteRouterScore {
            expert: 3,
            value: f32::NAN,
        },
        ExecutionError::NonFiniteExpertScale {
            expert: 3,
            value: f32::INFINITY,
            operand: OPERAND.into(),
        },
        ExecutionError::KernelOperandUnsuitable {
            kernel: KERNEL,
            operand: OPERAND.into(),
            reason: OperandUnsuitability::NonDirectView {
                view: "transpose".into(),
            },
        },
        ExecutionError::KernelActivationUnsupported {
            kernel: KERNEL,
            activation: "ReLU".into(),
        },
        ExecutionError::ExpertOutOfRange {
            expert: 900,
            population: 128,
        },
        ExecutionError::SelectedExpertNotResident {
            expert: 90,
            bank: "layer 5 bank 0".into(),
            resident: 8,
            population: 128,
        },
    ]
}

// ── Residency is not a failure ─────────────────────────────────────────────

#[test]
fn a_non_resident_expert_is_a_residency_refusal_and_nothing_else() {
    // The whole reason this taxonomy exists. The route was correct, the expert
    // exists in the model, and a shard that does not hold it is behaving as
    // designed — so this must not indict the execution.
    let err = ExecutionError::SelectedExpertNotResident {
        expert: 90,
        bank: "layer 5 bank 0".into(),
        resident: 8,
        population: 128,
    };
    assert_eq!(err.refusal(), RefusalKind::Residency);

    let verdict = Verdict::from(&err);
    assert_eq!(verdict, Verdict::Refused(RefusalKind::Residency));
    assert!(!verdict.indicts_execution());
    assert!(!verdict.agreed());
    assert!(!verdict.is_contracted());
}

#[test]
fn an_out_of_range_expert_is_a_binding_defect_not_a_residency_miss() {
    // The pair that makes the distinction legible: one means fetch the
    // operand, the other means the index is wrong. Collapsing them would leave
    // an operator unable to tell which.
    let err = ExecutionError::ExpertOutOfRange {
        expert: 900,
        population: 128,
    };
    assert_eq!(err.refusal(), RefusalKind::BindingDefect);
    assert!(Verdict::from(&err).indicts_execution());
}

#[test]
fn a_missing_kernel_is_unsupported_and_does_not_indict_the_execution() {
    // A valid representation with a valid semantic view can have no admissible
    // kernel. That is a gap, not a defect — nobody should reject the index.
    for err in [
        ExecutionError::KernelActivationUnsupported {
            kernel: KERNEL,
            activation: "ReLU".into(),
        },
        ExecutionError::KernelOperandUnsuitable {
            kernel: KERNEL,
            operand: OPERAND.into(),
            reason: OperandUnsuitability::Arrangement {
                found: "gate+up".into(),
                wanted: "one fused gate+up region",
            },
        },
    ] {
        assert_eq!(err.refusal(), RefusalKind::Unsupported, "{err}");
        assert!(!Verdict::from(&err).indicts_execution(), "{err}");
    }
}

#[test]
fn every_error_variant_is_classified() {
    // `refusal` is an exhaustive match, so this cannot fail to compile — but it
    // can fail to be *exercised*, which is what leaves a new variant silently
    // inheriting whatever arm it was pattern-matched into.
    let errors = every_error();
    assert_eq!(errors.len(), 12, "a variant joined ExecutionError");
    for err in &errors {
        assert!(RefusalKind::ALL.contains(&err.refusal()), "{err}");
    }
}

#[test]
fn nothing_but_a_non_resident_expert_classifies_as_residency() {
    // The dangerous default. `Residency` is the one kind that reads as
    // "nothing is wrong", so an unclassified failure inheriting it would make a
    // sweep report a defect as normal shard behaviour.
    let residency: Vec<String> = every_error()
        .iter()
        .filter(|e| e.refusal() == RefusalKind::Residency)
        .map(ToString::to_string)
        .collect();
    assert_eq!(residency.len(), 1, "{residency:?}");
    assert!(residency[0].contains("not resident"), "{residency:?}");
}

// ── Asserted versus observed ───────────────────────────────────────────────

#[test]
fn an_identical_result_is_exact_only_when_it_was_asserted() {
    assert_eq!(
        Verdict::from_comparison(true, true, true),
        Verdict::Exact,
        "an asserted checkpoint that matched is a contract"
    );
    assert_eq!(
        Verdict::from_comparison(true, true, false),
        Verdict::ObservedExact,
        "the same bytes, watched rather than contracted, are not"
    );
}

#[test]
fn only_exact_is_a_contract() {
    assert!(Verdict::Exact.is_contracted());
    for verdict in [
        Verdict::ObservedExact,
        Verdict::Equivalent,
        Verdict::NumericMismatch,
        Verdict::Refused(RefusalKind::Residency),
    ] {
        assert!(
            !verdict.is_contracted(),
            "{verdict} must not read as a guarantee"
        );
    }
}

#[test]
fn observed_exact_still_counts_as_agreement() {
    // It is weaker than `Exact`, not a failure. A sweep that treated it as one
    // would report a bit-identical block as a mismatch.
    assert!(Verdict::ObservedExact.agreed());
    assert!(!Verdict::ObservedExact.indicts_execution());
}

#[test]
fn a_difference_is_equivalent_or_a_mismatch_by_the_tolerance_alone() {
    // Whether it was asserted is irrelevant once the values differ: an
    // asserted checkpoint that missed is not "more equivalent" than a watched
    // one.
    assert_eq!(
        Verdict::from_comparison(false, true, true),
        Verdict::Equivalent
    );
    assert_eq!(
        Verdict::from_comparison(false, true, false),
        Verdict::Equivalent
    );
    assert_eq!(
        Verdict::from_comparison(false, false, true),
        Verdict::NumericMismatch
    );
    assert_eq!(
        Verdict::from_comparison(false, false, false),
        Verdict::NumericMismatch
    );
}

#[test]
fn only_a_numeric_mismatch_and_a_binding_defect_indict_the_execution() {
    assert!(Verdict::NumericMismatch.indicts_execution());
    assert!(Verdict::Refused(RefusalKind::BindingDefect).indicts_execution());
    for verdict in [
        Verdict::Exact,
        Verdict::ObservedExact,
        Verdict::Equivalent,
        Verdict::Refused(RefusalKind::Residency),
        Verdict::Refused(RefusalKind::Unsupported),
    ] {
        assert!(!verdict.indicts_execution(), "{verdict}");
    }
}

// ── Reporting ──────────────────────────────────────────────────────────────

#[test]
fn a_sweep_can_take_its_worst_outcome_by_maximum() {
    // The ordering exists so a summary line needs no bespoke fold. Weakest
    // claims sort last.
    let mut outcomes = [
        Verdict::Equivalent,
        Verdict::Exact,
        Verdict::Refused(RefusalKind::Residency),
        Verdict::ObservedExact,
    ];
    outcomes.sort_unstable();
    assert_eq!(outcomes.first(), Some(&Verdict::Exact));
    assert_eq!(
        outcomes.iter().max(),
        Some(&Verdict::Refused(RefusalKind::Residency))
    );
}

#[test]
fn every_verdict_has_a_distinct_name() {
    // Names go into a per-layer table that a human scans, so two outcomes
    // sharing one would make the table unreadable exactly where it matters.
    let mut names: Vec<&str> = vec![
        Verdict::Exact.name(),
        Verdict::ObservedExact.name(),
        Verdict::Equivalent.name(),
        Verdict::NumericMismatch.name(),
    ];
    names.extend(RefusalKind::ALL.iter().map(|k| Verdict::Refused(*k).name()));
    let count = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), count, "two verdicts share a name");
}

#[test]
fn a_verdict_displays_as_its_name() {
    assert_eq!(Verdict::Exact.to_string(), "exact");
    assert_eq!(
        Verdict::Refused(RefusalKind::Residency).to_string(),
        RefusalKind::Residency.to_string()
    );
    assert_eq!(RefusalKind::BindingDefect.to_string(), "binding_defect");
}

// ── The classification crossing a crate boundary ───────────────────────────

#[test]
fn the_boundary_trait_reports_the_same_kind_as_the_inherent_method() {
    // `ExecutionRefusal::kind` delegates to `refusal()` precisely so the two
    // cannot disagree — but a delegation nothing calls is a delegation nobody
    // has checked. This walks every variant through both.
    use larql_execution::ExecutionRefusal;
    for err in every_error() {
        assert_eq!(
            ExecutionRefusal::kind(&err),
            err.refusal(),
            "the boundary trait and the inherent method disagree for {err}"
        );
    }
}

#[test]
fn a_boxed_execution_error_keeps_its_classification_and_its_detail() {
    // The reason the trait exists: `larql-compute` cannot name `ExecutionError`
    // but can name `dyn ExecutionRefusal`, and both levels must survive the
    // crossing — the response category *and* the concrete diagnosis.
    let boxed: larql_execution::BoxRefusal = Box::new(ExecutionError::SelectedExpertNotResident {
        expert: 90,
        bank: "layer 5 bank 0".into(),
        resident: 8,
        population: 128,
    });
    assert_eq!(boxed.kind(), RefusalKind::Residency);
    assert!(!boxed.kind().indicts_the_artifact());
    // The detail is still there, not flattened to the category.
    let text = boxed.to_string();
    assert!(text.contains("expert 90"), "{text}");
    assert!(text.contains("8 of 128"), "{text}");
}
