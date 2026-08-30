//! Colocated tests for `traversal` — complete inference cases.
//!
//! These are deliberately whole-scenario rather than per-enum: the value of a
//! traversal is the verdict it reaches over a realistic selection, and the
//! per-variant behaviour is covered in `role`, `compatibility` and
//! `coordinate`. Each case below is a situation an operator could actually
//! hit, and asserts both the verdict and the diagnosis.

use std::collections::BTreeMap;

use crate::format::lyrw2::region_format::{Packing, RegionFormat};
use crate::format::lyrw2::region_role::RegionRole;
use crate::format::moe_manifest::programme::Programme;

use super::authority::Fidelity;
use super::compatibility::{IncompatibilityShape, SegmentCompatibility};
use super::coordinate::AbsenceKind;
use super::role::{OperandAvailability, ReferenceSupport};
use super::selection::{BankSelection, SelectedRegion};
use super::traversal::{traverse_bank, ReferenceCodecs, ReferenceExecution};

const LAYER: u32 = 12;
const BANK: u16 = 0;
const GATE: RegionRole = RegionRole::Gate;
const UP: RegionRole = RegionRole::Up;
const FUSED: RegionRole = RegionRole::GateUpFused;
const DOWN: RegionRole = RegionRole::Down;

/// A build that reads everything — isolates coverage behaviour from codec
/// behaviour.
struct AllCodecs;
impl ReferenceCodecs for AllCodecs {
    fn supports(&self, _: RegionFormat) -> bool {
        true
    }
    fn supports_packing(&self, _: Packing) -> bool {
        true
    }
}

/// A build that cannot read NVFP4 — the "newer artifact" case.
struct NoNvfp4;
impl ReferenceCodecs for NoNvfp4 {
    fn supports(&self, f: RegionFormat) -> bool {
        f != RegionFormat::Nvfp4
    }
    fn supports_packing(&self, _: Packing) -> bool {
        true
    }
}

fn region(format: RegionFormat) -> SelectedRegion {
    SelectedRegion {
        format,
        packing: Packing::RowMajor,
        fidelity: Fidelity::SourceEquivalent,
    }
}

/// Build a segmented selection from `(role, segments)` pairs.
fn selection(required: &[u16], roles: &[(RegionRole, &[u16])]) -> BankSelection {
    let mut regions = BTreeMap::new();
    for (role, segs) in roles {
        for s in *segs {
            regions.insert((Some(*s), *role), region(RegionFormat::Q6K));
        }
    }
    BankSelection {
        bank_id: BANK,
        required_segments: required.to_vec(),
        regions,
        omitted_roles: Vec::new(),
    }
}

// ── 1. Both alternatives physically present ────────────────────────────────

#[test]
fn both_alternatives_present_means_both_survive_for_kernel_binding() {
    let sel = selection(
        &[0, 1],
        &[
            (FUSED, &[0, 1]),
            (GATE, &[0, 1]),
            (UP, &[0, 1]),
            (DOWN, &[0, 1]),
        ],
    );
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);
    assert!(r.is_executable());
    assert_eq!(
        r.successful_alternatives().len(),
        2,
        "kernel binding must receive both layouts"
    );
    assert!(r.closest_failure().is_none());
}

// ── 2. Decomposed incompatible, fused complete ─────────────────────────────

#[test]
fn a_layer_is_executable_through_fused_when_decomposed_is_incompatible() {
    // gate covers {0}, up covers {1} — decomposed cannot run. The fused
    // layout is whole, so the LAYER is fine and the decomposed failure is
    // evidence rather than a defect.
    let sel = selection(
        &[0, 1],
        &[(FUSED, &[0, 1]), (DOWN, &[0, 1]), (GATE, &[0]), (UP, &[1])],
    );
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);

    assert!(r.is_executable());
    assert_eq!(r.successful_alternatives().len(), 1);
    assert_eq!(
        r.successful_alternatives()[0].alternative,
        &[FUSED, DOWN][..]
    );
    // No closest failure is reported while something succeeds.
    assert!(r.closest_failure().is_none());
    // ...but the invalid decomposed selection is still surfaced.
    assert!(r.has_invalid_selection());
}

// ── 3. Pairwise overlap, empty global intersection ─────────────────────────

#[test]
fn three_roles_overlapping_pairwise_but_not_globally_are_incompatible() {
    let sel = selection(
        &[0, 1, 2],
        &[(GATE, &[0, 1]), (UP, &[1, 2]), (DOWN, &[2, 0])],
    );
    let r = traverse_bank(LAYER, Programme::GatedMlpFusedFc1V1, &sel, &AllCodecs);
    // The fused programme needs gate_up_fused, which is absent — tier 1.
    assert!(!r.is_executable());

    let decomposed = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);
    let alt = decomposed
        .alternatives
        .iter()
        .find(|a| a.alternative == &[GATE, UP, DOWN][..])
        .unwrap();
    match alt.compatibility.as_ref().unwrap() {
        SegmentCompatibility::Incompatible { shape, common, .. } => {
            assert_eq!(*shape, IncompatibilityShape::NoCommonCoverage);
            assert!(common.is_empty());
        }
        other => panic!("expected incompatible, got {other:?}"),
    }
}

// ── 4. All segments present, codec unsupported ─────────────────────────────

#[test]
fn an_unreadable_codec_is_a_whole_role_failure_not_partial_coverage() {
    // The bytes are all there. This build simply cannot read them, so the role
    // contributes nowhere — tier 1, with the codec named.
    let mut sel = selection(&[0, 1], &[(FUSED, &[0, 1]), (DOWN, &[0, 1])]);
    for s in [0u16, 1] {
        sel.regions
            .insert((Some(s), DOWN), region(RegionFormat::Nvfp4));
    }
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &NoNvfp4);

    assert!(!r.is_executable());
    let alt = &r.alternatives[0];
    assert!(matches!(
        alt.reference_execution,
        ReferenceExecution::BlockedByOperands { .. }
    ));
    // Compatibility is moot and deliberately not evaluated.
    assert!(alt.compatibility.is_none());

    let down = alt
        .operands
        .iter()
        .find(|o| o.coordinate.role == DOWN)
        .unwrap();
    assert_eq!(down.availability, OperandAvailability::Present);
    assert_eq!(
        down.reference_support,
        ReferenceSupport::UnsupportedFormat(RegionFormat::Nvfp4)
    );
    // Present-but-unreadable is not an index defect — the artifact is intact.
    assert!(!down.availability.indicates_defect());
}

// ── 5. Equal incomplete coverage ───────────────────────────────────────────

#[test]
fn equally_short_coverage_is_partial_not_incompatible() {
    let sel = selection(&[0, 1, 2], &[(FUSED, &[0, 1]), (DOWN, &[0, 1])]);
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);

    assert!(!r.is_executable());
    let alt = &r.alternatives[0];
    assert_eq!(
        alt.reference_execution,
        ReferenceExecution::BlockedByIncompleteSegments
    );
    assert!(
        !alt.is_invalid_selection(),
        "consistent shortfall is not a contradiction"
    );
    assert!(matches!(
        alt.compatibility.as_ref().unwrap(),
        SegmentCompatibility::Partial { .. }
    ));
}

// ── 6. Gate-only deliberate omission ───────────────────────────────────────

#[test]
fn a_deliberate_gate_only_slice_is_not_reported_as_corruption() {
    // The browse-slice case. Decode is unavailable, but nothing is broken and
    // the selection is not invalid.
    let mut sel = selection(&[0], &[(GATE, &[0])]);
    sel.omitted_roles = vec![UP, DOWN, FUSED];
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);

    assert!(!r.is_executable());
    assert!(!r.has_invalid_selection());

    let failure = r.closest_failure().unwrap();
    let down = failure
        .operands
        .iter()
        .find(|o| o.coordinate.role == DOWN)
        .unwrap();
    assert_eq!(
        down.availability,
        OperandAvailability::Absent(AbsenceKind::OmittedBySelection)
    );
    assert!(!down.availability.indicates_defect());
}

// ── 7. Gate-only accidental absence ────────────────────────────────────────

#[test]
fn an_accidental_gate_only_index_is_not_laundered_into_a_browse_slice() {
    // Physically identical to case 6 in what is *missing*, but nothing
    // declared the omission. Provenance must survive: this is a defect.
    let sel = selection(&[0], &[(GATE, &[0])]);
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);

    assert!(!r.is_executable());
    let failure = r.closest_failure().unwrap();
    let down = failure
        .operands
        .iter()
        .find(|o| o.coordinate.role == DOWN)
        .unwrap();
    assert_eq!(
        down.availability,
        OperandAvailability::Absent(AbsenceKind::AbsentEverywhere)
    );
    assert!(
        down.availability.indicates_defect(),
        "undeclared absence must stay a defect"
    );
}

#[test]
fn declared_and_undeclared_absence_differ_only_in_provenance() {
    // Same physical shape, same admission result, different diagnosis — the
    // property that stops a corrupt index presenting as an intentional slice.
    let mut declared = selection(&[0], &[(GATE, &[0])]);
    declared.omitted_roles = vec![UP, DOWN, FUSED];
    let accidental = selection(&[0], &[(GATE, &[0])]);

    let a = traverse_bank(LAYER, Programme::GatedMlpV1, &declared, &AllCodecs);
    let b = traverse_bank(LAYER, Programme::GatedMlpV1, &accidental, &AllCodecs);

    assert_eq!(a.is_executable(), b.is_executable());
    assert!(!a.has_invalid_selection() && !b.has_invalid_selection());

    let defect = |r: &super::traversal::BankCapabilityReport| {
        r.closest_failure()
            .unwrap()
            .operands
            .iter()
            .any(|o| o.availability.indicates_defect())
    };
    assert!(!defect(&a), "declared omission is not a defect");
    assert!(defect(&b), "undeclared absence is");
}

// ── 8. Same bytes, no grouped kernel ───────────────────────────────────────

#[test]
fn traversal_stops_at_reference_executability_and_names_no_kernel() {
    // Traversal cannot express a kernel binding at all — that is the whole
    // point of the split, and it is enforced by the type rather than by
    // convention.
    let sel = selection(&[0, 1], &[(FUSED, &[0, 1]), (DOWN, &[0, 1])]);
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);
    assert_eq!(
        r.successful_alternatives()[0].reference_execution,
        ReferenceExecution::Executable
    );
}

// ── 10. Alternative-order permutation ──────────────────────────────────────

#[test]
fn the_successful_set_does_not_depend_on_selection_insertion_order() {
    let forward = selection(&[0, 1], &[(FUSED, &[0, 1]), (DOWN, &[1, 0])]);
    let reverse = selection(&[1, 0], &[(DOWN, &[0, 1]), (FUSED, &[1, 0])]);
    let a = traverse_bank(LAYER, Programme::GatedMlpV1, &forward, &AllCodecs);
    let b = traverse_bank(LAYER, Programme::GatedMlpV1, &reverse, &AllCodecs);
    assert_eq!(
        a.successful_alternatives().len(),
        b.successful_alternatives().len()
    );
    assert_eq!(a.is_executable(), b.is_executable());
}

#[test]
fn a_closest_failure_is_chosen_by_fewest_unusable_roles() {
    // Fused is missing one role; decomposed is missing two. The fused layout
    // is the closer failure and is the one worth reporting.
    let sel = selection(&[0], &[(FUSED, &[0])]);
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);
    assert!(!r.is_executable());
    let failure = r.closest_failure().unwrap();
    assert_eq!(failure.alternative, &[FUSED, DOWN][..]);
    assert_eq!(failure.unusable_role_count(), 1);
}

// ── Unsegmented banks ──────────────────────────────────────────────────────

#[test]
fn an_unsegmented_bank_executes_on_presence_alone() {
    let sel = BankSelection::unsegmented(
        BANK,
        &[FUSED, DOWN],
        RegionFormat::F16,
        Packing::RowMajor,
        Fidelity::SourceExact,
    );
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);
    assert!(r.is_executable());
    assert_eq!(r.required_population, Vec::<u16>::new());
}

#[test]
fn an_unsegmented_bank_missing_a_role_still_fails_tier_one() {
    let sel = BankSelection::unsegmented(
        BANK,
        &[FUSED],
        RegionFormat::F16,
        Packing::RowMajor,
        Fidelity::SourceExact,
    );
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);
    assert!(!r.is_executable());
    assert!(matches!(
        r.closest_failure().unwrap().reference_execution,
        ReferenceExecution::BlockedByOperands { .. }
    ));
}

#[test]
fn gpt_oss_without_bias_fails_on_the_bias_role() {
    let sel = selection(&[0], &[(FUSED, &[0]), (DOWN, &[0])]);
    let r = traverse_bank(LAYER, Programme::GptOssExpertV1, &sel, &AllCodecs);
    assert!(!r.is_executable());
    let blocked = &r.closest_failure().unwrap().reference_execution;
    match blocked {
        ReferenceExecution::BlockedByOperands { roles } => {
            assert_eq!(roles, &vec![RegionRole::Bias]);
        }
        other => panic!("expected operand block, got {other:?}"),
    }
}

#[test]
fn every_diagnosis_carries_the_full_coordinate() {
    let sel = selection(&[0, 1], &[(FUSED, &[0, 1])]);
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);
    let down = r
        .closest_failure()
        .unwrap()
        .operands
        .iter()
        .find(|o| o.coordinate.role == DOWN)
        .unwrap();
    let s = down.describe();
    assert!(s.contains("layer 12"), "{s}");
    assert!(s.contains("bank 0"), "{s}");
    assert!(s.contains("role down"), "{s}");
}

#[test]
fn regions_outside_the_required_population_are_named_as_such() {
    // The bytes exist in segments 7-8; this selection requires 0-1. That is a
    // resolution fault, not a missing variant, and the repair differs.
    let mut sel = selection(&[0, 1], &[(FUSED, &[0, 1])]);
    for s in [7u16, 8] {
        sel.regions
            .insert((Some(s), DOWN), region(RegionFormat::Q6K));
    }
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);
    assert!(!r.is_executable());

    let down = r
        .closest_failure()
        .unwrap()
        .operands
        .iter()
        .find(|o| o.coordinate.role == DOWN)
        .unwrap();
    match &down.availability {
        OperandAvailability::Absent(AbsenceKind::PresentOutsidePopulation { found, required }) => {
            assert_eq!(found, &vec![7, 8]);
            assert_eq!(required, &vec![0, 1]);
        }
        other => panic!("expected outside-population, got {other:?}"),
    }
}

#[test]
fn every_reference_execution_verdict_renders_distinguishably() {
    let verdicts = [
        ReferenceExecution::Executable,
        ReferenceExecution::BlockedByOperands { roles: vec![DOWN] },
        ReferenceExecution::BlockedByIncompatibleSegments,
        ReferenceExecution::BlockedByIncompleteSegments,
    ];
    let rendered: Vec<String> = verdicts.iter().map(|v| v.describe()).collect();
    for i in 0..rendered.len() {
        for j in (i + 1)..rendered.len() {
            assert_ne!(rendered[i], rendered[j], "{i} vs {j}");
        }
    }
    assert!(rendered[0].contains("reference-executable"));
    assert!(rendered[1].contains("down"));
    assert!(verdicts[0].is_executable());
    assert!(!verdicts[1].is_executable());
}

#[test]
fn a_bank_with_no_usable_role_reports_every_blocking_role() {
    let sel = selection(&[0], &[]);
    let r = traverse_bank(LAYER, Programme::GatedMlpV1, &sel, &AllCodecs);
    match &r.closest_failure().unwrap().reference_execution {
        ReferenceExecution::BlockedByOperands { roles } => {
            assert_eq!(roles.len(), 2, "{roles:?}");
        }
        other => panic!("expected operand block, got {other:?}"),
    }
}
