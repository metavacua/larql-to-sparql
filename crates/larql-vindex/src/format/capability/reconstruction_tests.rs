//! Colocated tests for `reconstruction` — the COMPILE contract.
//!
//! The load-bearing property is scope: canonical reconstruction must depend on
//! the document's declared baselines and on nothing a profile chose. The tests
//! that matter most are therefore the ones that *change serving policy and
//! assert the reconstruction is unmoved*.

use crate::format::lyrw2::region_format::{Packing, RegionFormat};
use crate::format::lyrw2::region_role::RegionRole;

use super::authority::{derive_authority, AuthorityInputs, Fidelity};
use super::coordinate::RegionCoordinate;
use super::reconstruction::{
    CanonicalSelection, CatalogueVariant, ReconstructionFailure, RegionSetCatalogue,
    TensorReconstruction,
};

const BASELINE: &str = "exact-q6k";
const SIBLING: &str = "native-mxfp4";

fn everything_decodes(_: RegionFormat, _: Packing) -> bool {
    true
}

fn nothing_decodes(_: RegionFormat, _: Packing) -> bool {
    false
}

fn coordinate(layer: u32) -> RegionCoordinate {
    RegionCoordinate::new(layer, 0, None, RegionRole::GateUpFused)
}

fn variant(name: &str, fidelity: Fidelity, present: bool) -> CatalogueVariant {
    CatalogueVariant {
        name: name.into(),
        format: RegionFormat::Q6K,
        packing: Packing::BlocksWithScalesInline,
        fidelity,
        present,
    }
}

/// A region set with a canonical Q6_K baseline and an MXFP4 sibling — the
/// §9.1 example, where a serving profile might prefer the sibling.
fn two_variant_set(layer: u32) -> RegionSetCatalogue {
    RegionSetCatalogue {
        coordinate: coordinate(layer),
        baseline: BASELINE.into(),
        variants: vec![
            variant(BASELINE, Fidelity::SourceEquivalent, true),
            variant(SIBLING, Fidelity::SourceExact, true),
        ],
    }
}

#[test]
fn the_declared_baseline_is_chosen_not_the_most_faithful_variant() {
    // The sibling is source-EXACT and the baseline only source-equivalent.
    // Canonical still means "declared baseline": COMPILE reproduces the
    // document's canonical form, not the best available encoding.
    let sel = CanonicalSelection::from_catalogue(&[two_variant_set(0)], &everything_decodes);
    assert!(sel.is_complete());
    assert_eq!(sel.chosen.len(), 1);
    assert_eq!(sel.chosen[0].1.name, BASELINE);
    assert_eq!(sel.fidelities(), vec![Fidelity::SourceEquivalent]);
}

#[test]
fn reconstruction_authority_can_be_approximate_and_still_be_canonical() {
    // Baseline status is a pointer, not a fidelity promotion. A baseline
    // quantised from BF16 reconstructs canonically at numerically-approximate,
    // which is why the operation is not called ReconstructOriginal.
    let lossy = RegionSetCatalogue {
        baseline: BASELINE.into(),
        variants: vec![variant(BASELINE, Fidelity::NumericallyApproximate, true)],
        ..two_variant_set(0)
    };
    let sel = CanonicalSelection::from_catalogue(&[lossy], &everything_decodes);
    assert!(sel.is_complete());
    let authority = derive_authority(&AuthorityInputs {
        selected_fidelities: sel.fidelities(),
        execution_complete: true,
        structural: None,
    });
    assert_eq!(authority, Fidelity::NumericallyApproximate);
}

#[test]
fn an_absent_baseline_never_falls_back_to_a_present_sibling() {
    // Falling back would silently change what COMPILE emits — the exact
    // failure the canonical contract exists to prevent.
    let mut set = two_variant_set(0);
    set.variants[0].present = false;
    let sel = CanonicalSelection::from_catalogue(&[set], &everything_decodes);

    assert!(!sel.is_complete());
    assert!(sel.chosen.is_empty(), "must not substitute the sibling");
    assert!(matches!(
        sel.failures[0],
        ReconstructionFailure::BaselineAbsent { .. }
    ));
}

#[test]
fn a_baseline_naming_an_undeclared_variant_lists_what_was_declared() {
    let mut set = two_variant_set(0);
    set.baseline = "native-nvfp4".into();
    let sel = CanonicalSelection::from_catalogue(&[set], &everything_decodes);
    match &sel.failures[0] {
        ReconstructionFailure::BaselineNotDeclared { declared, .. } => {
            assert_eq!(declared, &vec![BASELINE.to_string(), SIBLING.to_string()]);
        }
        other => panic!("expected not-declared, got {other:?}"),
    }
}

#[test]
fn a_region_set_with_no_baseline_is_refused() {
    let mut set = two_variant_set(0);
    set.baseline = "  ".into();
    let sel = CanonicalSelection::from_catalogue(&[set], &everything_decodes);
    assert!(matches!(
        sel.failures[0],
        ReconstructionFailure::NoDeclaredBaseline { .. }
    ));
}

#[test]
fn an_undecodable_baseline_names_both_format_and_packing() {
    let sel = CanonicalSelection::from_catalogue(&[two_variant_set(0)], &nothing_decodes);
    match &sel.failures[0] {
        ReconstructionFailure::BaselineUndecodable {
            format, packing, ..
        } => {
            assert_eq!(*format, RegionFormat::Q6K);
            assert_eq!(*packing, Packing::BlocksWithScalesInline);
        }
        other => panic!("expected undecodable, got {other:?}"),
    }
}

#[test]
fn one_broken_region_set_does_not_discard_the_others() {
    // Reconstruction reports every failure it finds; a single bad set must not
    // abort the survey, or the operator fixes one problem per run.
    let mut broken = two_variant_set(1);
    broken.variants[0].present = false;
    let sel = CanonicalSelection::from_catalogue(
        &[two_variant_set(0), broken, two_variant_set(2)],
        &everything_decodes,
    );
    assert_eq!(sel.chosen.len(), 2);
    assert_eq!(sel.failures.len(), 1);
    assert!(!sel.is_complete());
}

#[test]
fn every_failure_carries_its_coordinate() {
    let mut set = two_variant_set(37);
    set.variants[0].present = false;
    let sel = CanonicalSelection::from_catalogue(&[set], &everything_decodes);
    assert!(sel.failures[0].describe().contains("layer 37"));
}

#[test]
fn each_failure_kind_renders_distinguishably() {
    let rendered: Vec<String> = [
        ReconstructionFailure::NoDeclaredBaseline {
            coordinate: coordinate(0),
        },
        ReconstructionFailure::BaselineNotDeclared {
            coordinate: coordinate(0),
            baseline: "x".into(),
            declared: vec!["y".into()],
        },
        ReconstructionFailure::BaselineAbsent {
            coordinate: coordinate(0),
            baseline: "x".into(),
        },
        ReconstructionFailure::BaselineUndecodable {
            coordinate: coordinate(0),
            format: RegionFormat::Nvfp4,
            packing: Packing::RowMajor,
        },
        ReconstructionFailure::RecipeRoleMissing {
            output: "w1".into(),
            role: RegionRole::Gate,
        },
        ReconstructionFailure::IncompletePair {
            output: "w1".into(),
            present: RegionRole::Scales,
            missing: RegionRole::Down,
        },
    ]
    .iter()
    .map(|f| f.describe())
    .collect();
    for i in 0..rendered.len() {
        for j in (i + 1)..rendered.len() {
            assert_ne!(rendered[i], rendered[j], "{i} vs {j}");
        }
    }
}

#[test]
fn an_empty_catalogue_reconstructs_vacuously() {
    let sel = CanonicalSelection::from_catalogue(&[], &everything_decodes);
    assert!(sel.is_complete());
    assert!(sel.fidelities().is_empty());
}

// ── Recipes target checkpoint tensors, not stored regions ──────────────────

#[test]
fn a_fused_region_can_split_into_two_checkpoint_tensors() {
    // Storage fuses gate and up; the checkpoint keeps them apart. COMPILE must
    // reproduce the checkpoint's vocabulary, not the index's.
    let r = TensorReconstruction::SplitFused {
        role: RegionRole::GateUpFused,
        outputs: vec!["w1.weight".into(), "w3.weight".into()],
    };
    assert_eq!(r.required_roles(), vec![RegionRole::GateUpFused]);
    assert_eq!(r.outputs().len(), 2);
}

#[test]
fn separate_roles_can_join_into_one_checkpoint_tensor() {
    // The mirror direction, for a checkpoint whose canonical key is fused.
    let r = TensorReconstruction::JoinRoles {
        roles: vec![RegionRole::Gate, RegionRole::Up],
        output: "gate_up_proj.weight".into(),
    };
    assert_eq!(r.required_roles().len(), 2);
    assert_eq!(r.outputs(), vec!["gate_up_proj.weight".to_string()]);
}

#[test]
fn a_values_scales_pair_requires_both_halves() {
    let r = TensorReconstruction::PairedValuesScales {
        values: RegionRole::Down,
        scales: RegionRole::Scales,
        output: "w2.weight".into(),
    };
    assert_eq!(
        r.required_roles(),
        vec![RegionRole::Down, RegionRole::Scales]
    );
}

#[test]
fn a_direct_recipe_maps_one_role_to_one_tensor() {
    let r = TensorReconstruction::Direct {
        role: RegionRole::Down,
        output: "w2.weight".into(),
    };
    assert_eq!(r.required_roles(), vec![RegionRole::Down]);
    assert_eq!(r.outputs(), vec!["w2.weight".to_string()]);
}
