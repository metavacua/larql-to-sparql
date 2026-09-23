//! Colocated tests for `binding` — identity, views and deduplication.
//!
//! The headline case throughout is the tied LM head: one tensor, two semantic
//! roles, two views, one allocation.

use super::authority::Fidelity;
use super::binding::{
    catalogue_conflicts, physical_extents, total_bytes, ComponentBinding, ComponentView,
    PhysicalStorageId, RepresentationIdentity, ViewError,
};
use super::component::{ComponentContract, TensorKind};

const VOCAB: u32 = 32_000;
const HIDDEN: u32 = 4_096;
const DIGEST: &str = "sha256:aabbccddeeff0011";

fn extent(offset: u64, length: u64) -> PhysicalStorageId {
    PhysicalStorageId::new(DIGEST, offset, length)
}

fn binding(name: &str, storage: PhysicalStorageId, view: ComponentView) -> ComponentBinding {
    ComponentBinding {
        representation: RepresentationIdentity::new("embeddings", name),
        storage,
        view,
        fidelity: Fidelity::SourceExact,
    }
}

// ── Identity is about bytes, not names ─────────────────────────────────────

#[test]
fn two_variant_names_for_one_extent_deduplicate_to_one_component() {
    // The correction: folding the variant into physical identity would defeat
    // the deduplication it exists for.
    let a = binding("baseline", extent(0, 1_024), ComponentView::Direct);
    let b = ComponentBinding {
        representation: RepresentationIdentity::new("embeddings", "alias"),
        ..a.clone()
    };
    assert_ne!(a.representation, b.representation);
    assert_eq!(physical_extents(&[a, b]).len(), 1);
}

#[test]
fn one_tensor_serving_two_roles_is_one_allocation() {
    // Tied LM head: two bindings, two views, one extent.
    let shared = extent(0, 1_024);
    let bindings = [
        binding("baseline", shared.clone(), ComponentView::Direct),
        binding("baseline", shared, ComponentView::Transpose),
    ];
    assert_eq!(bindings.len(), 2, "both semantic uses survive");
    assert_eq!(physical_extents(&bindings).len(), 1);
    assert_eq!(total_bytes(&bindings), 1_024, "counted once, not twice");
}

#[test]
fn a_duplicated_logical_tensor_is_two_components() {
    // Same logical tensor written twice is two allocations, however tied it
    // is conceptually.
    let bindings = [
        binding("baseline", extent(0, 1_024), ComponentView::Direct),
        binding("copy", extent(4_096, 1_024), ComponentView::Direct),
    ];
    assert_eq!(physical_extents(&bindings).len(), 2);
    assert_eq!(total_bytes(&bindings), 2_048);
}

#[test]
fn only_exact_extents_are_treated_as_the_same_bytes() {
    let whole = extent(0, 1_024);
    let half = extent(0, 512);
    assert!(whole.is_same_extent(&extent(0, 1_024)));
    assert!(!whole.is_same_extent(&half));
}

#[test]
fn partial_overlap_is_detected_and_never_deduplicated() {
    // A sub-range is either legitimate or a catalogue error; neither makes it
    // "the same component".
    let a = extent(0, 1_024);
    let b = extent(512, 1_024);
    assert!(a.overlaps_partially(&b));
    assert!(!a.is_same_extent(&b));
    assert_eq!(
        physical_extents(&[
            binding("a", a, ComponentView::Direct),
            binding("b", b, ComponentView::Direct),
        ])
        .len(),
        2
    );
}

#[test]
fn extents_in_different_files_never_overlap() {
    let a = PhysicalStorageId::new("sha256:aaaa", 0, 1_024);
    let b = PhysicalStorageId::new("sha256:bbbb", 0, 1_024);
    assert!(!a.overlaps_partially(&b));
    assert!(!a.is_same_extent(&b));
}

#[test]
fn adjacent_extents_do_not_overlap() {
    let a = extent(0, 1_024);
    let b = extent(1_024, 1_024);
    assert!(!a.overlaps_partially(&b));
}

// ── Views transform contracts, not values ──────────────────────────────────

#[test]
fn a_transpose_turns_an_embedding_into_an_lm_head_shape() {
    // The case a raw-storage contract check would have rejected.
    let storage = ComponentContract::matrix(VOCAB, HIDDEN);
    let viewed = ComponentView::Transpose.apply_to(&storage).unwrap();
    assert_eq!(viewed, ComponentContract::matrix(HIDDEN, VOCAB));
}

#[test]
fn a_direct_view_leaves_the_contract_alone() {
    let storage = ComponentContract::matrix(HIDDEN, VOCAB);
    assert_eq!(ComponentView::Direct.apply_to(&storage).unwrap(), storage);
}

#[test]
fn transposing_a_vector_is_refused_rather_than_silently_accepted() {
    let err = ComponentView::Transpose
        .apply_to(&ComponentContract::vector(HIDDEN))
        .unwrap_err();
    assert!(matches!(err, ViewError::TransposeOfNonMatrix { .. }));
    assert!(err.to_string().contains("matrices only"));
}

#[test]
fn a_slice_narrows_one_dimension() {
    let storage = ComponentContract::matrix(VOCAB, HIDDEN);
    let viewed = ComponentView::Slice {
        dim: 0,
        start: 0,
        len: 1_000,
    }
    .apply_to(&storage)
    .unwrap();
    assert_eq!(viewed.shape, vec![1_000, HIDDEN]);
    assert_eq!(viewed.kind, TensorKind::Matrix);
}

#[test]
fn a_slice_past_the_end_is_refused_with_both_bounds() {
    let err = ComponentView::Slice {
        dim: 0,
        start: VOCAB - 10,
        len: 100,
    }
    .apply_to(&ComponentContract::matrix(VOCAB, HIDDEN))
    .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("out of range"), "{text}");
    assert!(text.contains(&VOCAB.to_string()), "{text}");
}

#[test]
fn a_slice_on_a_missing_dimension_is_refused() {
    let err = ComponentView::Slice {
        dim: 5,
        start: 0,
        len: 1,
    }
    .apply_to(&ComponentContract::vector(HIDDEN))
    .unwrap_err();
    assert!(matches!(err, ViewError::SliceDimOutOfRange { .. }));
}

#[test]
fn no_view_alters_stored_values() {
    // Which is why views never touch authority. They affect kernel
    // eligibility and access efficiency instead.
    for view in [
        ComponentView::Direct,
        ComponentView::Transpose,
        ComponentView::Slice {
            dim: 0,
            start: 0,
            len: 1,
        },
    ] {
        assert!(!view.alters_values(), "{}", view.describe());
    }
}

#[test]
fn a_transposed_binding_keeps_the_fidelity_of_its_bytes() {
    let direct = binding("baseline", extent(0, 1_024), ComponentView::Direct);
    let transposed = ComponentBinding {
        view: ComponentView::Transpose,
        ..direct.clone()
    };
    assert_eq!(direct.fidelity, transposed.fidelity);
}

// ── Catalogue conflicts ────────────────────────────────────────────────────

#[test]
fn two_declarations_disagreeing_about_one_extent_are_a_catalogue_defect() {
    // Not two components — the bytes cannot be both.
    let shared = extent(0, 1_024);
    let a = ComponentBinding {
        representation: RepresentationIdentity::new("gate_up", "exact-q6k"),
        storage: shared.clone(),
        view: ComponentView::Direct,
        fidelity: Fidelity::SourceEquivalent,
    };
    let b = ComponentBinding {
        representation: RepresentationIdentity::new("gate_up", "native-mxfp4"),
        storage: shared,
        view: ComponentView::Direct,
        fidelity: Fidelity::SourceExact,
    };
    let conflicts = catalogue_conflicts(&[a, b]);
    assert_eq!(conflicts.len(), 1);
    let text = conflicts[0].to_string();
    assert!(text.contains("fidelity"), "{text}");
    assert!(text.contains("cannot be both"), "{text}");
}

#[test]
fn agreeing_declarations_of_one_extent_are_not_a_conflict() {
    let shared = extent(0, 1_024);
    let conflicts = catalogue_conflicts(&[
        binding("baseline", shared.clone(), ComponentView::Direct),
        binding("alias", shared, ComponentView::Transpose),
    ]);
    assert!(conflicts.is_empty(), "differing views are not a conflict");
}

#[test]
fn declarations_of_different_extents_are_never_a_conflict() {
    let conflicts = catalogue_conflicts(&[
        ComponentBinding {
            fidelity: Fidelity::SourceExact,
            ..binding("a", extent(0, 1_024), ComponentView::Direct)
        },
        ComponentBinding {
            fidelity: Fidelity::NumericallyApproximate,
            ..binding("b", extent(2_048, 1_024), ComponentView::Direct)
        },
    ]);
    assert!(conflicts.is_empty());
}

#[test]
fn a_binding_describes_representation_extent_and_view() {
    let s = binding("baseline", extent(64, 1_024), ComponentView::Transpose).describe();
    assert!(s.contains("embeddings::baseline"), "{s}");
    assert!(s.contains("64..1088"), "{s}");
    assert!(s.contains("transposed"), "{s}");
}

#[test]
fn no_bindings_means_no_bytes() {
    assert_eq!(total_bytes(&[]), 0);
    assert!(physical_extents(&[]).is_empty());
}
