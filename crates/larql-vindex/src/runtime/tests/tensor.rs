//! Colocated tests for `tensor` — decoding, views and refusals.
//!
//! The view cases matter most. Role-coordinate indexing is what lets one
//! executor serve tied, transposed and sliced operands, and a transpose that
//! silently reads the wrong element still produces a well-shaped result.

use crate::format::capability::binding::{ComponentView, RepresentationIdentity};
use crate::format::capability::component::ComponentContract;
use crate::format::lyrw2::region_format::RegionFormat;

use crate::runtime::axis::Axis;
use crate::runtime::consts::COL_DIM;
use crate::runtime::error::{ExecutionError, OperandUnsuitability};
use crate::runtime::tensor::BoundTensor;

use super::support::{q4k_bytes, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS};

const REGION_SET: &str = "gate";
const VARIANT: &str = "test";

fn identity() -> RepresentationIdentity {
    RepresentationIdentity::new(REGION_SET, VARIANT)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A 2×3 matrix: rows [1,2,3] and [4,5,6].
fn matrix_bytes() -> Vec<u8> {
    f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

fn matrix(view: ComponentView) -> BoundTensor<'static> {
    // Leaked so the tensor can be returned; test-only, bounded and tiny.
    let bytes: &'static [u8] = Box::leak(matrix_bytes().into_boxed_slice());
    BoundTensor::new(
        identity(),
        bytes,
        RegionFormat::F32,
        ComponentContract::matrix(2, 3),
        view,
    )
    .unwrap()
}

// ── Direct reads ───────────────────────────────────────────────────────────

#[test]
fn a_direct_matrix_reads_rows_in_storage_order() {
    let t = matrix(ComponentView::Direct);
    assert_eq!(t.rows(), 2);
    assert_eq!(t.cols(), 3);
    assert_eq!(t.row(0).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(t.row(1).unwrap(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn len_counts_elements_not_bytes() {
    let t = matrix(ComponentView::Direct);
    assert_eq!(t.len(), 6);
    assert!(!t.is_empty());
}

#[test]
fn a_vector_reads_whole() {
    let bytes = f32_bytes(&[0.5, -1.5, 2.5]);
    let t = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::F32,
        ComponentContract::vector(3),
    )
    .unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![0.5, -1.5, 2.5]);
    assert_eq!(t.cols(), 1, "a vector has one column per row");
}

// ── Views ──────────────────────────────────────────────────────────────────

#[test]
fn a_transposed_matrix_swaps_the_role_shape_and_the_element_order() {
    // The tied-LM-head case: [2,3] storage serving a [3,2] role.
    let t = matrix(ComponentView::Transpose);
    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.row(0).unwrap(), vec![1.0, 4.0]);
    assert_eq!(t.row(2).unwrap(), vec![3.0, 6.0]);
}

#[test]
fn a_row_slice_offsets_into_storage() {
    let t = matrix(ComponentView::Slice {
        dim: 0,
        start: 1,
        len: 1,
    });
    assert_eq!(t.rows(), 1);
    assert_eq!(t.row(0).unwrap(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn a_column_slice_narrows_each_row() {
    let t = matrix(ComponentView::Slice {
        dim: 1,
        start: 1,
        len: 2,
    });
    assert_eq!(t.cols(), 2);
    assert_eq!(t.row(0).unwrap(), vec![2.0, 3.0]);
}

#[test]
fn a_view_that_cannot_apply_is_refused_at_bind_time() {
    let bytes = f32_bytes(&[1.0, 2.0]);
    let err = BoundTensor::new(
        identity(),
        &bytes,
        RegionFormat::F32,
        ComponentContract::vector(2),
        ComponentView::Transpose,
    )
    .unwrap_err();
    assert!(matches!(err, ExecutionError::UnsupportedView { .. }));
}

// ── Encodings ──────────────────────────────────────────────────────────────

#[test]
fn bf16_decodes_through_the_shared_widening() {
    // 1.0 = 0x3F80, -1.0 = 0xBF80.
    let bytes = vec![0x80, 0x3F, 0x80, 0xBF];
    let t = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::BF16,
        ComponentContract::vector(2),
    )
    .unwrap();
    assert_eq!(t.to_vec().unwrap(), vec![1.0, -1.0]);
}

#[test]
fn f16_decodes_through_the_shared_subnormal_safe_path() {
    // 1.0 = 0x3C00; the smallest positive subnormal = 0x0001.
    let bytes = vec![0x00, 0x3C, 0x01, 0x00];
    let t = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::F16,
        ComponentContract::vector(2),
    )
    .unwrap();
    let values = t.to_vec().unwrap();
    assert_eq!(values[0], 1.0);
    assert!(values[1] > 0.0 && values[1] < 1e-6, "{}", values[1]);
}

#[test]
fn a_quantised_region_binds_but_refuses_a_per_element_read() {
    // The refusal moved, deliberately. A Q4_K region has known geometry, so it
    // binds — the whole point of binding one is to hand its blocks to a kernel
    // that reads them. What it cannot do is serve a per-element read: there is
    // no per-element slot, and inventing a stride would produce a well-shaped
    // tensor of noise rather than an error.
    //
    // Not a defect in the index — a missing reference kernel, and the message
    // must say which one.
    let bytes = vec![0u8; Q4K_BLOCK_BYTES];
    let bound = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::Q4K,
        ComponentContract::vector(2),
    )
    .expect("a Q4_K region with registered geometry binds");
    let text = bound.to_vec().unwrap_err().to_string();
    assert!(text.contains("q4_k"), "{text}");
    assert!(text.contains(REGION_SET), "{text}");
}

#[test]
fn a_codec_with_no_registered_geometry_is_refused_at_bind() {
    // The counter-case. Without a block layout there is no way to say how many
    // bytes the shape needs, so the region cannot be sized, let alone read.
    let bytes = vec![0u8; Q4K_BLOCK_BYTES];
    let err = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::Mxfp4,
        ComponentContract::vector(2),
    )
    .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("mxfp4"), "{text}");
    assert!(text.contains(REGION_SET), "{text}");
}

// ── Refusals ───────────────────────────────────────────────────────────────

#[test]
fn a_region_too_short_for_its_shape_is_refused_with_both_sizes() {
    let bytes = f32_bytes(&[1.0, 2.0]);
    let err = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::F32,
        ComponentContract::matrix(2, 3),
    )
    .unwrap_err();
    let ExecutionError::ShortRegion { needed, found, .. } = err else {
        panic!("expected a short-region refusal, got {err}");
    };
    assert_eq!((needed, found), (24, 8));
}

#[test]
fn a_row_past_the_end_is_refused() {
    let err = matrix(ComponentView::Direct).row(2).unwrap_err();
    assert!(matches!(err, ExecutionError::RowOutOfRange { row: 2, .. }));
}

#[test]
fn a_shape_assertion_names_the_operand_and_the_axis() {
    let err = matrix(ComponentView::Direct)
        .require_matrix(2, 5)
        .unwrap_err();
    let ExecutionError::DimensionMismatch { axis, operand, .. } = &err else {
        panic!("expected a dimension mismatch, got {err}");
    };
    assert_eq!(*axis, Axis::Columns);
    assert!(operand.contains(REGION_SET), "{operand}");
}

#[test]
fn a_vector_asserted_as_a_matrix_is_refused_as_a_kind_error_first() {
    // Kind before dimensions: asking whether a vector has the right column
    // count is not a question.
    let bytes = f32_bytes(&[1.0, 2.0]);
    let t = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::F32,
        ComponentContract::vector(2),
    )
    .unwrap();
    assert!(matches!(
        t.require_matrix(1, 2),
        Err(ExecutionError::NotAMatrix { .. })
    ));
}

// ── Quantisation padding is not observable through the view ────────────────
//
// Gemma's Q4_K `down` region is stored `[hidden, 768]` — the logical 704
// rounded up to a 256-multiple — while `gate_up` is unpadded. Binding the
// stored width and slicing to the logical one is how VINDEX3 expresses
// `physical shape != semantic operand shape` without rewriting the index,
// materialising a cropped copy, or padding the activation in the generic
// runtime.
//
// These use a poisoned tail: the padding columns hold values large enough
// that any leak into a dot product would be unmistakable.

/// Logical width, stored width, and a padding value that cannot hide.
const LOGICAL_COLS: u32 = 4;
const STORED_COLS: u32 = 6;
const POISON: f32 = 1.0e9;

/// `[2, 6]` where columns 4 and 5 are poison.
fn poisoned_tail() -> BoundTensor<'static> {
    let mut values = Vec::new();
    for row in 0..2 {
        for col in 0..STORED_COLS {
            values.push(if col < LOGICAL_COLS {
                (row * LOGICAL_COLS + col) as f32 + 1.0
            } else {
                POISON
            });
        }
    }
    let bytes: &'static [u8] = Box::leak(f32_bytes(&values).into_boxed_slice());
    BoundTensor::new(
        identity(),
        bytes,
        RegionFormat::F32,
        ComponentContract::matrix(2, STORED_COLS),
        ComponentView::Slice {
            dim: 1,
            start: 0,
            len: LOGICAL_COLS,
        },
    )
    .unwrap()
}

#[test]
fn a_padded_tail_is_invisible_to_the_logical_view() {
    let t = poisoned_tail();
    assert_eq!(
        t.cols(),
        LOGICAL_COLS as usize,
        "role sees the logical width"
    );
    for row in 0..t.rows() {
        let values = t.row(row).unwrap();
        assert_eq!(values.len(), LOGICAL_COLS as usize);
        assert!(
            values.iter().all(|v| v.abs() < POISON),
            "row {row} observed padding: {values:?}"
        );
    }
}

#[test]
fn the_logical_view_reads_the_right_values_not_merely_the_right_count() {
    // A view that returned the correct *number* of elements from the wrong
    // offsets would pass the test above.
    let t = poisoned_tail();
    assert_eq!(t.row(0).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.row(1).unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn the_padding_really_is_in_the_stored_bytes() {
    // Guards the guard: if the fixture never wrote poison, every assertion
    // above would pass against a tensor with nothing to leak.
    let t = poisoned_tail();
    let direct = BoundTensor::direct(
        identity(),
        t.bytes_for_test(),
        RegionFormat::F32,
        ComponentContract::matrix(2, STORED_COLS),
    )
    .unwrap();
    assert_eq!(direct.row(0).unwrap()[LOGICAL_COLS as usize], POISON);
}

#[test]
fn the_logical_width_is_what_a_shape_assertion_checks() {
    // A kernel bound to this operand must be told the logical width, or it
    // would read the padding as data.
    let t = poisoned_tail();
    t.require_matrix(2, LOGICAL_COLS as usize).unwrap();
    assert!(t.require_matrix(2, STORED_COLS as usize).is_err());
}

#[test]
fn a_bound_tensor_reports_its_provenance() {
    let t = matrix(ComponentView::Transpose);
    assert_eq!(t.format(), RegionFormat::F32);
    assert_eq!(*t.view(), ComponentView::Transpose);
    assert_eq!(t.representation().region_set, REGION_SET);
    assert!(t.describe().contains(VARIANT));
    assert_eq!(t.contract(), &ComponentContract::matrix(3, 2));
}

// ── Handing an f32 operand to a kernel that takes `&[f32]` ─────────────────
//
// The contract is hand-over-or-refuse. A bridge that dequantised, repacked
// into a kernel-shaped temporary and then called the kernel could reach
// numerical parity while proving nothing about the binding, so each refusal
// below names a different remedy rather than "conversion failed".

#[test]
fn a_direct_f32_matrix_hands_over_its_own_bytes() {
    let t = matrix(ComponentView::Direct);
    let values = t.as_f32_slice().unwrap();
    assert_eq!(values, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // The region's own memory, not a copy.
    assert!(std::ptr::eq(
        values.as_ptr().cast::<u8>(),
        t.bytes_for_test().as_ptr()
    ));
}

#[test]
fn a_non_f32_encoding_is_refused_by_format() {
    // Remedy: bind another variant, or a kernel for this format.
    let bytes = vec![0x80, 0x3F, 0x80, 0xBF];
    let t = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::BF16,
        ComponentContract::vector(2),
    )
    .unwrap();
    let err = t.as_f32_slice().unwrap_err();
    assert!(
        matches!(err, OperandUnsuitability::ElementFormat { .. }),
        "{err}"
    );
    assert!(err.to_string().contains("bf16"), "{err}");
}

#[test]
fn an_f32_operand_read_through_a_view_is_refused() {
    // Remedy: a view-aware kernel, or a repacked variant. Unlike `as_blocks`,
    // no view is honoured here: the kernel indexes the slice directly, so a
    // transpose or a narrowed row would silently read the wrong elements.
    for view in [
        ComponentView::Transpose,
        ComponentView::Slice {
            dim: COL_DIM,
            start: 0,
            len: 2,
        },
    ] {
        let err = matrix(view.clone()).as_f32_slice().unwrap_err();
        assert!(
            matches!(err, OperandUnsuitability::NonDirectView { .. }),
            "{view:?} should be refused, got {err}"
        );
    }
}

#[test]
fn a_misaligned_base_is_refused_as_alignment_not_as_corruption() {
    // Remedy: an aligned copy. Reached by binding at a one-byte offset into an
    // f32-aligned buffer, which is what an unaligned region offset produces.
    let aligned: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(aligned.as_ptr().cast::<u8>(), aligned.len() * 4) };
    let t = BoundTensor::direct(
        identity(),
        &bytes[1..],
        RegionFormat::F32,
        ComponentContract::vector(2),
    )
    .unwrap();
    assert!(matches!(
        t.as_f32_slice().unwrap_err(),
        OperandUnsuitability::MisalignedBase { .. }
    ));
}

// ── Handing blocks to a block-native kernel ────────────────────────────────
//
// `as_blocks` is the blocked counterpart of `as_f32_slice`: hand over the
// region's own bytes or refuse. Every refusal below is a *different* remedy,
// which is the whole reason they are separate causes.

/// Two rows of one super-block each — the smallest shape that can be wrong
/// about a row stride.
const BLOCK_ROWS: u32 = 2;
const SEED: usize = 3;

fn q4k_rows(cols: usize) -> BoundTensor<'static> {
    let bytes = q4k_bytes(BLOCK_ROWS as usize, cols, SEED);
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    BoundTensor::direct(
        identity(),
        leaked,
        RegionFormat::Q4K,
        ComponentContract::matrix(BLOCK_ROWS, cols as u32),
    )
    .expect("well-formed q4_k fixture")
}

#[test]
fn a_direct_q4k_matrix_hands_over_exactly_its_rows() {
    let t = q4k_rows(Q4K_BLOCK_ELEMS);
    let blocks = t.as_blocks(RegionFormat::Q4K).unwrap();
    assert_eq!(blocks.rows, BLOCK_ROWS as usize);
    assert_eq!(blocks.storage_cols, Q4K_BLOCK_ELEMS);
    assert_eq!(blocks.role_cols, Q4K_BLOCK_ELEMS);
    assert_eq!(blocks.row_bytes, Q4K_BLOCK_BYTES);
    assert_eq!(blocks.padding_cols(), 0);
    assert_eq!(blocks.bytes.len(), BLOCK_ROWS as usize * Q4K_BLOCK_BYTES);
}

#[test]
fn the_handed_over_bytes_are_the_regions_own() {
    // Not a copy, not a repack: the same address. A bridge that materialised a
    // kernel-shaped temporary could reach identical numbers while proving
    // nothing about the binding.
    let t = q4k_rows(Q4K_BLOCK_ELEMS);
    let blocks = t.as_blocks(RegionFormat::Q4K).unwrap();
    assert!(std::ptr::eq(
        blocks.bytes.as_ptr(),
        t.bytes_for_test().as_ptr()
    ));
}

#[test]
fn a_column_prefix_slice_reports_both_extents() {
    // The padded-`down` case. The kernel needs the stored width to stride
    // correctly and the role width to know what the operation means, so the
    // view is honoured rather than refused.
    let stored = 2 * Q4K_BLOCK_ELEMS;
    let role = stored - 1;
    let bytes = q4k_bytes(BLOCK_ROWS as usize, stored, SEED);
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let t = BoundTensor::new(
        identity(),
        leaked,
        RegionFormat::Q4K,
        ComponentContract::matrix(BLOCK_ROWS, stored as u32),
        ComponentView::Slice {
            dim: COL_DIM,
            start: 0,
            len: role as u32,
        },
    )
    .unwrap();

    let blocks = t.as_blocks(RegionFormat::Q4K).unwrap();
    assert_eq!(blocks.storage_cols, stored);
    assert_eq!(blocks.role_cols, role);
    assert_eq!(blocks.padding_cols(), 1);
    assert_eq!(blocks.row_bytes, 2 * Q4K_BLOCK_BYTES);
}

#[test]
fn a_different_codec_is_refused_by_format() {
    // Remedy: bind another variant, or a kernel for this format.
    let t = q4k_rows(Q4K_BLOCK_ELEMS);
    assert!(matches!(
        t.as_blocks(RegionFormat::Q6K).unwrap_err(),
        OperandUnsuitability::ElementFormat { .. }
    ));
}

#[test]
fn a_directly_addressed_region_is_refused_by_format() {
    // f32 has no blocks to hand over, and the message must not pretend the
    // problem is the view or the length.
    let t = matrix(ComponentView::Direct);
    assert!(matches!(
        t.as_blocks(RegionFormat::F32).unwrap_err(),
        OperandUnsuitability::ElementFormat { .. }
    ));
}

#[test]
fn a_transpose_is_refused_because_the_bytes_are_not_the_operand() {
    // Remedy: a view-aware kernel, or a repacked variant. A transposed read of
    // a blocked region touches every block for one logical row.
    let bytes = q4k_bytes(BLOCK_ROWS as usize, Q4K_BLOCK_ELEMS, SEED);
    let t = BoundTensor::new(
        identity(),
        &bytes,
        RegionFormat::Q4K,
        ComponentContract::matrix(BLOCK_ROWS, Q4K_BLOCK_ELEMS as u32),
        ComponentView::Transpose,
    )
    .unwrap();
    assert!(matches!(
        t.as_blocks(RegionFormat::Q4K).unwrap_err(),
        OperandUnsuitability::NonDirectView { .. }
    ));
}

#[test]
fn a_row_slice_is_refused_even_though_a_column_slice_is_not() {
    // The pair that makes the rule legible: a column prefix keeps whole stored
    // rows and is honoured; a row slice starts partway into the region, so the
    // bytes handed over would be the wrong ones.
    let bytes = q4k_bytes(BLOCK_ROWS as usize, Q4K_BLOCK_ELEMS, SEED);
    let t = BoundTensor::new(
        identity(),
        &bytes,
        RegionFormat::Q4K,
        ComponentContract::matrix(BLOCK_ROWS, Q4K_BLOCK_ELEMS as u32),
        ComponentView::Slice {
            dim: 0,
            start: 0,
            len: 1,
        },
    )
    .unwrap();
    assert!(matches!(
        t.as_blocks(RegionFormat::Q4K).unwrap_err(),
        OperandUnsuitability::NonDirectView { .. }
    ));
}

#[test]
fn a_column_slice_that_does_not_start_at_zero_is_refused() {
    let stored = 2 * Q4K_BLOCK_ELEMS;
    let bytes = q4k_bytes(BLOCK_ROWS as usize, stored, SEED);
    let t = BoundTensor::new(
        identity(),
        &bytes,
        RegionFormat::Q4K,
        ComponentContract::matrix(BLOCK_ROWS, stored as u32),
        ComponentView::Slice {
            dim: COL_DIM,
            start: 1,
            len: (stored - 1) as u32,
        },
    )
    .unwrap();
    assert!(matches!(
        t.as_blocks(RegionFormat::Q4K).unwrap_err(),
        OperandUnsuitability::NonDirectView { .. }
    ));
}

#[test]
fn rows_that_are_not_whole_blocks_are_refused_by_alignment() {
    // Remedy: repack, or pad the extent. There is no row stride to read at, so
    // the kernel could only guess one.
    let ragged = Q4K_BLOCK_ELEMS - 1;
    let bytes = vec![0u8; BLOCK_ROWS as usize * Q4K_BLOCK_BYTES];
    let t = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::Q4K,
        ComponentContract::matrix(BLOCK_ROWS, ragged as u32),
    )
    .unwrap();
    let err = t.as_blocks(RegionFormat::Q4K).unwrap_err();
    assert!(
        matches!(
            err,
            OperandUnsuitability::BlockAlignment { found, block, .. }
                if found == ragged && block == Q4K_BLOCK_ELEMS
        ),
        "{err}"
    );
}

#[test]
fn a_region_too_short_for_its_rows_is_refused_by_length() {
    // Remedy: reject the index — this one is the defect. Whole-region sizing
    // rounds up once; per-row sizing rounds up per row, so a two-row region
    // whose rows each need padding is longer than the flat count suggests.
    let stored = Q4K_BLOCK_ELEMS;
    let rows = 3usize;
    let bytes = vec![0u8; (rows - 1) * Q4K_BLOCK_BYTES];
    let t = BoundTensor::direct(
        identity(),
        &bytes,
        RegionFormat::Q4K,
        ComponentContract::matrix(rows as u32, stored as u32),
    );
    // Whole-region sizing already catches this one at bind time, which is the
    // stronger place to catch it.
    assert!(matches!(t.unwrap_err(), ExecutionError::ShortRegion { .. }));
}
