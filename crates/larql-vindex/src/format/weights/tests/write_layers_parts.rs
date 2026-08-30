//! Colocated tests for `quantize_dense_entry` — the separate-tensor
//! expert assembler.
//!
//! The load-bearing assertion is the **row order** inside `gate_up`: gate rows
//! first, then up rows, contiguous. Reversing it does not crash and does not
//! change a single byte count — it silently swaps the two halves of every
//! expert's GLU, which is the plausible-wrong-numbers failure this codebase
//! keeps designing against. So the order is pinned against the same split the
//! consumer performs (`cpu/ops/moe/expert` slices at `inter * hidden`), not
//! against a hand-written byte string.

use super::super::write_layers::{quantize_dense_entry, LayerWeightFormat};

const INTER: usize = 2;
const HIDDEN: usize = 256; // Q4_K needs a 256 multiple on the contracted dim
const GATE_FILL: f32 = 1.0;
const UP_FILL: f32 = -1.0;
const DOWN_FILL: f32 = 0.5;

fn gate() -> Vec<f32> {
    vec![GATE_FILL; INTER * HIDDEN]
}

fn up() -> Vec<f32> {
    vec![UP_FILL; INTER * HIDDEN]
}

fn down() -> Vec<f32> {
    vec![DOWN_FILL; HIDDEN * INTER]
}

#[test]
fn f32_entry_places_gate_rows_before_up_rows() {
    // At F32 the payload is the input verbatim, so the split is directly
    // observable — the property every quantised format must also preserve.
    let entry = quantize_dense_entry(
        &gate(),
        &up(),
        &down(),
        INTER,
        HIDDEN,
        LayerWeightFormat::F32,
    )
    .unwrap();

    let floats: Vec<f32> = entry
        .gate_up
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(floats.len(), 2 * INTER * HIDDEN);
    let (gate_half, up_half) = floats.split_at(INTER * HIDDEN);
    assert!(
        gate_half.iter().all(|&v| v == GATE_FILL),
        "gate rows must come first"
    );
    assert!(
        up_half.iter().all(|&v| v == UP_FILL),
        "up rows must come second"
    );
}

#[test]
fn gate_up_is_twice_the_single_projection_size() {
    let entry = quantize_dense_entry(
        &gate(),
        &up(),
        &down(),
        INTER,
        HIDDEN,
        LayerWeightFormat::F32,
    )
    .unwrap();
    assert_eq!(entry.gate_up.len(), 2 * INTER * HIDDEN * 4);
}

#[test]
fn down_is_padded_to_the_block_boundary() {
    // inter = 2 pads to 256 under a block format, so `down` is written at
    // [hidden, 256] rather than [hidden, 2].
    let entry = quantize_dense_entry(
        &gate(),
        &up(),
        &down(),
        INTER,
        HIDDEN,
        LayerWeightFormat::F32,
    )
    .unwrap();
    assert_eq!(entry.down.len(), HIDDEN * 256 * 4);
}

#[test]
fn a_quantised_entry_is_smaller_than_the_f32_one() {
    let f32_entry = quantize_dense_entry(
        &gate(),
        &up(),
        &down(),
        INTER,
        HIDDEN,
        LayerWeightFormat::F32,
    )
    .unwrap();
    let q4k_entry = quantize_dense_entry(
        &gate(),
        &up(),
        &down(),
        INTER,
        HIDDEN,
        LayerWeightFormat::Q4_K,
    )
    .unwrap();
    assert!(q4k_entry.gate_up.len() < f32_entry.gate_up.len());
    assert!(q4k_entry.down.len() < f32_entry.down.len());
}

#[test]
fn a_short_gate_is_refused_by_shape() {
    let err = quantize_dense_entry(
        &vec![GATE_FILL; INTER * HIDDEN - 1],
        &up(),
        &down(),
        INTER,
        HIDDEN,
        LayerWeightFormat::F32,
    )
    .unwrap_err();
    assert!(err.to_string().contains("gate/up"), "{err}");
}

#[test]
fn a_short_up_is_refused_by_shape() {
    let err = quantize_dense_entry(
        &gate(),
        &[UP_FILL; 3],
        &down(),
        INTER,
        HIDDEN,
        LayerWeightFormat::F32,
    )
    .unwrap_err();
    assert!(err.to_string().contains("gate/up"), "{err}");
}

#[test]
fn a_transposed_down_is_refused_rather_than_reinterpreted() {
    // [inter, hidden] instead of [hidden, inter] has the *same element count*,
    // so only an explicit shape check catches it. Here the counts differ so the
    // check fires; the guard exists because the symmetric case would not.
    let err = quantize_dense_entry(
        &gate(),
        &up(),
        &vec![DOWN_FILL; HIDDEN * INTER - 1],
        INTER,
        HIDDEN,
        LayerWeightFormat::F32,
    )
    .unwrap_err();
    assert!(err.to_string().contains("down"), "{err}");
}

// ── quantize_dense_entry: the shape-refusal arms ─────────────────────────
//
// A mis-shaped input would otherwise quantise happily into a store whose
// row stride disagrees with every reader (the padded-row geometry).

#[test]
fn dense_entry_refuses_a_mis_sized_gate_or_up() {
    use crate::format::weights::write_layers::{quantize_dense_entry, LayerWeightFormat};
    let err = quantize_dense_entry(
        &[1.0; 3], // gate: not inter × hidden
        &[1.0; 4],
        &[1.0; 4],
        2,
        2,
        LayerWeightFormat::F32,
    )
    .unwrap_err();
    assert!(err.to_string().contains("gate/up must each be"), "{err}");
}

#[test]
fn dense_entry_refuses_a_mis_sized_down() {
    use crate::format::weights::write_layers::{quantize_dense_entry, LayerWeightFormat};
    let err = quantize_dense_entry(
        &[1.0; 4],
        &[1.0; 4],
        &[1.0; 3], // down: not hidden × inter
        2,
        2,
        LayerWeightFormat::F32,
    )
    .unwrap_err();
    assert!(err.to_string().contains("down must be"), "{err}");
}

/// Gate/up rows are stored at the padded width and the two halves must not
/// interleave: gate rows land first, then up rows, each padded rowwise.
/// Pinned on values, not sizes — a concatenate-then-pad (padding the fused
/// buffer as if rows were 2×hidden wide) produces the same byte count with
/// the wrong layout.
#[test]
fn dense_entry_pads_each_half_rowwise_before_concatenation() {
    use crate::format::weights::write_layers::{quantize_dense_entry, LayerWeightFormat};
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let entry = quantize_dense_entry(
        &[1.0, 2.0, 3.0, 4.0], // gate [2,2]
        &[5.0, 6.0, 7.0, 8.0], // up   [2,2]
        &[9.0, 10.0, 11.0, 12.0],
        2,
        2,
        LayerWeightFormat::F32,
    )
    .unwrap();
    let floats: Vec<f32> = entry
        .gate_up
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    // Row r of gate at r*block; up rows start at 2*block.
    assert_eq!(&floats[0..2], &[1.0, 2.0]);
    assert_eq!(floats[2], 0.0, "row padding must be zero");
    assert_eq!(&floats[block..block + 2], &[3.0, 4.0]);
    assert_eq!(&floats[2 * block..2 * block + 2], &[5.0, 6.0]);
    assert_eq!(&floats[3 * block..3 * block + 2], &[7.0, 8.0]);
}

/// The wire code and the registry tag must agree, per variant, in both
/// directions.
///
/// `as_u32` is what goes on disk and `registry_tag` is what the loader
/// matches to pick a decoder — the two are written as separate `match`
/// arms over the same enum, so nothing but a test stops one gaining a
/// variant the other spells differently. A store whose header says 5 and
/// whose registry says `Q4_K` decodes a Q6_K expert bank as Q4_K garbage,
/// which is the failure `registry_tag`'s own doc comment records.
#[test]
fn every_layer_format_round_trips_its_code_and_tag() {
    use std::collections::BTreeSet;

    // Every variant, listed here so adding one to the enum without adding
    // it here leaves the new arm visibly unpinned rather than silently
    // untested.
    const ALL: [LayerWeightFormat; 8] = [
        LayerWeightFormat::F32,
        LayerWeightFormat::F16,
        LayerWeightFormat::BF16,
        LayerWeightFormat::Q4_0,
        LayerWeightFormat::Q4_K,
        LayerWeightFormat::Q6_K,
        LayerWeightFormat::Q8_0,
        LayerWeightFormat::FP4,
    ];

    let mut codes = BTreeSet::new();
    let mut tags = BTreeSet::new();
    for f in ALL {
        assert!(
            codes.insert(f.as_u32()),
            "{f:?} reuses wire code {}",
            f.as_u32()
        );
        assert!(tags.insert(f.registry_tag()), "{f:?} reuses tag");
        assert!(!f.registry_tag().is_empty());
    }
    // Codes are the on-disk contract: they must be the dense 0..8 range the
    // header parser matches on, not merely distinct.
    assert_eq!(
        codes.iter().copied().collect::<Vec<_>>(),
        (0..ALL.len() as u32).collect::<Vec<_>>()
    );
}

/// A header carrying a code no variant claims is refused, not defaulted.
///
/// Defaulting an unknown quant code to Q4_K is precisely how a future
/// format would be decoded as garbage by an older binary.
#[test]
fn an_unknown_quant_code_is_refused_by_the_header_parser() {
    use super::super::write_layers::{parse_layer_weights_header, LayerWeightFormat};

    // Build the smallest header the parser accepts, then vary only the
    // quant word — so a refusal can only be about that field.
    // Spelled out rather than imported: MAGIC/FORMAT_VERSION are private to
    // `write_layers`, and a literal header is also a check that the on-disk
    // shape is what this test thinks it is.
    let header = |quant: u32| -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"LYRW"); // magic
        v.extend_from_slice(&1u32.to_le_bytes()); // format_version
        v.extend_from_slice(&quant.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes()); // num_entries: no table
        v.extend_from_slice(&8u32.to_le_bytes()); // inter
        v.extend_from_slice(&16u32.to_le_bytes()); // hidden
        v.resize(256, 0); // comfortably past HEADER_BYTES
        v
    };

    let known = parse_layer_weights_header(&header(LayerWeightFormat::Q6_K.as_u32()));
    assert!(known.is_some(), "a known code must parse");
    // MXFP4 took code 8 — the refusal boundary moved with it rather than
    // disappearing. This header's layout block reads as two zero words,
    // which is Inline + ContiguousHalves.
    let mxfp4 = parse_layer_weights_header(&header(LayerWeightFormat::MXFP4.as_u32()))
        .expect("MXFP4 is a known code now");
    assert_eq!(mxfp4.format, LayerWeightFormat::MXFP4);
    // One past the last variant, and a far-future one.
    assert!(parse_layer_weights_header(&header(9)).is_none());
    assert!(parse_layer_weights_header(&header(9_999)).is_none());
}

/// `LayerEntry`'s Debug reports sizes, never contents.
///
/// It is printed in refusal messages on multi-megabyte expert banks; a
/// derived Debug would dump the byte vectors into a diagnostic.
#[test]
fn layer_entry_debug_reports_sizes_not_bytes() {
    use super::super::write_layers::LayerEntry;

    let entry = LayerEntry {
        gate_up: vec![7u8; 33],
        down: vec![9u8; 5],
        scales: None,
    };
    let rendered = format!("{entry:?}");
    assert!(rendered.contains("33"), "{rendered}");
    assert!(rendered.contains('5'), "{rendered}");
    assert!(
        !rendered.contains("7, 7, 7"),
        "Debug must not dump payload bytes: {rendered}"
    );
}
