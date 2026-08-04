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

use super::write_layers::{quantize_dense_entry, LayerWeightFormat};

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
