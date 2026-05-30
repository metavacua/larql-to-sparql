//! Cross-check the lifted `dequantize_q4_k` in `cpu::ops::q4_common` against
//! `larql_models::quant::ggml::dequantize_q4_k` (the original source). Both
//! must produce bit-identical output for the same Q4_K bytes.
//!
//! Catches silent drift between the two implementations during refactors.
//!
//! Gated on the `heavy_tests` feature — see `test_correctness.rs` for
//! the rationale.
#![cfg(feature = "heavy_tests")]

use larql_compute::cpu::ops::q4_common::{dequantize_q4_k, quantize_q4_k};

#[test]
fn q4k_lifted_matches_larql_models_reference() {
    // Three super-blocks of varied data: smooth ramp, sparse spikes, noise.
    let n = 256 * 3;
    let mut data: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        let v = if i % 64 == 0 {
            (t * 4.0).sin() * 2.5
        } else {
            (t - 0.5) * 1.7
        };
        data.push(v);
    }

    let bytes = quantize_q4_k(&data);
    assert_eq!(
        bytes.len(),
        144 * 3,
        "Q4_K = 144 bytes per 256-elem super-block"
    );

    let lifted = dequantize_q4_k(&bytes, n);
    let reference =
        larql_models::quant::ggml::dequantize_q4_k(&bytes, n).expect("reference dequant");

    assert_eq!(lifted.len(), reference.len(), "length mismatch");
    for (i, (a, b)) in lifted.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bit drift at element {i}: lifted={a} reference={b}"
        );
    }
}

#[test]
fn q4k_round_trip_within_quant_noise() {
    // Smooth ramp [-1, 1]: worst case for block-level scales.
    let data: Vec<f32> = (0..256 * 4)
        .map(|i| (i as f32 / (256.0 * 4.0 - 1.0)) * 2.0 - 1.0)
        .collect();
    let bytes = quantize_q4_k(&data);
    let decoded = dequantize_q4_k(&bytes, data.len());

    let max_err: f32 = data
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    // Q4 nibble step ≈ 0.13 over 2.0 range; allow 2× for sub-block bias.
    assert!(max_err < 0.12, "Q4_K round-trip max error {max_err}");
}

#[test]
fn q4k_misaligned_input_returns_empty() {
    // n_elements not a multiple of 256 → empty fallback (no panic).
    let bytes = vec![0u8; 144];
    let out = dequantize_q4_k(&bytes, 200);
    assert!(out.is_empty());
}

#[test]
fn q4k_truncated_input_returns_empty() {
    // bytes too short for the requested element count.
    let bytes = vec![0u8; 100]; // < 144
    let out = dequantize_q4_k(&bytes, 256);
    assert!(out.is_empty());
}
