//! Reference-kernel unit gates: each op against hand-computed values.

use larql_models::config::{Activation, NormType};

use crate::format::vindex3::opplan::exec::kernels::{
    activate, matvec, norm, rope_rotate, sigmoid, softcap, softmax,
};

#[test]
fn matvec_is_row_major_out_by_in() {
    // w = [[1,2],[3,4],[5,6]] (3x2), x = [10, 100] → [210, 430, 650].
    let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = matvec(&w, 3, 2, &[10.0, 100.0]);
    assert_eq!(y, vec![210.0, 430.0, 650.0]);
}

#[test]
fn rms_norm_matches_hand_computation() {
    // x = [3, 4]: rms = sqrt(12.5), weight 2.0, offset 0.
    let y = norm(NormType::RmsNorm, &[3.0, 4.0], &[2.0, 2.0], 0.0, 0.0);
    let rms = (12.5f64).sqrt();
    assert!((y[0] as f64 - 3.0 / rms * 2.0).abs() < 1e-6);
    assert!((y[1] as f64 - 4.0 / rms * 2.0).abs() < 1e-6);
    // Weight offset adds before multiplying.
    let offset = norm(NormType::RmsNorm, &[3.0, 4.0], &[1.0, 1.0], 1.0, 0.0);
    assert!((offset[0] - y[0]).abs() < 1e-6);
    // Parameter-free: empty weight = unit gain, statistic only.
    let free = norm(NormType::RmsNorm, &[3.0, 4.0], &[], 0.0, 0.0);
    assert!((free[0] as f64 - 3.0 / rms).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "norm weight must be empty or 3 long, got 2")]
fn rms_norm_refuses_a_short_weight_rather_than_padding_it() {
    // A weight that is neither empty nor `x`-length is a geometry bug.
    // Padding the tail would return finite-but-wrong numbers, surfacing
    // later as unexplained drift in a parity table instead of here.
    norm(NormType::RmsNorm, &[1.0, 2.0, 3.0], &[1.0, 1.0], 0.0, 0.0);
}

#[test]
#[should_panic(expected = "norm weight must be empty or 3 long, got 4")]
fn layer_norm_refuses_an_over_long_weight() {
    norm(
        NormType::LayerNorm,
        &[1.0, 2.0, 3.0],
        &[1.0, 1.0, 1.0, 1.0],
        0.0,
        0.0,
    );
}

#[test]
fn parameter_free_norm_ignores_the_weight_offset() {
    // Weightless normalisation is the statistic alone: an offset that
    // would apply to a stored weight must not leak in through the
    // empty-weight path.
    let free = norm(NormType::RmsNorm, &[3.0, 4.0], &[], 5.0, 0.0);
    let rms = (12.5f64).sqrt();
    assert!((free[0] as f64 - 3.0 / rms).abs() < 1e-6);
    assert!((free[1] as f64 - 4.0 / rms).abs() < 1e-6);
}

#[test]
fn layer_norm_centres_and_scales() {
    // x = [1, 3]: mean 2, var 1 → normalised [-1, 1].
    let y = norm(NormType::LayerNorm, &[1.0, 3.0], &[1.0, 1.0], 0.0, 0.0);
    assert!((y[0] + 1.0).abs() < 1e-6);
    assert!((y[1] - 1.0).abs() < 1e-6);
}

#[test]
fn activations_match_reference_points() {
    assert_eq!(activate(Activation::Relu, -2.0), 0.0);
    assert_eq!(activate(Activation::Relu, 2.0), 2.0);
    assert!((activate(Activation::Silu, 1.0) - 1.0 * sigmoid(1.0)).abs() < 1e-7);
    // GELU(1) ≈ 0.841345; tanh approximation ≈ 0.841192.
    assert!((activate(Activation::Gelu, 1.0) - 0.841_345).abs() < 1e-3);
    assert!((activate(Activation::GeluTanh, 1.0) - 0.841_192).abs() < 1e-4);
    assert_eq!(activate(Activation::Gelu, 0.0), 0.0);
}

#[test]
fn softmax_is_stable_and_normalised() {
    let mut s = vec![1000.0, 1001.0, 1002.0];
    softmax(&mut s);
    let sum: f32 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(s[2] > s[1] && s[1] > s[0]);
}

#[test]
fn rope_is_identity_at_position_zero_and_norm_preserving() {
    let mut head = vec![1.0, 2.0, 3.0, 4.0];
    rope_rotate(&mut head, 0, 10_000.0);
    assert_eq!(head, vec![1.0, 2.0, 3.0, 4.0]);

    rope_rotate(&mut head, 7, 10_000.0);
    assert_ne!(head, vec![1.0, 2.0, 3.0, 4.0]);
    // Rotation preserves the norm of each rotated pair.
    let norm_sq: f32 = head.iter().map(|v| v * v).sum();
    assert!((norm_sq - 30.0).abs() < 1e-4);
}

#[test]
fn softcap_saturates_at_the_cap() {
    assert!(softcap(1000.0, 20.0) <= 20.0);
    assert!(softcap(1000.0, 20.0) > 19.9);
    assert!((softcap(0.1, 20.0) - 0.1).abs() < 1e-4);
    assert!(softcap(-1000.0, 20.0) >= -20.0);
}
