//! Colocated tests for `kernels` — the reference primitives.

use larql_compute::Activation;

use crate::runtime::kernels::{activate, dot, renormalize, softmax, top_k};

#[test]
fn a_dot_product_contracts_two_vectors() {
    assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    assert_eq!(dot(&[], &[]), 0.0);
}

#[test]
fn a_dot_product_stops_at_the_shorter_operand() {
    // Zip semantics, stated so a future rewrite to indexing keeps them.
    assert_eq!(dot(&[1.0, 2.0, 3.0], &[1.0, 1.0]), 3.0);
}

// ── Softmax ────────────────────────────────────────────────────────────────

#[test]
fn softmax_produces_a_distribution() {
    let mut v = vec![1.0, 2.0, 3.0];
    softmax(&mut v);
    assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(v[2] > v[1] && v[1] > v[0], "order preserved");
}

#[test]
fn softmax_is_shift_invariant() {
    let mut a = vec![1.0, 2.0, 3.0];
    let mut b = vec![101.0, 102.0, 103.0];
    softmax(&mut a);
    softmax(&mut b);
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-6, "{x} vs {y}");
    }
}

#[test]
fn softmax_survives_large_inputs_without_overflowing() {
    // The reason for the max-shift: exp(1000) is inf, and inf/inf is NaN.
    let mut v = vec![1000.0, 1000.5];
    softmax(&mut v);
    assert!(v.iter().all(|x| x.is_finite()), "{v:?}");
    assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn softmax_of_nothing_does_nothing() {
    let mut v: Vec<f32> = Vec::new();
    softmax(&mut v);
    assert!(v.is_empty());
}

// ── Top-k ──────────────────────────────────────────────────────────────────

#[test]
fn top_k_returns_the_highest_scores_descending() {
    let picked = top_k(&[0.1, 0.5, 0.2, 0.9], 2);
    assert_eq!(picked, vec![(3, 0.9), (1, 0.5)]);
}

#[test]
fn top_k_breaks_ties_toward_the_lower_index() {
    // Deterministic on purpose — an oracle that disagrees with itself between
    // runs cannot be used to check anything.
    let picked = top_k(&[0.5, 0.5, 0.5], 2);
    assert_eq!(
        picked.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[test]
fn top_k_larger_than_the_population_returns_everything() {
    assert_eq!(top_k(&[0.2, 0.8], 10).len(), 2);
}

#[test]
fn top_k_of_zero_selects_nothing() {
    assert!(top_k(&[0.2, 0.8], 0).is_empty());
}

// ── Renormalisation ────────────────────────────────────────────────────────

#[test]
fn renormalisation_makes_weights_sum_to_one() {
    let mut w = vec![0.3, 0.1];
    renormalize(&mut w);
    assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!((w[0] - 0.75).abs() < 1e-6);
}

#[test]
fn renormalising_a_zero_sum_leaves_it_alone() {
    // Dividing by zero here would turn a degenerate routing into NaNs that
    // propagate through the whole residual stream.
    let mut w = vec![0.0, 0.0];
    renormalize(&mut w);
    assert_eq!(w, vec![0.0, 0.0]);
}

// ── Activations ────────────────────────────────────────────────────────────

#[test]
fn every_activation_fixes_zero_at_zero() {
    for activation in [
        Activation::Silu,
        Activation::GeluTanh,
        Activation::GeluExact,
        Activation::ReLU,
    ] {
        let mut v = vec![0.0f32];
        activate(activation, &mut v);
        assert!(v[0].abs() < 1e-6, "{activation:?} maps 0 to {}", v[0]);
    }
}

#[test]
fn silu_matches_its_definition() {
    let mut v = vec![1.0f32];
    activate(Activation::Silu, &mut v);
    assert!((v[0] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
}

#[test]
fn relu_clamps_negatives() {
    let mut v = vec![-2.0, 3.0];
    activate(Activation::ReLU, &mut v);
    assert_eq!(v, vec![0.0, 3.0]);
}

#[test]
fn the_two_gelus_agree_closely() {
    // The tanh approximation and the erf form differ by well under a percent;
    // a much larger gap would mean one of them is wrong.
    for x in [-2.0f32, -0.5, 0.5, 2.0] {
        let mut tanh_form = vec![x];
        let mut exact_form = vec![x];
        activate(Activation::GeluTanh, &mut tanh_form);
        activate(Activation::GeluExact, &mut exact_form);
        assert!(
            (tanh_form[0] - exact_form[0]).abs() < 1e-2,
            "at {x}: {} vs {}",
            tanh_form[0],
            exact_form[0]
        );
    }
}

#[test]
fn gelu_is_monotone_over_the_range_that_matters() {
    let mut v: Vec<f32> = (0..20).map(|i| i as f32 * 0.25).collect();
    activate(Activation::GeluTanh, &mut v);
    assert!(v.windows(2).all(|w| w[1] >= w[0]), "{v:?}");
}

#[test]
fn activations_apply_elementwise_across_a_slice() {
    let mut v = vec![-1.0, 0.0, 1.0];
    activate(Activation::ReLU, &mut v);
    assert_eq!(v, vec![0.0, 0.0, 1.0]);
}
