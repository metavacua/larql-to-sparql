//! Reference numeric primitives.
//!
//! Plain, allocating, obviously-correct implementations. They exist to say
//! what the answer should be, so that a fast kernel has something to be
//! checked against. Nothing here should be optimised: the moment this file
//! becomes clever it stops being usable as an oracle.
//!
//! # Tie-breaking is a contract here, not an accident
//!
//! The incumbent MoE path selects top-k with `sort_unstable_by` over
//! `partial_cmp`. That is **unspecified for ties** — not necessarily random,
//! and it may well reproduce the same order within a build, but it promises
//! nothing. A reference that offers no ordering guarantee cannot serve as an
//! oracle, so this one commits:
//!
//! ```text
//! score descending, then expert id ascending
//! ```
//!
//! The comparison uses `f32::total_cmp`, a genuine total order, rather than
//! `partial_cmp` with a fallback. A fallback to `Equal` hands incomparable
//! values whatever position the sort happens to leave them in, which is
//! precisely the unspecified behaviour being replaced.
//!
//! Non-finite scores are refused upstream in `execute` rather than ordered:
//! a NaN that acquires a position by sort mechanics is a routing decision
//! nobody made.

use larql_compute::Activation;

/// `out[r] = dot(matrix_row(r), x)`, with rows supplied by the caller.
///
/// Taking a row-reader rather than a slice is what lets one implementation
/// serve fused and decomposed storage, quantised and dense, transposed and
/// direct — the arrangement is resolved by the reader, not here.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// In-place softmax, max-shifted for stability.
pub fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// The `k` highest scores, descending, ties resolved to the lower index.
pub fn top_k(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    top_k_with_margin(scores, k).0
}

/// Top-k plus the **boundary margin**: `score[k-1] - score[k]`.
///
/// The margin is how much room the selection had. It is `None` when there is
/// no boundary — `k` is zero, or `k` covers the whole population.
///
/// It exists for parity triage. When VINDEX2 and VINDEX3 disagree about which
/// experts ran, a margin of exactly zero says the two implementations met an
/// exact tie and resolved it under different policies; a wide margin says they
/// disagreed about the *scores*, which is a weight-decoding or arithmetic
/// fault. Those need completely different investigations, and without the
/// margin the two are indistinguishable from the outside.
pub fn top_k_with_margin(scores: &[f32], k: usize) -> (Vec<(usize, f32)>, Option<f32>) {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    // `total_cmp` rather than `partial_cmp(..).unwrap_or(Equal)`: a total
    // order, so the result does not depend on sort mechanics for any input.
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    let k = k.min(scores.len());
    let margin = (k > 0 && k < indexed.len()).then(|| indexed[k - 1].1 - indexed[k].1);
    indexed.truncate(k);
    (indexed, margin)
}

/// Rescale so the selected weights sum to one. A zero sum is left alone.
pub fn renormalize(weights: &mut [f32]) {
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
}

/// Apply an activation elementwise, sharing `larql-compute`'s vocabulary.
pub fn activate(activation: Activation, v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = apply(activation, *x);
    }
}

fn apply(activation: Activation, x: f32) -> f32 {
    match activation {
        Activation::Silu => x / (1.0 + (-x).exp()),
        Activation::GeluTanh => {
            const SQRT_2_OVER_PI: f32 = 0.797_884_6;
            const CUBIC_COEFF: f32 = 0.044_715;
            let inner = SQRT_2_OVER_PI * (x + CUBIC_COEFF * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
        Activation::GeluExact => {
            const SQRT_2: f32 = std::f32::consts::SQRT_2;
            0.5 * x * (1.0 + erf(x / SQRT_2))
        }
        Activation::ReLU => x.max(0.0),
    }
}

/// Abramowitz–Stegun 7.1.26. Reference-grade, not bit-exact with libm.
fn erf(x: f32) -> f32 {
    const A: [f32; 5] = [
        0.254_829_6,
        -0.284_496_74,
        1.421_413_7,
        -1.453_152,
        1.061_405_4,
    ];
    const P: f32 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = A.iter().rev().fold(0.0f32, |acc, &coeff| (acc + coeff) * t);
    sign * (1.0 - poly * (-x * x).exp())
}
