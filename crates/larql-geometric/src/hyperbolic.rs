// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean

/// Möbius addition: u ⊕ v in the Poincaré ball.
pub fn mobius_add(u: &[f32], v: &[f32]) -> Vec<f32> {
    let d = u.len();
    debug_assert_eq!(d, v.len());
    let uu: f32 = u.iter().map(|x| x*x).sum();
    let vv: f32 = v.iter().map(|x| x*x).sum();
    let uv: f32 = u.iter().zip(v.iter()).map(|(a,b)| a*b).sum();
    let denom = (1.0 + 2.0*uv + uu*vv).max(1e-8);
    let a = 1.0 + 2.0*uv + vv;
    let b = 1.0 - uu;
    (0..d).map(|i| (a * u[i] + b * v[i]) / denom).collect()
}

/// Negate in Poincaré ball (just flip sign in ℝ^d).
pub fn mobius_neg(u: &[f32]) -> Vec<f32> {
    u.iter().map(|x| -x).collect()
}

/// Geodesic distance: d(u, v) = 2 · arctanh(|| (-u) ⊕ v ||).
pub fn geodesic_dist(u: &[f32], v: &[f32]) -> f32 {
    let neg_u = mobius_neg(u);
    let diff = mobius_add(&neg_u, v);
    let norm: f32 = diff.iter().map(|x| x*x).sum::<f32>().sqrt();
    let norm = norm.min(1.0 - 1e-6);
    2.0 * norm.atanh()
}

/// Exponential map at origin: exp_0(v) = tanh(||v||/2) · v / ||v||.
pub fn exp_map_zero(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
    if norm < 1e-8 { return vec![0.0; v.len()]; }
    let scale = (norm / 2.0).tanh() / norm;
    v.iter().map(|x| x * scale).collect()
}
