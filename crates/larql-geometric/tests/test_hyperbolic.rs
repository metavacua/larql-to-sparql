// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use larql_geometric::hyperbolic::{exp_map_zero, geodesic_dist, mobius_add};
use larql_geometric::attention::{AttnInput, AttnBackend};
use larql_geometric::attention::hyperbolic_attn::HyperbolicAttn;

#[test]
fn test_geodesic_dist_origin_to_origin() {
    let z = vec![0.0f32; 4];
    let d = geodesic_dist(&z, &z);
    assert!(d.abs() < 1e-6, "d(0,0) should be 0, got {d}");
}

#[test]
fn test_geodesic_dist_positive() {
    let a = vec![0.3f32, 0.0, 0.0, 0.0];
    let b = vec![0.0f32, 0.3, 0.0, 0.0];
    assert!(geodesic_dist(&a, &b) > 0.0);
}

#[test]
fn test_exp_map_zero_lands_in_ball() {
    let v = vec![1.5f32, -0.8, 2.0, 0.1];
    let p = exp_map_zero(&v);
    let norm_sq: f32 = p.iter().map(|x| x*x).sum();
    assert!(norm_sq < 1.0, "exp_map_zero result has norm_sq={norm_sq}, must be < 1.0");
}

#[test]
fn test_hyperbolic_attn_output_shape() {
    let q: Vec<f32> = (0..4*8).map(|i| ((i as f32)*0.1).sin()).collect();
    let k = q.clone();
    let v: Vec<f32> = (0..4*8).map(|i| (i as f32) * 0.01).collect();
    let input = AttnInput { q: &q, k: &k, v: &v, seq_len: 4, head_dim: 8 };
    let backend = HyperbolicAttn::new(1.0);
    let out = backend.forward(&input);
    assert_eq!(out.out.len(), 4 * 8);
    assert!(out.out.iter().all(|x| x.is_finite()), "output has NaN/Inf");
}
