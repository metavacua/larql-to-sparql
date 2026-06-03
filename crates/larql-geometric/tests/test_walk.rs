// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use larql_geometric::attention::{AttnInput, AttnBackend};
use larql_geometric::attention::walk::WalkAttn;
use larql_geometric::attention::reference::standard_attn;

fn make_input(n: usize, d: usize, seed: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> = (0..n*d).map(|i| ((i as f32 + seed)*0.11).sin()).collect();
    let k: Vec<f32> = (0..n*d).map(|i| ((i as f32 + seed)*0.07 + 0.3).cos()).collect();
    let v: Vec<f32> = (0..n*d).map(|i| ((i as f32 + seed)*0.05).sin() * 0.5).collect();
    (q, k, v)
}

#[test]
fn test_walk_k_equals_n_matches_standard() {
    let (q, k, v) = make_input(6, 8, 1.0);
    let input = AttnInput { q: &q, k: &k, v: &v, seq_len: 6, head_dim: 8 };
    let walk = WalkAttn::new(6, 8);  // K=n, route_dim=head_dim → exact match
    let std_out = standard_attn(&input);
    let walk_out = walk.forward(&input);
    for (a, b) in walk_out.out.iter().zip(std_out.out.iter()) {
        assert!((a - b).abs() < 1e-4, "walk K=n differs from standard: {a} vs {b}");
    }
}

#[test]
fn test_walk_k1_attends_single_key() {
    let (q, k, v) = make_input(4, 8, 2.0);
    let input = AttnInput { q: &q, k: &k, v: &v, seq_len: 4, head_dim: 8 };
    let walk = WalkAttn::new(1, 8);  // K=1
    let out = walk.forward(&input);
    for i in 0..4 {
        let out_row = &out.out[i*8..(i+1)*8];
        let matches_some_v = (0..4).any(|j| {
            let v_row = &v[j*8..(j+1)*8];
            out_row.iter().zip(v_row.iter()).all(|(a,b)| (a-b).abs() < 1e-5)
        });
        assert!(matches_some_v, "K=1 output at {i} should equal exactly one V row");
    }
}

#[test]
fn test_walk_output_shape() {
    let (q, k, v) = make_input(8, 16, 0.0);
    let input = AttnInput { q: &q, k: &k, v: &v, seq_len: 8, head_dim: 16 };
    let out = WalkAttn::new(3, 16).forward(&input);
    assert_eq!(out.out.len(), 8 * 16);
    assert!(out.out.iter().all(|x| x.is_finite()));
}
