// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use larql_geometric::attention::{AttnInput, AttnBackend};
use larql_geometric::attention::linear_complex::ComplexLinearAttn;

fn tiny_qkv(seq: usize, dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> = (0..seq*dim).map(|i| ((i as f32)*0.07).sin()).collect();
    let k: Vec<f32> = (0..seq*dim).map(|i| ((i as f32)*0.13 + 0.5).cos()).collect();
    let v: Vec<f32> = (0..seq*dim).map(|i| ((i as f32)*0.05).sin() * 0.5).collect();
    (q, k, v)
}

#[test]
fn test_complex_linear_attn_output_shape() {
    let (q, k, v) = tiny_qkv(4, 8);
    let input = AttnInput { q: &q, k: &k, v: &v, seq_len: 4, head_dim: 8 };
    let backend = ComplexLinearAttn::new();
    let out = backend.forward(&input);
    assert_eq!(out.out.len(), 4 * 8);
    assert_eq!(out.seq_len, 4);
    assert_eq!(out.head_dim, 8);
}

#[test]
fn test_complex_linear_attn_no_nan() {
    let (q, k, v) = tiny_qkv(8, 16);
    let input = AttnInput { q: &q, k: &k, v: &v, seq_len: 8, head_dim: 16 };
    let backend = ComplexLinearAttn::new();
    let out = backend.forward(&input);
    assert!(out.out.iter().all(|x| x.is_finite()), "output contains NaN/Inf");
}

#[test]
#[ignore = "requires vindex at ~/larql-vindexes/smollm2-360m"]
fn test_vindex_loader_reads_qkvo() {
    use larql_geometric::vindex_loader::load_layer_attn;
    let path = std::path::Path::new(&std::env::var("HOME").unwrap()).join("larql-vindexes/smollm2-360m");
    let weights = load_layer_attn(&path, 0).expect("failed to load layer 0 attn weights");
    assert_eq!(weights.q_proj.len(), 960 * 960, "q_proj wrong size");
}
