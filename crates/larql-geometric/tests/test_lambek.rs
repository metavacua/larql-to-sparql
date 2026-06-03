// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use larql_geometric::attention::{AttnInput, AttnBackend};
use larql_geometric::attention::lambek::LambekAttn;

#[test]
fn test_lambek_exchange_rule_violated() {
    let d = 8;
    let backend = LambekAttn::default_init(d);

    // AB: [token_0.3, token_0.7]
    let q_ab: Vec<f32> = (0..2*d).map(|i| if i < d { 0.3 } else { 0.7 }).collect();
    let out_ab = backend.forward(&AttnInput { q: &q_ab, k: &q_ab, v: &q_ab, seq_len: 2, head_dim: d });

    // BA: [token_0.7, token_0.3]  (swapped)
    let q_ba: Vec<f32> = (0..2*d).map(|i| if i < d { 0.7 } else { 0.3 }).collect();
    let out_ba = backend.forward(&AttnInput { q: &q_ba, k: &q_ba, v: &q_ba, seq_len: 2, head_dim: d });

    // Output at position 0 MUST differ between AB and BA
    let max_diff: f32 = out_ab.out[..d].iter()
        .zip(out_ba.out[..d].iter())
        .map(|(a,b)| (a-b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-4, "Exchange rule holds (max_diff={max_diff:.6}) — backend is broken");
}

#[test]
fn test_lambek_output_shape() {
    let d = 8;
    let q: Vec<f32> = (0..6*d).map(|i| ((i as f32)*0.1).sin()).collect();
    let k: Vec<f32> = (0..6*d).map(|i| ((i as f32)*0.07+0.5).cos()).collect();
    let v: Vec<f32> = (0..6*d).map(|i| ((i as f32)*0.05).sin()*0.5).collect();
    let input = AttnInput { q: &q, k: &k, v: &v, seq_len: 6, head_dim: d };
    let out = LambekAttn::default_init(d).forward(&input);
    assert_eq!(out.out.len(), 6 * d);
    assert!(out.out.iter().all(|x| x.is_finite()));
}
