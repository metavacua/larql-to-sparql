// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use super::{AttnInput, AttnOutput};

/// Standard softmax attention O(n²) — reference only, not for production use.
pub fn standard_attn(inp: &AttnInput) -> AttnOutput {
    let n = inp.seq_len;
    let d = inp.head_dim;
    let scale = (d as f32).sqrt().recip();
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        let q_row = &inp.q[i*d..(i+1)*d];
        let scores: Vec<f32> = (0..n).map(|j| {
            let k_row = &inp.k[j*d..(j+1)*d];
            q_row.iter().zip(k_row.iter()).map(|(a,b)| a*b).sum::<f32>() * scale
        }).collect();
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_e: f32 = exps.iter().sum::<f32>().max(1e-8);
        let weights: Vec<f32> = exps.iter().map(|e| e / sum_e).collect();
        for j in 0..n {
            let v_row = &inp.v[j*d..(j+1)*d];
            for k in 0..d { out[i*d + k] += weights[j] * v_row[k]; }
        }
    }
    AttnOutput { out, seq_len: n, head_dim: d }
}
