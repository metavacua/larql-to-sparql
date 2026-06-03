// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use crate::attention::{AttnBackend, AttnInput, AttnOutput};

pub struct WalkAttn {
    top_k: usize,
    route_dim: usize,
}

impl WalkAttn {
    pub fn new(top_k: usize, route_dim: usize) -> Self { Self { top_k, route_dim } }
}

impl AttnBackend for WalkAttn {
    fn name(&self) -> &str { "walk" }

    fn forward(&self, inp: &AttnInput) -> AttnOutput {
        debug_assert_eq!(inp.q.len(), inp.seq_len * inp.head_dim);
        debug_assert_eq!(inp.k.len(), inp.seq_len * inp.head_dim);
        debug_assert_eq!(inp.v.len(), inp.seq_len * inp.head_dim);
        let n = inp.seq_len;
        let d = inp.head_dim;
        let k = self.top_k.min(n);
        let r = self.route_dim.min(d);
        let scale = (d as f32).sqrt().recip();

        // Cheap routing: score all keys using first r dims only
        let q_routes: Vec<&[f32]> = (0..n).map(|i| &inp.q[i*d..i*d+r]).collect();
        let k_routes: Vec<&[f32]> = (0..n).map(|i| &inp.k[i*d..i*d+r]).collect();

        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            // Find top-K keys by routing score
            let mut route_scores: Vec<(usize, f32)> = (0..n).map(|j| {
                let s: f32 = q_routes[i].iter().zip(k_routes[j].iter()).map(|(a,b)| a*b).sum();
                (j, s)
            }).collect();
            route_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_indices: Vec<usize> = route_scores[..k].iter().map(|(idx,_)| *idx).collect();

            // Full-dim attention over top-K keys
            let full_scores: Vec<f32> = top_indices.iter().map(|&j| {
                let q_row = &inp.q[i*d..(i+1)*d];
                let k_row = &inp.k[j*d..(j+1)*d];
                q_row.iter().zip(k_row.iter()).map(|(a,b)| a*b).sum::<f32>() * scale
            }).collect();

            let max_s = full_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = full_scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_e: f32 = exps.iter().sum::<f32>().max(1e-8);

            for (idx_k, &j) in top_indices.iter().enumerate() {
                let w = exps[idx_k] / sum_e;
                let v_row = &inp.v[j*d..(j+1)*d];
                for dim in 0..d { out[i*d + dim] += w * v_row[dim]; }
            }
        }
        AttnOutput { out, seq_len: n, head_dim: d }
    }
}
