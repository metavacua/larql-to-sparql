// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use crate::attention::{AttnBackend, AttnInput, AttnOutput};

const NUM_BUCKETS: usize = 5;

/// Non-commutative attention: directional scale factors break the exchange rule.
/// d_scales[direction][bucket][dim]: direction 0=past(j<i), 1=future(j>i), 2=self
pub struct LambekAttn {
    d_scales: Vec<Vec<Vec<f32>>>,  // [3][NUM_BUCKETS][head_dim]
}

fn dist_bucket(delta: usize) -> usize {
    if delta == 0 { 0 } else { ((delta as f32).log2() as usize + 1).min(NUM_BUCKETS - 1) }
}

impl LambekAttn {
    /// Non-trivial init: forward (past) scales differ from backward (future) scales.
    pub fn default_init(head_dim: usize) -> Self {
        let mut d_scales = vec![vec![vec![1.0f32; head_dim]; NUM_BUCKETS]; 3];
        for bucket in 0..NUM_BUCKETS {
            let decay = 1.0 / (1.0 + bucket as f32 * 0.3);
            for dim in 0..head_dim {
                // Past (dir=0): amplified — attends more strongly to recent context
                d_scales[0][bucket][dim] = 1.2 * decay * (1.0 + (dim as f32 * 0.1).sin() * 0.2);
                // Future (dir=1): dampened — attends less to future context
                d_scales[1][bucket][dim] = 0.8 * decay * (1.0 + (dim as f32 * 0.1).cos() * 0.1);
                // Self (dir=2): neutral
                d_scales[2][bucket][dim] = 1.0;
            }
        }
        Self { d_scales }
    }
}

impl AttnBackend for LambekAttn {
    fn name(&self) -> &str { "lambek-directional" }

    fn forward(&self, inp: &AttnInput) -> AttnOutput {
        debug_assert_eq!(inp.q.len(), inp.seq_len * inp.head_dim);
        debug_assert_eq!(inp.k.len(), inp.seq_len * inp.head_dim);
        debug_assert_eq!(inp.v.len(), inp.seq_len * inp.head_dim);
        let n = inp.seq_len;
        let d = inp.head_dim;
        let scale = (d as f32).sqrt().recip();
        let mut out = vec![0.0f32; n * d];

        for i in 0..n {
            let q_row = &inp.q[i*d..(i+1)*d];
            let mut weights = vec![0.0f32; n];

            for j in 0..n {
                let k_row = &inp.k[j*d..(j+1)*d];
                let dir = if j < i { 0 } else if j > i { 1 } else { 2 };
                let bucket = dist_bucket(i.abs_diff(j));
                let dir_scale = &self.d_scales[dir][bucket];

                // Directional dot: Q_i · (K_j ⊙ D[dir][bucket]) / √d
                weights[j] = q_row.iter()
                    .zip(k_row.iter())
                    .zip(dir_scale.iter())
                    .map(|((q, k), s)| q * k * s)
                    .sum::<f32>() * scale;
            }

            let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = weights.iter().map(|w| (w - max_w).exp()).collect();
            let sum_e: f32 = exps.iter().sum::<f32>().max(1e-8);

            for j in 0..n {
                let w = exps[j] / sum_e;
                let v_row = &inp.v[j*d..(j+1)*d];
                for k in 0..d { out[i*d + k] += w * v_row[k]; }
            }
        }
        AttnOutput { out, seq_len: n, head_dim: d }
    }
}
