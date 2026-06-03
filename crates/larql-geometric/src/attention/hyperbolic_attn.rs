// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use crate::attention::{AttnBackend, AttnInput, AttnOutput};
use crate::hyperbolic::{exp_map_zero, geodesic_dist};

pub struct HyperbolicAttn {
    temperature: f32,
}

impl HyperbolicAttn {
    pub fn new(temperature: f32) -> Self { Self { temperature } }
}

impl AttnBackend for HyperbolicAttn {
    fn name(&self) -> &str { "hyperbolic" }

    fn forward(&self, inp: &AttnInput) -> AttnOutput {
        let n = inp.seq_len;
        let d = inp.head_dim;
        let tau = self.temperature;

        let q_hyp: Vec<Vec<f32>> = (0..n).map(|i| exp_map_zero(&inp.q[i*d..(i+1)*d])).collect();
        let k_hyp: Vec<Vec<f32>> = (0..n).map(|i| exp_map_zero(&inp.k[i*d..(i+1)*d])).collect();

        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            let weights: Vec<f32> = (0..n).map(|j| {
                let dist = geodesic_dist(&q_hyp[i], &k_hyp[j]);
                (-dist * dist / tau).exp()
            }).collect();

            let w_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let w_exp: Vec<f32> = weights.iter().map(|w| (w - w_max).exp()).collect();
            let w_sum: f32 = w_exp.iter().sum::<f32>().max(1e-8);
            let w_norm: Vec<f32> = w_exp.iter().map(|w| w / w_sum).collect();

            for j in 0..n {
                let v_row = &inp.v[j*d..(j+1)*d];
                for k in 0..d { out[i*d + k] += w_norm[j] * v_row[k]; }
            }
        }
        AttnOutput { out, seq_len: n, head_dim: d }
    }
}
