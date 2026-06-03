// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use crate::attention::{AttnBackend, AttnInput, AttnOutput};
use crate::complex::{C32, phi};

pub struct ComplexLinearAttn;

impl ComplexLinearAttn {
    pub fn new() -> Self { Self }
}

impl Default for ComplexLinearAttn {
    fn default() -> Self { Self::new() }
}

impl AttnBackend for ComplexLinearAttn {
    fn name(&self) -> &str { "complex-linear" }

    fn forward(&self, inp: &AttnInput) -> AttnOutput {
        let n = inp.seq_len;
        let d = inp.head_dim;

        // Pass 1: KtV[i,j] = Σ_t φ(K_t[i])* · V_t[j]  (shape d×d complex)
        let mut ktv = vec![C32::new(0.0, 0.0); d * d];
        for t in 0..n {
            let k_row = &inp.k[t*d..(t+1)*d];
            let v_row = &inp.v[t*d..(t+1)*d];
            let phi_k_conj: Vec<C32> = phi(k_row).into_iter().map(|c| c.conj()).collect();
            for i in 0..d {
                for j in 0..d {
                    ktv[i*d + j] = ktv[i*d + j] + C32::new(
                        phi_k_conj[i].re * v_row[j],
                        phi_k_conj[i].im * v_row[j],
                    );
                }
            }
        }

        // Pass 2: out[t,j] = Re[ Σ_i φ(Q_t[i]) · KtV[i,j] ] / n
        let mut out = vec![0.0f32; n * d];
        let scale = 1.0 / n as f32;
        for t in 0..n {
            let q_row = &inp.q[t*d..(t+1)*d];
            let phi_q = phi(q_row);
            for j in 0..d {
                let mut sum = C32::new(0.0, 0.0);
                for i in 0..d {
                    sum = sum + phi_q[i] * ktv[i*d + j];
                }
                out[t*d + j] = sum.re * scale;
            }
        }

        AttnOutput { out, seq_len: n, head_dim: d }
    }
}
