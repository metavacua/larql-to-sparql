// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
pub mod hyperbolic_attn;
pub mod lambek;
pub mod linear_complex;
pub mod reference;
pub mod walk;

/// Q, K, V are row-major f32 slices of shape [seq_len, head_dim].
pub struct AttnInput<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub seq_len: usize,
    pub head_dim: usize,
}

pub struct AttnOutput {
    pub out: Vec<f32>,
    pub seq_len: usize,
    pub head_dim: usize,
}

pub trait AttnBackend {
    fn name(&self) -> &str;
    fn forward(&self, input: &AttnInput) -> AttnOutput;
}
