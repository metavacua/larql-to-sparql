// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
use wasm_bindgen::prelude::*;
use crate::attention::{AttnBackend, AttnInput};
use crate::attention::hyperbolic_attn::HyperbolicAttn;
use crate::attention::lambek::LambekAttn;
use crate::attention::linear_complex::ComplexLinearAttn;
use crate::attention::walk::WalkAttn;

#[wasm_bindgen]
pub struct GeometricAttn {
    backend: String,
    seq_len: usize,
    head_dim: usize,
}

#[wasm_bindgen]
impl GeometricAttn {
    #[wasm_bindgen(constructor)]
    pub fn new(backend: &str, seq_len: usize, head_dim: usize) -> Self {
        Self { backend: backend.to_string(), seq_len, head_dim }
    }

    /// Run the selected backend. q/k/v are Float32Arrays of length seq_len*head_dim.
    /// Returns a new Vec<f32> (becomes Float32Array on the JS side).
    pub fn run(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let input = AttnInput { q, k, v, seq_len: self.seq_len, head_dim: self.head_dim };
        match self.backend.as_str() {
            "complex-linear" => ComplexLinearAttn::new().forward(&input).out,
            "hyperbolic"     => HyperbolicAttn::new(1.0).forward(&input).out,
            "walk-k4"        => WalkAttn::new(4, self.head_dim).forward(&input).out,
            "lambek"         => LambekAttn::default_init(self.head_dim).forward(&input).out,
            _                => ComplexLinearAttn::new().forward(&input).out,
        }
    }
}

#[wasm_bindgen(start)]
pub fn on_start() {
    #[cfg(feature = "browser")]
    console_error_panic_hook::set_once();
}
