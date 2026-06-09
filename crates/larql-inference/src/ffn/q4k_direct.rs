//! `Q4kDirectFfn` — bypasses Q4_K → f32 dequantisation in the FFN forward.
//!
//! `WeightFfn` reads dequantised f32 tensors out of `weights.tensors`. That's
//! correct but expensive on CPU: every decode-step would materialise ~10 GB of
//! f32 FFN weights for Gemma 3 4B (PR #138 fixed the per-step retomputation
//! by caching the f32 across steps — but you still need to *read* the 10 GB
//! through memory for every GEMV). llama.cpp's CPU path avoids this by doing
//! Q4_K × Q8_K matvec directly on the quantised bytes, which is what closes
//! the remaining ~19× gap to its 14 tok/s on Gemma 3 4B.
//!
//! The direct-matvec compute kernels for FFN gate/up/down already exist as
//! [`crate::vindex::q4k_ffn_forward_layer_q8k`] (built on the AVX2 kernels in
//! `larql_compute::cpu::ops::q4k_q8k_dot`). This file is the thin
//! [`FfnBackend`] adapter that lets `run_ffn` dispatch through them instead
//! of `dense_ffn_forward`.
//!
//! Decode-step semantics — input must be a `(1, hidden)` matrix per call.
//! Multi-row (prefill) inputs are handled by looping per row.

use larql_compute::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
use larql_models::ModelArchitecture;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::FfnBackend;
use crate::vindex::q4k_ffn_forward_layer_q8k;

/// Q4_K direct-matvec FFN backend.
///
/// Holds borrowed references to the vindex (for raw Q4_K bytes) and the
/// architecture (for activation kind). Has no reference to `ModelWeights`,
/// so it composes with the `&mut ModelWeights` flow in
/// `predict_q4k_hidden_with_cache` without borrow conflicts.
pub struct Q4kDirectFfn<'a> {
    pub arch: &'a dyn ModelArchitecture,
    pub index: &'a VectorIndex,
}

impl<'a> FfnBackend for Q4kDirectFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let seq = x.shape()[0];
        let hidden = x.shape()[1];
        if seq == 1 {
            let row = x.row(0);
            let x_flat: Vec<f32> = row.iter().copied().collect();
            let h_q8k = quantize_x_to_q8k(&x_flat);
            return q4k_ffn_forward_layer_q8k(self.arch, self.index, layer, &h_q8k);
        }
        // Prefill (multi-row) — loop per row. Each Q8K quantisation + matvec is
        // O(intermediate × hidden) memory-bandwidth-bound, so per-row dispatch
        // amortises well at typical prompt lengths.
        let mut out = Array2::<f32>::zeros((seq, hidden));
        for r in 0..seq {
            let row = x.row(r);
            let x_flat: Vec<f32> = row.iter().copied().collect();
            let h_q8k = quantize_x_to_q8k(&x_flat);
            let row_out = q4k_ffn_forward_layer_q8k(self.arch, self.index, layer, &h_q8k);
            out.row_mut(r).assign(&row_out.row(0));
        }
        out
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        // The decode hot path (`run_ffn` with `capture_activation = false`)
        // never calls this. Diagnostic / intervention callers that need the
        // pre-down activation should use `WeightFfn` for now. Return a zero
        // placeholder so the trait surface stays uniform without exposing
        // the wrong tensor.
        let out = self.forward(layer, x);
        let intermediate = self.index.num_features(layer);
        let seq = x.shape()[0];
        (out, Array2::<f32>::zeros((seq, intermediate)))
    }

    fn name(&self) -> &str {
        "q4k-direct"
    }
}
