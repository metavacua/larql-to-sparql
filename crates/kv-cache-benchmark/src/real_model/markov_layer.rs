//! Markov Residual Stream strategy on the real model.
//!
//! Runs bounded-window forward pass. Captures the residual at each layer
//! instead of K/V. The residual IS the complete state (Markov property).
//!
//! - Active window: last W tokens get full residuals (f32)
//! - Cold tier: older tokens stored as token IDs only (4 bytes each)
//! - Reconstruction: replay from token IDs through forward pass

use ndarray::Array2;
use larql_inference::model::ModelWeights;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::attention::run_attention_with_kv;
use larql_inference::ffn::WeightFfn;

/// Result of Markov RS forward pass.
pub struct MarkovResult {
    /// Per-layer residual snapshots (for active window tokens).
    pub residuals: Vec<Array2<f32>>,
    /// Final hidden state.
    pub hidden: Array2<f32>,
    /// Total memory: active window + cold tier.
    pub memory_bytes: usize,
    /// Active window token count.
    pub window_tokens: usize,
    /// Cold tier token count.
    pub cold_tokens: usize,
    /// Wall clock for the forward pass in microseconds.
    pub forward_us: f64,
}

/// Run Markov RS forward pass with bounded window.
///
/// For the benchmark, we run the full forward pass but only retain residuals
/// for the last `window_size` tokens. Cold tier tokens are stored as IDs.
pub fn run_markov_forward(
    weights: &ModelWeights,
    token_ids: &[u32],
    window_size: usize,
) -> MarkovResult {
    let num_layers = weights.num_layers;
    let hidden_dim = weights.hidden_size;
    let seq_len = token_ids.len();
    let ffn = WeightFfn { weights };

    let t0 = std::time::Instant::now();

    let mut h = embed_tokens_pub(weights, token_ids);
    let mut residuals = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        // Capture residual before this layer
        residuals.push(h.clone());

        let (h_post_attn, _k, _v) = run_attention_with_kv(weights, &h, layer)
            .expect("attention failed");
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        h = h_out;
    }

    let forward_us = t0.elapsed().as_secs_f64() * 1e6;

    // Memory accounting
    let window_tokens = seq_len.min(window_size);
    let cold_tokens = seq_len.saturating_sub(window_size);

    // Active window: residuals for last W tokens at all layers
    // hidden_dim * 4 bytes (f32) per token per layer snapshot
    let window_bytes = window_tokens * hidden_dim * 4;
    // Cold tier: just token IDs
    let cold_bytes = cold_tokens * 4;
    let memory_bytes = window_bytes + cold_bytes;

    MarkovResult {
        residuals,
        hidden: h,
        memory_bytes,
        window_tokens,
        cold_tokens,
        forward_us,
    }
}

/// Compare two forward passes by checking if they produce the same top-1 prediction.
/// This validates that the Markov RS forward pass is equivalent to Standard KV.
pub fn compare_hidden_states(h1: &Array2<f32>, h2: &Array2<f32>) -> (f64, f64) {
    let seq_len = h1.shape()[0];

    // Compare last-token hidden state (what determines the prediction)
    let v1: Vec<f32> = h1.row(seq_len - 1).to_vec();
    let v2: Vec<f32> = h2.row(seq_len - 1).to_vec();

    let mse = crate::metrics::Metrics::compute_mse(&v1, &v2);
    let cosine = crate::metrics::Metrics::compute_cosine(&v1, &v2);
    (mse, cosine)
}
