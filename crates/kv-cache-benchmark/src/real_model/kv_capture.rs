//! K/V capture from the real attention forward pass.
//!
//! Runs `run_attention_with_kv()` per layer and collects the post-RoPE K and V
//! tensors. These are the ground-truth vectors that TurboQuant compresses.

use ndarray::Array2;
use larql_inference::model::ModelWeights;
use larql_inference::attention::run_attention_with_kv;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::ffn::WeightFfn;

/// Captured K/V tensors from a full forward pass.
pub struct KvCapture {
    /// Per-layer K tensors (post-RoPE). Shape: [seq_len, num_kv_heads * head_dim]
    pub keys: Vec<Array2<f32>>,
    /// Per-layer V tensors. Shape: [seq_len, num_kv_heads * head_dim]
    pub values: Vec<Array2<f32>>,
    /// Final hidden state after all layers.
    pub hidden: Array2<f32>,
    /// Number of layers captured.
    pub num_layers: usize,
}

/// Run a full forward pass capturing K/V at every layer.
/// This is the Standard KV baseline — what llama.cpp, vLLM, MLX all do.
pub fn capture_kv(weights: &ModelWeights, token_ids: &[u32]) -> KvCapture {
    let num_layers = weights.num_layers;
    let ffn = WeightFfn { weights };

    let mut h = embed_tokens_pub(weights, token_ids);
    let mut keys = Vec::with_capacity(num_layers);
    let mut values = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let (h_post_attn, k_rope, v) = run_attention_with_kv(weights, &h, layer)
            .expect("attention failed");

        keys.push(k_rope);
        values.push(v);

        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        h = h_out;
    }

    KvCapture {
        keys,
        values,
        hidden: h,
        num_layers,
    }
}

/// Memory used by captured KV in bytes (FP16 simulation).
pub fn kv_memory_bytes(capture: &KvCapture) -> usize {
    let mut total = 0;
    for k in &capture.keys {
        // FP16: 2 bytes per element
        total += k.len() * 2;
    }
    for v in &capture.values {
        total += v.len() * 2;
    }
    total
}
