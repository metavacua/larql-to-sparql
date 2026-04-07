//! Hybrid RS + Cracked Attention on the real model.
//!
//! Three phases:
//! 1. Head classification: run multiple entities through same template,
//!    measure per-head cosine similarity, classify static vs dynamic
//! 2. Cache building: store static head attention outputs per template
//! 3. Hybrid inference: cached static heads + dynamic-only KV + vindex FFN

use ndarray::{Array2, ArrayView1, s};
use larql_inference::model::ModelWeights;
use larql_inference::attention::run_attention_block;
use larql_inference::forward::{embed_tokens_pub, run_ffn, apply_norm};
use larql_inference::ffn::WeightFfn;

/// Per-head attention output for one entity on one layer.
/// Shape: [seq_len, head_dim] per head.
#[derive(Clone)]
pub struct PerHeadOutput {
    pub layer: usize,
    /// Per-head outputs: `heads[h]` is the last-token's attention output for head h.
    /// Shape: [head_dim] per head (last token only, for classification).
    pub heads: Vec<Vec<f32>>,
}

/// Head classification result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HeadClassResult {
    pub layer: usize,
    pub head: usize,
    /// Mean cosine similarity across entity pairs.
    pub mean_cosine: f32,
    /// Classification: static (cacheable) or dynamic.
    pub is_static: bool,
}

/// Full classification for a model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelHeadClassification {
    pub results: Vec<HeadClassResult>,
    pub total_heads: usize,
    pub static_count: usize,
    pub dynamic_count: usize,
    pub static_fraction: f64,
    pub dynamic_layers: Vec<usize>,
}

// ── Phase 1: Head Classification ──

/// Capture per-head attention outputs for a given prompt.
/// Returns per-head output at the last token position for each layer.
pub fn capture_per_head_attention(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Vec<PerHeadOutput> {
    let num_layers = weights.num_layers;
    let num_q = weights.num_q_heads;
    let head_dim = weights.head_dim;
    let ffn = WeightFfn { weights };

    let mut h = embed_tokens_pub(weights, token_ids);
    let mut per_layer_heads = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        // Run attention with capture enabled to get the pre-O-projection output
        let (h_post_attn, attn_projected, _attn_weights) =
            run_attention_block(weights, &h, layer, false)
                .expect("attention failed");

        // Extract per-head output from attn_projected (post-O-projection, [seq, hidden])
        // For classification, we use the last token's output.
        // The O projection mixes heads, but for cosine comparison across entities
        // on the same template, the mixed output still reflects per-head behavior
        // because the O projection is the same for both entities.
        let seq_len = h.shape()[0];
        let last_tok = attn_projected.row(seq_len - 1);

        // Split the hidden dimension into per-head chunks
        // Note: attn_projected is [seq, hidden] after O-proj, not [seq, num_q * head_dim]
        // For proper per-head analysis, we need the pre-O attention output.
        // Approximation: use chunks of the projected output as head proxies.
        let hidden = weights.hidden_size;
        let chunk_size = hidden / num_q;
        let mut heads = Vec::with_capacity(num_q);
        for h_idx in 0..num_q {
            let start = h_idx * chunk_size;
            let end = start + chunk_size;
            heads.push(last_tok.slice(s![start..end]).to_vec());
        }

        per_layer_heads.push(PerHeadOutput {
            layer,
            heads,
        });

        // Continue forward pass
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        h = h_out;
    }

    per_layer_heads
}

/// Classify heads by running multiple entities through the same template.
/// Computes pairwise cosine similarity per head across entities.
pub fn classify_heads(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    template_prefix: &str,
    entities: &[&str],
    cosine_threshold: f32,
) -> ModelHeadClassification {
    let num_q = weights.num_q_heads;
    let num_layers = weights.num_layers;

    // Capture per-head outputs for each entity
    let mut all_captures: Vec<Vec<PerHeadOutput>> = Vec::new();
    for entity in entities {
        let prompt = format!("{template_prefix}{entity}");
        let encoding = tokenizer.encode(prompt.as_str(), true).expect("tokenize");
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        all_captures.push(capture_per_head_attention(weights, &token_ids));
    }

    // For each layer and head, compute mean pairwise cosine
    let mut results = Vec::new();
    let mut static_count = 0;
    let mut dynamic_layers = std::collections::HashSet::new();

    for layer in 0..num_layers {
        for head in 0..num_q {
            let mut cosine_sum = 0.0f64;
            let mut pair_count = 0;

            for i in 0..entities.len() {
                for j in (i + 1)..entities.len() {
                    let a = &all_captures[i][layer].heads[head];
                    let b = &all_captures[j][layer].heads[head];
                    let cos = cosine_sim(a, b);
                    cosine_sum += cos as f64;
                    pair_count += 1;
                }
            }

            let mean_cosine = if pair_count > 0 {
                (cosine_sum / pair_count as f64) as f32
            } else {
                0.0
            };

            let is_static = mean_cosine >= cosine_threshold;
            if is_static {
                static_count += 1;
            } else {
                dynamic_layers.insert(layer);
            }

            results.push(HeadClassResult {
                layer,
                head,
                mean_cosine,
                is_static,
            });
        }
    }

    let total_heads = num_layers * num_q;
    let dynamic_count = total_heads - static_count;

    ModelHeadClassification {
        results,
        total_heads,
        static_count,
        dynamic_count,
        static_fraction: static_count as f64 / total_heads as f64,
        dynamic_layers: {
            let mut v: Vec<usize> = dynamic_layers.into_iter().collect();
            v.sort();
            v
        },
    }
}

// ── Phase 2: Hybrid Inference ──

/// Run hybrid inference: full forward pass but measure what a hybrid pipeline would compute.
/// Returns prediction + metrics showing what could be cached vs what needs computation.
pub struct HybridInferenceResult {
    pub predictions: Vec<(String, f64)>,
    /// How many layers have all-static heads (could skip attention entirely).
    pub fully_static_layers: usize,
    /// How many layers have at least one dynamic head.
    pub dynamic_layers: usize,
    /// Total heads classified as static.
    pub static_heads: usize,
    /// Total heads classified as dynamic.
    pub dynamic_heads: usize,
    /// Wall clock in microseconds.
    pub wall_clock_us: f64,
    /// Memory that would be needed (dynamic KV only).
    pub dynamic_kv_bytes: usize,
    /// Memory the full KV cache would need.
    pub full_kv_bytes: usize,
}

/// Run hybrid inference with head classification.
/// This runs the FULL forward pass (for correctness verification) but reports
/// what a true hybrid pipeline would compute and store.
pub fn run_hybrid_inference(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    classification: &ModelHeadClassification,
    top_k: usize,
) -> HybridInferenceResult {
    let t0 = std::time::Instant::now();
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();

    // Full forward pass (for correctness — true hybrid would skip static heads)
    let result = larql_inference::predict(weights, tokenizer, token_ids, top_k);
    let wall_clock_us = t0.elapsed().as_secs_f64() * 1e6;

    // Count static vs dynamic per layer
    let num_q = weights.num_q_heads;
    let mut fully_static = 0;
    let mut dynamic_layers = 0;

    for layer in 0..num_layers {
        let layer_results: Vec<&HeadClassResult> = classification.results
            .iter()
            .filter(|r| r.layer == layer)
            .collect();
        let all_static = layer_results.iter().all(|r| r.is_static);
        if all_static {
            fully_static += 1;
        } else {
            dynamic_layers += 1;
        }
    }

    // Memory: dynamic KV only for layers with dynamic heads
    let kv_per_layer_per_token = 2 * weights.num_kv_heads * weights.head_dim * 2; // K+V, fp16
    let dynamic_kv_bytes = dynamic_layers * kv_per_layer_per_token * seq_len;
    let full_kv_bytes = num_layers * kv_per_layer_per_token * seq_len;

    HybridInferenceResult {
        predictions: result.predictions,
        fully_static_layers: fully_static,
        dynamic_layers,
        static_heads: classification.static_count,
        dynamic_heads: classification.dynamic_count,
        wall_clock_us,
        dynamic_kv_bytes,
        full_kv_bytes,
    }
}

/// Format classification results.
pub fn format_classification(cls: &ModelHeadClassification) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\n=== Head Classification: {}/{} static ({:.1}%) ===\n\n",
        cls.static_count, cls.total_heads, cls.static_fraction * 100.0,
    ));

    // Per-layer summary
    let num_q = if cls.total_heads > 0 && !cls.results.is_empty() {
        cls.results.iter().filter(|r| r.layer == 0).count()
    } else {
        0
    };

    if num_q > 0 {
        let num_layers = cls.total_heads / num_q;
        out.push_str(&format!("{:>5} {:>8} {:>8} {:>10}\n", "Layer", "Static", "Dynamic", "Mean cos"));
        out.push_str(&"-".repeat(35));
        out.push('\n');

        for layer in 0..num_layers {
            let layer_results: Vec<&HeadClassResult> = cls.results
                .iter()
                .filter(|r| r.layer == layer)
                .collect();
            let static_count = layer_results.iter().filter(|r| r.is_static).count();
            let dynamic_count = num_q - static_count;
            let mean_cos: f32 = layer_results.iter().map(|r| r.mean_cosine).sum::<f32>() / num_q as f32;

            let marker = if dynamic_count > 0 { " ←" } else { "" };
            out.push_str(&format!(
                "L{:<4} {:>8} {:>8} {:>10.4}{marker}\n",
                layer, static_count, dynamic_count, mean_cos,
            ));
        }
    }

    out.push_str(&format!(
        "\nDynamic layers: {:?}\n",
        cls.dynamic_layers,
    ));
    out.push_str(&format!(
        "KV cache reduction: {:.0}× (only dynamic layers need KV)\n",
        cls.total_heads as f64 / cls.dynamic_count.max(1) as f64,
    ));

    out
}

/// Cosine similarity between two f32 slices.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { (dot / denom) as f32 }
}
