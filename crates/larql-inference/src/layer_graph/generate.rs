//! Token generation loop — GPU prefill + KV-cached decode

use larql_compute::ComputeBackend;
use crate::model::ModelWeights;
use super::CachedLayerGraph;

/// Top-K logits lookup that transparently handles models with tied
/// input/output embeddings (Gemma 2/3/4) whose vindex has no dedicated
/// `lm_head.bin` / `lm_head_q4.bin`.
///
/// Resolution order:
/// 1. Vindex-native KNN (`lm_head_knn_backend`) — fastest, used when the
///    vindex was built with a separate lm_head.
/// 2. CPU gemv against `weights.lm_head` — the loader fills this from
///    `embed.clone()` for tied-embedding models, so it's always populated
///    even when no lm_head file is present.
///
/// The second path is O(vocab * hidden) floats through the CPU, but that's
/// a one-shot matvec per generated token — negligible compared to the
/// per-layer attention + FFN. It lets every model generate tokens through
/// the Metal pipeline regardless of how its vindex was packaged.
fn lm_head_topk(
    index: &larql_vindex::VectorIndex,
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
    backend: &dyn ComputeBackend,
) -> Vec<(u32, f32)> {
    let hits = index.lm_head_knn_backend(query, top_k, backend);
    if !hits.is_empty() {
        return hits;
    }
    cpu_lm_head_topk(weights, query, top_k)
}

fn cpu_lm_head_topk(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
) -> Vec<(u32, f32)> {
    let lm = &weights.lm_head;
    if lm.is_empty() || query.is_empty() {
        return Vec::new();
    }
    let vocab = lm.shape()[0];
    let hidden = lm.shape()[1];
    if hidden != query.len() {
        return Vec::new();
    }
    // Single matvec: lm @ query → [vocab]. Use ndarray's BLAS-backed dot.
    let scores = lm.dot(query);
    let mut indexed: Vec<(u32, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(i, s)| (i as u32, s))
        .collect();
    let k = top_k.min(indexed.len());
    if k > 0 && k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
    }
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Drop any NaN scores from the tail — they'd otherwise sort unpredictably.
    indexed.retain(|(_, s)| s.is_finite());
    let _ = vocab;
    indexed
}

/// Multi-token generation: GPU prefill → decode loop with KV cache.
///
/// 1. GPU prefill: full_pipeline_q4 populates KV cache for all layers
/// 2. Decode loop: decode_token reads from KV cache, generates one token at a time
/// 3. Logits: vindex lm_head KNN (no dense matmul)
///
/// Returns: Vec of (token_string, probability) for each generated token,
/// plus timing (prefill_ms, per_token_ms).
#[allow(clippy::too_many_arguments)]
pub fn generate(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> GenerateResult {
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let gate_index: &dyn larql_vindex::GateIndex = index;

    // Build layer descriptors
    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
        (Some(mmap), true)
    } else {
        (gate_index.interleaved_q4_mmap_ref(), false)
    };
    let has_q4k = index.attn_q4k_layer_data(layer_range.start).is_some();
    let has_q8 = index.attn_q8_layer_data(layer_range.start).is_some();

    if !backend.has_q4() || q4_ffn.is_none() {
        let r = super::predict::predict_honest(weights, tokenizer, token_ids, 5, index, backend, cached_layers, layer_range);
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
        };
    }

    let q4_ffn_mmap = q4_ffn.unwrap();
    let intermediate = gate_index.num_features(layer_range.start);
    if intermediate == 0 || (!has_q4k && !has_q8) {
        let r = super::predict::predict_honest(weights, tokenizer, token_ids, 5, index, backend, cached_layers, layer_range);
        return GenerateResult {
            tokens: r.predictions.into_iter().take(1).collect(),
            prefill_ms: 0.0,
            decode_ms: vec![],
        };
    }

    let q4_ffn_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 148
    } else {
        intermediate * hidden / 32 * 18
    };

    let ffn_format = if ffn_is_q4k { larql_compute::QuantFormat::Q4_K } else { larql_compute::QuantFormat::Q4_0 };

    let num_layers = weights.num_layers;
    let layers = super::pipeline_layer::build_pipeline_layers(
        weights, index, 0..num_layers,
        q4_ffn_mmap, q4_ffn_per_matrix, ffn_format,
    );

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(layer_range.start) as f32;

    // ── Phase 1: GPU prefill ──
    let prefill_start = std::time::Instant::now();
    backend.reset_kv_cache();
    let seq_len = token_ids.len();

    let h_embed = crate::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();

    let softcap_val = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm_val = arch.attn_q_norm_key(0).is_some();

    let h_vec = backend.prefill_q4(
        &layers, &x, hidden, intermediate, q_dim, kv_dim,
        seq_len, weights.num_q_heads, weights.num_kv_heads, weights.head_dim,
        rope, qk_norm_val, softcap_val,
    ).unwrap_or_else(|| {
        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        let mut h = h_embed.clone();
        for layer in 0..num_layers {
            let (h_post_attn, _, _) =
                crate::attention::run_attention_block_gpu(weights, &h, layer, false, None).unwrap();
            let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h = h_out;
        }
        h.as_slice().unwrap_or(&[]).to_vec()
    });

    let h = ndarray::Array2::from_shape_vec((seq_len, hidden), h_vec).unwrap_or(h_embed);

    let h_1d = {
        let h_final = crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);
        h_final.row(seq_len - 1).to_owned()
    };
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // Sample first token
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut decode_ms = Vec::with_capacity(max_tokens);

    let first_hits = lm_head_topk(index, weights, &h_1d, 5, backend);
    if let Some(&(tid, score)) = first_hits.first() {
        let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default().trim().to_string();
        let prob = super::logits::softmax_prob(score, &first_hits, weights.arch.logits_scaling(), weights.arch.final_logit_softcapping());
        tokens.push((tok_str, prob));
    }

    // ── Phase 2: GPU decode loop ──
    let mut current_token_id = first_hits.first().map(|&(tid, _)| tid).unwrap_or(0);
    let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);

    for _step in 1..max_tokens {
        let decode_start = std::time::Instant::now();

        let h_tok = crate::forward::embed_tokens_pub(weights, &[current_token_id]);
        let x_dec: Vec<f32> = h_tok.row(0).to_vec();
        let result = backend.decode_token(
            &layers, &x_dec, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
        );

        if let Some(h_out) = result {
            let h_arr = ndarray::Array2::from_shape_vec((1, hidden), h_out).unwrap();
            let h_final = crate::forward::apply_norm(weights, &h_arr, weights.arch.final_norm_key(), norm_offset);
            let h_1d = h_final.row(0).to_owned();

            let hits = lm_head_topk(index, weights, &h_1d, 5, backend);
            let step_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            decode_ms.push(step_ms);

            if let Some(&(tid, score)) = hits.first() {
                let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default().trim().to_string();
                let prob = super::logits::softmax_prob(score, &hits, weights.arch.logits_scaling(), weights.arch.final_logit_softcapping());
                let is_eos = tok_str == "<eos>" || tok_str == "</s>" || tok_str == "<|endoftext|>";
                tokens.push((tok_str, prob));
                current_token_id = tid;
                if is_eos { break; }
            } else { break; }
        } else {
            // GPU failed — CPU fallback
            let mut h_dec = h_tok;
            for layer in 0..num_layers {
                let (h_post_attn, _, _) =
                    crate::attention::run_attention_block_gpu(weights, &h_dec, layer, false, None).unwrap();
                let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
                h_dec = h_out;
            }
            let h_final = crate::forward::apply_norm(weights, &h_dec, weights.arch.final_norm_key(), norm_offset);
            let h_1d = h_final.row(0).to_owned();
            let hits = lm_head_topk(index, weights, &h_1d, 5, backend);
            let step_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            decode_ms.push(step_ms);
            if let Some(&(tid, score)) = hits.first() {
                let tok_str = tokenizer.decode(&[tid], true).unwrap_or_default().trim().to_string();
                let prob = super::logits::softmax_prob(score, &hits, weights.arch.logits_scaling(), weights.arch.final_logit_softcapping());
                let is_eos = tok_str == "<eos>" || tok_str == "</s>" || tok_str == "<|endoftext|>";
                tokens.push((tok_str, prob));
                current_token_id = tid;
                if is_eos { break; }
            } else { break; }
        }
    }

    GenerateResult { tokens, prefill_ms, decode_ms }
}

/// Result of multi-token generation.
pub struct GenerateResult {
    pub tokens: Vec<(String, f64)>,
    pub prefill_ms: f64,
    pub decode_ms: Vec<f64>,
}

impl GenerateResult {
    pub fn avg_decode_ms(&self) -> f64 {
        if self.decode_ms.is_empty() { 0.0 }
        else { self.decode_ms.iter().sum::<f64>() / self.decode_ms.len() as f64 }
    }

    pub fn decode_tok_s(&self) -> f64 {
        let avg = self.avg_decode_ms();
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    }

    pub fn text(&self) -> String {
        self.tokens.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join("")
    }
}
