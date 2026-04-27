//! Graph Walk strategy on the real model.
//!
//! Uses vindex walk: gate KNN → feature → down KNN → token prediction.
//! No forward pass for factual queries — graph lookup only.
//!
//! Three tiers:
//!   A: cached template walk (<0.1ms)
//!   B: dynamic graph walk (1-5ms)
//!   C: fallback to Markov RS (~200ms)

use larql_inference::model::ModelWeights;
use larql_inference::forward::embed_tokens_pub;
use larql_vindex::VectorIndex;
use crate::graph_walk::walk_state::{WalkState, WalkTier};

/// Result of graph walk prediction.
pub struct GraphWalkResult {
    /// Top-K predictions as (token_string, score).
    pub predictions: Vec<(String, f64)>,
    /// Which tier resolved this query.
    pub tier: WalkTier,
    /// Wall clock in microseconds.
    pub latency_us: f64,
    /// Memory used (per-conversation: just token IDs).
    pub memory_bytes: usize,
}

/// Run graph walk prediction: embed → gate KNN → feature → down KNN → token.
///
/// For factual queries, this bypasses the forward pass entirely.
/// The embedding is used to seed the gate KNN, then vindex resolves the answer.
pub fn run_graph_walk(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    index: &VectorIndex,
    token_ids: &[u32],
    top_k: usize,
) -> GraphWalkResult {
    let seq_len = token_ids.len();
    let t0 = std::time::Instant::now();

    // Detect query type from tokens
    let token_strs: Vec<String> = token_ids
        .iter()
        .filter_map(|&id| tokenizer.decode(&[id], true).ok())
        .collect();
    let token_refs: Vec<&str> = token_strs.iter().map(|s| s.as_str()).collect();
    let walk_state = WalkState::from_tokens(&token_refs);

    let predictions = match walk_state.tier {
        WalkTier::CachedTemplate | WalkTier::DynamicWalk => {
            // Graph walk path: embed tokens, use last-token residual for gate KNN
            let h = embed_tokens_pub(weights, token_ids);
            let last_row = h.row(seq_len - 1).to_owned();

            // Vindex walk: score against gate vectors, get top features, resolve tokens
            let walk_trace = index.walk(&last_row, &critical_layers(weights.num_layers), top_k);

            // Collect predictions from walk trace
            let mut preds: Vec<(String, f64)> = Vec::new();
            for (_layer, hits) in &walk_trace.layers {
                for hit in hits {
                    let token_str = tokenizer
                        .decode(&[hit.meta.top_token_id], true)
                        .unwrap_or_default();
                    if !token_str.is_empty() {
                        preds.push((token_str.trim().to_string(), hit.gate_score as f64));
                    }
                }
            }

            // Deduplicate and sort by score
            preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            preds.dedup_by(|a, b| a.0 == b.0);
            preds.truncate(top_k);

            // If walk produced no results, fall back to lm_head KNN
            if preds.is_empty() {
                let lm_hits = index.lm_head_knn(&last_row, top_k);
                for (tok_id, score) in lm_hits {
                    if let Ok(s) = tokenizer.decode(&[tok_id], true) {
                        preds.push((s.trim().to_string(), score as f64));
                    }
                }
            }

            preds
        }
        WalkTier::MarkovFallback => {
            // Fallback: full forward pass via standard predict
            let result = larql_inference::predict(weights, tokenizer, token_ids, top_k);
            result.predictions
        }
    };

    let latency_us = t0.elapsed().as_secs_f64() * 1e6;

    GraphWalkResult {
        predictions,
        tier: walk_state.tier,
        latency_us,
        memory_bytes: seq_len * 4, // token IDs only
    }
}

/// Run graph walk using vindex logits (lm_head KNN) for final prediction.
/// This is the simpler path: full forward pass through walk FFN, then KNN logits.
/// Requires a WalkLayerGraph (dense attention + walk FFN per layer).
pub fn run_graph_walk_vindex_logits(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    index: &VectorIndex,
    token_ids: &[u32],
    top_k: usize,
) -> GraphWalkResult {
    let seq_len = token_ids.len();
    let t0 = std::time::Instant::now();

    // Build a WalkLayerGraph: dense attention + walk FFN
    let walk_ffn = larql_inference::WalkFfn::new_unlimited(weights, index);
    let walk_graph = larql_inference::WalkLayerGraph {
        ffn: &walk_ffn,
        backend: None,
    };

    // Use the existing predict_with_graph_vindex_logits pipeline
    let result = larql_inference::predict_with_graph_vindex_logits(
        weights, tokenizer, token_ids, top_k, &walk_graph, index,
    );

    let latency_us = t0.elapsed().as_secs_f64() * 1e6;

    GraphWalkResult {
        predictions: result.predictions,
        tier: WalkTier::DynamicWalk,
        latency_us,
        memory_bytes: seq_len * 4,
    }
}

/// Critical layers for graph walk (where factual retrieval happens).
/// Based on measured data: L13 task classifier, L15 confidence router, L24-L26 factual.
fn critical_layers(num_layers: usize) -> Vec<usize> {
    let mut layers = vec![13, 15, 24, 25, 26];
    layers.retain(|&l| l < num_layers);
    layers
}
