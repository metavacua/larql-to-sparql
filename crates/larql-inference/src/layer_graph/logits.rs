//! Logits computation — final norm + vindex KNN + softmax.

use ndarray::Array2;

use larql_compute::ComputeBackend;
use crate::model::ModelWeights;

/// Shared logits computation: final norm + vindex KNN + softmax.
pub fn finalize_logits(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h: &Array2<f32>,
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    norm_offset: f32,
) -> crate::forward::PredictResult {
    let h_final = crate::forward::apply_norm(weights, h, weights.arch.final_norm_key(), norm_offset);
    let seq_len = h_final.shape()[0];
    let last_row = h_final.row(seq_len - 1).to_owned();

    let hits = index.lm_head_knn_backend(&last_row, top_k, backend);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    let scaled: Vec<(u32, f32)> = hits.iter().map(|&(tid, score)| {
        let mut logit = score * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        (tid, logit)
    }).collect();

    let max_logit = scaled.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled.iter().map(|(_, l)| ((*l - max_logit) as f64).exp()).sum();
    let predictions = scaled.iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer.decode(&[tid], true).ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult { predictions, token_ids: Vec::new() }
}

/// Softmax probability of a single score within a set of hits.
pub(super) fn softmax_prob(score: f32, hits: &[(u32, f32)], logits_scale: f32, softcap: Option<f32>) -> f64 {
    let inv_scale = 1.0 / logits_scale;
    let scaled: Vec<f32> = hits.iter().map(|&(_, s)| {
        let mut l = s * inv_scale;
        if let Some(cap) = softcap { l = (l / cap).tanh() * cap; }
        l
    }).collect();
    let max_l = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled.iter().map(|l| ((*l - max_l) as f64).exp()).sum();
    let mut target = score * inv_scale;
    if let Some(cap) = softcap { target = (target / cap).tanh() * cap; }
    ((target - max_l) as f64).exp() / exp_sum
}
