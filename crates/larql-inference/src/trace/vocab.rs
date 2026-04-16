//! Vocabulary projection helpers — project residual vectors through lm_head.

use ndarray::Array2;
use crate::model::ModelWeights;
use larql_models::NormType;

/// Project a vector through final_norm → lm_head → logits.
pub fn project_to_logits(weights: &ModelWeights, vec: &[f32]) -> Vec<f32> {
    let hidden = weights.hidden_size;
    let norm_offset = weights.arch.norm_weight_offset();

    let v = Array2::from_shape_vec((1, hidden), vec.to_vec()).unwrap();
    let normed = apply_norm(weights, &v, weights.arch.final_norm_key(), norm_offset);
    let normed_row = normed.row(0);

    let logits_scale = weights.arch.logits_scaling();
    let softcap = weights.arch.final_logit_softcapping();
    let mut logits = Vec::with_capacity(weights.vocab_size);
    for tok_id in 0..weights.vocab_size {
        let lm_row = weights.lm_head.row(tok_id);
        let dot: f64 = normed_row.iter()
            .zip(lm_row.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let mut logit = (dot / logits_scale as f64) as f32;
        if let Some(cap) = softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logits.push(logit);
    }
    logits
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    logits.iter().map(|&l| (((l - max) as f64).exp() / exp_sum) as f32).collect()
}

pub fn top_k_from_logits(logits: &[f32], tokenizer: &tokenizers::Tokenizer, k: usize) -> Vec<(String, f32)> {
    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    let k = k.min(indexed.len());
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter()
        .filter_map(|(idx, prob)| {
            tokenizer.decode(&[idx as u32], true).ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect()
}

pub fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn apply_norm(
    weights: &ModelWeights, x: &Array2<f32>, weight_key: &str, norm_offset: f32,
) -> Array2<f32> {
    match weights.arch.norm_type() {
        NormType::LayerNorm => {
            let bias_key = weight_key.replace(".weight", ".bias");
            crate::residual::layer_norm(x, weights.vectors.get(weight_key), weights.vectors.get(&bias_key))
        }
        _ => crate::residual::rms_norm(x, weights.vectors.get(weight_key), norm_offset),
    }
}
