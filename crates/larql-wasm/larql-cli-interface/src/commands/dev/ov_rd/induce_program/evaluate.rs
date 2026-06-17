// PromptResult fields (id, stratum, …) are accumulated for a future
// diagnostic dump; suppress until the viewer is wired.
#![allow(dead_code)]

use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::super::address::attention_argmax;
use super::super::metrics::{
    argmax, bool_rate, kl_logp, log_softmax, mean, percentile, top_k_indices,
};
use super::super::oracle_pq_forward::{final_logits, forward_q4k_predicted_address_mode_d_head};
use super::super::program::{BehaviorMetrics, PositionContext, Program};
use super::context::FitContext;

/// Per-prompt result from a single proposal evaluation.
pub struct PromptResult {
    pub id: String,
    pub stratum: String,
    pub kl: f64,
    pub top1_agree: bool,
    pub baseline_top1_in_top5: bool,
}

/// Evaluate `program` using pre-captured oracle codes and baselines.
///
/// This avoids re-running oracle PQ or attention capture — only the Mode D
/// injection forward pass runs per proposal per prompt.
pub fn evaluate_program_fast(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    fit: &FitContext,
    program: &Program,
) -> Result<(Vec<PromptResult>, BehaviorMetrics), Box<dyn std::error::Error>> {
    let target_group = fit.group;
    let mut results = Vec::with_capacity(fit.captures.len());

    for capture in &fit.captures {
        let remapped: Vec<Vec<usize>> = capture
            .oracle_codes
            .iter()
            .enumerate()
            .map(|(pos, oracle_codes)| {
                let mut codes = oracle_codes.clone();
                let original = oracle_codes[target_group];

                let attn_row = capture
                    .attention_rows
                    .get(pos)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let attn_argmax = attention_argmax(attn_row, pos);
                let ctx = PositionContext {
                    stratum: capture.stratum.clone(),
                    position: pos,
                    token_id: capture.token_ids.get(pos).copied().unwrap_or(0),
                    prev_token_id: (pos > 0)
                        .then(|| capture.token_ids.get(pos - 1).copied())
                        .flatten(),
                    attends_bos: attn_argmax == 0,
                    attends_prev: pos > 0 && attn_argmax + 1 == pos,
                    original_code: original,
                    current_code: original,
                };
                codes[target_group] = program.apply_to_code(original, &ctx);
                codes
            })
            .collect();

        let replacement_delta = {
            let flat: Vec<f32> = (0..remapped.len())
                .flat_map(|pos| {
                    fit.mode_d_table.delta_for_position_codes_with_stratum(
                        pos,
                        &remapped[pos],
                        &capture.stratum,
                    )
                })
                .collect();
            Array2::from_shape_vec((capture.token_ids.len(), weights.hidden_size), flat)
                .map_err(|e| format!("delta shape: {e}"))?
        };
        let program_h = if let Some(ref b) = fit.metal {
            if let Some(h) = super::super::metal_backend::try_metal(
                weights,
                &capture.token_ids,
                index,
                fit.head.layer,
                fit.head.head,
                &replacement_delta,
                b,
            ) {
                h
            } else {
                forward_q4k_predicted_address_mode_d_head(
                    weights,
                    &capture.token_ids,
                    index,
                    fit.head,
                    &fit.mode_d_table,
                    &remapped,
                    &capture.stratum,
                )?
            }
        } else {
            forward_q4k_predicted_address_mode_d_head(
                weights,
                &capture.token_ids,
                index,
                fit.head,
                &fit.mode_d_table,
                &remapped,
                &capture.stratum,
            )?
        };
        let program_logits = final_logits(weights, &program_h);
        let program_logp = log_softmax(&program_logits);
        let program_top1 = argmax(&program_logits);
        let program_top5 = top_k_indices(&program_logits, 5);

        results.push(PromptResult {
            id: capture.id.clone(),
            stratum: capture.stratum.clone(),
            kl: kl_logp(&capture.baseline_logp, &program_logp),
            top1_agree: capture.baseline_top1 == program_top1,
            baseline_top1_in_top5: program_top5.contains(&capture.baseline_top1),
        });
    }

    let kls: Vec<f64> = results.iter().map(|r| r.kl).collect();
    let metrics = BehaviorMetrics {
        mean_kl: mean(&kls),
        p95_kl: percentile(kls.clone(), 0.95),
        max_kl: kls.iter().cloned().fold(0.0_f64, f64::max),
        top1: bool_rate(results.iter().map(|r| r.top1_agree)),
        top5: bool_rate(results.iter().map(|r| r.baseline_top1_in_top5)),
    };

    Ok((results, metrics))
}
