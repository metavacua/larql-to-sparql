use larql_vindex::VectorIndex;

use super::address::address_match_report;
use super::metrics::{argmax, kl_logp, log_softmax, top_k_indices};
use super::oracle_pq_forward::{final_logits, forward_q4k_predicted_address_mode_d_head};
use super::pq::ModeDTable;
use super::reports::AddressProbePromptReport;
use super::types::HeadId;

pub(super) fn evaluate_predicted_address(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    mode_d_table: &ModeDTable,
    predicted_codes_by_position: &[Vec<usize>],
    stratum: &str,
    label: &str,
    baseline_logp: &[f64],
    baseline_top1: u32,
    oracle_codes_by_position: &[Vec<usize>],
) -> Result<AddressProbePromptReport, Box<dyn std::error::Error>> {
    let address_match = address_match_report(oracle_codes_by_position, predicted_codes_by_position);
    let predicted_hidden = forward_q4k_predicted_address_mode_d_head(
        weights,
        token_ids,
        index,
        head,
        mode_d_table,
        predicted_codes_by_position,
        stratum,
    )?;
    let predicted_logits = final_logits(weights, &predicted_hidden);
    let predicted_logp = log_softmax(&predicted_logits);
    let predicted_top1 = argmax(&predicted_logits);
    let predicted_top5 = top_k_indices(&predicted_logits, 5);

    Ok(AddressProbePromptReport {
        id: label.to_string(),
        stratum: stratum.to_string(),
        kl: kl_logp(baseline_logp, &predicted_logp),
        positions: oracle_codes_by_position.len(),
        groups_correct: address_match.groups_correct,
        groups_total: address_match.groups_total,
        exact_address_match: address_match.exact_address_match,
        top1_agree: baseline_top1 == predicted_top1,
        baseline_top1_in_predicted_top5: predicted_top5.contains(&baseline_top1),
    })
}
