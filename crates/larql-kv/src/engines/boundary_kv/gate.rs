//! Boundary-frame gate helpers for `BoundaryKvEngine`.
//!
//! Pure functions: take residuals + config, produce a
//! `BoundaryFrame`. The engine's `maybe_emit_frame` path in
//! `engine.rs` calls [`build_frame`] at every chunk boundary; the
//! other helpers ([`reconstruct_logits`], [`encode_for_decision`],
//! [`last_row_flat`]) are factored out for clarity + reuse.
//!
//! Codec choice is driven by the [`BoundaryGate`]'s decision (see
//! `larql_boundary::gate::apply`): compressed `Int8Clip3Sigma`,
//! uncompressed `bf16`, or cold-replay / reject. Verify-agreement
//! mode (codec_int8 round-trip + lm_head re-project) feeds the
//! gate's `boundary_agreement` metric.

use larql_boundary::{
    codec::{bf16 as codec_bf16, int8 as codec_int8},
    metadata, BoundaryAgreement, BoundaryCompression, BoundaryContract, BoundaryDecision,
    BoundaryFrame,
};
use larql_inference::forward::hidden_to_raw_logits;
use larql_inference::model::ModelWeights;
use ndarray::Array2;

use crate::engines::boundary_kv::engine::BoundaryKvEngineConfig;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Build a frame for the boundary ending at `token_end`.
///
/// `sequence_id` is the effective chain id (the engine's
/// `current_sequence_id()`, which differs from `config.sequence_id`
/// after a re-prefill) — the frame's `boundary_id` is derived from it.
pub(super) fn build_frame(
    weights: &ModelWeights,
    hidden: &Array2<f32>,
    config: &BoundaryKvEngineConfig,
    sequence_id: &str,
    token_start: u64,
    token_end: u64,
) -> BoundaryFrame {
    // Guard: this function is pub(super) and callable directly, so it must
    // handle a 0-row hidden itself (the engine's `maybe_emit_frame` also
    // pre-checks). No residual exists → emit an empty Unknown frame rather
    // than underflowing `rows - 1` in the reconstruction path.
    if hidden.shape()[0] == 0 || hidden.shape()[1] == 0 {
        return empty_hidden_frame(weights, config, sequence_id, token_start, token_end);
    }
    let residual = last_row_flat(hidden);
    let hidden_size = residual.len() as u32;
    let raw_logits = hidden_to_raw_logits(weights, hidden);
    let hat_logits = if config.verify_agreement {
        Some(reconstruct_logits(weights, hidden, &residual))
    } else {
        None
    };
    let mut meta = metadata::compute(&raw_logits, hat_logits.as_deref());
    let decision = larql_boundary::gate::apply(&mut meta, &config.gate_config);

    let (compression_scheme, contract_level, payload) = encode_for_decision(&decision, &residual);

    let boundary_id = format!("{sequence_id}:{token_end}");
    BoundaryFrame {
        version: 1,
        model_id: config.identity.model_id.clone(),
        model_revision: config.identity.model_revision.clone(),
        tokenizer_revision: config.identity.tokenizer_revision.clone(),
        architecture: config.identity.architecture.clone(),
        boundary_id,
        sequence_id: sequence_id.to_string(),
        token_start,
        token_end,
        layer: weights.num_layers.saturating_sub(1) as u16,
        hidden_size,
        compression_scheme,
        contract_level,
        payload,
        raw_top1_token: meta.raw_top1_token,
        raw_logit_margin: meta.raw_logit_margin,
        raw_top1_prob: Some(meta.raw_top1_prob),
        compressed_top1_token: meta.compressed_top1_token,
        boundary_agreement: meta.boundary_agreement,
        codec_fragile: meta.codec_fragile,
        boundary_fragile: meta.boundary_fragile,
        fallback_policy: config.gate_config.fallback_policy.clone(),
        fallback_ref: None,
        calibration_run_id: None,
        residual_hash: None,
        token_hash: None,
    }
}

/// Degenerate frame for a 0-row (or 0-column) hidden: no residual, no
/// logits, `Unknown` contract, `NotChecked` agreement. Never useful for
/// resume — it exists so a direct `build_frame` call cannot panic.
fn empty_hidden_frame(
    weights: &ModelWeights,
    config: &BoundaryKvEngineConfig,
    sequence_id: &str,
    token_start: u64,
    token_end: u64,
) -> BoundaryFrame {
    BoundaryFrame {
        version: 1,
        model_id: config.identity.model_id.clone(),
        model_revision: config.identity.model_revision.clone(),
        tokenizer_revision: config.identity.tokenizer_revision.clone(),
        architecture: config.identity.architecture.clone(),
        boundary_id: format!("{sequence_id}:{token_end}"),
        sequence_id: sequence_id.to_string(),
        token_start,
        token_end,
        layer: weights.num_layers.saturating_sub(1) as u16,
        hidden_size: 0,
        compression_scheme: BoundaryCompression::None,
        contract_level: BoundaryContract::Unknown,
        payload: Vec::new(),
        raw_top1_token: 0,
        raw_logit_margin: 0.0,
        raw_top1_prob: None,
        compressed_top1_token: None,
        boundary_agreement: BoundaryAgreement::NotChecked,
        codec_fragile: false,
        boundary_fragile: false,
        fallback_policy: config.gate_config.fallback_policy.clone(),
        fallback_ref: None,
        calibration_run_id: None,
        residual_hash: None,
        token_hash: None,
    }
}

pub(super) fn last_row_flat(hidden: &Array2<f32>) -> Vec<f32> {
    let rows = hidden.shape()[0];
    if rows == 0 {
        return Vec::new();
    }
    hidden.row(rows - 1).to_vec()
}

/// Run the compressed-residual forward (encode → decode → re-project
/// to logits) so the gate can populate `boundary_agreement`. Uses
/// the same `lm_head` as the raw path so any margin shift is purely
/// codec-induced.
fn reconstruct_logits(weights: &ModelWeights, hidden: &Array2<f32>, residual: &[f32]) -> Vec<f32> {
    let encoded = codec_int8::encode(residual);
    let decoded = codec_int8::decode(&encoded);
    let mut hat = hidden.clone();
    let rows = hat.shape()[0];
    let cols = hat.shape()[1];
    let last = rows - 1;
    for j in 0..cols {
        hat[[last, j]] = decoded[j];
    }
    hidden_to_raw_logits(weights, &hat)
}

fn encode_for_decision(
    decision: &BoundaryDecision,
    residual: &[f32],
) -> (BoundaryCompression, BoundaryContract, Vec<u8>) {
    match decision {
        BoundaryDecision::CompressedOk { contract } => {
            let payload = codec_int8::encode(residual).to_bytes();
            (
                BoundaryCompression::Int8Clip3Sigma,
                contract.clone(),
                payload,
            )
        }
        BoundaryDecision::UseBf16 => (
            BoundaryCompression::None,
            BoundaryContract::Calibrating,
            codec_bf16::encode(residual),
        ),
        BoundaryDecision::UseColdReplay | BoundaryDecision::Reject => (
            BoundaryCompression::None,
            BoundaryContract::Unknown,
            Vec::new(),
        ),
    }
}
