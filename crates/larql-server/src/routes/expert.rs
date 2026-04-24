//! POST /v1/expert/{layer}/{expert_id} — remote expert endpoint for MoE inference.
//!
//! A shard server started with `--experts START-END` owns a contiguous range of
//! experts. The inference client routes individual expert calls to the right
//! shard rather than running all experts locally.
//!
//! # Single expert
//!   POST /v1/expert/{layer}/{expert_id}
//!   Body: {"residual": [f32...]}
//!   Response: {"output": [f32...], "latency_ms": f64}
//!
//! # Batch (multiple experts in one round-trip)
//!   POST /v1/expert/batch
//!   Body: {"requests": [{"layer": usize, "expert_id": usize, "residual": [f32...]}, ...]}
//!   Response: {"results": [{"layer": usize, "expert_id": usize, "output": [f32...]}, ...], "latency_ms": f64}

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use serde::{Deserialize, Serialize};

use crate::error::ServerError;
use crate::state::AppState;
use larql_inference;

// ── Request / response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SingleExpertRequest {
    pub residual: Vec<f32>,
}

#[derive(Serialize)]
pub struct SingleExpertResponse {
    pub output: Vec<f32>,
    pub latency_ms: f64,
}

#[derive(Deserialize)]
pub struct BatchExpertItem {
    pub layer: usize,
    pub expert_id: usize,
    pub residual: Vec<f32>,
}

#[derive(Deserialize)]
pub struct BatchExpertRequest {
    pub requests: Vec<BatchExpertItem>,
}

#[derive(Serialize)]
pub struct BatchExpertResult {
    pub layer: usize,
    pub expert_id: usize,
    pub output: Vec<f32>,
}

#[derive(Serialize)]
pub struct BatchExpertResponse {
    pub results: Vec<BatchExpertResult>,
    pub latency_ms: f64,
}

// ── Core computation ──────────────────────────────────────────────────────────

fn run_expert(
    state: &AppState,
    layer: usize,
    expert_id: usize,
    residual: &[f32],
) -> Result<Vec<f32>, ServerError> {
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;

    // Ownership check: reject if this shard doesn't own this expert.
    if let Some((start, end)) = model.expert_filter {
        if expert_id < start || expert_id > end {
            return Err(ServerError::BadRequest(format!(
                "expert {expert_id} not owned by this shard (owns {start}–{end})"
            )));
        }
    }

    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;

    let arch = &*weights.arch;

    if !arch.is_hybrid_moe() {
        return Err(ServerError::BadRequest(
            "model is not a hybrid MoE — no expert endpoints available".into(),
        ));
    }

    let hidden = model.config.hidden_size;
    if residual.len() != hidden {
        return Err(ServerError::BadRequest(format!(
            "residual length {} != hidden_size {hidden}",
            residual.len()
        )));
    }

    // Retrieve MoE weight keys.
    let gate_up_key = arch
        .packed_experts_gate_up_key(layer)
        .ok_or_else(|| ServerError::BadRequest(format!("no MoE gate/up weights for layer {layer}")))?;
    let down_key = arch
        .packed_experts_down_key(layer)
        .ok_or_else(|| ServerError::BadRequest(format!("no MoE down weights for layer {layer}")))?;

    let experts_gate_up = weights
        .get_packed_bytes(&gate_up_key)
        .ok_or_else(|| ServerError::Internal(format!("gate_up bytes missing for layer {layer}")))?;
    let experts_down = weights
        .get_packed_bytes(&down_key)
        .ok_or_else(|| ServerError::Internal(format!("down bytes missing for layer {layer}")))?;

    let inter = arch.moe_intermediate_size();
    let activation = larql_inference::activation_from_arch(arch);

    let output = if let Some(norm_key) = arch.moe_pre_experts_norm_key(layer) {
        let pre_experts_norm = weights
            .vectors
            .get(&norm_key)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        larql_inference::run_single_expert_with_norm(
            residual,
            experts_gate_up,
            experts_down,
            expert_id,
            inter,
            pre_experts_norm,
            arch.norm_weight_offset(),
            arch.norm_eps(),
            activation,
        )
    } else {
        larql_inference::run_single_expert(
            residual,
            experts_gate_up,
            experts_down,
            expert_id,
            inter,
            activation,
        )
    };

    Ok(output)
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

pub async fn handle_expert(
    State(state): State<Arc<AppState>>,
    Path((layer, expert_id)): Path<(usize, usize)>,
    Json(req): Json<SingleExpertRequest>,
) -> Result<Json<SingleExpertResponse>, ServerError> {
    state.bump_requests();
    let start = std::time::Instant::now();

    let output = tokio::task::spawn_blocking(move || {
        run_expert(&state, layer, expert_id, &req.residual)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(SingleExpertResponse { output, latency_ms }))
}

pub async fn handle_expert_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchExpertRequest>,
) -> Result<Json<BatchExpertResponse>, ServerError> {
    state.bump_requests();
    let start = std::time::Instant::now();

    let results = tokio::task::spawn_blocking(move || {
        req.requests
            .iter()
            .map(|item| {
                run_expert(&state, item.layer, item.expert_id, &item.residual).map(|output| {
                    BatchExpertResult {
                        layer: item.layer,
                        expert_id: item.expert_id,
                        output,
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(BatchExpertResponse { results, latency_ms }))
}
