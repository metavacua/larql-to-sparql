//! POST /v1/walk-ffn — decoupled inference protocol.
//!
//! Client sends a residual vector, server runs gate KNN + down projection,
//! returns the FFN output. This enables distributed inference where the client
//! runs attention locally and the server provides the sparse FFN computation.
//!
//! Single-layer mode:
//!   POST /v1/walk-ffn {"layer": 26, "residual": [0.12, -0.34, ...]}
//!   → {"output": [feature_idx, feature_idx, ...], "scores": [score, score, ...]}
//!
//! Batched mode (all layers in one request):
//!   POST /v1/walk-ffn {"layers": [0,1,...,33], "residual": [0.12, -0.34, ...]}
//!   → {"results": [{"layer": 0, "features": [...], "scores": [...]}, ...]}

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct WalkFfnRequest {
    /// Single layer mode.
    #[serde(default)]
    pub layer: Option<usize>,
    /// Batched mode — all layers.
    #[serde(default)]
    pub layers: Option<Vec<usize>>,
    /// Residual vector (hidden_size floats).
    pub residual: Vec<f32>,
    /// Top-K features to select.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize { 8092 }

fn run_walk_ffn(
    state: &AppState,
    req: &WalkFfnRequest,
) -> Result<serde_json::Value, ServerError> {
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;

    let patched = model.patched.blocking_read();

    if req.residual.len() != model.config.hidden_size {
        return Err(ServerError::BadRequest(format!(
            "residual has {} elements, expected {} (hidden_size)",
            req.residual.len(),
            model.config.hidden_size
        )));
    }

    let query = larql_vindex::ndarray::Array1::from_vec(req.residual.clone());
    let start = std::time::Instant::now();

    let scan_layers: Vec<usize> = if let Some(ref layers) = req.layers {
        layers.clone()
    } else if let Some(layer) = req.layer {
        vec![layer]
    } else {
        return Err(ServerError::BadRequest("must provide 'layer' or 'layers'".into()));
    };

    let mut results = Vec::with_capacity(scan_layers.len());
    for &layer in &scan_layers {
        let hits = patched.gate_knn(layer, &query, req.top_k);
        let features: Vec<usize> = hits.iter().map(|(f, _)| *f).collect();
        let scores: Vec<f32> = hits.iter().map(|(_, s)| (*s * 100.0).round() / 100.0).collect();
        results.push(serde_json::json!({
            "layer": layer,
            "features": features,
            "scores": scores,
        }));
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    if scan_layers.len() == 1 {
        // Single layer — flat response.
        let r = &results[0];
        Ok(serde_json::json!({
            "layer": r["layer"],
            "features": r["features"],
            "scores": r["scores"],
            "latency_ms": (latency_ms * 10.0).round() / 10.0,
        }))
    } else {
        // Batched — array response.
        Ok(serde_json::json!({
            "results": results,
            "latency_ms": (latency_ms * 10.0).round() / 10.0,
        }))
    }
}

pub async fn handle_walk_ffn(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WalkFfnRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let result = tokio::task::spawn_blocking(move || run_walk_ffn(&state, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
