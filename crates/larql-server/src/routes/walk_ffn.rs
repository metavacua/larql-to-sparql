//! POST /v1/walk-ffn — decoupled inference protocol.
//!
//! Client sends a residual vector, server runs either (a) gate KNN only, or
//! (b) the full FFN compute, and returns the result. This enables distributed
//! inference where the client runs attention locally and the server provides
//! the sparse FFN computation.
//!
//! # Features-only mode (default)
//!
//! Single layer:
//!   POST /v1/walk-ffn {"layer": 26, "residual": [0.12, -0.34, ...]}
//!   → {"layer": 26, "features": [f0, f1, ...], "scores": [s0, s1, ...]}
//!
//! Batched:
//!   POST /v1/walk-ffn {"layers": [0,1,...], "residual": [...]}
//!   → {"results": [{"layer": 0, "features": [...], "scores": [...]}, ...]}
//!
//! # Full-output mode (`"full_output": true`)
//!
//! Returns the FFN output vectors for each requested layer, computed via the
//! same `WalkFfn` path used by local inference (gate KNN → activation → up
//! gather → down projection, architecture-correct).
//!
//! The `residual` field is a row-major flat array of length `seq_len *
//! hidden_size`. `seq_len` defaults to 1 and lets the server process a whole
//! sequence (prefill) in one round trip. Output mirrors the shape.
//!
//! Single layer:
//!   POST /v1/walk-ffn {"layer": 26, "residual": [...], "seq_len": 1,
//!                       "full_output": true}
//!   → {"layer": 26, "output": [...], "seq_len": 1}
//!
//! Batched:
//!   POST /v1/walk-ffn {"layers": [...], "residual": [...], "seq_len": N,
//!                       "full_output": true}
//!   → {"results": [{"layer": N, "output": [...], "seq_len": N}, ...]}
//!
//! Full-output mode triggers lazy loading of model weights. On first call it
//! mmaps the vindex weight files; subsequent calls reuse the loaded state.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize)]
pub struct WalkFfnRequest {
    /// Single layer mode.
    #[serde(default)]
    pub layer: Option<usize>,
    /// Batched mode — multiple layers in one request.
    #[serde(default)]
    pub layers: Option<Vec<usize>>,
    /// Residual vector(s), row-major flat. Length must be `seq_len *
    /// hidden_size`. Features-only mode requires `seq_len == 1` (only the
    /// first `hidden_size` elements are consulted).
    pub residual: Vec<f32>,
    /// Sequence length — number of residual rows in the flat `residual`
    /// array. Defaults to 1. Ignored in features-only mode.
    #[serde(default = "default_seq_len")]
    pub seq_len: usize,
    /// Top-K features to select. Ignored in `full_output` mode (WalkFfn uses
    /// its own unlimited-K default there).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// When true, return the computed FFN output vector per layer instead of
    /// feature indices + scores. Requires loadable model weights.
    #[serde(default)]
    pub full_output: bool,
}

fn default_seq_len() -> usize { 1 }

fn default_top_k() -> usize { 8092 }

fn run_walk_ffn(
    state: &AppState,
    req: &WalkFfnRequest,
) -> Result<serde_json::Value, ServerError> {
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;

    let hidden = model.config.hidden_size;
    let expected_len = if req.full_output {
        req.seq_len
            .checked_mul(hidden)
            .ok_or_else(|| ServerError::BadRequest("seq_len * hidden overflow".into()))?
    } else {
        hidden
    };
    if req.residual.len() != expected_len {
        return Err(ServerError::BadRequest(format!(
            "residual has {} elements, expected {expected_len} (seq_len={} * hidden_size={hidden})",
            req.residual.len(),
            if req.full_output { req.seq_len } else { 1 },
        )));
    }
    if req.full_output && req.seq_len == 0 {
        return Err(ServerError::BadRequest("seq_len must be >= 1".into()));
    }

    let scan_layers: Vec<usize> = if let Some(ref layers) = req.layers {
        layers.clone()
    } else if let Some(layer) = req.layer {
        vec![layer]
    } else {
        return Err(ServerError::BadRequest(
            "must provide 'layer' or 'layers'".into(),
        ));
    };

    let start = std::time::Instant::now();

    if req.full_output {
        run_full_output(model, req, &scan_layers, start)
    } else {
        run_features_only(model, req, &scan_layers, start)
    }
}

fn run_features_only(
    model: &LoadedModel,
    req: &WalkFfnRequest,
    scan_layers: &[usize],
    start: std::time::Instant,
) -> Result<serde_json::Value, ServerError> {
    let patched = model.patched.blocking_read();
    let query = larql_vindex::ndarray::Array1::from_vec(req.residual.clone());

    let mut results = Vec::with_capacity(scan_layers.len());
    for &layer in scan_layers {
        let hits = patched.gate_knn(layer, &query, req.top_k);
        let features: Vec<usize> = hits.iter().map(|(f, _)| *f).collect();
        let scores: Vec<f32> = hits
            .iter()
            .map(|(_, s)| (*s * 100.0).round() / 100.0)
            .collect();
        results.push(serde_json::json!({
            "layer": layer,
            "features": features,
            "scores": scores,
        }));
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let latency_rounded = (latency_ms * 10.0).round() / 10.0;

    if scan_layers.len() == 1 {
        let r = &results[0];
        Ok(serde_json::json!({
            "layer": r["layer"],
            "features": r["features"],
            "scores": r["scores"],
            "latency_ms": latency_rounded,
        }))
    } else {
        Ok(serde_json::json!({
            "results": results,
            "latency_ms": latency_rounded,
        }))
    }
}

fn run_full_output(
    model: &LoadedModel,
    req: &WalkFfnRequest,
    scan_layers: &[usize],
    start: std::time::Instant,
) -> Result<serde_json::Value, ServerError> {
    use larql_inference::ffn::FfnBackend;
    use larql_vindex::ndarray::Array2;

    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;

    let patched = model.patched.blocking_read();
    let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited(weights, &*patched);

    // WalkFfn expects Array2 shaped [seq_len, hidden]; the wire format is row-major.
    let hidden = model.config.hidden_size;
    let seq_len = req.seq_len;
    let x = Array2::from_shape_vec((seq_len, hidden), req.residual.clone())
        .map_err(|e| ServerError::Internal(format!("reshape residual: {e}")))?;

    let mut results = Vec::with_capacity(scan_layers.len());
    for &layer in scan_layers {
        if layer >= model.config.num_layers {
            return Err(ServerError::BadRequest(format!(
                "layer {layer} out of range (num_layers = {})",
                model.config.num_layers
            )));
        }
        let out = walk_ffn.forward(layer, &x);
        // out shape is [seq_len, hidden] — flatten row-major.
        let output: Vec<f32> = out.into_iter().collect();
        debug_assert_eq!(output.len(), seq_len * hidden);
        results.push(serde_json::json!({
            "layer": layer,
            "output": output,
            "seq_len": seq_len,
        }));
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let latency_rounded = (latency_ms * 10.0).round() / 10.0;

    if scan_layers.len() == 1 {
        let r = &results[0];
        Ok(serde_json::json!({
            "layer": r["layer"],
            "output": r["output"],
            "seq_len": r["seq_len"],
            "latency_ms": latency_rounded,
        }))
    } else {
        Ok(serde_json::json!({
            "results": results,
            "seq_len": seq_len,
            "latency_ms": latency_rounded,
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
