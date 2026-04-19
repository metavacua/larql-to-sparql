//! POST /v1/walk-ffn — decoupled inference protocol.
//!
//! L2 FFN cache: single-position (`seq_len == 1`) requests with `full_output`
//! check the per-model L2 cache before running WalkFfn. Cache key is derived
//! from the gate-KNN feature IDs for that layer (same scheme as L1).
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
use larql_vindex::GateIndex as _;
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

    // Reject layers outside this shard's owned range before touching any pages.
    {
        let patched = model.patched.blocking_read();
        let base = patched.base();
        for &layer in &scan_layers {
            if !base.is_layer_owned(layer) {
                let range_desc = match base.owned_layer_range() {
                    Some((s, e)) => format!("{s}–{}", e - 1),
                    None => "all".into(),
                };
                return Err(ServerError::BadRequest(format!(
                    "layer {layer} not served by this shard (owned: {range_desc})"
                )));
            }
        }
    }

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
    // Q4_K vindexes take a per-layer dequantise path: the FFN weights live
    // in `interleaved_q4k.bin` (mmap) and are never materialised into
    // `weights.tensors` at load time. For each requested layer we dequant
    // gate/up/down and run the FFN math directly. WalkFfn would panic
    // otherwise (it reads `weights.tensors[ffn_gate_key]` etc.).
    let is_q4k = model.config.quant == larql_vindex::QuantFormat::Q4k;
    let walk_ffn = if is_q4k {
        None
    } else {
        Some(larql_inference::vindex::WalkFfn::new_unlimited(weights, &*patched))
    };

    // WalkFfn expects Array2 shaped [seq_len, hidden]; the wire format is row-major.
    let hidden = model.config.hidden_size;
    let seq_len = req.seq_len;
    let x = Array2::from_shape_vec((seq_len, hidden), req.residual.clone())
        .map_err(|e| ServerError::Internal(format!("reshape residual: {e}")))?;

    // L2 cache is only consulted for single-position requests (autoregressive step).
    let use_l2_cache = seq_len == 1;

    let mut results = Vec::with_capacity(scan_layers.len());
    for &layer in scan_layers {
        if layer >= model.config.num_layers {
            return Err(ServerError::BadRequest(format!(
                "layer {layer} out of range (num_layers = {})",
                model.config.num_layers
            )));
        }

        // L2 cache check: compute gate-KNN key, look up before running FFN.
        // Skip when the layer has active overrides (INSERT patches may change
        // down/up vectors without changing the gate, so the feature-ID key
        // would match but the output would be stale).
        let l2_key = if use_l2_cache && !(*patched).has_overrides_at(layer) {
            let x_1d = x.row(0).to_owned();
            let hits = patched.gate_knn(layer, &x_1d, req.top_k);
            let feat_ids: Vec<usize> = hits.iter().map(|(f, _)| *f).collect();
            let key = crate::ffn_l2_cache::FfnL2Cache::key(&feat_ids);
            if let Some(cached) = model.ffn_l2_cache.get(layer, key) {
                let output: Vec<f32> = (*cached).clone();
                results.push(serde_json::json!({
                    "layer": layer,
                    "output": output,
                    "seq_len": seq_len,
                }));
                continue;
            }
            Some(key)
        } else {
            None
        };

        let out = if let Some(ref wf) = walk_ffn {
            wf.forward(layer, &x)
        } else {
            larql_inference::vindex::q4k_ffn_forward_layer(
                &*weights.arch, patched.base(), layer, &x,
            )
        };
        // out shape is [seq_len, hidden] — flatten row-major.
        let output: Vec<f32> = out.into_iter().collect();
        debug_assert_eq!(output.len(), seq_len * hidden);

        // L2 cache insert (non-blocking, best-effort).
        if let Some(key) = l2_key {
            model.ffn_l2_cache.insert(layer, key, output.clone());
        }

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
    let result = tokio::task::spawn_blocking(move || {
        let result = run_walk_ffn(&state, &req)?;
        // Opt-in hard-RSS bound: after the request completes, advise the
        // kernel to drop resident mmap pages. Done inside spawn_blocking so
        // the sync `blocking_read` on `patched` doesn't hit a tokio worker.
        if let Some(model) = state.model(None) {
            if model.release_mmap_after_request {
                let patched = model.patched.blocking_read();
                patched.base().release_mmap_pages();
            }
        }
        Ok::<_, ServerError>(result)
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(result))
}
