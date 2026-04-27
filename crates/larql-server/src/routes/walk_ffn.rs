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
//!
//! # Binary wire format (`Content-Type: application/x-larql-ffn`)
//!
//! Requires `full_output = true`. Eliminates JSON float parsing overhead.
//!
//! ## Request — single layer
//! ```text
//! Offset  Size  Field
//! 0       4     layer_index (u32 LE, must not be 0xFFFFFFFF)
//! 4       4     seq_len (u32 LE)
//! 8       4     flags (u32 LE, bit 0 = full_output, must be 1)
//! 12      4     top_k (u32 LE)
//! 16      N×4   residual (f32[] LE)
//! ```
//!
//! ## Request — batch
//! ```text
//! 0       4     BATCH_MARKER = 0xFFFFFFFF
//! 4       4     num_layers (u32 LE)
//! 8       K×4   layer_indices (u32[] LE)
//! 8+K*4   4     seq_len (u32 LE)
//! 12+K*4  4     flags (u32 LE)
//! 16+K*4  4     top_k (u32 LE)
//! 20+K*4  N×4   residual (f32[] LE)
//! ```
//!
//! ## Response — single layer
//! ```text
//! 0       4     layer (u32 LE)
//! 4       4     seq_len (u32 LE)
//! 8       4     latency_ms (f32 LE)
//! 12      N×4   output (f32[] LE)
//! ```
//!
//! ## Response — batch
//! ```text
//! 0       4     BATCH_MARKER = 0xFFFFFFFF
//! 4       4     num_results (u32 LE)
//! 8       4     latency_ms (f32 LE)
//! Per result:
//!   0     4     layer (u32 LE)
//!   4     4     seq_len (u32 LE)
//!   8     4     num_output_floats (u32 LE)
//!   12    M×4   output (f32[] LE)
//! ```

use std::sync::Arc;

use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::Response;
use larql_vindex::GateIndex as _;
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

pub(crate) const BINARY_CT: &str = "application/x-larql-ffn";
pub(crate) const BATCH_MARKER: u32 = 0xFFFF_FFFF;

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

// ── Typed output structs (shared by JSON + binary encoders) ──────────────────

pub(crate) struct FfnEntry {
    pub(crate) layer: usize,
    pub(crate) output: Vec<f32>,
}

pub(crate) struct FfnOutput {
    pub(crate) entries: Vec<FfnEntry>,
    pub(crate) seq_len: usize,
    pub(crate) latency_ms: f64,
}

// ── Binary codec ─────────────────────────────────────────────────────────────

/// Decode a binary-format request body into a [`WalkFfnRequest`].
pub(crate) fn decode_binary_request(body: &[u8]) -> Result<WalkFfnRequest, ServerError> {
    if body.len() < 16 {
        return Err(ServerError::BadRequest("binary: body too short (need ≥ 16 bytes)".into()));
    }

    let first = u32::from_le_bytes(body[0..4].try_into().unwrap());

    let (layer, layers, header_end) = if first == BATCH_MARKER {
        if body.len() < 8 {
            return Err(ServerError::BadRequest("binary batch: truncated num_layers".into()));
        }
        let n = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
        let layers_end = 8 + n * 4;
        if body.len() < layers_end {
            return Err(ServerError::BadRequest(format!(
                "binary batch: body too short for {n} layer indices"
            )));
        }
        let layers: Vec<usize> = (0..n)
            .map(|i| {
                u32::from_le_bytes(body[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize
            })
            .collect();
        (None, Some(layers), layers_end)
    } else {
        (Some(first as usize), None, 4)
    };

    if body.len() < header_end + 12 {
        return Err(ServerError::BadRequest(
            "binary: truncated fixed header (seq_len/flags/top_k)".into(),
        ));
    }
    let seq_len =
        u32::from_le_bytes(body[header_end..header_end + 4].try_into().unwrap()) as usize;
    let flags =
        u32::from_le_bytes(body[header_end + 4..header_end + 8].try_into().unwrap());
    let top_k =
        u32::from_le_bytes(body[header_end + 8..header_end + 12].try_into().unwrap()) as usize;
    let full_output = (flags & 1) != 0;

    let residual_bytes = &body[header_end + 12..];
    if !residual_bytes.len().is_multiple_of(4) {
        return Err(ServerError::BadRequest(
            "binary: residual byte length is not a multiple of 4".into(),
        ));
    }
    let residual: Vec<f32> = residual_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    Ok(WalkFfnRequest {
        layer,
        layers,
        residual,
        seq_len,
        top_k,
        full_output,
    })
}

/// Encode an [`FfnOutput`] as the binary response format.
pub(crate) fn encode_binary_output(out: &FfnOutput) -> Vec<u8> {
    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let mut buf = Vec::with_capacity(12 + entry.output.len() * 4);
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for &v in &entry.output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    } else {
        let num = out.entries.len();
        let mut buf = Vec::with_capacity(12 + num * 12);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for &v in &entry.output {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] as the existing JSON response format (unchanged wire
/// contract for JSON clients).
fn encode_json_full_output(out: &FfnOutput) -> serde_json::Value {
    let latency_rounded = (out.latency_ms * 10.0).round() / 10.0;
    if out.entries.len() == 1 {
        let e = &out.entries[0];
        serde_json::json!({
            "layer": e.layer,
            "output": e.output,
            "seq_len": out.seq_len,
            "latency_ms": latency_rounded,
        })
    } else {
        let results: Vec<serde_json::Value> = out
            .entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "layer": e.layer,
                    "output": e.output,
                    "seq_len": out.seq_len,
                })
            })
            .collect();
        serde_json::json!({
            "results": results,
            "seq_len": out.seq_len,
            "latency_ms": latency_rounded,
        })
    }
}

// ── Request helpers ───────────────────────────────────────────────────────────

fn collect_scan_layers(req: &WalkFfnRequest) -> Result<Vec<usize>, ServerError> {
    if let Some(ref layers) = req.layers {
        Ok(layers.clone())
    } else if let Some(layer) = req.layer {
        Ok(vec![layer])
    } else {
        Err(ServerError::BadRequest(
            "must provide 'layer' or 'layers'".into(),
        ))
    }
}

fn validate_residual(req: &WalkFfnRequest, hidden: usize) -> Result<(), ServerError> {
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
    Ok(())
}

fn validate_owned(model: &LoadedModel, scan_layers: &[usize]) -> Result<(), ServerError> {
    let patched = model.patched.blocking_read();
    let base = patched.base();
    for &layer in scan_layers {
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
    Ok(())
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Architecture-correct FFN forward pass for one or more layers.
/// Returns a typed [`FfnOutput`] used by both JSON and binary encoders.
pub(crate) fn run_full_output_core(
    model: &LoadedModel,
    req: &WalkFfnRequest,
    scan_layers: &[usize],
    start: std::time::Instant,
) -> Result<FfnOutput, ServerError> {
    use larql_inference::ffn::FfnBackend;
    use larql_vindex::ndarray::Array2;

    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;

    let patched = model.patched.blocking_read();
    let is_q4k = model.config.quant == larql_vindex::QuantFormat::Q4k;
    let walk_ffn = if is_q4k {
        None
    } else {
        Some(larql_inference::vindex::WalkFfn::new_unlimited(weights, &*patched))
    };

    let hidden = model.config.hidden_size;
    let seq_len = req.seq_len;
    let x = Array2::from_shape_vec((seq_len, hidden), req.residual.clone())
        .map_err(|e| ServerError::Internal(format!("reshape residual: {e}")))?;

    let use_l2_cache = seq_len == 1;

    let mut entries = Vec::with_capacity(scan_layers.len());
    for &layer in scan_layers {
        if layer >= model.config.num_layers {
            return Err(ServerError::BadRequest(format!(
                "layer {layer} out of range (num_layers = {})",
                model.config.num_layers
            )));
        }

        let l2_key = if use_l2_cache && !(*patched).has_overrides_at(layer) {
            let x_1d = x.row(0).to_owned();
            let hits = patched.gate_knn(layer, &x_1d, req.top_k);
            let feat_ids: Vec<usize> = hits.iter().map(|(f, _)| *f).collect();
            let key = crate::ffn_l2_cache::FfnL2Cache::key(&feat_ids);
            if let Some(cached) = model.ffn_l2_cache.get(layer, key) {
                entries.push(FfnEntry {
                    layer,
                    output: (*cached).clone(),
                });
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
                &*weights.arch,
                patched.base(),
                layer,
                &x,
            )
        };
        let output: Vec<f32> = out.into_iter().collect();
        debug_assert_eq!(output.len(), seq_len * hidden);

        if let Some(key) = l2_key {
            model.ffn_l2_cache.insert(layer, key, output.clone());
        }

        entries.push(FfnEntry { layer, output });
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(FfnOutput { entries, seq_len, latency_ms })
}

fn run_full_output(
    model: &LoadedModel,
    req: &WalkFfnRequest,
    scan_layers: &[usize],
    start: std::time::Instant,
) -> Result<serde_json::Value, ServerError> {
    let out = run_full_output_core(model, req, scan_layers, start)?;
    Ok(encode_json_full_output(&out))
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

fn run_walk_ffn(
    state: &AppState,
    req: &WalkFfnRequest,
) -> Result<serde_json::Value, ServerError> {
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;

    let hidden = model.config.hidden_size;
    validate_residual(req, hidden)?;

    let scan_layers = collect_scan_layers(req)?;
    validate_owned(model, &scan_layers)?;

    let start = std::time::Instant::now();

    if req.full_output {
        run_full_output(model, req, &scan_layers, start)
    } else {
        run_features_only(model, req, &scan_layers, start)
    }
}

// ── HTTP handler ──────────────────────────────────────────────────────────────

pub async fn handle_walk_ffn(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Result<Response, ServerError> {
    state.bump_requests();

    let is_binary = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.starts_with(BINARY_CT))
        .unwrap_or(false);

    let body = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| ServerError::BadRequest(format!("read body: {e}")))?;

    if is_binary {
        let req = decode_binary_request(&body)?;
        if !req.full_output {
            return Err(ServerError::BadRequest(
                "binary wire format requires full_output = true".into(),
            ));
        }
        let result = tokio::task::spawn_blocking(move || {
            let model = state
                .model(None)
                .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
            validate_residual(&req, model.config.hidden_size)?;
            let scan_layers = collect_scan_layers(&req)?;
            validate_owned(model, &scan_layers)?;
            let start = std::time::Instant::now();
            let out = run_full_output_core(model, &req, &scan_layers, start)?;
            if model.release_mmap_after_request {
                let patched = model.patched.blocking_read();
                patched.base().release_mmap_pages();
            }
            Ok::<_, ServerError>(out)
        })
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;

        let bytes = encode_binary_output(&result);
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, BINARY_CT)
            .body(axum::body::Body::from(bytes))
            .unwrap());
    }

    // JSON path — original behaviour preserved.
    let req: WalkFfnRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("invalid JSON: {e}")))?;

    let result = tokio::task::spawn_blocking(move || {
        let result = run_walk_ffn(&state, &req)?;
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

    let json_bytes = serde_json::to_vec(&result)
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(json_bytes))
        .unwrap())
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── decode_binary_request ─────────────────────────────────────────────────

    fn make_single_binary(
        layer: u32,
        seq_len: u32,
        full_output: bool,
        top_k: u32,
        residual: &[f32],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&(full_output as u32).to_le_bytes());
        buf.extend_from_slice(&top_k.to_le_bytes());
        for &v in residual {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn make_batch_binary(
        layers: &[u32],
        seq_len: u32,
        full_output: bool,
        top_k: u32,
        residual: &[f32],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(layers.len() as u32).to_le_bytes());
        for &l in layers {
            buf.extend_from_slice(&l.to_le_bytes());
        }
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&(full_output as u32).to_le_bytes());
        buf.extend_from_slice(&top_k.to_le_bytes());
        for &v in residual {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn decode_single_layer_request() {
        let residual = vec![0.1f32, -0.2, 0.3, 0.4];
        let body = make_single_binary(7, 1, true, 256, &residual);
        let req = decode_binary_request(&body).unwrap();
        assert_eq!(req.layer, Some(7));
        assert!(req.layers.is_none());
        assert_eq!(req.seq_len, 1);
        assert!(req.full_output);
        assert_eq!(req.top_k, 256);
        assert_eq!(req.residual.len(), 4);
        assert!((req.residual[0] - 0.1f32).abs() < 1e-6);
        assert!((req.residual[1] - (-0.2f32)).abs() < 1e-6);
    }

    #[test]
    fn decode_batch_request() {
        let residual = vec![1.0f32, 2.0, 3.0, 4.0];
        let body = make_batch_binary(&[5, 20, 30], 1, true, 512, &residual);
        let req = decode_binary_request(&body).unwrap();
        assert!(req.layer.is_none());
        assert_eq!(req.layers.as_deref(), Some([5, 20, 30].as_slice()));
        assert!(req.full_output);
        assert_eq!(req.top_k, 512);
        assert_eq!(req.residual.len(), 4);
    }

    #[test]
    fn decode_features_only_binary() {
        let residual = vec![1.0f32, 0.0, 0.0, 0.0];
        let body = make_single_binary(3, 1, false, 8092, &residual);
        let req = decode_binary_request(&body).unwrap();
        assert!(!req.full_output);
    }

    #[test]
    fn decode_binary_truncated_body() {
        let result = decode_binary_request(&[0u8; 4]);
        assert!(result.is_err(), "should fail on truncated body");
    }

    #[test]
    fn decode_binary_empty_body() {
        let result = decode_binary_request(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_binary_batch_truncated_layers() {
        // Claims 10 layers but only provides 2.
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&10u32.to_le_bytes()); // num_layers = 10
        buf.extend_from_slice(&0u32.to_le_bytes()); // only 1 layer provided
        buf.extend_from_slice(&0u32.to_le_bytes()); // padding
        let result = decode_binary_request(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn decode_binary_odd_residual_length() {
        // Residual bytes not a multiple of 4.
        let mut body = make_single_binary(0, 1, true, 8092, &[1.0, 2.0]);
        body.push(0xff); // extra byte → not multiple of 4
        let result = decode_binary_request(&body);
        assert!(result.is_err());
    }

    // ── encode_binary_output ──────────────────────────────────────────────────

    #[test]
    fn encode_single_entry_output() {
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 5,
                output: vec![1.0f32, -2.0, 3.5],
            }],
            seq_len: 1,
            latency_ms: 7.3,
        };
        let bytes = encode_binary_output(&out);
        // Single: [layer u32][seq_len u32][latency f32][output f32*3]
        assert_eq!(bytes.len(), 4 + 4 + 4 + 3 * 4);
        let layer = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let seq_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let latency = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(layer, 5);
        assert_eq!(seq_len, 1);
        assert!((latency - 7.3f32).abs() < 0.01);
        let v0 = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert!((v0 - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn encode_batch_output() {
        let out = FfnOutput {
            entries: vec![
                FfnEntry { layer: 5, output: vec![1.0f32, 2.0] },
                FfnEntry { layer: 20, output: vec![3.0f32, 4.0] },
            ],
            seq_len: 1,
            latency_ms: 15.0,
        };
        let bytes = encode_binary_output(&out);
        let marker = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num_results = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(num_results, 2);
        let latency = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert!((latency - 15.0f32).abs() < 0.01);
        // First entry
        let layer0 = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(layer0, 5);
        let num_floats0 = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(num_floats0, 2);
    }

    #[test]
    fn binary_roundtrip_float_preservation() {
        let original_output = vec![0.12345f32, -9.87654, 1e-7, f32::MAX / 2.0];
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 10,
                output: original_output.clone(),
            }],
            seq_len: 1,
            latency_ms: 1.0,
        };
        let bytes = encode_binary_output(&out);
        // Decode back
        let decoded_floats: Vec<f32> = bytes[12..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_floats.len(), original_output.len());
        for (a, b) in decoded_floats.iter().zip(original_output.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "float bits differ: {a} vs {b}");
        }
    }

    // ── encode_json_full_output ───────────────────────────────────────────────

    #[test]
    fn json_single_layer_format() {
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 26,
                output: vec![0.1f32, 0.2],
            }],
            seq_len: 1,
            latency_ms: 10.0,
        };
        let v = encode_json_full_output(&out);
        assert_eq!(v["layer"].as_u64(), Some(26));
        assert_eq!(v["seq_len"].as_u64(), Some(1));
        assert!(v.get("output").is_some());
        assert!(v.get("latency_ms").is_some());
        assert!(v.get("results").is_none());
    }

    #[test]
    fn json_batch_format() {
        let out = FfnOutput {
            entries: vec![
                FfnEntry { layer: 0, output: vec![1.0f32] },
                FfnEntry { layer: 1, output: vec![2.0f32] },
            ],
            seq_len: 2,
            latency_ms: 20.0,
        };
        let v = encode_json_full_output(&out);
        assert!(v.get("results").is_some());
        let results = v["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["layer"].as_u64(), Some(0));
    }
}
