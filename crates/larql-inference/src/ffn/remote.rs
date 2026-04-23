//! RemoteWalkBackend — FFN backend that dispatches to a `larql-server` over
//! HTTP instead of computing locally.
//!
//! Implements the same [`FfnBackend`] trait as [`WalkFfn`], so it slots into
//! `predict_with_ffn` and the rest of the forward-pass code with zero
//! changes.
//!
//! Wire protocol: POST `/v1/walk-ffn` with `full_output: true`. The server
//! runs the architecture-correct WalkFfn path (gate KNN → activation → up
//! gather → down projection) and returns the hidden-size FFN output per
//! layer. See [`crate::ffn::FfnBackend`] for the trait and
//! `crates/larql-server/src/routes/walk_ffn.rs` for the endpoint.
//!
//! The residual is sent row-major as `seq_len × hidden` floats; output
//! mirrors the shape. One HTTP round trip per `forward()` call.
//!
//! # Wire format
//!
//! By default `RemoteWalkBackend` uses the binary wire format
//! (`Content-Type: application/x-larql-ffn`), which eliminates JSON float
//! serialization overhead (~0.5 ms/hop on a Gemma 3 4B hidden layer).
//!
//! ## Binary request — single layer
//! ```text
//! 0       4     layer_index (u32 LE)
//! 4       4     seq_len (u32 LE)
//! 8       4     flags (u32 LE, bit 0 = full_output = 1)
//! 12      4     top_k (u32 LE, unused in full_output mode)
//! 16      N×4   residual (f32[] LE)
//! ```
//!
//! ## Binary request — batch
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
//! ## Binary response — single layer
//! ```text
//! 0       4     layer (u32 LE)
//! 4       4     seq_len (u32 LE)
//! 8       4     latency_ms (f32 LE)
//! 12      N×4   output (f32[] LE)
//! ```
//!
//! ## Binary response — batch
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

use std::collections::HashMap;
use std::time::Duration;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::ffn::FfnBackend;

const BINARY_CT: &str = "application/x-larql-ffn";
const BATCH_MARKER: u32 = 0xFFFF_FFFF;

/// Client config for talking to a remote FFN server.
#[derive(Clone, Debug)]
pub struct RemoteFfnConfig {
    /// Base URL, e.g. `"https://ffn.example.com:8080"`. Trailing slash
    /// stripped automatically.
    pub base_url: String,
    /// Per-request timeout. Applied to both connect and read.
    pub timeout: Duration,
}

impl RemoteFfnConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            timeout: Duration::from_secs(60),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Remote FFN backend. Holds a blocking HTTP client plus the server URL.
///
/// Cloning is cheap — the underlying `reqwest::blocking::Client` is
/// connection-pooled and `Arc`-shared.
pub struct RemoteWalkBackend {
    config: RemoteFfnConfig,
    client: reqwest::blocking::Client,
    hidden_size: usize,
}

impl RemoteWalkBackend {
    /// Build a backend. Performs a one-shot health check against
    /// `/v1/stats` so we fail fast if the server is unreachable at
    /// construction time rather than mid-forward-pass.
    pub fn connect(config: RemoteFfnConfig) -> Result<Self, RemoteFfnError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| RemoteFfnError::Client(e.to_string()))?;

        let stats_url = format!("{}/v1/stats", config.base_url);
        let resp = client.get(&stats_url).send().map_err(|e| {
            RemoteFfnError::Unreachable {
                url: stats_url.clone(),
                cause: e.to_string(),
            }
        })?;
        if !resp.status().is_success() {
            return Err(RemoteFfnError::ServerError {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default(),
            });
        }
        let stats: serde_json::Value = resp
            .json()
            .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
        let hidden_size = stats["hidden_size"].as_u64().ok_or_else(|| {
            RemoteFfnError::BadResponse("stats missing hidden_size".into())
        })? as usize;

        Ok(Self { config, client, hidden_size })
    }

    /// Hidden size advertised by the remote server.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Single-layer FFN call using the binary wire format.
    /// Returns a `Vec<f32>` of length `seq_len * hidden_size`, row-major.
    fn call_single(
        &self,
        layer: usize,
        residual_flat: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, RemoteFfnError> {
        let url = format!("{}/v1/walk-ffn", self.config.base_url);
        let body = encode_binary_request(Some(layer), None, residual_flat, seq_len, true, 8092);

        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, BINARY_CT)
            .body(body)
            .send()
            .map_err(|e| RemoteFfnError::Http {
                layer,
                cause: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(RemoteFfnError::ServerError {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default(),
            });
        }

        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let resp_bytes = resp
            .bytes()
            .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;

        let output = if ct.starts_with(BINARY_CT) {
            let (_, floats) = decode_binary_single(&resp_bytes)
                .map_err(RemoteFfnError::BadResponse)?;
            floats
        } else {
            // Fallback: server returned JSON.
            let parsed: WalkFfnSingleResponse = serde_json::from_slice(&resp_bytes)
                .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
            parsed.output
        };

        let expected = seq_len * self.hidden_size;
        if output.len() != expected {
            return Err(RemoteFfnError::BadResponse(format!(
                "layer {layer}: expected {expected} output floats, got {}",
                output.len()
            )));
        }
        Ok(output)
    }

    /// Batch FFN call — sends all `layers` in one round trip using the binary
    /// wire format. Returns a map from layer index to output floats.
    ///
    /// The server must serve all requested layers (i.e. they must all be in
    /// the same shard). For cross-shard batches, route through `larql-router`
    /// using JSON.
    pub fn call_batch(
        &self,
        layers: &[usize],
        residual_flat: &[f32],
        seq_len: usize,
    ) -> Result<HashMap<usize, Vec<f32>>, RemoteFfnError> {
        let url = format!("{}/v1/walk-ffn", self.config.base_url);
        let body =
            encode_binary_request(None, Some(layers), residual_flat, seq_len, true, 8092);

        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, BINARY_CT)
            .body(body)
            .send()
            .map_err(|e| RemoteFfnError::Http {
                layer: layers.first().copied().unwrap_or(0),
                cause: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(RemoteFfnError::ServerError {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default(),
            });
        }

        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let resp_bytes = resp
            .bytes()
            .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;

        if ct.starts_with(BINARY_CT) {
            decode_binary_batch(&resp_bytes).map_err(RemoteFfnError::BadResponse)
        } else {
            // Fallback: JSON batch response.
            let v: serde_json::Value = serde_json::from_slice(&resp_bytes)
                .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
            let mut out = HashMap::new();
            // Single-layer JSON response.
            if let Some(layer) = v.get("layer").and_then(|l| l.as_u64()) {
                let floats = json_output_floats(&v)?;
                out.insert(layer as usize, floats);
                return Ok(out);
            }
            // Multi-layer JSON response.
            if let Some(results) = v.get("results").and_then(|r| r.as_array()) {
                for entry in results {
                    let layer = entry["layer"].as_u64().ok_or_else(|| {
                        RemoteFfnError::BadResponse("batch JSON: missing layer".into())
                    })? as usize;
                    let floats = json_output_floats(entry)?;
                    out.insert(layer, floats);
                }
                return Ok(out);
            }
            Err(RemoteFfnError::BadResponse(
                "batch response has neither 'layer' nor 'results'".into(),
            ))
        }
    }

    /// Measure round-trip latency breakdown over `n` calls.
    ///
    /// Sends a zero residual batch covering `layers` each time and reports:
    /// - `total_ms`: wall-clock time measured by the client
    /// - `server_ms`: compute time reported by the server in the response header
    /// - `overhead_ms`: `total_ms - server_ms` (HTTP + TCP + framing)
    ///
    /// First call is a warmup (excluded from stats). Results are averaged over
    /// the remaining `n - 1` calls.
    pub fn probe_latency(
        &self,
        layers: &[usize],
        n: usize,
    ) -> Result<RemoteLatencyStats, RemoteFfnError> {
        assert!(n >= 2, "probe_latency: need at least 2 calls (1 warmup + 1 measured)");
        let residual = vec![0.0f32; self.hidden_size];
        let url = format!("{}/v1/walk-ffn", self.config.base_url);
        let body = encode_binary_request(None, Some(layers), &residual, 1, true, 8092);

        let mut totals = Vec::with_capacity(n - 1);
        let mut servers = Vec::with_capacity(n - 1);

        for i in 0..n {
            let t0 = std::time::Instant::now();
            let resp = self
                .client
                .post(&url)
                .header(reqwest::header::CONTENT_TYPE, BINARY_CT)
                .body(body.clone())
                .send()
                .map_err(|e| RemoteFfnError::Http { layer: layers[0], cause: e.to_string() })?;
            if !resp.status().is_success() {
                return Err(RemoteFfnError::ServerError {
                    status: resp.status().as_u16(),
                    body: resp.text().unwrap_or_default(),
                });
            }
            let resp_bytes =
                resp.bytes().map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // Extract server-reported latency from bytes 8-11 of response.
            let server_ms = extract_response_latency_ms(&resp_bytes);

            if i > 0 {
                // Skip warmup call.
                totals.push(total_ms);
                servers.push(server_ms);
            }
        }

        let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let total_ms = avg(&totals);
        let server_ms = avg(&servers);
        Ok(RemoteLatencyStats {
            total_ms,
            server_ms,
            overhead_ms: total_ms - server_ms,
            hidden_size: self.hidden_size,
            num_layers: layers.len(),
            samples: n - 1,
        })
    }

    /// Run the full FFN forward pass for every layer in `layers`, returning
    /// a map from layer → `Array2<f32>` shaped `[seq_len, hidden]`.
    ///
    /// All layers are sent in a single HTTP round trip (binary batch format).
    pub fn forward_all_layers(
        &self,
        layers: &[usize],
        x: &Array2<f32>,
    ) -> Result<HashMap<usize, Array2<f32>>, RemoteFfnError> {
        let seq_len = x.shape()[0];
        let hidden = x.shape()[1];
        assert_eq!(
            hidden, self.hidden_size,
            "RemoteWalkBackend: input hidden {hidden} != server hidden {}",
            self.hidden_size
        );
        let residual_flat: Vec<f32> = x.iter().copied().collect();
        let flat_map = self.call_batch(layers, &residual_flat, seq_len)?;
        let mut result = HashMap::with_capacity(flat_map.len());
        for (layer, floats) in flat_map {
            if floats.len() != seq_len * hidden {
                return Err(RemoteFfnError::BadResponse(format!(
                    "layer {layer}: expected {} output floats, got {}",
                    seq_len * hidden,
                    floats.len()
                )));
            }
            let arr = Array2::from_shape_vec((seq_len, hidden), floats)
                .expect("shape validated above");
            result.insert(layer, arr);
        }
        Ok(result)
    }
}

impl FfnBackend for RemoteWalkBackend {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.shape()[0];
        let hidden = x.shape()[1];
        assert_eq!(
            hidden, self.hidden_size,
            "RemoteWalkBackend: input hidden {hidden} != server hidden {}",
            self.hidden_size
        );

        let residual_flat: Vec<f32> = x.iter().copied().collect();
        let output = self
            .call_single(layer, &residual_flat, seq_len)
            .unwrap_or_else(|e| {
                panic!("RemoteWalkBackend layer {layer}: {e}")
            });

        Array2::from_shape_vec((seq_len, hidden), output)
            .expect("RemoteWalkBackend: server output shape mismatch (validated above)")
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let out = self.forward(layer, x);
        let seq_len = x.shape()[0];
        let zeros = Array2::<f32>::zeros((seq_len, 1));
        (out, zeros)
    }

    fn name(&self) -> &str {
        "remote-walk"
    }
}

// ── Latency profiling ────────────────────────────────────────────────────────

/// Breakdown returned by [`RemoteWalkBackend::probe_latency`].
#[derive(Debug, Clone)]
pub struct RemoteLatencyStats {
    /// Wall-clock round-trip (client-measured), averaged over `samples` calls.
    pub total_ms: f64,
    /// FFN compute time reported by the server in the binary response header.
    pub server_ms: f64,
    /// `total_ms - server_ms`: HTTP framing + TCP + serialization overhead.
    pub overhead_ms: f64,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub samples: usize,
}

impl std::fmt::Display for RemoteLatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "layers={} hidden={} samples={}\n  total    {:7.2} ms\n  server   {:7.2} ms  (FFN compute)\n  overhead {:7.2} ms  (HTTP + TCP + framing)",
            self.num_layers, self.hidden_size, self.samples,
            self.total_ms, self.server_ms, self.overhead_ms,
        )
    }
}

/// Extract the `latency_ms` f32 embedded at bytes 8-11 of a binary response.
/// Returns 0.0 if the body is too short or the value is non-finite.
fn extract_response_latency_ms(body: &[u8]) -> f64 {
    if body.len() < 12 {
        return 0.0;
    }
    // Both single-layer and batch responses have latency_ms at offset 8.
    let v = f32::from_le_bytes(body[8..12].try_into().unwrap());
    if v.is_finite() { v as f64 } else { 0.0 }
}

// ── Binary codec ──────────────────────────────────────────────────────────────

/// Encode a request as binary.
/// `layer` and `layers` are mutually exclusive; pass `None` for the unused one.
pub(crate) fn encode_binary_request(
    layer: Option<usize>,
    layers: Option<&[usize]>,
    residual: &[f32],
    seq_len: usize,
    full_output: bool,
    top_k: usize,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16 + residual.len() * 4);

    if let Some(ls) = layers {
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(ls.len() as u32).to_le_bytes());
        for &l in ls {
            buf.extend_from_slice(&(l as u32).to_le_bytes());
        }
    } else {
        let l = layer.unwrap_or(0) as u32;
        buf.extend_from_slice(&l.to_le_bytes());
    }

    buf.extend_from_slice(&(seq_len as u32).to_le_bytes());
    buf.extend_from_slice(&(full_output as u32).to_le_bytes());
    buf.extend_from_slice(&(top_k as u32).to_le_bytes());
    for &v in residual {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Decode a binary single-layer full_output response.
/// Returns `(layer, output_floats)`.
pub(crate) fn decode_binary_single(body: &[u8]) -> Result<(usize, Vec<f32>), String> {
    if body.len() < 12 {
        return Err(format!("binary response too short: {} bytes", body.len()));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer response but got batch marker".into());
    }
    let layer = marker as usize;
    // bytes 4-7: seq_len (ignored here — caller validates against expected shape)
    // bytes 8-11: latency f32
    let floats: Vec<f32> = body[12..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((layer, floats))
}

/// Decode a binary batch full_output response.
/// Returns a map from layer → output floats.
pub(crate) fn decode_binary_batch(body: &[u8]) -> Result<HashMap<usize, Vec<f32>>, String> {
    if body.len() < 12 {
        return Err(format!("binary batch response too short: {} bytes", body.len()));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());

    // Single-layer response — accept it as a batch of 1.
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single(body)?;
        let mut m = HashMap::new();
        m.insert(layer, floats);
        return Ok(m);
    }

    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    // bytes 8-11: latency f32 (skip)
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity(num_results);

    for _ in 0..num_results {
        if body.len() < offset + 12 {
            return Err("binary batch: truncated result header".into());
        }
        let layer = u32::from_le_bytes(body[offset..offset + 4].try_into().unwrap()) as usize;
        // offset+4: seq_len (skip)
        let num_floats =
            u32::from_le_bytes(body[offset + 8..offset + 12].try_into().unwrap()) as usize;
        offset += 12;
        let bytes_needed = num_floats * 4;
        if body.len() < offset + bytes_needed {
            return Err(format!(
                "binary batch: truncated output for layer {layer}: need {bytes_needed}, have {}",
                body.len() - offset
            ));
        }
        let floats: Vec<f32> = body[offset..offset + bytes_needed]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        offset += bytes_needed;
        out.insert(layer, floats);
    }
    Ok(out)
}

// ── JSON fallback helpers ─────────────────────────────────────────────────────

fn json_output_floats(v: &serde_json::Value) -> Result<Vec<f32>, RemoteFfnError> {
    v.get("output")
        .and_then(|o| o.as_array())
        .ok_or_else(|| RemoteFfnError::BadResponse("missing 'output' array".into()))
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect()
        })
}

// ── wire types (JSON fallback) ────────────────────────────────────────────────

#[derive(Serialize)]
#[allow(dead_code)]
struct WalkFfnHttpRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    layers: Option<Vec<usize>>,
    residual: Vec<f32>,
    seq_len: usize,
    full_output: bool,
}

#[derive(Deserialize)]
struct WalkFfnSingleResponse {
    #[allow(dead_code)]
    layer: usize,
    output: Vec<f32>,
    #[allow(dead_code)]
    seq_len: usize,
}

// ── error type ────────────────────────────────────────────────────────────────

#[derive(thiserror::Error, Debug)]
pub enum RemoteFfnError {
    #[error("remote FFN client setup failed: {0}")]
    Client(String),

    #[error("remote FFN server unreachable at {url}: {cause}")]
    Unreachable { url: String, cause: String },

    #[error("remote FFN HTTP call for layer {layer} failed: {cause}")]
    Http { layer: usize, cause: String },

    #[error("remote FFN server returned {status}: {body}")]
    ServerError { status: u16, body: String },

    #[error("remote FFN bad response: {0}")]
    BadResponse(String),
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── RemoteFfnConfig ───────────────────────────────────────────────────────

    #[test]
    fn config_strips_trailing_slash() {
        let c = RemoteFfnConfig::new("https://example.com:8080/");
        assert_eq!(c.base_url, "https://example.com:8080");
    }

    #[test]
    fn config_strips_multiple_trailing_slashes() {
        let c = RemoteFfnConfig::new("https://example.com:8080///");
        assert_eq!(c.base_url, "https://example.com:8080");
    }

    #[test]
    fn config_preserves_url_without_trailing_slash() {
        let c = RemoteFfnConfig::new("http://127.0.0.1:8080");
        assert_eq!(c.base_url, "http://127.0.0.1:8080");
    }

    #[test]
    fn config_default_timeout_is_nontrivial() {
        let c = RemoteFfnConfig::new("http://x");
        assert!(c.timeout.as_secs() >= 10);
    }

    #[test]
    fn config_with_timeout_overrides_default() {
        let c = RemoteFfnConfig::new("http://x").with_timeout(Duration::from_secs(5));
        assert_eq!(c.timeout.as_secs(), 5);
    }

    // ── JSON serialisation (unchanged) ────────────────────────────────────────

    #[test]
    fn request_serializes_with_seq_len_and_full_output() {
        let req = WalkFfnHttpRequest {
            layer: Some(3),
            layers: None,
            residual: vec![0.1, -0.2, 0.3, 0.4],
            seq_len: 2,
            full_output: true,
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["layer"], 3);
        assert_eq!(v["seq_len"], 2);
        assert_eq!(v["full_output"], true);
        assert!(
            v.get("layers").is_none() || v["layers"].is_null(),
            "layers should not appear when None, got: {v}"
        );
        assert_eq!(v["residual"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn response_deserializes_hidden_vector() {
        let json = serde_json::json!({
            "layer": 5,
            "output": [0.1, 0.2, 0.3, 0.4, 0.5],
            "seq_len": 1,
            "latency_ms": 2.5,
        });
        let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.layer, 5);
        assert_eq!(parsed.output.len(), 5);
        assert_eq!(parsed.seq_len, 1);
    }

    #[test]
    fn response_deserializes_multi_token_output() {
        let flat: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let json = serde_json::json!({
            "layer": 0,
            "output": flat,
            "seq_len": 3,
        });
        let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.output.len(), 12);
        assert_eq!(parsed.seq_len, 3);
    }

    #[test]
    fn error_display_messages_are_actionable() {
        let e = RemoteFfnError::Unreachable {
            url: "http://nope:1234".into(),
            cause: "connection refused".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("http://nope:1234"));
        assert!(s.contains("connection refused"));

        let e = RemoteFfnError::Http {
            layer: 7,
            cause: "timed out".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("layer 7"));
        assert!(s.contains("timed out"));

        let e = RemoteFfnError::ServerError {
            status: 503,
            body: "service unavailable".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("503"));
        assert!(s.contains("service unavailable"));
    }

    #[test]
    fn connect_fails_fast_on_unreachable_url() {
        let cfg =
            RemoteFfnConfig::new("http://127.0.0.1:1").with_timeout(Duration::from_millis(500));
        match RemoteWalkBackend::connect(cfg) {
            Ok(_) => panic!("expected connect to fail against 127.0.0.1:1"),
            Err(RemoteFfnError::Unreachable { url, .. }) => {
                assert!(url.contains("127.0.0.1:1"));
            }
            Err(other) => panic!("expected Unreachable, got {other:?}"),
        }
    }

    // ── encode_binary_request ─────────────────────────────────────────────────

    #[test]
    fn encode_single_layer_header() {
        let residual = vec![1.0f32, 2.0, 3.0, 4.0];
        let body = encode_binary_request(Some(7), None, &residual, 1, true, 256);
        // First u32 = layer index
        let layer = u32::from_le_bytes(body[0..4].try_into().unwrap());
        assert_eq!(layer, 7);
        let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
        assert_eq!(seq_len, 1);
        let flags = u32::from_le_bytes(body[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 1); // full_output
        let top_k = u32::from_le_bytes(body[12..16].try_into().unwrap());
        assert_eq!(top_k, 256);
        assert_eq!(body.len(), 16 + 4 * 4);
    }

    #[test]
    fn encode_batch_header() {
        let residual = vec![0.5f32; 4];
        let body = encode_binary_request(None, Some(&[5, 20, 30]), &residual, 1, true, 512);
        let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num_layers = u32::from_le_bytes(body[4..8].try_into().unwrap());
        assert_eq!(num_layers, 3);
        let l0 = u32::from_le_bytes(body[8..12].try_into().unwrap());
        let l1 = u32::from_le_bytes(body[12..16].try_into().unwrap());
        let l2 = u32::from_le_bytes(body[16..20].try_into().unwrap());
        assert_eq!((l0, l1, l2), (5, 20, 30));
    }

    #[test]
    fn encode_residual_values_preserved() {
        let residual = vec![-1.5f32, 0.0, 3.25];
        let body = encode_binary_request(Some(0), None, &residual, 1, true, 8092);
        let offset = 16; // 4 header u32s × 4 bytes
        let v0 = f32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
        let v1 = f32::from_le_bytes(body[offset + 4..offset + 8].try_into().unwrap());
        let v2 = f32::from_le_bytes(body[offset + 8..offset + 12].try_into().unwrap());
        assert_eq!(v0.to_bits(), (-1.5f32).to_bits());
        assert_eq!(v1.to_bits(), 0.0f32.to_bits());
        assert!((v2 - 3.25f32).abs() < 1e-5);
    }

    // ── decode_binary_single ──────────────────────────────────────────────────

    fn make_single_response(layer: u32, seq_len: u32, latency: f32, output: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&latency.to_le_bytes());
        for &v in output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn make_batch_response(latency: f32, entries: &[(u32, &[f32])]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        buf.extend_from_slice(&latency.to_le_bytes());
        for &(layer, floats) in entries {
            buf.extend_from_slice(&layer.to_le_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
            buf.extend_from_slice(&(floats.len() as u32).to_le_bytes());
            for &v in floats {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn decode_single_response_correct() {
        let output = vec![1.0f32, -2.0, 3.5];
        let body = make_single_response(5, 1, 7.3, &output);
        let (layer, floats) = decode_binary_single(&body).unwrap();
        assert_eq!(layer, 5);
        assert_eq!(floats.len(), 3);
        assert!((floats[0] - 1.0).abs() < 1e-6);
        assert!((floats[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn decode_single_response_rejects_batch_marker() {
        let body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
        let result = decode_binary_single(&body);
        assert!(result.is_err());
    }

    #[test]
    fn decode_single_response_too_short() {
        let result = decode_binary_single(&[0u8; 8]);
        assert!(result.is_err());
    }

    // ── decode_binary_batch ───────────────────────────────────────────────────

    #[test]
    fn decode_batch_response_correct() {
        let body = make_batch_response(
            15.0,
            &[(5, &[1.0, 2.0]), (20, &[3.0, 4.0])],
        );
        let map = decode_binary_batch(&body).unwrap();
        assert_eq!(map.len(), 2);
        let v5 = map.get(&5).unwrap();
        assert_eq!(v5.len(), 2);
        assert!((v5[0] - 1.0).abs() < 1e-6);
        let v20 = map.get(&20).unwrap();
        assert!((v20[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn decode_batch_accepts_single_response() {
        // A server returning single-layer response to a same-shard batch.
        let output = vec![7.0f32, 8.0];
        let body = make_single_response(10, 1, 5.0, &output);
        let map = decode_binary_batch(&body).unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&10));
    }

    #[test]
    fn decode_batch_truncated_returns_error() {
        let mut body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
        body.truncate(body.len() - 4); // cut off last float
        let result = decode_binary_batch(&body);
        assert!(result.is_err());
    }

    #[test]
    fn binary_request_response_roundtrip() {
        // Encode a single-layer request, then simulate what the server echoes.
        let residual = vec![0.1f32, 0.2, 0.3, 0.4];
        let req = encode_binary_request(Some(5), None, &residual, 1, true, 8092);
        // Simulate server extracting the layer.
        let layer = u32::from_le_bytes(req[0..4].try_into().unwrap());
        assert_eq!(layer, 5);

        // Simulate server response.
        let output = vec![0.9f32, 0.8, 0.7, 0.6];
        let resp = make_single_response(layer, 1, 8.5, &output);
        let (resp_layer, floats) = decode_binary_single(&resp).unwrap();
        assert_eq!(resp_layer as u32, layer);
        assert_eq!(floats, output);
    }
}
