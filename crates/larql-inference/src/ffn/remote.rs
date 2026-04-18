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

use std::time::Duration;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::ffn::FfnBackend;

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

    /// Hidden size advertised by the remote server. Use this to size local
    /// tensors before the first `forward()` call.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Single-layer FFN call. Returns a `Vec<f32>` of length
    /// `seq_len * hidden_size`, row-major.
    fn call_single(
        &self,
        layer: usize,
        residual_flat: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, RemoteFfnError> {
        let url = format!("{}/v1/walk-ffn", self.config.base_url);
        let req = WalkFfnHttpRequest {
            layer: Some(layer),
            layers: None,
            residual: residual_flat.to_vec(),
            seq_len,
            full_output: true,
        };

        let resp = self
            .client
            .post(&url)
            .json(&req)
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

        let parsed: WalkFfnSingleResponse = resp
            .json()
            .map_err(|e| RemoteFfnError::BadResponse(e.to_string()))?;

        let expected = seq_len * self.hidden_size;
        if parsed.output.len() != expected {
            return Err(RemoteFfnError::BadResponse(format!(
                "layer {layer}: expected {expected} output floats, got {}",
                parsed.output.len()
            )));
        }
        Ok(parsed.output)
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
                // FfnBackend::forward has no Result in its signature; all
                // local backends panic on catastrophic failure. Match that
                // contract. Callers wanting graceful remote-outage handling
                // should use `call_single` directly.
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
        // The server only returns the final FFN output. The pre-down
        // activation is not on the wire (deliberately — it's the largest
        // intermediate, `[seq_len, intermediate_size]`). Callers that need
        // the activation must use a local backend.
        let out = self.forward(layer, x);
        let seq_len = x.shape()[0];
        let zeros = Array2::<f32>::zeros((seq_len, 1));
        (out, zeros)
    }

    fn name(&self) -> &str {
        "remote-walk"
    }
}

// ── wire types ──────────────────────────────────────────────────────────

#[derive(Serialize)]
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

// ── error type ──────────────────────────────────────────────────────────

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

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

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
        // `layers: None` should not be emitted.
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
        // Use a reserved port that should never respond.
        // Short timeout so the test doesn't wait the default 60 s.
        let cfg =
            RemoteFfnConfig::new("http://127.0.0.1:1").with_timeout(Duration::from_millis(500));
        // `RemoteWalkBackend` holds a `reqwest::blocking::Client` which is
        // !Debug, so we can't use `.unwrap_err()`. Match explicitly.
        match RemoteWalkBackend::connect(cfg) {
            Ok(_) => panic!("expected connect to fail against 127.0.0.1:1"),
            Err(RemoteFfnError::Unreachable { url, .. }) => {
                assert!(url.contains("127.0.0.1:1"));
            }
            Err(other) => panic!("expected Unreachable, got {other:?}"),
        }
    }
}
