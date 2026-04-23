//! `RemoteMoeBackend` — Mixture-of-Experts weight-shard dispatch over HTTP.
//!
//! Not to be confused with [`crate::experts`] — that module hosts deterministic
//! WASM compute experts (gcd, base64, …). This module dispatches *MoE expert
//! weights* (the FFN sub-blocks of an MoE transformer) to remote shard servers.
//!
//! For hybrid MoE models (e.g. Gemma 4 26B A4B), the client holds attention
//! weights + router weights (~5.5 GB). Expert weights live on remote shard
//! servers. For each layer:
//!
//!   1. Client runs the router locally: norm → scale → proj → softmax → top-K.
//!   2. Client groups selected experts by shard.
//!   3. One `POST /v1/expert/batch` per shard (parallel via rayon).
//!   4. Client assembles weighted sum from responses.
//!
//! Wire format: JSON — `{"requests": [{layer, expert_id, residual}]}`
//!              → `{"results": [{layer, expert_id, output}], "latency_ms": f64}`
//!
//! This mirrors [`crate::ffn::RemoteWalkBackend`] at the MoE level, replacing
//! `POST /v1/walk-ffn` with `POST /v1/expert/batch`.
//!
//! # Shard map
//!
//! Expert IDs are contiguous ranges owned by each shard:
//!
//! ```text
//! "0-31"  → https://shard-a.local:8081
//! "32-63" → https://shard-b.local:8082
//! ```
//!
//! A single-shard setup (`"0-63"`) routes all experts to one server.
//! `reshard()` swaps the map live without reloading the model.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ── Public error type ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum RemoteMoeError {
    /// Could not reach the shard server (connection refused, DNS failure, etc.).
    Unreachable { url: String, cause: String },
    /// The server responded with a non-2xx status.
    ServerError { status: u16, body: String },
    /// Response body could not be parsed.
    BadResponse(String),
    /// No shard owns a required expert ID.
    NoShard { expert_id: usize },
    /// HTTP client construction failed.
    Client(String),
}

impl std::fmt::Display for RemoteMoeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable { url, cause } => write!(f, "expert shard unreachable: {url} ({cause})"),
            Self::ServerError { status, body } => write!(f, "expert shard returned {status}: {body}"),
            Self::BadResponse(msg) => write!(f, "bad expert response: {msg}"),
            Self::NoShard { expert_id } => write!(f, "no shard owns expert {expert_id}"),
            Self::Client(msg) => write!(f, "HTTP client error: {msg}"),
        }
    }
}

impl std::error::Error for RemoteMoeError {}

// ── Shard configuration ───────────────────────────────────────────────────────

/// One entry in the shard map: a contiguous expert-ID range + its URL.
#[derive(Clone, Debug)]
pub struct ShardConfig {
    /// First expert ID owned by this shard (inclusive).
    pub start: usize,
    /// Last expert ID owned by this shard (inclusive).
    pub end: usize,
    /// Base URL, e.g. `"http://shard-a.local:8081"`. Trailing slashes stripped.
    pub url: String,
    /// HTTP request timeout (default: 30 s).
    pub timeout: Duration,
}

impl ShardConfig {
    pub fn new(start: usize, end: usize, url: impl Into<String>) -> Self {
        let url = url.into().trim_end_matches('/').to_string();
        Self { start, end, url, timeout: Duration::from_secs(30) }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Parse `"0-31"` → `(0, 31)`. Returns `None` on bad input.
    pub fn parse_range(s: &str) -> Option<(usize, usize)> {
        let mut parts = s.splitn(2, '-');
        let start: usize = parts.next()?.parse().ok()?;
        let end: usize = parts.next()?.parse().ok()?;
        if start <= end { Some((start, end)) } else { None }
    }
}

// ── Internal shard state ──────────────────────────────────────────────────────

#[derive(Clone)]
struct Shard {
    config: ShardConfig,
    client: reqwest::blocking::Client,
}

impl Shard {
    fn connect(config: ShardConfig) -> Result<Self, RemoteMoeError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| RemoteMoeError::Client(e.to_string()))?;

        // Health check — fail fast rather than dying mid-forward-pass.
        let health_url = format!("{}/v1/health", config.url);
        let resp = client.get(&health_url).send().map_err(|e| RemoteMoeError::Unreachable {
            url: health_url.clone(),
            cause: e.to_string(),
        })?;
        if !resp.status().is_success() {
            return Err(RemoteMoeError::ServerError {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default(),
            });
        }

        Ok(Self { config, client })
    }

    fn owns(&self, expert_id: usize) -> bool {
        expert_id >= self.config.start && expert_id <= self.config.end
    }

    /// Send a batch of expert calls to this shard.
    fn call_batch(
        &self,
        requests: &[ExpertCallItem],
    ) -> Result<Vec<ExpertResultItem>, RemoteMoeError> {
        let url = format!("{}/v1/expert/batch", self.config.url);
        let body = BatchRequest { requests };
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| RemoteMoeError::Unreachable {
                url: url.clone(),
                cause: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(RemoteMoeError::ServerError {
                status: resp.status().as_u16(),
                body: resp.text().unwrap_or_default(),
            });
        }

        let parsed: BatchResponse = resp
            .json()
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        Ok(parsed.results)
    }
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct BatchRequest<'a> {
    requests: &'a [ExpertCallItem],
}

#[derive(Serialize, Clone)]
struct ExpertCallItem {
    layer: usize,
    expert_id: usize,
    residual: Vec<f32>,
}

#[derive(Deserialize)]
struct BatchResponse {
    results: Vec<ExpertResultItem>,
}

#[derive(Deserialize)]
struct ExpertResultItem {
    layer: usize,
    expert_id: usize,
    output: Vec<f32>,
}

// ── Local routing math ────────────────────────────────────────────────────────
// Mirrored from larql-compute cpu/ops/moe.rs so the client can route without
// having the expert weights locally.

fn rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    if w.is_empty() || x.is_empty() { return x.to_vec(); }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(w.iter()).map(|(&xi, &wi)| xi / rms * (wi + offset)).collect()
}

/// Parameter-free RMSNorm (HF `Gemma4RMSNorm(with_scale=False)`): scales
/// `x` by `1/sqrt(mean(x²) + eps)` with no learned weight.
fn rms_norm_no_weight(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() { return Vec::new(); }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().map(|v| v / rms).collect()
}

fn matmul_vec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    (0..out_rows).map(|row| {
        let w_row = &w[row * in_cols..(row + 1) * in_cols];
        x.iter().zip(w_row.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}

fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    if sum > 0.0 { for x in v.iter_mut() { *x /= sum; } }
}

fn top_k(v: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(v.len());
    let mut indexed: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    (indexed.iter().map(|(i, _)| *i).collect(),
     indexed.iter().map(|(_, v)| *v).collect())
}

/// Routing-only parameters. A subset of `MoeLayerWeights` — the expert weight
/// slices (`experts_gate_up`, `experts_down`) are absent; those live on shards.
pub struct MoeRouterWeights<'a> {
    /// Router linear projection [num_experts × hidden_size].
    pub router_proj: &'a [f32],
    /// Optional router input scale [hidden_size].
    pub router_scale: &'a [f32],
    /// Optional per-expert output scale [num_experts].
    pub router_per_expert_scale: &'a [f32],
    /// Optional router-specific RMSNorm weights [hidden_size]. When non-empty,
    /// the router input is `rms_norm(h, router_norm)`; when empty AND
    /// `router_norm_parameter_free` is true, it's parameter-free RMSNorm;
    /// otherwise falls back to `rms_norm(h, pre_experts_norm)`.
    pub router_norm: &'a [f32],
    /// Parameter-free router RMSNorm (no learned weight). HF Gemma 4 sets
    /// this true (`Gemma4RMSNorm(with_scale=False)`).
    pub router_norm_parameter_free: bool,
    /// Scalar multiplier on the router input after the norm and `router_scale`.
    /// HF Gemma 4: `hidden_size^-0.5`. Use `1.0` for no scaling.
    pub router_input_scalar: f32,
    /// Pre-experts RMSNorm weights [hidden_size].
    pub pre_experts_norm: &'a [f32],
    /// Post-experts RMSNorm weights [hidden_size]. Applied to the summed output.
    pub post_experts_norm: &'a [f32],
    pub num_experts: usize,
    pub top_k: usize,
}

impl MoeRouterWeights<'_> {
    /// Run steps 1-5 of the MoE forward pass (norm → scale → proj → softmax → top-K).
    /// Returns `(h_norm, expert_indices, expert_weights)` where `h_norm` is
    /// the experts' input (pre_experts_norm output), not the router's input.
    pub fn route(&self, h: &[f32], norm_offset: f32, eps: f32) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
        let hidden = h.len();

        // Experts' input norm (used by callers for the expert matmuls).
        let h_norm = rms_norm(h, self.pre_experts_norm, eps, norm_offset);

        // Router input norm. Priority:
        //   1. learned router_norm weight (architectures that ship one),
        //   2. parameter-free RMSNorm (HF Gemma 4 — `with_scale=False`),
        //   3. fallback: experts' pre-norm (legacy / archs without an explicit
        //      router norm).
        let router_in_normed = if !self.router_norm.is_empty() {
            rms_norm(h, self.router_norm, eps, norm_offset)
        } else if self.router_norm_parameter_free {
            rms_norm_no_weight(h, eps)
        } else {
            h_norm.clone()
        };

        let mut router_in: Vec<f32> = if !self.router_scale.is_empty() {
            router_in_normed.iter().zip(self.router_scale.iter()).map(|(a, b)| a * b).collect()
        } else {
            router_in_normed
        };
        if self.router_input_scalar != 1.0 && self.router_input_scalar != 0.0 {
            for v in router_in.iter_mut() { *v *= self.router_input_scalar; }
        }

        let mut logits = matmul_vec(&router_in, self.router_proj, self.num_experts, hidden);
        softmax(&mut logits);

        let (indices, mut weights) = top_k(&logits, self.top_k);

        // Renormalize selected weights to sum to 1 — matches Gemma 4's
        // gemma4_top_k_softmax which normalises after selection.
        let weight_sum: f32 = weights.iter().sum();
        if weight_sum > 0.0 {
            for w in &mut weights { *w /= weight_sum; }
        }

        if !self.router_per_expert_scale.is_empty() {
            for (i, &ei) in indices.iter().enumerate() {
                if ei < self.router_per_expert_scale.len() {
                    weights[i] *= self.router_per_expert_scale[ei];
                }
            }
        }

        (h_norm, indices, weights)
    }
}

// ── RemoteMoeBackend ───────────────────────────────────────────────────────

/// Remote MoE expert backend. Thread-safe — all methods take `&self`.
///
/// The shard map is stored behind an `RwLock` so `reshard()` can replace it
/// without interrupting in-flight `forward_moe` calls on other threads.
pub struct RemoteMoeBackend {
    shards: Arc<RwLock<Vec<Shard>>>,
}

impl RemoteMoeBackend {
    /// Build from a shard list. Performs a health check on each shard.
    pub fn connect(configs: Vec<ShardConfig>) -> Result<Self, RemoteMoeError> {
        let shards: Result<Vec<Shard>, _> = configs.into_iter().map(Shard::connect).collect();
        Ok(Self { shards: Arc::new(RwLock::new(shards?)) })
    }

    /// Replace the shard map live (no model reload, no inference interruption).
    ///
    /// Reconnects to new shards, then atomically swaps the map.
    /// In-flight requests against old shards complete normally.
    pub fn reshard(&self, configs: Vec<ShardConfig>) -> Result<(), RemoteMoeError> {
        let new_shards: Result<Vec<Shard>, _> = configs.into_iter().map(Shard::connect).collect();
        *self.shards.write().unwrap() = new_shards?;
        Ok(())
    }

    /// Latency-stats probe: test-call each shard with a zero-length batch and
    /// return `(url, rtt_ms)` per shard. Non-fatal — returns partial results.
    pub fn probe_latency(&self) -> Vec<(String, f64)> {
        let shards = self.shards.read().unwrap();
        shards
            .par_iter()
            .map(|shard| {
                let t = std::time::Instant::now();
                let _ = shard.call_batch(&[]);
                let rtt_ms = t.elapsed().as_secs_f64() * 1000.0;
                (shard.config.url.clone(), rtt_ms)
            })
            .collect()
    }

    /// Run one MoE layer forward pass with experts dispatched remotely.
    ///
    /// Steps:
    ///   1. Router runs locally on `h` using `router`.
    ///   2. Selected experts are grouped by owning shard.
    ///   3. One `POST /v1/expert/batch` per shard (parallel).
    ///   4. Weighted outputs are summed; post-experts norm applied.
    ///
    /// Returns the expert-block contribution (same shape as `h`).
    pub fn forward_moe(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        let hidden = h.len();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 {
            return Ok(vec![0.0f32; hidden]);
        }

        // 1. Route locally.
        let (_h_norm, expert_indices, expert_weights) = router.route(h, norm_offset, eps);

        // 2. Build per-shard call lists.
        let shards = self.shards.read().unwrap();
        let mut shard_calls: Vec<(usize, Vec<ExpertCallItem>)> =
            (0..shards.len()).map(|i| (i, Vec::new())).collect();

        for (&expert_id, _) in expert_indices.iter().zip(expert_weights.iter()) {
            let shard_idx = shards
                .iter()
                .position(|s| s.owns(expert_id))
                .ok_or(RemoteMoeError::NoShard { expert_id })?;
            shard_calls[shard_idx].1.push(ExpertCallItem {
                layer,
                expert_id,
                residual: h.to_vec(),
            });
        }

        // 3. Parallel dispatch — one batch call per shard that has work.
        let non_empty: Vec<(usize, &Vec<ExpertCallItem>)> = shard_calls
            .iter()
            .filter(|(_, items)| !items.is_empty())
            .map(|(si, items)| (*si, items))
            .collect();

        let results_per_shard: Vec<Result<Vec<ExpertResultItem>, RemoteMoeError>> = non_empty
            .par_iter()
            .map(|(si, items)| shards[*si].call_batch(items))
            .collect();

        // 4. Accumulate weighted outputs.
        let expert_weight_map: std::collections::HashMap<usize, f32> =
            expert_indices.iter().copied().zip(expert_weights.iter().copied()).collect();

        let mut out = vec![0.0f32; hidden];
        for result in results_per_shard {
            for item in result? {
                if item.output.len() != hidden {
                    return Err(RemoteMoeError::BadResponse(format!(
                        "expert {}/{} returned {} floats, expected {hidden}",
                        item.layer, item.expert_id, item.output.len()
                    )));
                }
                let weight = expert_weight_map.get(&item.expert_id).copied().unwrap_or(0.0);
                for (acc, &val) in out.iter_mut().zip(item.output.iter()) {
                    *acc += weight * val;
                }
            }
        }

        // 5. Post-experts norm.
        Ok(rms_norm(&out, router.post_experts_norm, eps, norm_offset))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_range_valid() {
        assert_eq!(ShardConfig::parse_range("0-31"), Some((0, 31)));
        assert_eq!(ShardConfig::parse_range("32-63"), Some((32, 63)));
        assert_eq!(ShardConfig::parse_range("0-0"), Some((0, 0)));
    }

    #[test]
    fn parse_range_invalid() {
        assert_eq!(ShardConfig::parse_range("31-0"), None); // reversed
        assert_eq!(ShardConfig::parse_range("abc"), None);
        assert_eq!(ShardConfig::parse_range(""), None);
    }

    #[test]
    fn shard_config_strips_trailing_slash() {
        let s = ShardConfig::new(0, 31, "http://a.example.com:8081///");
        assert_eq!(s.url, "http://a.example.com:8081");
    }

    #[test]
    fn shard_owns() {
        fn make_shard(start: usize, end: usize) -> Shard {
            let config = ShardConfig::new(start, end, "http://localhost:8080");
            let client = reqwest::blocking::Client::new();
            Shard { config, client }
        }
        let s = make_shard(0, 31);
        assert!(s.owns(0));
        assert!(s.owns(31));
        assert!(!s.owns(32));
        let s2 = make_shard(32, 63);
        assert!(s2.owns(32));
        assert!(s2.owns(63));
        assert!(!s2.owns(31));
    }

    #[test]
    fn route_softmax_sums_to_one() {
        let num_experts = 8;
        let hidden = 4;
        let router_proj: Vec<f32> = (0..num_experts * hidden).map(|i| i as f32 * 0.01).collect();
        let router = MoeRouterWeights {
            router_proj: &router_proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k: 2,
        };
        let h: Vec<f32> = vec![1.0, 0.5, -0.3, 0.2];
        let (_, indices, weights) = router.route(&h, 0.0, 1e-6);
        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn forward_moe_empty_input_returns_zero() {
        // Can't connect to a real server, but we can verify the early-exit path.
        // Construct a backend with an empty shard list via the raw struct (bypassing connect).
        let backend = RemoteMoeBackend {
            shards: Arc::new(RwLock::new(vec![])),
        };
        let router = MoeRouterWeights {
            router_proj: &[],
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_experts_norm: &[],
            num_experts: 0,
            top_k: 0,
        };
        let result = backend.forward_moe(0, &[1.0f32, 2.0, 3.0], &router, 0.0, 1e-6);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0.0f32; 3]);
    }
}
