//! Public types for the walk-ffn endpoint and the in-process RAII guard
//! that tracks `requests_in_flight` for GT6 drain.
//!
//! Split out of the previous monolithic `walk_ffn.rs` so the request
//! shape + binary constants are reachable from the codec, validators,
//! and handler without circular imports.

use serde::Deserialize;

/// RAII guard that decrements the `requests_in_flight` counter on drop.
/// Used by [`super::handler::handle_walk_ffn`] so the GT6 drain protocol
/// (ADR-0011 §Phase B2) sees an accurate in-flight count even when the
/// handler errors out before sending a response.
pub(crate) struct RifGuard(pub(crate) std::sync::Arc<std::sync::atomic::AtomicU32>);

impl Drop for RifGuard {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        // Saturating sub to avoid wrapping if something incremented 0 and dropped twice.
        let prev = self
            .0
            .fetch_update(Ordering::Release, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
        let _ = prev;
    }
}

/// Register a compute request against the first loaded model for GT6 drain
/// and heartbeat visibility: bumps `requests_in_flight` (decremented when
/// the returned guard drops) and the cumulative `requests_total` that the
/// grid announce loop diffs into `HeartbeatMsg.req_per_sec`.
///
/// Every model-compute handler (walk-ffn, walk-ffn-q8k, all expert
/// endpoints) must hold one of these for its full duration — a handler
/// that skips it is invisible to drain and can be reassigned mid-request
/// (ROADMAP hardening item 13).
pub(crate) fn track_model_request(state: &crate::state::AppState) -> Option<RifGuard> {
    state.first_model().map(|m| {
        use std::sync::atomic::Ordering;
        m.requests_in_flight.fetch_add(1, Ordering::Relaxed);
        m.requests_total.fetch_add(1, Ordering::Relaxed);
        RifGuard(m.requests_in_flight.clone())
    })
}

// Wire constants are single-sourced in the shared codec (ROADMAP
// hardening item 16). The handler now detects the inbound format via
// `crate::wire::request_wire_format` (f32/f16/i8 Content-Type dispatch),
// so no CT constant is re-exported here; `BATCH_MARKER` lives at
// `larql_inference::ffn::remote::BATCH_MARKER`.

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
    /// When true, `residual` is `h_post_attn` (post-attention, pre-norm). The
    /// server runs the full hybrid MoE layer: dense-FFN + remote expert dispatch
    /// + combine + outer norm. Requires `full_output: true` and the server to
    ///   have `--moe-shards` configured.
    #[serde(default)]
    pub moe_layer: bool,
}

fn default_seq_len() -> usize {
    1
}

/// Default `top_k` for walk-ffn requests that omit the field.
///
/// NOTE: 8092 is suspiciously close to 8192 (2^13) and may be a historic
/// typo, but it is the value clients have been served with — changing it
/// changes served behavior, so it is kept as-is.
const DEFAULT_WALK_FFN_TOP_K: usize = 8092;

fn default_top_k() -> usize {
    DEFAULT_WALK_FFN_TOP_K
}

// ── Typed output structs (shared by JSON + binary encoders) ──────────────────
//
// `FfnEntry`/`FfnOutput` moved into the shared codec alongside the
// encoders that consume them; re-exported to preserve the
// `super::types::{FfnEntry, FfnOutput}` paths used by `core`/`binary`.
pub(crate) use larql_inference::ffn::remote::{FfnEntry, FfnOutput};
