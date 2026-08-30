//! AppState: loaded vindex + config, shared across all handlers.
//!
//! Split across:
//! - `loaded_model.rs` — `LoadedModel`, one bound VINDEX2 model plus
//!   its lazy-loaded weights and per-model counters/caches.
//! - `model_set.rs`    — `ModelSet` (the coherent V2+V3 snapshot),
//!   `ServedModel`, and every `AppState` method that resolves a
//!   request to a model.
//! - `lifecycle.rs`    — `RouterTopology` (the invariant that dynamic
//!   model lifecycle mutation cannot outgrow the router axum was
//!   actually built with, `docs/runtime-lifecycle-design.md` §7) and
//!   `LifecycleState` (the single-slot load/unload state machine
//!   `routes/runtime_lifecycle.rs` drives).
//! - this file          — `AppState` itself and small free-standing
//!   helpers that don't belong to any of the above.
//!
//! The split is pure file-size hygiene (see `docs/runtime-lifecycle-design.md`
//! §1 for the reasoning behind `ModelSet` itself) — every type here
//! stays reachable at its original `crate::state::*` path via the
//! re-exports below.

mod lifecycle;
mod loaded_model;
mod model_set;

pub use lifecycle::{
    decide_load, decide_unload, LifecycleError, LifecycleState, LoadDecision, RouterTopology,
    UnloadDecision,
};
pub use loaded_model::LoadedModel;
pub use model_set::{ModelSet, ServedModel};

use std::collections::HashMap;
use std::sync::Arc;

use larql_vindex::format::filenames::FEATURE_LABELS_JSON;

use crate::cache::DescribeCache;
use crate::session::SessionManager;

/// Shared application state.
pub struct AppState {
    /// Every model this process has bound (see [`ModelSet`]). All
    /// resolution goes through the methods below, which take the read
    /// guard only long enough to find and clone the `Arc` they need,
    /// then release it — no inference work, and no `.await`, ever
    /// happens while the guard is held.
    pub model_set: std::sync::RwLock<ModelSet>,
    /// The router topology `bootstrap::serve` actually built axum
    /// with, frozen once at construction. This is *not* derivable
    /// from `model_set`'s live count once lifecycle mutation exists —
    /// see [`RouterTopology`] and `docs/runtime-lifecycle-design.md`
    /// §7. Every lifecycle mutation must go through
    /// [`AppState::validate_lifecycle_mutation`], which reads this.
    pub router_topology: RouterTopology,
    /// The single-slot load/unload state flag `routes::runtime_lifecycle`
    /// drives. Held only long enough to check-and-set — the actual
    /// load/drain work runs with the lock released, so a concurrent
    /// second lifecycle call sees the flag immediately (and rejects
    /// outright) instead of blocking behind it.
    pub lifecycle: std::sync::Mutex<LifecycleState>,
    /// Server start time for uptime reporting.
    pub started_at: std::time::Instant,
    /// Request counter.
    pub requests_served: std::sync::atomic::AtomicU64,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Per-session PatchedVindex manager.
    pub sessions: SessionManager,
    /// Stored Responses-API envelopes + conversations, backing
    /// `store` / `previous_response_id` and `GET /v1/responses/{id}`.
    pub responses: crate::response_store::ResponseStore,
    /// N1 — resident KV continuation states for chained V3 responses
    /// (`previous_response_id`); see [`crate::response_kv`].
    pub v3_kv: crate::response_kv::ResponseKvCache,
    /// DESCRIBE result cache.
    pub describe_cache: DescribeCache,
    /// Server-side hard timeout for `/v1/infer` and friends.  When
    /// the wall-time of the spawn_blocking future exceeds this, the
    /// handler responds 504 and drops the JoinHandle.  The blocking
    /// thread is *not* killed (we don't have cooperative cancel on
    /// the inference path) — it runs to completion in the
    /// background and its result is discarded.  Default: 60s; set
    /// to 0 to disable.  See BUG-infer-deadlock §5.6.
    pub infer_timeout: std::time::Duration,
    /// Server-wide performance/activity recorder backing
    /// `GET /v1/runtime` (see [`crate::runtime_stats`]). `Arc`-wrapped
    /// so a streaming generation handler can clone the recorder alone
    /// into its `spawn_blocking` closure, independently of the rest of
    /// `AppState`.
    pub runtime: Arc<crate::runtime_stats::RuntimeRecorder>,
}

/// Compute elapsed milliseconds from `start`, rounded to one decimal place.
pub fn elapsed_ms(start: std::time::Instant) -> f64 {
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (ms * 10.0).round() / 10.0
}

/// Load probe-confirmed feature labels from feature_labels.json.
/// Format: {"L{layer}_F{feature}": "relation_name", ...}
pub fn load_probe_labels(vindex_path: &std::path::Path) -> HashMap<(usize, usize), String> {
    let path = vindex_path.join(FEATURE_LABELS_JSON);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return HashMap::new(),
    };
    let obj: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let map = match obj.as_object() {
        Some(m) => m,
        None => return HashMap::new(),
    };

    let mut labels = HashMap::new();
    for (key, value) in map {
        if let Some(rel) = value.as_str() {
            let parts: Vec<&str> = key.split('_').collect();
            if parts.len() == 2 {
                if let (Some(layer), Some(feat)) = (
                    parts[0]
                        .strip_prefix('L')
                        .and_then(|s| s.parse::<usize>().ok()),
                    parts[1]
                        .strip_prefix('F')
                        .and_then(|s| s.parse::<usize>().ok()),
                ) {
                    labels.insert((layer, feat), rel.to_string());
                }
            }
        }
    }
    labels
}

/// Derive a short model ID from the full model name.
/// "google/gemma-3-4b-it" → "gemma-3-4b-it"
pub fn model_id_from_name(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}

#[cfg(test)]
mod tests {
    //! `load_probe_labels` / `model_id_from_name` — the two free
    //! functions that stayed in this file rather than moving into
    //! `loaded_model.rs` or `model_set.rs`.
    use super::*;

    #[test]
    fn model_id_from_name_strips_the_org_prefix() {
        assert_eq!(model_id_from_name("google/gemma-3-4b-it"), "gemma-3-4b-it");
    }

    #[test]
    fn model_id_from_name_is_a_no_op_without_a_slash() {
        assert_eq!(model_id_from_name("standalone"), "standalone");
    }

    #[test]
    fn load_probe_labels_missing_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_probe_labels(dir.path()).is_empty());
    }

    #[test]
    fn load_probe_labels_parses_layer_feature_keys() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("feature_labels.json"),
            r#"{"L3_F12": "capital_of", "malformed": "skipped"}"#,
        )
        .unwrap();
        let labels = load_probe_labels(dir.path());
        assert_eq!(labels.get(&(3, 12)), Some(&"capital_of".to_string()));
        assert_eq!(
            labels.len(),
            1,
            "the malformed key must be skipped, not panic"
        );
    }

    #[test]
    fn load_probe_labels_malformed_json_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("feature_labels.json"), "not json").unwrap();
        assert!(load_probe_labels(dir.path()).is_empty());
    }
}
