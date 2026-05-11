//! GT6 dynamic rebalancing background task.
//!
//! Runs every `check_interval` seconds and checks for per-layer latency
//! imbalance across replicated shards. When a shard is measurably slower
//! than its peers (ratio > `imbalance_threshold`) and a spare available
//! server exists to replace it, the rebalancer sends `UnassignMsg` to the
//! slow server and triggers gap-fill for the freed layer range.
//!
//! The server receives `UnassignMsg`, drains in-flight requests (up to 30s),
//! sends `DroppingMsg(reason="reassigned")`, and re-enters the available pool.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::{debug, info};

use larql_router_protocol::{RouterMessage, RouterPayload, UnassignMsg};

use crate::grid::GridState;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct RebalancerConfig {
    /// How often to run the imbalance check.
    pub check_interval: Duration,
    /// Trigger rebalancing when max(avg_ms) / min(avg_ms) exceeds this ratio
    /// across replicas covering the same layer for at least `sustained_window`.
    pub imbalance_threshold: f32,
    /// Sustained imbalance window before action is taken.
    pub sustained_window: Duration,
}

impl Default for RebalancerConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            imbalance_threshold: 2.0,
            sustained_window: Duration::from_secs(60),
        }
    }
}

impl RebalancerConfig {
    pub fn from_cli(interval_secs: u64, threshold: f32) -> Self {
        Self {
            check_interval: Duration::from_secs(interval_secs),
            imbalance_threshold: threshold,
            sustained_window: Duration::from_secs(interval_secs * 2),
        }
    }
}

// ── Per-layer imbalance tracker ───────────────────────────────────────────────

/// Tracks how long a given layer has been in imbalanced state.
#[derive(Default)]
struct ImbalanceTracker {
    /// (model_id, layer) → first_seen_imbalanced
    first_seen: HashMap<(String, u32), std::time::Instant>,
}

impl ImbalanceTracker {
    /// Record that this layer is currently imbalanced. Returns true if the
    /// imbalance has been sustained long enough to trigger action.
    fn record(&mut self, key: (String, u32), sustained: Duration) -> bool {
        let entry = self
            .first_seen
            .entry(key)
            .or_insert_with(std::time::Instant::now);
        entry.elapsed() >= sustained
    }

    /// Clear a layer's imbalance record (it is now balanced or was acted on).
    fn clear(&mut self, key: &(String, u32)) {
        self.first_seen.remove(key);
    }
}

// ── Rebalancer task ───────────────────────────────────────────────────────────

/// Spawn the rebalancer background task.
/// Returns immediately; the task runs for the process lifetime.
pub fn spawn(state: Arc<RwLock<GridState>>, cfg: RebalancerConfig) {
    tokio::spawn(rebalancer_task(state, cfg));
}

async fn rebalancer_task(state: Arc<RwLock<GridState>>, cfg: RebalancerConfig) {
    let mut interval = tokio::time::interval(cfg.check_interval);
    let mut tracker = ImbalanceTracker::default();

    loop {
        interval.tick().await;
        check_imbalance(&state, &cfg, &mut tracker).await;
    }
}

async fn check_imbalance(
    state: &Arc<RwLock<GridState>>,
    cfg: &RebalancerConfig,
    tracker: &mut ImbalanceTracker,
) {
    // Collect per-layer latency data across all servers.
    // Group by (model_id, layer) → Vec<(server_id, avg_ms)>.
    let snapshot = {
        let guard = state.read().await;
        let mut by_layer: HashMap<(String, u32), Vec<(String, f32)>> = HashMap::new();
        for (sid, entry) in guard.servers() {
            for (&layer, &(avg_ms, _p99)) in &entry.layer_latencies {
                by_layer
                    .entry((entry.model_id.clone(), layer))
                    .or_default()
                    .push((sid.clone(), avg_ms));
            }
        }
        let has_available = guard.has_available_servers();
        (by_layer, has_available)
    };

    let (by_layer, has_available) = snapshot;

    // Only rebalance if there is a spare server ready to take over.
    if !has_available {
        debug!("Rebalancer: no available servers — skipping imbalance check");
        return;
    }

    for ((model_id, layer), mut servers) in by_layer {
        if servers.len() < 2 {
            // Can't detect imbalance without at least 2 replicas.
            tracker.clear(&(model_id, layer));
            continue;
        }

        servers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let min_ms = servers.first().map(|(_, ms)| *ms).unwrap_or(0.0);
        let max_ms = servers.last().map(|(_, ms)| *ms).unwrap_or(0.0);
        let slowest_server_id = servers.last().map(|(id, _)| id.clone());

        if min_ms <= 0.0 {
            continue;
        }

        let ratio = max_ms / min_ms;
        let key = (model_id.clone(), layer);

        if ratio > cfg.imbalance_threshold {
            let sustained = tracker.record(key.clone(), cfg.sustained_window);
            if sustained {
                // Imbalance has persisted long enough — send UnassignMsg.
                if let Some(ref server_id) = slowest_server_id {
                    info!(
                        model_id = %model_id,
                        layer,
                        ratio = %format!("{ratio:.1}×"),
                        server_id = %server_id,
                        "Rebalancer: sustained imbalance detected — sending UnassignMsg"
                    );
                    send_unassign(state, server_id, &model_id, layer).await;
                    tracker.clear(&key);
                }
            } else {
                debug!(
                    model_id = %model_id,
                    layer,
                    ratio = %format!("{ratio:.1}×"),
                    "Rebalancer: imbalance observed (not yet sustained)"
                );
            }
        } else {
            tracker.clear(&key);
        }
    }
}

/// Send `UnassignMsg` to the serving server identified by `server_id`.
/// The sender channel is stored in `GridState::serving_senders`.
async fn send_unassign(
    state: &Arc<RwLock<GridState>>,
    server_id: &str,
    model_id: &str,
    layer: u32,
) {
    let guard = state.read().await;
    if let Some(tx) = guard.serving_sender(server_id) {
        let msg = RouterMessage {
            payload: Some(RouterPayload::Unassign(UnassignMsg {
                model_id: model_id.to_owned(),
                layer_start: layer,
                layer_end: layer,
                reason: "rebalancing".to_owned(),
            })),
        };
        if let Err(e) = tx.try_send(Ok(msg)) {
            tracing::warn!(server_id, "Rebalancer: failed to send UnassignMsg: {e}");
        }
    } else {
        tracing::warn!(
            server_id,
            "Rebalancer: no sender for server — already disconnected?"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imbalance_tracker_records_and_clears() {
        let mut t = ImbalanceTracker::default();
        let key = ("model".to_string(), 5u32);
        // First record: not sustained yet (window = 1 hour).
        assert!(!t.record(key.clone(), Duration::from_secs(3600)));
        // Clear and re-record: still fresh.
        t.clear(&key);
        assert!(!t.record(key.clone(), Duration::from_secs(3600)));
        // With zero window: sustained immediately.
        let key2 = ("model".to_string(), 6u32);
        assert!(t.record(key2, Duration::from_secs(0)));
    }

    #[test]
    fn rebalancer_config_defaults() {
        let cfg = RebalancerConfig::default();
        assert_eq!(cfg.check_interval, Duration::from_secs(30));
        assert_eq!(cfg.imbalance_threshold, 2.0);
    }
}
