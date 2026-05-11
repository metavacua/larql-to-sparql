//! Per-layer latency tracker for grid heartbeats.
//!
//! Collects compute latency per layer as walk-ffn requests complete.
//! Snapshots are included in `HeartbeatMsg.layer_stats` every 10 seconds
//! so the router can route by actual layer latency rather than by global
//! request count.
//!
//! Two metrics per layer:
//! - `avg_ms`: EMA (α=0.1) — smooth, low overhead, updated every request.
//! - `p99_ms`: ring-buffer p99 over the last 100 requests — tail latency.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use larql_router_protocol::LayerLatency;

const RING_CAPACITY: usize = 100;
const EMA_ALPHA: f32 = 0.1;

struct LayerStats {
    ema_ms: f32,
    ring: VecDeque<f32>,
}

impl LayerStats {
    fn new(first_ms: f32) -> Self {
        let mut ring = VecDeque::with_capacity(RING_CAPACITY);
        ring.push_back(first_ms);
        Self {
            ema_ms: first_ms,
            ring,
        }
    }

    fn record(&mut self, ms: f32) {
        self.ema_ms = EMA_ALPHA * ms + (1.0 - EMA_ALPHA) * self.ema_ms;
        if self.ring.len() == RING_CAPACITY {
            self.ring.pop_front();
        }
        self.ring.push_back(ms);
    }

    fn p99(&self) -> f32 {
        if self.ring.is_empty() {
            return 0.0;
        }
        let mut vals: Vec<f32> = self.ring.iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((vals.len() as f32 * 0.99) as usize).min(vals.len() - 1);
        vals[idx]
    }
}

/// Thread-safe per-layer latency tracker.
///
/// Shared between walk-ffn handlers (write) and the announce heartbeat
/// sender (read). The lock is held only for brief HashMap + VecDeque
/// updates — never during FFN computation.
pub struct LayerLatencyTracker {
    inner: Mutex<HashMap<u32, LayerStats>>,
}

impl Default for LayerLatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerLatencyTracker {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Record `latency_ms` for `layer`. Called once per layer per request.
    pub fn record(&self, layer: u32, latency_ms: f32) {
        if let Ok(mut guard) = self.inner.lock() {
            guard
                .entry(layer)
                .and_modify(|s| s.record(latency_ms))
                .or_insert_with(|| LayerStats::new(latency_ms));
        }
    }

    /// Snapshot current stats as proto `LayerLatency` values, sorted by layer.
    /// Returns an empty vec when no requests have been recorded yet.
    pub fn snapshot(&self) -> Vec<LayerLatency> {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        let mut out: Vec<LayerLatency> = guard
            .iter()
            .map(|(&layer, stats)| LayerLatency {
                layer,
                avg_ms: stats.ema_ms,
                p99_ms: stats.p99(),
            })
            .collect();
        out.sort_by_key(|l| l.layer);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_snapshot_single_layer() {
        let t = LayerLatencyTracker::new();
        t.record(5, 10.0);
        t.record(5, 20.0);
        let snap = t.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].layer, 5);
        assert!(snap[0].avg_ms > 0.0);
        assert!(snap[0].p99_ms >= 10.0);
    }

    #[test]
    fn snapshot_sorted_by_layer() {
        let t = LayerLatencyTracker::new();
        t.record(2, 1.0);
        t.record(0, 1.0);
        t.record(1, 1.0);
        let snap = t.snapshot();
        assert_eq!(snap.iter().map(|l| l.layer).collect::<Vec<_>>(), [0, 1, 2]);
    }

    #[test]
    fn empty_tracker_returns_empty_snapshot() {
        let t = LayerLatencyTracker::new();
        assert!(t.snapshot().is_empty());
    }

    #[test]
    fn ring_buffer_caps_at_capacity() {
        let t = LayerLatencyTracker::new();
        for i in 0..=RING_CAPACITY + 10 {
            t.record(0, i as f32);
        }
        let guard = t.inner.lock().unwrap();
        let stats = guard.get(&0).unwrap();
        assert_eq!(stats.ring.len(), RING_CAPACITY);
    }
}
