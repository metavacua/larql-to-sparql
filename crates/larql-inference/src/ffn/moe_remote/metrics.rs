//! Lightweight transport-byte accounting for remote MoE dispatch.
//!
//! Enabled by `LARQL_MOE_BYTES=1`; also enabled when `LARQL_MOE_TIMING=1` so
//! timing runs get byte/shard accounting without another flag. Both reads
//! go through [`RemoteMoeRuntime`] so [`record_call`] / [`record_skip`]
//! pay no per-call env-var cost.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use super::runtime::RemoteMoeRuntime;

#[derive(Clone, Copy, Debug, Default)]
pub struct ShardTransportTotals {
    pub calls: u64,
    pub skipped: u64,
    pub request_bytes: u64,
    pub response_bytes: u64,
    pub active_experts: u64,
}

#[derive(Clone, Debug, Default)]
pub struct TransportSnapshot {
    by_shard: BTreeMap<String, ShardTransportTotals>,
}

impl TransportSnapshot {
    fn delta_since(&self, before: &TransportSnapshot) -> TransportSnapshot {
        let mut out = BTreeMap::new();
        for (shard, after) in &self.by_shard {
            let before = before.by_shard.get(shard).copied().unwrap_or_default();
            let delta = ShardTransportTotals {
                calls: after.calls.saturating_sub(before.calls),
                skipped: after.skipped.saturating_sub(before.skipped),
                request_bytes: after.request_bytes.saturating_sub(before.request_bytes),
                response_bytes: after.response_bytes.saturating_sub(before.response_bytes),
                active_experts: after.active_experts.saturating_sub(before.active_experts),
            };
            if delta.calls > 0
                || delta.skipped > 0
                || delta.request_bytes > 0
                || delta.response_bytes > 0
            {
                out.insert(shard.clone(), delta);
            }
        }
        TransportSnapshot { by_shard: out }
    }

    fn total(&self) -> ShardTransportTotals {
        self.by_shard
            .values()
            .fold(ShardTransportTotals::default(), |mut acc, v| {
                acc.calls += v.calls;
                acc.skipped += v.skipped;
                acc.request_bytes += v.request_bytes;
                acc.response_bytes += v.response_bytes;
                acc.active_experts += v.active_experts;
                acc
            })
    }
}

fn state() -> &'static Mutex<BTreeMap<String, ShardTransportTotals>> {
    static STATE: OnceLock<Mutex<BTreeMap<String, ShardTransportTotals>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub fn enabled() -> bool {
    RemoteMoeRuntime::get().moe_bytes_enabled
}

pub fn shard_timing_enabled() -> bool {
    RemoteMoeRuntime::get().moe_shard_timing
}

pub fn snapshot() -> TransportSnapshot {
    let by_shard = state().lock().map(|g| g.clone()).unwrap_or_default();
    TransportSnapshot { by_shard }
}

pub fn record_call(
    shard: &str,
    request_bytes: usize,
    response_bytes: usize,
    active_experts: usize,
) {
    if !enabled() {
        return;
    }
    if let Ok(mut guard) = state().lock() {
        let totals = guard.entry(shard.to_owned()).or_default();
        totals.calls += 1;
        totals.request_bytes += request_bytes as u64;
        totals.response_bytes += response_bytes as u64;
        totals.active_experts += active_experts as u64;
    }
}

pub fn record_skip(shard: &str) {
    if !enabled() {
        return;
    }
    if let Ok(mut guard) = state().lock() {
        guard.entry(shard.to_owned()).or_default().skipped += 1;
    }
}

pub fn print_delta(label: &str, tok_idx: usize, before: &TransportSnapshot) {
    if !enabled() {
        return;
    }
    let delta = snapshot().delta_since(before);
    let total = delta.total();
    if total.calls == 0 && total.skipped == 0 {
        return;
    }
    let total_bytes = total.request_bytes + total.response_bytes;
    eprintln!(
        "[moe-bytes] {label} tok={tok_idx} calls={} skipped={} req={}B resp={}B total={}B active_experts={}",
        total.calls,
        total.skipped,
        total.request_bytes,
        total.response_bytes,
        total_bytes,
        total.active_experts,
    );
    for (shard, totals) in delta.by_shard {
        let shard_bytes = totals.request_bytes + totals.response_bytes;
        eprintln!(
            "[moe-bytes]   shard={shard} calls={} skipped={} req={}B resp={}B total={}B active_experts={}",
            totals.calls,
            totals.skipped,
            totals.request_bytes,
            totals.response_bytes,
            shard_bytes,
            totals.active_experts,
        );
    }
}

pub fn print_summary(label: &str, before: &TransportSnapshot, measured_tokens: usize) {
    if !enabled() {
        return;
    }
    let delta = snapshot().delta_since(before);
    let total = delta.total();
    if total.calls == 0 && total.skipped == 0 {
        return;
    }
    let denom = measured_tokens.max(1) as f64;
    let total_bytes = total.request_bytes + total.response_bytes;
    eprintln!(
        "[moe-bytes] {label} SUMMARY tokens={} calls={} skipped={} req={}B resp={}B total={}B active_experts={}",
        measured_tokens,
        total.calls,
        total.skipped,
        total.request_bytes,
        total.response_bytes,
        total_bytes,
        total.active_experts,
    );
    eprintln!(
        "[moe-bytes]   per-token avg: calls={:.2} skipped={:.2} req={:.0}B resp={:.0}B total={:.0}B active_experts={:.2}",
        total.calls as f64 / denom,
        total.skipped as f64 / denom,
        total.request_bytes as f64 / denom,
        total.response_bytes as f64 / denom,
        total_bytes as f64 / denom,
        total.active_experts as f64 / denom,
    );
    for (shard, totals) in delta.by_shard {
        let shard_bytes = totals.request_bytes + totals.response_bytes;
        eprintln!(
            "[moe-bytes]   shard={shard} calls={} skipped={} req={}B resp={}B total={}B active_experts={}",
            totals.calls,
            totals.skipped,
            totals.request_bytes,
            totals.response_bytes,
            shard_bytes,
            totals.active_experts,
        );
    }
}
