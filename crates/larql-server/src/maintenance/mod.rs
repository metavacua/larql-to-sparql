//! Periodic maintenance sweeper for bounded in-memory stores.
//!
//! Every in-memory map the server holds per client (rate-limit buckets,
//! session overlays, …) must have its stale entries dropped somewhere, or
//! the process grows without bound. This module is that somewhere: the
//! bootstrap registers one [`SweepTarget`] per store and [`spawn`]s a
//! single background task that runs them all on a fixed interval.
//!
//! The module is deliberately decoupled from the stores it sweeps — a
//! target is just a named async closure returning how many entries it
//! removed — so a new bounded store only has to register a closure here,
//! and the sweeper needs no knowledge of `AppState`.

#[cfg(test)]
mod tests;

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use tokio::task::JoinHandle;
use tracing::{debug, info};

/// How often the sweeper wakes: once a minute. Stale-entry windows (session
/// TTL, rate-limit GC) are minutes-to-hours, so a minute of eviction latency
/// is invisible while keeping the idle wakeup cost negligible.
pub const DEFAULT_SWEEP_INTERVAL_SECS: u64 = 60;

/// Boxed async sweep operation; resolves to how many entries were removed.
type SweepFuture = Pin<Box<dyn Future<Output = usize> + Send>>;

/// One named store to sweep.
pub struct SweepTarget {
    name: &'static str,
    run: Box<dyn Fn() -> SweepFuture + Send + Sync>,
}

impl SweepTarget {
    /// Wrap an async closure as a sweep target. The closure is invoked once
    /// per sweep and returns how many entries it evicted.
    pub fn new<F, Fut>(name: &'static str, run: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = usize> + Send + 'static,
    {
        Self {
            name,
            run: Box::new(move || Box::pin(run())),
        }
    }

    /// The target's log name.
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// Run every target once; returns `(name, evicted)` per target in
/// registration order.
pub async fn sweep_once(targets: &[SweepTarget]) -> Vec<(&'static str, usize)> {
    let mut report = Vec::with_capacity(targets.len());
    for target in targets {
        let evicted = (target.run)().await;
        report.push((target.name, evicted));
    }
    report
}

/// Spawn the background sweeper: runs [`sweep_once`] over `targets` every
/// `interval`, forever. The returned handle can be aborted on shutdown;
/// dropping it detaches the task (fine for a process-lifetime daemon).
pub fn spawn(interval: Duration, targets: Vec<SweepTarget>) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;
            let report = sweep_once(&targets).await;
            let total: usize = report.iter().map(|(_, n)| n).sum();
            if total > 0 {
                info!("maintenance sweep evicted {total} stale entries: {report:?}");
            } else {
                debug!("maintenance sweep: nothing stale");
            }
        }
    })
}
