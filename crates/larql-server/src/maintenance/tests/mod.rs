//! Unit tests for the maintenance sweeper: report aggregation and the
//! periodic spawn loop.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Fast interval for the spawn-loop test; only pacing, not a contract.
const TEST_SWEEP_INTERVAL: Duration = Duration::from_millis(5);

/// Generous upper bound for the spawn-loop test to observe two sweeps.
const TEST_WAIT_BUDGET: Duration = Duration::from_secs(5);

#[test]
fn target_reports_its_name() {
    let target = SweepTarget::new("buckets", || async { 0 });
    assert_eq!(target.name(), "buckets");
}

#[tokio::test]
async fn sweep_once_reports_each_target_in_order() {
    let targets = vec![
        SweepTarget::new("first", || async { 2 }),
        SweepTarget::new("second", || async { 0 }),
        SweepTarget::new("third", || async { 7 }),
    ];
    let report = sweep_once(&targets).await;
    assert_eq!(report, vec![("first", 2), ("second", 0), ("third", 7)]);
}

#[tokio::test]
async fn sweep_once_with_no_targets_is_empty() {
    assert!(sweep_once(&[]).await.is_empty());
}

#[tokio::test]
async fn sweep_once_invokes_the_closure_each_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_in = Arc::clone(&calls);
    let targets = vec![SweepTarget::new("counted", move || {
        let calls = Arc::clone(&calls_in);
        async move { calls.fetch_add(1, Ordering::SeqCst) }
    })];
    sweep_once(&targets).await;
    sweep_once(&targets).await;
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn spawn_runs_targets_periodically() {
    let sweeps = Arc::new(AtomicUsize::new(0));
    let sweeps_in = Arc::clone(&sweeps);
    let handle = spawn(
        TEST_SWEEP_INTERVAL,
        vec![SweepTarget::new("ticker", move || {
            let sweeps = Arc::clone(&sweeps_in);
            async move {
                sweeps.fetch_add(1, Ordering::SeqCst);
                1
            }
        })],
    );

    // Wait until the loop has demonstrably run more than once.
    let observed = tokio::time::timeout(TEST_WAIT_BUDGET, async {
        while sweeps.load(Ordering::SeqCst) < 2 {
            tokio::time::sleep(TEST_SWEEP_INTERVAL).await;
        }
    })
    .await;
    handle.abort();

    assert!(
        observed.is_ok(),
        "sweeper did not run twice within {TEST_WAIT_BUDGET:?}"
    );
}
