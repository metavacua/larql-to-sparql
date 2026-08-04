//! Phase-timing sink + env-var trace plumbing for the walk paths.
//!
//! Two independent observability channels live here:
//!
//! - [`PhaseTimingsHandle`] — atomic counters filled by the
//!   `sparse:parallel_q4k_down` branch (attach via
//!   `WalkFfn::with_phase_timings`).
//! - The `LARQL_WALK_TRACE` env var — a stderr echo of every
//!   dispatch-trace entry, cached per thread so the hot path never
//!   calls `getenv` more than once per thread.

/// Phase-timing sink for `sparse:parallel_q4k_down`. All counters are
/// `AtomicU64` so the rayon-parallel scan can record without locking.
/// Times are sums across every invocation of the branch (a per-position
/// call) — divide by `calls` for a per-call average.
#[derive(Debug, Default)]
pub struct PhaseTimingsHandle {
    /// Time inside the per-position gate KNN dispatch (`gate_walk`
    /// → `gate_knn_q4` → `gate_knn` fallback chain). Counts once per
    /// (position, layer) — i.e. once per `parallel_q4k_down` call.
    pub gate_knn_ns: std::sync::atomic::AtomicU64,
    /// Time inside `kquant_ffn_layer(layer, FFN_DOWN)` — should be ~0 once the
    /// dequantised down cache is warm.
    pub cache_fetch_ns: std::sync::atomic::AtomicU64,
    /// Time spent in the `par_chunks().map().collect()` scan — the
    /// per-feature up-dot + scaled-add loop. Expected to dominate.
    pub parallel_scan_ns: std::sync::atomic::AtomicU64,
    /// Time spent summing per-thread partials into the output row.
    pub reduce_ns: std::sync::atomic::AtomicU64,
    /// Number of times the parallel_q4k_down branch fired.
    pub calls: std::sync::atomic::AtomicU64,
}

// Thread-local cache of the LARQL_WALK_TRACE env var so we don't
// getenv on every layer. Set once per thread on first access; the
// env var is typically static across a process lifetime.
thread_local! {
    static WALK_TRACE_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// `LARQL_WALK_TRACE=1` — emit the per-feature walk trace to stderr.
pub(super) const ENV_WALK_TRACE: &str = "LARQL_WALK_TRACE";

pub(super) fn walk_trace_env_enabled() -> bool {
    WALK_TRACE_ENABLED.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        // Route through the override-aware `options` helper so tests can toggle
        // this via the thread-local override (NOT `std::env::set_var`, which
        // races concurrent `getenv` on the decode path → SIGSEGV).
        let enabled = larql_compute::options::env_value(ENV_WALK_TRACE).as_deref() == Some("1");
        c.set(Some(enabled));
        enabled
    })
}
