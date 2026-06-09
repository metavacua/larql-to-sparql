//! Lightweight per-stage decode profiler for the DSv4 forward.
//!
//! Off by default (zero-cost: a single `OnceLock<bool>` check). When
//! `LARQL_DSV4_PROFILE=1` is set, [`timed`] accumulates wall time + call
//! count per named stage into a global table; the bench (or any driver)
//! calls [`reset`] before a decode step and [`report`] after to see the
//! attention / routed-MoE / shared-expert / mHC split.
//!
//! Stages nest (a `timed("ffn", …)` wraps `timed("ffn.routed", …)` +
//! `timed("ffn.shared", …)`); the report prints each accumulated bucket,
//! so child buckets sum to (slightly less than) their parent.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LARQL_DSV4_PROFILE").is_ok())
}

/// stage name -> (total nanos, call count)
static ACC: Mutex<BTreeMap<&'static str, (u128, u64)>> = Mutex::new(BTreeMap::new());

/// Time `f` under `name` when profiling is enabled; otherwise just run it.
pub fn timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let out = f();
    let ns = t.elapsed().as_nanos();
    let mut g = ACC.lock().expect("dsv4_profile lock");
    let e = g.entry(name).or_insert((0, 0));
    e.0 += ns;
    e.1 += 1;
    out
}

/// Clear all accumulated timings (call before the region you want to measure).
pub fn reset() {
    if enabled() {
        ACC.lock().expect("dsv4_profile lock").clear();
    }
}

/// Render the accumulated per-stage table (empty string when disabled or
/// nothing recorded). Sorted by total time descending.
pub fn report() -> String {
    if !enabled() {
        return String::new();
    }
    let g = ACC.lock().expect("dsv4_profile lock");
    if g.is_empty() {
        return String::new();
    }
    let mut rows: Vec<(&'static str, u128, u64)> =
        g.iter().map(|(k, (ns, c))| (*k, *ns, *c)).collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let total_top: u128 = rows
        .iter()
        .filter(|(k, _, _)| !k.contains('.'))
        .map(|(_, ns, _)| *ns)
        .sum();
    let mut s = String::from("  --- DSv4 decode profile (LARQL_DSV4_PROFILE) ---\n");
    s.push_str(&format!(
        "  {:<16} {:>10} {:>8} {:>10} {:>7}\n",
        "stage", "total_ms", "calls", "ms/call", "%top"
    ));
    for (name, ns, calls) in rows {
        let ms = ns as f64 / 1e6;
        let pct = if total_top > 0 && !name.contains('.') {
            100.0 * ns as f64 / total_top as f64
        } else {
            f64::NAN
        };
        let pct_str = if pct.is_nan() {
            "   .".to_string()
        } else {
            format!("{pct:6.1}")
        };
        s.push_str(&format!(
            "  {:<16} {:>10.2} {:>8} {:>10.3} {:>7}\n",
            name,
            ms,
            calls,
            ms / calls.max(1) as f64,
            pct_str
        ));
    }
    s
}
