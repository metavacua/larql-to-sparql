//! Row-note formatters for the KV-engine bench table.
//!
//! Each function renders one clause of the `notes` column. They are the
//! fields that say what a row actually measured — which dispatch shape
//! ran, how the step split between forward and lm_head, and how much K/V
//! the engine really holds — so they are kept together and tested
//! together, separately from the numeric summarisation in `engine.rs`.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Render the dispatch-shape prefix for an engine row, e.g.
/// `"[per-layer→host] "`. Empty when the engine does not report a shape
/// (no prefill, or an engine with only one).
///
/// This is the field that tells a reader whether the row measured the
/// backend they asked for. A windowed engine on `--backends metal`
/// declines the fused path and runs its whole forward on the CPU, while
/// the row label still says `[metal (GPU)]` — worth ~15 ms/token and
/// otherwise only inferable by noticing the timings look wrong.
pub(super) fn format_dispatch_note(
    path: Option<larql_inference::kv_engine::DispatchPath>,
    per_layer_is_host_delegated: bool,
) -> String {
    match path {
        Some(p) => format!("[{}] ", p.describe(per_layer_is_host_delegated)),
        None => String::new(),
    }
}

/// Render the memory-footprint note for an engine row: hot, cold, and the
/// compression ratio relative to a Standard KV (FP16) baseline.
///
/// `total` is what the engine reports owning. An engine that delegates
/// its K/V to a backend-resident cache (the coarse whole-model handle on
/// Metal, for instance) owns nothing itself and reports 0 — that is
/// "not accounted here", NOT "uses no memory", and a ratio computed
/// against it would be unbounded. Say so instead of printing a number:
/// a fabricated `0×` / `239×` in this column is worse than a blank.
pub(super) fn format_kv_memory_note(total: usize, cold: usize, kv_ref: usize) -> String {
    if total == 0 {
        return "engine-side kv unaccounted (backend-resident)".to_string();
    }
    let hot = total.saturating_sub(cold);
    let ratio = kv_ref as f64 / total as f64;
    // Whole numbers for the big compression wins, one decimal below the
    // threshold — see RATIO_DECIMAL_THRESHOLD for why that matters.
    let ratio_str = if ratio < RATIO_DECIMAL_THRESHOLD {
        format!("{ratio:.1}×")
    } else {
        format!("{ratio:.0}×")
    };
    format!(
        "hot={:.1}MB cold={:.1}MB  {ratio_str} vs std-kv",
        hot as f64 / BYTES_PER_MIB,
        cold as f64 / BYTES_PER_MIB,
    )
}

/// Bytes in the MB unit this table prints (mebibytes, matching how the
/// rest of the tooling reports model and cache sizes).
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

/// Below this the ratio gets a decimal place; at or above it, none.
/// A genuine 0.5x (an engine using twice the reference) must not print
/// as "0x", which reads as a missing measurement rather than a loss.
const RATIO_DECIMAL_THRESHOLD: f64 = 10.0;

/// Render the per-step split for an engine row: how much of the measured
/// step was the forward, and how much the lm_head + next-token pick.
///
/// Both halves are inside the timed step (see `engine_runtime`), so
/// `fwd = step - head`. Reporting the split keeps the forward-only
/// number recoverable without making `tok_per_s` incomparable against
/// the reference backend rows, which have always measured a whole token.
pub(super) fn format_step_split_note(step_ms: f64, head_ms: f64) -> String {
    let fwd_ms = (step_ms - head_ms).max(0.0);
    format!("fwd={fwd_ms:.2}ms head={head_ms:.2}ms")
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::kv_engine::DispatchPath;

    // ── format_kv_memory_note ────────────────────────────────────────────

    #[test]
    fn kv_memory_note_normal_case() {
        // total = 16 MB, cold = 4 MB → hot = 12 MB. Ratio = 64/16 = 4.
        let s = format_kv_memory_note(16 * 1024 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024);
        assert!(s.contains("hot=12.0MB"));
        assert!(s.contains("cold=4.0MB"));
        assert!(s.contains("4.0× vs std-kv"));
    }

    #[test]
    fn kv_memory_note_sub_unit_ratio_is_not_rounded_to_zero() {
        // An engine using 2× the reference is a real 0.5×; printing "0×"
        // made a loss indistinguishable from a missing measurement.
        let s = format_kv_memory_note(112 * 1024 * 1024, 0, 56 * 1024 * 1024);
        assert!(s.contains("0.5× vs std-kv"), "got {s:?}");
    }

    #[test]
    fn kv_memory_note_large_ratio_drops_the_decimal() {
        let s = format_kv_memory_note(1024 * 1024, 0, 240 * 1024 * 1024);
        assert!(s.contains("240× vs std-kv"), "got {s:?}");
    }

    #[test]
    fn kv_memory_note_zero_total_declines_to_invent_a_ratio() {
        // An engine that delegates K/V to a backend-resident cache reports
        // 0 bytes owned. The old code printed "0× vs std-kv", which reads
        // as a measurement; it isn't one.
        let s = format_kv_memory_note(0, 0, 1024);
        assert!(s.contains("unaccounted"), "got {s:?}");
        assert!(!s.contains("×"), "must not fabricate a ratio: {s:?}");
        assert!(
            !s.contains("hot="),
            "must not imply a hot-tier reading: {s:?}"
        );
    }

    #[test]
    fn kv_memory_note_clamps_hot_when_cold_exceeds_total() {
        // Engine bug guard: cold > total shouldn't underflow.
        let s = format_kv_memory_note(1024, 4096, 0);
        assert!(s.contains("hot=0.0MB"));
    }

    // ── format_dispatch_note ─────────────────────────────────────────────

    #[test]
    fn dispatch_note_is_empty_when_the_engine_reports_no_shape() {
        assert_eq!(format_dispatch_note(None, true), "");
        assert_eq!(format_dispatch_note(None, false), "");
    }

    #[test]
    fn dispatch_note_marks_host_delegated_per_layer() {
        assert_eq!(
            format_dispatch_note(Some(DispatchPath::PerLayer), true),
            "[per-layer→host] "
        );
    }

    #[test]
    fn dispatch_note_leaves_native_per_layer_unmarked() {
        assert_eq!(
            format_dispatch_note(Some(DispatchPath::PerLayer), false),
            "[per-layer] "
        );
    }

    #[test]
    fn dispatch_note_never_marks_coarse_as_host() {
        // Coarse doesn't touch the per-layer surface, so the backend's
        // delegation answer must not leak into its label.
        assert_eq!(
            format_dispatch_note(Some(DispatchPath::Coarse), true),
            "[coarse] "
        );
    }

    // ── format_step_split_note ───────────────────────────────────────────

    #[test]
    fn step_split_note_reports_fwd_as_step_minus_head() {
        let s = format_step_split_note(10.0, 6.5);
        assert!(s.contains("fwd=3.50ms"), "got {s:?}");
        assert!(s.contains("head=6.50ms"), "got {s:?}");
    }

    #[test]
    fn step_split_note_clamps_negative_fwd() {
        // Timer noise could make head marginally exceed the step total;
        // clamp rather than print a negative forward cost.
        let s = format_step_split_note(1.0, 1.2);
        assert!(s.contains("fwd=0.00ms"), "got {s:?}");
    }
}
