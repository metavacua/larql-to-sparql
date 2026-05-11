//! Per-stage decode timing — the shape that replaces the deleted
//! `decode_profile.rs` duplicate.
//!
//! This module ships the **public API** ([`ProfileTimings`] +
//! [`MetalBackend::decode_token_with_profile`]) so that callers
//! (notably `larql-inference::layer_graph::generate` under
//! `LARQL_PROFILE_SPLIT=1`) can request per-stage timing without
//! a parallel decode path.
//!
//! Implementation (2026-05-02): when `LARQL_PROFILE_SPLIT=1` (or
//! `LARQL_DECODE_STAGE_TIMING=1`) is set, `decode_token_with_moe_split_fn`
//! inserts paired commit/wait boundaries between the attention block and
//! the FFN block on every layer. The resulting per-stage GPU times land
//! in a thread-local cell so [`MetalBackend::decode_token_split_profile`]
//! can read them back.
//!
//! Granularity today is **attention vs full FFN block**:
//! - `attn_ms` — Steps 1.5–5: QK-norm + RoPE + V-norm + KV append/attend
//!   + O proj + post-attn residual + ffn-input norm.
//! - `gate_up_ms` — the **entire FFN block**: gate + up + activation
//!   (GEGLU/SiLU) + down + post-FFN residual.
//! - `down_ms` — **0 for now**, reserved for the next-finer split that
//!   breaks `encode_ffn_step` into `gate_up` and `down` phases.
//!
//! Cost: ~2 commit/waits per layer × 34 = ~68/token of cmd-buffer
//! overhead (~2–3 ms on M3 Max). This is measurement-only mode; the
//! production decode path is unchanged when the env var is unset.

/// Per-stage wall-clock decode timings in milliseconds.
///
/// Filled by [`MetalBackend::decode_token_with_profile`]. Today
/// `attn_ms` carries the whole-token cost; per-stage split is on the
/// roadmap (see ROADMAP P1: "Restore per-stage decode profiling via a
/// `Profile` decorator").
#[derive(Debug, Default, Clone, Copy)]
pub struct ProfileTimings {
    /// Wall time for the attention side of the layer:
    /// input norm → QKV proj → QK-norm → RoPE → KV-attend → O proj.
    /// Today receives the whole-token cost as a placeholder.
    pub attn_ms: f64,
    /// Wall time for the FFN gate + up + activation. Zero today.
    pub gate_up_ms: f64,
    /// Wall time for the FFN down projection + post-FFN residual + scalar.
    /// Zero today.
    pub down_ms: f64,
}

/// True iff `LARQL_PROFILE_SPLIT=1` (or the legacy alias
/// `LARQL_DECODE_STAGE_TIMING=1`) is set in the environment. Decode
/// honours either flag for paired-commit per-stage profiling.
pub fn split_profile_requested() -> bool {
    crate::options::split_profile_requested()
}

thread_local! {
    /// Most recent per-stage timing recorded by
    /// `decode_token_with_moe_split_fn` when `LARQL_PROFILE_SPLIT=1`.
    /// `decode_token_split_profile` reads back from this cell.
    static LAST_SPLIT_TIMINGS: std::cell::Cell<Option<ProfileTimings>> =
        const { std::cell::Cell::new(None) };
}

/// Store the latest per-stage timing for the current thread. Called by
/// `decode_token_with_moe_split_fn` at the end of a token when
/// [`split_profile_requested`] returned true.
pub(crate) fn store_last_split_timings(t: ProfileTimings) {
    LAST_SPLIT_TIMINGS.with(|cell| cell.set(Some(t)));
}

/// Take and clear the most recent per-stage timing recorded on the
/// current thread. Returns `None` if `LARQL_PROFILE_SPLIT` was not set
/// for the most recent decode call.
pub fn take_last_split_timings() -> Option<ProfileTimings> {
    LAST_SPLIT_TIMINGS.with(|cell| cell.take())
}

impl ProfileTimings {
    /// Sum across the three buckets — the whole-token cost.
    pub fn total_ms(&self) -> f64 {
        self.attn_ms + self.gate_up_ms + self.down_ms
    }

    /// Format a `[profile-split] …` line in the same shape the old
    /// `decode_profile.rs` printed. Used by `larql-inference::generate`
    /// under `LARQL_PROFILE_SPLIT=1`.
    pub fn format_summary(&self, num_layers: usize) -> String {
        let total = self.total_ms();
        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
        let per_layer = if num_layers > 0 {
            total / num_layers as f64
        } else {
            0.0
        };
        format!(
            "[profile-split] {num_layers} layers — \
             attn={:.2}ms ({:.0}%)  gate+up={:.2}ms ({:.0}%)  \
             down={:.2}ms ({:.0}%)  total={:.2}ms ({per_layer:.3}ms/layer)",
            self.attn_ms,
            pct(self.attn_ms),
            self.gate_up_ms,
            pct(self.gate_up_ms),
            self.down_ms,
            pct(self.down_ms),
            total,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_ms_sums_buckets() {
        let p = ProfileTimings {
            attn_ms: 1.5,
            gate_up_ms: 2.5,
            down_ms: 1.0,
        };
        assert!((p.total_ms() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn format_summary_handles_zero_total() {
        let p = ProfileTimings::default();
        let s = p.format_summary(34);
        // No NaN-percent panics, total prints as 0.00.
        assert!(s.contains("total=0.00ms"));
        assert!(s.contains("34 layers"));
    }

    #[test]
    fn format_summary_includes_per_layer_average() {
        let p = ProfileTimings {
            attn_ms: 6.0,
            gate_up_ms: 3.0,
            down_ms: 1.0,
        };
        let s = p.format_summary(10);
        // total = 10.0, per-layer = 1.0
        assert!(s.contains("total=10.00ms"));
        assert!(s.contains("1.000ms/layer"));
    }
}
