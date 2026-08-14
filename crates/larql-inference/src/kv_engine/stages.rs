//! [`DecodeStageSummary`] — per-step timing averages for a completed run.

/// Per-step averages for a completed engine run. Returned from
/// [`KvEngine::stage_summary`] when profiling was enabled at engine
/// construction.
#[derive(Debug, Clone)]
pub struct DecodeStageSummary {
    pub engine: String,
    pub backend: String,
    pub steps: usize,
    pub avg_embed_us: f64,
    /// K/V recompute from stored residuals (MarkovRS only). Split by tier.
    pub avg_recompute_cold_us: f64,
    pub avg_recompute_hot_us: f64,
    pub avg_attention_us: f64,
    pub avg_ffn_us: f64,
    pub avg_total_decode_us: f64,
    /// W10 instrumentation: time spent inside the backend's
    /// `coarse_decode_step_with_state_masked` call — kernel run +
    /// state-dump readback (skipped under HOnly / None). Zero on
    /// non-dispatch paths and on engines that don't capture state.
    pub avg_state_capture_us: f64,
    /// W10 instrumentation: cumulative time inside per-layer handle
    /// materialise calls (`StateHandle::into_array`). Tracks the
    /// CPU bridge cost from the captured dump to engine-owned
    /// `Array2`s. Zero under None mask (engine drops handles
    /// without materialising).
    pub avg_state_materialise_us: f64,
    /// W10 instrumentation: cumulative time appending materialised
    /// state into engine slabs (`append_row` calls). Tracks
    /// `rs.stored` / `rs.hot_kv` growth. Zero under None mask.
    pub avg_state_append_us: f64,
}

impl DecodeStageSummary {
    pub fn avg_recompute_total_us(&self) -> f64 {
        self.avg_recompute_cold_us + self.avg_recompute_hot_us
    }

    /// Print a human-readable breakdown table.
    pub fn print(&self) {
        let total = self.avg_total_decode_us;
        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };

        println!(
            "\nStage breakdown  ({}, {}, {} decode steps avg):",
            self.engine, self.backend, self.steps
        );
        println!("  {:<25} {:>8}  {:>6}", "Stage", "avg_us", "%");
        println!("  {}", "-".repeat(45));
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "embed",
            self.avg_embed_us,
            pct(self.avg_embed_us)
        );
        if self.avg_recompute_total_us() > 0.0 {
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "recompute_kv (cold)",
                self.avg_recompute_cold_us,
                pct(self.avg_recompute_cold_us)
            );
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "recompute_kv (hot)",
                self.avg_recompute_hot_us,
                pct(self.avg_recompute_hot_us)
            );
        }
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "attention",
            self.avg_attention_us,
            pct(self.avg_attention_us)
        );
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "ffn",
            self.avg_ffn_us,
            pct(self.avg_ffn_us)
        );
        // W10 instrumentation: only print state lines when populated
        // (avoids noise on engines that don't capture state).
        let state_total =
            self.avg_state_capture_us + self.avg_state_materialise_us + self.avg_state_append_us;
        if state_total > 0.0 {
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "state_capture",
                self.avg_state_capture_us,
                pct(self.avg_state_capture_us)
            );
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "state_materialise",
                self.avg_state_materialise_us,
                pct(self.avg_state_materialise_us)
            );
            println!(
                "  {:<25} {:>8.1}  {:>5.1}%",
                "state_append",
                self.avg_state_append_us,
                pct(self.avg_state_append_us)
            );
        }
        println!("  {}", "-".repeat(45));
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "total (measured)", total, 100.0
        );
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_stage_summary_recompute_total() {
        let s = DecodeStageSummary {
            engine: "test".into(),
            backend: "cpu".into(),
            steps: 10,
            avg_embed_us: 1.0,
            avg_recompute_cold_us: 2.0,
            avg_recompute_hot_us: 3.0,
            avg_attention_us: 4.0,
            avg_ffn_us: 5.0,
            avg_total_decode_us: 15.0,
            avg_state_capture_us: 0.0,
            avg_state_materialise_us: 0.0,
            avg_state_append_us: 0.0,
        };
        assert_eq!(s.avg_recompute_total_us(), 5.0);
    }

    /// Cover `DecodeStageSummary::print` — both the recompute>0 branch and
    /// the total>0 percentage branch. Output goes to stdout (captured by the
    /// test harness); this is a smoke test for the formatting code path.
    #[test]
    fn decode_stage_summary_print_with_recompute() {
        let s = DecodeStageSummary {
            engine: "markov-rs".into(),
            backend: "cpu".into(),
            steps: 10,
            avg_embed_us: 100.0,
            avg_recompute_cold_us: 500.0,
            avg_recompute_hot_us: 300.0,
            avg_attention_us: 1500.0,
            avg_ffn_us: 800.0,
            avg_total_decode_us: 3200.0,
            avg_state_capture_us: 0.0,
            avg_state_materialise_us: 0.0,
            avg_state_append_us: 0.0,
        };
        s.print();
    }

    /// `print` must also handle the no-recompute, zero-total branch — the
    /// `pct` fallback when `avg_total_decode_us == 0.0` and the
    /// `avg_recompute_total_us() == 0` short-circuit.
    #[test]
    fn decode_stage_summary_print_no_recompute_zero_total() {
        let s = DecodeStageSummary {
            engine: "no-cache".into(),
            backend: "metal".into(),
            steps: 0,
            avg_embed_us: 0.0,
            avg_recompute_cold_us: 0.0,
            avg_recompute_hot_us: 0.0,
            avg_attention_us: 0.0,
            avg_ffn_us: 0.0,
            avg_total_decode_us: 0.0,
            avg_state_capture_us: 0.0,
            avg_state_materialise_us: 0.0,
            avg_state_append_us: 0.0,
        };
        s.print();
    }
}
