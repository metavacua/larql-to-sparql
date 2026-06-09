//! Per-stage timing for KV-cache engines.
//!
//! Enable by constructing engines with `with_profiling(true)`. Each decode
//! step accumulates per-stage wall-clock times; call `stage_summary()` after
//! decoding to retrieve averaged results.
//!
//! Overhead when disabled: one branch per stage (zero-cost in release builds
//! when the compiler inlines `if self.profiling { ... }`).

use std::time::Instant;

/// Accumulator for a single timing stage. Add new samples with `record`.
#[derive(Debug, Clone, Default)]
pub struct StageAccumulator {
    pub total_us: f64,
    pub count: usize,
}

impl StageAccumulator {
    pub fn record(&mut self, t: Instant) {
        self.total_us += t.elapsed().as_secs_f64() * 1e6;
        self.count += 1;
    }

    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_us / self.count as f64
        }
    }
}

/// Per-step averages for a completed engine run.
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
        println!("  {}", "-".repeat(45));
        println!(
            "  {:<25} {:>8.1}  {:>5.1}%",
            "total (measured)", total, 100.0
        );
        println!();
    }
}

/// Per-engine profiling state.
/// Field layout matches `MarkovResidualEngine` — add more engines as needed.
#[derive(Debug, Default)]
pub struct EngineProfiler {
    pub embed: StageAccumulator,
    pub recompute_cold: StageAccumulator,
    pub recompute_hot: StageAccumulator,
    pub attention: StageAccumulator,
    pub ffn: StageAccumulator,
    pub decode_total: StageAccumulator,
}

impl EngineProfiler {
    pub fn summary(&self, engine: &str, backend: &str) -> DecodeStageSummary {
        DecodeStageSummary {
            engine: engine.to_string(),
            backend: backend.to_string(),
            steps: self.decode_total.count,
            avg_embed_us: self.embed.avg_us(),
            avg_recompute_cold_us: self.recompute_cold.avg_us(),
            avg_recompute_hot_us: self.recompute_hot.avg_us(),
            avg_attention_us: self.attention.avg_us(),
            avg_ffn_us: self.ffn.avg_us(),
            avg_total_decode_us: self.decode_total.avg_us(),
        }
    }
}
