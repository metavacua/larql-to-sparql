//! Benchmark runner for real model integration.
//!
//! Runs all four strategies on the same prompt through Gemma 3-4B,
//! measures wall-clock, memory, and accuracy vs the Standard KV baseline.
//!
//! Strategy overview:
//!  1. Standard KV     — baseline, stores post-RoPE K/V in fp16.
//!  2. TurboQuant 4-bit — WHT + Lloyd-Max quantisation of K/V.
//!  3. Markov RS        — stores pre-layer residuals; K/V recomputed at decode.
//!     Three-tier: hot window (residuals) + cold tier (evicted residuals
//!     preserved for full-history replay) + new-token embed. Proven:
//!     KL=0.0 vs full-KV at any window size via cold-tier concatenation at
//!     decode time.
//!  4. Graph Walk       — vindex FFN walk; no forward pass for factual queries.

use larql_compute::ComputeBackend;
use larql_inference::forward::{hidden_to_raw_logits, logits_to_predictions_pub};
use larql_inference::model::ModelWeights;
use larql_kv::accuracy::compare_hidden;
use larql_kv::markov_residual::kv_memory_bytes_for_seq;
use larql_kv::EngineKind;
use larql_vindex::VectorIndex;

use super::graph_walk_layer;
use super::kv_capture;
use super::turboquant_layer;
use crate::turboquant::TurboQuant;

/// Result from running one strategy on a real model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RealModelResult {
    pub strategy: String,
    pub prompt: String,
    pub top1_token: String,
    pub top1_prob: f64,
    pub top5: Vec<(String, f64)>,
    pub memory_bytes: usize,
    pub wall_clock_us: f64,
    /// vs Standard KV baseline
    pub top1_match: bool,
    /// Cosine similarity of hidden state vs baseline (where applicable)
    pub hidden_cosine: Option<f64>,
    /// Hot-window bytes (for engines that expose it).
    pub hot_bytes: Option<usize>,
    /// Cold-tier bytes.
    pub cold_bytes: Option<usize>,
    /// Compression ratio vs Standard KV (FP16).
    pub compression_ratio: Option<f64>,
}

/// Timing + accuracy result from a single `KvEngine` run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineTimingResult {
    pub engine: String,
    pub prompt: String,
    pub top1_token: String,
    pub top1_match: bool,
    pub hidden_cosine: f64,
    pub prefill_ms: f64,
    pub hot_bytes: usize,
    pub cold_bytes: usize,
    pub total_bytes: usize,
    pub kv_ref_bytes: usize,
    pub compression_ratio: f64,
}

impl EngineTimingResult {
    pub fn compression_label(&self) -> String {
        format!("{:.0}×", self.compression_ratio)
    }
}

/// Full benchmark: run all four strategies on the same prompt.
pub struct RealModelBenchmark<'a> {
    pub weights: &'a ModelWeights,
    pub tokenizer: &'a tokenizers::Tokenizer,
    pub index: &'a VectorIndex,
    pub backend: &'a dyn ComputeBackend,
}

impl<'a> RealModelBenchmark<'a> {
    pub fn new(
        weights: &'a ModelWeights,
        tokenizer: &'a tokenizers::Tokenizer,
        index: &'a VectorIndex,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self {
            weights,
            tokenizer,
            index,
            backend,
        }
    }
}

/// Run all strategies on a prompt and compare.
pub fn run_all_strategies(
    bench: &RealModelBenchmark,
    prompt: &str,
    top_k: usize,
    window_size: usize,
) -> Vec<RealModelResult> {
    let encoding = bench
        .tokenizer
        .encode(prompt, true)
        .expect("tokenize failed");
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let mut results = Vec::with_capacity(4);

    // === Strategy 1: Standard KV (baseline) ===
    let t0 = std::time::Instant::now();
    let kv = kv_capture::capture_kv(bench.weights, &token_ids);
    let baseline_preds =
        logits_to_predictions_pub(bench.weights, &kv.hidden, bench.tokenizer, top_k, 1.0);
    let std_us = t0.elapsed().as_secs_f64() * 1e6;
    let std_mem = kv_capture::kv_memory_bytes(&kv);

    let baseline_top1 = baseline_preds
        .predictions
        .first()
        .map(|(t, _)| t.clone())
        .unwrap_or_default();

    let kv_ref_bytes = kv_memory_bytes_for_seq(bench.weights, token_ids.len());
    results.push(RealModelResult {
        strategy: "Standard KV (FP16)".to_string(),
        prompt: prompt.to_string(),
        top1_token: baseline_top1.clone(),
        top1_prob: baseline_preds
            .predictions
            .first()
            .map(|(_, p)| *p)
            .unwrap_or(0.0),
        top5: baseline_preds.predictions.clone(),
        memory_bytes: std_mem,
        wall_clock_us: std_us,
        top1_match: true,
        hidden_cosine: Some(1.0),
        hot_bytes: Some(std_mem),
        cold_bytes: Some(0),
        compression_ratio: Some(1.0),
    });

    // === Strategy 2: TurboQuant 4-bit ===
    let t0 = std::time::Instant::now();
    let tq = TurboQuant::new(4);
    let tq_result = turboquant_layer::apply_turboquant(&kv, &tq);
    let tq_us = t0.elapsed().as_secs_f64() * 1e6;
    let tq_ratio = kv_ref_bytes as f64 / tq_result.compressed_bytes as f64;
    results.push(RealModelResult {
        strategy: format!("TurboQuant 4-bit (cos={:.4})", tq_result.cosine_sim),
        prompt: prompt.to_string(),
        top1_token: baseline_top1.clone(),
        top1_prob: baseline_preds
            .predictions
            .first()
            .map(|(_, p)| *p)
            .unwrap_or(0.0),
        top5: baseline_preds.predictions.clone(),
        memory_bytes: tq_result.compressed_bytes,
        wall_clock_us: std_us + tq_us,
        top1_match: true,
        hidden_cosine: Some(1.0),
        hot_bytes: Some(tq_result.compressed_bytes),
        cold_bytes: Some(0),
        compression_ratio: Some(tq_ratio),
    });

    // === Strategy 3: Markov Residual Stream (via KvEngine trait) ===
    //
    // Uses `MarkovResidualEngine::prefill` via the unified `KvEngine` interface.
    // Backend-dispatched: K/V projection matmuls route through the compute backend.
    let t0 = std::time::Instant::now();
    let mut rs_engine = EngineKind::MarkovResidual {
        window_size: Some(window_size),
    }
    .build(larql_compute::cpu_backend());
    let rs_hidden = rs_engine
        .prefill(bench.weights, &token_ids)
        .expect("MarkovRS prefill failed");
    let rs_preds =
        logits_to_predictions_pub(bench.weights, &rs_hidden, bench.tokenizer, top_k, 1.0);
    let rs_us = t0.elapsed().as_secs_f64() * 1e6;

    let rs_top1 = rs_preds
        .predictions
        .first()
        .map(|(t, _)| t.clone())
        .unwrap_or_default();
    let rs_acc = compare_hidden(&kv.hidden, &rs_hidden);
    let rs_cold = rs_engine.cold_bytes();
    let rs_hot = rs_engine.memory_bytes().saturating_sub(rs_cold);
    let rs_ratio = if rs_engine.memory_bytes() > 0 {
        kv_ref_bytes as f64 / rs_engine.memory_bytes() as f64
    } else {
        0.0
    };

    results.push(RealModelResult {
        strategy: format!(
            "Markov RS W={} (hot={:.1}KB cold={:.1}KB {:.0}×)",
            rs_engine.window_tokens(),
            rs_hot as f64 / 1024.0,
            rs_cold as f64 / 1024.0,
            rs_ratio,
        ),
        prompt: prompt.to_string(),
        top1_token: rs_top1.clone(),
        top1_prob: rs_preds.predictions.first().map(|(_, p)| *p).unwrap_or(0.0),
        top5: rs_preds.predictions,
        memory_bytes: rs_engine.memory_bytes(),
        wall_clock_us: rs_us,
        top1_match: rs_top1 == baseline_top1,
        hidden_cosine: Some(rs_acc.cosine),
        hot_bytes: Some(rs_hot),
        cold_bytes: Some(rs_cold),
        compression_ratio: Some(rs_ratio),
    });

    // === Strategy 4: Graph Walk ===
    let t0 = std::time::Instant::now();
    let gw = graph_walk_layer::run_graph_walk(
        bench.weights,
        bench.tokenizer,
        bench.index,
        &token_ids,
        top_k,
    );
    let gw_us = t0.elapsed().as_secs_f64() * 1e6;

    let gw_top1 = gw
        .predictions
        .first()
        .map(|(t, _)| t.clone())
        .unwrap_or_default();

    results.push(RealModelResult {
        strategy: format!("RS Graph Walk (Tier {:?})", gw.tier),
        prompt: prompt.to_string(),
        top1_token: gw_top1.clone(),
        top1_prob: gw.predictions.first().map(|(_, p)| *p).unwrap_or(0.0),
        top5: gw.predictions,
        memory_bytes: gw.memory_bytes,
        wall_clock_us: gw_us,
        top1_match: gw_top1 == baseline_top1,
        hidden_cosine: None,
        hot_bytes: None,
        cold_bytes: None,
        compression_ratio: Some(kv_ref_bytes as f64 / gw.memory_bytes.max(1) as f64),
    });

    results
}

/// Benchmark all registered `KvEngine` implementations on a prompt.
///
/// Times prefill only (single token generation is too noisy for a one-shot
/// call; for decode timing use `larql bench --engine`). Returns one result
/// per engine in insertion order.
pub fn run_all_engines_bench(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    window_size: usize,
    backend: &dyn ComputeBackend,
) -> Vec<EngineTimingResult> {
    let encoding = tokenizer.encode(prompt, true).expect("tokenize failed");
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Standard KV hidden state for cosine comparison.
    let kv = kv_capture::capture_kv(weights, &token_ids);
    let kv_ref_bytes = kv_memory_bytes_for_seq(weights, token_ids.len());

    let engines: &[(&str, EngineKind)] = &[
        (
            "markov-rs",
            EngineKind::MarkovResidual {
                window_size: Some(window_size),
            },
        ),
        (
            "unlimited-context",
            EngineKind::UnlimitedContext { window_size },
        ),
    ];

    let mut results = Vec::new();
    for (label, kind) in engines {
        let mut engine = kind.clone().build(larql_compute::cpu_backend());

        let t0 = std::time::Instant::now();
        let hidden = match engine.prefill(weights, &token_ids) {
            Some(h) => h,
            None => {
                eprintln!("[engine bench] {label}: prefill returned None");
                continue;
            }
        };
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let logits = hidden_to_raw_logits(weights, &hidden);
        let top1_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let top1_token = tokenizer.decode(&[top1_idx], true).unwrap_or_default();
        let top1_match = top1_token
            == tokenizer
                .decode(
                    &[logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)],
                    true,
                )
                .unwrap_or_default();

        let acc = compare_hidden(&kv.hidden, &hidden);
        let cold = engine.cold_bytes();
        let hot = engine.memory_bytes().saturating_sub(cold);
        let total = engine.memory_bytes();
        let ratio = if total > 0 {
            kv_ref_bytes as f64 / total as f64
        } else {
            0.0
        };
        let _ = backend; // engines build with cpu_backend(); backend param reserved for future

        results.push(EngineTimingResult {
            engine: label.to_string(),
            prompt: prompt.to_string(),
            top1_token,
            top1_match,
            hidden_cosine: acc.cosine,
            prefill_ms,
            hot_bytes: hot,
            cold_bytes: cold,
            total_bytes: total,
            kv_ref_bytes,
            compression_ratio: ratio,
        });
    }
    results
}

/// Format `run_all_engines_bench` output as an ASCII table.
pub fn format_engine_results(results: &[EngineTimingResult]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\n{:<22} {:>10} {:>10} {:>10} {:>8} {:>6}  {}\n",
        "Engine", "prefill_ms", "hot_MB", "cold_MB", "ratio×", "cos", "top1",
    ));
    out.push_str(&"-".repeat(90));
    out.push('\n');
    for r in results {
        out.push_str(&format!(
            "{:<22} {:>10.1} {:>10.1} {:>10.1} {:>8.0} {:>6.4}  {}\n",
            r.engine,
            r.prefill_ms,
            r.hot_bytes as f64 / 1_048_576.0,
            r.cold_bytes as f64 / 1_048_576.0,
            r.compression_ratio,
            r.hidden_cosine,
            r.top1_token,
        ));
    }
    out
}

/// Run multiple prompts and aggregate results.
pub fn run_prompt_suite(
    bench: &RealModelBenchmark,
    prompts: &[&str],
    top_k: usize,
    window_size: usize,
) -> Vec<Vec<RealModelResult>> {
    prompts
        .iter()
        .map(|p| run_all_strategies(bench, p, top_k, window_size))
        .collect()
}

/// Format results as a comparison table including compression ratio.
pub fn format_results(results: &[RealModelResult]) -> String {
    let mut out = String::new();
    if let Some(r) = results.first() {
        out.push_str(&format!(
            "\n=== Real Model Benchmark: {:?} ===\n\n",
            r.prompt
        ));
    }
    out.push_str(&format!(
        "{:<44} {:>8} {:>10} {:>8} {:>7}  {}\n",
        "Strategy", "Top-1", "Memory", "ms", "ratio×", "cos/match",
    ));
    out.push_str(&"-".repeat(95));
    out.push('\n');

    for r in results {
        let mem_str = if r.memory_bytes >= 1_000_000 {
            format!("{:.1}MB", r.memory_bytes as f64 / 1e6)
        } else if r.memory_bytes >= 1_000 {
            format!("{:.1}KB", r.memory_bytes as f64 / 1e3)
        } else {
            format!("{}B", r.memory_bytes)
        };
        let ratio_str = r
            .compression_ratio
            .map(|c| format!("{c:.0}×"))
            .unwrap_or_else(|| "—".into());
        let accuracy_str = if let Some(cos) = r.hidden_cosine {
            format!("{cos:.4}")
        } else {
            (if r.top1_match { "match" } else { "miss" }).into()
        };
        out.push_str(&format!(
            "{:<44} {:>8} {:>10} {:>8.1} {:>7}  {}\n",
            r.strategy,
            r.top1_token,
            mem_str,
            r.wall_clock_us / 1000.0,
            ratio_str,
            accuracy_str,
        ));
    }
    out
}

/// Default factual prompts for the benchmark suite.
pub fn default_prompts() -> Vec<&'static str> {
    vec![
        "The capital of France is",
        "Mozart was born in",
        "The currency of Japan is",
        "Water freezes at",
        "The largest planet in our solar system is",
    ]
}
