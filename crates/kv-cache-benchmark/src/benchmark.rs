/// Benchmark runner: sweeps context lengths × strategies × models.
/// Outputs JSON + formatted table.

use crate::{KvStrategy, StrategyResult, run_strategy_benchmark, model_config::ModelConfig};
use rand::prelude::*;

/// Context lengths to sweep.
pub const CONTEXT_LENGTHS: &[usize] = &[
    512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 370_000,
];

/// Run all strategies across all context lengths for a given model.
pub fn run_sweep(
    config: &ModelConfig,
    strategies: &[&dyn KvStrategy],
    context_lengths: &[usize],
    seed: u64,
) -> Vec<StrategyResult> {
    let mut results = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);

    for &seq_len in context_lengths {
        for strategy in strategies {
            let result = run_strategy_benchmark(*strategy, config, seq_len, &mut rng);
            results.push(result);
        }
    }

    results
}

/// Memory-only sweep (no encode/decode, just analytical formula).
/// Fast — can run for all models including 70B.
pub fn memory_sweep(
    config: &ModelConfig,
    strategies: &[&dyn KvStrategy],
    context_lengths: &[usize],
) -> Vec<MemoryPoint> {
    let mut points = Vec::new();

    for &seq_len in context_lengths {
        for strategy in strategies {
            points.push(MemoryPoint {
                strategy_name: strategy.name().to_string(),
                model_name: config.name.to_string(),
                seq_len,
                memory_bytes: strategy.memory_bytes(config, seq_len),
            });
        }
    }

    points
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryPoint {
    pub strategy_name: String,
    pub model_name: String,
    pub seq_len: usize,
    pub memory_bytes: usize,
}

/// Multi-turn simulation result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MultiTurnResult {
    pub strategy_name: String,
    pub turn: usize,
    pub cumulative_tokens: usize,
    pub memory_bytes: usize,
    pub wall_clock_us: f64,
}

/// Simulate a multi-turn conversation.
pub fn multi_turn_simulation(
    config: &ModelConfig,
    strategies: &[&dyn KvStrategy],
    num_turns: usize,
    tokens_per_turn: usize,
    seed: u64,
) -> Vec<MultiTurnResult> {
    let mut results = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let dim = config.kv_dim();

    for strategy in strategies {
        let mut cumulative_tokens = 0;

        for turn in 1..=num_turns {
            cumulative_tokens += tokens_per_turn;
            let num_vectors = cumulative_tokens * config.layers * config.kv_heads;

            // Generate vectors for this turn's cumulative context
            let keys: Vec<Vec<f32>> = (0..num_vectors.min(1000)) // Cap to keep fast
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
                .collect();
            let values: Vec<Vec<f32>> = (0..num_vectors.min(1000))
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
                .collect();

            let t0 = std::time::Instant::now();
            let encoded = strategy.encode(&keys, &values);
            let _ = strategy.decode(&encoded, keys.len(), dim);
            let wall_clock_us = t0.elapsed().as_secs_f64() * 1e6;

            results.push(MultiTurnResult {
                strategy_name: strategy.name().to_string(),
                turn,
                cumulative_tokens,
                memory_bytes: strategy.memory_bytes(config, cumulative_tokens),
                wall_clock_us,
            });
        }
    }

    results
}

/// Format results as a comparative table (the video frame).
pub fn format_comparative_table(
    config: &ModelConfig,
    strategies: &[&dyn KvStrategy],
) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n=== KV Cache Strategy Comparison: {} ===\n\n", config.name));

    // Dynamic header based on number of strategies
    let col_width = 15;
    out.push_str(&format!("{:<25}", "Context Length"));
    for s in strategies {
        out.push_str(&format!(" {:>width$}", s.name(), width = col_width));
    }
    out.push('\n');
    out.push_str(&"-".repeat(25 + strategies.len() * (col_width + 1)));
    out.push('\n');

    for &seq_len in &[512, 4096, 32768, 131072, 370_000usize] {
        out.push_str(&format!("{:<25}", format_tokens(seq_len)));
        for strategy in strategies {
            let mem = strategy.memory_bytes(config, seq_len);
            out.push_str(&format!(" {:>width$}", format_bytes(mem), width = col_width));
        }
        out.push('\n');
    }

    out.push_str("\n--- Computation per token ---\n\n");
    // 5-column computation table
    let ops = [
        //                     StdKV         TQ4           MarkovRS      BoundaryRS    GraphWalk
        ("Attention matmul", "34 layers",  "34 layers",  "window only","window only","ELIMINATED*"),
        ("FFN matmul",       "34 layers",  "34 layers",  "34 layers",  "34 layers",  "ELIMINATED*"),
        ("Logits matmul",    "1x",         "1x",         "1x",         "1x",         "ELIMINATED*"),
        ("KV cache write",   "34 layers",  "34L + quant","none",       "none",       "none"),
        ("Cold K/V replay",  "none",       "none",       "none",       "bdy+ids",    "none"),
        ("Graph lookup",     "none",       "none",       "none",       "none",       "3 per hop*"),
    ];

    out.push_str(&format!(
        "{:<20} {:>14} {:>14} {:>12} {:>13} {:>13}\n",
        "Operation", "Standard KV", "TurboQuant", "Markov RS", "Boundary RS", "Graph Walk"
    ));
    out.push_str(&"-".repeat(92));
    out.push('\n');

    for (op, std, tq, mrs, brs, gw) in &ops {
        out.push_str(&format!(
            "{:<20} {:>14} {:>14} {:>12} {:>13} {:>13}\n",
            op, std, tq, mrs, brs, gw,
        ));
    }

    out.push_str("\n* Graph Walk requires cracked attention (not yet implemented). Falls back to Markov RS.\n");

    out
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1e3)
    } else {
        format!("{} B", bytes)
    }
}

fn format_tokens(tokens: usize) -> String {
    if tokens >= 1_000 {
        format!("{}K tokens", tokens / 1000)
    } else {
        format!("{} tokens", tokens)
    }
}

/// Write results to JSON file.
pub fn write_json(results: &[StrategyResult], path: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(path, json)
}

/// Write memory sweep to JSON file.
pub fn write_memory_json(points: &[MemoryPoint], path: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(points)?;
    std::fs::write(path, json)
}
