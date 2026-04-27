//! Multi-Turn Simulation Demo
//!
//! Simulates a 25-turn conversation and shows how each strategy's
//! memory evolves. Produces the crossover plot data.
//!
//! Usage:
//!   cargo run --example multi_turn_demo

fn main() {
    use kv_cache_benchmark::*;
    use kv_cache_benchmark::benchmark;
    use kv_cache_benchmark::model_config::ModelConfig;
    use kv_cache_benchmark::standard_kv::StandardKv;
    use kv_cache_benchmark::turboquant::TurboQuant;
    use kv_cache_benchmark::markov_residual::MarkovResidual;
    use kv_cache_benchmark::graph_walk::GraphWalk;

    let config = ModelConfig::gemma_4b();
    let num_turns = 25;
    let tokens_per_turn = 150;

    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let graph = GraphWalk::gemma_4b();

    println!("=== Multi-Turn Memory Simulation: {} ===", config.name);
    println!("  {} turns, {} tokens/turn\n", num_turns, tokens_per_turn);

    // Header
    println!(
        "{:>5}  {:>8}  {:>12}  {:>12}  {:>12}  {:>12}",
        "Turn", "Tokens", "Standard KV", "TurboQ 4b", "Markov RS", "Graph Walk",
    );
    println!("{}", "-".repeat(80));

    for turn in 1..=num_turns {
        let cumulative = turn * tokens_per_turn;

        let mem_std = standard.memory_bytes(&config, cumulative);
        let mem_tq = tq4.memory_bytes(&config, cumulative);
        let mem_mrk = markov.memory_bytes(&config, cumulative);
        let mem_gw = graph.memory_bytes(cumulative);

        println!(
            "{:>5}  {:>8}  {:>12}  {:>12}  {:>12}  {:>12}",
            turn,
            cumulative,
            format_bytes(mem_std),
            format_bytes(mem_tq),
            format_bytes(mem_mrk),
            format_bytes(mem_gw),
        );
    }

    // Summary
    let final_tokens = num_turns * tokens_per_turn;
    println!("\n=== At {} tokens (turn {}) ===\n", final_tokens, num_turns);

    let strategies: Vec<(&str, usize)> = vec![
        ("Standard KV", standard.memory_bytes(&config, final_tokens)),
        ("TurboQuant 4b", tq4.memory_bytes(&config, final_tokens)),
        ("Markov RS", markov.memory_bytes(&config, final_tokens)),
        ("Graph Walk", graph.memory_bytes(final_tokens)),
    ];

    let baseline = strategies[0].1;
    for (name, mem) in &strategies {
        let ratio = if *mem > 0 { baseline as f64 / *mem as f64 } else { 0.0 };
        println!("  {:<15} {:>12}  ({:.1}× vs baseline)", name, format_bytes(*mem), ratio);
    }

    // Full comparative table (KV-reconstructing strategies only).
    let all: Vec<&dyn KvStrategy> = vec![&standard, &tq4, &markov];
    println!("{}", benchmark::format_comparative_table(&config, &all));

    // Crossover analysis
    println!("\n=== Crossover Analysis ===\n");
    println!("Standard KV grows linearly: every turn adds {} per token",
        format_bytes(config.kv_bytes_per_token()));
    println!("Markov RS is bounded: window = 512 tokens, cold tier = 4 bytes/token");
    println!("Graph Walk is constant: per-conversation = token IDs only (requires cracked attention)");

    // Find crossover point where Markov RS < Standard KV
    for turn in 1..=50 {
        let tokens = turn * tokens_per_turn;
        let std_mem = standard.memory_bytes(&config, tokens);
        let mrk_mem = markov.memory_bytes(&config, tokens);
        if mrk_mem < std_mem {
            println!("\nMarkov RS < Standard KV at turn {} ({} tokens)", turn, tokens);
            break;
        }
    }
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
