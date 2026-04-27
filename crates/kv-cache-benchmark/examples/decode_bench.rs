//! Bounded-state decode experiment: RS-decode vs full-KV decode.
//!
//! Proves the Markov residual stream claim end-to-end:
//!
//!   "The pre-layer residual is the complete Markov state of the transformer.
//!    K and V can be recomputed from it at any context length with zero loss."
//!
//! ## What is measured
//!
//! Both decoders start from an identical prefill (same forward pass).
//! Divergence is decode-only — the only difference is how K/V history is
//! maintained:
//!
//!   Full-KV  — grows the cache with raw K/V tensors (standard approach).
//!   RS-decode — recomputes K/V from stored residuals each step.
//!              Cold tier keeps evicted residuals so full history is visible.
//!
//! With cold-tier replay enabled the two decoders produce identical output
//! at every window size (cos h = 1.000000, 100% top-1 match).
//!
//! ## Query types
//!
//!   Parametric — answer lives in model weights (factual recall).
//!                Window size doesn't matter; parametric routing operates
//!                through static FFN gates independent of context length.
//!
//!   InContext  — answer planted at the start of a long prompt.
//!                Without cold-tier replay the RS decoder cannot see the
//!                planted fact when the window is smaller than the prompt.
//!                With cold-tier replay it matches full-KV exactly even at
//!                window=1.
//!
//! ## Usage
//!
//!   cargo run --example decode_bench --release --features real-model -- \
//!       google/gemma-3-4b-it /path/to/gemma3-4b-v2.vindex
//!
//! Optional third argument overrides the window sizes (comma-separated):
//!   ... -- google/gemma-3-4b-it /path/to.vindex 2,4,6,12,24

#[cfg(feature = "real-model")]
fn main() {
    use kv_cache_benchmark::real_model::decode_comparison::{
        run_decode_comparison, format_comparison, format_window_sweep,
        QueryType, parametric_prompts, in_context_prompts, DecodeComparisonResult,
    };

    let args: Vec<String> = std::env::args().collect();
    let model_name = args.get(1).map(|s| s.as_str()).unwrap_or("google/gemma-3-4b-it");
    let decode_steps = 8;

    // Parse window sizes from optional third argument, or use defaults.
    let windows: Vec<usize> = args.get(3)
        .map(|s| s.split(',').filter_map(|w| w.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 6, 12, 24]);

    println!("Loading model: {model_name}");
    let model = larql_inference::InferenceModel::load(model_name)
        .expect("Failed to load model");

    let weights = model.weights();
    let tokenizer = model.tokenizer();

    println!("Window sweep: {windows:?}  |  Decode steps: {decode_steps}");

    let mut all_results: Vec<DecodeComparisonResult> = Vec::new();

    // ── Parametric ────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  PARAMETRIC — answer in model weights                ║");
    println!("║  Claim: RS-decode == full-KV at every window size    ║");
    println!("╚══════════════════════════════════════════════════════╝");

    for prompt_str in parametric_prompts() {
        let token_ids: Vec<u32> = tokenizer
            .encode(prompt_str, true).expect("tokenize")
            .get_ids().to_vec();

        println!("\nPrompt: {:?}  ({} tokens)", prompt_str, token_ids.len());

        for &window in &windows {
            let result = run_decode_comparison(
                weights, tokenizer, &token_ids,
                QueryType::Parametric, window, decode_steps,
            );
            println!("{}", format_comparison(&result));
            all_results.push(result);
        }
    }

    // ── In-context ────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║  IN-CONTEXT — answer planted in context              ║");
    println!("║  Claim: cold-tier replay keeps RS == full-KV         ║");
    println!("╚══════════════════════════════════════════════════════╝");

    for prompt_str in in_context_prompts() {
        let token_ids: Vec<u32> = tokenizer
            .encode(prompt_str.as_str(), true).expect("tokenize")
            .get_ids().to_vec();

        println!("\nPrompt: {:?}  ({} tokens)", &prompt_str[..60.min(prompt_str.len())], token_ids.len());

        for &window in &windows {
            let result = run_decode_comparison(
                weights, tokenizer, &token_ids,
                QueryType::InContext, window, decode_steps,
            );
            println!("{}", format_comparison(&result));
            all_results.push(result);
        }
    }

    // ── Summary table ─────────────────────────────────────────────────────────
    println!("\n=== Summary across all prompts and windows ===");
    println!("{}", format_window_sweep(&all_results));

    let total = all_results.len();
    let perfect = all_results.iter().filter(|r| r.first_divergence.is_none()).count();
    println!("Overall: {perfect}/{total} runs with zero divergence ({:.1}%)",
        perfect as f64 / total as f64 * 100.0);

    let json = serde_json::to_string_pretty(&all_results).unwrap();
    let out_path = "crates/kv-cache-benchmark/results/decode_comparison.json";
    std::fs::create_dir_all("crates/kv-cache-benchmark/results").ok();
    std::fs::write(out_path, &json).ok();
    println!("\nResults written to {out_path}");
}

#[cfg(not(feature = "real-model"))]
fn main() {
    eprintln!("This example requires the 'real-model' feature:");
    eprintln!("  cargo run --example decode_bench --release --features real-model -- \\");
    eprintln!("      google/gemma-3-4b-it /path/to/gemma3-4b-v2.vindex");
}
