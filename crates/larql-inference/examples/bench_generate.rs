//! Generate benchmark: CPU prefill → GPU decode loop.
//! Proves the compute crate's 59 tok/s on a real model.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference --example bench_generate -- \
//!     --vindex output/gemma3-4b-v2.vindex

use larql_inference::{
    generate, InferenceModel, CachedLayerGraph, default_backend,
};
use larql_inference::ffn::WeightFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
        i += 1;
    }

    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_lm_head(&vindex_path)?;
    let _ = index.load_lm_head_q4(&vindex_path);
    let _ = index.load_attn_q4k(&vindex_path);
    let _ = index.load_attn_q8(&vindex_path);
    let _ = index.load_interleaved_q4(&vindex_path);
    let _ = index.load_interleaved_q4k(&vindex_path);

    let gpu_be = default_backend();
    let dense_ffn = WeightFfn { weights };
    let cached_layers: Vec<usize> = (0..=12).collect();
    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let cache = CachedLayerGraph::build(weights, &token_ids, &cached_layers, &dense_ffn);

    println!("╔═══════════════════════════════════════════════╗");
    println!("║       LARQL Generate Benchmark                ║");
    println!("╚═══════════════════════════════════════════════╝");
    println!();
    println!("  Prompt: \"{prompt}\" ({} tokens)", token_ids.len());
    println!("  Backend: {}", gpu_be.name());
    println!("  Layers: {} (cached 0-12, compute 13-{})", num_layers, num_layers - 1);
    println!();

    let result = generate(
        weights, tokenizer, &token_ids, 20,
        &index, &*gpu_be, &cache, 13..num_layers,
    );

    println!("  Prefill:       {:.0}ms", result.prefill_ms);
    println!("  Generated:     \"{}\"", result.text());
    println!("  Tokens:        {}", result.tokens.len());
    println!();

    if !result.decode_ms.is_empty() {
        println!("  Decode timing:");
        for (i, ms) in result.decode_ms.iter().enumerate() {
            let tok = &result.tokens[i + 1].0;
            println!("    Token {}: {:>8} {:>7.1}ms  ({:.0} tok/s)", i + 1, tok, ms, 1000.0 / ms);
        }
        println!();
        println!("  Average decode: {:.1}ms/tok = {:.0} tok/s", result.avg_decode_ms(), result.decode_tok_s());
    }

    println!();
    println!("  ┌───────────────────────────────────────────┐");
    println!("  │ Prefill: {:>6.0}ms (one-time)              │", result.prefill_ms);
    if result.decode_ms.is_empty() {
        println!("  │ Decode:  (no GPU decode tokens)           │");
    } else {
        println!("  │ Decode:  {:>6.1}ms/tok = {:>3.0} tok/s          │", result.avg_decode_ms(), result.decode_tok_s());
    }
    println!("  │ Ollama:    8.5ms/tok = 117 tok/s          │");
    println!("  └───────────────────────────────────────────┘");

    Ok(())
}
