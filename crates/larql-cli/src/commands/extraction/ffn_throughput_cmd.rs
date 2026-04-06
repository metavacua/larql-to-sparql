use std::time::Instant;

use clap::Args;
use larql_inference::{
    CachedFfn, InferenceModel, FfnBackend,
};

#[derive(Args)]
pub struct FfnThroughputArgs {
    /// Model path or HuggingFace model ID.
    #[arg(short, long)]
    model: String,

    /// Prompt to calibrate cache from.
    #[arg(short, long, default_value = "The capital of France is")]
    prompt: String,

    /// Number of tokens to simulate.
    #[arg(short, long, default_value = "100000")]
    tokens: usize,
}

pub fn run(args: FfnThroughputArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;

    let encoding = model.tokenizer().encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    eprintln!("Calibrating cache...");
    let cached_ffn = CachedFfn::calibrate(weights, &token_ids);

    // Method 1: Current approach — clone per call via FfnBackend trait
    let x1 = larql_inference::ndarray::Array2::<f32>::zeros((1, hidden));
    let start = Instant::now();
    for _ in 0..args.tokens {
        for layer in 0..num_layers {
            let _ = cached_ffn.forward(layer, &x1);
        }
    }
    let clone_ms = start.elapsed().as_secs_f64() * 1000.0;
    let clone_tok_s = args.tokens as f64 / (clone_ms / 1000.0);

    // Method 2: Direct memcpy into pre-allocated buffer
    let cache_vecs = cached_ffn.get_cache_vecs();
    let mut out_buf = vec![0.0f32; hidden];
    let start = Instant::now();
    for _ in 0..args.tokens {
        for layer in 0..num_layers {
            if let Some(cached) = cache_vecs.get(&layer) {
                // Copy just the last position's row into the pre-allocated buffer
                let seq_len = cached.shape()[0];
                let last_row = cached.row(seq_len - 1);
                out_buf.copy_from_slice(last_row.as_slice().unwrap());
            }
        }
    }
    let memcpy_ms = start.elapsed().as_secs_f64() * 1000.0;
    let memcpy_tok_s = args.tokens as f64 / (memcpy_ms / 1000.0);

    // Method 3: ArcArray clone (refcount bump only, no data copy)
    let start = Instant::now();
    for _ in 0..args.tokens {
        for layer in 0..num_layers {
            if let Some(cached) = cache_vecs.get(&layer) {
                let _ref = cached.clone(); // refcount bump, O(1)
            }
        }
    }
    let arc_ms = start.elapsed().as_secs_f64() * 1000.0;
    let arc_tok_s = args.tokens as f64 / (arc_ms / 1000.0);

    // Method 4: Raw pointer read (no copy, just dereference)
    let start = Instant::now();
    let mut checksum = 0.0f32;
    for _ in 0..args.tokens {
        for layer in 0..num_layers {
            if let Some(cached) = cache_vecs.get(&layer) {
                let seq_len = cached.shape()[0];
                checksum += cached[[seq_len - 1, 0]] + cached[[seq_len - 1, hidden - 1]];
            }
        }
    }
    let read_ms = start.elapsed().as_secs_f64() * 1000.0;
    let read_tok_s = args.tokens as f64 / (read_ms / 1000.0);

    println!();
    println!("FFN Throughput — {} tokens, {} layers, hidden={}", args.tokens, num_layers, hidden);
    println!("{}", "=".repeat(65));
    println!("{:>25} {:>10} {:>12} {:>12}", "Method", "Total ms", "us/tok", "tok/s");
    println!("{}", "-".repeat(65));
    println!("{:>25} {:>10.1} {:>12.1} {:>12.0}",
        "clone (current)", clone_ms, clone_ms * 1000.0 / args.tokens as f64, clone_tok_s);
    println!("{:>25} {:>10.1} {:>12.1} {:>12.0}",
        "memcpy (pre-alloc)", memcpy_ms, memcpy_ms * 1000.0 / args.tokens as f64, memcpy_tok_s);
    println!("{:>25} {:>10.1} {:>12.1} {:>12.0}",
        "arc clone (refcount)", arc_ms, arc_ms * 1000.0 / args.tokens as f64, arc_tok_s);
    println!("{:>25} {:>10.1} {:>12.1} {:>12.0}",
        "read-only (no copy)", read_ms, read_ms * 1000.0 / args.tokens as f64, read_tok_s);

    println!();
    let bytes_per_tok = num_layers as f64 * hidden as f64 * 4.0;
    println!("  Bytes/token: {:.0} ({} layers × {} × 4B)", bytes_per_tok, num_layers, hidden);
    println!("  Bandwidth at 100K tok/s: {:.1} GB/s", bytes_per_tok * 100_000.0 / 1e9);
    println!("  (checksum: {:.2} — prevents optimizer elimination)", checksum);

    Ok(())
}
