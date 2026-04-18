//! ffn_profile — per-phase FFN timing on a loaded vindex.
//!
//! Times each stage of a K=full walk at one layer:
//!   gate_scores_batch, q4k_matmul_transb(up), q4k_matmul_transb(down).
//! Prints medians across iterations. Run:
//!   cargo run --release -p larql-inference --example ffn_profile -- \
//!     --model MODEL --vindex DIR [--layer 30] [--seq-len 6] [--iters 20]
//!
//! The total should roughly match the FFN slice of walk_ffn_sparse's fast
//! path (gate + up + silu/gelu elementwise + down). If it's << the forward
//! total, the bottleneck is attention or orchestration, not the FFN.

use std::path::PathBuf;
use std::time::Instant;

use larql_inference::{default_backend, InferenceModel};
use larql_vindex::{SilentLoadCallbacks, VectorIndex, GateIndex};

struct Args {
    model: String,
    vindex: PathBuf,
    layer: usize,
    seq_len: usize,
    iters: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut vindex = PathBuf::new();
    let mut layer = 0;
    let mut seq_len = 6;
    let mut iters = 10;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model = args[i].clone(); }
            "--vindex" => { i += 1; vindex = PathBuf::from(&args[i]); }
            "--layer" => { i += 1; layer = args[i].parse().unwrap_or(0); }
            "--seq-len" => { i += 1; seq_len = args[i].parse().unwrap_or(6); }
            "--iters" => { i += 1; iters = args[i].parse().unwrap_or(10); }
            _ => {}
        }
        i += 1;
    }
    if model.is_empty() || !vindex.is_dir() {
        eprintln!("Usage: ffn_profile --model M --vindex D [--layer N] [--seq-len N] [--iters N]");
        std::process::exit(1);
    }
    Args { model, vindex, layer, seq_len, iters }
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() as f64) * p).floor() as usize;
    samples[idx.min(samples.len() - 1)]
}

fn median(samples: &mut [f64]) -> f64 { percentile(samples, 0.5) }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    println!("=== FFN Profile ===\n");
    println!("Model:   {}", args.model);
    println!("Vindex:  {}", args.vindex.display());
    println!("Layer:   {}", args.layer);
    println!("seq_len: {}", args.seq_len);
    println!("iters:   {}\n", args.iters);

    let t0 = Instant::now();
    let model = InferenceModel::load_walk_only(&args.model)?;
    let weights = model.weights();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    println!("Loaded: {num_layers} layers, hidden={hidden} (took {:.1}s)", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&args.vindex, &mut cb)?;
    println!("Vindex: {} vectors (took {:.1}s)\n", index.total_gate_vectors(), t0.elapsed().as_secs_f64());

    let intermediate = index.num_features(args.layer);
    println!("Layer {} shape: intermediate={}, hidden={}", args.layer, intermediate, hidden);

    let backend = default_backend();
    let backend_ref: Option<&dyn larql_compute::ComputeBackend> = Some(&*backend);

    // Synthetic x: [seq_len, hidden] random-ish, just for timing.
    let x_vec: Vec<f32> = (0..args.seq_len * hidden).map(|i| ((i as f32 * 0.001).sin() * 0.1)).collect();
    let x = ndarray::Array2::from_shape_vec((args.seq_len, hidden), x_vec.clone())?;
    let x_flat: &[f32] = x.as_slice().unwrap();

    // Warmup — make sure mmap pages and Q4K metadata are hot.
    for _ in 0..2 {
        let _ = index.gate_scores_batch(args.layer, &x);
        let _ = index.q4k_matmul_transb(args.layer, 1, x_flat, args.seq_len, backend_ref);
    }

    // --- Gate scores ---
    let mut gate_ms = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t = Instant::now();
        let _ = index.gate_scores_batch(args.layer, &x);
        gate_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Up Q4K matmul ---
    let mut up_ms = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t = Instant::now();
        let _ = index.q4k_matmul_transb(args.layer, 1, x_flat, args.seq_len, backend_ref);
        up_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    // --- Down Q6K matmul (needs activation shaped [seq, intermediate]) ---
    let act_vec: Vec<f32> = (0..args.seq_len * intermediate).map(|i| ((i as f32 * 0.002).cos() * 0.1)).collect();
    let mut down_ms = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t = Instant::now();
        let _ = index.q4k_matmul_transb(args.layer, 2, &act_vec, args.seq_len, backend_ref);
        down_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let g_med = median(&mut gate_ms.clone());
    let u_med = median(&mut up_ms.clone());
    let d_med = median(&mut down_ms.clone());
    let g_p99 = percentile(&mut gate_ms, 0.99);
    let u_p99 = percentile(&mut up_ms, 0.99);
    let d_p99 = percentile(&mut down_ms, 0.99);

    println!("\n--- Per-phase medians @ layer {} (seq_len={}) ---", args.layer, args.seq_len);
    println!("  {:<28}  median   p99", "phase");
    println!("  {}", "-".repeat(58));
    println!("  {:<28}  {:>6.1}ms  {:>6.1}ms", "gate_scores_batch (f16)", g_med, g_p99);
    println!("  {:<28}  {:>6.1}ms  {:>6.1}ms", "q4k_matmul_transb (up)", u_med, u_p99);
    println!("  {:<28}  {:>6.1}ms  {:>6.1}ms", "q4k_matmul_transb (down)", d_med, d_p99);
    println!("  {}", "-".repeat(58));
    let layer_total = g_med + u_med + d_med;
    println!("  {:<28}  {:>6.1}ms", "per-layer FFN total", layer_total);
    println!("  {:<28}  {:>6.1}ms", format!("× {num_layers} layers est. FFN"), layer_total * num_layers as f64);

    Ok(())
}
