//! Walk inference benchmark — measures tokens/sec with dense attention + vindex FFN
//! at all 34 layers. This is the target architecture: zero FFN matmul.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_walk_inference -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex path/to/gemma3-4b.vindex

use std::time::Instant;

use larql_inference::{
    predict, predict_with_ffn,
    InferenceModel, WeightFfn,
    vindex::WalkFfn,
};
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_name = String::new();
    let mut vindex_path = std::path::PathBuf::new();
    let mut top_k = 8092;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_name = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
            "--top-k" => { i += 1; top_k = args[i].parse().unwrap(); }
            _ => {}
        }
        i += 1;
    }
    if model_name.is_empty() || !vindex_path.is_dir() {
        eprintln!("Usage: bench_walk_inference --model MODEL --vindex PATH [--top-k N]");
        std::process::exit(1);
    }

    println!("=== Walk Inference Benchmark ===\n");

    // Load
    let t0 = Instant::now();
    let model = InferenceModel::load(&model_name)?;
    println!("Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    println!("Vindex loaded in {:.1}s ({} vectors)", t0.elapsed().as_secs_f64(), index.total_gate_vectors());

    // Pre-decode f16 gate vectors (skip for f32 — already zero-copy mmap)
    let t0 = Instant::now();
    index.warmup();
    let warmup_s = t0.elapsed().as_secs_f64();
    if warmup_s > 0.01 {
        println!("Gate warmup in {warmup_s:.1}s (f16→f32 pre-decode)");
    } else {
        println!("Gate warmup: skipped (f32, zero-copy mmap)");
    }

    // Load feature-major vectors for mmap walk
    match index.load_down_features(&vindex_path) {
        Ok(()) => println!("Down features: loaded"),
        Err(_) => println!("Down features: not found"),
    }
    match index.load_up_features(&vindex_path) {
        Ok(()) => println!("Up features: loaded (full mmap FFN enabled)"),
        Err(_) => println!("Up features: not found"),
    }

    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    println!("{num_layers} layers, hidden={}, top_k={top_k}\n", weights.hidden_size);

    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("Prompt: \"{prompt}\" ({} tokens)\n", token_ids.len());

    // ── Dense baseline ──
    println!("--- Dense (all matmul) ---");
    // Warmup
    let _ = predict(weights, tokenizer, &token_ids, 5);

    let n = 3;
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = predict(weights, tokenizer, &token_ids, 5);
    }
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let dense_result = predict(weights, tokenizer, &token_ids, 5);
    let (dense_tok, dense_prob) = dense_result.predictions.first()
        .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();
    println!("  {dense_tok} ({:.2}%)  {dense_ms:.0}ms/token  ({:.1} tok/s)",
        dense_prob * 100.0, 1000.0 / dense_ms);

    // ── Walk brute-force (vindex FFN, all layers) ──
    println!("\n--- Walk brute-force (dense attention + vindex FFN, all {num_layers} layers) ---");
    let walk_ffn = WalkFfn::new(weights, &index, top_k);

    // Warmup (also primes the f16 decode cache)
    let _ = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);

    let t0 = Instant::now();
    for _ in 0..n {
        let _ = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);
    }
    let walk_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let walk_result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);
    let (walk_tok, walk_prob) = walk_result.predictions.first()
        .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();
    println!("  {walk_tok} ({:.2}%)  {walk_ms:.0}ms/token  ({:.1} tok/s)",
        walk_prob * 100.0, 1000.0 / walk_ms);

    // ── Component breakdown ──
    println!("\n--- Component breakdown (layer 0, 3 iters) ---");

    let weight_ffn = WeightFfn { weights };
    let h = larql_inference::forward_to_layer(weights, &token_ids, 0);
    let h_norm = larql_inference::ndarray::Array2::from_shape_fn(
        (h.shape()[0], h.shape()[1]), |(i, j)| h[[i, j]]
    );

    // Dense FFN
    let t0 = Instant::now();
    for _ in 0..3 {
        use larql_inference::FfnBackend;
        let _ = weight_ffn.forward(0, &h_norm);
    }
    let dense_ffn_ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
    println!("  Dense FFN layer 0:      {dense_ffn_ms:.1}ms");

    // Walk FFN (end-to-end: gate KNN + sparse)
    let t0 = Instant::now();
    for _ in 0..3 {
        use larql_inference::FfnBackend;
        let _ = walk_ffn.forward(0, &h_norm);
    }
    let walk_ffn_ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
    println!("  Walk FFN layer 0:       {walk_ffn_ms:.1}ms");

    // Gate KNN alone (batch gemm)
    let t0 = Instant::now();
    for _ in 0..3 {
        let _ = index.gate_knn_batch(0, &h_norm, top_k);
    }
    let gate_ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
    println!("  Gate KNN (batch gemm):  {gate_ms:.1}ms");

    // Sparse FFN alone (given pre-selected features)
    let features = index.gate_knn_batch(0, &h_norm, top_k);
    let t0 = Instant::now();
    for _ in 0..3 {
        let _ = larql_inference::ffn::sparse_compute::sparse_ffn_forward(weights, 0, &h_norm, &features);
    }
    let sparse_ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
    println!("  Sparse FFN (K={}):  {sparse_ms:.1}ms", features.len());
    println!();
    println!("  Gate KNN:    {:.0}% of walk FFN time", gate_ms / walk_ffn_ms.max(0.01) * 100.0);
    println!("  Sparse FFN:  {:.0}% of walk FFN time", sparse_ms / walk_ffn_ms.max(0.01) * 100.0);
    println!("  Dense/Walk:  {:.1}x", dense_ffn_ms / walk_ffn_ms.max(0.01));

    // ── Walk HNSW ──
    println!("\n--- Walk HNSW (graph search, all {num_layers} layers) ---");
    index.enable_hnsw(200);
    let walk_hnsw = WalkFfn::new(weights, &index, top_k);

    println!("  Building HNSW indexes (one-time, dim=64 projected)...");
    let t0 = Instant::now();
    let _ = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_hnsw);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  HNSW build + first query: {build_ms:.0}ms");

    let t0 = Instant::now();
    for _ in 0..n {
        let _ = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_hnsw);
    }
    let hnsw_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let hnsw_result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_hnsw);
    let (hnsw_tok, hnsw_prob) = hnsw_result.predictions.first()
        .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();
    println!("  {hnsw_tok} ({:.2}%)  {hnsw_ms:.0}ms/token  ({:.1} tok/s)",
        hnsw_prob * 100.0, 1000.0 / hnsw_ms);
    index.disable_hnsw();

    // ── Summary ──
    println!("\n--- Summary ---\n");
    println!("  Dense:       {dense_ms:>8.0}ms  ({:.1} tok/s)  {dense_tok} ({:.2}%)",
        1000.0 / dense_ms, dense_prob * 100.0);
    println!("  Walk brute:  {walk_ms:>8.0}ms  ({:.1} tok/s)  {walk_tok} ({:.2}%)",
        1000.0 / walk_ms, walk_prob * 100.0);
    println!("  Walk HNSW:   {hnsw_ms:>8.0}ms  ({:.1} tok/s)  {hnsw_tok} ({:.2}%)",
        1000.0 / hnsw_ms, hnsw_prob * 100.0);
    println!();
    println!("  Brute vs HNSW: {:.1}x", walk_ms / hnsw_ms.max(0.1));
    println!("  Dense vs HNSW: {:.1}x", dense_ms / hnsw_ms.max(0.1));
    println!("  Predictions: dense={dense_tok} brute={walk_tok} hnsw={hnsw_tok}");

    println!("\n=== Done ===");
    Ok(())
}
