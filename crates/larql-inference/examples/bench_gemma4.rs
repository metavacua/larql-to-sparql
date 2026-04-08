//! Gemma 4 E2B end-to-end benchmark with component breakdown.
//!
//! Profiles: embedding, RoPE (sliding vs global), GQA attention, FFN,
//! PLE, layer_scalar, KV sharing overhead, and full forward pass.
//! Compares against Ollama if available.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_gemma4 [-- model_name]
//!   Default: google/gemma-4-E2B-it

use std::time::Instant;
use ndarray::Array2;

use larql_inference::attention::{apply_rope, apply_rope_partial, gqa_attention_with_weights};
use larql_inference::forward::{embed_tokens_pub, apply_norm, dot_proj, predict, forward_to_layer};
use larql_inference::residual::{rms_norm_heads, rms_norm_heads_no_weight};
use larql_models::{load_model_dir, resolve_model_path};

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) -> f64 {
    for _ in 0..2.min(iters) { f(); }
    let t0 = Instant::now();
    for _ in 0..iters { f(); }
    let per_iter = t0.elapsed().as_micros() as f64 / iters as f64;
    if per_iter > 10_000.0 {
        println!("  {name:<50} {:>8.2} ms  ({iters} iters)", per_iter / 1000.0);
    } else {
        println!("  {name:<50} {:>8.1} us  ({iters} iters)", per_iter);
    }
    per_iter
}

fn main() {
    let model_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "google/gemma-4-E2B-it".to_string());

    println!("=== Gemma 4 Component Benchmark ===\n");

    // ── Load model ──
    let t0 = Instant::now();
    let path = resolve_model_path(&model_name).expect("model not found");
    let weights = load_model_dir(&path).expect("failed to load");
    println!("Loaded {} in {:.1}s", model_name, t0.elapsed().as_secs_f64());
    println!("  {} layers, hidden={}, vocab={}\n",
        weights.num_layers, weights.hidden_size, weights.vocab_size);

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let token_ids: Vec<u32> = vec![818, 5279, 529, 7001, 563]; // "The capital of France is"
    let seq_len = token_ids.len();

    // ── 1. Embedding ──
    println!("--- Component Benchmarks (seq={seq_len}) ---\n");

    bench("Embedding + scale", 1000, || {
        let _ = embed_tokens_pub(&weights, &token_ids);
    });

    let h_embed = embed_tokens_pub(&weights, &token_ids);

    // ── 2. Input norm ──
    let norm_key = arch.input_layernorm_key(0);
    bench("RMSNorm (input, L0)", 1000, || {
        let _ = apply_norm(&weights, &h_embed, &norm_key, arch.norm_weight_offset());
    });

    let h_norm = apply_norm(&weights, &h_embed, &norm_key, arch.norm_weight_offset());

    // ── 3. Q/K/V projections ──
    let w_q = weights.tensors.get(&arch.attn_q_key(0)).unwrap();
    let w_k = weights.tensors.get(&arch.attn_k_key(0)).unwrap();
    let w_v = weights.tensors.get(&arch.attn_v_key(0)).unwrap();

    bench("Q projection (sliding L0)", 500, || { let _ = dot_proj(&h_norm, w_q); });
    bench("K projection (sliding L0)", 500, || { let _ = dot_proj(&h_norm, w_k); });

    let q = dot_proj(&h_norm, w_q);
    let k = dot_proj(&h_norm, w_k);
    let v = dot_proj(&h_norm, w_v);

    // ── 4. QK norm ──
    let hd_sliding = arch.head_dim_for_layer(0);
    let nq = arch.num_q_heads_for_layer(0);
    let nkv = arch.num_kv_heads_for_layer(0);
    let qk_w = weights.vectors.get(&arch.attn_q_norm_key(0).unwrap()).unwrap();
    let kk_w = weights.vectors.get(&arch.attn_k_norm_key(0).unwrap()).unwrap();

    bench("QK-norm Q (sliding, per-head)", 1000, || {
        let _ = rms_norm_heads(&q, qk_w, nq, hd_sliding, 0.0);
    });

    // ── 5. V-norm ──
    bench("V-norm (parameter-free, per-head)", 1000, || {
        let _ = rms_norm_heads_no_weight(&v, nkv, hd_sliding);
    });

    // ── 6. RoPE ──
    let q_normed = rms_norm_heads(&q, qk_w, nq, hd_sliding, 0.0);
    let k_normed = rms_norm_heads(&k, kk_w, nkv, hd_sliding, 0.0);

    bench("RoPE Q (sliding, 8×256, full)", 1000, || {
        let _ = apply_rope(&q_normed, nq, hd_sliding, 10_000.0);
    });
    bench("RoPE K (sliding, 1×256, full)", 1000, || {
        let _ = apply_rope(&k_normed, nkv, hd_sliding, 10_000.0);
    });

    // Global layer RoPE (if applicable)
    if arch.head_dim_for_layer(4) != hd_sliding {
        let hd_global = arch.head_dim_for_layer(4);
        let w_q_g = weights.tensors.get(&arch.attn_q_key(4)).unwrap();
        let h_norm_g = apply_norm(&weights, &h_embed, &arch.input_layernorm_key(4), arch.norm_weight_offset());
        let q_g = dot_proj(&h_norm_g, w_q_g);
        let frac = arch.rotary_fraction_for_layer(4);

        bench(&format!("RoPE Q (global, {nq}×{hd_global}, {:.0}%)", frac * 100.0), 1000, || {
            let _ = apply_rope_partial(&q_g, nq, hd_global, 1_000_000.0, frac);
        });
    }

    // ── 7. GQA attention ──
    let q_rope = apply_rope(&q_normed, nq, hd_sliding, 10_000.0);
    let k_rope = apply_rope(&k_normed, nkv, hd_sliding, 10_000.0);
    let v_normed = rms_norm_heads_no_weight(&v, nkv, hd_sliding);
    let reps = nq / nkv;

    bench("GQA attention (sliding, scale=1.0)", 500, || {
        let _ = gqa_attention_with_weights(
            &q_rope, &k_rope, &v_normed, nq, hd_sliding, reps, 1.0, seq_len, false, None);
    });

    // ── 8. FFN ──
    let w_gate = weights.tensors.get(&arch.ffn_gate_key(0)).unwrap();
    let w_up = weights.tensors.get(&arch.ffn_up_key(0)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(0)).unwrap();
    let inter = w_gate.shape()[0];

    bench(&format!("FFN gate proj ({inter}×{hidden})"), 200, || { let _ = dot_proj(&h_norm, w_gate); });
    bench("FFN full (gate+up+act+down)", 100, || {
        let gate = dot_proj(&h_norm, w_gate);
        let up = dot_proj(&h_norm, w_up);
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let activated: Array2<f32> = gate.mapv(|x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }) * &up;
        let _ = dot_proj(&activated, w_down);
    });

    // ── 9. Full forward pass ──
    println!("\n--- Full Forward Pass ---\n");

    let tokenizer_path = path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).expect("tokenizer");

    let full_us = bench("predict (5 tokens, all layers)", 3, || {
        let _ = predict(&weights, &tokenizer, &token_ids, 5);
    });

    let per_layer = full_us / weights.num_layers as f64;
    println!("\n  Per-layer avg: {per_layer:.0} us ({} layers)", weights.num_layers);
    println!("  Throughput: {:.1} queries/sec\n", 1_000_000.0 / full_us);

    // ── 10. Layer-by-layer timing (first 5 + last 5) ──
    println!("--- Per-Layer Timing (forward_to_layer delta) ---\n");

    let mut prev_time = 0.0f64;
    let layers_to_check: Vec<usize> = (0..5).chain(weights.num_layers-3..weights.num_layers).collect();

    for &stop in &layers_to_check {
        let t0 = Instant::now();
        let _ = forward_to_layer(&weights, &token_ids, stop);
        let elapsed = t0.elapsed().as_micros() as f64;
        let delta = elapsed - prev_time;
        let layer_type = if arch.is_sliding_window_layer(stop) { "sliding" } else { "GLOBAL " };
        let kv_src = arch.kv_shared_source_layer(stop);
        let sharing = kv_src.map_or("own KV".to_string(), |s| format!("KV←L{s}"));
        println!("  L{stop:2} ({layer_type}, {sharing}): {delta:>8.0} us (cumulative: {elapsed:.0} us)");
        prev_time = elapsed;
    }

    // ── 11. Ollama comparison (if available) ──
    println!("\n--- Ollama Comparison ---\n");

    let ollama_result = std::process::Command::new("ollama")
        .args(["list"])
        .output();

    match ollama_result {
        Ok(output) if output.status.success() => {
            let list = String::from_utf8_lossy(&output.stdout);
            let has_gemma4 = list.lines().any(|l| l.contains("gemma-4") || l.contains("gemma4"));
            if has_gemma4 {
                println!("  Ollama has a Gemma 4 model. Benchmarking...");
                // Run Ollama with timing
                let t0 = Instant::now();
                let result = std::process::Command::new("ollama")
                    .args(["run", "gemma4:e2b", "--verbose", "The capital of France is"])
                    .output();
                let ollama_ms = t0.elapsed().as_millis();

                match result {
                    Ok(out) => {
                        let resp = String::from_utf8_lossy(&out.stdout);
                        println!("  Ollama response: {}", resp.trim().lines().next().unwrap_or("(empty)"));
                        println!("  Ollama time: {ollama_ms} ms");
                        println!("  LARQL time:  {:.0} ms", full_us / 1000.0);
                        let ratio = ollama_ms as f64 / (full_us / 1000.0);
                        println!("  Ratio: Ollama is {ratio:.1}x vs LARQL (CPU f32)");
                    }
                    Err(e) => println!("  Ollama run failed: {e}"),
                }
            } else {
                println!("  Ollama available but no Gemma 4 model found.");
                println!("  Install with: ollama pull gemma4:e2b");
            }
        }
        _ => {
            println!("  Ollama not installed. Install from https://ollama.com");
            println!("  Then: ollama pull gemma4:e2b");
        }
    }

    // ── Summary ──
    println!("\n--- Summary ---\n");
    let result = predict(&weights, &tokenizer, &token_ids, 3);
    println!("  Model: {model_name}");
    println!("  Predict: {:.0} ms ({:.1} qps)", full_us / 1000.0, 1_000_000.0 / full_us);
    println!("  Top prediction: {} ({:.1}%)",
        result.predictions[0].0, result.predictions[0].1 * 100.0);
    println!();
}
