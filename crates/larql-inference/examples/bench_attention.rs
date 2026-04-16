//! Fused attention benchmark — measures the online-softmax kernel
//! against a naive materialized-scores reference at transformer-realistic
//! dimensions. Synthetic data only — no model weights needed.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_attention

use ndarray::Array2;
use std::time::Instant;

use larql_inference::attention::{gqa_attention, gqa_attention_with_weights};

/// Deterministic f32 matrix.
fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

/// Naive reference: materialized [seq, seq] scores matrix per head.
#[allow(clippy::too_many_arguments)]
fn reference_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f32,
    seq_len: usize,
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));

    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        // Materialize full [seq, seq] scores matrix
        let q_head = q.slice(ndarray::s![.., q_off..q_off + head_dim]);
        let k_head = k.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let mut scores = q_head.dot(&k_head.t()).mapv(|v| v * scale);

        // Causal mask + softmax
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = -1e9;
            }
            let max_val = scores.row(i).iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for j in 0..seq_len {
                let e = ((scores[[i, j]] - max_val) as f64).exp();
                scores[[i, j]] = e as f32;
                sum += e;
            }
            let inv = (1.0 / sum) as f32;
            for j in 0..seq_len {
                scores[[i, j]] *= inv;
            }
        }

        // scores @ V
        let v_head = v.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let attn_v = scores.dot(&v_head);
        for i in 0..seq_len {
            for d in 0..head_dim {
                out[[i, q_off + d]] = attn_v[[i, d]];
            }
        }
    }
    out
}

fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) -> f64 {
    for _ in 0..3.min(iters) {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per_iter = t0.elapsed().as_micros() as f64 / iters as f64;
    if per_iter > 10_000.0 {
        println!("  {name:<55} {:.2} ms  ({iters} iters)", per_iter / 1000.0);
    } else {
        println!("  {name:<55} {:.1} us  ({iters} iters)", per_iter);
    }
    per_iter
}

fn main() {
    println!("=== Fused Attention Benchmark ===");
    println!("Online softmax (fused) vs materialized scores (reference)\n");

    // ── 1. Correctness verification at each test size ──
    println!("--- Correctness Verification ---\n");

    let configs: Vec<(&str, usize, usize, usize, usize)> = vec![
        ("Gemma-3 4B (scaled)", 6, 32, 10, 2),
        ("Single head", 6, 64, 1, 1),
        ("Long seq", 48, 32, 4, 4),
        ("Full Gemma-3 head_dim", 6, 256, 10, 2),
        ("Very long seq", 128, 32, 2, 2),
    ];

    for (label, seq, hd, nq, nkv) in &configs {
        let reps = nq / nkv;
        let scale = 1.0 / (*hd as f64).sqrt();
        let q = synth_matrix(*seq, nq * hd, 42);
        let k = synth_matrix(*seq, nkv * hd, 43);
        let v = synth_matrix(*seq, nkv * hd, 44);

        let fused = gqa_attention(&q, &k, &v, *nq, *hd, reps, scale, *seq);
        let naive = reference_attention(&q, &k, &v, *nq, *hd, reps, scale as f32, *seq);
        let diff = max_diff(&fused, &naive);

        let scores_bytes = seq * seq * nq * 4; // [seq, seq] per head, f32
        let fused_bytes = seq * hd * 8; // acc per head, f64
        let status = if diff < 1e-3 { "PASS" } else { "FAIL" };

        println!(
            "  {status} {label:<30} seq={seq:<4} heads={nq:<3} hd={hd:<4} max_diff={diff:.2e}  scores_mem={:.1}KB fused_mem={:.1}KB",
            scores_bytes as f64 / 1024.0,
            fused_bytes as f64 / 1024.0,
        );
    }

    // ── 2. Performance: fused vs reference at varying seq lengths ──
    println!("\n--- Sequence Length Scaling (Gemma-3 4B config: 10 Q, 2 KV, hd=32) ---\n");

    let hd = 32;
    let nq = 10;
    let nkv = 2;
    let reps = nq / nkv;
    let scale = 1.0 / (hd as f64).sqrt();

    for &seq in &[1, 6, 12, 24, 48, 96, 192] {
        let q = synth_matrix(seq, nq * hd, 100 + seq as u64);
        let k = synth_matrix(seq, nkv * hd, 200 + seq as u64);
        let v = synth_matrix(seq, nkv * hd, 300 + seq as u64);

        let iters = if seq <= 24 { 200 } else if seq <= 96 { 50 } else { 10 };

        let fused_us = bench(
            &format!("Fused     seq={seq:<4} ({nq} heads, hd={hd})"),
            iters,
            || { let _ = gqa_attention(&q, &k, &v, nq, hd, reps, scale, seq); },
        );

        let ref_us = bench(
            &format!("Reference seq={seq:<4} ({nq} heads, hd={hd})"),
            iters,
            || { let _ = reference_attention(&q, &k, &v, nq, hd, reps, scale as f32, seq); },
        );

        let ratio = ref_us / fused_us.max(0.1);
        let scores_kb = (seq * seq * nq * 4) as f64 / 1024.0;
        if ratio > 1.0 {
            println!("    -> Fused {ratio:.1}x faster, saves {scores_kb:.1}KB scores matrix\n");
        } else {
            println!("    -> Reference {:.1}x faster, scores matrix = {scores_kb:.1}KB\n", 1.0 / ratio);
        }
    }

    // ── 3. Head dimension scaling ──
    println!("--- Head Dimension Scaling (seq=6, 10 heads) ---\n");

    let seq = 6;
    let nq = 10;
    let nkv = 2;
    let reps = nq / nkv;

    for &hd in &[32, 64, 128, 256] {
        let scale = 1.0 / (hd as f64).sqrt();
        let q = synth_matrix(seq, nq * hd, 500 + hd as u64);
        let k = synth_matrix(seq, nkv * hd, 600 + hd as u64);
        let v = synth_matrix(seq, nkv * hd, 700 + hd as u64);

        let fused_us = bench(
            &format!("Fused     hd={hd:<4} ({nq} heads, seq={seq})"),
            200,
            || { let _ = gqa_attention(&q, &k, &v, nq, hd, reps, scale, seq); },
        );

        let ref_us = bench(
            &format!("Reference hd={hd:<4} ({nq} heads, seq={seq})"),
            200,
            || { let _ = reference_attention(&q, &k, &v, nq, hd, reps, scale as f32, seq); },
        );

        let ratio = ref_us / fused_us.max(0.1);
        if ratio > 1.0 {
            println!("    -> Fused {ratio:.1}x faster\n");
        } else {
            println!("    -> Reference {:.1}x faster\n", 1.0 / ratio);
        }
    }

    // ── 4. Attention capture overhead ──
    println!("--- Capture Overhead (seq=6, Gemma-3 config) ---\n");

    let hd = 32;
    let scale = 1.0 / (hd as f64).sqrt();
    let q = synth_matrix(6, nq * hd, 800);
    let k = synth_matrix(6, nkv * hd, 801);
    let v = synth_matrix(6, nkv * hd, 802);

    let no_cap_us = bench("Fused without capture", 500, || {
        let _ = gqa_attention_with_weights(&q, &k, &v, nq, hd, reps, scale, 6, false, None);
    });

    let cap_us = bench("Fused with capture", 500, || {
        let _ = gqa_attention_with_weights(&q, &k, &v, nq, hd, reps, scale, 6, true, None);
    });

    let overhead = (cap_us / no_cap_us.max(0.1) - 1.0) * 100.0;
    println!("    -> Capture overhead: {overhead:.1}%\n");

    // ── 5. Softcap overhead ──
    println!("--- Softcap Overhead (Gemma2-style) ---\n");

    let no_softcap_us = bench("Fused without softcap", 500, || {
        let _ = gqa_attention_with_weights(&q, &k, &v, nq, hd, reps, scale, 6, false, None);
    });

    let softcap_us = bench("Fused with softcap=50.0", 500, || {
        let _ = gqa_attention_with_weights(&q, &k, &v, nq, hd, reps, scale, 6, false, Some(50.0));
    });

    let overhead = (softcap_us / no_softcap_us.max(0.1) - 1.0) * 100.0;
    println!("    -> Softcap overhead: {overhead:.1}%\n");

    // ── 6. Memory comparison ──
    println!("--- Memory: Materialized vs Fused ---\n");
    println!("  {:>6}  {:>10}  {:>10}  {:>8}", "seq", "scores_mat", "fused_acc", "savings");
    for &seq in &[6, 24, 128, 512, 1024, 2048] {
        let scores_bytes = seq * seq * nq * std::mem::size_of::<f32>();
        let fused_bytes = seq * 256 * std::mem::size_of::<f64>(); // acc per position, head_dim=256
        let savings = if scores_bytes > fused_bytes {
            format!("{:.0}x", scores_bytes as f64 / fused_bytes as f64)
        } else {
            "n/a".to_string()
        };
        println!(
            "  {:>6}  {:>9.1}KB  {:>9.1}KB  {:>8}",
            seq,
            scores_bytes as f64 / 1024.0,
            fused_bytes as f64 / 1024.0,
            savings,
        );
    }

    println!("\n=== Done ===");
}
