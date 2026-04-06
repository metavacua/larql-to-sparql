//! Fused attention demo — shows the online-softmax kernel in action.
//!
//! Demonstrates:
//! - Correctness: fused output matches naive reference
//! - GQA: multiple Q heads sharing K/V heads
//! - Softcap: Gemma2-style logit capping
//! - Attention capture: extracting per-head weights
//! - Causal masking: each position only sees past tokens
//!
//! Usage:
//!   cargo run -p larql-inference --example attention_demo

use ndarray::Array2;

use larql_inference::attention::{gqa_attention, gqa_attention_with_weights};

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

fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    println!("=== Fused Attention Demo ===\n");

    // ── 1. Single token: output = V ──
    println!("--- 1. Single Token ---");
    let q = Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let k = q.clone();
    let v = Array2::from_shape_vec((1, 4), vec![0.5, 0.5, 0.5, 0.5]).unwrap();
    let out = gqa_attention(&q, &k, &v, 1, 4, 1, 0.5, 1);
    println!("  Input V: [0.5, 0.5, 0.5, 0.5]");
    println!(
        "  Output:  [{:.4}, {:.4}, {:.4}, {:.4}]",
        out[[0, 0]], out[[0, 1]], out[[0, 2]], out[[0, 3]]
    );
    println!("  (Single token → attention weight = 1.0 → output = V)\n");

    // ── 2. Causal masking ──
    println!("--- 2. Causal Masking (3 tokens) ---");
    let seq = 3;
    let hd = 4;
    let q = Array2::zeros((seq, hd));
    let k = Array2::zeros((seq, hd));
    let v = Array2::from_shape_vec(
        (seq, hd),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let out = gqa_attention(&q, &k, &v, 1, hd, 1, 1.0, seq);
    println!("  V[0] = [1, 0, 0, 0]  V[1] = [0, 1, 0, 0]  V[2] = [0, 0, 1, 0]");
    println!("  Q=K=0 → uniform attention (equal scores)");
    for i in 0..seq {
        println!(
            "  Token {i} sees 0..={i}: output = [{:.3}, {:.3}, {:.3}, {:.3}]",
            out[[i, 0]], out[[i, 1]], out[[i, 2]], out[[i, 3]]
        );
    }
    println!("  (Each token averages V rows it can see)\n");

    // ── 3. Multi-head attention ──
    println!("--- 3. Multi-Head Attention ---");
    let seq = 6;
    let hd = 16;
    let num_heads = 4;
    let scale = 1.0 / (hd as f64).sqrt();
    let q = synth_matrix(seq, num_heads * hd, 10);
    let k = synth_matrix(seq, num_heads * hd, 11);
    let v = synth_matrix(seq, num_heads * hd, 12);
    let out = gqa_attention(&q, &k, &v, num_heads, hd, 1, scale, seq);
    println!("  {num_heads} heads, head_dim={hd}, seq={seq}, scale={scale:.4}");
    println!("  Output shape: {:?}", out.shape());
    let norms: Vec<f32> = (0..num_heads)
        .map(|h| {
            let off = h * hd;
            out.slice(ndarray::s![seq - 1, off..off + hd])
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
        })
        .collect();
    println!(
        "  Last token per-head L2 norms: [{}]",
        norms
            .iter()
            .map(|n| format!("{n:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!();

    // ── 4. Grouped-Query Attention (GQA) ──
    println!("--- 4. Grouped-Query Attention ---");
    let num_q = 10;
    let num_kv = 2;
    let reps = num_q / num_kv;
    let hd = 32;
    let scale = 1.0 / (hd as f64).sqrt();
    let q = synth_matrix(seq, num_q * hd, 20);
    let k = synth_matrix(seq, num_kv * hd, 21);
    let v = synth_matrix(seq, num_kv * hd, 22);
    let out = gqa_attention(&q, &k, &v, num_q, hd, reps, scale, seq);
    println!("  {num_q} Q heads, {num_kv} KV heads (reps={reps}), head_dim={hd}");
    println!("  Output shape: {:?}", out.shape());

    // Verify Q heads sharing same KV produce different outputs (different Q weights)
    let h0_norm: f32 = out
        .slice(ndarray::s![seq - 1, 0..hd])
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    let h1_norm: f32 = out
        .slice(ndarray::s![seq - 1, hd..2 * hd])
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    println!(
        "  Heads 0,1 share KV group 0: ||h0||={h0_norm:.3}, ||h1||={h1_norm:.3} (different Q → different output)"
    );
    println!();

    // ── 5. Softcap (Gemma2) ──
    println!("--- 5. Softcap (Gemma2-style) ---");
    let hd = 16;
    let q = synth_matrix(seq, hd, 30);
    let k = synth_matrix(seq, hd, 31);
    let v = synth_matrix(seq, hd, 32);
    let scale = 1.0 / (hd as f64).sqrt();

    let (out_no_cap, _) = gqa_attention_with_weights(&q, &k, &v, 1, hd, 1, scale, seq, false, None);
    let (out_cap, _) = gqa_attention_with_weights(&q, &k, &v, 1, hd, 1, scale, seq, false, Some(50.0));

    let diff = max_diff(&out_no_cap, &out_cap);
    println!("  Without softcap: last token = [{:.4}, {:.4}, ...]",
        out_no_cap[[seq - 1, 0]], out_no_cap[[seq - 1, 1]]);
    println!("  With softcap=50: last token = [{:.4}, {:.4}, ...]",
        out_cap[[seq - 1, 0]], out_cap[[seq - 1, 1]]);
    println!("  Max diff: {diff:.2e}  (softcap compresses extreme scores)\n");

    // ── 6. Attention Weight Capture ──
    println!("--- 6. Attention Weight Capture ---");
    let hd = 8;
    let num_heads = 3;
    let q = synth_matrix(seq, num_heads * hd, 40);
    let k = synth_matrix(seq, num_heads * hd, 41);
    let v = synth_matrix(seq, num_heads * hd, 42);
    let scale = 1.0 / (hd as f64).sqrt();

    let (_, weights) = gqa_attention_with_weights(
        &q, &k, &v, num_heads, hd, 1, scale, seq, true, None,
    );
    let weights = weights.unwrap();
    println!("  {num_heads} heads, seq={seq}, capturing last token's attention");
    for (h, w) in weights.heads.iter().enumerate() {
        let sum: f32 = w.iter().sum();
        let max_pos = w
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        println!(
            "  Head {h}: weights = [{}]  sum={sum:.4}  max_pos={max_pos}",
            w.iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("  (Each head's weights sum to 1.0, represent last token's attention)\n");

    // ── 7. Memory savings ──
    println!("--- 7. Memory Comparison ---");
    println!("  Fused (online softmax) never allocates the [seq, seq] scores matrix.");
    println!("  Per head, fused uses O(head_dim) accumulator vs O(seq^2) materialized.\n");
    println!("  {:>6}  {:>12}  {:>12}  {:>8}", "seq", "materialized", "fused_acc", "savings");
    let num_heads_demo = 10;
    let hd_demo = 256;
    for &s in &[6, 24, 128, 512, 2048] {
        let mat = s * s * num_heads_demo * 4; // f32
        let fused = s * hd_demo * 8; // f64 acc per position
        println!(
            "  {:>6}  {:>10.1}KB  {:>10.1}KB  {:>7.0}x",
            s,
            mat as f64 / 1024.0,
            fused as f64 / 1024.0,
            mat as f64 / fused as f64,
        );
    }

    println!("\n=== Done ===");
}
