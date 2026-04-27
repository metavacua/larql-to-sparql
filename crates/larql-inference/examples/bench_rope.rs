//! RoPE benchmark — measures rotary position embedding at transformer-realistic
//! dimensions. Compares full vs partial rotation, and scaling across seq lengths.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_rope

use ndarray::Array2;
use std::time::Instant;

use larql_inference::attention::{apply_rope, apply_rope_partial};

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

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) -> f64 {
    for _ in 0..3.min(iters) { f(); }
    let t0 = Instant::now();
    for _ in 0..iters { f(); }
    let per_iter = t0.elapsed().as_micros() as f64 / iters as f64;
    if per_iter > 10_000.0 {
        println!("  {name:<55} {:.2} ms  ({iters} iters)", per_iter / 1000.0);
    } else {
        println!("  {name:<55} {:.1} us  ({iters} iters)", per_iter);
    }
    per_iter
}

fn main() {
    println!("=== RoPE Benchmark ===\n");

    // ── 1. Full rotation: head_dim scaling ──
    println!("--- Full Rotation: Head Dimension Scaling (seq=6, 8 heads) ---\n");

    for &hd in &[64, 128, 256, 512] {
        let nq = 8;
        let seq = 6;
        let base = 10000.0;
        let x = synth_matrix(seq, nq * hd, 42 + hd as u64);

        bench(
            &format!("apply_rope         hd={hd:<4} ({nq} heads, seq={seq})"),
            1000,
            || { let _ = apply_rope(&x, nq, hd, base); },
        );
    }

    // ── 2. Partial vs full rotation ──
    println!("\n--- Partial vs Full Rotation (Gemma 4 global layers) ---\n");

    let hd = 512;
    let nq = 8;
    let seq = 6;

    let x = synth_matrix(seq, nq * hd, 100);

    let full_us = bench(
        &format!("Full rotation       hd={hd} (fraction=1.0)"),
        1000,
        || { let _ = apply_rope_partial(&x, nq, hd, 1_000_000.0, 1.0); },
    );

    let partial_us = bench(
        &format!("Partial rotation    hd={hd} (fraction=0.25)"),
        1000,
        || { let _ = apply_rope_partial(&x, nq, hd, 1_000_000.0, 0.25); },
    );

    let speedup = full_us / partial_us.max(0.1);
    println!("    -> Partial 0.25 is {speedup:.1}x faster (rotates 128 of {hd} dims)\n");

    // ── 3. Sequence length scaling ──
    println!("--- Sequence Length Scaling (hd=256, 8 heads) ---\n");

    let hd = 256;
    let nq = 8;
    let base = 10000.0;

    for &seq in &[1, 6, 12, 24, 48, 128, 512] {
        let x = synth_matrix(seq, nq * hd, 200 + seq as u64);
        let iters = if seq <= 48 { 500 } else { 50 };

        bench(
            &format!("apply_rope         seq={seq:<4} ({nq}×{hd}={} dims)", nq * hd),
            iters,
            || { let _ = apply_rope(&x, nq, hd, base); },
        );
    }

    // ── 4. Gemma 4 E2B config: sliding vs global ──
    println!("\n--- Gemma 4 E2B: Sliding vs Global Layer RoPE ---\n");

    let seq = 6;

    // Sliding: 8 heads, hd=256, full rotation, theta=10k
    let x_sliding = synth_matrix(seq, 8 * 256, 300);
    let sliding_us = bench(
        "Sliding  (8×256, full, θ=10k)",
        1000,
        || { let _ = apply_rope(&x_sliding, 8, 256, 10_000.0); },
    );

    // Global: 8 heads, hd=512, 25% rotation, theta=1M
    let x_global = synth_matrix(seq, 8 * 512, 301);
    let global_us = bench(
        "Global   (8×512, 25%, θ=1M)",
        1000,
        || { let _ = apply_rope_partial(&x_global, 8, 512, 1_000_000.0, 0.25); },
    );

    println!("    -> Sliding: {sliding_us:.1}us, Global: {global_us:.1}us");
    println!("    -> Global is {:.1}x vs sliding (larger head_dim but less rotation)\n",
        global_us / sliding_us.max(0.1));

    // ── 5. Correctness: partial fraction=1.0 matches full ──
    println!("--- Correctness Verification ---\n");

    let x = synth_matrix(6, 8 * 256, 400);
    let full = apply_rope(&x, 8, 256, 10_000.0);
    let partial = apply_rope_partial(&x, 8, 256, 10_000.0, 1.0);
    let diff: f32 = full.iter().zip(partial.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  partial(1.0) vs full: max_diff = {diff:.2e} {}\n",
        if diff < 1e-6 { "PASS" } else { "FAIL" });

    // Partial preserves non-rotated dims
    let x = synth_matrix(6, 8 * 512, 401);
    let result = apply_rope_partial(&x, 8, 512, 1_000_000.0, 0.25);
    let rotary_dim = 128; // 512 * 0.25
    let mut preserved = true;
    for pos in 0..6 {
        for h in 0..8 {
            let off = h * 512;
            for d in rotary_dim..512 {
                if (result[[pos, off + d]] - x[[pos, off + d]]).abs() > 1e-6 {
                    preserved = false;
                }
            }
        }
    }
    println!("  partial(0.25) preserves dims [128..512]: {} \n",
        if preserved { "PASS" } else { "FAIL" });

    println!("=== Done ===");
}
