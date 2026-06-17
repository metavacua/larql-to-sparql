//! Demo: `ridge_decomposition_solve` — the closed-form ridge solve
//! that underlies MEMIT-style weight edits.
//!
//! Solves   ΔW = T^T (K K^T + λI)^{-1} K
//!
//! Run:  cargo run --release -p larql-compute --example demo_ridge_solve
//!
//! Walks three regimes:
//!   1. Orthonormal keys → exact reconstruction.
//!   2. Near-singular keys → λ rescues the system; recon degrades.
//!   3. High-d random keys → realistic MEMIT shapes (Gemma 4B-ish).

extern crate blas_src;

use larql_compute::cpu::ops::linalg::ridge_decomposition_solve;
use ndarray::{Array1, Array2};
use std::time::Instant;

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn cosine(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn report(label: &str, keys: &Array2<f32>, targets: &Array2<f32>, lambda: f32) {
    let n = keys.nrows();
    let d = keys.ncols();
    let t0 = Instant::now();
    let delta_w = match ridge_decomposition_solve(keys, targets, lambda) {
        Ok(d) => d,
        Err(e) => {
            println!("  [{label}] N={n} d={d} λ={lambda:.0e} → ERROR: {e}");
            return;
        }
    };
    let elapsed = t0.elapsed();

    let mut min_cos = f32::INFINITY;
    let mut sum_cos = 0.0_f32;
    for i in 0..n {
        let recon = delta_w.dot(&keys.row(i));
        let cos = cosine(&recon, &targets.row(i).to_owned());
        min_cos = min_cos.min(cos);
        sum_cos += cos;
    }
    let frob: f32 = delta_w.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "  [{label:<22}] N={n:<3} d={d:<5} λ={lambda:.0e}  mean_cos={:.4} min_cos={:.4} \
         ‖ΔW‖={:>8.2}  ({:>6.2}ms)",
        sum_cos / n as f32,
        min_cos,
        frob,
        elapsed.as_secs_f64() * 1e3,
    );
}

fn main() {
    println!("=== ridge_decomposition_solve demo ===\n");

    // ── Regime 1: orthonormal keys ──
    println!("Regime 1 — orthonormal keys (exact reconstruction expected):");
    let n = 6;
    let d = 16;
    let mut keys = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        keys[[i, i]] = 1.0;
    }
    let mut targets = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        targets[[i, (i + n) % d]] = 1.0;
    }
    report("orthonormal", &keys, &targets, 1e-6);

    // ── Regime 2: near-singular keys ──
    println!("\nRegime 2 — keys share dominant direction (template-like, exp 8 case):");
    let n = 8;
    let d = 16;
    let template = Array1::<f32>::from_shape_fn(d, |i| (i as f32 * 0.3).sin());
    let mut keys = Array2::<f32>::zeros((n, d));
    let mut state = 7u64;
    for i in 0..n {
        for j in 0..d {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            keys[[i, j]] = template[j] * 100.0 + noise * 0.1;
        }
    }
    let targets = synth(n, d, 99);
    for &lambda in &[1e-6_f32, 1e-3, 1e-1, 1.0] {
        report("template+noise", &keys, &targets, lambda);
    }

    // ── Regime 3: realistic MEMIT scale ──
    println!("\nRegime 3 — MEMIT-realistic shapes (random keys, hidden_dim ≥ 576):");
    for &(n, d) in &[(10usize, 576usize), (30, 576), (60, 2560), (120, 2560)] {
        let keys = synth(n, d, 1);
        let targets = synth(n, d, 2);
        report("random@hidden", &keys, &targets, 1e-3);
    }

    println!("\nDone.");
}
