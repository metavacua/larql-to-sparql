//! CPU-1A2: what the dense projections around Gated DeltaNet cost, and
//! whether the BLAS-backed path is faster at each shape.
//!
//! Measurement, not a pass/fail gate — env-gated so it never runs in CI:
//!
//! ```text
//! QW_PROJ_BENCH=1 cargo test --release projection_bench -- --nocapture
//! ```
//!
//! The two tiny projections (`in_proj_a`/`in_proj_b`, 48 x 5120) are here
//! precisely because BLAS call overhead may LOSE on them. A strategy that
//! routed all five through BLAS for conceptual tidiness would be choosing
//! consistency over measurement.

use std::time::Instant;

use crate::format::vindex3::fixtures::lcg_values;
use crate::format::vindex3::opplan::exec::kernels::matvec;

/// Qwen3.8's real projection shapes, `(name, out_dim, in_dim)`.
const SHAPES: &[(&str, usize, usize)] = &[
    ("delta in_proj_qkv", 10240, 5120),
    ("delta in_proj_z", 6144, 5120),
    ("delta out_proj", 5120, 6144),
    ("delta in_proj_a", 48, 5120),
    ("ffn gate/up (control)", 17408, 5120),
];

fn bench(label: &str, out_dim: usize, in_dim: usize) {
    let w = lcg_values(out_dim * in_dim, 11);
    let x = lcg_values(in_dim, 22);
    let macs = (out_dim * in_dim) as f64;
    let bytes = (out_dim * in_dim * 4) as f64;

    // Iterations scaled so every shape gets a comparable amount of work;
    // one pass over a 210 MB weight is already long, a 48-row one is not.
    let iters = (2_000_000_000.0 / macs).clamp(3.0, 400.0) as usize;

    let mut sink = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..iters {
        let y = matvec(&w, out_dim, in_dim, &x);
        sink += y[0];
    }
    let scalar = t0.elapsed().as_secs_f64() / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        let y = larql_compute::cpu::ops::moe::math::matmul_vec(&x, &w, out_dim, in_dim);
        sink += y[0];
    }
    let blas = t1.elapsed().as_secs_f64() / iters as f64;

    // Same inputs, so a disagreement is the arithmetic, not the data.
    let a = matvec(&w, out_dim, in_dim, &x);
    let b = larql_compute::cpu::ops::moe::math::matmul_vec(&x, &w, out_dim, in_dim);
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (p, q) in a.iter().zip(&b) {
        num += (*p as f64 - *q as f64).powi(2);
        den += (*q as f64).powi(2);
    }
    let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();

    println!(
        "  {label:22} {out_dim:>6}x{in_dim:<5}  scalar {:8.2} ms {:6.2} GB/s  |  \
         blas {:8.2} ms {:7.2} GB/s  |  {:6.1}x  rel {:.2e}",
        scalar * 1e3,
        bytes / scalar / 1e9,
        blas * 1e3,
        bytes / blas / 1e9,
        scalar / blas,
        rel
    );
    std::hint::black_box(sink);
}

#[test]
fn projection_bench() {
    if std::env::var("QW_PROJ_BENCH").is_err() {
        eprintln!("SKIP projection_bench: set QW_PROJ_BENCH=1");
        return;
    }
    println!("\n  Qwen3.8 dense projections — scalar vs larql-compute BLAS\n");
    for (label, out_dim, in_dim) in SHAPES {
        bench(label, *out_dim, *in_dim);
    }
    println!();
}
