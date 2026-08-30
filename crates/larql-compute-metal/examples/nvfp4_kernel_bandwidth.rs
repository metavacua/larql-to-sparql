//! What bandwidth does `nvfp4_matvec` actually achieve?
//!
//! The decode spends ~427 us per submission. The driver round-trip floor
//! is 19 us, so the remainder is either weight reading at whatever rate
//! the kernel sustains, or our own per-call software cost. Those imply
//! completely different fixes, and the difference between them is one
//! number: the kernel's achieved GB/s.
//!
//! Measured against the ~380 GB/s roofline this machine sustains. A
//! kernel near the roofline means the submission time is real work and
//! the only lever is fewer bytes or better kernels; a kernel far below
//! it means the dispatch geometry is leaving bandwidth on the floor.

use larql_compute::backend::matmul::MatMul;
use larql_models::quant::nvfp4;
use std::time::Instant;

/// Muse-Glimmer's real per-layer projection shapes.
const SHAPES: &[(&str, usize, usize)] = &[
    ("ffn gate/up  [19968, 6656]", 19968, 6656),
    ("ffn down     [6656, 19968]", 6656, 19968),
    ("attn q       [4096, 6656]", 4096, 6656),
    ("attn o       [6656, 4096]", 6656, 4096),
    // The head: 10x the FFN shapes. If a fixed per-call cost dominates,
    // this is where it amortises away and the kernel approaches roofline.
    ("head       [202048, 6656]", 202048, 6656),
];
const ITERS: usize = 60;

fn main() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device");
        std::process::exit(2);
    };
    println!(
        "{:<28}{:>10}{:>11}{:>11}{:>10}",
        "shape", "MB", "ms/call", "GB/s", "% of 380"
    );
    let mut total_mb = 0.0;
    let mut total_ms = 0.0;
    for &(name, n, k) in SHAPES {
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i % 977) as f32 / 977.0) - 0.5)
            .collect();
        let m = nvfp4::quantize(&values, n, k).expect("quantise");
        let x: Vec<f32> = (0..k).map(|i| (i % 13) as f32 * 0.01).collect();
        let bytes = m.packed.len() + m.scales.len();

        // Warm: first touch pays upload and wiring, which is not what
        // steady decode pays.
        for _ in 0..5 {
            gpu.nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, n, k)
                .expect("kernel");
        }
        let t = Instant::now();
        for _ in 0..ITERS {
            gpu.nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, n, k)
                .expect("kernel");
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
        let gbs = bytes as f64 / (ms / 1e3) / 1e9;
        println!(
            "{name:<28}{:>10.1}{:>11.3}{:>11.1}{:>9.0}%",
            bytes as f64 / 1e6,
            ms,
            gbs,
            100.0 * gbs / 380.0
        );
        total_mb += bytes as f64 / 1e6;
        total_ms += ms;
    }
    println!();
    println!(
        "one layer's four projections: {:.1} MB in {:.3} ms -> {:.1} GB/s",
        total_mb,
        total_ms,
        total_mb / 1e3 / (total_ms / 1e3)
    );
}
