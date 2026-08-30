//! Does the ~230 us commit-to-GPU-start latency pipeline away?
use larql_models::quant::nvfp4;
fn main() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        std::process::exit(2)
    };
    for &(name, n, k) in &[
        ("attn q  15 MB", 4096usize, 6656usize),
        ("ffn up  75 MB", 19968, 6656),
    ] {
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i % 977) as f32 / 977.0) - 0.5)
            .collect();
        let m = nvfp4::quantize(&values, n, k).expect("quantise");
        let x: Vec<f32> = (0..k).map(|i| (i % 13) as f32 * 0.01).collect();
        print!("{name:<15}");
        for depth in [1usize, 2, 4, 8, 16, 32] {
            for _ in 0..3 {
                gpu.nvfp4_pipelined_cost(&m.packed, &m.scales, m.tensor_scale, &x, n, k, depth);
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let v = gpu
                    .nvfp4_pipelined_cost(&m.packed, &m.scales, m.tensor_scale, &x, n, k, depth)
                    .unwrap();
                best = best.min(v);
            }
            print!("  d{depth}={best:.0}us");
        }
        println!();
    }
    println!("\nper-dispatch cost at each queue depth (best of 5)");
}
