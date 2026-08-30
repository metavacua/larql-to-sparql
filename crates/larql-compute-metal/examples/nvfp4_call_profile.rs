//! R4a: attribute the ~300 us per-call intercept.
//!
//! Three shapes spanning 50x in bytes. Anything roughly constant across
//! them is the fixed tax; anything proportional is real work.

use larql_models::quant::nvfp4;

const SHAPES: &[(&str, usize, usize)] = &[
    ("attn q   15 MB", 4096, 6656),
    ("ffn up   75 MB", 19968, 6656),
    ("head    756 MB", 202048, 6656),
];
const ITERS: usize = 40;

fn main() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device");
        std::process::exit(2);
    };
    println!(
        "{:<16}{:>8}{:>8}{:>8}{:>8}{:>8}{:>9}{:>9}{:>9}{:>9}",
        "shape", "input", "bufacq", "encode", "commit", "wait", "gpu", "queue", "readbk", "total"
    );
    for &(name, n, k) in SHAPES {
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i % 977) as f32 / 977.0) - 0.5)
            .collect();
        let m = nvfp4::quantize(&values, n, k).expect("quantise");
        let x: Vec<f32> = (0..k).map(|i| (i % 13) as f32 * 0.01).collect();

        for _ in 0..5 {
            gpu.profile_nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, n, k)
                .expect("profile");
        }
        let mut acc = [0.0f64; 9];
        for _ in 0..ITERS {
            let p = gpu
                .profile_nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, n, k)
                .expect("profile");
            for (a, v) in acc.iter_mut().zip([
                p.input_stage,
                p.buffer_acquire,
                p.encode,
                p.commit,
                p.wait,
                p.gpu_span,
                p.commit_to_gpu_start,
                p.readback,
                p.total,
            ]) {
                *a += v;
            }
        }
        let a: Vec<f64> = acc.iter().map(|v| v / ITERS as f64).collect();
        println!(
            "{name:<16}{:>8.1}{:>8.1}{:>8.1}{:>8.1}{:>8.1}{:>9.1}{:>9.1}{:>9.1}{:>9.1}",
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]
        );
    }
    println!("\nall values microseconds. 'gpu' is GPUEndTime-GPUStartTime;");
    println!("'queue' is commit->completion wall minus the GPU's own span.");
}
