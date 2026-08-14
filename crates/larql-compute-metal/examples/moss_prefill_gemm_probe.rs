//! MOSS prefill GEMM probe — can the existing Metal kernels beat the
//! CPU prefill ceiling at the speech model's exact shapes?
//!
//! Context (`docs/tts-funnel.md`, TTFA gate): the 343-row backbone
//! prefill is 0.97 TFLOP; CPU tops out ≈ 456 ms of matrix work (AMX
//! BLAS) + ~570 ms overhead, so the <500 ms TTFA gate needs Metal.
//! Two candidate kernels already exist — the 32×32-tile f32 `sgemm`
//! (`matmul_transb`) and the Q4_K matmul (`q4k_matmul`, twice falsified
//! for *Gemma* prefill as L1-thrash-bound; MOSS shapes are smaller, so
//! it gets its own reading rather than an assumption).
//!
//! Method: the four distinct projection shapes of the MOSS backbone,
//! timed cache-warm (weights re-used across iterations — models a
//! weight-staged implementation, not per-call upload), best-of-REPS,
//! scaled by per-shape use counts to a whole-prefill total.
//!
//! Usage:
//!   cargo run --release -p larql-compute-metal --example moss_prefill_gemm_probe

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute::backend::{ComputeBackend, MatMul, QuantMatVec};
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    use ndarray::Array2;
    use std::time::Instant;

    /// The parity fixture's prefill rows.
    const SEQ: usize = 343;
    /// Best-of repetitions per shape per kernel.
    const REPS: usize = 5;
    /// MOSS backbone layer count.
    const LAYERS: usize = 28;

    // (name, out_rows, in_cols, uses per prefill)
    let shapes: [(&str, usize, usize, usize); 4] = [
        ("q/o  2048x2048", 2048, 2048, 2 * LAYERS),
        ("k/v   512x2048", 512, 2048, 2 * LAYERS),
        ("g/u  6144x2048", 6144, 2048, 2 * LAYERS),
        ("down 2048x6144", 2048, 6144, LAYERS),
    ];

    let Some(backend) = larql_compute_metal::metal_backend() else {
        eprintln!("no Metal device");
        std::process::exit(1);
    };

    let mut total_f32 = 0.0f64;
    let mut total_q4k = 0.0f64;
    let mut total_flop = 0.0f64;
    println!("MOSS prefill GEMM probe: seq {SEQ}, best-of-{REPS}, cache-warm\n");
    println!("shape            uses   f32 sgemm      q4k matmul");
    for (name, rows, cols, uses) in shapes {
        let weight = Array2::<f32>::from_shape_fn((rows, cols), |(r, c)| {
            ((r * 31 + c * 7) % 23) as f32 * 0.01 - 0.11
        });
        let x = Array2::<f32>::from_shape_fn((SEQ, cols), |(r, c)| {
            ((r * 13 + c * 3) % 19) as f32 * 0.01 - 0.09
        });
        let packed = quantize_q4_k(weight.as_slice().unwrap());
        let flop = 2.0 * rows as f64 * cols as f64 * SEQ as f64;

        let mut best_f32 = f64::INFINITY;
        for _ in 0..REPS {
            let start = Instant::now();
            let out = backend.matmul_transb(x.view(), weight.view());
            let elapsed = start.elapsed().as_secs_f64();
            assert_eq!(out.dim(), (SEQ, rows));
            best_f32 = best_f32.min(elapsed);
        }

        let mut best_q4k = f64::INFINITY;
        let mut q4k_supported = true;
        for _ in 0..REPS {
            let start = Instant::now();
            match backend.q4k_matmul(&packed, x.as_slice().unwrap(), rows, cols, SEQ) {
                Some(out) => {
                    assert_eq!(out.len(), SEQ * rows);
                    best_q4k = best_q4k.min(start.elapsed().as_secs_f64());
                }
                None => {
                    q4k_supported = false;
                    break;
                }
            }
        }

        total_f32 += best_f32 * uses as f64;
        total_flop += flop * uses as f64;
        let q4k_cell = if q4k_supported {
            total_q4k += best_q4k * uses as f64;
            format!(
                "{:7.2} ms ({:4.0} GF)",
                best_q4k * 1000.0,
                flop / best_q4k / 1e9
            )
        } else {
            total_q4k = f64::NAN;
            "unsupported".to_string()
        };
        println!(
            "{name}   x{uses:2}  {:7.2} ms ({:4.0} GF)   {q4k_cell}",
            best_f32 * 1000.0,
            flop / best_f32 / 1e9
        );
    }

    println!(
        "\nwhole-prefill matrix work ({:.2} TFLOP):",
        total_flop / 1e12
    );
    println!(
        "  f32 sgemm    {:6.0} ms  ({:.0} GFLOPS)",
        total_f32 * 1000.0,
        total_flop / total_f32 / 1e9
    );
    if total_q4k.is_finite() {
        println!(
            "  q4k matmul   {:6.0} ms  ({:.0} GFLOPS)",
            total_q4k * 1000.0,
            total_flop / total_q4k / 1e9
        );
    }
    println!("\ncontext: CPU AMX BLAS measured 456 ms; TTFA gate is <500 ms total");
    println!("(per-call staging excluded — a real implementation stages weights once)");
}
