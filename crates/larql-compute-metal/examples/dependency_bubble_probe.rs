//! Does a tiny single-threadgroup reduction between GEMVs cost the GEMV
//! its bandwidth? (The in-situ vs isolated discrepancy, A-12.)
//!
//! In the lowered token every projection GEMV is data-dependent on an
//! RMS norm — one threadgroup, ~1024 threads, the rest of the GPU idle —
//! and the ledger reads attn.proj at ~208 GB/s where the isolated bench
//! reads ~280 on the same shape. Two explanations divide cleanly:
//!
//!   - kernel arithmetic (would show in isolation too) — already ruled
//!     out by the bench;
//!   - a pipeline bubble at every serialization point: the norm drains
//!     the machine, the following GEMV pays a ramp to refill it.
//!
//! Arms, same GEMV kernel and bytes throughout (x2 NVFP4, rotated banks):
//!   A: GEMV-only chain (the isolated number);
//!   B: rms_norm → GEMV alternating (the token's real dependency shape);
//!   C: rms_norm(dependent) → GEMV where the GEMV reads a DIFFERENT,
//!      constant x — same dispatches as B but the GEMV does not depend
//!      on the norm, so Metal may overlap them. B−C is the dependency
//!      cost; C−A is the norm's own occupancy/execution cost.
//!
//! Per-boundary bubble = (B − A) / chain − norm's own time.
use larql_compute_metal::lowering::profile::gpu_span_ms;
use larql_compute_metal::lowering::{MatvecOperands, Nvfp4Kernel};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

const CHAIN: usize = 48;
const REPS: usize = 5;

fn main() {
    let Some(gpu) = MetalBackend::new() else {
        std::process::exit(2)
    };
    // gpt-oss QKV-ish shape.
    let (n, k) = (5120usize, 2880usize);
    let values: Vec<f32> = (0..n * k)
        .map(|i| ((i % 977) as f32 / 977.0) - 0.5)
        .collect();
    let m = nvfp4::quantize(&values, n, k).expect("quantise");
    let bytes = (m.packed.len() + m.scales.len()) as f64;
    let copies = ((256usize << 20) / (bytes as usize)).clamp(1, CHAIN);
    let packed: Vec<Vec<u8>> = (0..copies).map(|_| m.packed.clone()).collect();
    let scales: Vec<Vec<u8>> = (0..copies).map(|_| m.scales.clone()).collect();
    let packed: Vec<_> = packed.iter().map(|b| gpu.lowering_weight(b)).collect();
    let scales: Vec<_> = scales.iter().map(|b| gpu.lowering_weight(b)).collect();

    let x0: Vec<f32> = (0..k).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect();
    let w: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.2).collect();
    let xb = gpu.lowering_upload(&x0).expect("x");
    let wb = gpu.lowering_upload(&w).expect("w");
    let normed = gpu.lowering_scratch(k);
    let xconst = gpu.lowering_upload(&x0).expect("xc");
    let out = gpu.lowering_scratch(n);

    let run = |mode: u8| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..REPS {
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for c in 0..CHAIN {
                if mode > 0 {
                    // out (previous GEMV result, first k floats) -> normed:
                    // a real dependency on the previous iteration.
                    larql_compute_metal::stages::input_norm::encode_f32(
                        enc,
                        &gpu.norms.rms_norm_pipeline,
                        &out,
                        0,
                        &wb,
                        &normed,
                        0,
                        k,
                        1e-6,
                        1.0,
                    );
                }
                let x = match mode {
                    1 => &normed, // GEMV depends on the norm (token shape)
                    2 => &xconst, // same dispatches, GEMV independent
                    _ => &xb,
                };
                gpu.encode_nvfp4_kernel(
                    Nvfp4Kernel::X2,
                    enc,
                    &MatvecOperands {
                        packed: &packed[c % packed.len()],
                        scales: &scales[c % scales.len()],
                        x,
                        out: &out,
                        out_offset: 0,
                        n,
                        k,
                    },
                    m.tensor_scale,
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            best = best.min(gpu_span_ms(&cmd) * 1e3 / CHAIN as f64);
        }
        best
    };

    let a = run(0);
    let b = run(1);
    let c = run(2);
    println!("shape [{n},{k}], {:.1} MB, chain {CHAIN}", bytes / 1e6);
    println!(
        "A  GEMV only              : {a:>7.1} µs  {:>4.0} GB/s",
        bytes / (a / 1e6) / 1e9
    );
    println!(
        "B  norm -> dependent GEMV : {b:>7.1} µs  {:>4.0} GB/s",
        bytes / (b / 1e6) / 1e9
    );
    println!(
        "C  norm ;  independent GEMV: {c:>6.1} µs  {:>4.0} GB/s",
        bytes / (c / 1e6) / 1e9
    );
    println!(
        "per boundary: dependency bubble (B−C) = {:.1} µs; norm own cost (C−A) = {:.1} µs",
        b - c,
        c - a
    );
}
