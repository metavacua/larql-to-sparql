//! GEMV arms on the ledger's shapes, measured the way the lowered token
//! runs them — `CHAIN` dependent dispatches in ONE command buffer, GPU
//! span off the command buffer — and fitted per arm to
//!
//! ```text
//! T(shape) = α + bytes / B
//! ```
//!
//! so a variant is judged on what it moves: α (fixed per-dispatch ramp)
//! or B (sustained bandwidth). At the small shapes the GB/s lens lies —
//! a [2560,2560] NVFP4 GEMV is ~11 µs over a 4.7 µs byte floor while its
//! f16 twin is 21 over 16.7: both pay the same ramp, the quantised one
//! shows it as a worse ratio — so the table reports µs over floor too.
//!
//! Arms: f16 GEMV, NVFP4 v1 (production), v2 (falsified decode
//! hypothesis), and the sweep (groups per lane per step × rows
//! per threadgroup). NVFP4 arms are checked against v1 to fp32 rounding.
//!
//! The chain rotates over `ROTATE_BYTES` worth of distinct copies of the
//! matrix: re-reading ONE small matrix 40× measures the system-level
//! cache (f16 read 625–752 GB/s on 12–24 MB shapes, above DRAM roofline,
//! in the first version of this bench), and the lowered token streams
//! every weight from DRAM exactly once. With rotation B is DRAM
//! bandwidth; the cache-fed number is still informative for one thing —
//! a kernel that cannot beat the roofline *from cache* is compute-bound.
//!
//! Run on AC; on battery v1 alone drifts 213↔280 GB/s between runs.
use larql_compute_metal::lowering::profile::gpu_span_ms;
use larql_compute_metal::lowering::{
    MatvecOperands, MatvecTarget, Nvfp4Kernel, Nvfp4Segment, PreNorm,
};
use larql_compute_metal::MetalBackend;
use larql_models::quant::half::f32_to_f16;
use larql_models::quant::nvfp4;

const CHAIN: usize = 40;
const REPS: usize = 5;
/// Distinct weight bytes the chain cycles through — past the M3 Max
/// system cache, so the steady state reads from DRAM.
const ROTATE_BYTES: usize = 256 << 20;
const GPU_READ_CEILING_GB_S: f64 = 367.0;

/// (name, n rows, k cols) — the ledger's projection shapes.
const SHAPES: &[(&str, usize, usize)] = &[
    ("granite3b q    [2560,2560]", 2560, 2560),
    ("granite3b gate [8192,2560]", 8192, 2560),
    ("granite3b down [2560,8192]", 2560, 8192),
    ("gemma4 q       [4096,2816]", 4096, 2816),
    ("gemma4 k/v     [2048,2816]", 2048, 2816),
    ("gemma4 dense   [2112,2816]", 2112, 2816),
    ("gptoss q       [4096,2880]", 4096, 2880),
    ("gptoss o       [2880,4096]", 2880, 4096),
    ("glimmer q      [4096,6656]", 4096, 6656),
    ("glimmer o      [6656,4096]", 6656, 4096),
    ("granite8b gate [12800,4096]", 12800, 4096),
    ("head          [100352,2560]", 100352, 2560),
];

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    F16,
    Nvfp4(Nvfp4Kernel),
    /// The segmented x2 kernel with ONE segment (what a fused Q+K+V or a
    /// residual-folded o-proj pays per byte), with/without the residual.
    Seg1,
    Seg1Residual,
    /// x2 with the RMS norm of X folded into the prologue (rung 2d).
    X2PreNorm,
    /// Form B: normalised X staged in threadgroup memory.
    X2PreNormStaged,
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::F16 => "f16".into(),
            Arm::Nvfp4(k) => format!("nvfp4.{}", k.name()),
            Arm::Seg1 => "nvfp4.seg1".into(),
            Arm::Seg1Residual => "nvfp4.seg1+res".into(),
            Arm::X2PreNorm => "nvfp4.x2+norm".into(),
            Arm::X2PreNormStaged => "nvfp4.x2+normTG".into(),
        }
    }
}

#[derive(Clone)]
struct Sample {
    bytes: f64,
    us: f64,
}

/// Least-squares `us = alpha + bytes / B`; returns (alpha µs, B GB/s, r²).
fn fit(samples: &[Sample]) -> (f64, f64, f64) {
    let n = samples.len() as f64;
    let mx = samples.iter().map(|s| s.bytes).sum::<f64>() / n;
    let my = samples.iter().map(|s| s.us).sum::<f64>() / n;
    let sxx: f64 = samples.iter().map(|s| (s.bytes - mx).powi(2)).sum();
    let sxy: f64 = samples.iter().map(|s| (s.bytes - mx) * (s.us - my)).sum();
    let slope = sxy / sxx; // µs per byte
    let alpha = my - slope * mx;
    let ss_res: f64 = samples
        .iter()
        .map(|s| (s.us - (alpha + slope * s.bytes)).powi(2))
        .sum();
    let ss_tot: f64 = samples.iter().map(|s| (s.us - my).powi(2)).sum();
    // µs per byte → bytes per second = 1e6 / slope; GB/s = /1e9.
    (alpha, 1e6 / slope / 1e9, 1.0 - ss_res / ss_tot)
}

fn main() {
    let Some(gpu) = MetalBackend::new() else {
        std::process::exit(2)
    };
    let arms: Vec<Arm> = std::iter::once(Arm::F16)
        .chain(Nvfp4Kernel::ALL.into_iter().map(Arm::Nvfp4))
        .chain([
            Arm::Seg1,
            Arm::Seg1Residual,
            Arm::X2PreNorm,
            Arm::X2PreNormStaged,
        ])
        .collect();
    let mut per_arm: Vec<Vec<Sample>> = vec![Vec::new(); arms.len()];

    println!("per shape: µs per call (best of {REPS}, chain {CHAIN}); floor = bytes/{GPU_READ_CEILING_GB_S:.0} GB/s; NVFP4 arms vs v1 rel_rms");
    for &(name, n, k) in SHAPES {
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i % 977) as f32 / 977.0) - 0.5)
            .collect();
        let x: Vec<f32> = (0..k).map(|i| (i % 13) as f32 * 0.01 - 0.05).collect();
        let xb = gpu.lowering_upload(&x).expect("x");
        let out = gpu.lowering_scratch(n);
        let nv = nvfp4::quantize(&values, n, k).expect("nvfp4");
        let f16: Vec<u8> = values
            .iter()
            .flat_map(|v| f32_to_f16(*v).to_le_bytes())
            .collect();
        // Copies so the chain does not re-read one cache-resident matrix.
        let nv_bytes = nv.packed.len() + nv.scales.len();
        let nv_copies = (ROTATE_BYTES / nv_bytes).clamp(1, CHAIN);
        let f16_copies = (ROTATE_BYTES / f16.len()).clamp(1, CHAIN);
        let nv_packed: Vec<Vec<u8>> = (0..nv_copies).map(|_| nv.packed.clone()).collect();
        let nv_scales: Vec<Vec<u8>> = (0..nv_copies).map(|_| nv.scales.clone()).collect();
        let f16_copies_v: Vec<Vec<u8>> = (0..f16_copies).map(|_| f16.clone()).collect();
        let nv_packed: Vec<_> = nv_packed.iter().map(|b| gpu.lowering_weight(b)).collect();
        let nv_scales: Vec<_> = nv_scales.iter().map(|b| gpu.lowering_weight(b)).collect();
        let f16_bufs: Vec<_> = f16_copies_v
            .iter()
            .map(|b| gpu.lowering_weight(b))
            .collect();
        println!("{name}");
        let mut v1_out: Option<Vec<f32>> = None;
        for (ai, arm) in arms.iter().enumerate() {
            let bytes = match arm {
                Arm::F16 => f16.len(),
                _ => nv.packed.len() + nv.scales.len(),
            } as f64;
            let mut best = f64::MAX;
            for _ in 0..REPS {
                let cmd = gpu.new_lowering_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for c in 0..CHAIN {
                    match arm {
                        Arm::F16 => gpu.encode_f16_matvec(
                            enc,
                            &f16_bufs[c % f16_bufs.len()],
                            &MatvecTarget {
                                x: &xb,
                                out: &out,
                                out_offset: 0,
                                n,
                                k,
                            },
                        ),
                        Arm::Nvfp4(which) => gpu.encode_nvfp4_kernel(
                            *which,
                            enc,
                            &MatvecOperands {
                                packed: &nv_packed[c % nv_packed.len()],
                                scales: &nv_scales[c % nv_scales.len()],
                                x: &xb,
                                out: &out,
                                out_offset: 0,
                                n,
                                k,
                            },
                            nv.tensor_scale,
                        ),
                        Arm::X2PreNorm => gpu.encode_nvfp4_matvec_prenorm(
                            enc,
                            &MatvecOperands {
                                packed: &nv_packed[c % nv_packed.len()],
                                scales: &nv_scales[c % nv_scales.len()],
                                x: &xb,
                                out: &out,
                                out_offset: 0,
                                n,
                                k,
                            },
                            nv.tensor_scale,
                            &PreNorm {
                                weight: &xb,
                                eps: 1e-6,
                                offset: 0.0,
                            },
                        ),
                        Arm::X2PreNormStaged => {
                            if k <= MetalBackend::PRENORM_STAGED_MAX_K {
                                gpu.encode_nvfp4_matvec_prenorm_staged(
                                    enc,
                                    &MatvecOperands {
                                        packed: &nv_packed[c % nv_packed.len()],
                                        scales: &nv_scales[c % nv_scales.len()],
                                        x: &xb,
                                        out: &out,
                                        out_offset: 0,
                                        n,
                                        k,
                                    },
                                    nv.tensor_scale,
                                    &PreNorm {
                                        weight: &xb,
                                        eps: 1e-6,
                                        offset: 0.0,
                                    },
                                )
                            }
                        }
                        Arm::Seg1 | Arm::Seg1Residual => {
                            let seg = Nvfp4Segment {
                                packed: &nv_packed[c % nv_packed.len()],
                                packed_offset: 0,
                                scales: &nv_scales[c % nv_scales.len()],
                                scales_offset: 0,
                                tensor_scale: nv.tensor_scale,
                                out: &out,
                                out_offset: 0,
                                n,
                            };
                            let residual = (*arm == Arm::Seg1Residual).then_some(&out);
                            gpu.encode_nvfp4_matvec_segments_residual(enc, &xb, k, &[seg], residual)
                        }
                    }
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                best = best.min(gpu_span_ms(&cmd) * 1e3 / CHAIN as f64);
            }
            let floor_us = bytes / GPU_READ_CEILING_GB_S / 1e3;
            let got = gpu.lowering_readback(&out, n).expect("readback");
            let parity = match arm {
                Arm::Nvfp4(Nvfp4Kernel::V1) => {
                    v1_out = Some(got);
                    String::new()
                }
                Arm::Nvfp4(_) | Arm::Seg1 => {
                    let r = v1_out.as_ref().expect("v1 ran first");
                    let (mut num, mut den) = (0.0f64, 0.0f64);
                    for (a, b) in r.iter().zip(&got) {
                        num += ((a - b) as f64).powi(2);
                        den += (*a as f64).powi(2);
                    }
                    format!("  rel_rms {:.1e}", (num / den.max(1e-30)).sqrt())
                }
                _ => String::new(),
            };
            println!(
                "  {:<12} {:>8.1} MB {:>9.1} us  floor {:>7.1}  over {:>7.1}  {:>5.0} GB/s{parity}",
                arm.name(),
                bytes / 1e6,
                best,
                floor_us,
                best - floor_us,
                bytes / (best / 1e6) / 1e9
            );
            per_arm[ai].push(Sample { bytes, us: best });
        }
        gpu.recycle_lowering_scratch(out);
        gpu.recycle_lowering_scratch(xb);
    }

    println!();
    println!(
        "fit T = α + bytes/B over {} shapes (head excluded: it would dominate the slope)",
        SHAPES.len() - 1
    );
    println!(
        "  {:<12} {:>10} {:>10} {:>6}",
        "arm", "α µs", "B GB/s", "r²"
    );
    for (ai, arm) in arms.iter().enumerate() {
        let samples: Vec<Sample> = per_arm[ai]
            .iter()
            .take(SHAPES.len() - 1)
            .map(|s| Sample {
                bytes: s.bytes,
                us: s.us,
            })
            .collect();
        let (alpha, b, r2) = fit(&samples);
        println!("  {:<12} {alpha:>10.2} {b:>10.0} {r2:>6.3}", arm.name());
    }
}
