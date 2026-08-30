//! GPT-OSS expert α/B decomposition (A-12 expert pass, step 1).
//!
//! The ledger prices the routed-FFN stage at ~57% of roofline. Before
//! touching the kernel, decompose the number the way A-5 taught:
//! same bytes, same shape, progressively more routing machinery —
//!
//! | arm        | machinery                                            |
//! |------------|------------------------------------------------------|
//! | f16x4      | attainable control (equal shape, more bytes)         |
//! | matvec1x4  | contiguous per-expert `mxfp4_matvec`, no indirection |
//! | grouped1x4 | production grouped kernel, one slot per dispatch     |
//! | grouped4   | production grouped kernel, 4 slots in one dispatch — |
//! |            | exactly the decode's gate/up/down dispatch           |
//!
//! matvec1x4 vs grouped1x4 prices the offset-table indirection; grouped1x4
//! vs grouped4 prices the slot-grid parallelism; the f16 arm says what the
//! GPU sustains on this geometry. Chained in one command buffer, GPU span
//! from the command buffer, rotating ≥256 MB of distinct expert banks so
//! the steady state reads DRAM, not the SLC (the A-5 bench trap).
//!
//! Weights are synthesised MXFP4 bytes (random nibbles, e8m0 scales near
//! 1.0): bandwidth does not care about values, and cross-arm parity is
//! checked on the identical bytes.
//!
//! Run on AC:
//!   cargo run --release -p larql-compute-metal --example moe_expert_alpha_b

extern crate blas_src;

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute_metal::lowering::profile::gpu_span_ms;
    use larql_compute_metal::lowering::{MatvecOperands, MatvecTarget};
    use larql_compute_metal::MetalBackend;
    use metal::MTLSize;

    // GPT-OSS expert geometry: one gate/up half or the down projection.
    const N: usize = 2880;
    const K: usize = 2880;
    const TOP_K: usize = 4;
    const GROUPS: usize = K / 32;
    const ROW_BYTES: usize = GROUPS * 16; // packed nibbles per row
    const SCALE_BYTES: usize = GROUPS; // e8m0 per row
    const CHAIN: usize = 24; // one dispatch per layer, like a token
    const REPS: usize = 5;
    const ROOFLINE: f64 = 367.0;

    let Some(gpu) = MetalBackend::new() else {
        eprintln!("Metal unavailable");
        return;
    };

    // Distinct banks so CHAIN dispatches never re-read cached bytes:
    // each bank holds TOP_K experts' payloads+scales (~17.6 MB).
    let per_expert_p = N * ROW_BYTES;
    let per_expert_s = N * SCALE_BYTES;
    let bank_bytes = TOP_K * (per_expert_p + per_expert_s);
    let n_banks = (256usize << 20).div_ceil(bank_bytes).min(CHAIN);
    println!(
        "expert [{N},{K}] x{TOP_K}: {:.1} MB per dispatch, {n_banks} banks rotated",
        bank_bytes as f64 / 1e6
    );

    let mut state = 0x9e3779b9u32;
    let mut rand_bytes = |len: usize| -> Vec<u8> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect()
    };
    // Payloads: random nibbles. Scales: e8m0 126..=128 (0.5..2.0) so
    // magnitudes stay sane for the parity check.
    let banks_p: Vec<Vec<u8>> = (0..n_banks)
        .map(|_| rand_bytes(TOP_K * per_expert_p))
        .collect();
    let banks_s: Vec<Vec<u8>> = (0..n_banks)
        .map(|_| {
            rand_bytes(TOP_K * per_expert_s)
                .into_iter()
                .map(|b| 126 + (b % 3))
                .collect()
        })
        .collect();
    let banks_p_buf: Vec<_> = banks_p.iter().map(|b| gpu.lowering_weight(b)).collect();
    let banks_s_buf: Vec<_> = banks_s.iter().map(|b| gpu.lowering_weight(b)).collect();
    // Per-expert slices of the same bytes, for the arms whose encode has
    // no weight offset (address-keyed buffer cache: a subslice is its own
    // buffer, same underlying allocation).
    let expert_p: Vec<Vec<_>> = banks_p
        .iter()
        .map(|b| {
            (0..TOP_K)
                .map(|e| gpu.lowering_weight(&b[e * per_expert_p..(e + 1) * per_expert_p]))
                .collect()
        })
        .collect();
    let expert_s: Vec<Vec<_>> = banks_s
        .iter()
        .map(|b| {
            (0..TOP_K)
                .map(|e| gpu.lowering_weight(&b[e * per_expert_s..(e + 1) * per_expert_s]))
                .collect()
        })
        .collect();
    // f16 control: same shape, per-expert buffers (values irrelevant).
    let f16_banks: Vec<Vec<u8>> = (0..n_banks)
        .map(|_| rand_bytes(TOP_K * N * K * 2))
        .collect();
    let f16_experts: Vec<Vec<_>> = f16_banks
        .iter()
        .map(|b| {
            (0..TOP_K)
                .map(|e| gpu.lowering_weight(&b[e * N * K * 2..(e + 1) * N * K * 2]))
                .collect()
        })
        .collect();

    // Offset tables: slot s -> expert s's base inside the bank.
    let offs: Vec<u32> = (0..TOP_K).map(|s| (s * per_expert_p) as u32).collect();
    let soffs: Vec<u32> = (0..TOP_K).map(|s| (s * per_expert_s) as u32).collect();
    let offs_bytes: Vec<u8> = offs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let soffs_bytes: Vec<u8> = soffs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let offs_buf = gpu.lowering_weight(&offs_bytes);
    let soffs_buf = gpu.lowering_weight(&soffs_bytes);
    let zero_off = gpu.lowering_weight(&0u32.to_le_bytes());

    let x: Vec<f32> = (0..K).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    let out = gpu.lowering_scratch(TOP_K * N);

    let (kh, _) = (
        &gpu.quant.mxfp4_grouped_pipeline,
        gpu.quant.mxfp4_grouped_binding,
    );
    assert!(
        format!("{:?}", gpu.quant.mxfp4_grouped_arm).contains("Vec"),
        "expected the production vec arm, got {:?}",
        gpu.quant.mxfp4_grouped_arm
    );

    let set_u32 = |enc: &metal::ComputeCommandEncoderRef, idx: u64, v: u32| {
        enc.set_bytes(idx, 4, &v as *const u32 as *const std::ffi::c_void);
    };

    // One grouped dispatch over `slots` slots of bank `b`, slot base `s0`.
    let grouped = |enc: &metal::ComputeCommandEncoderRef, b: usize, slots: usize, s0: usize| {
        let row_tiles = (N as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&banks_p_buf[b]), (s0 * per_expert_p) as u64);
        enc.set_buffer(1, Some(&offs_buf), 0);
        enc.set_buffer(2, Some(&banks_s_buf[b]), (s0 * per_expert_s) as u64);
        enc.set_buffer(3, Some(&soffs_buf), 0);
        enc.set_buffer(4, Some(&xb), 0);
        enc.set_buffer(5, Some(&out), (s0 * N * 4) as u64);
        set_u32(enc, 6, N as u32);
        set_u32(enc, 7, K as u32);
        set_u32(enc, 8, 0); // shared X
        set_u32(enc, 9, 0); // ROWBASE identity
        set_u32(enc, 10, 1); // ROWSTRIDE identity
        enc.dispatch_thread_groups(
            MTLSize::new(row_tiles, slots as u64, 1),
            MTLSize::new(kh.threads_per_tg, 1, 1),
        );
    };
    // Same kernel, offsets forced to zero and the base pointer moved: the
    // single-slot arm reads through the same indirection with slot 0.
    let grouped_one = |enc: &metal::ComputeCommandEncoderRef, b: usize, e: usize| {
        let row_tiles = (N as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&banks_p_buf[b]), (e * per_expert_p) as u64);
        enc.set_buffer(1, Some(&zero_off), 0);
        enc.set_buffer(2, Some(&banks_s_buf[b]), (e * per_expert_s) as u64);
        enc.set_buffer(3, Some(&zero_off), 0);
        enc.set_buffer(4, Some(&xb), 0);
        enc.set_buffer(5, Some(&out), (e * N * 4) as u64);
        set_u32(enc, 6, N as u32);
        set_u32(enc, 7, K as u32);
        set_u32(enc, 8, 0);
        set_u32(enc, 9, 0);
        set_u32(enc, 10, 1);
        enc.dispatch_thread_groups(
            MTLSize::new(row_tiles, 1, 1),
            MTLSize::new(kh.threads_per_tg, 1, 1),
        );
    };

    // A2x2 (production) and the 313→346 candidates.
    let kh2 = &gpu.quant.mxfp4_grouped_x2_pipeline;
    let kh2p = &gpu.quant.mxfp4_grouped_x2p_pipeline;
    let kh4 = &gpu.quant.mxfp4_grouped_x4_pipeline;
    let grouped_arm = |enc: &metal::ComputeCommandEncoderRef,
                       kh: &larql_compute_metal::kernels::KernelHandle,
                       b: usize,
                       slots: usize| {
        let row_tiles = (N as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&banks_p_buf[b]), 0);
        enc.set_buffer(1, Some(&offs_buf), 0);
        enc.set_buffer(2, Some(&banks_s_buf[b]), 0);
        enc.set_buffer(3, Some(&soffs_buf), 0);
        enc.set_buffer(4, Some(&xb), 0);
        enc.set_buffer(5, Some(&out), 0);
        set_u32(enc, 6, N as u32);
        set_u32(enc, 7, K as u32);
        set_u32(enc, 8, 0);
        set_u32(enc, 9, 0);
        set_u32(enc, 10, 1);
        enc.dispatch_thread_groups(
            MTLSize::new(row_tiles, slots as u64, 1),
            MTLSize::new(kh.threads_per_tg, 1, 1),
        );
    };

    #[derive(Clone, Copy, PartialEq)]
    enum Arm {
        F16x4,
        Matvec1x4,
        Grouped1x4,
        Grouped4,
        GroupedX2,
        GroupedX2P,
        GroupedX4,
    }
    let arms = [
        Arm::F16x4,
        Arm::Matvec1x4,
        Arm::Grouped1x4,
        Arm::Grouped4,
        Arm::GroupedX2,
        Arm::GroupedX2P,
        Arm::GroupedX4,
    ];
    let name = |a: Arm| match a {
        Arm::F16x4 => "f16 x4 (attainable)",
        Arm::Matvec1x4 => "mxfp4_matvec x4 (no indirection)",
        Arm::Grouped1x4 => "grouped kernel, 1 slot x4",
        Arm::Grouped4 => "grouped kernel, 4 slots x1 (production)",
        Arm::GroupedX2 => "grouped x2, 4 slots x1 (production)",
        Arm::GroupedX2P => "grouped x2 + byte-pair LUT",
        Arm::GroupedX4 => "grouped x4 (4 rows/lane)",
    };

    println!(
        "{:<40}{:>10}{:>10}{:>10}{:>9}",
        "arm", "µs/layer", "floor µs", "over", "GB/s"
    );
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for arm in arms {
        let bytes = match arm {
            Arm::F16x4 => (TOP_K * N * K * 2) as f64,
            _ => bank_bytes as f64,
        };
        let mut best = f64::MAX;
        for _ in 0..REPS {
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for c in 0..CHAIN {
                let b = c % n_banks;
                match arm {
                    Arm::F16x4 => {
                        for (e, w) in f16_experts[b].iter().enumerate() {
                            gpu.encode_f16_matvec(
                                enc,
                                w,
                                &MatvecTarget {
                                    x: &xb,
                                    out: &out,
                                    out_offset: (e * N * 4) as u64,
                                    n: N,
                                    k: K,
                                },
                            );
                        }
                    }
                    Arm::Matvec1x4 => {
                        for e in 0..TOP_K {
                            gpu.encode_mxfp4_matvec(
                                enc,
                                &MatvecOperands {
                                    packed: &expert_p[b][e],
                                    scales: &expert_s[b][e],
                                    x: &xb,
                                    out: &out,
                                    out_offset: (e * N * 4) as u64,
                                    n: N,
                                    k: K,
                                },
                            );
                        }
                    }
                    Arm::Grouped1x4 => {
                        for e in 0..TOP_K {
                            grouped_one(enc, b, e);
                        }
                    }
                    Arm::Grouped4 => grouped(enc, b, TOP_K, 0),
                    Arm::GroupedX2 => grouped_arm(enc, kh2, b, TOP_K),
                    Arm::GroupedX2P => grouped_arm(enc, kh2p, b, TOP_K),
                    Arm::GroupedX4 => grouped_arm(enc, kh4, b, TOP_K),
                }
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            best = best.min(gpu_span_ms(&cmd) * 1e3 / CHAIN as f64);
        }
        let floor = bytes / ROOFLINE / 1e3;
        println!(
            "{:<40}{:>10.1}{:>10.1}{:>10.1}{:>9.0}",
            name(arm),
            best,
            floor,
            best - floor,
            bytes / (best / 1e6) / 1e9
        );
        outputs.push(gpu.lowering_readback(&out, TOP_K * N).expect("readback"));
    }
    // Parity: the MXFP4 arms decode identical bytes.
    let (m, g1, g4, gx2) = (&outputs[1], &outputs[2], &outputs[3], &outputs[4]);
    assert_eq!(g1, g4, "grouped 1-slot vs 4-slot differ");
    assert_eq!(g4, gx2, "x2 arm diverged from the production kernel");
    for (name, o) in [("x2p", &outputs[5]), ("x4", &outputs[6])] {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (a, b) in gx2.iter().zip(o.iter()) {
            num += ((a - b) as f64).powi(2);
            den += (*a as f64).powi(2);
        }
        let rel = (num / den.max(1e-30)).sqrt();
        assert!(rel < 1e-6, "{name} rel_rms {rel}");
        println!("{name} vs x2 rel_rms {rel:.1e}");
    }
    let close = m
        .iter()
        .zip(g4)
        .all(|(a, b)| (a - b).abs() <= 1e-3 * a.abs().max(1.0));
    println!(
        "parity: grouped1x4 == grouped4 bit-for-bit; matvec vs grouped {}",
        if close { "≤1e-3 rel" } else { "DIFFER" }
    );
    gpu.recycle_lowering_scratch(out);
    gpu.recycle_lowering_scratch(xb);
}
