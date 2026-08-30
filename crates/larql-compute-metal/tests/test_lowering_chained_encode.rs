//! G6a foundation: does encoding a chain into **one** command buffer give
//! the same numbers as one-command-buffer-per-matvec, and does it remove
//! the queue-starvation tax?
//!
//! Both halves matter and neither alone is enough. Faster-but-different
//! is not an optimisation, and identical-but-equally-slow would mean the
//! premise is wrong. So this asserts bit-equality first and reports the
//! speedup second.
//!
//! Bit-equality is the right bar here, unusually: the two arms run the
//! *same kernel* with the *same arguments* and the same per-row reduction
//! order. Only the scheduling differs, so any difference at all would
//! signal a real hazard — a missing dependency between chained
//! dispatches, or a recycled buffer being read after reuse.
//!
//! The chain feeds each matvec's output into the next call's input, which
//! is the dependency shape a real layer has (QKV -> attention -> o_proj
//! -> FFN). Metal's default serial-dispatch encoder must order them; if
//! it did not, this test would fail non-deterministically rather than
//! silently produce plausible numbers.

#![cfg(target_os = "macos")]

use larql_compute::backend::matmul::MatMul;
use larql_models::quant::nvfp4;
use std::time::Instant;

/// Square so a chain can feed itself without reshaping.
const DIM: usize = 2048;
/// Long enough that queue depth has somewhere to go — a real Glimmer
/// token issues 209.
const CHAIN: usize = 32;

fn matrices(count: usize) -> Vec<nvfp4::Nvfp4Matrix> {
    (0..count)
        .map(|c| {
            // Near-identity scaling keeps the chain's magnitudes stable
            // over 32 hops; a random operator would under/overflow and
            // the comparison would be between two sets of infinities.
            let values: Vec<f32> = (0..DIM * DIM)
                .map(|i| {
                    let (r, col) = (i / DIM, i % DIM);
                    if r == col {
                        1.0
                    } else {
                        (((i + c) % 31) as f32 - 15.0) * 1e-4
                    }
                })
                .collect();
            nvfp4::quantize(&values, DIM, DIM).expect("quantise")
        })
        .collect()
}

#[test]
fn chained_encode_is_bit_identical_and_removes_the_starvation_tax() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let mats = matrices(CHAIN);
    let x0: Vec<f32> = (0..DIM).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();

    // ── Arm 1: today's shape — one command buffer per matvec, host
    // round trip between each.
    let run_per_call = || {
        let mut x = x0.clone();
        for m in &mats {
            x = gpu
                .nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, DIM, DIM)
                .expect("gemv");
        }
        x
    };

    // ── Arm 2: the lowered shape — one command buffer, device-resident
    // intermediates, a single wait.
    let run_chained = || {
        let x_buf = gpu.lowering_upload(&x0).expect("upload");
        let mut bufs = vec![x_buf];
        for _ in 0..CHAIN {
            bufs.push(gpu.lowering_scratch(DIM));
        }
        let weights: Vec<_> = mats
            .iter()
            .map(|m| {
                (
                    gpu.lowering_weight(&m.packed),
                    gpu.lowering_weight(&m.scales),
                    m.tensor_scale,
                )
            })
            .collect();

        let cmd = gpu.new_lowering_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for (i, (p, s, ts)) in weights.iter().enumerate() {
            let op = larql_compute_metal::lowering::MatvecOperands {
                packed: p,
                scales: s,
                x: &bufs[i],
                out: &bufs[i + 1],
                out_offset: 0,
                n: DIM,
                k: DIM,
            };
            gpu.encode_nvfp4_matvec(enc, &op, *ts);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let out = gpu.lowering_readback(&bufs[CHAIN], DIM).expect("readback");
        for b in bufs {
            gpu.recycle_lowering_scratch(b);
        }
        out
    };

    // Warm both paths: first touch pays weight upload and wiring.
    let _ = run_per_call();
    let _ = run_chained();

    let t = Instant::now();
    let a = run_per_call();
    let per_call_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let b = run_chained();
    let chained_ms = t.elapsed().as_secs_f64() * 1e3;

    assert_eq!(a.len(), DIM);
    assert!(
        a.iter().all(|v| v.is_finite()),
        "per-call chain produced non-finite values"
    );
    assert_eq!(
        a, b,
        "one command buffer must compute exactly what {CHAIN} command buffers did — \
         a difference means the chained dispatches are not correctly ordered"
    );

    eprintln!(
        "chain of {CHAIN} [{DIM}x{DIM}] nvfp4 matvecs:\n  \
         per-call ({CHAIN} CBs): {per_call_ms:.2} ms ({:.0} us/matvec)\n  \
         chained  (1 CB):        {chained_ms:.2} ms ({:.0} us/matvec)\n  \
         speedup: {:.2}x",
        per_call_ms * 1e3 / CHAIN as f64,
        chained_ms * 1e3 / CHAIN as f64,
        per_call_ms / chained_ms,
    );
    assert!(
        chained_ms < per_call_ms,
        "chaining must not be slower: {chained_ms:.2} vs {per_call_ms:.2} ms"
    );
}
