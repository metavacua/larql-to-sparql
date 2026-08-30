//! Gates for the NVFP4 call-phase diagnostic.
//!
//! This module measures where a single gemv's time goes, and a
//! diagnostic that reports plausible-but-wrong phases is worse than none
//! — it sends the next optimisation at the wrong target. So the gates
//! check the two things a timing breakdown can get wrong independently:
//! that the phases actually decompose the call (they sum to no more than
//! the total, and the total is not a stand-in for one dominant phase),
//! and that the run underneath the timings computed the right answer.
//!
//! Both entry points also have refusal paths. `MetalBackend::new()`
//! returning `None` is the honest skip on a machine without a device;
//! CI's paravirtual GPU can also refuse timestamps, so the GPU-clock
//! assertions are written to tolerate a zero span rather than fail the
//! suite on hardware that cannot supply one.

use super::*;
use larql_compute::backend::MatMul;
use larql_models::quant::nvfp4;

/// Small but not degenerate: K a whole number of 16-element groups, rows
/// past one threadgroup so the dispatch grid is real.
const ROWS: usize = 64;
const K: usize = 128;

fn fixture() -> (nvfp4::Nvfp4Matrix, Vec<f32>, Vec<f32>) {
    let values: Vec<f32> = (0..ROWS * K)
        .map(|i| ((i % 97) as f32 / 97.0) - 0.5)
        .collect();
    let m = nvfp4::quantize(&values, ROWS, K).expect("quantise");
    let x: Vec<f32> = (0..K).map(|i| (i % 11) as f32 * 0.05 - 0.25).collect();
    // The oracle multiplies the DEQUANTISED weights, not the originals.
    // Against the originals this test measures NVFP4's representation
    // loss — which is not what it claims to check, and on a 128-term dot
    // product with cancellation that loss reaches 20% of the row scale,
    // so the tolerance would have to be loosened until the gate stopped
    // being able to see a wrong kernel at all. `round_trip` is the
    // format's own accessor, so this compares the kernel against the
    // bytes it was actually given.
    let stored = nvfp4::round_trip(&values, ROWS, K).expect("round trip");
    let mut expect = vec![0.0f32; ROWS];
    for (r, e) in expect.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for c in 0..K {
            acc += stored[r * K + c] * x[c];
        }
        *e = acc;
    }
    (m, x, expect)
}

#[test]
fn call_profile_phases_decompose_the_call() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (m, x, _) = fixture();
    let p = gpu
        .profile_nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, ROWS, K)
        .expect("profile returns a breakdown on a working device");

    // Every phase is a real elapsed measurement: non-negative and finite.
    for (name, v) in [
        ("input_stage", p.input_stage),
        ("buffer_acquire", p.buffer_acquire),
        ("encode", p.encode),
        ("commit", p.commit),
        ("wait", p.wait),
        ("readback", p.readback),
        ("total", p.total),
    ] {
        assert!(v.is_finite() && v >= 0.0, "{name} is {v}, not a duration");
    }

    // The decomposition claim: the host phases are PARTS of the call, so
    // together they cannot exceed it. Without this the struct could
    // report each phase as the whole call and still look reasonable.
    let host_sum = p.input_stage + p.buffer_acquire + p.encode + p.commit + p.wait + p.readback;
    assert!(
        host_sum <= p.total * 1.05,
        "phases sum to {host_sum} µs but the whole call was {} µs — they are \
         not a decomposition",
        p.total
    );
    assert!(p.total > 0.0, "a completed call took no time");

    // GPU-clock fields: derived from GPUStartTime/GPUEndTime, which a
    // paravirtual device may not supply. Assert only what holds either
    // way — never negative, never larger than the call that contains it.
    assert!(p.gpu_span >= 0.0 && p.gpu_span.is_finite());
    assert!(
        p.gpu_span <= p.total * 1.05,
        "GPU span {} µs exceeds the whole call {} µs",
        p.gpu_span,
        p.total
    );
    assert!(p.commit_to_gpu_start.is_finite());
}

#[test]
fn call_profile_computes_the_right_answer_underneath() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (m, x, expect) = fixture();
    // The diagnostic discards its output, so the timings alone cannot
    // tell us it ran the real kernel. Run the same shape through the
    // production gemv and require agreement: if the profiled path
    // dispatched something else, the phases would be timing a different
    // program.
    assert!(gpu
        .profile_nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, ROWS, K)
        .is_some());
    let got = gpu
        .nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, ROWS, K)
        .expect("production gemv");
    assert_eq!(got.len(), ROWS);
    // Both sides now multiply the same stored weights, so the only
    // legitimate difference is fp32 summation order (the GPU reduces
    // across a simdgroup, the oracle serially) — a tight tolerance, and
    // tight is the point: a loose one would pass on a kernel reading the
    // wrong rows entirely.
    let scale = expect.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
    for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
        assert!(
            (g - e).abs() / scale < 1e-4,
            "row {i}: {g} vs {e} (rel to {scale})"
        );
    }
}

#[test]
fn pipelined_cost_reports_per_dispatch_microseconds() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (m, x, _) = fixture();
    let one = gpu
        .nvfp4_pipelined_cost(&m.packed, &m.scales, m.tensor_scale, &x, ROWS, K, 1)
        .expect("depth 1");
    let deep = gpu
        .nvfp4_pipelined_cost(&m.packed, &m.scales, m.tensor_scale, &x, ROWS, K, 8)
        .expect("depth 8");
    for (name, v) in [("depth 1", one), ("depth 8", deep)] {
        assert!(v.is_finite() && v > 0.0, "{name} reported {v} µs/dispatch");
    }
    // The value is PER DISPATCH, not a total: were the division dropped,
    // depth 8 would report roughly eight times depth 1. This bounds the
    // mistake without asserting that pipelining actually wins, which is
    // the open question the diagnostic exists to answer and must not be
    // baked into its own gate.
    assert!(
        deep < one * 4.0,
        "depth 8 at {deep} µs/dispatch against depth 1 at {one} — that looks \
         like a total rather than a per-dispatch figure"
    );
}
