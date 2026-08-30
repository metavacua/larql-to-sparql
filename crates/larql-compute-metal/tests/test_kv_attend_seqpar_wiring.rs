//! KV-B1 wiring gate: `encode_kv_attend_seqpar` vs `encode_kv_attend`.
//!
//! `test_kernel_kv_attention_seqpar` dispatches the pipelines by hand, so it
//! proves the *kernel* reassociates correctly but says nothing about the
//! Rust that feeds it. Decode calls the `encode_*` functions, not the
//! pipelines, and those are where an argument can be marshalled to the wrong
//! buffer index or a pipeline selected on the wrong side of
//! `SHORT_ATTENTION_SPAN`.
//!
//! It also closes a hole in that gate's coverage. The kernel gate pins
//! `window_size = 0`, `has_sinks = 0`, `softcap = 0` — but gpt-oss-20b, the
//! model this work was done for, runs **attention sinks** on every layer and
//! a **128 sliding window** on half of them. Those arguments reach phases
//! 1-2, which the seqpar kernel copies from the production kernel, so a
//! divergence there is entirely possible and was previously untested.
//!
//! Reference is `encode_kv_attend` under the identical arguments, so this
//! measures only the seqpar substitution.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute_metal::ops::kv_cache::{encode_kv_attend, encode_kv_attend_seqpar, LayerKVCache};
use larql_compute_metal::MetalBackend;

const NUM_Q_HEADS: usize = 64;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const MAX_SEQ: usize = 4096;

/// gpt-oss-20b's sliding window. Half its layers use this; the other half
/// pass 0 (full attention).
const GPT_OSS_SLIDING_WINDOW: u32 = 128;

/// Reassociation tolerance, relative to the row's own scale. Calibrated by
/// `negative_control_*` below, which measures a real defect at ~1e-1.
const TOL: f32 = 1e-4;

fn synth(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Mixed signs across a 10^3 magnitude spread: a dropped or double-counted
/// slice then moves the result far outside float-reassociation noise, which
/// a uniform-positive fixture would hide.
fn adversarial_v(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((s >> 33) as f32) / (u32::MAX as f32);
            let mag = 10f32.powf(u * 3.0 - 1.5);
            if i % 2 == 0 {
                mag
            } else {
                -mag
            }
        })
        .collect()
}

struct Rig {
    metal: MetalBackend,
    cache: LayerKVCache,
    q: metal::Buffer,
    out: metal::Buffer,
    sink_vals: Vec<f32>,
}

fn rig(span: u32) -> Rig {
    let metal = MetalBackend::new().expect("Metal device");
    let bufs = metal.bufs();
    let mut cache = LayerKVCache::new(bufs, MAX_SEQ, NUM_KV_HEADS, HEAD_DIM);
    cache.current_len = (span - 1) as usize;
    cache.abs_position = (span - 1) as usize;

    let rows = MAX_SEQ * NUM_KV_HEADS * HEAD_DIM;
    let k = synth(rows, 0x51);
    let v = adversarial_v(rows, 0xADDE);
    unsafe {
        std::ptr::copy_nonoverlapping(k.as_ptr(), cache.k_cache.contents() as *mut f32, rows);
        std::ptr::copy_nonoverlapping(v.as_ptr(), cache.v_cache.contents() as *mut f32, rows);
    }

    let q = bufs.transient_from_f32(&synth(NUM_Q_HEADS * HEAD_DIM, 0x99));
    let out = bufs.output((NUM_Q_HEADS * HEAD_DIM * 4) as u64);
    // Sinks at a scale comparable to the logits, so they actually shift the
    // softmax rather than underflowing out of it.
    let sink_vals = synth(NUM_Q_HEADS, 0x7E5);
    Rig {
        metal,
        cache,
        q,
        out,
        sink_vals,
    }
}

/// Drive one encoder through the real wiring. `slices == 0` selects the
/// production encoder.
fn attend(rig: &Rig, slices: usize, window: u32, with_sinks: bool, softcap: f32) -> Vec<f32> {
    let m = &rig.metal;
    let a = &m.attention;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let sinks = with_sinks.then_some(rig.sink_vals.as_slice());

    let cmd = m.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    if slices == 0 {
        encode_kv_attend(
            enc,
            &rig.cache,
            &a.kv_attend_pipeline,
            Some(&a.kv_attend_long_pipeline),
            &rig.q,
            &rig.out,
            NUM_Q_HEADS,
            scale,
            window,
            sinks,
            softcap,
        );
    } else {
        encode_kv_attend_seqpar(
            enc,
            &rig.cache,
            &a.kv_attend_seqpar_pipeline,
            &a.kv_attend_seqpar_long_pipeline,
            &rig.q,
            &rig.out,
            NUM_Q_HEADS,
            scale,
            window,
            sinks,
            softcap,
            slices,
        );
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    larql_compute_metal::buffers::read_buffer_f32(&rig.out, NUM_Q_HEADS * HEAD_DIM)
}

/// Max deviation relative to the reference row's own scale, so a near-zero
/// element cannot manufacture a huge relative error.
fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(0.0f32, f32::max)
}

/// The full argument cross-product decode actually uses, at spans either
/// side of `SHORT_ATTENTION_SPAN` so both pipeline selections are covered.
#[test]
fn seqpar_wiring_matches_production_across_sinks_window_and_softcap() {
    if MetalBackend::new().is_none() {
        return;
    }
    for &span in &[64u32, 129, 512, 1024, 1025, 2048] {
        let r = rig(span);
        for &window in &[0u32, GPT_OSS_SLIDING_WINDOW] {
            for &with_sinks in &[false, true] {
                for &softcap in &[0.0f32, 30.0] {
                    let reference = attend(&r, 0, window, with_sinks, softcap);
                    for slices in [2usize, 4, 8, 16] {
                        let got = attend(&r, slices, window, with_sinks, softcap);
                        let rel = max_rel(&reference, &got);
                        assert!(
                            rel <= TOL,
                            "span {span} window {window} sinks {with_sinks} softcap {softcap} \
                             slices {slices}: max rel {rel:.3e} > {TOL:.0e}"
                        );
                    }
                }
            }
        }
    }
}

/// NEGATIVE CONTROL. The test above compares two encoders that share a
/// fixture, so it would also pass if `attend` ignored its arguments. Feed
/// the seqpar arm a *different* window and require the comparison to fail:
/// that proves the harness is sensitive to the arguments it claims to test.
#[test]
fn negative_control_wiring_gate_detects_a_wrong_window() {
    if MetalBackend::new().is_none() {
        return;
    }
    let span = 512u32;
    let r = rig(span);
    let reference = attend(&r, 0, 0, true, 0.0);
    let wrong_window = attend(&r, 4, GPT_OSS_SLIDING_WINDOW, true, 0.0);
    let rel = max_rel(&reference, &wrong_window);
    assert!(
        rel > TOL * 10.0,
        "attending over a 128-window instead of the full 512 span moved the \
         output by only {rel:.3e} — the gate cannot see the window argument \
         at all, so its window coverage above is vacuous"
    );
}

/// The reduction is order-fixed so repeated dispatches agree bitwise.
/// Non-determinism here is invisible to sampling but desyncs the Shannon
/// arithmetic coder, which is why this is asserted on bits.
#[test]
fn seqpar_wiring_is_deterministic_across_repeats() {
    if MetalBackend::new().is_none() {
        return;
    }
    for &span in &[512u32, 2048] {
        let r = rig(span);
        let first = attend(&r, 8, GPT_OSS_SLIDING_WINDOW, true, 0.0);
        for _ in 0..4 {
            let again = attend(&r, 8, GPT_OSS_SLIDING_WINDOW, true, 0.0);
            assert_eq!(
                first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "span {span}: repeated dispatch differed through the wiring"
            );
        }
    }
}
