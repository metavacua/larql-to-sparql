//! KV-B1 gate for the **fused** append+attend kernel.
//!
//! `kv_attention_seqpar` fixed the serial weighted-V loop on the unfused
//! attend path, which decode only reaches when the span exceeds
//! `SHORT_ATTENTION_SPAN`. Below that, `kv_append_attend_fused` is the
//! default (`LARQL_FUSED_KV_APPEND_ATTEND`) and carried the identical
//! defect — so this kernel, not the unfused one, serves the common case:
//! every sliding-window layer at every depth, and every full-attention
//! layer up to depth 1024.
//!
//! `kv_append_attend_fused_seqpar` must therefore match it. The comparison
//! is tolerance-based, not bitwise, because splitting the accumulation
//! across slices reassociates the sum.
//!
//! Coverage is the argument cross-product decode really uses: gpt-oss-20b
//! runs **attention sinks** on every layer and a **128 sliding window** on
//! half of them, and both feed phases 1-2, which this kernel copies from
//! the baseline.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute_metal::ops::kv_cache::LayerKVCache;
use larql_compute_metal::MetalBackend;

const NUM_Q_HEADS: usize = 64;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const MAX_SEQ: usize = 2048;

/// gpt-oss-20b's sliding window.
const GPT_OSS_SLIDING_WINDOW: u32 = 128;

/// Reassociation tolerance relative to the row scale; `negative_control_*`
/// measures a real defect far above it.
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

/// Mixed signs over a 10^3 magnitude spread, so a mis-mapped slice moves
/// the result far outside reassociation noise.
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
    new_k: metal::Buffer,
    new_v: metal::Buffer,
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
    // Phase 0 writes these into the cache at pos = T-1. Both arms write the
    // same values, so running them against one cache is idempotent.
    let new_k = bufs.transient_from_f32(&synth(NUM_KV_HEADS * HEAD_DIM, 0x2B));
    let new_v = bufs.transient_from_f32(&adversarial_v(NUM_KV_HEADS * HEAD_DIM, 0x3C));
    let sink_vals = synth(NUM_Q_HEADS, 0x7E5);
    Rig {
        metal,
        cache,
        q,
        out,
        new_k,
        new_v,
        sink_vals,
    }
}

/// `slices == 0` dispatches the baseline fused kernel at its production
/// width (`head_dim` threads); otherwise the seqpar variant at
/// `slices * head_dim`.
fn attend(rig: &Rig, slices: usize, window: u32, with_sinks: bool, softcap: f32) -> Vec<f32> {
    let m = &rig.metal;
    let a = &m.attention;
    let pipeline = if slices == 0 {
        &a.kv_append_attend_fused_pipeline
    } else {
        &a.kv_append_attend_fused_seqpar_pipeline
    };
    let threads = if slices == 0 {
        HEAD_DIM as u64
    } else {
        (slices * HEAD_DIM) as u64
    };

    let t_val = (rig.cache.current_len + 1) as u32;
    let hd = HEAD_DIM as u32;
    let nq = NUM_Q_HEADS as u32;
    let nkv = NUM_KV_HEADS as u32;
    let has_sinks = u32::from(with_sinks);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let sink_buf = m.bufs().transient_from_f32(&rig.sink_vals);

    let cmd = m.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&rig.q), 0);
    enc.set_buffer(1, Some(&rig.cache.k_cache), 0);
    enc.set_buffer(2, Some(&rig.cache.v_cache), 0);
    enc.set_buffer(3, Some(&rig.out), 0);
    enc.set_bytes(4, 4, &t_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &hd as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &nq as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &nkv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &window as *const u32 as *const std::ffi::c_void);
    enc.set_buffer(10, Some(&rig.new_k), 0);
    enc.set_buffer(11, Some(&rig.new_v), 0);
    enc.set_buffer(12, Some(&sink_buf), 0);
    enc.set_bytes(13, 4, &has_sinks as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(14, 4, &softcap as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(NUM_Q_HEADS as u64, 1, 1),
        metal::MTLSize::new(threads, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    larql_compute_metal::buffers::read_buffer_f32(&rig.out, NUM_Q_HEADS * HEAD_DIM)
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(0.0f32, f32::max)
}

/// Spans stop at `SHORT_ATTENTION_SPAN` because `tg_scores[1024]` is what
/// bounds this kernel — past it decode takes the unfused path, which its
/// own gate covers.
#[test]
fn fused_seqpar_matches_baseline_across_sinks_window_and_softcap() {
    if MetalBackend::new().is_none() {
        return;
    }
    for &span in &[
        1u32, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 512, 1023, 1024,
    ] {
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
                            "span {span} window {window} sinks {with_sinks} \
                             softcap {softcap} slices {slices}: max rel {rel:.3e} > {TOL:.0e}"
                        );
                    }
                }
            }
        }
    }
}

/// Phase 0 must still land: the seqpar variant widens the threadgroup, and
/// the append loop is `for (d = tid; d < head_dim; d += tg_sz)` — if that
/// were mis-strided the newest position would be stale, which the parity
/// test above cannot see because both arms would write it identically.
/// Here the cache row at `pos` is poisoned first, so only a correct append
/// can reproduce the expected output.
#[test]
fn fused_seqpar_phase0_appends_the_new_row() {
    if MetalBackend::new().is_none() {
        return;
    }
    let span = 256u32;
    let r = rig(span);
    let reference = attend(&r, 0, 0, true, 0.0);

    let pos = (span - 1) as usize;
    let row = pos * NUM_KV_HEADS * HEAD_DIM;
    unsafe {
        let kp = r.cache.k_cache.contents() as *mut f32;
        let vp = r.cache.v_cache.contents() as *mut f32;
        for i in 0..NUM_KV_HEADS * HEAD_DIM {
            *kp.add(row + i) = 12345.0;
            *vp.add(row + i) = -54321.0;
        }
    }

    // Poison must be overwritten by phase 0, restoring parity exactly.
    let after = attend(&r, 8, 0, true, 0.0);
    let rel = max_rel(&reference, &after);
    assert!(
        rel <= TOL,
        "poisoned position {pos} survived: max rel {rel:.3e} — the seqpar \
         variant's phase-0 append does not cover the full row at this \
         threadgroup width"
    );
}

/// NEGATIVE CONTROL. Perturb one attended position's V row and require the
/// comparison to fail well outside `TOL`, proving the gate could see a
/// broken reduction at all.
#[test]
fn negative_control_a_single_perturbed_position_breaks_parity() {
    if MetalBackend::new().is_none() {
        return;
    }
    let span = 512u32;
    let r = rig(span);
    let reference = attend(&r, 0, 0, true, 0.0);

    let victim = (span as usize / 2) * NUM_KV_HEADS * HEAD_DIM;
    let saved: Vec<f32> = unsafe {
        let p = r.cache.v_cache.contents() as *mut f32;
        let s = std::slice::from_raw_parts(p.add(victim), HEAD_DIM).to_vec();
        for d in 0..HEAD_DIM {
            *p.add(victim + d) += 25.0;
        }
        s
    };
    let perturbed = attend(&r, 4, 0, true, 0.0);
    unsafe {
        let p = r.cache.v_cache.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(saved.as_ptr(), p.add(victim), HEAD_DIM);
    }

    let rel = max_rel(&reference, &perturbed);
    assert!(
        rel > TOL * 10.0,
        "perturbing one attended position moved the output by only {rel:.3e}, \
         inside the parity tolerance — the gate cannot tell a correct \
         reduction from a broken one"
    );

    let restored = attend(&r, 4, 0, true, 0.0);
    assert!(
        max_rel(&reference, &restored) <= TOL,
        "restore failed; the fixture is not stable"
    );
}

/// Fixed-order slice reduction ⇒ bitwise-stable across repeats.
#[test]
fn fused_seqpar_is_deterministic_across_repeats() {
    if MetalBackend::new().is_none() {
        return;
    }
    for &span in &[128u32, 1024] {
        let r = rig(span);
        let first = attend(&r, 8, GPT_OSS_SLIDING_WINDOW, true, 0.0);
        for _ in 0..4 {
            let again = attend(&r, 8, GPT_OSS_SLIDING_WINDOW, true, 0.0);
            assert_eq!(
                first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "span {span}: repeated dispatch differed"
            );
        }
    }
}
