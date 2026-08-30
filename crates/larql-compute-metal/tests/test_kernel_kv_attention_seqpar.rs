//! KV-B1 gate: sequence-parallel phase 3 against the production kernel.
//!
//! `kv_attention_seqpar` splits the weighted-V accumulation across
//! `tg_sz / head_dim` sequence slices and reduces them in fixed order. That
//! reassociates the sum, so the candidate is **not** bitwise equal to
//! `kv_attention` — the gate uses a calibrated tolerance instead, the same
//! call rung E made about the MoE combine.
//!
//! The tolerance is only meaningful if the fixture can actually expose a
//! bad reduction, so the V values are adversarial: mixed signs and mixed
//! magnitudes, so that dropping or double-counting any slice changes the
//! result by far more than reassociation does. `negative_control_*` proves
//! that by construction — it perturbs exactly one slice's contribution and
//! requires the comparison to fail.
//!
//! Spans are chosen around every boundary the kernels have: the simdgroup
//! width (31/32/33), `head_dim` (63/64/65), the sliding-window default
//! (127/128/129), and `SHORT_ATTENTION_SPAN` (1024/1025), plus 1 and 2048.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute_metal::ops::kv_cache::LayerKVCache;
use larql_compute_metal::MetalBackend;

const NUM_Q_HEADS: usize = 64;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;

/// Every boundary in the two kernels, plus the degenerate span.
const SPANS: &[u32] = &[
    1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 1023, 1024, 1025, 1536,
    2048,
];

/// Adversarial by construction: alternating signs and a 10^3 magnitude
/// spread, so a slice that is dropped, doubled, or offset moves the result
/// far outside float reassociation noise. A uniform-positive fixture would
/// hide exactly those defects.
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

fn synth(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

struct Rig {
    metal: MetalBackend,
    cache: LayerKVCache,
    q: metal::Buffer,
    out: metal::Buffer,
    sinks: metal::Buffer,
}

fn rig(span: u32, v_scale_seed: u64) -> Rig {
    let metal = MetalBackend::new().expect("Metal device");
    let bufs = metal.bufs();
    let mut cache = LayerKVCache::new(bufs, 4096, NUM_KV_HEADS, HEAD_DIM);
    cache.current_len = (span - 1) as usize;
    cache.abs_position = (span - 1) as usize;

    let rows = 4096 * NUM_KV_HEADS * HEAD_DIM;
    let k = synth(rows, 0x51);
    let v = adversarial_v(rows, v_scale_seed);
    unsafe {
        std::ptr::copy_nonoverlapping(k.as_ptr(), cache.k_cache.contents() as *mut f32, rows);
        std::ptr::copy_nonoverlapping(v.as_ptr(), cache.v_cache.contents() as *mut f32, rows);
    }

    let q_data = synth(NUM_Q_HEADS * HEAD_DIM, 0x99);
    let q = bufs.transient_from_f32(&q_data);
    let out = bufs.output((NUM_Q_HEADS * HEAD_DIM * 4) as u64);
    let sinks = bufs.output((NUM_Q_HEADS * 4) as u64);
    Rig {
        metal,
        cache,
        q,
        out,
        sinks,
    }
}

/// Dispatch one attention kernel. `slices == 0` selects the production
/// kernel at its production geometry (`head_dim` threads); otherwise the
/// sequence-parallel kernel with `slices * head_dim` threads.
///
/// `t_shift` and `drop_slice` exist ONLY for the negative control.
fn run(rig: &Rig, span: u32, slices: usize) -> Vec<f32> {
    let m = &rig.metal;
    let pipeline = match (slices, span > 1024) {
        (0, false) => &m.attention.kv_attend_pipeline,
        (0, true) => &m.attention.kv_attend_long_pipeline,
        (_, false) => &m.attention.kv_attend_seqpar_pipeline,
        (_, true) => &m.attention.kv_attend_seqpar_long_pipeline,
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
    let win = 0u32;
    let has_sinks = 0u32;
    let softcap = 0.0f32;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

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
    enc.set_bytes(9, 4, &win as *const u32 as *const std::ffi::c_void);
    enc.set_buffer(10, Some(&rig.sinks), 0);
    enc.set_bytes(11, 4, &has_sinks as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(12, 4, &softcap as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(NUM_Q_HEADS as u64, 1, 1),
        metal::MTLSize::new(threads, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    larql_compute_metal::buffers::read_buffer_f32(&rig.out, NUM_Q_HEADS * HEAD_DIM)
}

/// Max relative deviation, measured against the row's own scale so a
/// near-zero output element cannot manufacture a huge relative error.
fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(0.0f32, f32::max)
}

/// Reassociation tolerance. The weighted-V sum is over `span` terms of
/// mixed sign at a 10^3 magnitude spread, so f32 cancellation is real;
/// 1e-4 relative to the row scale is comfortably above that and far below
/// what any slice-mapping defect produces (the negative control measures
/// the latter at ~1e-1).
const TOL: f32 = 1e-4;

#[test]
fn seqpar_matches_production_across_every_boundary_span() {
    if MetalBackend::new().is_none() {
        return;
    }
    for &span in SPANS {
        let r = rig(span, 0xADDE);
        let reference = run(&r, span, 0);
        for slices in [2usize, 4, 8, 16] {
            let candidate = run(&r, span, slices);
            let rel = max_rel(&reference, &candidate);
            assert!(
                rel <= TOL,
                "span {span}, {slices} slices: max rel {rel:.3e} > {TOL:.0e}\n  \
                 ref[..4]={:?}\n  got[..4]={:?}",
                &reference[..4],
                &candidate[..4]
            );
        }
    }
}

/// A span shorter than the slice count leaves trailing slices with nothing
/// to accumulate. Their partials must be the zero they were initialised
/// with, not stale threadgroup memory.
#[test]
fn seqpar_handles_spans_shorter_than_the_slice_count() {
    if MetalBackend::new().is_none() {
        return;
    }
    for &span in &[1u32, 2, 3, 5, 7] {
        let r = rig(span, 0xBEEF);
        let reference = run(&r, span, 0);
        for slices in [2usize, 4, 8, 16] {
            let candidate = run(&r, span, slices);
            let rel = max_rel(&reference, &candidate);
            assert!(
                rel <= TOL,
                "span {span} with {slices} slices (more slices than positions): \
                 max rel {rel:.3e}"
            );
        }
    }
}

/// Repeated dispatches must agree bitwise with each other: the fixed-order
/// slice reduction is what keeps this deterministic, and non-determinism
/// here is invisible to sampling but desyncs the Shannon arithmetic coder.
#[test]
fn seqpar_is_deterministic_across_repeats() {
    if MetalBackend::new().is_none() {
        return;
    }
    for &span in &[128u32, 1024, 2048] {
        let r = rig(span, 0x1234);
        let first = run(&r, span, 4);
        for _ in 0..6 {
            let again = run(&r, span, 4);
            assert_eq!(
                first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "span {span}: repeated dispatch differed — the slice reduction \
                 is not order-stable"
            );
        }
    }
}

/// NEGATIVE CONTROL. Calibrates the tolerance: perturb ONE position's V row
/// — the smallest defect a wrong slice mapping could produce — and require
/// the comparison to fail well outside `TOL`. Without this the parity test
/// above could pass on a kernel that ignored V entirely.
#[test]
fn negative_control_a_single_perturbed_position_breaks_parity() {
    if MetalBackend::new().is_none() {
        return;
    }
    let span = 512u32;
    let r = rig(span, 0xC0DE);
    let reference = run(&r, span, 0);

    // Perturb position `span/2` for kv_head 0 — inside the attended range,
    // and reachable by exactly one slice under any slice count.
    let victim = (span as usize / 2) * NUM_KV_HEADS * HEAD_DIM;
    let saved: Vec<f32> = unsafe {
        let p = r.cache.v_cache.contents() as *mut f32;
        let s = std::slice::from_raw_parts(p.add(victim), HEAD_DIM).to_vec();
        for d in 0..HEAD_DIM {
            *p.add(victim + d) += 25.0;
        }
        s
    };

    let perturbed = run(&r, span, 4);
    let rel = max_rel(&reference, &perturbed);

    // Restore before asserting so a failure cannot poison later tests.
    unsafe {
        let p = r.cache.v_cache.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(saved.as_ptr(), p.add(victim), HEAD_DIM);
    }

    assert!(
        rel > TOL * 10.0,
        "perturbing one attended position moved the output by only {rel:.3e}, \
         which is within the parity tolerance — the gate cannot distinguish a \
         correct reduction from a broken one"
    );

    // Restore-green: the same comparison must pass again once V is back.
    let restored = run(&r, span, 4);
    assert!(
        max_rel(&reference, &restored) <= TOL,
        "restore failed; the fixture is not stable"
    );
}
