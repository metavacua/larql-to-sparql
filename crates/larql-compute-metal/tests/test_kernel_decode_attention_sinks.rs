//! Numerical parity for attention sinks in the two **decode** Metal
//! kernels, against a CPU reference.
//!
//! Prefill (`fused_attention`) is covered in
//! `test_kernel_fused_attention.rs`. These are its decode siblings:
//!
//! - `kv_append_attend_fused` — the default decode kernel: appends the
//!   new K/V row at `pos = T-1`, then attends over the cache.
//! - `attn_fused` — the fused QK-norm + RoPE + append + attend kernel.
//!
//! Both must reproduce the reference sink formulation: concatenate the
//! per-head sink to the logits, softmax jointly, drop the sink column —
//! so the emitted weights sum to less than one. See
//! `docs/k3-funnel.md` §4.6.

#![cfg(target_os = "macos")]

extern crate blas_src;

use std::ffi::c_void;

/// Real sink magnitudes measured on `openai/gpt-oss-20b`: mean +2.45,
/// max +8.19. Using the observed range keeps the test honest about the
/// regime the kernel actually runs in.
const SINKS: [f32; 2] = [2.45, 8.19];

/// Agreement bound. The kernels reduce in f32 while the reference
/// accumulates in f64, so exact equality is not expected.
const MAX_DIFF: f32 = 1e-4;

fn device_and_lib() -> Option<(metal::Device, metal::Library)> {
    let device = metal::Device::system_default()?;
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .expect("shader library compiles");
    Some((device, lib))
}

fn shared_buffer(device: &metal::Device, data: &[f32]) -> metal::Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const c_void,
        std::mem::size_of_val(data) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn read_back(buf: &metal::Buffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec() }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Reference decode attention over a cache of `t_len` positions.
///
/// `sink` is `None` for the ordinary softmax. Accumulates in f64 so the
/// comparison tests the kernel, not the reference's own rounding.
#[allow(clippy::too_many_arguments)]
fn cpu_decode_reference(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    t_len: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
    scale: f32,
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_q * head_dim];
    for head in 0..num_q {
        let kv_head = head / (num_q / num_kv);
        let mut logits = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[head * head_dim + d]
                    * k_cache[t * num_kv * head_dim + kv_head * head_dim + d];
            }
            logits.push((dot * scale) as f64);
        }
        // Joint max over logits and the sink, exactly as the reference
        // does after concatenating.
        let mut m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if let Some(s) = sinks {
            m = m.max(s[head] as f64);
        }
        let exps: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
        let mut denom: f64 = exps.iter().sum();
        if let Some(s) = sinks {
            // Denominator only — the sink has no output slot.
            denom += (s[head] as f64 - m).exp();
        }
        for d in 0..head_dim {
            let mut acc = 0.0f64;
            for (t, e) in exps.iter().enumerate() {
                acc += (e / denom) * v_cache[t * num_kv * head_dim + kv_head * head_dim + d] as f64;
            }
            out[head * head_dim + d] = acc as f32;
        }
    }
    out
}

// ── kv_append_attend_fused (the default decode kernel) ───────────────────

/// Dispatch `kv_append_attend_fused` for one decode step and return the
/// attention output. `t_len` is the cache length *after* the append.
#[allow(clippy::too_many_arguments)]
fn run_kv_append_attend_fused(
    device: &metal::Device,
    lib: &metal::Library,
    q: &[f32],
    cache_k: &[f32],
    cache_v: &[f32],
    new_k: &[f32],
    new_v: &[f32],
    t_len: u32,
    num_q: u32,
    num_kv: u32,
    head_dim: u32,
    scale: f32,
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function("kv_append_attend_fused", None).unwrap(),
        )
        .unwrap();
    let queue = device.new_command_queue();

    let out_len = (num_q * head_dim) as usize;
    let buf_q = shared_buffer(device, q);
    let buf_k = shared_buffer(device, cache_k);
    let buf_v = shared_buffer(device, cache_v);
    let buf_nk = shared_buffer(device, new_k);
    let buf_nv = shared_buffer(device, new_v);
    let buf_out = device.new_buffer(
        (out_len * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    let u = |v: &u32| v as *const u32 as *const c_void;
    let f = |v: &f32| v as *const f32 as *const c_void;
    enc.set_bytes(4, 4, u(&t_len));
    enc.set_bytes(5, 4, u(&head_dim));
    enc.set_bytes(6, 4, u(&num_q));
    enc.set_bytes(7, 4, u(&num_kv));
    enc.set_bytes(8, 4, f(&scale));
    let window_size = 0u32; // 0 = no sliding window
    enc.set_bytes(9, 4, u(&window_size));
    enc.set_buffer(10, Some(&buf_nk), 0);
    enc.set_buffer(11, Some(&buf_nv), 0);
    let placeholder = [0.0f32];
    let (sink_vals, has_sinks): (&[f32], u32) = match sinks {
        Some(s) => (s, 1),
        None => (&placeholder, 0),
    };
    enc.set_bytes(
        12,
        std::mem::size_of_val(sink_vals) as u64,
        sink_vals.as_ptr() as *const c_void,
    );
    enc.set_bytes(13, 4, u(&has_sinks));
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, 1, 1),
        metal::MTLSize::new(head_dim.min(256) as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    read_back(&buf_out, out_len)
}

/// Build a decode fixture: a cache with `t_len` slots where the last row
/// is zeroed (the kernel writes it from `new_k`/`new_v`), plus the
/// post-append cache the CPU reference should see.
#[allow(clippy::type_complexity)]
fn decode_fixture(
    t_len: usize,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    let q: Vec<f32> = (0..num_q * head_dim)
        .map(|i| (i as f32 * 0.37 + 1.0).sin() * 0.5)
        .collect();
    let cache_len = t_len * num_kv * head_dim;
    let mut cache_k: Vec<f32> = (0..cache_len)
        .map(|i| (i as f32 * 0.29 + 2.0).cos() * 0.5)
        .collect();
    let mut cache_v: Vec<f32> = (0..cache_len)
        .map(|i| (i as f32 * 0.19 + 3.0).sin() * 0.5)
        .collect();
    let new_k: Vec<f32> = (0..num_kv * head_dim)
        .map(|i| (i as f32 * 0.41 + 5.0).cos() * 0.5)
        .collect();
    let new_v: Vec<f32> = (0..num_kv * head_dim)
        .map(|i| (i as f32 * 0.23 + 7.0).sin() * 0.5)
        .collect();

    // The kernel appends at pos = t_len - 1, so zero that row in the
    // buffer it receives and splice the new row into the reference.
    let last_row = (t_len - 1) * num_kv * head_dim;
    for d in 0..num_kv * head_dim {
        cache_k[last_row + d] = 0.0;
        cache_v[last_row + d] = 0.0;
    }
    let mut ref_k = cache_k.clone();
    let mut ref_v = cache_v.clone();
    ref_k[last_row..last_row + num_kv * head_dim].copy_from_slice(&new_k);
    ref_v[last_row..last_row + num_kv * head_dim].copy_from_slice(&new_v);

    (q, cache_k, cache_v, new_k, new_v, ref_k, ref_v)
}

#[test]
fn kv_append_attend_fused_with_sinks_matches_cpu_reference() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5usize, 2usize, 2usize, 8usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, ck, cv, nk, nv, ref_k, ref_v) = decode_fixture(t_len, num_q, num_kv, head_dim);

    let gpu = run_kv_append_attend_fused(
        &device,
        &lib,
        &q,
        &ck,
        &cv,
        &nk,
        &nv,
        t_len as u32,
        num_q as u32,
        num_kv as u32,
        head_dim as u32,
        scale,
        Some(&SINKS),
    );
    let cpu = cpu_decode_reference(
        &q,
        &ref_k,
        &ref_v,
        t_len,
        num_q,
        num_kv,
        head_dim,
        scale,
        Some(&SINKS),
    );

    let diff = max_diff(&cpu, &gpu);
    assert!(
        diff < MAX_DIFF,
        "kv_append_attend_fused with sinks: max diff {diff}\nCPU: {:?}\nGPU: {:?}",
        &cpu[..8],
        &gpu[..8]
    );
}

#[test]
fn kv_append_attend_fused_without_sinks_is_unchanged() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5usize, 2usize, 2usize, 8usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, ck, cv, nk, nv, ref_k, ref_v) = decode_fixture(t_len, num_q, num_kv, head_dim);

    let gpu = run_kv_append_attend_fused(
        &device,
        &lib,
        &q,
        &ck,
        &cv,
        &nk,
        &nv,
        t_len as u32,
        num_q as u32,
        num_kv as u32,
        head_dim as u32,
        scale,
        None,
    );
    let cpu = cpu_decode_reference(
        &q, &ref_k, &ref_v, t_len, num_q, num_kv, head_dim, scale, None,
    );

    let diff = max_diff(&cpu, &gpu);
    assert!(
        diff < MAX_DIFF,
        "kv_append_attend_fused without sinks regressed: max diff {diff}"
    );
}

#[test]
fn kv_append_attend_fused_sinks_divert_attention_mass() {
    // A sink that dominates the logits must shrink the output toward
    // zero. Guards against the kernel accepting the binding but never
    // applying it — which the parity test alone could not distinguish
    // if both sides were wrong in the same way.
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5usize, 2usize, 2usize, 8usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, ck, cv, nk, nv, _, _) = decode_fixture(t_len, num_q, num_kv, head_dim);

    let run = |sinks: Option<&[f32]>| {
        run_kv_append_attend_fused(
            &device,
            &lib,
            &q,
            &ck,
            &cv,
            &nk,
            &nv,
            t_len as u32,
            num_q as u32,
            num_kv as u32,
            head_dim as u32,
            scale,
            sinks,
        )
    };
    let plain = run(None);
    let dominated = run(Some(&[40.0, 40.0]));

    let plain_mag: f32 = plain.iter().map(|x| x.abs()).sum();
    let dom_mag: f32 = dominated.iter().map(|x| x.abs()).sum();
    assert!(
        plain_mag > 1e-3,
        "degenerate fixture — no-sink output is already ~zero"
    );
    assert!(
        dom_mag < plain_mag * 0.01,
        "a dominant sink must absorb nearly all mass: {dom_mag} vs {plain_mag}"
    );
}

#[test]
fn kv_append_attend_fused_applies_each_heads_own_sink() {
    // Head 0 gets a negligible sink, head 1 a dominant one. Only head 1
    // should collapse — fails if the kernel indexes `sinks` wrongly or
    // reads a single value for every head.
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5usize, 2usize, 2usize, 8usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, ck, cv, nk, nv, _, _) = decode_fixture(t_len, num_q, num_kv, head_dim);

    let out = run_kv_append_attend_fused(
        &device,
        &lib,
        &q,
        &ck,
        &cv,
        &nk,
        &nv,
        t_len as u32,
        num_q as u32,
        num_kv as u32,
        head_dim as u32,
        scale,
        Some(&[-40.0, 40.0]),
    );
    let head0: f32 = out[..head_dim].iter().map(|x| x.abs()).sum();
    let head1: f32 = out[head_dim..].iter().map(|x| x.abs()).sum();
    assert!(head0 > 1e-3, "head 0 should be unaffected, got {head0}");
    assert!(
        head1 < 1e-3,
        "head 1's dominant sink should zero it, got {head1}"
    );
}

// ── attn_fused (QK-norm + RoPE + append + attend) ────────────────────────
//
// This kernel fuses RMS-norm and RoPE, so a value-level CPU reference
// would have to re-derive both — duplicating logic the norm and RoPE
// kernels already test, and testing the duplicate as much as the kernel.
//
// Instead these use the sink's *exact analytic signature*. Adding a sink
// changes only the denominator, so every weight in a head is multiplied
// by the same constant:
//
//     w_i(sink) = e_i / (S + e_s) = w_i(no sink) · S/(S + e_s)
//
// Since the output is a weighted sum of V, the whole per-head output
// vector scales by that one factor λ ∈ (0, 1). A uniform per-head
// rescaling is a fingerprint no other bug produces: applying the sink to
// the logits, per-element, or to the wrong head all break it.

/// Dispatch `attn_fused` for one decode step. Returns the attention output.
#[allow(clippy::too_many_arguments)]
fn run_attn_fused(
    device: &metal::Device,
    lib: &metal::Library,
    t_len: u32,
    num_q: u32,
    num_kv: u32,
    head_dim: u32,
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("attn_fused", None).unwrap())
        .unwrap();
    let queue = device.new_command_queue();

    let nq = num_q as usize;
    let nkv = num_kv as usize;
    let hd = head_dim as usize;
    let tl = t_len as usize;

    let q_in: Vec<f32> = (0..nq * hd)
        .map(|i| (i as f32 * 0.37 + 1.0).sin() * 0.5)
        .collect();
    let k_in: Vec<f32> = (0..nkv * hd)
        .map(|i| (i as f32 * 0.41 + 5.0).cos() * 0.5)
        .collect();
    let v_in: Vec<f32> = (0..nkv * hd)
        .map(|i| (i as f32 * 0.23 + 7.0).sin() * 0.5)
        .collect();
    let cache: Vec<f32> = (0..tl * nkv * hd)
        .map(|i| (i as f32 * 0.29 + 2.0).cos() * 0.5)
        .collect();
    let ones = vec![1.0f32; hd];

    let buf_q = shared_buffer(device, &q_in);
    let buf_k = shared_buffer(device, &k_in);
    let buf_v = shared_buffer(device, &v_in);
    let buf_kc = shared_buffer(device, &cache);
    let buf_vc = shared_buffer(device, &cache);
    let buf_qw = shared_buffer(device, &ones);
    let buf_kw = shared_buffer(device, &ones);
    let out_len = nq * hd;
    let buf_out = device.new_buffer(
        (out_len * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_kc), 0);
    enc.set_buffer(4, Some(&buf_vc), 0);
    enc.set_buffer(5, Some(&buf_out), 0);
    enc.set_buffer(6, Some(&buf_qw), 0);
    enc.set_buffer(7, Some(&buf_kw), 0);
    let u = |v: &u32| v as *const u32 as *const c_void;
    let f = |v: &f32| v as *const f32 as *const c_void;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    enc.set_bytes(8, 4, u(&t_len));
    enc.set_bytes(9, 4, u(&head_dim));
    enc.set_bytes(10, 4, u(&num_q));
    enc.set_bytes(11, 4, u(&num_kv));
    enc.set_bytes(12, 4, f(&scale));
    let (window, eps, qk_off, rope_base, rot) = (0u32, 1e-6f32, 0.0f32, 10000.0f32, 0u32);
    enc.set_bytes(13, 4, u(&window));
    enc.set_bytes(14, 4, f(&eps));
    enc.set_bytes(15, 4, f(&qk_off));
    enc.set_bytes(16, 4, f(&rope_base));
    enc.set_bytes(17, 4, u(&rot));
    let placeholder = [0.0f32];
    let (sink_vals, has_sinks): (&[f32], u32) = match sinks {
        Some(s) => (s, 1),
        None => (&placeholder, 0),
    };
    enc.set_bytes(
        18,
        std::mem::size_of_val(sink_vals) as u64,
        sink_vals.as_ptr() as *const c_void,
    );
    enc.set_bytes(19, 4, u(&has_sinks));

    let mut tg_w: u64 = 1;
    while tg_w < head_dim as u64 && tg_w < 32 {
        tg_w <<= 1;
    }
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, 1, 1),
        metal::MTLSize::new(tg_w, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    read_back(&buf_out, out_len)
}

/// Per-head scale factor between a sinked and an unsinked run, asserting
/// it is genuinely constant across the head's dimensions.
fn per_head_ratio(plain: &[f32], sinked: &[f32], head: usize, head_dim: usize) -> f32 {
    let lo = head * head_dim;
    let mut ratio = f32::NAN;
    for d in 0..head_dim {
        let (p, s) = (plain[lo + d], sinked[lo + d]);
        if p.abs() < 1e-4 {
            continue; // ratio is ill-conditioned where the output is ~0
        }
        let r = s / p;
        if ratio.is_nan() {
            ratio = r;
        } else {
            assert!(
                (r - ratio).abs() < 1e-3,
                "head {head} dim {d}: sink must rescale the head uniformly, \
                 got ratio {r} vs {ratio}"
            );
        }
    }
    assert!(
        !ratio.is_nan(),
        "head {head}: no usable dimensions in fixture"
    );
    ratio
}

#[test]
fn attn_fused_sink_rescales_each_head_uniformly() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5u32, 2u32, 2u32, 8u32);
    let plain = run_attn_fused(&device, &lib, t_len, num_q, num_kv, head_dim, None);
    let sinked = run_attn_fused(&device, &lib, t_len, num_q, num_kv, head_dim, Some(&SINKS));

    for head in 0..num_q as usize {
        let r = per_head_ratio(&plain, &sinked, head, head_dim as usize);
        assert!(
            r > 0.0 && r < 1.0,
            "head {head}: a sink must strictly shrink the output, got factor {r}"
        );
    }
}

#[test]
fn attn_fused_larger_sink_diverts_more_mass() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5u32, 2u32, 2u32, 8u32);
    let plain = run_attn_fused(&device, &lib, t_len, num_q, num_kv, head_dim, None);
    let small = run_attn_fused(
        &device,
        &lib,
        t_len,
        num_q,
        num_kv,
        head_dim,
        Some(&[0.0, 0.0]),
    );
    let large = run_attn_fused(
        &device,
        &lib,
        t_len,
        num_q,
        num_kv,
        head_dim,
        Some(&[4.0, 4.0]),
    );

    for head in 0..num_q as usize {
        let hd = head_dim as usize;
        let r_small = per_head_ratio(&plain, &small, head, hd);
        let r_large = per_head_ratio(&plain, &large, head, hd);
        assert!(
            r_large < r_small,
            "head {head}: a larger sink must divert more mass ({r_large} vs {r_small})"
        );
    }
}

#[test]
fn attn_fused_applies_each_heads_own_sink() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5u32, 2u32, 2u32, 8u32);
    let plain = run_attn_fused(&device, &lib, t_len, num_q, num_kv, head_dim, None);
    // Head 0 negligible, head 1 dominant.
    let mixed = run_attn_fused(
        &device,
        &lib,
        t_len,
        num_q,
        num_kv,
        head_dim,
        Some(&[-40.0, 40.0]),
    );

    let hd = head_dim as usize;
    let r0 = per_head_ratio(&plain, &mixed, 0, hd);
    assert!(
        (r0 - 1.0).abs() < 1e-3,
        "head 0's negligible sink should leave it unchanged, got factor {r0}"
    );
    let head1: f32 = mixed[hd..].iter().map(|x| x.abs()).sum();
    assert!(
        head1 < 1e-3,
        "head 1's dominant sink should zero it, got magnitude {head1}"
    );
}

#[test]
fn attn_fused_without_sinks_is_unchanged_by_the_new_bindings() {
    // Two no-sink runs must agree exactly: the added bindings must not
    // perturb the default path.
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let a = run_attn_fused(&device, &lib, 5, 2, 2, 8, None);
    let b = run_attn_fused(&device, &lib, 5, 2, 2, 8, Some(&[-1e30, -1e30]));
    let diff = max_diff(&a, &b);
    assert!(
        diff < MAX_DIFF,
        "an unreachably-negative sink must match the no-sink path: max diff {diff}"
    );
}
