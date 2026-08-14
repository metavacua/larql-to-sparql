//! Metal **prefill** attention must honour a layer's sliding window.
//!
//! ROADMAP M4. This is the M1 asymmetry inverted: there prefill roped
//! correctly and decode did not; here Metal *decode* windows correctly
//! (`decode/encode_attn.rs` → `effective_window_for` → `attention_span`)
//! and Metal *prefill* did not — `stages::attention::encode` took no
//! window argument at all, so every sliding layer attended the whole
//! prefix on GPU while CPU prefill windowed at `attention/block.rs`.
//!
//! Why this test is at kernel level and uses a tiny window: the
//! model-level parity suites prompt with "The capital of France is"
//! (~16 tokens) against Gemma 3's 1024-token window, so the windowed and
//! unwindowed paths are *identical* on that fixture. A fixture too small
//! to distinguish the candidate behaviours is an absent test, not a weak
//! one. `seq_len=48, window=8` puts 40 of 48 query positions strictly
//! outside the window, so the two behaviours cannot coincide.
//!
//! The reference is the production CPU path (`gqa_attention_windowed`),
//! not a hand-rolled reimplementation — the claim under test is that the
//! two *backends* agree, so the reference has to be the other backend.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::attention::gqa_attention_windowed;
use ndarray::Array2;

const SEQ_LEN: usize = 48;
const NUM_Q: usize = 2;
const NUM_KV: usize = 1;
const HEAD_DIM: usize = 16;
/// Small enough that most positions sit outside it. With `SEQ_LEN = 48`
/// only the first 8 queries are unaffected, so an unwindowed kernel
/// disagrees on 40 of 48 rows.
const WINDOW: usize = 8;

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect()
}

fn as_2d(v: &[f32], rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_vec((rows, cols), v.to_vec()).unwrap()
}

/// Run Metal prefill attention with `window` and return `[seq, num_q*head_dim]`.
fn metal_prefill(window: Option<usize>) -> Option<Vec<f32>> {
    let device = metal::Device::system_default()?;
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .ok()?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function("fused_attention", None).unwrap(),
        )
        .ok()?;
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let q = synth(SEQ_LEN * NUM_Q * HEAD_DIM, 1.0);
    let k = synth(SEQ_LEN * NUM_KV * HEAD_DIM, 2.0);
    let v = synth(SEQ_LEN * NUM_KV * HEAD_DIM, 3.0);

    let q_buf = bufs.get_f32(&q);
    let k_buf = bufs.get_f32(&k);
    let v_buf = bufs.get_f32(&v);
    let out_buf = bufs.output((q.len() * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    larql_compute_metal::stages::attention::encode(
        enc,
        &pipeline,
        &q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        SEQ_LEN,
        NUM_Q,
        NUM_KV,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        &larql_compute::attention::rope::RopeFreqPlan::unscaled(HEAD_DIM, 0usize, 10000.0),
        larql_compute_metal::stages::attention::Flags {
            use_qk_norm: false,
            // Compare pure attention: RoPE is M1's axis, not M4's.
            skip_rope: true,
            softcap: 0.0,
            rotary_dim: 0,
            window,
        },
        None,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out =
        unsafe { std::slice::from_raw_parts(out_buf.contents() as *const f32, q.len()) }.to_vec();
    Some(out)
}

fn cpu_prefill(window: Option<usize>) -> Vec<f32> {
    let q = synth(SEQ_LEN * NUM_Q * HEAD_DIM, 1.0);
    let k = synth(SEQ_LEN * NUM_KV * HEAD_DIM, 2.0);
    let v = synth(SEQ_LEN * NUM_KV * HEAD_DIM, 3.0);
    let out = gqa_attention_windowed(
        &as_2d(&q, SEQ_LEN, NUM_Q * HEAD_DIM),
        &as_2d(&k, SEQ_LEN, NUM_KV * HEAD_DIM),
        &as_2d(&v, SEQ_LEN, NUM_KV * HEAD_DIM),
        NUM_Q,
        HEAD_DIM,
        NUM_Q / NUM_KV,
        1.0 / (HEAD_DIM as f64).sqrt(),
        SEQ_LEN,
        None,
        None,
        window,
    );
    out.into_raw_vec_and_offset().0
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Fixture-adequacy guard, run first. If windowed and unwindowed CPU
/// attention agreed on this input, every assertion below would pass on a
/// kernel that ignores the window entirely.
#[test]
fn the_window_actually_binds_on_this_fixture() {
    let windowed = cpu_prefill(Some(WINDOW));
    let full = cpu_prefill(None);
    let d = max_abs(&windowed, &full);
    assert!(
        d > 1e-3,
        "window {WINDOW} over seq {SEQ_LEN} left the output unchanged (max|Δ| {d:.3e}); \
         the fixture cannot distinguish a windowed kernel from an unwindowed one"
    );
}

/// The M4 assertion: Metal prefill under a window must match CPU prefill
/// under the same window.
#[test]
fn metal_prefill_honours_the_sliding_window() {
    let Some(gpu) = metal_prefill(Some(WINDOW)) else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let cpu = cpu_prefill(Some(WINDOW));
    let d = max_abs(&gpu, &cpu);
    assert!(
        d < 1e-4,
        "Metal prefill disagrees with CPU prefill under a {WINDOW}-token window: \
         max|Δ| {d:.3e}. If this is ~the size of the unwindowed difference, the \
         kernel is attending the whole prefix."
    );
}

/// Control: with no window the two must still agree, so a failure above
/// is attributable to the window and not to the kernel generally.
#[test]
fn metal_prefill_matches_cpu_without_a_window() {
    let Some(gpu) = metal_prefill(None) else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let cpu = cpu_prefill(None);
    let d = max_abs(&gpu, &cpu);
    assert!(
        d < 1e-4,
        "unwindowed prefill already disagrees: max|Δ| {d:.3e}"
    );
}
