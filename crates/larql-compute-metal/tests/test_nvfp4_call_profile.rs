//! Drive `diag/nvfp4_call_profile.rs` end to end on a real device.
//!
//! The module splits one NVFP4 gemv call into host/GPU phases
//! (`profile_nvfp4_gemv`) and measures the per-dispatch cost of a
//! pipelined burst (`nvfp4_pipelined_cost`). Both are diagnostics, so
//! this file does not assert any latency number — those vary with the
//! machine and with contention. What it does pin:
//!
//! | arm | claim |
//! |---|---|
//! | 1 | every phase of a profiled call is finite and non-negative |
//! | 2 | the wall-clock phases (input, acquire, encode, commit, wait, readback) sum to no more than `total` — they are sub-intervals of it, so a profile whose parts exceed its whole is mis-attributed |
//! | 3 | `commit_to_gpu_start` is derived as `commit→done − gpu_span`, so it is bounded above by `commit + wait` plus timer slack |
//! | 4 | the GPU reported a completed span — `gpu_span > 0` — which is what distinguishes "we read real GPU timestamps" from a selector that returned zeros |
//! | 5 | profiling leaves the backend usable: a second call hits the weight cache rather than inserting new entries (cache size does not grow), and a real `nvfp4_gemv` afterwards still matches the CPU reference |
//! | 6 | `nvfp4_pipelined_cost` returns a finite, positive per-dispatch microsecond cost for `depth = 1` and for a deep burst, and the same cache-hit/correctness invariants hold |
//!
//! The `None` arms in both functions fire only when the pooled input
//! buffer's `contents()` is null. Shared-storage Metal buffers never
//! report a null contents pointer, so those arms are not reachable from
//! a test and are documented as such in the coverage report rather than
//! faked.

#![cfg(target_os = "macos")]

use larql_compute::backend::matmul::MatMul;
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4::{self, NVFP4_GROUP_ELEMS};

/// Deliberately not a multiple of the kernel's rows-per-threadgroup, so
/// the profiled dispatch exercises the tail-row guard like a real call.
const ROWS: usize = 13;
/// Groups per row — enough that the scale stream is a real stream.
const GROUPS_PER_ROW: usize = 24;
const K: usize = NVFP4_GROUP_ELEMS * GROUPS_PER_ROW;
/// A burst deep enough that queueing, if it pipelines at all, is visible.
const PIPELINE_DEPTH: usize = 8;
/// Single dispatch — the degenerate burst the deep one is compared to.
const SINGLE_DISPATCH: usize = 1;
/// Two `Instant` reads bracketing each phase are not free; the sum of
/// sub-intervals may exceed `total` by their own overhead. Generous
/// because the test must not flake under contention.
const TIMER_SLACK_US: f64 = 50.0;
/// Relative error budget for the post-profile gemv against the CPU
/// reference — reduction-order noise only.
const GEMV_REL_ERROR_BUDGET: f32 = 1e-4;

fn fixture_values(rows: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    (0..rows * k)
        .map(|i| {
            let group = (i % k) / NVFP4_GROUP_ELEMS;
            let octave = (group % 12) as i32;
            next() * (2.0f32).powi(-2 * octave)
        })
        .collect()
}

fn input_vector(k: usize) -> Vec<f32> {
    (0..k).map(|i| ((i as f32) * 0.017).sin()).collect()
}

fn cpu_reference(matrix: &nvfp4::Nvfp4Matrix, rows: usize, k: usize, x: &[f32]) -> Vec<f32> {
    let mut weights = vec![0.0f32; rows * k];
    nvfp4::dequantize_into(matrix, rows, k, &mut weights).expect("dequantise");
    (0..rows)
        .map(|r| {
            weights[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(w, v)| w * v)
                .sum()
        })
        .collect()
}

fn rel_error(reference: &[f32], got: &[f32]) -> f32 {
    let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    reference
        .iter()
        .zip(got)
        .map(|(r, g)| (r - g).abs())
        .fold(0.0f32, f32::max)
        / scale
}

struct Fixture {
    matrix: nvfp4::Nvfp4Matrix,
    x: Vec<f32>,
    reference: Vec<f32>,
}

fn fixture() -> Fixture {
    let values = fixture_values(ROWS, K, 7);
    let x = input_vector(K);
    let matrix = nvfp4::quantize(&values, ROWS, K).expect("quantise");
    let reference = cpu_reference(&matrix, ROWS, K, &x);
    Fixture {
        matrix,
        x,
        reference,
    }
}

/// The gemv the profiled kernel mirrors must still be right after the
/// diagnostic has run — proves the diagnostic recycled rather than
/// corrupted the shared pool and pipelines.
fn assert_gemv_still_matches_reference(gpu: &MetalBackend, fx: &Fixture) {
    let got = gpu
        .nvfp4_gemv(
            &fx.matrix.packed,
            &fx.matrix.scales,
            fx.matrix.tensor_scale,
            &fx.x,
            ROWS,
            K,
        )
        .expect("Metal backend must have an NVFP4 kernel");
    let err = rel_error(&fx.reference, &got);
    assert!(
        err < GEMV_REL_ERROR_BUDGET,
        "nvfp4_gemv after profiling disagrees with the CPU reference by {err}"
    );
}

fn assert_finite_non_negative(name: &str, value: f64) {
    assert!(
        value.is_finite() && value >= 0.0,
        "{name} must be a finite non-negative microsecond count, got {value}"
    );
}

/// Arms 1–5: one profiled call reports an internally consistent phase
/// breakdown with real GPU timestamps and leaves the backend intact.
#[test]
fn profile_nvfp4_gemv_reports_consistent_phases_and_real_gpu_span() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let fx = fixture();

    // Warm the pool so the measured call is a steady-state call, not a
    // first-touch allocation — that is what the module claims to model.
    let warm = gpu
        .profile_nvfp4_gemv(
            &fx.matrix.packed,
            &fx.matrix.scales,
            fx.matrix.tensor_scale,
            &fx.x,
            ROWS,
            K,
        )
        .expect("warm-up profile");
    assert_finite_non_negative("warm total", warm.total);

    let cache_before = gpu.cache_size();
    let p = gpu
        .profile_nvfp4_gemv(
            &fx.matrix.packed,
            &fx.matrix.scales,
            fx.matrix.tensor_scale,
            &fx.x,
            ROWS,
            K,
        )
        .expect("profile on a live device returns Some");

    // Arm 1.
    assert_finite_non_negative("input_stage", p.input_stage);
    assert_finite_non_negative("buffer_acquire", p.buffer_acquire);
    assert_finite_non_negative("encode", p.encode);
    assert_finite_non_negative("commit", p.commit);
    assert_finite_non_negative("wait", p.wait);
    assert_finite_non_negative("readback", p.readback);
    assert_finite_non_negative("total", p.total);
    assert!(p.gpu_span.is_finite(), "gpu_span must be finite");
    assert!(
        p.commit_to_gpu_start.is_finite(),
        "commit_to_gpu_start must be finite"
    );

    // Arm 2.
    let phase_sum = p.input_stage + p.buffer_acquire + p.encode + p.commit + p.wait + p.readback;
    assert!(
        phase_sum <= p.total + TIMER_SLACK_US,
        "wall phases ({phase_sum} us) exceed the whole call ({} us)",
        p.total
    );
    assert!(p.total > 0.0, "a real call cannot take zero wall time");

    // Arm 3: commit→done is commit + wait, and the derived queue latency
    // is that minus the GPU span.
    assert!(
        p.commit_to_gpu_start <= p.commit + p.wait + TIMER_SLACK_US,
        "derived queue latency {} us exceeds commit+wait {} us",
        p.commit_to_gpu_start,
        p.commit + p.wait
    );

    // Arm 4.
    assert!(
        p.gpu_span > 0.0,
        "GPUStartTime/GPUEndTime must bracket a completed dispatch; got span {} us",
        p.gpu_span
    );

    // Arm 5.
    assert!(
        gpu.cache_size() <= cache_before,
        "a warmed profile must hit the weight cache, not insert; cache grew {} -> {}",
        cache_before,
        gpu.cache_size()
    );
    assert_gemv_still_matches_reference(&gpu, &fx);
}

/// Arm 6: the pipelined-burst measurement returns a positive finite
/// per-dispatch cost at both the degenerate depth and a deep burst, hits
/// the weight cache on the second call, and leaves the gemv correct.
#[test]
fn nvfp4_pipelined_cost_reports_positive_per_dispatch_cost_and_recycles_scratch() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let fx = fixture();

    let single = gpu
        .nvfp4_pipelined_cost(
            &fx.matrix.packed,
            &fx.matrix.scales,
            fx.matrix.tensor_scale,
            &fx.x,
            ROWS,
            K,
            SINGLE_DISPATCH,
        )
        .expect("depth-1 burst on a live device returns Some");
    assert!(
        single.is_finite() && single > 0.0,
        "depth-1 per-dispatch cost must be finite and positive, got {single}"
    );

    let cache_before = gpu.cache_size();
    let deep = gpu
        .nvfp4_pipelined_cost(
            &fx.matrix.packed,
            &fx.matrix.scales,
            fx.matrix.tensor_scale,
            &fx.x,
            ROWS,
            K,
            PIPELINE_DEPTH,
        )
        .expect("deep burst on a live device returns Some");
    assert!(
        deep.is_finite() && deep > 0.0,
        "deep-burst per-dispatch cost must be finite and positive, got {deep}"
    );
    assert!(
        gpu.cache_size() <= cache_before,
        "a second burst must hit the weight cache, not insert; cache grew {} -> {}",
        cache_before,
        gpu.cache_size()
    );
    // The figure is PER DISPATCH, not a total for the burst. Were the
    // division dropped, a depth-N burst would report roughly N times the
    // depth-1 cost and every assertion above would still pass. This
    // bounds that mistake without asserting that pipelining actually
    // wins — which is the open question the diagnostic exists to answer
    // and must not be baked into its own gate.
    assert!(
        deep < single * PIPELINE_DEPTH as f64,
        "depth-{PIPELINE_DEPTH} reported {deep} against depth-1's {single} — that \
         looks like a burst total rather than a per-dispatch figure"
    );
    assert_gemv_still_matches_reference(&gpu, &fx);
}
