//! The stage profiler: does the `StageEncoders` seam attribute GPU time
//! to stages without changing what executes?
//!
//! Three claims, each with its own check:
//!
//! 1. **Same numbers.** A dependent chain encoded through `SingleEncoder`
//!    and through `StageProfiler` lands bit-identical output — the seam
//!    changes scheduling granularity only.
//! 2. **Attribution is real.** Each stage that dispatched work records a
//!    positive span; consecutive same-stage calls merge into one run; a
//!    stage that never dispatched records nothing; the stage sum is
//!    bounded by the command buffer's own GPU span.
//! 3. **Overflow is counted, not silent.** Past capacity the dispatches
//!    still execute (the output is still right) and the report says how
//!    many runs went unattributed.
//!
//! Plus the pure arithmetic (`profile_from_samples`, `StageProfile::add`)
//! on hand-built timestamp arrays, where the device is not involved.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

use larql_compute_metal::lowering::profile::{
    gpu_span_ms, profile_from_samples, SingleEncoder, Stage, StageEncoders, StageProfile,
    StageProfiler,
};
use larql_compute_metal::MetalBackend;

const LEN: usize = 2880;
/// Dependent steps per stage in the chain.
const STEPS: usize = 8;

/// Encode a chain through `encs`: `STEPS` residual-adds under each of the
/// given stages, each reading what the previous wrote. Returns nothing;
/// the caller owns the command buffer.
fn encode_chain(
    gpu: &MetalBackend,
    encs: &mut dyn StageEncoders,
    stages: &[Stage],
    a: &metal::Buffer,
    b: &metal::Buffer,
) {
    let mut i = 0usize;
    for &stage in stages {
        for _ in 0..STEPS {
            let enc = encs.stage(stage);
            let (x, y) = if i.is_multiple_of(2) { (a, b) } else { (b, a) };
            gpu.encode_residual_add(enc, x, y, y, LEN, 1.0);
            i += 1;
        }
    }
}

fn fresh_inputs(gpu: &MetalBackend) -> (metal::Buffer, metal::Buffer) {
    let a: Vec<f32> = (0..LEN).map(|i| (i % 7) as f32 * 0.5).collect();
    let b: Vec<f32> = (0..LEN).map(|i| 1.0 + (i % 3) as f32).collect();
    (
        gpu.lowering_upload(&a).expect("upload a"),
        gpu.lowering_upload(&b).expect("upload b"),
    )
}

#[test]
fn profiled_and_single_encoder_chains_are_bit_identical() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let stages = [Stage::AttnProj, Stage::AttnCore, Stage::DenseFfn];

    // Arm 1: production seam.
    let (a1, b1) = fresh_inputs(&gpu);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_chain(&gpu, &mut SingleEncoder(enc), &stages, &a1, &b1);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let single_a = gpu.lowering_readback(&a1, LEN).expect("read a");
    let single_b = gpu.lowering_readback(&b1, LEN).expect("read b");

    // Arm 2: profiler seam.
    let (a2, b2) = fresh_inputs(&gpu);
    let cmd = gpu.new_lowering_command_buffer();
    let Some(mut prof) = StageProfiler::new(&gpu.device_ref(), cmd.clone(), 64) else {
        eprintln!("no stage-boundary counters (CI paravirtual GPU); skipping");
        return;
    };
    encode_chain(&gpu, &mut prof, &stages, &a2, &b2);
    let (cmd, samples) = prof.finish();
    cmd.commit();
    cmd.wait_until_completed();
    let prof_a = gpu.lowering_readback(&a2, LEN).expect("read a");
    let prof_b = gpu.lowering_readback(&b2, LEN).expect("read b");

    assert_eq!(single_a, prof_a, "seam changed the numbers (a)");
    assert_eq!(single_b, prof_b, "seam changed the numbers (b)");

    let profile = samples.resolve().expect("resolve");
    for stage in stages {
        let ns = profile.stage_ns.get(&stage).copied().unwrap_or(0);
        assert!(ns > 0, "{stage:?} dispatched work but recorded {ns} ns");
        assert_eq!(
            profile.stage_runs.get(&stage).copied(),
            Some(1),
            "{STEPS} consecutive same-stage calls must merge into one run"
        );
    }
    assert!(
        !profile.stage_ns.contains_key(&Stage::Head),
        "a stage that never dispatched must not appear"
    );
    // The counters and `GPUStartTime`/`GPUEndTime` are read from different
    // clocks; they agree to within a few microseconds on a multi-ms span
    // (`examples/counter_stage_probe.rs`), so the bound carries a small
    // tolerance rather than demanding equality of two clocks.
    let gpu_ns = gpu_span_ms(&cmd) * 1e6;
    let tolerance_ns = gpu_ns * 0.05 + 50_000.0;
    assert!(
        (profile.attributed_ns() as f64) <= gpu_ns + tolerance_ns,
        "stage sum {} ns exceeds the command buffer's GPU span {gpu_ns:.0} ns",
        profile.attributed_ns(),
    );
    assert!(
        (profile.span_ns as f64) <= gpu_ns + tolerance_ns,
        "sampled span {} ns exceeds the command buffer's GPU span {gpu_ns:.0} ns",
        profile.span_ns
    );
    assert!(profile.gap_ns() <= profile.span_ns);
    assert_eq!(profile.overflowed, 0);
}

#[test]
fn revisiting_a_stage_opens_a_new_run() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let stages = [Stage::AttnNorm, Stage::AttnProj, Stage::AttnNorm];
    let (a, b) = fresh_inputs(&gpu);
    let cmd = gpu.new_lowering_command_buffer();
    let Some(mut prof) = StageProfiler::new(&gpu.device_ref(), cmd.clone(), 64) else {
        eprintln!("no stage-boundary counters (CI paravirtual GPU); skipping");
        return;
    };
    encode_chain(&gpu, &mut prof, &stages, &a, &b);
    let (cmd, samples) = prof.finish();
    cmd.commit();
    cmd.wait_until_completed();
    let profile = samples.resolve().expect("resolve");
    assert_eq!(profile.stage_runs.get(&Stage::AttnNorm).copied(), Some(2));
    assert_eq!(profile.stage_runs.get(&Stage::AttnProj).copied(), Some(1));
}

#[test]
fn overflow_past_capacity_still_executes_and_is_counted() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let stages = [
        Stage::AttnProj,
        Stage::AttnCore,
        Stage::DenseFfn,
        Stage::Head,
    ];

    let (a_ref, b_ref) = fresh_inputs(&gpu);
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_chain(&gpu, &mut SingleEncoder(enc), &stages, &a_ref, &b_ref);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let want = gpu.lowering_readback(&b_ref, LEN).expect("read");

    // Capacity 2: the third and fourth stages overflow.
    let (a, b) = fresh_inputs(&gpu);
    let cmd = gpu.new_lowering_command_buffer();
    let Some(mut prof) = StageProfiler::new(&gpu.device_ref(), cmd.clone(), 2) else {
        eprintln!("no stage-boundary counters (CI paravirtual GPU); skipping");
        return;
    };
    encode_chain(&gpu, &mut prof, &stages, &a, &b);
    let (cmd, samples) = prof.finish();
    cmd.commit();
    cmd.wait_until_completed();
    let got = gpu.lowering_readback(&b, LEN).expect("read");
    assert_eq!(want, got, "overflowed stages must still execute");

    let profile = samples.resolve().expect("resolve");
    assert_eq!(profile.stage_runs.len(), 2, "only two runs fit");
    // Two stages past capacity, each requested once per step after the
    // first (same-stage calls on the unsampled encoder still count as
    // requests the buffer could not hold).
    assert!(
        profile.overflowed >= 2,
        "overflow must be counted, got {}",
        profile.overflowed
    );
}

#[test]
fn single_encoder_returns_the_same_encoder_for_every_stage() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut single = SingleEncoder(enc);
    let first: *const _ = single.stage(Stage::AttnNorm);
    for stage in Stage::ALL {
        let again: *const _ = single.stage(stage);
        assert!(std::ptr::eq(first, again));
    }
    enc.end_encoding();
}

#[test]
fn profiler_refuses_a_sample_buffer_the_device_cannot_hold() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let cmd = gpu.new_lowering_command_buffer();
    // 2048 runs (4096 samples) is the documented ceiling; far past it
    // the device refuses and the constructor must say so, not panic.
    assert!(StageProfiler::new(&gpu.device_ref(), cmd, 1 << 20).is_none());
}

// ── pure arithmetic ────────────────────────────────────────────────────

#[test]
fn profile_from_samples_folds_runs_and_spans() {
    let runs = [
        (Stage::AttnProj, 0usize),
        (Stage::Head, 2),
        (Stage::AttnProj, 4),
    ];
    let ts = [100u64, 150, 160, 200, 210, 240];
    let p = profile_from_samples(&runs, &ts, 3);
    assert_eq!(p.stage_ns[&Stage::AttnProj], 50 + 30);
    assert_eq!(p.stage_ns[&Stage::Head], 40);
    assert_eq!(p.stage_runs[&Stage::AttnProj], 2);
    assert_eq!(p.stage_runs[&Stage::Head], 1);
    assert_eq!(p.span_ns, 240 - 100);
    assert_eq!(p.attributed_ns(), 120);
    assert_eq!(p.gap_ns(), 20);
    assert_eq!(p.overflowed, 3);
}

#[test]
fn profile_from_samples_skips_runs_past_the_timestamp_array() {
    let runs = [(Stage::AttnProj, 0usize), (Stage::Head, 2)];
    let ts = [10u64, 30];
    let p = profile_from_samples(&runs, &ts, 0);
    assert_eq!(p.stage_ns.len(), 1);
    assert_eq!(p.span_ns, 20);
    let empty = profile_from_samples(&[], &[], 0);
    assert_eq!(empty, StageProfile::default());
}

#[test]
fn profile_add_accumulates_across_tokens() {
    let mut acc = StageProfile::default();
    let one = StageProfile {
        stage_ns: BTreeMap::from([(Stage::Head, 5u64), (Stage::AttnCore, 7)]),
        stage_runs: BTreeMap::from([(Stage::Head, 1u32), (Stage::AttnCore, 2)]),
        span_ns: 20,
        overflowed: 1,
    };
    acc.add(&one);
    acc.add(&one);
    assert_eq!(acc.stage_ns[&Stage::Head], 10);
    assert_eq!(acc.stage_runs[&Stage::AttnCore], 4);
    assert_eq!(acc.span_ns, 40);
    assert_eq!(acc.overflowed, 2);
}

#[test]
fn stage_order_and_labels_are_distinct() {
    let labels: std::collections::BTreeSet<&str> = Stage::ALL.iter().map(|s| s.label()).collect();
    assert_eq!(labels.len(), Stage::ALL.len(), "labels must be distinct");
    for w in Stage::ALL.windows(2) {
        assert!(w[0] < w[1], "ALL must be in encode order");
    }
}
