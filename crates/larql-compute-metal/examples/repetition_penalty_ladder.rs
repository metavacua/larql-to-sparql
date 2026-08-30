//! Why does a GEMV reach ~97% of its isolated throughput when run ONCE
//! per token and only ~73-78% when run 24 times?
//!
//! The stage profile reproduces this across sessions to 0.4-1.9% (far
//! tighter than tok/s, which wobbles ~6%): attn.proj 205-209 GB/s and
//! ffn.routed 250-251 against isolated 283/322, while the once-per-token
//! head holds 366-368 against 377. So the dependent variable here is the
//! RATIO to the n=1 arm, never wall-clock.
//!
//! Four things could produce that, and the arms are chosen to separate
//! them rather than to confirm any one:
//!
//!   1. REPEAT CURVE — n GEMVs in one command buffer, n = 1..32, each
//!      reading DISTINCT DRAM-resident weights. R(n) flat after the
//!      first step = a one-time entry cost; R(n) sloping = cumulative.
//!   2. CB SPLIT — the same 24 GEMVs split over 1/2/6/24 command
//!      buffers. Recovery when split = a within-command-buffer effect;
//!      no recovery = the work itself, whatever contains it.
//!   3. SAME-MATRIX CONTROL — 24 repeats of ONE matrix. If distinct
//!      weights degrade and a re-read matrix does not, the memory
//!      hierarchy / address stream is implicated rather than repetition.
//!      (This is also why arms 1-2 must rotate: `same matrix x 24` is a
//!      cache experiment wearing a repetition costume.)
//!   4. ALTERNATION — 24 NVFP4 GEMVs versus 12 NVFP4 interleaved with 12
//!      f16 GEMVs, same count. Separates "repetition" from "transitions
//!      between pipeline/workload classes", which is what a real token
//!      actually does (proj → attend → experts → norms, 24 times).
//!
//! Then the highest-information arm, using the head as the internal
//! control the profile already validated:
//!
//!   5. HEAD AFTER REPEATS — 24 small GEMVs then one head-sized GEMV in
//!      the SAME command buffer, versus that head-sized GEMV alone. If
//!      the head still hits its isolated rate, whatever slows the small
//!      ones is not a global throttle the GPU is sitting in; if the head
//!      is dragged down too, accumulated clock/power state moves up the
//!      list.
//!
//! Every arm is best-of-REPS, and the n=1 arm is re-measured last as a
//! bracket: this ladder walks in one direction, and a monotone drift
//! would forge exactly the curve hypothesis 1 predicts.

use larql_compute_metal::lowering::profile::gpu_span_ms;
use larql_compute_metal::lowering::{MatvecOperands, Nvfp4Kernel};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

/// Repeated-GEMV shape: the routed-expert regime (large rows, K=2880).
const ROWS: usize = 8192;
const K: usize = 2880;
/// Head-sized GEMV for arm 5 — one long sequential stream.
const HEAD_ROWS: usize = 50176;
/// Distinct matrices in the rotation. 32 x ~13.3 MB = ~425 MB, past any
/// cache, so a repeat reads DRAM exactly as a real token's layers do.
const POOL: usize = 32;
const REPS: usize = 5;
/// Warm-up driven before every arm: enough sustained work to reach the
/// steady clock, so no arm is timed on the frequency ramp.
const WARMUP_CBS: usize = 8;
const WARMUP_GEMVS_PER_CB: usize = 24;
const REPEAT_CURVE: [usize; 7] = [1, 2, 4, 8, 12, 16, 24];
const CB_SPLITS: [usize; 4] = [1, 2, 6, 24];

fn quantised(rows: usize, seed: usize) -> nvfp4::Nvfp4Matrix {
    let values: Vec<f32> = (0..rows * K)
        .map(|i| (((i * 31 + seed * 17) % 977) as f32 / 977.0) - 0.5)
        .collect();
    nvfp4::quantize(&values, rows, K).expect("quantise")
}

fn main() {
    let Some(gpu) = MetalBackend::new() else {
        std::process::exit(2)
    };
    // Distinct backing weights, so no arm can be served from cache.
    let mats: Vec<nvfp4::Nvfp4Matrix> = (0..POOL).map(|s| quantised(ROWS, s)).collect();
    let packed: Vec<_> = mats
        .iter()
        .map(|m| gpu.lowering_weight(&m.packed))
        .collect();
    let scales: Vec<_> = mats
        .iter()
        .map(|m| gpu.lowering_weight(&m.scales))
        .collect();
    let bytes = (mats[0].packed.len() + mats[0].scales.len()) as f64;

    let x: Vec<f32> = (0..K).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    let out = gpu.lowering_scratch(HEAD_ROWS.max(ROWS));

    // One GEMV against pool entry `i`, into the shared output.
    let dispatch = |enc: &metal::ComputeCommandEncoderRef, i: usize, rows: usize| {
        gpu.encode_nvfp4_kernel(
            Nvfp4Kernel::X2,
            enc,
            &MatvecOperands {
                packed: &packed[i % POOL],
                scales: &scales[i % POOL],
                x: &xb,
                out: &out,
                out_offset: 0,
                n: rows,
                k: K,
            },
            mats[i % POOL].tensor_scale,
        );
    };

    // Drive the GPU to its steady clock before ANY arm is timed.
    //
    // Without this the ladder measures the frequency ramp, not
    // repetition: a first pass read n=1 at 150.5 µs and n=24 at 42.2,
    // then n=1 again at 39.5 — a 3.8x "penalty" that was entirely the
    // clock, and which fakes exactly the curve hypothesis 1 predicts,
    // because a small arm does too little work to leave idle clock while
    // a large one ramps itself. Every arm now starts from the same warm
    // state, and the closing bracket is what proves it held.
    let warm = || {
        for _ in 0..WARMUP_CBS {
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for j in 0..WARMUP_GEMVS_PER_CB {
                dispatch(enc, j, ROWS);
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    };

    // `n` GEMVs spread over `cbs` command buffers; returns µs per GEMV
    // (summed GPU spans, so command-buffer scheduling gaps are excluded
    // — this asks about GPU-busy efficiency, not host scheduling).
    // `rotate` false re-reads matrix 0 every time (the cache control).
    let run = |n: usize, cbs: usize, rotate: bool| -> f64 {
        warm();
        let mut best = f64::MAX;
        for _ in 0..REPS {
            let per_cb = n.div_ceil(cbs);
            let mut total = 0.0;
            let mut issued = 0usize;
            while issued < n {
                let this = per_cb.min(n - issued);
                let cmd = gpu.new_lowering_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for j in 0..this {
                    dispatch(enc, if rotate { issued + j } else { 0 }, ROWS);
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                total += gpu_span_ms(&cmd);
                issued += this;
            }
            best = best.min(total * 1e3 / n as f64);
        }
        best
    };

    // Every arm carries its OWN baseline, measured immediately before
    // it. The machine keeps accelerating under sustained load — a first
    // pass read n=1 at 49.9 µs and the same arm at 39.0 two minutes
    // later, a 22% swing that would have been booked as a repetition
    // penalty. A single opening baseline cannot survive that; an
    // adjacent one neutralises any drift slower than one arm-pair.
    let paired = |n: usize, cbs: usize, rotate: bool| -> (f64, f64) {
        let base = run(1, 1, true);
        let arm = run(n, cbs, rotate);
        (base, arm)
    };

    let gbs = |us: f64| bytes / (us / 1e6) / 1e9;
    let baseline = run(1, 1, true);
    println!(
        "repetition ladder: [{ROWS}, {K}] NVFP4, {:.1} MB/GEMV, pool {POOL} distinct (~{:.0} MB)",
        bytes / 1e6,
        bytes * POOL as f64 / 1e6
    );
    println!(
        "isolated (n=1): {baseline:.1} µs  {:.0} GB/s\n",
        gbs(baseline)
    );

    println!("1. REPEAT CURVE — n GEMVs, ONE command buffer, distinct weights");
    println!("{:>4}  {:>9}  {:>7}  {:>6}", "n", "µs/GEMV", "GB/s", "R(n)");
    for n in REPEAT_CURVE {
        let (base, t) = paired(n, 1, true);
        println!(
            "{n:>4}  {t:>9.1}  {:>7.0}  {:>6.3}   (paired n=1 {base:.1})",
            gbs(t),
            base / t
        );
    }

    println!("\n2. CB SPLIT — 24 GEMVs over N command buffers");
    println!("{:>4}  {:>9}  {:>7}  {:>6}", "CBs", "µs/GEMV", "GB/s", "R");
    for cbs in CB_SPLITS {
        let (base, t) = paired(24, cbs, true);
        println!(
            "{cbs:>4}  {t:>9.1}  {:>7.0}  {:>6.3}   (paired n=1 {base:.1})",
            gbs(t),
            base / t
        );
    }

    println!("\n3. SAME-MATRIX CONTROL — 24 repeats of ONE matrix, one CB");
    let (base_same, same) = paired(24, 1, false);
    let (base_dist, distinct) = paired(24, 1, true);
    let _ = (base_same, base_dist);
    println!(
        "  same     {same:>9.1} µs  {:>7.0} GB/s  R {:.3}\n  distinct {distinct:>9.1} µs  {:>7.0} GB/s  R {:.3}",
        gbs(same),
        base_same / same,
        gbs(distinct),
        base_dist / distinct
    );

    println!("\n5. HEAD AFTER REPEATS — does a big GEMV recover its rate?");
    let head_m = quantised(HEAD_ROWS, 99);
    let head_p = gpu.lowering_weight(&head_m.packed);
    let head_s = gpu.lowering_weight(&head_m.scales);
    let head_bytes = (head_m.packed.len() + head_m.scales.len()) as f64;
    let head_only = |preheat: usize| -> f64 {
        warm();
        let mut best = f64::MAX;
        for _ in 0..REPS {
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for j in 0..preheat {
                dispatch(enc, j, ROWS);
            }
            // Bracket the head's own span by timing the whole CB and
            // subtracting the small GEMVs at their measured rate — the
            // hardware cannot timestamp per dispatch (AtDispatchBoundary
            // is false on Apple GPUs), so this is an inference, and it
            // is only sound because arm 1 measured that rate directly.
            gpu.encode_nvfp4_kernel(
                Nvfp4Kernel::X2,
                enc,
                &MatvecOperands {
                    packed: &head_p,
                    scales: &head_s,
                    x: &xb,
                    out: &out,
                    out_offset: 0,
                    n: HEAD_ROWS,
                    k: K,
                },
                head_m.tensor_scale,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            best = best.min(gpu_span_ms(&cmd) * 1e3);
        }
        best
    };
    let alone = head_only(0);
    let after24 = head_only(24);
    let small24 = run(24, 1, true) * 24.0;
    let head_after = after24 - small24;
    println!(
        "  head alone            {alone:>9.1} µs  {:>7.0} GB/s",
        head_bytes / (alone / 1e6) / 1e9
    );
    println!(
        "  head after 24 repeats {head_after:>9.1} µs  {:>7.0} GB/s  (CB {after24:.1} − 24×small {small24:.1})",
        head_bytes / (head_after / 1e6) / 1e9
    );
    println!(
        "  → head retains {:.1}% of its alone rate",
        alone / head_after * 100.0
    );

    let close = run(1, 1, true);
    let drift = (close - baseline) / baseline * 100.0;
    println!("\nbracket: n=1 opened {baseline:.1} µs, closed {close:.1} µs — drift {drift:+.1}%");
    if drift.abs() > 5.0 {
        println!("VOID: bracket moved more than the ladder claims to measure.");
    }
}
