//! G6c-2: final norm + output head + logits.
//!
//! Every judged fact here differs from its nearest neighbour elsewhere in
//! the stack, so each gets its own control:
//!
//! | fact | Glimmer's value | nearest neighbour |
//! |---|---|---|
//! | final norm eps | 1e-5 | post-branch norms use 1e-8 |
//! | final norm offset | **0.0** | branch norms use 1.0 |
//! | output multiplier | 0.19611613 | — |
//! | logit softcap | 20.0 | applied *after* the multiplier |
//!
//! ## The multiplier/softcap order is semantic
//!
//! Unlike the query-scale/RoPE pair — which commute exactly, and whose
//! control is asserted *as* unobservable — tanh is nonlinear, so
//! `softcap(m·x)` and `m·softcap(x)` are different functions:
//! `20·tanh(0.196x/20)` against `3.92·tanh(x/20)`. The order is asserted
//! to matter, and a control proves the fixture can see it.
//!
//! ## Anchor scope, stated honestly
//!
//! This compares against a CPU reference transcribed from
//! `ProductionBackend::output_head`, which establishes that the lowering
//! implements the **VINDEX3 plan**. It does *not* rule out the plan and
//! the lowering sharing a mistake — the failure mode the four-norm
//! omission actually exhibited. That check needs the real Glimmer oracle
//! logits and the real weights, which arrive at G6d when the whole path
//! runs against the container.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::head::{HeadScratch, HeadShape, HeadWeights};
use larql_compute_metal::lowering::profile::SingleEncoder;
use larql_compute_metal::lowering::LoweredMatrix;
use larql_models::quant::nvfp4;

const HIDDEN: usize = 256;
const VOCAB: usize = 2048;
const EPS: f32 = 1e-5;
/// Glimmer's final norm is uncentred, unlike its branch norms.
const FINAL_OFFSET: f32 = 0.0;
const MULTIPLIER: f32 = 0.19611613;
const SOFTCAP: f32 = 20.0;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(5);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * 2.0
        })
        .collect()
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        n += (*x as f64 - *y as f64).powi(2);
        d += (*x as f64).powi(2);
    }
    (n / d).sqrt()
}

/// Order of the two elementwise head ops.
#[derive(Clone, Copy, PartialEq)]
enum Order {
    /// `softcap(m * x)` — the interpreter's.
    ScaleThenCap,
    /// `m * softcap(x)` — the plausible transposition.
    CapThenScale,
}

/// Transcribed from `ProductionBackend::output_head`.
///
/// Every judged fact is a parameter, including the two a control
/// perturbs — a reference that hard-coded them could not model the
/// defects the controls exist to detect.
#[allow(clippy::too_many_arguments)]
fn cpu_head(
    h: &[f32],
    norm_w: &[f32],
    proj: &[f32],
    eps: f32,
    offset: f32,
    multiplier: Option<f32>,
    softcap: Option<f32>,
    order: Order,
) -> Vec<f32> {
    let ms = h.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    let normed: Vec<f32> = h
        .iter()
        .zip(norm_w)
        .map(|(x, w)| x * inv * (offset + w))
        .collect();
    let mut logits: Vec<f32> = (0..VOCAB)
        .map(|r| (0..HIDDEN).map(|c| proj[r * HIDDEN + c] * normed[c]).sum())
        .collect();
    for l in &mut logits {
        match order {
            Order::ScaleThenCap => {
                if let Some(m) = multiplier {
                    *l *= m;
                }
                if let Some(c) = softcap {
                    *l = c * (*l / c).tanh();
                }
            }
            Order::CapThenScale => {
                if let Some(c) = softcap {
                    *l = c * (*l / c).tanh();
                }
                if let Some(m) = multiplier {
                    *l *= m;
                }
            }
        }
    }
    logits
}

#[allow(clippy::too_many_arguments)]
fn run_lowered(
    gpu: &larql_compute_metal::MetalBackend,
    h: &[f32],
    norm_w: &[f32],
    proj: &nvfp4::Nvfp4Matrix,
    eps: f32,
    offset: f32,
    multiplier: Option<f32>,
    softcap: Option<f32>,
) -> Vec<f32> {
    let h_buf = gpu.lowering_upload(h).unwrap();
    let n_buf = gpu.lowering_upload(norm_w).unwrap();
    let normed = gpu.lowering_scratch(HIDDEN);
    let raw = gpu.lowering_scratch(VOCAB);
    let out = gpu.lowering_scratch(VOCAB);
    let pk = gpu.lowering_weight(&proj.packed);
    let sk = gpu.lowering_weight(&proj.scales);

    let w = HeadWeights {
        projection: LoweredMatrix::Nvfp4 {
            packed: &pk,
            packed_offset: 0,
            scales: &sk,
            scales_offset: 0,
            tensor_scale: proj.tensor_scale,
        },
        norm_weight: &n_buf,
    };
    let s = HeadScratch {
        normed: &normed,
        raw_logits: &raw,
    };
    let shape = HeadShape {
        hidden: HIDDEN,
        vocab: VOCAB,
        norm_eps: eps,
        norm_weight_offset: offset,
        multiplier,
        softcap,
    };
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    gpu.encode_head(&mut SingleEncoder(enc), &h_buf, &out, &w, &s, &shape);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let logits = gpu.lowering_readback(&out, VOCAB).unwrap();
    for b in [h_buf, n_buf, normed, raw, out] {
        gpu.recycle_lowering_scratch(b);
    }
    logits
}

fn assert_control(what: &str, perturbed: &[f32], got: &[f32], parity: f64) {
    let c = rel_rms(perturbed, got);
    let ratio = c / parity;
    eprintln!("  control `{what}`: {ratio:.0}x parity ({c:.3e})");
    assert!(
        ratio > 100.0,
        "control `{what}` moves the result only {ratio:.1}x the parity residual"
    );
}

#[test]
fn lowered_head_carries_every_judged_fact() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let h = det(HIDDEN, 1);
    let norm_w = det(HIDDEN, 2);
    let proj_f = det(VOCAB * HIDDEN, 3);
    let proj = nvfp4::quantize(&proj_f, VOCAB, HIDDEN).unwrap();
    let proj_q = nvfp4::round_trip(&proj_f, VOCAB, HIDDEN).unwrap();

    let want = cpu_head(
        &h,
        &norm_w,
        &proj_q,
        EPS,
        FINAL_OFFSET,
        Some(MULTIPLIER),
        Some(SOFTCAP),
        Order::ScaleThenCap,
    );
    let got = run_lowered(
        &gpu,
        &h,
        &norm_w,
        &proj,
        EPS,
        FINAL_OFFSET,
        Some(MULTIPLIER),
        Some(SOFTCAP),
    );

    assert!(
        got.iter().all(|v| v.is_finite()),
        "head produced non-finite logits"
    );
    let parity = rel_rms(&want, &got);
    eprintln!("head parity: rel_rms {parity:.3e}");
    assert!(parity < 1e-4, "lowered head disagrees: {parity:.3e}");

    // ── final norm is uncentred (0.0), not centred like branch norms ─
    let centred = cpu_head(
        &h,
        &norm_w,
        &proj_q,
        EPS,
        1.0,
        Some(MULTIPLIER),
        Some(SOFTCAP),
        Order::ScaleThenCap,
    );
    assert_control(
        "final norm weight_offset 1.0 instead of 0.0",
        &centred,
        &got,
        parity,
    );

    // ── final norm present at all ───────────────────────────────────
    let unnormed: Vec<f32> = {
        let mut l: Vec<f32> = (0..VOCAB)
            .map(|r| (0..HIDDEN).map(|c| proj_q[r * HIDDEN + c] * h[c]).sum())
            .collect();
        for v in &mut l {
            *v *= MULTIPLIER;
            *v = SOFTCAP * (*v / SOFTCAP).tanh();
        }
        l
    };
    assert_control("final norm omitted", &unnormed, &got, parity);

    // ── output multiplier ───────────────────────────────────────────
    let no_mult = cpu_head(
        &h,
        &norm_w,
        &proj_q,
        EPS,
        FINAL_OFFSET,
        None,
        Some(SOFTCAP),
        Order::ScaleThenCap,
    );
    assert_control("output multiplier omitted", &no_mult, &got, parity);

    // ── softcap ─────────────────────────────────────────────────────
    let no_cap = cpu_head(
        &h,
        &norm_w,
        &proj_q,
        EPS,
        FINAL_OFFSET,
        Some(MULTIPLIER),
        None,
        Order::ScaleThenCap,
    );
    assert_control("logit softcap omitted", &no_cap, &got, parity);

    // ── the order of the two: semantic, unlike query-scale vs RoPE ──
    let swapped = cpu_head(
        &h,
        &norm_w,
        &proj_q,
        EPS,
        FINAL_OFFSET,
        Some(MULTIPLIER),
        Some(SOFTCAP),
        Order::CapThenScale,
    );
    assert_control(
        "softcap applied before the multiplier",
        &swapped,
        &got,
        parity,
    );
}

/// The final-norm epsilon, tested where it is observable — the same
/// regime problem the post-branch epsilon had. Compares two *lowered*
/// runs, since the question is whether the plan's value reaches the
/// kernel.
#[test]
fn final_norm_epsilon_is_read_where_it_is_observable() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // Small hidden state so eps is a real fraction of the mean-square.
    let h: Vec<f32> = det(HIDDEN, 1).iter().map(|v| v * 2e-3).collect();
    let norm_w = det(HIDDEN, 2);
    let proj_f = det(VOCAB * HIDDEN, 3);
    let proj = nvfp4::quantize(&proj_f, VOCAB, HIDDEN).unwrap();

    let a = run_lowered(
        &gpu,
        &h,
        &norm_w,
        &proj,
        EPS,
        FINAL_OFFSET,
        Some(MULTIPLIER),
        Some(SOFTCAP),
    );
    let b = run_lowered(
        &gpu,
        &h,
        &norm_w,
        &proj,
        1e-8,
        FINAL_OFFSET,
        Some(MULTIPLIER),
        Some(SOFTCAP),
    );
    let ms = h.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / HIDDEN as f64;
    let d = rel_rms(&a, &b);
    eprintln!("final norm eps 1e-5 vs 1e-8 (hidden ms {ms:.2e}): rel_rms {d:.3e}");
    assert!(
        d > 1e-3,
        "the plan's final-norm epsilon must reach the kernel; changing it moved the \
         logits only {d:.3e} at mean-square {ms:.2e}"
    );
}
