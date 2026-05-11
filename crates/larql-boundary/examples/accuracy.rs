//! Accuracy of the boundary codec: downstream quality metrics.
//!
//! MSE of the reconstructed residual is *not* the contract — the contract is
//! top-1 token preservation under `lm_head(final_norm(residual))`.
//!
//! This example demonstrates the accuracy metrics (top-1, top-5, KL, prob_ratio)
//! directly via Phase 2 `compute_metadata`, using synthetic logit vectors that
//! match the distribution profiles measured in Exp 43.
//!
//! It also shows what the Phase 1 codec does to the residual vector and why
//! residual MSE is misleading (dominated by the outlier saturation).
//!
//! Run with:
//!   cargo run -p larql-boundary --example accuracy

use larql_boundary::{
    codec::int8,
    metadata::{compute, BoundaryMetadata},
};

const D: usize = 2560;

// ── Helpers ────────────────────────────────────────────────────────────────

fn kl_nats(raw_lp: &[f32], hat_lp: &[f32]) -> f64 {
    raw_lp
        .iter()
        .zip(hat_lp.iter())
        .map(|(&lp, &lq)| {
            let p = lp.exp() as f64;
            if p < 1e-12 {
                0.0
            } else {
                p * (lp as f64 - lq as f64)
            }
        })
        .sum::<f64>()
        .max(0.0)
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&x| x - max).collect();
    let log_z = shifted.iter().map(|&x| x.exp()).sum::<f32>().ln();
    shifted.iter().map(|&x| x - log_z).collect()
}

fn top_k_contains_top1(raw_top1: u32, hat_lp: &[f32], k: usize) -> bool {
    let mut indexed: Vec<(usize, f32)> = hat_lp.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed[..k].iter().any(|(i, _)| *i == raw_top1 as usize)
}

fn print_accuracy(label: &str, meta: &BoundaryMetadata, raw_lp: &[f32], hat_lp: &[f32]) {
    let kl = kl_nats(raw_lp, hat_lp);
    let top5 = top_k_contains_top1(meta.raw_top1_token, hat_lp, 5);
    let top1 = matches!(
        meta.boundary_agreement,
        larql_boundary::BoundaryAgreement::Agrees
    );
    let p_raw = raw_lp[meta.raw_top1_token as usize].exp() as f64;
    let p_hat = hat_lp[meta.raw_top1_token as usize].exp() as f64;
    let prob_ratio = p_hat / p_raw.max(1e-12);

    println!("  {label}");
    println!(
        "    top-1 agree:       {} (agreement={:?})",
        if top1 { "✓" } else { "✗" },
        meta.boundary_agreement
    );
    println!("    top-5 agree:       {}", if top5 { "✓" } else { "✗" });
    println!("    KL (nats):         {kl:.4}");
    println!("    prob_ratio:        {prob_ratio:.3}  (p_hat[top1] / p_raw[top1])");
    println!(
        "    raw_log_prob_margin: {:.4}  (gate threshold=2.16)",
        meta.raw_log_prob_margin
    );
}

// ── Synthetic logit profiles from Exp 43 characterisation ─────────────────

/// Confident boundary: large margin between top-1 and top-2 (~tech_3).
/// The codec perturbation is small relative to the gap → top-1 preserved.
fn confident_logits_raw(vocab: usize) -> Vec<f32> {
    let mut l = vec![-5.0f32; vocab];
    l[42_001] = 9.0; // clear top-1
    l[17_003] = 0.0; // top-2: 9.0 logit gap
    l
}
fn confident_logits_hat(vocab: usize) -> Vec<f32> {
    // Codec shifts logits slightly but not enough to change top-1.
    let mut l = confident_logits_raw(vocab);
    l[42_001] -= 0.3; // small shift, still well above top-2
    l[17_003] += 0.2; // top-2 comes up slightly
    l
}

/// Fragile boundary: tiny margin (~qa_2 pattern).
/// The same codec perturbation flips the argmax.
fn fragile_logits_raw(vocab: usize) -> Vec<f32> {
    let mut l = vec![-5.0f32; vocab];
    l[42_001] = 0.12; // top-1 by only 0.12 logits
    l[17_003] = 0.00; // top-2 very close
    l
}
fn fragile_logits_hat(vocab: usize) -> Vec<f32> {
    // Same-magnitude perturbation now flips the winner.
    let mut l = fragile_logits_raw(vocab);
    l[42_001] -= 0.3; // now BELOW top-2 → argmax flips to 17_003
    l[17_003] += 0.2;
    l
}

fn main() {
    // Vocab size matches Gemma 3 4B (262 145).
    // Using a smaller subset for demo performance; real accuracy uses full vocab.
    let vocab = 262_145;

    println!("larql-boundary accuracy demonstration");
    println!("(Synthetic logit profiles matching Exp 43 measured distributions)");
    println!();

    // ── Phase 1: residual codec ───────────────────────────────────────────────
    println!("── Phase 1: Residual codec (what the bytes look like) ───────────────");
    let mut residual: Vec<f32> = (0..D)
        .map(|i| ((i as f32) * 0.0023).sin() * 1650.0)
        .collect();
    residual[42] = 94_208.0; // outlier matching Gemma 3 4B absmax
    residual[512] = -60_000.0; // outlier

    let payload = int8::encode(&residual);
    let hat_residual = int8::decode(&payload);

    let bf16_bytes = larql_boundary::codec::bf16::encode(&residual).len();
    let int8_bytes = payload.to_bytes().len();

    let total_mse: f32 = residual
        .iter()
        .zip(hat_residual.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / D as f32;
    let non_outlier_mse: f32 = residual
        .iter()
        .zip(hat_residual.iter())
        .enumerate()
        .filter(|(i, _)| *i != 42 && *i != 512)
        .map(|(_, (a, b))| (a - b).powi(2))
        .sum::<f32>()
        / (D - 2) as f32;

    println!("  bf16:            {bf16_bytes:>5} bytes  (near-lossless reference)");
    println!(
        "  int8_clip3σ:     {int8_bytes:>5} bytes  ({:.1}× compression)",
        bf16_bytes as f32 / int8_bytes as f32
    );
    println!("  total MSE:   {total_mse:>12.1}  ← outlier-dominated; not the contract");
    println!("  non-outlier MSE: {non_outlier_mse:>6.1}  ← bulk of the 2558 elements");
    println!();
    println!("  ⚠ MSE is not the accuracy contract. The contract is top-1 preservation");
    println!("    under lm_head(final_norm(residual)). See Phase 2 below.");
    println!();

    // ── Phase 2: downstream quality metrics ──────────────────────────────────
    println!("── Phase 2: Downstream quality (what the contract is actually about) ──");
    println!();
    println!("  Exp 43 measured (30 prompts, layer 33, Gemma 3 4B):");
    println!("    int8_clip3σ: top-1=98.7% mean, 93.3% min | top-5=100% | KL=2.0 nats");
    println!("    Accepted at threshold 2.16: 68.9% of boundaries");
    println!();

    use larql_boundary::gate::{apply, BoundaryDecision};

    let config = larql_boundary::BoundaryGateConfig {
        calibration_mode: false,
        min_log_prob_margin: 2.16,
        min_top1_prob: 0.5,
        ..Default::default()
    };

    // Confident boundary demo
    let raw_c = confident_logits_raw(vocab);
    let hat_c = confident_logits_hat(vocab);
    let raw_lp_c = log_softmax(&raw_c);
    let hat_lp_c = log_softmax(&hat_c);
    let mut meta_c = compute(&raw_c, Some(&hat_c));
    let decision_c = apply(&mut meta_c, &config);
    print_accuracy(
        "Confident boundary (~tech_3, margin≈9.0)",
        &meta_c,
        &raw_lp_c,
        &hat_lp_c,
    );
    println!(
        "    boundary_fragile: {}  → gate: {}",
        meta_c.boundary_fragile,
        match &decision_c {
            BoundaryDecision::CompressedOk { .. } => "COMPRESS ✓",
            _ => "bf16 fallback",
        }
    );
    println!();

    // Fragile boundary demo — codec disagrees (hard-reject path)
    let raw_f = fragile_logits_raw(vocab);
    let hat_f = fragile_logits_hat(vocab);
    let raw_lp_f = log_softmax(&raw_f);
    let hat_lp_f = log_softmax(&hat_f);
    let mut meta_f = compute(&raw_f, Some(&hat_f));
    let decision_f = apply(&mut meta_f, &config);
    print_accuracy(
        "Fragile boundary (~qa_2, margin≈0.1, codec disagrees)",
        &meta_f,
        &raw_lp_f,
        &hat_lp_f,
    );
    println!(
        "    codec_fragile: {}  boundary_fragile: {}  → gate: {}",
        meta_f.codec_fragile,
        meta_f.boundary_fragile,
        match &decision_f {
            BoundaryDecision::CompressedOk { .. } => "COMPRESS",
            BoundaryDecision::UseBf16 => "bf16 fallback ✓",
            _ => "other",
        }
    );
    println!();

    // ── Summary ────────────────────────────────────────────────────────────────
    println!("── Summary ──────────────────────────────────────────────────────────");
    println!("  The gate accepts confident boundaries → top-1 preserved, low KL.");
    println!("  The gate rejects fragile boundaries  → bf16 fallback avoids cascade.");
    println!("  MSE of ~300 (non-outlier) does not predict boundary acceptance;");
    println!("  log_prob_margin predicts continuation stability.");
}
