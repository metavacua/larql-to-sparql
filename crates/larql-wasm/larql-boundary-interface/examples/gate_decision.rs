//! Show gate decisions for four boundary types matching Exp 43 continuation tests.
//!
//! Run with:
//!   cargo run -p larql-boundary --example gate_decision

use larql_boundary::{
    gate::{BoundaryDecision, BoundaryGateConfig},
    metadata::{compute, BoundaryMetadata},
};

fn peaked_logits(top: usize, v_top: f32, v_second: f32, v_rest: f32, n: usize) -> Vec<f32> {
    let mut l = vec![v_rest; n];
    l[top] = v_top;
    if top > 0 {
        l[top - 1] = v_second;
    }
    l
}

fn print_decision(label: &str, meta: &mut BoundaryMetadata, config: &BoundaryGateConfig) {
    let decision = larql_boundary::gate::apply(meta, config);
    let symbol = match &decision {
        BoundaryDecision::CompressedOk { .. } => "✓  compressed",
        BoundaryDecision::UseBf16 => "~  bf16 fallback",
        BoundaryDecision::UseColdReplay => "↩  cold replay",
        BoundaryDecision::Reject => "✗  reject",
    };
    println!(
        "  {label:<25}  logit_margin={:.2}  top1_prob={:.3}  \
         agreement={:<12}  fragile={}  →  {symbol}",
        meta.raw_logit_margin,
        meta.raw_top1_prob,
        format!("{:?}", meta.boundary_agreement),
        meta.boundary_fragile,
    );
}

fn main() {
    const VOCAB: usize = 10_000;

    let config = BoundaryGateConfig {
        calibration_mode: false,
        min_log_prob_margin: 1.0,
        min_top1_prob: 0.5,
        ..Default::default()
    };

    println!(
        "Gate config: min_log_prob_margin={:.1}  min_top1_prob={:.1}",
        config.min_log_prob_margin, config.min_top1_prob
    );
    println!("(thresholds are UNCALIBRATED placeholders; Exp 44 Track A will fit real values)");
    println!();

    // tech_3 pattern: confident boundary, codec agrees → compresses
    let raw_confident = peaked_logits(42, 10.0, 1.0, 0.0, VOCAB);
    let mut meta_confident = compute(&raw_confident, Some(&raw_confident));
    print_decision("tech_3 (confident)", &mut meta_confident, &config);

    // qa_2 pattern: low-margin boundary, codec agrees → boundary_fragile
    let raw_fragile = peaked_logits(42, 1.1, 1.0, 0.9, VOCAB); // top-2 very close
    let mut meta_fragile = compute(&raw_fragile, Some(&raw_fragile));
    print_decision("qa_2 (fragile)", &mut meta_fragile, &config);

    // Codec disagrees (hard reject regardless of margin)
    let raw_disagree = peaked_logits(42, 10.0, 1.0, 0.0, VOCAB);
    let hat_disagree = peaked_logits(99, 10.0, 1.0, 0.0, VOCAB);
    let mut meta_disagree = compute(&raw_disagree, Some(&hat_disagree));
    print_decision("codec disagrees", &mut meta_disagree, &config);

    // NotChecked (sender skipped agreement check — treated as hard reject)
    let mut meta_not_checked = compute(&raw_confident, None);
    print_decision("not checked", &mut meta_not_checked, &config);

    println!();
    println!("Calibration mode (default — always bf16 until Exp 44):");
    let calib_config = BoundaryGateConfig::default(); // calibration_mode = true
    let mut m = compute(&raw_confident, Some(&raw_confident));
    print_decision("confident (calib mode)", &mut m, &calib_config);
}
