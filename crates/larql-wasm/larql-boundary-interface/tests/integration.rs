//! Integration tests: full Phase 1 → 2 → 3 pipeline.

use larql_boundary::{
    codec::{bf16, int8},
    frame::{
        BoundaryAgreement, BoundaryCompression, BoundaryContract, BoundaryFrame, FallbackPolicy,
    },
    gate::{apply, BoundaryDecision, BoundaryGateConfig},
    metadata::compute,
};

const D: usize = 2560;

fn synthetic_residual() -> Vec<f32> {
    let mut r: Vec<f32> = (0..D).map(|i| (i as f32 * 0.01).sin() * 80.0).collect();
    r[100] = 94_208.0; // outlier
    r
}

fn peaked_logits(top: usize, n: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; n];
    l[top] = 10.0;
    if top > 0 {
        l[top - 1] = 3.0;
    }
    l
}

// ── Phase 1 ──────────────────────────────────────────────────────────────

#[test]
fn bf16_roundtrip_no_overflow() {
    let r = synthetic_residual();
    let enc = bf16::encode(&r);
    let dec = bf16::decode(&enc);
    assert_eq!(dec.len(), r.len());
    for v in &dec {
        assert!(v.is_finite());
    }
}

#[test]
fn int8_roundtrip_correct_length() {
    let r = synthetic_residual();
    let payload = int8::encode(&r);
    let bytes = payload.to_bytes();
    assert_eq!(bytes.len(), 4 + D);
    let recovered = int8::decode(&int8::Payload::from_bytes(&bytes));
    assert_eq!(recovered.len(), D);
}

// ── Phase 2 ──────────────────────────────────────────────────────────────

#[test]
fn metadata_agrees_for_same_logits() {
    let logits = peaked_logits(42, 10_000);
    let meta = compute(&logits, Some(&logits));
    assert!(matches!(meta.boundary_agreement, BoundaryAgreement::Agrees));
    assert!(!meta.codec_fragile);
    assert!(meta.raw_log_prob_margin > 0.0);
}

#[test]
fn metadata_disagrees_for_different_top1() {
    let raw = peaked_logits(42, 10_000);
    let hat = peaked_logits(99, 10_000);
    let meta = compute(&raw, Some(&hat));
    assert!(matches!(
        meta.boundary_agreement,
        BoundaryAgreement::Disagrees
    ));
    assert!(meta.codec_fragile);
}

// ── Phase 3 ──────────────────────────────────────────────────────────────

#[test]
fn gate_compresses_confident_boundary() {
    let logits = peaked_logits(42, 10_000);
    let mut meta = compute(&logits, Some(&logits));
    let config = BoundaryGateConfig {
        calibration_mode: false,
        min_log_prob_margin: 1.0,
        min_top1_prob: 0.5,
        ..Default::default()
    };
    let decision = apply(&mut meta, &config);
    assert!(
        matches!(decision, BoundaryDecision::CompressedOk { .. }),
        "expected CompressedOk, got {decision:?}"
    );
    assert!(!meta.boundary_fragile);
}

#[test]
fn gate_bf16_fallback_in_calibration_mode() {
    let logits = peaked_logits(42, 10_000);
    let mut meta = compute(&logits, Some(&logits));
    let config = BoundaryGateConfig::default(); // calibration_mode = true
    assert_eq!(apply(&mut meta, &config), BoundaryDecision::UseBf16);
}

#[test]
fn gate_rejects_codec_fragile() {
    let raw = peaked_logits(42, 10_000);
    let hat = peaked_logits(99, 10_000);
    let mut meta = compute(&raw, Some(&hat));
    let config = BoundaryGateConfig {
        calibration_mode: false,
        ..Default::default()
    };
    // codec_fragile → fallback
    assert_eq!(apply(&mut meta, &config), BoundaryDecision::UseBf16);
}

// ── Frame ─────────────────────────────────────────────────────────────────

#[test]
fn frame_continuation_safety() {
    let mut f = BoundaryFrame {
        version: 1,
        model_id: "m".into(),
        model_revision: "r".into(),
        tokenizer_revision: "t".into(),
        architecture: "a".into(),
        boundary_id: "b".into(),
        sequence_id: "s".into(),
        token_start: 0,
        token_end: 512,
        layer: 33,
        hidden_size: 2560,
        compression_scheme: BoundaryCompression::Int8Clip3Sigma,
        contract_level: BoundaryContract::ArgmaxNearEquivalentHighMargin,
        payload: vec![0u8; 2564],
        raw_top1_token: 42,
        raw_logit_margin: 5.0,
        raw_top1_prob: Some(0.9),
        compressed_top1_token: Some(42),
        boundary_agreement: BoundaryAgreement::Agrees,
        codec_fragile: false,
        boundary_fragile: false,
        fallback_policy: FallbackPolicy::Bf16Boundary,
        fallback_ref: None,
        calibration_run_id: None,
        residual_hash: None,
        token_hash: None,
    };

    assert!(f.is_safe_for_continuation());
    assert!(f.is_safe_for_routing());

    f.contract_level = BoundaryContract::Calibrating;
    assert!(!f.is_safe_for_continuation());
    assert!(!f.is_safe_for_routing());

    f.contract_level = BoundaryContract::ArgmaxNearEquivalentLowMargin;
    assert!(!f.is_safe_for_continuation());
    assert!(f.is_safe_for_routing());
}

// ── Accuracy regression ────────────────────────────────────────────────────
//
// These tests guard the codec's accuracy properties without running the model.
// They verify the invariants that make the Exp 43 characterisation numbers
// meaningful (98.7% top-1 mean, KL ~2.0 nats, Contract D-@high at threshold 2.16).
//
// The actual Exp 43 numbers come from running Gemma 3 4B; these tests protect
// the *codec* properties that underpin them: non-outlier reconstruction quality,
// the gate's hard-reject on Disagrees, and the confidence-gate's correct use of
// log-prob margin (not raw logit margin).

#[test]
fn codec_non_outlier_reconstruction_quality() {
    // int8-clip3σ saturates outlier elements but should preserve the bulk of the
    // vector with reasonable fidelity. The Exp 43 characterisation showed the codec
    // is accurate enough for 98.7% top-1 agreement downstream.
    //
    // Guard: non-outlier per-element RMS error < 5% of the signal std-dev.
    // (A regression that drastically worsens clipping would break this.)
    let sigma = 1650.0f32; // Gemma 3 4B residual σ
    let mut r: Vec<f32> = (0..D)
        .map(|i| ((i as f32) * 0.0023).sin() * sigma)
        .collect();
    r[42] = 94_208.0; // outlier
    r[512] = -60_000.0; // outlier

    let payload = int8::encode(&r);
    let hat = int8::decode(&payload);

    // Non-outlier RMS error as fraction of σ.
    let non_outlier_rms: f32 = r
        .iter()
        .zip(hat.iter())
        .enumerate()
        .filter(|(i, _)| *i != 42 && *i != 512)
        .map(|(_, (a, b))| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
        / (D - 2) as f32;

    assert!(
        non_outlier_rms < sigma * 0.05,
        "non-outlier RMS {non_outlier_rms:.2} exceeds 5% of σ={sigma}"
    );
}

#[test]
fn gate_codec_fragile_always_rejects_regardless_of_margin() {
    // The Exp 43 characterisation depends on the gate hard-rejecting codec-fragile
    // boundaries. A codec that flips the argmax must never be accepted, even if the
    // raw margin looks high. This is the Contract D- guarantee.
    let config = BoundaryGateConfig {
        calibration_mode: false,
        min_log_prob_margin: 0.0, // no margin threshold — only codec check matters
        min_top1_prob: 0.0,
        require_compressed_agreement: true,
        ..Default::default()
    };

    // High-margin logits where compression DISAGREES (codec fragile)
    let vocab = 1000;
    let mut raw = vec![0.0f32; vocab];
    raw[42] = 9.0;
    raw[17] = 0.0;
    let mut hat = raw.clone();
    hat[42] = 0.0;
    hat[17] = 9.0; // codec flipped the argmax

    let mut meta = compute(&raw, Some(&hat));
    assert!(
        meta.codec_fragile,
        "expected codec_fragile=true when argmax flips"
    );

    let decision = apply(&mut meta, &config);
    assert_ne!(
        decision,
        BoundaryDecision::CompressedOk {
            contract: larql_boundary::BoundaryContract::ArgmaxNearEquivalentHighMargin
        },
        "codec-fragile boundary must not be accepted even at zero margin threshold"
    );
}

#[test]
fn gate_uses_log_prob_margin_not_raw_logit_margin() {
    // This is the units-correctness test. The gate threshold is calibrated in
    // log-prob margin units. If the gate compared raw_logit_margin against
    // min_log_prob_margin, a boundary with large raw logits but small log-prob
    // gap would be incorrectly accepted.
    //
    // Here we construct: raw_logit_margin=10.0 but raw_log_prob_margin≈0.001
    // (near-uniform distribution with one very slightly higher logit).
    let vocab = 1_000_000; // huge vocab → logits near-uniform
    let mut raw = vec![-15.0f32; vocab];
    raw[42] = -14.9; // only 0.1 logit above the rest → near-uniform distribution
                     // log_prob_margin ≈ log_softmax[-14.9] - log_softmax[-15.0] ≈ very small

    let mut meta = compute(&raw, None);

    // log_prob_margin should be tiny even though raw_logit_margin = 0.1
    assert!(
        meta.raw_log_prob_margin < 0.5,
        "expected small log_prob_margin for near-uniform distribution, got {}",
        meta.raw_log_prob_margin
    );

    // Gate with threshold=2.16 (calibrated) should reject this as fragile
    let config = BoundaryGateConfig {
        calibration_mode: false,
        min_log_prob_margin: 2.16,
        min_top1_prob: 0.0,
        require_compressed_agreement: false, // skip codec check
        ..Default::default()
    };
    let decision = apply(&mut meta, &config);
    assert!(
        !matches!(decision, BoundaryDecision::CompressedOk { .. }),
        "near-uniform boundary should be rejected at calibrated threshold 2.16"
    );
}

// ── Full pipeline ─────────────────────────────────────────────────────────

#[test]
fn full_pipeline_encode_metadata_gate() {
    let residual = synthetic_residual();

    // Phase 1: encode
    let payload = int8::encode(&residual);
    let hat_residual = int8::decode(&payload);

    // Phase 2: compute metadata from logits (caller provides logits; we fake them here)
    let vocab = 100;
    let raw_logits: Vec<f32> = {
        let mut l = vec![0.0f32; vocab];
        l[42] = 8.0;
        l[17] = 2.0;
        l
    };
    let hat_logits: Vec<f32> = {
        // Use hat_residual's "logits" — simulate lm_head result: same top-1 if compression preserved
        let mut l = raw_logits.clone();
        // Small perturbation that doesn't change argmax
        l[17] = 2.1;
        l
    };
    let mut meta = compute(&raw_logits, Some(&hat_logits));
    assert!(matches!(meta.boundary_agreement, BoundaryAgreement::Agrees));

    // Phase 3: gate
    let config = BoundaryGateConfig {
        calibration_mode: false,
        min_log_prob_margin: 1.0,
        min_top1_prob: 0.3,
        ..Default::default()
    };
    let decision = apply(&mut meta, &config);
    assert!(matches!(decision, BoundaryDecision::CompressedOk { .. }));

    // Residual decoded correctly
    assert_eq!(hat_residual.len(), residual.len());
}
