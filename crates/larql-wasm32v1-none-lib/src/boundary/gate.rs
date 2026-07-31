//! Phase 3 — confidence gate.

use super::frame::{BoundaryAgreement, BoundaryContract, FallbackPolicy};
use super::metadata::BoundaryMetadata;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundaryGateConfig {
    pub min_log_prob_margin: f32,
    pub min_top1_prob: f32,
    pub require_compressed_agreement: bool,
    pub fallback_policy: FallbackPolicy,
    pub calibration_mode: bool,
}

impl Default for BoundaryGateConfig {
    fn default() -> Self {
        Self {
            min_log_prob_margin: 1.0,
            min_top1_prob: 0.5,
            require_compressed_agreement: true,
            fallback_policy: FallbackPolicy::Bf16Boundary,
            calibration_mode: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundaryDecision {
    CompressedOk { contract: BoundaryContract },
    UseBf16,
    UseColdReplay,
    Reject,
}

/// Apply the gate to one boundary's metadata.
///
/// Mutates `metadata.boundary_fragile` in place.
pub fn apply(metadata: &mut BoundaryMetadata, config: &BoundaryGateConfig) -> BoundaryDecision {
    if config.calibration_mode {
        metadata.boundary_fragile = is_fragile(metadata, config);
        return BoundaryDecision::UseBf16;
    }

    let agreement_fails = !matches!(metadata.boundary_agreement, BoundaryAgreement::Agrees);
    if config.require_compressed_agreement && agreement_fails {
        metadata.boundary_fragile = false;
        return to_fallback(config);
    }

    let fragile = is_fragile(metadata, config);
    metadata.boundary_fragile = fragile;
    if fragile {
        return to_fallback(config);
    }

    BoundaryDecision::CompressedOk {
        contract: BoundaryContract::ArgmaxNearEquivalentHighMargin,
    }
}

fn is_fragile(meta: &BoundaryMetadata, config: &BoundaryGateConfig) -> bool {
    meta.raw_log_prob_margin < config.min_log_prob_margin
        || meta.raw_top1_prob < config.min_top1_prob
}

fn to_fallback(config: &BoundaryGateConfig) -> BoundaryDecision {
    match config.fallback_policy {
        FallbackPolicy::None | FallbackPolicy::RejectIfUnsafe => BoundaryDecision::Reject,
        FallbackPolicy::Bf16Boundary => BoundaryDecision::UseBf16,
        FallbackPolicy::ColdReplay => BoundaryDecision::UseColdReplay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::frame::BoundaryAgreement;
    use super::super::metadata::BoundaryMetadata;

    fn meta(logit_margin: f32, top1_prob: f32, agreement: BoundaryAgreement) -> BoundaryMetadata {
        BoundaryMetadata {
            raw_top1_token: 42,
            compressed_top1_token: Some(42),
            boundary_agreement: agreement.clone(),
            raw_logit_margin: logit_margin,
            raw_log_prob_margin: logit_margin * 0.9,
            raw_top1_prob: top1_prob,
            codec_fragile: matches!(agreement, BoundaryAgreement::Disagrees),
            boundary_fragile: false,
        }
    }

    fn live() -> BoundaryGateConfig {
        BoundaryGateConfig {
            calibration_mode: false,
            ..Default::default()
        }
    }

    #[test]
    fn calibration_mode_always_bf16() {
        let config = BoundaryGateConfig::default();
        let mut m = meta(10.0, 0.99, BoundaryAgreement::Agrees);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::UseBf16);
    }

    #[test]
    fn disagrees_hard_rejects() {
        let config = live();
        let mut m = meta(5.0, 0.9, BoundaryAgreement::Disagrees);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::UseBf16);
    }

    #[test]
    fn not_checked_hard_rejects() {
        let config = live();
        let mut m = meta(5.0, 0.9, BoundaryAgreement::NotChecked);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::UseBf16);
    }

    #[test]
    fn low_margin_is_boundary_fragile() {
        let mut config = live();
        config.min_log_prob_margin = 2.0;
        let mut m = meta(0.5, 0.9, BoundaryAgreement::Agrees);
        let decision = apply(&mut m, &config);
        assert!(m.boundary_fragile, "expected boundary_fragile = true");
        assert_eq!(decision, BoundaryDecision::UseBf16);
    }

    #[test]
    fn confident_boundary_compresses() {
        let config = live();
        let mut m = meta(3.0, 0.8, BoundaryAgreement::Agrees);
        let decision = apply(&mut m, &config);
        assert!(!m.boundary_fragile);
        assert!(matches!(decision, BoundaryDecision::CompressedOk { .. }));
    }

    #[test]
    fn cold_replay_fallback() {
        let mut config = live();
        config.fallback_policy = FallbackPolicy::ColdReplay;
        let mut m = meta(0.1, 0.3, BoundaryAgreement::Agrees);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::UseColdReplay);
    }

    #[test]
    fn reject_if_unsafe_fallback() {
        let mut config = live();
        config.fallback_policy = FallbackPolicy::RejectIfUnsafe;
        let mut m = meta(0.1, 0.3, BoundaryAgreement::Agrees);
        assert_eq!(apply(&mut m, &config), BoundaryDecision::Reject);
    }
}
