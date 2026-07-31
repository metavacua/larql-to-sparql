//! Wire format types for BOUNDARY ref frames.

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum BoundaryCompression {
    None,
    Int8Clip3Sigma,
    Int8Absmax,
    Int4Clip3Sigma,
}

impl BoundaryCompression {
    pub fn is_compressed(&self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum BoundaryContract {
    Exact,
    DistributionFaithful,
    DistributionSimilar,
    ArgmaxEquivalent,
    ArgmaxNearEquivalentHighMargin,
    ArgmaxNearEquivalentLowMargin,
    CandidateSet,
    Calibrating,
    Unknown,
}

impl BoundaryContract {
    pub fn is_safe_for_continuation(&self) -> bool {
        matches!(
            self,
            Self::Exact
                | Self::DistributionFaithful
                | Self::DistributionSimilar
                | Self::ArgmaxEquivalent
                | Self::ArgmaxNearEquivalentHighMargin
        )
    }

    pub fn is_safe_for_routing(&self) -> bool {
        matches!(
            self,
            Self::Exact
                | Self::DistributionFaithful
                | Self::DistributionSimilar
                | Self::ArgmaxEquivalent
                | Self::ArgmaxNearEquivalentHighMargin
                | Self::ArgmaxNearEquivalentLowMargin
                | Self::CandidateSet
        )
    }
}

/// Tri-state result of the sender's compressed-vs-raw agreement check.
///
/// `NotChecked` must be treated as `Disagrees` by receivers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum BoundaryAgreement {
    NotChecked,
    Agrees,
    Disagrees,
}

impl BoundaryAgreement {
    pub fn is_hard_reject(&self) -> bool {
        matches!(self, Self::Disagrees | Self::NotChecked)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum FallbackPolicy {
    None,
    Bf16Boundary,
    ColdReplay,
    RejectIfUnsafe,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundaryFrame {
    pub version: u16,
    pub model_id: String,
    pub model_revision: String,
    pub tokenizer_revision: String,
    pub architecture: String,
    pub boundary_id: String,
    pub sequence_id: String,
    pub token_start: u64,
    pub token_end: u64,
    pub layer: u16,
    pub hidden_size: u32,
    pub compression_scheme: BoundaryCompression,
    pub contract_level: BoundaryContract,
    pub payload: Vec<u8>,
    pub raw_top1_token: u32,
    pub raw_logit_margin: f32,
    pub raw_top1_prob: Option<f32>,
    pub compressed_top1_token: Option<u32>,
    pub boundary_agreement: BoundaryAgreement,
    pub codec_fragile: bool,
    pub boundary_fragile: bool,
    pub fallback_policy: FallbackPolicy,
    pub fallback_ref: Option<String>,
    pub calibration_run_id: Option<String>,
    pub residual_hash: Option<[u8; 32]>,
    pub token_hash: Option<[u8; 32]>,
}

impl BoundaryFrame {
    pub fn is_compressed(&self) -> bool {
        self.compression_scheme.is_compressed()
    }

    pub fn is_safe_for_continuation(&self) -> bool {
        if matches!(self.contract_level, BoundaryContract::Calibrating) {
            return false;
        }
        if self.codec_fragile || self.boundary_agreement.is_hard_reject() {
            return false;
        }
        self.contract_level.is_safe_for_continuation()
    }

    pub fn is_safe_for_routing(&self) -> bool {
        if matches!(self.contract_level, BoundaryContract::Calibrating) {
            return false;
        }
        if self.codec_fragile {
            return false;
        }
        if self.is_compressed() && self.boundary_agreement.is_hard_reject() {
            return false;
        }
        self.contract_level.is_safe_for_routing()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_frame() -> BoundaryFrame {
        BoundaryFrame {
            version: 1,
            model_id: "test".into(),
            model_revision: "abc".into(),
            tokenizer_revision: "def".into(),
            architecture: "test-arch".into(),
            boundary_id: "b0".into(),
            sequence_id: "s0".into(),
            token_start: 0,
            token_end: 512,
            layer: 33,
            hidden_size: 2560,
            compression_scheme: BoundaryCompression::None,
            contract_level: BoundaryContract::Exact,
            payload: alloc::vec![],
            raw_top1_token: 42,
            raw_logit_margin: 5.0,
            raw_top1_prob: Some(0.9),
            compressed_top1_token: None,
            boundary_agreement: BoundaryAgreement::NotChecked,
            codec_fragile: false,
            boundary_fragile: false,
            fallback_policy: FallbackPolicy::Bf16Boundary,
            fallback_ref: None,
            calibration_run_id: None,
            residual_hash: None,
            token_hash: None,
        }
    }

    #[test]
    fn continuation_safety_rejects_calibrating() {
        let mut frame = minimal_frame();
        frame.contract_level = BoundaryContract::Calibrating;
        assert!(!frame.is_safe_for_continuation());
        assert!(!frame.is_safe_for_routing());
    }

    #[test]
    fn continuation_safety_rejects_not_checked_compressed() {
        let mut frame = minimal_frame();
        frame.compression_scheme = BoundaryCompression::Int8Clip3Sigma;
        frame.boundary_agreement = BoundaryAgreement::NotChecked;
        assert!(!frame.is_safe_for_routing());
    }

    #[test]
    fn continuation_safety_accepts_highmargin() {
        let mut frame = minimal_frame();
        frame.contract_level = BoundaryContract::ArgmaxNearEquivalentHighMargin;
        frame.boundary_agreement = BoundaryAgreement::Agrees;
        assert!(frame.is_safe_for_continuation());
        assert!(frame.is_safe_for_routing());
    }

    #[test]
    fn routing_accepts_lowmargin_rejects_for_continuation() {
        let mut frame = minimal_frame();
        frame.contract_level = BoundaryContract::ArgmaxNearEquivalentLowMargin;
        frame.boundary_agreement = BoundaryAgreement::Agrees;
        assert!(!frame.is_safe_for_continuation());
        assert!(frame.is_safe_for_routing());
    }
}
