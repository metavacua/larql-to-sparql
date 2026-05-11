//! Wire format types for BOUNDARY ref frames.
//!
//! These enums and structs map directly to the protocol spec in
//! `~/chris-source/chris-experiments/shannon/43_residual_stream_codec/BOUNDARY_REF_PROTOCOL.md`.
//!
//! Full serialisation (protobuf, HTTP JSON) is handled by `larql-server`.
//! This module defines the canonical Rust representation.

// ── Enums ─────────────────────────────────────────────────────────────────

/// Compression scheme applied to the residual payload.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryCompression {
    /// Raw bfloat16; no information loss. See [`codec::bf16`].
    None,
    /// Per-vector 3σ clipping + int8. Contract D-. See [`codec::int8`].
    Int8Clip3Sigma,
    /// Per-vector absmax int8. Contract C (distribution-similar). Experimental.
    Int8Absmax,
    /// Per-vector 3σ clipping + int4. Contract E. Experimental.
    Int4Clip3Sigma,
}

impl BoundaryCompression {
    pub fn is_compressed(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Behavioural contract on the boundary frame.
///
/// Defined by the (KL, top-1, top-5) criteria in the protocol spec §6.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryContract {
    /// A: bf16 reference; KL=0, top-1=100%.
    Exact,
    /// B: KL < 0.05, top-1 ≥ 99%.
    DistributionFaithful,
    /// C: KL < 0.5, top-5 ≥ 95%.
    DistributionSimilar,
    /// D: top-1 ≥ 99%.
    ArgmaxEquivalent,
    /// D-@high: margin ≥ gate threshold; continuation-safe.
    ArgmaxNearEquivalentHighMargin,
    /// D-@low: margin < gate threshold; single-token routing only.
    ArgmaxNearEquivalentLowMargin,
    /// E: top-5 ≥ 90%; candidate generation / reranking.
    CandidateSet,
    /// Emitted during `calibration_mode = true`. Thresholds were not yet
    /// fitted when this frame was written. Must not cross trust boundaries.
    Calibrating,
    Unknown,
}

impl BoundaryContract {
    /// True iff this contract level is suitable for long greedy continuation.
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

    /// True iff this contract level is suitable for single-token routing.
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
/// `NotChecked` must be treated as `Disagrees` by receivers. See §10.7.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryAgreement {
    /// Sender did not run the compressed forward pass.
    /// Receiver must treat this identically to `Disagrees`.
    NotChecked,
    /// `argmax(hat_logits) == argmax(raw_logits)`.
    Agrees,
    /// `argmax(hat_logits) != argmax(raw_logits)`. Hard reject.
    Disagrees,
}

impl BoundaryAgreement {
    /// True iff the receiver should hard-reject a compressed frame with this
    /// agreement value (i.e., the codec changed or did not check the argmax).
    pub fn is_hard_reject(&self) -> bool {
        matches!(self, Self::Disagrees | Self::NotChecked)
    }
}

/// What the sender wants the receiver to do when the compressed frame is
/// rejected or unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackPolicy {
    None,
    Bf16Boundary,
    ColdReplay,
    RejectIfUnsafe,
}

// ── BoundaryFrame ──────────────────────────────────────────────────────────

/// The BOUNDARY ref frame: residual checkpoint + contract + confidence metadata.
///
/// Carries everything a receiver needs to decide whether to accept the
/// compressed state and continue from it. See the protocol spec for field
/// semantics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoundaryFrame {
    pub version: u16,

    // ── Model identity ──────────────────────────────────────────────────────
    pub model_id: String,
    /// Content hash of the weights. Canonical identity for mismatch checks.
    pub model_revision: String,
    /// Hash of the tokenizer config + vocab. Versioned independently from
    /// model weights. Required for cold-replay `token_hash` lookup.
    pub tokenizer_revision: String,
    /// Human-readable label ("gemma3-4b"). Not used for matching.
    pub architecture: String,

    // ── Boundary location ───────────────────────────────────────────────────
    pub boundary_id: String,
    pub sequence_id: String,
    pub token_start: u64,
    pub token_end: u64,
    pub layer: u16,
    pub hidden_size: u32,

    // ── Compression ─────────────────────────────────────────────────────────
    pub compression_scheme: BoundaryCompression,
    pub contract_level: BoundaryContract,

    // ── Payload ─────────────────────────────────────────────────────────────
    pub payload: Vec<u8>,

    // ── Confidence metadata ─────────────────────────────────────────────────
    pub raw_top1_token: u32,
    /// Primary gate input: `logits[top1] - logits[top2]` in raw logit units.
    pub raw_logit_margin: f32,
    /// Secondary gate input. Also used for human-readable diagnostics.
    pub raw_top1_prob: Option<f32>,
    pub compressed_top1_token: Option<u32>,
    pub boundary_agreement: BoundaryAgreement,
    /// Hard-reject signal: `boundary_agreement == Disagrees`.
    pub codec_fragile: bool,
    /// Soft-warning signal: margin below calibrated threshold.
    pub boundary_fragile: bool,

    // ── Fallback ─────────────────────────────────────────────────────────────
    pub fallback_policy: FallbackPolicy,
    /// v0.1: always `None`. Protocol TBD (v0.2 cold-replay sketch in spec §3).
    pub fallback_ref: Option<String>,

    // ── Calibration identity ─────────────────────────────────────────────────
    /// Present when `calibration_mode = true`. Part of the contract for frames
    /// that cross trust boundaries during calibration. Receivers must hard-reject
    /// frames with unknown or mismatched `calibration_run_id`. See spec §10.6.
    pub calibration_run_id: Option<String>,

    // ── Integrity ─────────────────────────────────────────────────────────────
    /// SHA-256 of the uncompressed bf16 payload. Tamper detection and
    /// cross-session deduplication.
    pub residual_hash: Option<[u8; 32]>,
    /// SHA-256 of source token IDs `[token_start..token_end]`.
    /// Only valid for `tokenizer_revision` that produced the tokens.
    /// Cold-replay lookup key (v0.2).
    pub token_hash: Option<[u8; 32]>,
}

impl BoundaryFrame {
    /// Whether this frame carries a compressed residual (vs raw bf16).
    pub fn is_compressed(&self) -> bool {
        self.compression_scheme.is_compressed()
    }

    /// Whether this frame is safe for long greedy continuation.
    pub fn is_safe_for_continuation(&self) -> bool {
        if matches!(self.contract_level, BoundaryContract::Calibrating) {
            return false;
        }
        if self.codec_fragile || self.boundary_agreement.is_hard_reject() {
            return false;
        }
        self.contract_level.is_safe_for_continuation()
    }

    /// Whether this frame is safe for single-token routing.
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
            payload: vec![],
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
}
