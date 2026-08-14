//! Shared fixtures for the integration gates.

use std::sync::Mutex;

use larql_compute::ffn::WeightFfn;
use larql_inference::ffn::FfnBackend;
use larql_inference::model::ModelWeights;
use ndarray::Array2;

use crate::engines::semantic_promotion::*;

/// Token span the fixtures plant a source authority over.
pub const SPAN_START: u64 = 4;
pub const SPAN_TOKENS: [u32; 4] = [7, 4, 3, 1];
/// Payload length of the fixture answer, in tokens.
pub const PAYLOAD_TOKENS: u32 = 4;

/// EXP-25 per-position KL traces, in bits, at 65K distance.
pub mod exp25 {
    /// `copy` / canonical — clean throughout; peak 0.0441 at the first
    /// termination token, inside the 0.05 tolerance.
    pub const COPY_CANONICAL: &[f32] = &[0.0001, 0.0, 0.0, 0.0, 0.0441, 0.0, 0.0, 0.0004];
    /// `transform_inc` / canonical — payload under 0.001, then 0.1472 on
    /// the first termination token.
    pub const INC_CANONICAL: &[f32] = &[0.0008, 0.0003, 0.0001, 0.001, 0.1472, 0.0044];
    /// `transform_rev` / canonical — payload under 0.023, then 0.0610.
    pub const REV_CANONICAL: &[f32] = &[0.0223, 0.0061, 0.0, 0.0028, 0.061, 0.0107];
    /// `transform_inc` / derived — clean; peak 0.0218 past termination.
    pub const INC_DERIVED: &[f32] = &[0.0001, 0.0, 0.0, 0.0008, 0.0003, 0.0117, 0.0001, 0.0218];
    /// `transform_rev` / derived — 14.0197 on the first payload token.
    pub const REV_DERIVED: &[f32] = &[14.0197, 0.007, 0.0, 0.0012, 0.0313, 0.273];
}

/// Score a KL trace into metrics at the default tolerance.
///
/// Top-1 equality is derived from the bits here purely as a fixture
/// convenience; production callers pass measured argmax agreement.
pub fn metrics_from(kls: &[f32]) -> QualificationMetrics {
    let trace: Vec<_> = kls
        .iter()
        .map(|&kl_bits| PositionScore {
            kl_bits,
            top1_equal: kl_bits < 1.0,
        })
        .collect();
    score_trajectory(&trace, PAYLOAD_TOKENS, DEFAULT_KL_TOLERANCE_BITS).unwrap()
}

pub fn regime() -> RegimeFingerprint {
    RegimeFingerprint::new(
        ModelFingerprint::new("synthetic-test-weights"),
        RuntimeFingerprint::new("semantic-promotion(standard)/cpu"),
        PromptProtocolId::new("terse-number-v1"),
        PlacementProtocolId::new("record-at-question-boundary"),
    )
}

pub fn signature(operator: &str) -> OperationSignature {
    OperationSignature::new(operator, ["operand"], "integer", 1)
}

/// A source authority in `engine`'s current session epoch. Reading the
/// epoch off the engine rather than hardcoding it keeps the fixture
/// honest across re-prefills, which bump the epoch.
pub fn source_for(engine: &SemanticPromotionEngine) -> SourceAuthority {
    SourceAuthority::from_tokens(engine.session_epoch(), SPAN_START, &SPAN_TOKENS)
}

pub fn canonical_fact(provenance: Vec<AuthorityId>) -> SemanticRecord {
    SemanticRecord::CanonicalFact(CanonicalFact {
        // Overwritten by the graph on registration.
        id: RecordId::from_counter(0),
        subject: SemanticAtom::new("Meridian"),
        relation: SemanticAtom::new("code"),
        value: SemanticValue::new("integer", "7431"),
        provenance,
        schema_version: 1,
    })
}

pub fn operation(id: OperationId, operator: &str, operands: Vec<RecordId>) -> OperationSpec {
    OperationSpec {
        id,
        class: OperationClass::Transform,
        operands,
        answer_boundary: AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS),
        signature: signature(operator),
    }
}

/// A wrapped `StandardEngine` in the default (`Observe`) configuration.
pub fn wrapped_standard() -> SemanticPromotionEngine {
    let base = Box::new(crate::engines::standard::StandardEngine::with_backend(
        None,
        larql_inference::cpu_engine_backend(),
    ));
    SemanticPromotionEngine::new(base, SemanticPromotionConfig::default()).unwrap()
}

pub fn bare_standard() -> crate::engines::standard::StandardEngine {
    crate::engines::standard::StandardEngine::with_backend(
        None,
        larql_inference::cpu_engine_backend(),
    )
}

/// FFN backend that records the layer sequence it was asked for, so the
/// dispatch gate can compare wrapped and unwrapped call streams.
pub struct RecordingFfn<'a> {
    inner: WeightFfn<'a>,
    calls: Mutex<Vec<usize>>,
}

impl<'a> RecordingFfn<'a> {
    pub fn new(weights: &'a ModelWeights) -> Self {
        Self {
            inner: WeightFfn { weights },
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<usize> {
        self.calls.lock().expect("recording ffn poisoned").clone()
    }
}

impl FfnBackend for RecordingFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.calls
            .lock()
            .expect("recording ffn poisoned")
            .push(layer);
        self.inner.forward(layer, x)
    }

    fn name(&self) -> &str {
        "recording"
    }
}
