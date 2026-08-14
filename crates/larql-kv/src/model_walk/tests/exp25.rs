//! EXP-25 encoded as graphs the planner walks.
//!
//! Each builder registers the same objects in the engine and in the walk
//! graph, so a planned walk can actually be executed. The certificates
//! carry the measured per-position KL traces, so what the planner is
//! allowed to emit is decided by the experiment, not by a fixture
//! constant.

use crate::model_walk::{ModelEdge, ModelGraph, ModelNode};
use crate::semantic_promotion::*;

/// Logical span the source authority occupies in the fixture session.
pub const SPAN_START: u64 = 4;
pub const SPAN_TOKENS: [u32; 4] = [7, 4, 3, 1];
/// Payload length of every EXP-25 answer, in tokens.
pub const PAYLOAD_TOKENS: u32 = 4;
/// Tokens the promoted record renders to.
pub const RECORD_TOKENS: [u32; 3] = [10, 11, 12];

/// Measured per-position KL bits at 65K distance (EXP-25).
pub mod kl {
    /// `copy` / canonical — clean; peak 0.0441 at first termination.
    pub const COPY_CANONICAL: &[f32] = &[0.0001, 0.0, 0.0, 0.0, 0.0441, 0.0, 0.0, 0.0004];
    /// `transform_inc` / canonical — 0.1472 at first termination.
    pub const INC_CANONICAL: &[f32] = &[0.0008, 0.0003, 0.0001, 0.001, 0.1472, 0.0044];
    /// `transform_rev` / canonical — 0.0610 at first termination.
    pub const REV_CANONICAL: &[f32] = &[0.0223, 0.0061, 0.0, 0.0028, 0.061, 0.0107];
    /// `transform_inc` / derived — clean; peak 0.0218 past termination.
    pub const INC_DERIVED: &[f32] = &[0.0001, 0.0, 0.0, 0.0008, 0.0003, 0.0117, 0.0001, 0.0218];
    /// `transform_rev` / derived — 14.0197 on the first payload token.
    pub const REV_DERIVED: &[f32] = &[14.0197, 0.007, 0.0, 0.0012, 0.0313, 0.273];
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

fn metrics(kls: &[f32]) -> QualificationMetrics {
    let trace: Vec<_> = kls
        .iter()
        .map(|&kl_bits| PositionScore {
            kl_bits,
            top1_equal: kl_bits < 1.0,
        })
        .collect();
    score_trajectory(&trace, PAYLOAD_TOKENS, DEFAULT_KL_TOLERANCE_BITS).unwrap()
}

/// Build a certificate whose scored object is the strongest the numbers
/// actually support — trajectory if they clear it, payload otherwise.
/// Returns `None` when the numbers support neither, which is EXP-25's
/// `transform_rev` / derived row.
///
/// `consumption` is a separate axis and is never inferred from the KLs.
/// EXP-25CF found a record that *asserts* a computed result being read as
/// a fresh operand in both histories, so a certificate has to say which
/// the model actually did.
#[allow(clippy::too_many_arguments)]
pub fn certificate_from_measurement(
    id: QualificationId,
    record: RecordId,
    record_digest: [u8; 32],
    operator: &str,
    kls: &[f32],
    consumption: ConsumptionMode,
) -> Option<QualificationCertificate> {
    let m = metrics(kls);
    let build = |scored_object| QualificationCertificate {
        id,
        key: CapabilityKey::for_operation(
            &regime(),
            record_digest,
            1,
            MaterialisationKind::TokenSequence,
            1,
            RecordMaterialisation::token_sequence(RECORD_TOKENS.to_vec()).content_digest(),
            signature(operator),
        ),
        record,
        scored_object,
        metrics: m,
        tolerance_bits: DEFAULT_KL_TOLERANCE_BITS,
        established_over: RegimeEvidence::distance_invariant(),
        consumption,
        general_state_evidence: None,
    };
    for object in [
        ScoredObject::ReachableTrajectory {
            payload_tokens: PAYLOAD_TOKENS,
        },
        ScoredObject::PayloadOnly {
            payload_tokens: PAYLOAD_TOKENS,
        },
    ] {
        if m.qualifies(object, DEFAULT_KL_TOLERANCE_BITS) {
            return Some(build(object));
        }
    }
    None
}

/// A session with a source, a canonical record and (optionally) a
/// derived record, plus a walk graph describing the same objects.
pub struct Fixture {
    pub engine: SemanticPromotionEngine,
    pub graph: ModelGraph,
    pub authority: AuthorityId,
    pub canonical: RecordId,
    pub derived: Option<RecordId>,
    pub operation: OperationId,
}

/// Which record forms the fixture should register.
#[derive(Clone, Copy, PartialEq)]
pub struct Forms {
    pub canonical: bool,
    pub derived: bool,
}

impl Forms {
    pub const CANONICAL_ONLY: Self = Self {
        canonical: true,
        derived: false,
    };
    pub const BOTH: Self = Self {
        canonical: true,
        derived: true,
    };
    pub const DERIVED_ONLY: Self = Self {
        canonical: false,
        derived: true,
    };
}

fn engine() -> SemanticPromotionEngine {
    engine_with(
        SemanticPromotionConfig::default(),
        PromotionEngineCapabilities::ffn_only(),
    )
}

/// An engine with an explicit config and capability declaration.
pub fn engine_with(
    config: SemanticPromotionConfig,
    capabilities: PromotionEngineCapabilities,
) -> SemanticPromotionEngine {
    let base = Box::new(crate::engines::standard::StandardEngine::with_backend(
        None,
        larql_inference::cpu_engine_backend(),
    ));
    let mut engine =
        SemanticPromotionEngine::with_capabilities(base, config, capabilities).unwrap();
    engine.set_regime(regime());
    engine
}

/// Assemble the fixture for one EXP-25 question.
///
/// `canonical_kls` / `derived_kls` are the measured traces for the two
/// record forms under `operator`.
pub fn fixture(
    operator: &str,
    forms: Forms,
    canonical_kls: &[f32],
    derived_kls: &[f32],
) -> Fixture {
    build(engine(), operator, forms, canonical_kls, derived_kls)
}

/// Same fixture over an engine with an explicit config and capability
/// declaration. The engine may already have prefilled — registrations
/// happen after, so they land in the current session epoch.
pub fn fixture_with(
    operator: &str,
    forms: Forms,
    canonical_kls: &[f32],
    derived_kls: &[f32],
    config: SemanticPromotionConfig,
    capabilities: PromotionEngineCapabilities,
) -> Fixture {
    build(
        engine_with(config, capabilities),
        operator,
        forms,
        canonical_kls,
        derived_kls,
    )
}

/// Build over an engine that has already prefilled `prompt`.
///
/// Order matters: a prefill opens a new session epoch and resets the
/// session's registrations, so sources and records must be registered
/// *after* it or their references go stale.
#[allow(clippy::too_many_arguments)]
pub fn fixture_prefilled(
    operator: &str,
    forms: Forms,
    canonical_kls: &[f32],
    derived_kls: &[f32],
    config: SemanticPromotionConfig,
    capabilities: PromotionEngineCapabilities,
    weights: &larql_inference::model::ModelWeights,
    prompt: &[u32],
) -> Fixture {
    let mut engine = engine_with(config, capabilities);
    let ffn = larql_compute::ffn::WeightFfn { weights };
    <SemanticPromotionEngine as crate::KvEngine>::prefill(&mut engine, weights, &ffn, prompt)
        .expect("fixture prefill");
    build(engine, operator, forms, canonical_kls, derived_kls)
}

fn build(
    mut engine: SemanticPromotionEngine,
    operator: &str,
    forms: Forms,
    canonical_kls: &[f32],
    derived_kls: &[f32],
) -> Fixture {
    let mut graph = ModelGraph::new();

    let source = SourceAuthority::from_tokens(engine.session_epoch(), SPAN_START, &SPAN_TOKENS);
    let authority = engine.register_source(source).unwrap();
    let source_node = ModelNode::SourceAuthority(authority);
    graph.add_node(source_node);

    let operation = OperationId::from_counter(1);

    let canonical_record = SemanticRecord::CanonicalFact(CanonicalFact {
        id: RecordId::from_counter(0),
        subject: SemanticAtom::new("Meridian"),
        relation: SemanticAtom::new("code"),
        value: SemanticValue::new("integer", "7431"),
        provenance: vec![authority],
        schema_version: 1,
    });
    let canonical_digest = canonical_record.content_digest();
    let canonical = engine
        .register_record(
            canonical_record,
            RecordMaterialisation::token_sequence(RECORD_TOKENS.to_vec()),
        )
        .unwrap();

    let derived_record = SemanticRecord::DerivedClaim(DerivedClaim {
        id: RecordId::from_counter(0),
        operation: signature(operator),
        operands: vec![canonical],
        result: SemanticValue::new("integer", "7432"),
        provenance: vec![authority],
        schema_version: 1,
    });
    let derived_digest = derived_record.content_digest();
    let derived = forms.derived.then(|| {
        engine
            .register_record(
                derived_record,
                RecordMaterialisation::token_sequence(RECORD_TOKENS.to_vec()),
            )
            .unwrap()
    });

    let mut next_qualification = 1u128;
    let mut register =
        |graph: &mut ModelGraph, node: ModelNode, record, digest: [u8; 32], kls: &[f32]| {
            graph.relate(node, ModelEdge::Asserts, source_node);
            graph.relate(node, ModelEdge::OperandOf, ModelNode::Operation(operation));
            if node.is_derived_claim() {
                graph.relate(node, ModelEdge::Discharges, ModelNode::Operation(operation));
            }
            // A derived record asserts a computed answer; whether the model
            // consumes it as one is what EXP-25CF measured, and at 4K it did
            // not. Fixtures therefore say `OperandForOperation` unless a
            // caller has evidence otherwise.
            if let Some(cert) = certificate_from_measurement(
                QualificationId::from_counter(next_qualification),
                record,
                digest,
                operator,
                kls,
                ConsumptionMode::OperandForOperation,
            ) {
                graph.add_certificate(cert);
            }
            next_qualification += 1;
        };

    if forms.canonical {
        register(
            &mut graph,
            ModelNode::CanonicalFact(canonical),
            canonical,
            canonical_digest,
            canonical_kls,
        );
    }
    if let Some(d) = derived {
        register(
            &mut graph,
            ModelNode::DerivedClaim(d),
            d,
            derived_digest,
            derived_kls,
        );
    }

    // The spec carries an arity-correct *template* binding. Which record
    // actually fills the slot is the walk's decision, and the executor
    // rebinds before opening the operation.
    let template_operand = if forms.canonical {
        canonical
    } else {
        derived.expect("a fixture must register at least one record form")
    };
    graph.add_operation(OperationSpec {
        id: operation,
        class: OperationClass::Transform,
        operands: vec![template_operand],
        answer_boundary: AnswerBoundary::ExactPayloadLength(PAYLOAD_TOKENS),
        signature: signature(operator),
    });

    // Costing inputs: how big the replaced span is, and how compact
    // each record's rendering is.
    graph.set_source_span_tokens(authority, SPAN_TOKENS.len() as u64);
    graph.set_record_tokens(canonical, RECORD_TOKENS.len() as u64);
    if let Some(d) = derived {
        graph.set_record_tokens(d, RECORD_TOKENS.len() as u64);
    }

    Fixture {
        engine,
        graph,
        authority,
        canonical,
        derived,
        operation,
    }
}

/// Re-declare the source span's length, so a fixture can model a
/// promotion that retires a large stretch of context rather than the
/// four planted tokens.
///
/// The engine's real span stays four tokens — this only changes what the
/// *planner* is told it would be retiring, which is the quantity the
/// cost model turns on.
pub fn with_modelled_span(mut f: Fixture, span_tokens: u64) -> Fixture {
    f.graph.set_source_span_tokens(f.authority, span_tokens);
    f
}

/// EXP-25 `copy`: canonical only, trajectory-clean.
pub fn copy_fixture() -> Fixture {
    fixture("identity", Forms::CANONICAL_ONLY, kl::COPY_CANONICAL, &[])
}

/// EXP-25 `transform_inc`, canonical form only.
pub fn increment_canonical_fixture() -> Fixture {
    fixture("increment", Forms::CANONICAL_ONLY, kl::INC_CANONICAL, &[])
}

/// EXP-25 `transform_rev`, canonical form only.
pub fn reversal_canonical_fixture() -> Fixture {
    fixture(
        "reverse_digits",
        Forms::CANONICAL_ONLY,
        kl::REV_CANONICAL,
        &[],
    )
}

/// EXP-25 `transform_inc` with both forms available — the planner has to
/// choose between payload-only canonical and trajectory-clean derived.
pub fn increment_both_forms_fixture() -> Fixture {
    fixture("increment", Forms::BOTH, kl::INC_CANONICAL, kl::INC_DERIVED)
}

/// EXP-25 `transform_rev`, derived form only — the row that fails on
/// payload and can be certified for nothing.
pub fn reversal_derived_only_fixture() -> Fixture {
    fixture("reverse_digits", Forms::DERIVED_ONLY, &[], kl::REV_DERIVED)
}
