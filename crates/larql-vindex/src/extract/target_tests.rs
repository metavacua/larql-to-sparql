//! Route-admission tests.
//!
//! The property under test is that the decision is made from *capabilities*
//! and never from a model family, and that a native request which cannot be
//! honoured fails loudly instead of quietly producing the transcoded
//! artifact the caller was trying to avoid.

use super::*;

/// A source that can do everything native needs — the shape GPT-OSS
/// presents once its raw streams are reachable.
fn capable() -> SourceCapabilities {
    SourceCapabilities::from_expert_format(ExpertFormat::PackedMxfp4, Some(RegionFormat::Mxfp4))
        .with_scale_streams(true)
        .with_gate_up_layout(RegionLayout::Interleaved)
}

// ── the capability is the format's, not the family's ─────────────────────

/// `PackedMxfp4` is the only expert format that carries split scales, and
/// it says so itself. Nothing here consults a model name.
#[test]
fn split_scale_streams_is_a_property_of_the_format() {
    assert!(ExpertFormat::PackedMxfp4.has_split_scale_streams());
    assert!(!ExpertFormat::PackedBF16.has_split_scale_streams());
    assert!(!ExpertFormat::PerExpert.has_split_scale_streams());
}

/// Derivation fills in what the format knows and deliberately leaves the
/// rest unanswered — availability and stored layout are facts about the
/// artifact, not the format class.
#[test]
fn derivation_leaves_artifact_facts_to_the_caller() {
    let caps = SourceCapabilities::from_expert_format(
        ExpertFormat::PackedMxfp4,
        Some(RegionFormat::Mxfp4),
    );
    assert!(caps.split_scale_streams);
    assert!(
        !caps.scale_streams_available,
        "whether a source can hand over raw bytes is not knowable from the format"
    );
    assert!(!caps.gate_up_layout.is_declared());
}

// ── admission ────────────────────────────────────────────────────────────

#[test]
fn a_fully_capable_source_is_admitted_for_native() {
    assert_eq!(
        admit(ExtractionRequest::Native, &capable()).unwrap(),
        ExtractionTarget::NativeV3
    );
}

/// An inline-scale format is still eligible for native V3 — the native
/// route is not MXFP4-only. Gemma reaches it today with scales inline.
#[test]
fn an_inline_scale_source_is_admitted_without_scale_streams() {
    let caps =
        SourceCapabilities::from_expert_format(ExpertFormat::PackedBF16, Some(RegionFormat::BF16))
            .with_gate_up_layout(RegionLayout::ContiguousHalves);
    assert!(!caps.scale_streams_available, "and it does not need them");
    assert_eq!(
        admit(ExtractionRequest::Native, &caps).unwrap(),
        ExtractionTarget::NativeV3
    );
}

/// The legacy route transcodes whatever it is given, so it is admitted
/// even from a source that could not go native.
#[test]
fn the_legacy_route_is_always_admitted() {
    let helpless = SourceCapabilities::from_expert_format(ExpertFormat::PackedMxfp4, None);
    assert_eq!(
        admit(ExtractionRequest::Legacy, &helpless).unwrap(),
        ExtractionTarget::LegacyKQuant
    );
}

// ── Auto is policy, and is not the same as a refusal ─────────────────────

/// `Auto` resolves to the legacy route **even from a fully capable
/// source**, so every existing caller's behaviour is unchanged while the
/// native path is being qualified. Native is opt-in until #4 passes.
#[test]
fn auto_stays_on_the_legacy_route_even_when_native_is_possible() {
    let caps = capable();
    assert_eq!(
        admit(ExtractionRequest::Native, &caps).unwrap(),
        ExtractionTarget::NativeV3,
        "the source really is capable"
    );
    assert_eq!(
        admit(ExtractionRequest::Auto, &caps).unwrap(),
        ExtractionTarget::LegacyKQuant,
        "yet Auto must not opt a caller in"
    );
}

/// `Auto` never errors, whatever the source can or cannot do — it made no
/// demand to be refused. This is what keeps "a caller was told no" and "a
/// caller never asked" distinguishable at the call site.
#[test]
fn auto_never_refuses_because_it_demanded_nothing() {
    for caps in [
        capable(),
        capable().with_scale_streams(false),
        SourceCapabilities::from_expert_format(ExpertFormat::PackedMxfp4, None),
    ] {
        assert_eq!(
            admit(ExtractionRequest::Auto, &caps).unwrap(),
            ExtractionTarget::LegacyKQuant,
            "{caps:?}"
        );
    }
}

/// The same incapable source: `Auto` proceeds, `Native` refuses. If these
/// collapsed to one behaviour, a caller asking for native bytes could
/// receive transcoded ones and compare them against the transcode.
#[test]
fn auto_and_native_diverge_on_an_incapable_source() {
    let incapable = capable().with_scale_streams(false);
    assert!(admit(ExtractionRequest::Auto, &incapable).is_ok());
    assert!(admit(ExtractionRequest::Native, &incapable).is_err());
}

// ── refusals: no silent fallback, ever ───────────────────────────────────

fn refusal(caps: &SourceCapabilities) -> String {
    let err = admit(ExtractionRequest::Native, caps)
        .expect_err("this source cannot satisfy a native request");
    err.to_string()
}

/// The load-bearing property: a native request that cannot be honoured
/// **errors**. It must never come back as `LegacyKQuant`, because the
/// caller would then compare transcoded bytes against the transcode they
/// were trying to qualify against.
#[test]
fn an_unsatisfiable_native_request_never_returns_the_legacy_target() {
    for caps in [
        SourceCapabilities::from_expert_format(ExpertFormat::PackedMxfp4, None),
        capable().with_scale_streams(false),
        capable().with_gate_up_layout(RegionLayout::Unspecified),
    ] {
        let result = admit(ExtractionRequest::Native, &caps);
        assert!(result.is_err(), "must refuse, not downgrade: {caps:?}");
        assert_ne!(
            result.ok(),
            Some(ExtractionTarget::LegacyKQuant),
            "a refusal must not be spelled as the other target"
        );
    }
}

#[test]
fn an_unrepresentable_encoding_is_refused_by_name() {
    let caps = SourceCapabilities::from_expert_format(ExpertFormat::PackedMxfp4, None)
        .with_scale_streams(true)
        .with_gate_up_layout(RegionLayout::Interleaved);
    let msg = refusal(&caps);
    assert!(msg.contains("region format"), "{msg}");
    assert!(msg.contains("transcode"), "{msg}");
}

/// The gap that matters most in practice: the format declares split
/// scales, but the source cannot produce them. Writing the payload alone
/// would yield a bank whose every group decodes at 2^0 — finite, plausible
/// and wrong.
#[test]
fn a_split_scale_format_without_reachable_streams_is_refused() {
    let msg = refusal(&capable().with_scale_streams(false));
    assert!(msg.contains("separate stream"), "{msg}");
    assert!(msg.contains("2^0"), "{msg}");
}

#[test]
fn an_undeclared_fused_layout_is_refused() {
    let msg = refusal(&capable().with_gate_up_layout(RegionLayout::Unspecified));
    assert!(msg.contains("undeclared"), "{msg}");
    assert!(msg.contains("interleaved"), "{msg}");
}

/// Every refusal points at the legacy route explicitly, so a caller that
/// wanted an artifact at all knows what to ask for next — without the
/// admission function making that choice for them.
#[test]
fn every_refusal_names_the_alternative_without_taking_it() {
    for caps in [
        SourceCapabilities::from_expert_format(ExpertFormat::PackedMxfp4, None),
        capable().with_scale_streams(false),
        capable().with_gate_up_layout(RegionLayout::Unspecified),
    ] {
        let msg = refusal(&caps);
        assert!(msg.contains("legacy"), "{msg}");
        assert!(msg.contains("explicitly"), "{msg}");
    }
}
