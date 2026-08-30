//! Declared-vs-resolved value comparison.

use super::support::{glimmer_shaped_target, known_dense, FIXTURE_LAYERS};
use crate::format::vindex3::plan::compare::compare;
use crate::format::vindex3::plan::{Finding, FindingCategory, SemanticClass};

fn find<'a>(findings: &'a [Finding], subject: &str) -> &'a Finding {
    findings
        .iter()
        .find(|f| f.subject == subject)
        .unwrap_or_else(|| panic!("no finding for {subject}"))
}

#[test]
fn honoured_topology_scalars_are_representable() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target(dir.path());
    let findings = compare(&inventory);
    for subject in [
        "text_config.hidden_size",
        "text_config.num_hidden_layers",
        "text_config.num_key_value_heads",
        "text_config.sliding_window",
    ] {
        let finding = find(&findings, subject);
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{subject}"
        );
        assert!(!finding.blocks());
    }
}

/// The NoPE round trip: per-layer declared θ with zeros on global layers is
/// honoured — resolution answers `PositionPolicy::None` on exactly the
/// sentinel layers, so declared and resolved agree, per layer.
#[test]
fn nope_layer_rope_theta_is_honoured_per_layer() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target(dir.path());
    let findings = compare(&inventory);
    let finding = find(&findings, "text_config.layer_rope_theta");
    assert_eq!(finding.category, FindingCategory::Representable);
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert!(!finding.blocks());
    assert!(
        finding.detail.contains("NoPE"),
        "detail must state the policies were honoured incl. NoPE: {}",
        finding.detail
    );
}

/// The counter-case the comparator exists for: a resolver that answers a
/// rotary policy on a declared-NoPE layer mismatches on exactly that layer.
#[test]
fn a_resolver_without_nope_mismatches_on_the_sentinel_layers() {
    use larql_models::config::PositionPolicy;
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = glimmer_shaped_target(dir.path());
    // Corrupt resolution the way the pre-PositionPolicy build behaved:
    // every layer rotary at the uniform base.
    for layer in &mut inventory.resolved.layers {
        layer.position = PositionPolicy::Rope { theta: 500000.0 };
    }
    let findings = compare(&inventory);
    let finding = find(&findings, "text_config.layer_rope_theta");
    assert_eq!(finding.category, FindingCategory::Mismatched);
    assert!(finding.blocks());
    let expected = FIXTURE_LAYERS / 4;
    assert!(
        finding
            .detail
            .contains(&format!("{expected} of {FIXTURE_LAYERS}")),
        "detail must count the disagreeing layers: {}",
        finding.detail
    );
}

/// With a per-layer declaration present, the uniform rope comparator stands
/// down — one authority per fact.
#[test]
fn uniform_rope_defers_to_the_per_layer_declaration() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target(dir.path());
    let findings = compare(&inventory);
    assert!(
        !findings
            .iter()
            .any(|f| f.subject.ends_with("rope_parameters.rope_theta")),
        "uniform comparator must defer to layer_rope_theta"
    );
}

/// On a model with no per-layer array, the uniform comparator runs — and
/// passes when resolution honours the declared base.
#[test]
fn uniform_rope_checks_when_it_owns_the_fact() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = known_dense(dir.path());
    let findings = compare(&inventory);
    let finding = find(&findings, "rope_theta");
    assert_eq!(finding.category, FindingCategory::Representable);
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
}

#[test]
fn declared_layer_types_honoured_by_resolution() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target(dir.path());
    let findings = compare(&inventory);
    let finding = find(&findings, "text_config.layer_types");
    assert_eq!(finding.category, FindingCategory::Representable);
}

/// A mismatch fabricated directly: declared hidden_size ≠ resolved.
#[test]
fn scalar_disagreement_is_a_blocking_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = glimmer_shaped_target(dir.path());
    // Corrupt the resolved side; the declared fact stays.
    inventory.resolved.hidden_size += 1;
    let findings = compare(&inventory);
    let finding = find(&findings, "text_config.hidden_size");
    assert_eq!(finding.category, FindingCategory::Mismatched);
    assert!(finding.blocks());
    assert!(finding.declared.is_some() && finding.resolved.is_some());
}
