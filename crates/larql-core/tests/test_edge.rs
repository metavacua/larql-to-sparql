use std::collections::HashSet;

use larql_core::*;

#[test]
fn test_edge_new_defaults() {
    let e = Edge::new("France", "capital-of", "Paris");
    assert_eq!(e.subject, "France");
    assert_eq!(e.relation, "capital-of");
    assert_eq!(e.object, "Paris");
    assert!((e.confidence - 1.0).abs() < f64::EPSILON);
    assert_eq!(e.source, SourceType::Unknown);
    assert!(e.metadata.is_none());
    assert!(e.injection.is_none());
}

#[test]
fn test_edge_builder() {
    let e = Edge::new("France", "capital-of", "Paris")
        .with_confidence(0.89)
        .with_source(SourceType::Parametric)
        .with_metadata("model", serde_json::json!("gemma-3"))
        .with_metadata("passes", serde_json::json!(3));

    assert!((e.confidence - 0.89).abs() < f64::EPSILON);
    assert_eq!(e.source, SourceType::Parametric);
    let meta = e.metadata.as_ref().unwrap();
    assert_eq!(meta.len(), 2);
    assert_eq!(meta["model"], "gemma-3");
    assert_eq!(meta["passes"], 3);
}

#[test]
fn test_confidence_clamped() {
    let high = Edge::new("A", "r", "B").with_confidence(1.5);
    assert!((high.confidence - 1.0).abs() < f64::EPSILON);

    let low = Edge::new("A", "r", "B").with_confidence(-0.5);
    assert!((low.confidence - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_edge_equality_ignores_confidence() {
    let e1 = Edge::new("France", "capital-of", "Paris").with_confidence(0.89);
    let e2 = Edge::new("France", "capital-of", "Paris").with_confidence(0.50);
    assert_eq!(e1, e2);
}

#[test]
fn test_edge_equality_different_triple() {
    let e1 = Edge::new("France", "capital-of", "Paris");
    let e2 = Edge::new("France", "capital-of", "Lyon");
    assert_ne!(e1, e2);
}

#[test]
fn test_edge_hash_consistency() {
    let e1 = Edge::new("France", "capital-of", "Paris").with_confidence(0.89);
    let e2 = Edge::new("France", "capital-of", "Paris").with_confidence(0.50);

    let mut set = HashSet::new();
    set.insert(e1);
    set.insert(e2);
    assert_eq!(set.len(), 1); // same triple = same hash
}

#[test]
fn test_triple() {
    let e = Edge::new("France", "capital-of", "Paris");
    let t = e.triple();
    assert_eq!(t.0, "France");
    assert_eq!(t.1, "capital-of");
    assert_eq!(t.2, "Paris");
}

#[test]
fn test_compact_roundtrip() {
    let original = Edge::new("France", "capital-of", "Paris")
        .with_confidence(0.89)
        .with_source(SourceType::Parametric)
        .with_metadata("key", serde_json::json!("value"));

    let compact = larql_core::core::edge::CompactEdge::from(&original);
    assert_eq!(compact.s, "France");
    assert_eq!(compact.r, "capital-of");
    assert_eq!(compact.o, "Paris");
    assert!((compact.c - 0.89).abs() < f64::EPSILON);
    assert_eq!(compact.src.as_deref(), Some("parametric"));

    let restored = Edge::from(compact);
    assert_eq!(restored.subject, "France");
    assert_eq!(restored.relation, "capital-of");
    assert_eq!(restored.object, "Paris");
    assert!((restored.confidence - 0.89).abs() < f64::EPSILON);
    assert_eq!(restored.source, SourceType::Parametric);
    assert!(restored.metadata.is_some());
}

#[test]
fn test_compact_unknown_source_omitted() {
    let e = Edge::new("A", "r", "B");
    let compact = larql_core::core::edge::CompactEdge::from(&e);
    assert!(compact.src.is_none());
}

#[test]
fn test_compact_json_serialization() {
    let e = Edge::new("France", "capital-of", "Paris")
        .with_confidence(0.89)
        .with_source(SourceType::Parametric);

    let compact = larql_core::core::edge::CompactEdge::from(&e);
    let json = serde_json::to_value(&compact).unwrap();

    assert_eq!(json["s"], "France");
    assert_eq!(json["r"], "capital-of");
    assert_eq!(json["o"], "Paris");
    assert!((json["c"].as_f64().unwrap() - 0.89).abs() < f64::EPSILON);
    assert_eq!(json["src"], "parametric");
    // meta should be absent (skip_serializing_if)
    assert!(json.get("meta").is_none());
    assert!(json.get("inj").is_none());
}

#[test]
fn test_source_type_as_str() {
    assert_eq!(SourceType::Parametric.as_str(), "parametric");
    assert_eq!(SourceType::Document.as_str(), "document");
    assert_eq!(SourceType::Installed.as_str(), "installed");
    assert_eq!(SourceType::Wikidata.as_str(), "wikidata");
    assert_eq!(SourceType::Manual.as_str(), "manual");
    assert_eq!(SourceType::Unknown.as_str(), "unknown");
}
