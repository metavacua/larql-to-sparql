use std::collections::HashSet;

use larql_core::core::schema::{RelationMeta, Schema, TypeRule};

#[test]
fn test_empty_schema() {
    let s = Schema::new();
    assert!(s.names().is_empty());
    assert!(!s.has("anything"));
    assert!(s.get("anything").is_none());
}

#[test]
fn test_add_and_get() {
    let mut s = Schema::new();
    s.add(RelationMeta {
        name: "capital-of".to_string(),
        subject_types: vec!["country".to_string()],
        object_types: vec!["city".to_string()],
        reversible: true,
        reverse_name: None,
    });

    assert!(s.has("capital-of"));
    let meta = s.get("capital-of").unwrap();
    assert_eq!(meta.subject_types, vec!["country"]);
    assert_eq!(meta.object_types, vec!["city"]);
    assert!(meta.reversible);
}

#[test]
fn test_type_inference_outgoing() {
    let mut s = Schema::new();
    s.add_type_rule(TypeRule {
        node_type: "country".to_string(),
        outgoing: vec!["capital-of".to_string(), "currency".to_string()],
        incoming: vec![],
    });

    let out: HashSet<String> = ["capital-of".to_string()].into();
    let inp: HashSet<String> = HashSet::new();
    assert_eq!(s.infer_type(&out, &inp), Some("country".to_string()));
}

#[test]
fn test_type_inference_incoming() {
    let mut s = Schema::new();
    s.add_type_rule(TypeRule {
        node_type: "city".to_string(),
        outgoing: vec![],
        incoming: vec!["capital-of".to_string()],
    });

    let out: HashSet<String> = HashSet::new();
    let inp: HashSet<String> = ["capital-of".to_string()].into();
    assert_eq!(s.infer_type(&out, &inp), Some("city".to_string()));
}

#[test]
fn test_type_inference_no_match() {
    let mut s = Schema::new();
    s.add_type_rule(TypeRule {
        node_type: "country".to_string(),
        outgoing: vec!["capital-of".to_string()],
        incoming: vec![],
    });

    let out: HashSet<String> = ["unrelated".to_string()].into();
    let inp: HashSet<String> = HashSet::new();
    assert_eq!(s.infer_type(&out, &inp), None);
}

#[test]
fn test_type_inference_first_match_wins() {
    let mut s = Schema::new();
    s.add_type_rule(TypeRule {
        node_type: "person".to_string(),
        outgoing: vec!["birthplace".to_string()],
        incoming: vec![],
    });
    s.add_type_rule(TypeRule {
        node_type: "organization".to_string(),
        outgoing: vec!["birthplace".to_string()],
        incoming: vec![],
    });

    let out: HashSet<String> = ["birthplace".to_string()].into();
    let inp: HashSet<String> = HashSet::new();
    // First rule wins
    assert_eq!(s.infer_type(&out, &inp), Some("person".to_string()));
}

#[test]
fn test_schema_json_roundtrip() {
    let mut s = Schema::new();
    s.add(RelationMeta {
        name: "capital-of".to_string(),
        subject_types: vec!["country".to_string()],
        object_types: vec!["city".to_string()],
        reversible: true,
        reverse_name: None,
    });
    s.add_type_rule(TypeRule {
        node_type: "country".to_string(),
        outgoing: vec!["capital-of".to_string()],
        incoming: vec![],
    });

    let json = s.to_json_value();
    let restored = Schema::from_json_value(&json);

    assert!(restored.has("capital-of"));
    assert_eq!(restored.type_rules().len(), 1);
    assert_eq!(restored.type_rules()[0].node_type, "country");
}
