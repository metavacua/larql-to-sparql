use larql_core::*;

/// Load the example graph (matches Python format) and verify every field.
#[test]
fn test_load_python_produced_graph() {
    let g = load("../../examples/gemma_4b_knowledge.json").unwrap();

    assert_eq!(g.edge_count(), 10);

    // Spot-check edges
    assert!(g.exists("France", "capital-of", "Paris"));
    assert!(g.exists("France", "language-of", "French"));
    assert!(g.exists("France", "continent", "Europe"));
    assert!(g.exists("France", "currency", "Euro"));
    assert!(g.exists("Germany", "capital-of", "Berlin"));
    assert!(g.exists("Paris", "located-in", "France"));
    assert!(g.exists("Berlin", "located-in", "Germany"));
}

#[test]
fn test_python_graph_confidence() {
    let g = load("../../examples/gemma_4b_knowledge.json").unwrap();

    let france_cap = g.select("France", Some("capital-of"));
    assert_eq!(france_cap.len(), 1);
    assert!((france_cap[0].confidence - 0.89).abs() < 0.001);

    let paris_loc = g.select("Paris", Some("located-in"));
    assert_eq!(paris_loc.len(), 1);
    assert!((paris_loc[0].confidence - 0.98).abs() < 0.001);
}

#[test]
fn test_python_graph_source() {
    let g = load("../../examples/gemma_4b_knowledge.json").unwrap();

    for edge in g.edges() {
        assert_eq!(edge.source, SourceType::Parametric);
    }
}

#[test]
fn test_python_graph_schema() {
    let g = load("../../examples/gemma_4b_knowledge.json").unwrap();

    assert!(g.schema.has("capital-of"));
    assert!(g.schema.has("language-of"));
    assert!(g.schema.has("currency"));
    assert!(g.schema.has("continent"));
    assert!(g.schema.has("located-in"));

    let cap = g.schema.get("capital-of").unwrap();
    assert_eq!(cap.subject_types, vec!["country"]);
    assert_eq!(cap.object_types, vec!["city"]);
    assert!(cap.reversible);
}

#[test]
fn test_python_graph_type_rules() {
    let g = load("../../examples/gemma_4b_knowledge.json").unwrap();

    assert_eq!(g.schema.type_rules().len(), 3);

    // France should be inferred as "country"
    let france = g.node("France").unwrap();
    assert_eq!(france.node_type, Some("country".to_string()));

    // Paris should be inferred as "city"
    let paris = g.node("Paris").unwrap();
    assert_eq!(paris.node_type, Some("city".to_string()));
}

#[test]
fn test_python_graph_stats() {
    let g = load("../../examples/gemma_4b_knowledge.json").unwrap();
    let stats = g.stats();

    assert_eq!(stats.edges, 10);
    assert_eq!(stats.entities, 8);
    assert_eq!(stats.relations, 5);
    assert_eq!(stats.connected_components, 1);
}

/// Save the Python graph as JSON, reload, verify identical.
#[test]
fn test_python_graph_json_roundtrip() {
    let original = load("../../examples/gemma_4b_knowledge.json").unwrap();
    let path = std::env::temp_dir().join("test_python_compat.larql.json");

    save(&original, &path).unwrap();
    let reloaded = load(&path).unwrap();

    assert_eq!(original.edge_count(), reloaded.edge_count());
    for edge in original.edges() {
        assert!(
            reloaded.exists(&edge.subject, &edge.relation, &edge.object),
            "missing: {} --{}--> {}",
            edge.subject,
            edge.relation,
            edge.object
        );
    }

    std::fs::remove_file(&path).ok();
}

/// Save the Python graph as MessagePack, reload, verify identical.
#[test]
fn test_python_graph_msgpack_roundtrip() {
    let original = load("../../examples/gemma_4b_knowledge.json").unwrap();
    let path = std::env::temp_dir().join("test_python_compat.larql.bin");

    save(&original, &path).unwrap();
    let reloaded = load(&path).unwrap();

    assert_eq!(original.edge_count(), reloaded.edge_count());
    for edge in original.edges() {
        assert!(reloaded.exists(&edge.subject, &edge.relation, &edge.object));
    }

    std::fs::remove_file(&path).ok();
}
