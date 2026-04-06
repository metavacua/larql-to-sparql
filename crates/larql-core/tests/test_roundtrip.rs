use larql_core::*;

fn sample_graph() -> Graph {
    let mut g = Graph::new();
    g.add_edge(
        Edge::new("France", "capital-of", "Paris")
            .with_confidence(0.89)
            .with_source(SourceType::Parametric),
    );
    g.add_edge(
        Edge::new("France", "currency", "Euro")
            .with_confidence(0.91)
            .with_source(SourceType::Parametric),
    );
    g.add_edge(
        Edge::new("Germany", "capital-of", "Berlin")
            .with_confidence(0.81)
            .with_source(SourceType::Wikidata),
    );
    g.add_edge(
        Edge::new("Paris", "located-in", "France")
            .with_confidence(0.98)
            .with_source(SourceType::Parametric)
            .with_metadata("model", serde_json::json!("gemma-3")),
    );
    g
}

fn assert_graphs_equal(a: &Graph, b: &Graph) {
    assert_eq!(a.edge_count(), b.edge_count());
    for edge in a.edges() {
        assert!(
            b.exists(&edge.subject, &edge.relation, &edge.object),
            "missing edge: {} --{}--> {}",
            edge.subject,
            edge.relation,
            edge.object
        );
    }
}

// ── JSON roundtrip ──

#[test]
fn test_json_value_roundtrip() {
    let g = sample_graph();
    let json = g.to_json_value();
    let restored = Graph::from_json_value(&json).unwrap();
    assert_graphs_equal(&g, &restored);
}

#[test]
fn test_json_bytes_roundtrip() {
    let g = sample_graph();
    let bytes = to_bytes(&g, Format::Json).unwrap();
    let restored = from_bytes(&bytes, Format::Json).unwrap();
    assert_graphs_equal(&g, &restored);
}

#[test]
fn test_json_file_roundtrip() {
    let g = sample_graph();
    let path = std::env::temp_dir().join("test_roundtrip.larql.json");
    save(&g, &path).unwrap();
    let restored = load(&path).unwrap();
    assert_graphs_equal(&g, &restored);
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_json_preserves_confidence() {
    let g = sample_graph();
    let json = g.to_json_value();
    let restored = Graph::from_json_value(&json).unwrap();

    for orig in g.edges() {
        let restored_edges = restored.select(&orig.subject, Some(&orig.relation));
        let matched = restored_edges
            .iter()
            .find(|e| e.object == orig.object)
            .unwrap();
        assert!(
            (orig.confidence - matched.confidence).abs() < 0.001,
            "confidence mismatch for {} --{}--> {}: {} vs {}",
            orig.subject,
            orig.relation,
            orig.object,
            orig.confidence,
            matched.confidence
        );
    }
}

#[test]
fn test_json_preserves_source() {
    let g = sample_graph();
    let json = g.to_json_value();
    let restored = Graph::from_json_value(&json).unwrap();

    let france_cap = restored.select("France", Some("capital-of"));
    assert_eq!(france_cap[0].source, SourceType::Parametric);

    let germany_cap = restored.select("Germany", Some("capital-of"));
    assert_eq!(germany_cap[0].source, SourceType::Wikidata);
}

#[test]
fn test_json_preserves_metadata() {
    let g = sample_graph();
    let json = g.to_json_value();
    let restored = Graph::from_json_value(&json).unwrap();

    let paris = restored.select("Paris", Some("located-in"));
    let meta = paris[0].metadata.as_ref().unwrap();
    assert_eq!(meta["model"], "gemma-3");
}

#[test]
fn test_json_format_structure() {
    let g = sample_graph();
    let json = g.to_json_value();

    assert_eq!(json["larql_version"], "0.1.0");
    assert!(json["metadata"].is_object());
    assert!(json["schema"].is_object());
    assert!(json["edges"].is_array());
    assert_eq!(json["edges"].as_array().unwrap().len(), 4);

    // Check compact edge keys
    let edge = &json["edges"][0];
    assert!(edge.get("s").is_some());
    assert!(edge.get("r").is_some());
    assert!(edge.get("o").is_some());
    assert!(edge.get("c").is_some());
}

// ── MessagePack roundtrip ──

#[test]
fn test_msgpack_bytes_roundtrip() {
    let g = sample_graph();
    let bytes = to_bytes(&g, Format::MessagePack).unwrap();
    let restored = from_bytes(&bytes, Format::MessagePack).unwrap();
    assert_graphs_equal(&g, &restored);
}

#[test]
fn test_msgpack_file_roundtrip() {
    let g = sample_graph();
    let path = std::env::temp_dir().join("test_roundtrip.larql.bin");
    save(&g, &path).unwrap();
    let restored = load(&path).unwrap();
    assert_graphs_equal(&g, &restored);
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_msgpack_preserves_confidence() {
    let g = sample_graph();
    let bytes = to_bytes(&g, Format::MessagePack).unwrap();
    let restored = from_bytes(&bytes, Format::MessagePack).unwrap();

    for orig in g.edges() {
        let restored_edges = restored.select(&orig.subject, Some(&orig.relation));
        let matched = restored_edges
            .iter()
            .find(|e| e.object == orig.object)
            .unwrap();
        assert!(
            (orig.confidence - matched.confidence).abs() < 0.001,
            "confidence mismatch"
        );
    }
}

#[test]
fn test_msgpack_smaller_than_json() {
    let g = sample_graph();
    let json_bytes = to_bytes(&g, Format::Json).unwrap();
    let msgpack_bytes = to_bytes(&g, Format::MessagePack).unwrap();
    assert!(
        msgpack_bytes.len() < json_bytes.len(),
        "msgpack {} should be smaller than json {}",
        msgpack_bytes.len(),
        json_bytes.len()
    );
}

// ── Cross-format ──

#[test]
fn test_json_to_msgpack_roundtrip() {
    let g = sample_graph();
    let json_bytes = to_bytes(&g, Format::Json).unwrap();
    let intermediate = from_bytes(&json_bytes, Format::Json).unwrap();
    let msgpack_bytes = to_bytes(&intermediate, Format::MessagePack).unwrap();
    let final_graph = from_bytes(&msgpack_bytes, Format::MessagePack).unwrap();
    assert_graphs_equal(&g, &final_graph);
}

// ── Format detection ──

#[test]
fn test_format_from_path() {
    assert_eq!(Format::from_path("graph.larql.json"), Some(Format::Json));
    assert_eq!(Format::from_path("graph.json"), Some(Format::Json));
    assert_eq!(
        Format::from_path("graph.larql.bin"),
        Some(Format::MessagePack)
    );
    assert_eq!(Format::from_path("graph.bin"), Some(Format::MessagePack));
    assert_eq!(
        Format::from_path("graph.msgpack"),
        Some(Format::MessagePack)
    );
    assert_eq!(Format::from_path("graph.txt"), None);
}

// ── Empty graph ──

#[test]
fn test_empty_graph_roundtrip() {
    let g = Graph::new();
    let json = g.to_json_value();
    let restored = Graph::from_json_value(&json).unwrap();
    assert_eq!(restored.edge_count(), 0);
}
