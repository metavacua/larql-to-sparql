//! Demonstrate graph filtering — select edges by confidence, layer, relation, etc.
//!
//! Run: cargo run -p larql-core --example filter_demo

use larql_core::*;

fn main() {
    println!("=== LARQL Filter Demo ===\n");

    // Build a sample extraction graph with metadata
    let mut graph = Graph::new();

    // Knowledge edges with layer/selectivity metadata (simulating weight extraction output)
    let edges = vec![
        ("France", "capital-of", "Paris", 0.92, 26, 0.85),
        ("Germany", "capital-of", "Berlin", 0.88, 26, 0.78),
        ("Japan", "capital-of", "Tokyo", 0.91, 27, 0.82),
        ("France", "language-of", "French", 0.45, 10, 0.22),
        ("Germany", "language-of", "German", 0.41, 10, 0.19),
        ("France", "continent", "Europe", 0.72, 15, 0.55),
        ("Japan", "continent", "Asia", 0.68, 15, 0.48),
        ("Paris", "located-in", "France", 0.35, 8, 0.12),
        ("Berlin", "located-in", "Germany", 0.33, 8, 0.10),
        ("Tokyo", "located-in", "Japan", 0.30, 8, 0.09),
    ];

    for (s, r, o, conf, layer, sel) in edges {
        graph.add_edge(
            Edge::new(s, r, o)
                .with_confidence(conf)
                .with_source(SourceType::Parametric)
                .with_metadata("layer", serde_json::json!(layer))
                .with_metadata("selectivity", serde_json::json!(sel)),
        );
    }

    println!("Original graph: {} edges\n", graph.edge_count());

    // ── Filter by confidence ──
    let high_conf = filter_graph(
        &graph,
        &FilterConfig {
            min_confidence: Some(0.7),
            ..Default::default()
        },
    );
    println!(
        "min_confidence >= 0.7:   {} edges (removed {})",
        high_conf.edge_count(),
        graph.edge_count() - high_conf.edge_count()
    );

    // ── Filter by layer range ──
    let knowledge_layers = filter_graph(
        &graph,
        &FilterConfig {
            min_layer: Some(20),
            ..Default::default()
        },
    );
    println!(
        "layers >= 20:            {} edges (knowledge layers only)",
        knowledge_layers.edge_count()
    );

    // ── Filter by selectivity ──
    let selective = filter_graph(
        &graph,
        &FilterConfig {
            min_selectivity: Some(0.5),
            ..Default::default()
        },
    );
    println!(
        "selectivity >= 0.5:      {} edges (factual edges)",
        selective.edge_count()
    );

    // ── Filter by relation ──
    let capitals = filter_graph(
        &graph,
        &FilterConfig {
            relations: Some(vec!["capital-of".to_string()]),
            ..Default::default()
        },
    );
    println!(
        "relation = capital-of:   {} edges",
        capitals.edge_count()
    );

    // ── Exclude relation ──
    let no_located = filter_graph(
        &graph,
        &FilterConfig {
            exclude_relations: Some(vec!["located-in".to_string()]),
            ..Default::default()
        },
    );
    println!(
        "exclude located-in:      {} edges",
        no_located.edge_count()
    );

    // ── Subject contains ──
    let france = filter_graph(
        &graph,
        &FilterConfig {
            subject_contains: Some("France".to_string()),
            ..Default::default()
        },
    );
    println!(
        "subject contains France: {} edges",
        france.edge_count()
    );

    // ── Combined filters ──
    let best = filter_graph(
        &graph,
        &FilterConfig {
            min_confidence: Some(0.8),
            min_layer: Some(20),
            min_selectivity: Some(0.7),
            ..Default::default()
        },
    );
    println!(
        "\nCombined (conf>=0.8, layer>=20, sel>=0.7): {} edges",
        best.edge_count()
    );
    for edge in best.edges() {
        println!(
            "  {} --{}--> {} (conf={:.2})",
            edge.subject, edge.relation, edge.object, edge.confidence
        );
    }

    println!("\n=== Done ===");
}
