use larql_core::*;

// ── Shortest path ──

#[test]
fn test_shortest_path_direct() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("A", "r", "B").with_confidence(0.9));

    let (cost, path) = shortest_path(&g, "A", "B").unwrap();
    assert!((cost - 0.1).abs() < 0.001); // weight = 1.0 - 0.9
    assert_eq!(path.len(), 1);
    assert_eq!(path[0].subject, "A");
    assert_eq!(path[0].object, "B");
}

#[test]
fn test_shortest_path_multi_hop() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("A", "r", "B").with_confidence(0.9));
    g.add_edge(Edge::new("B", "r", "C").with_confidence(0.8));

    let (cost, path) = shortest_path(&g, "A", "C").unwrap();
    assert!((cost - 0.3).abs() < 0.001); // 0.1 + 0.2
    assert_eq!(path.len(), 2);
}

#[test]
fn test_shortest_path_prefers_high_confidence() {
    let mut g = Graph::new();
    // Direct route: low confidence
    g.add_edge(Edge::new("A", "r", "C").with_confidence(0.5));
    // Indirect route: high confidence
    g.add_edge(Edge::new("A", "r", "B").with_confidence(0.99));
    g.add_edge(Edge::new("B", "r", "C").with_confidence(0.99));

    let (cost, path) = shortest_path(&g, "A", "C").unwrap();
    // Direct: 0.5, Indirect: 0.01 + 0.01 = 0.02
    assert!(cost < 0.1); // indirect is cheaper
    assert_eq!(path.len(), 2);
}

#[test]
fn test_shortest_path_no_route() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("A", "r", "B").with_confidence(0.9));
    g.add_edge(Edge::new("C", "r", "D").with_confidence(0.9));

    assert!(shortest_path(&g, "A", "D").is_none());
}

#[test]
fn test_shortest_path_same_node() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("A", "r", "B").with_confidence(0.9));

    let (cost, path) = shortest_path(&g, "A", "A").unwrap();
    assert!((cost - 0.0).abs() < f64::EPSILON);
    assert!(path.is_empty());
}

// ── Merge ──

#[test]
fn test_merge_graphs() {
    let mut g1 = Graph::new();
    g1.add_edge(Edge::new("France", "capital-of", "Paris"));

    let mut g2 = Graph::new();
    g2.add_edge(Edge::new("Germany", "capital-of", "Berlin"));
    g2.add_edge(Edge::new("France", "capital-of", "Paris")); // duplicate

    let added = merge_graphs(&mut g1, &g2);
    assert_eq!(added, 1); // only Berlin added, Paris skipped
    assert_eq!(g1.edge_count(), 2);
    assert!(g1.exists("Germany", "capital-of", "Berlin"));
}

#[test]
fn test_merge_empty_into_existing() {
    let mut g1 = Graph::new();
    g1.add_edge(Edge::new("France", "capital-of", "Paris"));

    let g2 = Graph::new();
    let added = merge_graphs(&mut g1, &g2);
    assert_eq!(added, 0);
    assert_eq!(g1.edge_count(), 1);
}

#[test]
fn test_merge_into_empty() {
    let mut g1 = Graph::new();
    let mut g2 = Graph::new();
    g2.add_edge(Edge::new("France", "capital-of", "Paris"));

    let added = merge_graphs(&mut g1, &g2);
    assert_eq!(added, 1);
    assert_eq!(g1.edge_count(), 1);
}
