use std::{collections::HashSet, path::PathBuf};

use clap::Args;
use larql_core::{
    core::graph::Graph,
    io::{load_graph, save_graph},
};

/// Consensus-merge two graphs.
///
/// Edges present in both A and B get confidence 1.0.
/// Edges unique to A or B get confidence 0.7.
pub fn consensus_merge(a: &Graph, b: &Graph) -> Graph {
    let set_a: HashSet<(String, String, String)> = a
        .edges()
        .iter()
        .map(|e| (e.subject.clone(), e.relation.clone(), e.object.clone()))
        .collect();
    let set_b: HashSet<(String, String, String)> = b
        .edges()
        .iter()
        .map(|e| (e.subject.clone(), e.relation.clone(), e.object.clone()))
        .collect();

    let mut out = Graph::new();
    for e in a.edges().iter() {
        let triple = (e.subject.clone(), e.relation.clone(), e.object.clone());
        let mut edge = e.clone();
        edge.confidence = if set_b.contains(&triple) { 1.0 } else { 0.7 };
        out.add_edge(edge);
    }
    for e in b.edges().iter() {
        let triple = (e.subject.clone(), e.relation.clone(), e.object.clone());
        if !set_a.contains(&triple) {
            let mut edge = e.clone();
            edge.confidence = 0.7;
            out.add_edge(edge);
        }
    }
    out
}

#[derive(Args)]
pub struct GraphDiffArgs {
    /// First graph (.larql.json), typically from extract-codebase.
    graph_a: PathBuf,
    /// Second graph (.larql.json), typically from extract-graphify.
    graph_b: PathBuf,
    /// Output merged graph path.
    #[arg(short, long, default_value = "merged.larql.json")]
    output: PathBuf,
}

pub fn run(args: GraphDiffArgs) -> Result<(), Box<dyn std::error::Error>> {
    let a = load_graph(&args.graph_a).map_err(|e| format!("{e}"))?;
    let b = load_graph(&args.graph_b).map_err(|e| format!("{e}"))?;
    let merged = consensus_merge(&a, &b);
    eprintln!(
        "merged: {} edges (was {} + {})",
        merged.edge_count(),
        a.edge_count(),
        b.edge_count()
    );
    save_graph(&merged, &args.output).map_err(|e| format!("{e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::{edge::Edge, graph::Graph};

    fn graph_a() -> Graph {
        let mut g = Graph::new();
        g.add_edge(Edge::new("A", "calls", "B"));
        g.add_edge(Edge::new("A", "calls", "C")); // unique to A
        g
    }

    fn graph_b() -> Graph {
        let mut g = Graph::new();
        g.add_edge(Edge::new("A", "calls", "B")); // shared
        g.add_edge(Edge::new("B", "calls", "D")); // unique to B
        g
    }

    #[test]
    fn shared_edge_has_confidence_one() {
        let merged = consensus_merge(&graph_a(), &graph_b());
        let edges = merged.edges();
        let shared = edges
            .iter()
            .find(|e| e.subject == "A" && e.relation == "calls" && e.object == "B")
            .expect("shared edge missing");
        assert!((shared.confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unique_edge_has_confidence_0_7() {
        let merged = consensus_merge(&graph_a(), &graph_b());
        let edges = merged.edges();
        let unique_a = edges
            .iter()
            .find(|e| e.subject == "A" && e.object == "C")
            .expect("unique-A edge missing");
        assert!(
            (unique_a.confidence - 0.7).abs() < 1e-6,
            "expected 0.7, got {}",
            unique_a.confidence
        );
    }
}
