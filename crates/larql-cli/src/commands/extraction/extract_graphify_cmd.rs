use std::{collections::HashMap, path::PathBuf};

use clap::Args;
use larql_core::{
    core::{edge::Edge, enums::SourceType, graph::Graph},
    io::save_graph,
};
use serde_json::Value;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum GraphifyError {
    #[error("missing 'nodes' array in graphify JSON")]
    MissingNodes,
    #[error("missing 'links' array in graphify JSON")]
    MissingLinks,
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("graph: {0}")]
    Graph(String),
}

/// φ-transform: convert a NetworkX node-link JSON value into a `larql_core::Graph`.
pub fn phi_transform(v: &Value) -> Result<Graph, GraphifyError> {
    let nodes_arr = v["nodes"].as_array().ok_or(GraphifyError::MissingNodes)?;
    let links_arr = v["links"].as_array().ok_or(GraphifyError::MissingLinks)?;

    // Build id → label map (id may be int or string in the JSON).
    let mut id_to_label: HashMap<String, String> = HashMap::new();
    for node in nodes_arr {
        let id = node["id"].to_string(); // stringify (handles int and str)
        let label = node["label"].as_str().unwrap_or("unknown").to_string();
        id_to_label.insert(id, label.clone());
    }

    let mut graph = Graph::new();

    // Node metadata edges
    for node in nodes_arr {
        let label = node["label"].as_str().unwrap_or("unknown");
        if let Some(kind) = node["kind"].as_str() {
            let mut e = Edge::new(label, "has_kind", kind);
            e.source = SourceType::Ast;
            e.confidence = 1.0;
            graph.add_edge(e);
        }
        if let Some(source_file) = node["source_file"].as_str() {
            let mut e = Edge::new(label, "defined_in", source_file);
            e.source = SourceType::Ast;
            e.confidence = 1.0;
            graph.add_edge(e);
        }
    }

    // Structural edges from links
    for link in links_arr {
        let src_id = link["source"].to_string();
        let tgt_id = link["target"].to_string();
        let relation = link["type"]
            .as_str()
            .or_else(|| link["relation"].as_str())
            .unwrap_or("references");
        let src_label = id_to_label
            .get(&src_id)
            .map(|s| s.as_str())
            .unwrap_or(src_id.as_str());
        let tgt_label = id_to_label
            .get(&tgt_id)
            .map(|s| s.as_str())
            .unwrap_or(tgt_id.as_str());
        let mut e = Edge::new(src_label, relation, tgt_label);
        e.source = SourceType::Ast;
        e.confidence = 1.0;
        graph.add_edge(e);
    }

    Ok(graph)
}

#[derive(Args)]
pub struct ExtractGraphifyArgs {
    /// Path to the graphify node-link JSON file.
    input: PathBuf,

    /// Output .larql.json path.
    #[arg(short, long, default_value = "graph.larql.json")]
    output: PathBuf,
}

pub fn run(args: ExtractGraphifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(&args.input)?;
    let v: Value = serde_json::from_str(&text)?;
    let graph = phi_transform(&v)?;
    eprintln!(
        "φ-transform: {} nodes, {} edges → {}",
        graph.node_count(),
        graph.edge_count(),
        args.output.display()
    );
    save_graph(&graph, &args.output).map_err(|e| format!("{e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_json() -> serde_json::Value {
        serde_json::json!({
            "nodes": [
                {"id": 0, "label": "mymod", "kind": "module", "source_file": "src/lib.rs"},
                {"id": 1, "label": "helper_fn", "kind": "function", "source_file": "src/lib.rs"},
                {"id": 2, "label": "std::fmt", "kind": "external"}
            ],
            "links": [
                {"source": 0, "target": 1, "type": "contains"},
                {"source": 1, "target": 2, "type": "imports"}
            ]
        })
    }

    #[test]
    fn phi_transform_produces_contains_edge() {
        let graph = phi_transform(&fixture_json()).unwrap();
        let edges = graph.edges();
        assert!(
            edges.iter().any(|e| e.relation == "contains"),
            "expected a 'contains' edge"
        );
    }

    #[test]
    fn phi_transform_produces_defined_in_edge() {
        let graph = phi_transform(&fixture_json()).unwrap();
        let edges = graph.edges();
        assert!(
            edges.iter().any(|e| e.relation == "defined_in"),
            "expected a 'defined_in' edge for source_file"
        );
    }
}
