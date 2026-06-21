use crate::core::graph::Graph;

#[cfg(not(target_arch = "wasm32"))]
use crate::core::graph::GraphError;
#[cfg(not(target_arch = "wasm32"))]
use crate::io::load_graph;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// An in-process graph query engine that answers prompts by describing graph entities.
///
/// Queries are expressed as simple prompts (`"DESCRIBE <entity>"`) and resolved
/// directly against an in-memory `Graph` — no HTTP or model inference required.
pub struct VindexProvider {
    graph: Graph,
}

impl VindexProvider {
    /// Create a `VindexProvider` from an existing in-memory graph.
    pub fn from_graph(graph: Graph) -> Self {
        Self { graph }
    }

    /// Load a `VindexProvider` from a `.larql.json` file on disk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: &Path) -> Result<Self, GraphError> {
        let graph = load_graph(path)?;
        Ok(Self::from_graph(graph))
    }

    /// Format a DescribeResult into a newline-separated edge list.
    fn format_result(r: &crate::core::graph::DescribeResult) -> String {
        let mut parts = Vec::new();
        for e in &r.outgoing {
            parts.push(format!("{} --[{}]--> {}", e.subject, e.relation, e.object));
        }
        for e in &r.incoming {
            parts.push(format!("{} --[{}]--> {}", e.subject, e.relation, e.object));
        }
        parts.join("\n")
    }

    /// Answer a prompt by querying the graph.
    ///
    /// Supported patterns:
    /// - `"DESCRIBE <entity>"` — returns all edges involving `<entity>`.
    /// - Any other prompt — falls back to describing the first whitespace-separated word.
    ///
    /// Returns an empty string when no edges are found.
    pub fn complete(&self, prompt: &str) -> String {
        if let Some(entity) = prompt.strip_prefix("DESCRIBE ") {
            let r = self.graph.describe(entity.trim());
            if r.outgoing.is_empty() && r.incoming.is_empty() {
                return String::new();
            }
            return Self::format_result(&r);
        }

        // Fallback: describe the first word of the prompt.
        let first = prompt.split_whitespace().next().unwrap_or("");
        if first.is_empty() {
            return String::new();
        }
        let r = self.graph.describe(first);
        if r.outgoing.is_empty() && r.incoming.is_empty() {
            return String::new();
        }
        Self::format_result(&r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::edge::Edge;

    #[test]
    fn complete_describe_returns_edges() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("alpha", "calls", "beta"));
        let provider = VindexProvider::from_graph(g);
        let result = provider.complete("DESCRIBE alpha");
        assert!(result.contains("alpha"), "response should mention alpha");
        assert!(result.contains("calls"), "response should include relation");
    }

    #[test]
    fn complete_describe_unknown_returns_empty() {
        let g = Graph::new();
        let provider = VindexProvider::from_graph(g);
        let result = provider.complete("DESCRIBE unknown_entity_xyz");
        assert_eq!(result, "", "unknown entity should return empty string");
    }
}
