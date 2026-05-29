//! Graph diffing — find added, removed, and changed edges between two graphs.

use crate::core::edge::Edge;
use crate::core::graph::Graph;

/// Result of diffing two graphs.
#[derive(Debug)]
pub struct GraphDiff {
    pub added: Vec<Edge>,
    pub removed: Vec<Edge>,
    pub changed: Vec<ChangedEdge>,
}

/// An edge whose confidence or metadata changed between graphs.
#[derive(Debug)]
pub struct ChangedEdge {
    pub old: Edge,
    pub new: Edge,
}

/// Compute the diff between two graphs.
/// `added` = in `new` but not `old`, `removed` = in `old` but not `new`,
/// `changed` = same triple but different confidence, source, metadata, or injection.
pub fn diff(old: &Graph, new: &Graph) -> GraphDiff {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    // Find added and changed
    for edge in new.edges() {
        if !old.exists(&edge.subject, &edge.relation, &edge.object) {
            added.push(edge.clone());
        } else {
            // Same triple exists — check if edge attributes changed.
            let old_edges = old.select(&edge.subject, Some(&edge.relation));
            if let Some(old_edge) = old_edges.iter().find(|e| e.object == edge.object) {
                if edge_changed(old_edge, edge) {
                    changed.push(ChangedEdge {
                        old: (*old_edge).clone(),
                        new: edge.clone(),
                    });
                }
            }
        }
    }

    // Find removed
    for edge in old.edges() {
        if !new.exists(&edge.subject, &edge.relation, &edge.object) {
            removed.push(edge.clone());
        }
    }

    GraphDiff {
        added,
        removed,
        changed,
    }
}

fn edge_changed(old: &Edge, new: &Edge) -> bool {
    (old.confidence - new.confidence).abs() > f64::EPSILON
        || old.source != new.source
        || old.metadata != new.metadata
        || old.injection != new.injection
}
