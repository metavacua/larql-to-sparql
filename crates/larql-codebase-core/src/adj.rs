use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};
use serde::{Deserialize, Serialize};

/// Maps node names to stable sorted indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeIndex {
    pub names: Vec<String>,
    pub index: BTreeMap<String, usize>,
}

/// Sparse adjacency list: n×n matrix represented as (row, col, weight) triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseAdj {
    pub n: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

/// Build a deterministic node index from raw edges (subject, object, confidence).
pub fn build_node_index(edges: &[(&str, &str, f64)]) -> NodeIndex {
    let mut name_set = BTreeMap::<String, ()>::new();
    for &(s, o, _) in edges {
        name_set.insert(s.to_string(), ());
        name_set.insert(o.to_string(), ());
    }
    // BTreeMap iterates in sorted order → deterministic
    let names: Vec<String> = name_set.into_keys().collect();
    let index: BTreeMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    NodeIndex { names, index }
}

/// Build a sparse adjacency from raw edges and a pre-built NodeIndex.
pub fn build_adjacency(edges: &[(&str, &str, f64)], idx: &NodeIndex) -> SparseAdj {
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    for &(s, o, w) in edges {
        if let (Some(&i), Some(&j)) = (idx.index.get(s), idx.index.get(o)) {
            entries.push((i, j, w));
        }
    }
    SparseAdj {
        n: idx.names.len(),
        entries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> Vec<(&'static str, &'static str, f64)> {
        vec![("A", "B", 1.0), ("B", "C", 1.0), ("C", "A", 1.0)]
    }

    #[test]
    fn node_index_covers_all_nodes() {
        let edges = triangle();
        let idx = build_node_index(&edges);
        assert_eq!(idx.names.len(), 3);
        assert!(idx.index.contains_key("A"));
        assert!(idx.index.contains_key("B"));
        assert!(idx.index.contains_key("C"));
    }

    #[test]
    fn adjacency_has_correct_entry_count() {
        let edges = triangle();
        let idx = build_node_index(&edges);
        let adj = build_adjacency(&edges, &idx);
        assert_eq!(adj.n, 3);
        assert_eq!(adj.entries.len(), 3);
    }

    #[test]
    fn node_index_is_sorted() {
        let edges = vec![("Z", "A", 1.0), ("M", "B", 1.0)];
        let idx = build_node_index(&edges);
        assert_eq!(idx.names, vec!["A", "B", "M", "Z"]);
    }
}
