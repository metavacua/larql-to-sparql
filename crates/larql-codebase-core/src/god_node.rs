use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};

use crate::adj::{NodeIndex, SparseAdj};

/// Degree distribution statistics for god-node detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegreeStats {
    pub mean: f64,
    pub std: f64,
    pub threshold: f64,
}

/// Compute degree statistics over all nodes in adj.
/// Degree of node i = sum of (in-degree + out-degree) counting each edge once per direction.
pub fn degree_stats(adj: &SparseAdj) -> DegreeStats {
    let mut deg = alloc::vec![0.0_f64; adj.n];
    for &(i, j, _) in &adj.entries {
        deg[i] += 1.0;
        deg[j] += 1.0;
    }
    let n = deg.len() as f64;
    let mean = deg.iter().sum::<f64>() / n;
    let variance = deg.iter().map(|&d| (d - mean) * (d - mean)).sum::<f64>() / n;
    let std = libm::sqrt(variance);
    DegreeStats {
        mean,
        std,
        threshold: mean + 3.0 * std,
    }
}

/// Return names of nodes whose total degree exceeds `mean + sigma * std`.
pub fn god_nodes(adj: &SparseAdj, idx: &NodeIndex, sigma: f64) -> Vec<String> {
    let mut deg = alloc::vec![0.0_f64; adj.n];
    for &(i, j, _) in &adj.entries {
        deg[i] += 1.0;
        deg[j] += 1.0;
    }
    let stats = degree_stats(adj);
    let threshold = stats.mean + sigma * stats.std;
    idx.names
        .iter()
        .enumerate()
        .filter(|&(i, _)| deg[i] > threshold)
        .map(|(_, name)| name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adj::{NodeIndex, SparseAdj};
    use alloc::{
        collections::BTreeMap,
        string::{String, ToString},
        vec,
    };

    fn hub_graph() -> (SparseAdj, NodeIndex) {
        // Hub "H" connects to 4 leaves. Leaves have degree 1, hub degree 4.
        let names: Vec<String> = vec![
            "H".to_string(),
            "L1".to_string(),
            "L2".to_string(),
            "L3".to_string(),
            "L4".to_string(),
        ];
        let index: BTreeMap<String, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let idx = NodeIndex { names, index };
        let adj = SparseAdj {
            n: 5,
            entries: vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)],
        };
        (adj, idx)
    }

    #[test]
    fn hub_is_a_god_node() {
        let (adj, idx) = hub_graph();
        let gods = god_nodes(&adj, &idx, 1.0);
        assert!(gods.contains(&"H".to_string()), "Hub must be a god node");
    }

    #[test]
    fn leaves_are_not_god_nodes() {
        let (adj, idx) = hub_graph();
        let gods = god_nodes(&adj, &idx, 1.0);
        for leaf in ["L1", "L2", "L3", "L4"] {
            assert!(
                !gods.contains(&leaf.to_string()),
                "{leaf} should not be a god node"
            );
        }
    }
}
