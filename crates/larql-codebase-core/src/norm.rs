use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::adj::SparseAdj;

/// Symmetrically-normalised adjacency: D^{-½} A D^{-½}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormAdj {
    pub n: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

/// Compute D^{-½} A D^{-½} where degree counts weighted edge contributions.
/// For each (i, j, w): entry value = w / sqrt(deg[i] * deg[j]).
pub fn symmetric_normalise(adj: &SparseAdj) -> NormAdj {
    let mut deg = alloc::vec![0.0_f64; adj.n];
    for &(i, j, w) in &adj.entries {
        deg[i] += w;
        deg[j] += w; // treat as undirected for normalisation
    }
    let inv_sqrt: Vec<f64> = deg
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / libm::sqrt(d) } else { 0.0 })
        .collect();
    let entries = adj
        .entries
        .iter()
        .map(|&(i, j, w)| (i, j, w * inv_sqrt[i] * inv_sqrt[j]))
        .collect();
    NormAdj { n: adj.n, entries }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adj::SparseAdj;

    fn star_adj() -> SparseAdj {
        // Hub node 0 connects to 3 leaves: 1, 2, 3. Degrees: hub=3, leaves=1.
        SparseAdj {
            n: 4,
            entries: vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)],
        }
    }

    #[test]
    fn normalised_hub_leaf_edge_value() {
        let adj = star_adj();
        let norm = symmetric_normalise(&adj);
        // deg[0]=3, deg[1]=1 → weight = 1/sqrt(3*1) ≈ 0.5774
        let (_, _, w) = norm.entries[0];
        let expected = 1.0_f64 / libm::sqrt(3.0_f64 * 1.0_f64);
        assert!((w - expected).abs() < 1e-10, "got {w}, expected {expected}");
    }

    #[test]
    fn normalised_entry_count_matches_input() {
        let adj = star_adj();
        let norm = symmetric_normalise(&adj);
        assert_eq!(norm.entries.len(), adj.entries.len());
    }

    #[test]
    fn normalised_weighted_edge() {
        // Single edge with weight 0.5 between nodes with degree 0.5 each
        // Expected: 0.5 * (1/sqrt(0.5)) * (1/sqrt(0.5)) = 0.5 * 2 * 2 = 2.0
        // OR: Create a case with weight 0.5 where both nodes have degree 2 (two edges each)
        // Node 0: edges (0,1,0.5) and (0,2,1.5) → degree 2.0
        // Node 1: edges (0,1,0.5) and (1,3,1.5) → degree 2.0
        // Expected for (0,1): 0.5 * (1/sqrt(2)) * (1/sqrt(2)) = 0.5 * 0.5 = 0.25
        let adj = SparseAdj {
            n: 4,
            entries: vec![
                (0, 1, 0.5),
                (0, 2, 1.5),
                (1, 3, 1.5),
            ],
        };
        let norm = symmetric_normalise(&adj);
        // First entry should be (0, 1) with weight 0.5
        let (_, _, w) = norm.entries[0];
        let expected = 0.5 * (1.0 / libm::sqrt(2.0)) * (1.0 / libm::sqrt(2.0));
        assert!((w - expected).abs() < 1e-10, "got {w}, expected {expected}");
    }
}
