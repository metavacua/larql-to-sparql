use serde::{Deserialize, Serialize};

/// Model architecture configuration derived from graph statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
}

/// Return the smallest power of two >= n, clamped to minimum `min`.
fn next_power_of_two_min(n: usize, min: usize) -> usize {
    let mut p = min;
    while p < n {
        p *= 2;
    }
    p
}

/// Size a BitNet architecture from graph statistics.
///
/// Rules (spec §4.2):
/// - hidden_size = next power of 2 >= ceil(sqrt(n_nodes)), minimum 64
/// - n_layers    = max(2, ceil(log2(n_edges / n_nodes)))
/// - n_heads     = max(4, hidden_size / 64)
/// - head_dim    = 64 (BitNet standard)
/// - ffn_dim     = hidden_size * 4
pub fn size_architecture(n_nodes: usize, n_edges: usize, _god_node_count: usize) -> ArchConfig {
    let sqrt_n = libm::ceil(libm::sqrt(n_nodes as f64)) as usize;
    let hidden_size = next_power_of_two_min(sqrt_n, 64);

    let avg_degree = if n_nodes > 0 {
        (n_edges as f64) / (n_nodes as f64)
    } else {
        1.0
    };
    // log2 of avg_degree, at least 2.0 to ensure n_layers >= 2
    let log2_deg = if avg_degree > 1.0 {
        libm::ceil(libm::log2(avg_degree)) as usize
    } else {
        0
    };
    let n_layers = log2_deg.max(2);

    let n_heads = (hidden_size / 64).max(4);
    let head_dim = 64;
    let ffn_dim = hidden_size * 4;

    ArchConfig {
        hidden_size,
        n_layers,
        n_heads,
        head_dim,
        ffn_dim,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_size_is_power_of_two() {
        let cfg = size_architecture(1000, 5000, 10);
        assert!(cfg.hidden_size.is_power_of_two());
        assert!(cfg.hidden_size >= 64);
    }

    #[test]
    fn head_dim_always_64() {
        let cfg = size_architecture(500, 2000, 5);
        assert_eq!(cfg.head_dim, 64);
    }

    #[test]
    fn ffn_is_4x_hidden() {
        let cfg = size_architecture(500, 2000, 5);
        assert_eq!(cfg.ffn_dim, cfg.hidden_size * 4);
    }

    #[test]
    fn n_layers_at_least_2() {
        let cfg = size_architecture(10, 10, 0);
        assert!(cfg.n_layers >= 2);
    }

    #[test]
    fn n_heads_at_least_4() {
        let cfg = size_architecture(10, 10, 0);
        assert!(cfg.n_heads >= 4);
    }

    #[test]
    fn min_graph_doesnt_panic() {
        let cfg = size_architecture(0, 0, 0);
        assert!(cfg.hidden_size >= 64);
        assert!(cfg.n_layers >= 2);
    }
}
