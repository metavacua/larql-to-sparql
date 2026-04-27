pub mod routing_table;
pub mod walk_state;
pub mod template;
pub mod fallback;

/// Residual Stream Graph Walk — projected architecture, memory-accounting only.
///
/// Target architecture once attention is cracked. The forward pass *is* a
/// graph walk:
///
/// - FFN graph:      gate KNN → feature → down KNN → token (348K features, proven)
/// - Attention graph: routing table (352 KB, 44 centroids) — requires cracked attention
/// - Residual stream: walk state connecting them (Markov cursor)
///
/// Three tiers in the target design:
///
/// - Tier A: cached template walk — known template, entity KNN only (<0.1 ms)
/// - Tier B: dynamic graph walk — full routing table lookup (~1–5 ms)
/// - Tier C: free-form fallback — full RS forward pass for anything outside the graph
///
/// # Why this type does not implement `KvStrategy`
///
/// `KvStrategy` promises encode/decode of K/V vectors. Graph Walk does not
/// reconstruct K/V — it replaces the forward pass with graph lookups. Piping
/// a `GraphWalk` through `run_strategy_benchmark` would produce meaningless
/// MSE / cosine numbers, so the encode/decode shape is deliberately absent.
///
/// What this type *does* provide is memory accounting: `memory_bytes` (per
/// conversation) and `shared_bytes` (one-time infrastructure). Use those to
/// populate memory-scaling tables without implying K/V reconstruction.
pub struct GraphWalk {
    /// Vindex size in bytes (shared, not per-conversation).
    pub vindex_bytes: usize,
    /// Routing table size in bytes.
    pub routing_table_bytes: usize,
    /// Number of features in the FFN graph.
    pub num_features: usize,
    /// Number of layers in the model.
    pub num_layers: usize,
}

impl GraphWalk {
    /// Default for Gemma 3-4B based on measured values.
    pub fn gemma_4b() -> Self {
        Self {
            vindex_bytes: 1_500_000_000, // 1.5 GB Q4 vindex
            routing_table_bytes: 360_448, // 352 KB routing table
            num_features: 348_000,
            num_layers: 34,
        }
    }

    /// Create with custom parameters.
    pub fn new(vindex_bytes: usize, routing_table_bytes: usize, num_features: usize, num_layers: usize) -> Self {
        Self {
            vindex_bytes,
            routing_table_bytes,
            num_features,
            num_layers,
        }
    }

    pub fn name(&self) -> &str {
        "RS Graph Walk"
    }

    /// Per-conversation storage: just token IDs (same as Markov RS cold tier).
    pub fn memory_bytes(&self, seq_len: usize) -> usize {
        seq_len * 4 // u32 token IDs
    }

    /// Shared infrastructure: vindex + routing table (one copy, not per-conversation).
    pub fn shared_bytes(&self) -> usize {
        self.vindex_bytes + self.routing_table_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_walk_memory_tiny() {
        let gw = GraphWalk::gemma_4b();

        // At 370K tokens: 370,000 × 4 = 1.48 MB per-conversation
        let mem = gw.memory_bytes(370_000);
        assert_eq!(mem, 370_000 * 4);
        assert!(mem < 2_000_000);
    }

    #[test]
    fn test_graph_walk_shared_infrastructure() {
        let gw = GraphWalk::gemma_4b();
        // ~1.5 GB shared (vindex + routing table)
        assert!(gw.shared_bytes() > 1_000_000_000);
        assert!(gw.shared_bytes() < 2_000_000_000);
    }
}
