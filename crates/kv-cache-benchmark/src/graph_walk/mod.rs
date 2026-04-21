pub mod routing_table;
pub mod walk_state;
pub mod template;
pub mod fallback;

use crate::{KvStrategy, model_config::ModelConfig};

/// Strategy 5: Residual Stream Graph Walk.
///
/// Target architecture once attention is cracked. Eliminates the forward pass itself.
/// The forward pass IS a graph walk:
///   FFN graph:       gate KNN → feature → down KNN → token (348K features in vindex, proven)
///   Attention graph:  routing table (352KB, 44 centroids) — requires cracked attention (TODO)
///   Residual stream:  the walk state connecting them (Markov cursor)
///
/// Current status: FFN graph walk is proven. Attention elimination requires cracked attention
/// which is not yet implemented. Until then Tier C (free-form) falls back to Markov RS.
///
/// Three tiers:
///   Tier A: cached template walk — known template, entity KNN only (<0.1ms)
///   Tier B: dynamic graph walk — full routing table lookup (~1-5ms)
///   Tier C: Markov RS fallback — full RS forward pass for anything outside the graph
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

    /// Per-conversation storage: just token IDs (same as Markov RS cold tier).
    fn conversation_bytes(&self, seq_len: usize) -> usize {
        seq_len * 4 // u32 token IDs
    }

    /// Shared infrastructure: vindex + routing table (one copy, not per-conversation).
    pub fn shared_bytes(&self) -> usize {
        self.vindex_bytes + self.routing_table_bytes
    }
}

impl KvStrategy for GraphWalk {
    fn name(&self) -> &str {
        "RS Graph Walk"
    }

    fn encode(&self, keys: &[Vec<f32>], _values: &[Vec<f32>]) -> Vec<u8> {
        // Graph Walk doesn't store K/V vectors at all.
        // Per-conversation state is token IDs only.
        let num_tokens = keys.len();
        let mut buf = Vec::with_capacity(4 + num_tokens * 4);
        buf.extend_from_slice(&(num_tokens as u32).to_le_bytes());
        for i in 0..num_tokens {
            buf.extend_from_slice(&(i as u32).to_le_bytes()); // Simulated token IDs
        }
        buf
    }

    fn decode(&self, _encoded: &[u8], num_vectors: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        // Graph Walk produces predictions via graph lookup, not K/V reconstruction.
        // For the synthetic benchmark, we return zeros to indicate "no K/V vectors".
        // Accuracy is measured differently: top-1 token match, not MSE on K/V.
        let keys = vec![vec![0.0f32; dim]; num_vectors];
        let values = vec![vec![0.0f32; dim]; num_vectors];
        (keys, values)
    }

    fn memory_bytes(&self, _config: &ModelConfig, seq_len: usize) -> usize {
        // Per-conversation: token IDs only.
        // The vindex is shared infrastructure — reported separately.
        self.conversation_bytes(seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_walk_memory_tiny() {
        let gw = GraphWalk::gemma_4b();
        let config = ModelConfig::gemma_4b();

        // At 370K tokens: 370,000 × 4 = 1.48 MB per-conversation
        let mem = gw.memory_bytes(&config, 370_000);
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

    #[test]
    fn test_graph_walk_no_kv_stored() {
        let gw = GraphWalk::gemma_4b();
        let keys = vec![vec![1.0f32; 256]; 100];
        let values = vec![vec![2.0f32; 256]; 100];
        let encoded = gw.encode(&keys, &values);
        // Should be just header + token IDs, not full vectors
        assert_eq!(encoded.len(), 4 + 100 * 4);
    }
}
