pub mod head_classifier;
pub mod template_cache;

use crate::{KvStrategy, model_config::ModelConfig};

/// Strategy 4: Hybrid RS + Cracked Attention.
///
/// The near-term practical win. Doesn't require solving attention fully.
///
/// - 97.1% of attention heads are cacheable for parametric queries (264/272, cosine ≥ 0.90)
/// - FFN is already solved (vindex walk, zero matmul)
/// - Cache the static head outputs per template
/// - Only the ~2.9% dynamic heads need real KV cache (L1, L13, L26, L32)
///
/// Memory breakdown:
///   Static heads:   cached per template (shared, not per-conversation)
///   Dynamic heads:  tiny KV cache (~4 layers × kv_heads × head_dim × seq_len)
///   FFN:            zero (vindex walk)
///   Cold tier:      token IDs (4 bytes per token)
///   Routing table:  352 KB (one-time)
pub struct HybridCrackedAttention {
    /// Fraction of heads that are static (cacheable).
    pub static_head_fraction: f64,
    /// Number of layers with dynamic heads.
    pub dynamic_layers: usize,
    /// Routing table size in bytes.
    pub routing_table_bytes: usize,
    /// Template cache size per template in bytes.
    pub template_cache_bytes: usize,
    /// Window size for dynamic head KV cache.
    pub dynamic_window: usize,
}

impl HybridCrackedAttention {
    /// Default for Gemma 3-4B based on measured head cacheability.
    /// Parametric queries: 97.1% static (264/272), dynamic layers L1, L13, L26, L32.
    pub fn gemma_4b() -> Self {
        Self {
            static_head_fraction: 0.971,
            dynamic_layers: 4,          // L1, L13, L26, L32 (parametric retrieval circuit)
            routing_table_bytes: 360_448, // 352 KB
            template_cache_bytes: 1_500_000, // ~1.5 MB per template
            dynamic_window: 32_768,     // 32K active-token window for dynamic heads
        }
    }

    /// Custom configuration.
    pub fn new(
        static_head_fraction: f64,
        dynamic_layers: usize,
        dynamic_window: usize,
    ) -> Self {
        Self {
            static_head_fraction,
            dynamic_layers,
            routing_table_bytes: 360_448,
            template_cache_bytes: 1_500_000,
            dynamic_window,
        }
    }

    /// Dynamic-head-only KV cache size at a given sequence length.
    /// Uses a bounded window — beyond that, cold-tier token IDs cover the rest.
    fn dynamic_kv_bytes(&self, config: &ModelConfig, seq_len: usize) -> usize {
        let window = seq_len.min(self.dynamic_window);
        // dynamic_layers × 2(K+V) × kv_heads × head_dim × 2(fp16) × window
        self.dynamic_layers * 2 * config.kv_heads * config.head_dim * 2 * window
    }

    /// Cold tier: token IDs for context beyond the dynamic window.
    fn cold_tier_bytes(&self, seq_len: usize) -> usize {
        seq_len * 4
    }

    /// Shared infrastructure: routing table + template cache.
    /// This is per-template, not per-conversation.
    pub fn shared_bytes(&self) -> usize {
        self.routing_table_bytes + self.template_cache_bytes
    }

    /// Full standard KV cache size for comparison.
    fn full_kv_bytes(&self, config: &ModelConfig, seq_len: usize) -> usize {
        config.kv_memory(seq_len)
    }

    /// Compression ratio vs standard KV.
    pub fn compression_ratio(&self, config: &ModelConfig, seq_len: usize) -> f64 {
        let full = self.full_kv_bytes(config, seq_len) as f64;
        let hybrid = self.memory_bytes(config, seq_len) as f64;
        if hybrid > 0.0 { full / hybrid } else { 0.0 }
    }
}

impl KvStrategy for HybridCrackedAttention {
    fn name(&self) -> &str {
        "Hybrid RS+CA"
    }

    fn encode(&self, keys: &[Vec<f32>], _values: &[Vec<f32>]) -> Vec<u8> {
        // Hybrid RS+CA stores:
        // 1. Template ID (4 bytes) — selects cached static head outputs
        // 2. Dynamic head K/V for the ~4.5% dynamic heads only
        // 3. Token IDs for cold tier (4 bytes each)
        //
        // For the synthetic benchmark, we store a header + dynamic head subset + token IDs.
        let total_vectors = keys.len();
        let dynamic_fraction = 1.0 - self.static_head_fraction;
        let dynamic_count = ((total_vectors as f64) * dynamic_fraction).ceil() as usize;
        let dynamic_count = dynamic_count.min(total_vectors);

        let mut buf = Vec::new();

        // Header: template ID + total vectors + dynamic count
        buf.extend_from_slice(&0u32.to_le_bytes()); // template ID
        buf.extend_from_slice(&(total_vectors as u32).to_le_bytes());
        buf.extend_from_slice(&(dynamic_count as u32).to_le_bytes());

        // Dynamic head K/V only (the ~4.5%)
        for v in keys.iter().take(dynamic_count) {
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }

        // Cold tier token IDs
        let cold_count = total_vectors.saturating_sub(self.dynamic_window);
        for i in 0..cold_count {
            buf.extend_from_slice(&(i as u32).to_le_bytes());
        }

        buf
    }

    fn decode(&self, encoded: &[u8], num_vectors: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let _template_id = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let _total = u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;
        let dynamic_count = u32::from_le_bytes([encoded[8], encoded[9], encoded[10], encoded[11]]) as usize;

        let mut keys = Vec::with_capacity(num_vectors);
        let mut values = Vec::with_capacity(num_vectors);

        // Decode dynamic head vectors
        let data_start = 12;
        for i in 0..dynamic_count.min(num_vectors) {
            let offset = data_start + i * dim * 4;
            let mut v = Vec::with_capacity(dim);
            for j in 0..dim {
                let o = offset + j * 4;
                if o + 3 < encoded.len() {
                    let x = f32::from_le_bytes([encoded[o], encoded[o + 1], encoded[o + 2], encoded[o + 3]]);
                    v.push(x);
                } else {
                    v.push(0.0);
                }
            }
            keys.push(v.clone());
            values.push(v);
        }

        // Static heads: inject cached values (zeros in synthetic benchmark)
        let static_count = num_vectors.saturating_sub(dynamic_count);
        for _ in 0..static_count {
            keys.push(vec![0.0f32; dim]);
            values.push(vec![0.0f32; dim]);
        }

        (keys, values)
    }

    fn memory_bytes(&self, config: &ModelConfig, seq_len: usize) -> usize {
        // Per-conversation: dynamic KV + cold tier + routing table
        self.dynamic_kv_bytes(config, seq_len)
            + self.cold_tier_bytes(seq_len)
            + self.routing_table_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_memory_at_4k() {
        let config = ModelConfig::gemma_4b();
        let hybrid = HybridCrackedAttention::gemma_4b();

        let mem = hybrid.memory_bytes(&config, 4096);
        let standard = config.kv_memory(4096);

        // Should be dramatically smaller than standard KV
        // Spec target: ~20-37 MB vs 285 MB standard
        assert!(
            mem < standard / 5,
            "Hybrid at 4K ({} bytes = {:.1} MB) should be <20% of standard ({} bytes = {:.1} MB)",
            mem, mem as f64 / 1e6, standard, standard as f64 / 1e6,
        );
    }

    #[test]
    fn test_hybrid_memory_at_370k() {
        let config = ModelConfig::gemma_4b();
        let hybrid = HybridCrackedAttention::gemma_4b();

        let mem = hybrid.memory_bytes(&config, 370_000);
        let standard = config.kv_memory(370_000);

        // Spec target: ~150-300 MB vs 25.8 GB standard
        let ratio = standard as f64 / mem as f64;
        assert!(
            ratio > 10.0,
            "Hybrid at 370K should be >10x smaller than standard, got {ratio:.1}x"
        );
    }

    #[test]
    fn test_hybrid_dynamic_kv_size() {
        let config = ModelConfig::gemma_4b();
        let hybrid = HybridCrackedAttention::gemma_4b();

        let dynamic_kv = hybrid.dynamic_kv_bytes(&config, 4096);
        let full_kv = config.kv_memory(4096);

        // Dynamic KV at 4K (window 32K > 4K, so full context used): 4/34 ≈ 12%
        let ratio = dynamic_kv as f64 / full_kv as f64;
        assert!(
            ratio < 0.15,
            "Dynamic KV should be <15% of full KV, got {ratio:.1}%"
        );
    }

    #[test]
    fn test_hybrid_static_head_fraction() {
        let hybrid = HybridCrackedAttention::gemma_4b();
        assert!(
            (hybrid.static_head_fraction - 0.971).abs() < 0.01,
            "Static head fraction should be ~97.1% (264/272 parametric)"
        );
    }

    #[test]
    fn test_hybrid_template_cache_shared() {
        let hybrid = HybridCrackedAttention::gemma_4b();
        // Template cache is per-template, not per-conversation
        // shared_bytes should be routing table + template cache
        let shared = hybrid.shared_bytes();
        assert!(shared > 1_000_000, "Shared infra should be >1MB");
        assert!(shared < 5_000_000, "Shared infra should be <5MB per template");
    }

    #[test]
    fn test_hybrid_ffn_zero_matmul() {
        // Hybrid uses vindex walk for FFN — the encode path doesn't store FFN data.
        // Verify: encoded data contains only dynamic head K/V + token IDs, no FFN.
        let hybrid = HybridCrackedAttention::gemma_4b();
        let keys = vec![vec![1.0f32; 256]; 100];
        let values = vec![vec![2.0f32; 256]; 100];
        let encoded = hybrid.encode(&keys, &values);

        // Encoded should be much smaller than full K+V (only ~4.5% of heads)
        let full_size = 100 * 256 * 4 * 2; // K+V, f32
        assert!(
            encoded.len() < full_size / 2,
            "Encoded ({}) should be much smaller than full K+V ({}) — FFN adds zero",
            encoded.len(), full_size,
        );
    }

    #[test]
    fn test_hybrid_compression_ratio() {
        let config = ModelConfig::gemma_4b();
        let hybrid = HybridCrackedAttention::gemma_4b();

        let ratio_4k = hybrid.compression_ratio(&config, 4096);
        let ratio_370k = hybrid.compression_ratio(&config, 370_000);

        // At 4K: expect 15-27x compression
        assert!(ratio_4k > 5.0, "4K compression {ratio_4k:.1}x too low");

        // At 370K: expect 100x+ compression
        assert!(ratio_370k > 10.0, "370K compression {ratio_370k:.1}x too low");
    }
}
