/// Boundary Residual Stream strategy.
///
/// The production form of the Markov RS insight from the Python experiments
/// (`unlimited_engine.py`, `rs_generator.py`). Keeps context fully unbounded
/// without O(context) memory growth.
///
/// ## Three tiers
///
/// ```text
/// ┌──────────────────────┬─────────────────────┬──────────────────┐
/// │  Boundary residual   │   Hot window         │  New token       │
/// │  1 vec / layer       │  W vecs / layer      │  embed only      │
/// │  fixed ~340 KB       │  fixed ~11 MB (W=32) │                  │
/// └──────────────────────┴─────────────────────┴──────────────────┘
///        Cold tier: token IDs only (4 bytes/token)
/// ```
///
/// - **Hot window** (`W` tokens): full f32 residuals per layer, recomputed
///   into K/V at each decode step. W is small (default 32) because the
///   boundary residual encodes all prior context.
///
/// - **Boundary residual**: one residual vector per layer at the window edge.
///   This is the Markov chain state — it encodes all information from all
///   tokens before the hot window. When the hot window slides forward, the
///   old boundary is discarded and the new one saved.
///   Size: `num_layers × hidden_dim × 4 bytes` ≈ 340 KB on Gemma 3-4B.
///
/// - **Cold tier**: token IDs only (u32, 4 bytes). No residuals stored.
///   When K/V for cold tokens is needed, replay forward from the boundary
///   residual through the cold token IDs (same as Python `extend()`).
///   Cost: 4 bytes/token regardless of model size.
///
/// ## Memory at scale (Gemma 3-4B, W=32, hidden=2560, 34 layers)
///
/// ```text
/// Context    Hot (W=32)   Boundary   Cold IDs   Total
/// ──────────────────────────────────────────────────────
///     512      11.2 MB    340 KB       2 KB    11.5 MB
///      4K      11.2 MB    340 KB      16 KB    11.6 MB
///     32K      11.2 MB    340 KB     128 KB    11.7 MB
///    131K      11.2 MB    340 KB     510 KB    12.1 MB
///    370K      11.2 MB    340 KB    1.48 MB    13.0 MB
/// ```
///
/// The total stays flat (~11-13 MB) regardless of context length —
/// unlike standard KV which grows to 25.8 GB at 370K.
///
/// ## Contrast with MarkovResidual
///
/// `MarkovResidual` (W=512) stores full residuals for all 512 hot-window
/// positions ≈ 178 MB fixed. `BoundaryResidual` (W=32) uses the boundary
/// vector to keep the window tiny: 11 MB. The trade-off is a forward replay
/// pass when accessing cold K/V (amortised — only needed when the cold token
/// becomes relevant to a decode query).

use crate::{KvStrategy, model_config::ModelConfig};

/// Strategy 6: Boundary Residual Stream.
///
/// Small hot window (W=32) + one boundary residual per layer + cold token IDs.
pub struct BoundaryResidual {
    /// Active window size. Default: 32 tokens.
    pub window_size: usize,
}

impl BoundaryResidual {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Default for Gemma 3-4B: window=32, matching the Python experiment window.
    pub fn gemma_4b() -> Self {
        Self::new(32)
    }

    /// Hot-window memory: W tokens × num_layers × hidden_dim × f32.
    pub fn hot_window_bytes(&self, config: &ModelConfig) -> usize {
        self.window_size * config.layers * config.hidden_dim * 4
    }

    /// Boundary residual memory: 1 vec per layer × hidden_dim × f32.
    /// Fixed cost ~340 KB on Gemma 3-4B; negligible in the total.
    pub fn boundary_bytes(&self, config: &ModelConfig) -> usize {
        config.layers * config.hidden_dim * 4
    }

    /// Cold-tier token ID memory: one u32 per cold token.
    pub fn cold_id_bytes(&self, seq_len: usize) -> usize {
        seq_len.saturating_sub(self.window_size) * 4
    }
}

impl KvStrategy for BoundaryResidual {
    fn name(&self) -> &str {
        "Boundary Residual Stream"
    }

    fn encode(&self, keys: &[Vec<f32>], _values: &[Vec<f32>]) -> Vec<u8> {
        // Simulate: store hot window + boundary residual + cold token IDs.
        // For the synthetic benchmark we emit a realistic-sized header.
        let total = keys.len();
        let window = total.min(self.window_size);
        let cold_count = total.saturating_sub(self.window_size);
        let dim = keys.first().map_or(0, |v| v.len());

        let mut buf = Vec::new();
        buf.extend_from_slice(&(total as u32).to_le_bytes());
        buf.extend_from_slice(&(window as u32).to_le_bytes());

        // Boundary residual: last hot-window key as proxy (dim × f32).
        let boundary_idx = if total > self.window_size { total - self.window_size - 1 } else { 0 };
        if !keys.is_empty() {
            for &x in &keys[boundary_idx] {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }

        // Hot window residuals (last W positions).
        let start = total.saturating_sub(window);
        for v in &keys[start..] {
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }

        // Cold tier: sequential token IDs.
        for i in 0..cold_count {
            buf.extend_from_slice(&(i as u32).to_le_bytes());
        }

        buf
    }

    fn decode(&self, encoded: &[u8], num_vectors: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        if encoded.len() < 8 {
            return (vec![vec![0.0; dim]; num_vectors], vec![vec![0.0; dim]; num_vectors]);
        }
        let total = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        let window = u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;
        let cold_count = total.saturating_sub(window);

        // Cold-tier tokens: replay from boundary (simulated as boundary vector for all cold).
        let boundary_start = 8;
        let mut boundary = vec![0.0f32; dim];
        for j in 0..dim {
            let o = boundary_start + j * 4;
            if o + 4 <= encoded.len() {
                boundary[j] = f32::from_le_bytes([encoded[o], encoded[o+1], encoded[o+2], encoded[o+3]]);
            }
        }

        let hot_start = boundary_start + dim * 4;
        let mut keys = Vec::with_capacity(num_vectors);
        let mut values = Vec::with_capacity(num_vectors);

        // Cold positions: reconstructed from boundary replay (approximated here).
        for _ in 0..cold_count {
            keys.push(boundary.clone());
            values.push(boundary.clone());
        }

        // Hot window: decode stored residuals.
        for i in 0..window.min(num_vectors.saturating_sub(cold_count)) {
            let offset = hot_start + i * dim * 4;
            let mut v = vec![0.0f32; dim];
            for j in 0..dim {
                let o = offset + j * 4;
                if o + 4 <= encoded.len() {
                    v[j] = f32::from_le_bytes([encoded[o], encoded[o+1], encoded[o+2], encoded[o+3]]);
                }
            }
            keys.push(v.clone());
            values.push(v);
        }

        (keys, values)
    }

    fn memory_bytes(&self, config: &ModelConfig, seq_len: usize) -> usize {
        self.hot_window_bytes(config) + self.boundary_bytes(config) + self.cold_id_bytes(seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_window_is_bounded() {
        let br = BoundaryResidual::gemma_4b();
        let config = ModelConfig::gemma_4b();

        let hot = br.hot_window_bytes(&config);
        // W=32, 34 layers, hidden=2560, f32 → 32 × 34 × 2560 × 4 = 11,206,656
        assert_eq!(hot, 32 * 34 * 2560 * 4);
    }

    #[test]
    fn boundary_is_fixed_per_model() {
        let br = BoundaryResidual::gemma_4b();
        let config = ModelConfig::gemma_4b();

        let b = br.boundary_bytes(&config);
        // 34 layers × 2560 × 4 = 348,160 bytes ≈ 340 KB
        assert_eq!(b, 34 * 2560 * 4);
        assert!(b < 400_000, "Boundary residual should be < 400 KB");
    }

    #[test]
    fn cold_id_bytes_is_4_per_cold_token() {
        let br = BoundaryResidual::new(32);
        assert_eq!(br.cold_id_bytes(1000), (1000 - 32) * 4);
        assert_eq!(br.cold_id_bytes(10), 0); // seq < window → no cold
    }

    #[test]
    fn total_stays_flat_at_scale() {
        let br = BoundaryResidual::gemma_4b();
        let config = ModelConfig::gemma_4b();

        let mem_4k   = br.memory_bytes(&config, 4_096);
        let mem_32k  = br.memory_bytes(&config, 32_768);
        let mem_131k = br.memory_bytes(&config, 131_072);
        let mem_370k = br.memory_bytes(&config, 370_000);

        // Cold IDs grow but are tiny (4 bytes/token). At 370K that's 1.48 MB
        // vs hot window of ~11.2 MB. Total stays in 11-13 MB range.
        assert!(mem_4k   < 15_000_000, "4K: {mem_4k}");
        assert!(mem_32k  < 15_000_000, "32K: {mem_32k}");
        assert!(mem_131k < 15_000_000, "131K: {mem_131k}");
        assert!(mem_370k < 15_000_000, "370K: {mem_370k}");

        // Growth from 4K to 370K is only cold IDs: (370K - 32 - (4K - 32)) × 4
        let growth = mem_370k - mem_4k;
        let expected_cold_growth = (370_000 - 4_096) * 4;
        assert_eq!(growth, expected_cold_growth);
    }

    #[test]
    fn much_smaller_than_standard_kv() {
        let br = BoundaryResidual::gemma_4b();
        let config = ModelConfig::gemma_4b();

        let br_mem    = br.memory_bytes(&config, 370_000);
        let kv_mem    = config.kv_memory(370_000);

        // Standard KV at 370K ≈ 25.8 GB; Boundary RS ≈ 13 MB → ~2000× compression.
        assert!(br_mem * 1000 < kv_mem,
            "Boundary RS ({br_mem}) should be >1000× smaller than standard KV ({kv_mem})");
    }

    #[test]
    fn encode_decode_roundtrip_shape() {
        let br = BoundaryResidual::new(4);
        let keys: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32; 16]).collect();
        let vals: Vec<Vec<f32>> = keys.clone();
        let encoded = br.encode(&keys, &vals);
        let (dk, dv) = br.decode(&encoded, 8, 16);
        assert_eq!(dk.len(), 8);
        assert_eq!(dv.len(), 8);
        assert_eq!(dk[0].len(), 16);
    }
}
