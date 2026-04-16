/// Residual checkpointing for fast reconstruction.
///
/// Stores residual snapshots at key layers (e.g., L0, L8, L16, L24, L33).
/// When reconstructing from cold tier, replay from nearest checkpoint
/// instead of full forward pass — max 8 layers of recompute.

/// Checkpoint configuration.
pub struct CheckpointConfig {
    /// Layer indices where checkpoints are stored.
    pub layers: Vec<usize>,
    /// Total layers in the model.
    pub total_layers: usize,
}

impl CheckpointConfig {
    pub fn new(layers: Vec<usize>, total_layers: usize) -> Self {
        Self {
            layers,
            total_layers,
        }
    }

    /// Default for Gemma 3-4B: checkpoints at L0, L8, L16, L24, L33.
    pub fn gemma_4b() -> Self {
        Self::new(vec![0, 8, 16, 24, 33], 34)
    }

    /// Maximum layers to recompute from nearest checkpoint.
    pub fn max_recompute(&self) -> usize {
        if self.layers.is_empty() {
            return self.total_layers;
        }

        let mut max_gap = 0;
        for w in self.layers.windows(2) {
            max_gap = max_gap.max(w[1] - w[0]);
        }
        // Also check from last checkpoint to end
        if let Some(&last) = self.layers.last() {
            max_gap = max_gap.max(self.total_layers - last);
        }
        max_gap
    }

    /// Find the nearest checkpoint at or before the given layer.
    pub fn nearest_checkpoint(&self, layer: usize) -> Option<usize> {
        self.layers.iter().rev().find(|&&l| l <= layer).copied()
    }

    /// Memory per token for checkpoints (bytes).
    pub fn memory_per_token(&self, hidden_dim: usize) -> usize {
        self.layers.len() * hidden_dim * 2 // fp16 residuals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_recompute() {
        let config = CheckpointConfig::gemma_4b();
        // Gaps: 8, 8, 8, 9, plus 34-33=1
        assert_eq!(config.max_recompute(), 9);
    }

    #[test]
    fn test_nearest_checkpoint() {
        let config = CheckpointConfig::gemma_4b();
        assert_eq!(config.nearest_checkpoint(0), Some(0));
        assert_eq!(config.nearest_checkpoint(5), Some(0));
        assert_eq!(config.nearest_checkpoint(8), Some(8));
        assert_eq!(config.nearest_checkpoint(30), Some(24));
        assert_eq!(config.nearest_checkpoint(33), Some(33));
    }
}
