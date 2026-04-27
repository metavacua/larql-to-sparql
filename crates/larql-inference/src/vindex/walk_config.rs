//! WalkFfnConfig — per-layer K schedule for the unified walk kernel.
//!
//! `None` selects the dense-equivalent mmap path for that layer
//! (interleaved / q4 / full_mmap — chosen internally based on what
//! the vindex exposes). `Some(k)` selects the sparse walk path
//! (gate KNN → top-K up dot products → GEGLU → K down accumulations).

#[derive(Debug, Clone)]
pub struct WalkFfnConfig {
    /// Per-layer K. None = dense walk (all features). Some(k) = top-K sparse.
    pub k_per_layer: Vec<Option<usize>>,
    /// Skip features whose |activation| falls below this threshold.
    /// 0.0 preserves dense equivalence.
    pub activation_floor: f32,
}

impl WalkFfnConfig {
    /// Dense walk for every layer. Produces the same math as the classic
    /// `gate @ up @ down` matmul pipeline, routed through mmap'd vectors.
    pub fn dense(num_layers: usize) -> Self {
        Self { k_per_layer: vec![None; num_layers], activation_floor: 0.0 }
    }

    /// Uniform sparse walk at K per layer.
    pub fn sparse(num_layers: usize, k: usize) -> Self {
        Self { k_per_layer: vec![Some(k); num_layers], activation_floor: 0.0 }
    }

    /// Dense for `0..sparse_from`, sparse-K from `sparse_from..num_layers`.
    /// Matches the "dense early, sparse late" split used in hybrid configs.
    pub fn hybrid(num_layers: usize, sparse_from: usize, k: usize) -> Self {
        let mut k_per_layer = vec![None; num_layers];
        for slot in &mut k_per_layer[sparse_from.min(num_layers)..] {
            *slot = Some(k);
        }
        Self { k_per_layer, activation_floor: 0.0 }
    }

    /// Set the activation magnitude floor. Default 0.0 (no skip).
    pub fn with_floor(mut self, floor: f32) -> Self {
        self.activation_floor = floor;
        self
    }

    /// K for a layer. Out-of-range layers fall through to the last entry
    /// (or None if the config is empty) — mirrors `LayerFfnRouter::get`.
    pub fn k_for(&self, layer: usize) -> Option<usize> {
        if self.k_per_layer.is_empty() {
            return None;
        }
        let idx = layer.min(self.k_per_layer.len() - 1);
        self.k_per_layer[idx]
    }

    /// True when this layer should take the sparse walk path.
    pub fn is_sparse(&self, layer: usize) -> bool {
        self.k_for(layer).is_some()
    }

    pub fn num_layers(&self) -> usize {
        self.k_per_layer.len()
    }
}

impl Default for WalkFfnConfig {
    /// Empty config — all layers resolve to dense (None). Callers
    /// should prefer the named constructors when num_layers is known.
    fn default() -> Self {
        Self { k_per_layer: Vec::new(), activation_floor: 0.0 }
    }
}
