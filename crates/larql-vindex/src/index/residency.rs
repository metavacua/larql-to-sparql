//! Adaptive layer residency — pin hot layers, stream cold ones.
//!
//! llama.cpp loads ALL weights or nothing. Partial offload kills speed (PCIe cliff).
//! Vindex has a gradient: more memory → more pinned layers → smoothly faster.
//!
//! ```text
//! Pinned layer:  Q4 data pre-loaded in RAM → Q4 matvec (GPU or CPU), no page faults
//! Mmap layer:    Q4 data on disk, paged on demand → Q4 matvec, cold penalty ~0.05ms
//! f32 layer:     Full f32 gate vectors → BLAS brute-force, 10-100x slower
//! ```
//!
//! The `ResidencyManager` tracks which layers are pinned and how much memory
//! is used. `auto_pin()` fills the budget with the most-queried layers.

use std::collections::HashMap;

/// Per-layer residency state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerState {
    /// Not loaded — will use f32 mmap fallback.
    Cold,
    /// Q4 data available via mmap (paged on demand by OS).
    MmapQ4,
    /// Q4 data pinned in memory (pre-loaded, no page faults).
    Pinned,
}

/// Manages adaptive layer residency within a memory budget.
///
/// Tracks which layers are pinned, their memory cost, and provides
/// automatic pinning based on a budget and access patterns.
pub struct ResidencyManager {
    num_layers: usize,
    hidden_size: usize,
    /// Per-layer feature counts (needed for Q4 size calculation).
    layer_features: Vec<usize>,
    /// Current state of each layer.
    states: Vec<LayerState>,
    /// Pinned Q4 data per layer (owned bytes, pre-loaded from mmap).
    pinned_data: HashMap<usize, Vec<u8>>,
    /// Memory budget in bytes.
    budget_bytes: usize,
    /// Current pinned memory usage in bytes.
    pinned_bytes: usize,
    /// Access counts per layer (for auto_pin LRU/frequency).
    access_counts: Vec<u64>,
}

impl ResidencyManager {
    /// Create a new residency manager.
    ///
    /// `budget_mb`: memory budget in megabytes for pinned layers.
    /// `num_layers`, `hidden_size`: model dimensions.
    /// `layer_features`: per-layer feature count (from vindex config).
    pub fn new(
        budget_mb: usize,
        num_layers: usize,
        hidden_size: usize,
        layer_features: Vec<usize>,
    ) -> Self {
        assert_eq!(layer_features.len(), num_layers);
        Self {
            num_layers,
            hidden_size,
            layer_features,
            states: vec![LayerState::Cold; num_layers],
            pinned_data: HashMap::new(),
            budget_bytes: budget_mb * 1_048_576,
            pinned_bytes: 0,
            access_counts: vec![0; num_layers],
        }
    }

    /// Q4 byte size for a layer's gate vectors.
    pub fn layer_q4_bytes(&self, layer: usize) -> usize {
        let floats = self.layer_features[layer] * self.hidden_size;
        floats / 32 * 18 // Q4_0: 18 bytes per 32 elements
    }

    /// Current state of a layer.
    pub fn state(&self, layer: usize) -> LayerState {
        self.states[layer]
    }

    /// Total pinned memory in bytes.
    pub fn pinned_bytes(&self) -> usize {
        self.pinned_bytes
    }

    /// Total pinned memory in MB.
    pub fn pinned_mb(&self) -> f64 {
        self.pinned_bytes as f64 / 1_048_576.0
    }

    /// Memory budget in bytes.
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Number of pinned layers.
    pub fn num_pinned(&self) -> usize {
        self.states.iter().filter(|&&s| s == LayerState::Pinned).count()
    }

    /// Set all layers to MmapQ4 state (Q4 file is loaded).
    pub fn mark_q4_available(&mut self) {
        for s in &mut self.states {
            if *s == LayerState::Cold {
                *s = LayerState::MmapQ4;
            }
        }
    }

    /// Pin a layer: copy its Q4 data from mmap into owned memory.
    /// Returns false if the layer would exceed the budget.
    pub fn pin_layer(&mut self, layer: usize, q4_data: &[u8]) -> bool {
        if layer >= self.num_layers { return false; }
        if self.states[layer] == LayerState::Pinned { return true; } // already pinned

        let cost = q4_data.len();
        if self.pinned_bytes + cost > self.budget_bytes {
            return false; // over budget
        }

        self.pinned_data.insert(layer, q4_data.to_vec());
        self.states[layer] = LayerState::Pinned;
        self.pinned_bytes += cost;
        true
    }

    /// Evict a pinned layer back to mmap state.
    pub fn evict_layer(&mut self, layer: usize) {
        if layer >= self.num_layers { return; }
        if self.states[layer] != LayerState::Pinned { return; }

        if let Some(data) = self.pinned_data.remove(&layer) {
            self.pinned_bytes -= data.len();
        }
        self.states[layer] = LayerState::MmapQ4;
    }

    /// Get pinned Q4 data for a layer (returns None if not pinned).
    pub fn pinned_q4(&self, layer: usize) -> Option<&[u8]> {
        self.pinned_data.get(&layer).map(|v| v.as_slice())
    }

    /// Record an access to a layer (for frequency-based auto_pin).
    pub fn record_access(&mut self, layer: usize) {
        if layer < self.num_layers {
            self.access_counts[layer] += 1;
        }
    }

    /// Automatically pin layers to fill the memory budget.
    ///
    /// Strategy: pin layers by frequency (most-accessed first), then by
    /// layer index (knowledge band layers tend to be accessed most).
    /// Requires a closure that provides Q4 data for a given layer.
    pub fn auto_pin<F>(&mut self, get_q4: F) -> usize
    where
        F: Fn(usize) -> Option<Vec<u8>>,
    {
        // Rank layers by access count (descending), then by index
        let mut candidates: Vec<usize> = (0..self.num_layers)
            .filter(|&l| self.states[l] != LayerState::Pinned && self.layer_features[l] > 0)
            .collect();
        candidates.sort_by(|&a, &b| {
            self.access_counts[b].cmp(&self.access_counts[a])
                .then(a.cmp(&b))
        });

        let mut pinned = 0;
        for layer in candidates {
            let cost = self.layer_q4_bytes(layer);
            if self.pinned_bytes + cost > self.budget_bytes {
                continue; // skip layers that don't fit
            }
            if let Some(data) = get_q4(layer) {
                if self.pin_layer(layer, &data) {
                    pinned += 1;
                }
            }
        }
        pinned
    }

    /// Pin layers within a specific range (e.g., knowledge band L14-27).
    /// Returns how many were pinned.
    pub fn pin_range<F>(&mut self, start: usize, end: usize, get_q4: F) -> usize
    where
        F: Fn(usize) -> Option<Vec<u8>>,
    {
        let mut pinned = 0;
        for layer in start..end.min(self.num_layers) {
            if self.states[layer] == LayerState::Pinned { continue; }
            let cost = self.layer_q4_bytes(layer);
            if self.pinned_bytes + cost > self.budget_bytes { break; }
            if let Some(data) = get_q4(layer) {
                if self.pin_layer(layer, &data) {
                    pinned += 1;
                }
            }
        }
        pinned
    }

    /// Summary string for diagnostics.
    pub fn summary(&self) -> String {
        let pinned = self.num_pinned();
        let mmap = self.states.iter().filter(|&&s| s == LayerState::MmapQ4).count();
        let cold = self.states.iter().filter(|&&s| s == LayerState::Cold).count();
        format!(
            "{} pinned ({:.1} MB / {:.1} MB budget), {} mmap, {} cold",
            pinned,
            self.pinned_mb(),
            self.budget_bytes as f64 / 1_048_576.0,
            mmap,
            cold,
        )
    }
}
