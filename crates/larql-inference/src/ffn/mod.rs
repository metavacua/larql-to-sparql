//! Feed-forward network computation — trait-based with pluggable backends.
//!
//! Production: [`WalkFfn`](crate::vindex::WalkFfn) — the unified walk kernel.
//! Full-K and sparse-K are the same code path parameterised by [`WalkFfnConfig`](crate::vindex::WalkFfnConfig).
//!
//! Reference: [`WeightFfn`] + [`SparseFfn`] live here for correctness/bench
//! comparison (see `examples/walk_correctness.rs`); they are not used in
//! production dispatch.

pub mod weight;
pub mod sparse;
pub mod sparse_compute;
pub mod remote;
#[cfg(test)]
mod tests;

use ndarray::Array2;

// ── Trait ──

/// FFN backend trait. Defines how a single layer's FFN is computed.
pub trait FfnBackend {
    /// Run the FFN for a given layer on the pre-FFN-normed residual.
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32>;

    /// Run FFN and also return the pre-down activation (for capture).
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>);

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

// ── Re-exports ──

pub use weight::WeightFfn;
pub use sparse::SparseFfn;
pub use remote::{RemoteFfnConfig, RemoteFfnError, RemoteWalkBackend, RemoteLatencyStats};
pub use sparse_compute::{
    sparse_ffn_forward, sparse_ffn_forward_with_overrides,
    sparse_ffn_forward_with_full_overrides, FeatureSlotOverride,
};

// ── Per-layer backend selection ──

/// Selects which FFN backend to use for each layer.
pub struct LayerFfnRouter<'a> {
    backends: Vec<&'a dyn FfnBackend>,
    num_layers: usize,
}

impl<'a> LayerFfnRouter<'a> {
    pub fn uniform(backend: &'a dyn FfnBackend, num_layers: usize) -> Self {
        Self { backends: vec![backend; num_layers], num_layers }
    }

    pub fn per_layer(backends: Vec<&'a dyn FfnBackend>) -> Self {
        let num_layers = backends.len();
        Self { backends, num_layers }
    }

    pub fn get(&self, layer: usize) -> &dyn FfnBackend {
        if layer < self.num_layers { self.backends[layer] }
        else { self.backends[self.num_layers - 1] }
    }
}

// ── Activation functions ──

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu_gate_up(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let activated = gate.mapv(|v| v * sigmoid(v));
    &activated * up
}

pub fn gelu_tanh_gate_up(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let activated = gate.mapv(gelu_tanh);
    &activated * up
}

pub fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

// Architecture-correct FFN computation is in weight::dense_ffn_forward
// and sparse_compute::sparse_ffn_forward. No legacy SiLU-only functions.
