use ndarray::Array2;

use larql_compute::ComputeBackend;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use super::{LayerGraph, LayerOutput};

/// Dense baseline: standard matmul attention + pluggable FFN backend.
/// This is today's working path — nothing changes, just wrapped in the trait.
pub struct DenseLayerGraph<'a> {
    pub ffn: &'a dyn FfnBackend,
    pub backend: Option<&'a dyn ComputeBackend>,
    pub capture_activation: bool,
    pub capture_attention: bool,
}

impl<'a> LayerGraph for DenseLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        // Attention: dense matmul (Q·K·V), optionally GPU-accelerated
        let (h_post_attn, _attn_proj, attn_weights) =
            crate::attention::run_attention_block_gpu(
                weights, h, layer, self.capture_attention, self.backend,
            )?;

        // FFN: delegated to backend (dense, walk, sparse, etc.)
        let (h_out, activation) = crate::forward::run_ffn(
            weights, &h_post_attn, layer, self.ffn, self.capture_activation,
        );

        Some(LayerOutput {
            residual: h_out,
            activation,
            attention: attn_weights,
        })
    }

    fn name(&self) -> &str {
        "dense"
    }
}

/// Per-layer graph selection: different layers can use different backends.
pub struct PerLayerGraph<'a> {
    layers: Vec<&'a dyn LayerGraph>,
}

impl<'a> PerLayerGraph<'a> {
    pub fn new(layers: Vec<&'a dyn LayerGraph>) -> Self {
        Self { layers }
    }

    pub fn get(&self, layer: usize) -> &'a dyn LayerGraph {
        if layer < self.layers.len() {
            self.layers[layer]
        } else {
            *self.layers.last().unwrap()
        }
    }
}

impl<'a> LayerGraph for PerLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        self.get(layer).forward_layer(weights, h, layer)
    }

    fn name(&self) -> &str {
        "per-layer"
    }
}
