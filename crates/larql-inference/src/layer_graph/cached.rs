use ndarray::Array2;

use larql_compute::ComputeBackend;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use super::{LayerGraph, LayerOutput, DenseLayerGraph, PerLayerGraph};

// ── Cached: precomputed layer output for fixed-routing regimes ──

/// Cached layer graph: returns a precomputed residual instead of computing.
/// For layers where the output is template-determined (L0-12 regime).
///
/// Build by running a dense forward pass for a template, capturing residuals,
/// then storing them. At inference, skip the computation entirely.
pub struct CachedLayerGraph {
    /// layer → cached residual [seq_len, hidden]. Keyed by layer index.
    cache: std::collections::HashMap<usize, Array2<f32>>,
}

impl CachedLayerGraph {
    /// Build a cache by running a dense forward pass and capturing residuals.
    /// `layers`: which layers to cache (e.g., 0..=12).
    pub fn build(
        weights: &ModelWeights,
        token_ids: &[u32],
        layers: &[usize],
        ffn: &dyn FfnBackend,
    ) -> Self {
        let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
        let mut cache = std::collections::HashMap::new();
        let max_layer = *layers.iter().max().unwrap_or(&0);

        for layer in 0..=max_layer.min(weights.num_layers - 1) {
            let graph = DenseLayerGraph { ffn, backend: None, capture_activation: false, capture_attention: false };
            if let Some(output) = graph.forward_layer(weights, &h, layer) {
                h = output.residual;
                if layers.contains(&layer) {
                    cache.insert(layer, h.clone());
                }
            }
        }
        Self { cache }
    }

    /// Build from an existing residual (e.g., from a previous forward pass).
    pub fn from_residuals(residuals: Vec<(usize, Array2<f32>)>) -> Self {
        Self { cache: residuals.into_iter().collect() }
    }

    pub fn has_layer(&self, layer: usize) -> bool {
        self.cache.contains_key(&layer)
    }

    pub fn num_cached(&self) -> usize {
        self.cache.len()
    }
}

impl LayerGraph for CachedLayerGraph {
    fn forward_layer(
        &self,
        _weights: &ModelWeights,
        _h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        let residual = self.cache.get(&layer)?.clone();
        Some(LayerOutput { residual, activation: None, attention: None })
    }

    fn name(&self) -> &str { "cached" }
}

/// Build a PerLayerGraph with cached layers for a detected template.
/// Returns the graph and the number of cached layers.
///
/// Layout:
///   cached_layers → CachedLayerGraph (skip computation)
///   remaining layers → fallback (dense/walk)
pub fn build_adaptive_graph<'a>(
    cache: &'a CachedLayerGraph,
    fallback: &'a dyn LayerGraph,
    num_layers: usize,
    cached_range: &std::ops::RangeInclusive<usize>,
) -> PerLayerGraph<'a> {
    let mut layers: Vec<&dyn LayerGraph> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        if cached_range.contains(&layer) && cache.has_layer(layer) {
            layers.push(cache);
        } else {
            layers.push(fallback);
        }
    }
    PerLayerGraph::new(layers)
}

/// Cached post-attention residuals and FFN-normed inputs for the split pass.
///
/// Built from one exact (interleaved) forward pass. Reused for all entities
/// that match the same template — attention is template-fixed (~99% identical).
pub struct AttentionCache {
    /// Per-layer FFN-normed last-token vector (the actual FFN input).
    pub ffn_inputs: Vec<Vec<f32>>,
    /// The final post-attention residual (for combining with FFN output).
    pub final_residual: Array2<f32>,
}

impl AttentionCache {
    /// Build by running one exact forward pass (interleaved attention + FFN)
    /// and capturing the FFN inputs at each walk layer.
    pub fn build(
        weights: &ModelWeights,
        token_ids: &[u32],
        cached_layers: &CachedLayerGraph,
        ffn: &dyn FfnBackend,
        layer_range: std::ops::Range<usize>,
    ) -> Self {
        let seq_len = token_ids.len();
        let arch = &*weights.arch;
        let norm_offset = arch.norm_weight_offset();

        // Run through cached layers first
        let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
        for layer in 0..layer_range.start {
            if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
                h = output.residual;
            }
        }

        // Run exact interleaved pass for walk layers, capturing FFN inputs
        let mut ffn_inputs = Vec::with_capacity(layer_range.len());
        for layer in layer_range {
            // Attention (exact)
            let (h_post_attn, _, _) =
                crate::attention::run_attention_block_gpu(weights, &h, layer, false, None)
                    .unwrap();

            // Capture FFN-normed input (last token)
            let pre_ffn_key = if arch.has_post_norms() {
                arch.pre_feedforward_layernorm_key(layer)
            } else {
                Some(arch.post_attention_layernorm_key(layer))
            };
            let h_ffn = match pre_ffn_key {
                Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
                None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
            };
            ffn_inputs.push(h_ffn.row(seq_len - 1).to_vec());

            // FFN (exact — for correct residual stream)
            let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, ffn, false);
            h = h_out;
        }

        AttentionCache { ffn_inputs, final_residual: h }
    }
}
