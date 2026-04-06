use ndarray::Array2;

use larql_compute::ComputeBackend;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use super::{LayerGraph, LayerOutput};

// ── Walk: dense attention + vindex walk FFN ──

/// Walk layer graph: dense attention + vindex walk FFN.
/// This is the working walk path, wrapped in the LayerGraph trait.
pub struct WalkLayerGraph<'a> {
    pub ffn: &'a dyn FfnBackend,
    pub backend: Option<&'a dyn ComputeBackend>,
}

impl<'a> LayerGraph for WalkLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block_gpu(weights, h, layer, false, self.backend)?;
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, self.ffn, false);
        Some(LayerOutput { residual: h_out, activation: None, attention: None })
    }

    fn name(&self) -> &str { "walk" }
}

// ── Pipelined: CPU attention + batched GPU Q4 FFN ──

/// Pipelined layer graph: runs attention on CPU, batches all FFN layers
/// in one Metal Q4 command buffer via `multi_layer_q4_ffn`.
///
/// The layer loop splits each layer into:
/// 1. Attention (CPU BLAS) → post-attention residual
/// 2. FFN norm → normalized input for FFN
/// 3. Q4 FFN (batched on GPU after all layers)
///
/// For layers where attention and FFN are interleaved (each FFN feeds next attention),
/// this runs a hybrid: per-layer attention + per-layer Q4 FFN via the compute backend,
/// but with the Q4 overhead amortized by reusing the GPU command queue.
pub struct PipelinedLayerGraph<'a> {
    pub index: &'a dyn larql_vindex::GateIndex,
    pub backend: &'a dyn ComputeBackend,
    pub layer_range: std::ops::Range<usize>,
}

impl<'a> LayerGraph for PipelinedLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        if !self.layer_range.contains(&layer) {
            return None;
        }

        // Attention: CPU BLAS (fast, no GPU overhead)
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block_gpu(weights, h, layer, false, None)?;

        // FFN: use WalkFfn which handles Q4 dispatch internally.
        // WalkFfn checks for Q4 interleaved data and routes to Metal Q4
        // when backend.has_q4(), falling back to f32 BLAS otherwise.
        // This ensures the norm/residual logic matches exactly.
        let walk_ffn = crate::vindex::WalkFfn::new_with_backend(
            weights, self.index, 8192, self.backend,
        );
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        Some(LayerOutput { residual: h_out, activation: None, attention: None })
    }

    fn name(&self) -> &str { "pipelined" }
}
