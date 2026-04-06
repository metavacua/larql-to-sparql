//! LayerGraph — pluggable per-layer routing for attention and FFN.
//!
//! The transformer layer loop receives a residual, routes through attention
//! and FFN, and produces the next residual. The mechanism behind each step
//! can vary:
//!
//! - Dense matmul (today's baseline)
//! - Walk/vindex (sparse FFN from mmap)
//! - Template cache (precomputed routing for known templates)
//! - Residual-adaptive graph (cluster-based routing)
//!
//! The `LayerGraph` trait abstracts this: given a residual, produce the
//! layer output. The implementation decides how attention and FFN are computed.

mod dense;
mod walk;
mod cached;
mod template;
mod predict;

use ndarray::Array2;

use crate::attention::AttentionWeights;
use larql_compute::ComputeBackend;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;

// Re-export everything publicly
pub use dense::*;
pub use walk::*;
pub use cached::*;
pub use template::*;
pub use predict::*;

/// Output of a single layer's computation.
pub struct LayerOutput {
    /// Post-layer residual (input to next layer).
    pub residual: Array2<f32>,
    /// Optional: FFN activation capture (for tracing/analysis).
    pub activation: Option<Array2<f32>>,
    /// Optional: attention weight capture (for tracing/analysis).
    pub attention: Option<AttentionWeights>,
}

/// Per-layer routing trait. Takes a residual, produces the next residual.
///
/// Implementations control both attention and FFN computation.
/// The residual is always the input. The mechanism changes.
pub trait LayerGraph {
    /// Run one transformer layer: attention + FFN + residuals.
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}
