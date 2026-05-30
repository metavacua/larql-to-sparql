//! Prediction-related types used across the forward pass.

use crate::attention::AttentionWeights;
use crate::ffn::FfnBackend;
#[allow(unused_imports)]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format, borrow::{Cow, ToOwned}, rc::Rc, sync::Arc, collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
/// Per-head attention pattern for the last token at one layer.
pub struct LayerAttentionCapture {
    pub layer: usize,
    pub weights: AttentionWeights,
}

/// Result of a forward trace — residuals and optional sparse activations.
pub struct TraceResult {
    pub residuals: Vec<(usize, Vec<f32>)>,
    pub activations: Vec<(usize, Vec<(usize, f32)>)>,
    pub attention: Vec<LayerAttentionCapture>,
}

/// Prediction result from a full forward pass.
pub struct PredictResult {
    pub predictions: Vec<(String, f64)>,
    /// Top-k token IDs parallel to `predictions`. `token_ids[i]`
    /// produced `predictions[i].0` when decoded. Used by autoregressive
    /// generators to append the argmax token without re-tokenizing the
    /// decoded string (which would drift on subword boundaries).
    pub token_ids: Vec<u32>,
}

/// Prediction result with per-layer residual capture.
pub struct PredictResultWithResiduals {
    pub predictions: Vec<(String, f64)>,
    pub residuals: Vec<Vec<f32>>,
}

/// Prediction result with per-layer attention captures and logit lens.
pub struct PredictResultWithAttention {
    pub predictions: Vec<(String, f64)>,
    pub attention: Vec<LayerAttentionCapture>,
    pub residuals: Vec<(usize, Vec<f32>)>,
}

/// Per-layer computation strategy.
pub enum LayerMode<'a> {
    Compute(&'a dyn FfnBackend),
    ScalarGain(f32),
    AttentionOnly,
}
