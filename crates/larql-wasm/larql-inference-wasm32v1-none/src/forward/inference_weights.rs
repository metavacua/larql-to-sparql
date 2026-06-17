//! Format-agnostic inference weight handle.
//!
//! `InferenceWeights` is the single loading point for any code that needs to
//! run `infer_patched` against a vindex. It detects the quantisation format
//! from `VindexConfig`, loads the right on-disk artefacts, and dispatches to
//! `infer_patched` or `infer_patched_q4k` without the caller branching on
//! `config.quant`.
//!
//! **Scope:** the INFER / INSERT KNN / EXPLAIN INFER pipeline. Specialised
//! callers (bench, generation, Metal) keep their own explicit paths.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(not(target_arch = "wasm32"))]
use larql_vindex::{
    GateIndex, IndexLoadCallbacks, KnnStore, QuantFormat, VectorIndex, VindexConfig, VindexError,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::model::ModelWeights;

#[cfg(not(target_arch = "wasm32"))]
use super::infer_patched::{infer_patched, infer_patched_q4k, InferPatchedResult};
#[cfg(not(target_arch = "wasm32"))]
use super::predict::predict;
use super::PredictResult;
#[cfg(not(target_arch = "wasm32"))]
/// An inference-ready weight handle that is agnostic to quantisation format.
///
/// Constructed via [`InferenceWeights::load`]. Callers use
/// [`InferenceWeights::infer_patched`] and [`InferenceWeights::as_weights`]
/// without branching on the underlying format.
#[allow(clippy::large_enum_variant)]
pub enum InferenceWeights {
    Dense(ModelWeights),
    Quantised {
        weights: ModelWeights,
        index: VectorIndex,
    },
}

#[cfg(not(target_arch = "wasm32"))]
impl InferenceWeights {
    #[cfg(not(target_arch = "wasm32"))]
    /// Load weights for the vindex at `path`, choosing the right artefacts
    /// based on `config.quant`. Returns `VindexError` on any I/O or parse
    /// failure so callers can map it to their own error type.
    pub fn load(
        path: &Path,
        config: &VindexConfig,
        cb: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, VindexError> {
        if config.quant != QuantFormat::None {
            let mut idx = VectorIndex::load_vindex(path, cb)?;
            idx.load_attn_q4k(path)?;
            idx.load_interleaved_q4k(path)?;
            let weights = larql_vindex::load_model_weights_q4k(path, cb)?;
            Ok(Self::Quantised {
                weights,
                index: idx,
            })
        } else {
            let weights = larql_vindex::load_model_weights(path, cb)?;
            Ok(Self::Dense(weights))
        }
    }

    /// `true` if backed by a quantised (q4k or later) format.
    pub fn is_quantised(&self) -> bool {
        matches!(self, Self::Quantised { .. })
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Borrow the underlying `ModelWeights` (arch + embeddings + norms).
    ///
    /// Always valid — both variants carry a `ModelWeights`. For the
    /// `Quantised` variant the attention/FFN tensor slots are empty; callers
    /// that need full attention tensors in memory must not use the dense path.
    pub fn as_weights(&self) -> &ModelWeights {
        match self {
            Self::Dense(w) => w,
            Self::Quantised { weights, .. } => weights,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Mutably borrow the underlying `ModelWeights`.
    pub fn as_weights_mut(&mut self) -> &mut ModelWeights {
        match self {
            Self::Dense(w) => w,
            Self::Quantised { weights, .. } => weights,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Run the shared INFER pipeline, dispatching to the correct forward path.
    ///
    /// Identical contract to [`infer_patched`] / [`infer_patched_q4k`]:
    /// unlimited walk FFN features, `KNN_COSINE_THRESHOLD = 0.75`, first
    /// stored layer wins. Callers do not branch on format.
    pub fn infer_patched(
        &mut self,
        tokenizer: &Tokenizer,
        gate_index: &dyn GateIndex,
        knn_store: Option<&KnnStore>,
        token_ids: &[u32],
        top_k: usize,
    ) -> InferPatchedResult {
        match self {
            Self::Dense(weights) => {
                infer_patched(weights, tokenizer, gate_index, knn_store, token_ids, top_k)
            }
            Self::Quantised { weights, index } => infer_patched_q4k(
                weights, tokenizer, gate_index, knn_store, token_ids, top_k, index,
            ),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Dense forward pass (no walk FFN, no KNN). Used for the
    /// `INFER COMPARE` dense side-by-side column.
    pub fn predict_dense(
        &mut self,
        tokenizer: &Tokenizer,
        token_ids: &[u32],
        top_k: usize,
    ) -> PredictResult {
        match self {
            Self::Dense(weights) => predict(weights, tokenizer, token_ids, top_k),
            Self::Quantised { weights, index } => {
                crate::vindex::predict_q4k(weights, tokenizer, token_ids, top_k, index)
            }
        }
    }
}
