//! Minimal engine implementations shared by this module's tests.
//!
//! Both exist to exercise the traits' *default* method bodies: they implement
//! only the required methods, so every default a real engine would override
//! stays on its default path. Shared here rather than duplicated per file
//! because [`AnyEngine`](super::AnyEngine)'s tests need both families, and a
//! second copy would drift from the one under test.

use super::{EngineError, EngineInfo, KvEngine, RetrievalEngine};
use crate::ffn::FfnBackend;
use crate::ModelWeights;
use ndarray::Array2;

/// Synthetic engine that only implements the required trait methods,
/// leaving every default (`window_tokens`, `cold_bytes`, `stage_summary`,
/// `prefill_quant`, `decode_step_quant`) to fire. Exercises the default
/// bodies that no shipped engine routes through (every concrete engine
/// overrides them).
pub(crate) struct DefaultsOnlyEngine {
    pub(crate) prefill_calls: usize,
    pub(crate) decode_calls: usize,
}

impl KvEngine for DefaultsOnlyEngine {
    fn name(&self) -> &str {
        "defaults-only"
    }
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: self.name().into(),
            description: "test fixture".into(),
            backend: "cpu".into(),
            config: String::new(),
        }
    }
    fn prefill(
        &mut self,
        _weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        _token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_calls += 1;
        Ok(Array2::zeros((1, 4)))
    }
    fn decode_step(
        &mut self,
        _weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        _token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_calls += 1;
        Ok(Array2::zeros((1, 4)))
    }
    fn memory_bytes(&self) -> usize {
        0
    }
}

pub(crate) struct StubRetrievalEngine {
    pub(crate) prefill_calls: usize,
    pub(crate) decode_calls: usize,
    pub(crate) last_token: Option<u32>,
}

impl RetrievalEngine for StubRetrievalEngine {
    fn name(&self) -> &str {
        "stub-retrieval"
    }
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: self.name().into(),
            description: "test fixture".into(),
            backend: "cpu".into(),
            config: String::new(),
        }
    }
    fn prefill(
        &mut self,
        _weights: &ModelWeights,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_calls += 1;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        Ok(Array2::zeros((1, 4)))
    }
    fn decode_step(
        &mut self,
        _weights: &ModelWeights,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_calls += 1;
        self.last_token = Some(token_id);
        Ok(Array2::zeros((1, 4)))
    }
    fn memory_bytes(&self) -> usize {
        16
    }
}
