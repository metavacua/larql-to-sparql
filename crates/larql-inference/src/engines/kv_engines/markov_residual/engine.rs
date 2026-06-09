//! MarkovResidualEngine — KvEngine implementation.

use larql_compute::{cpu_backend, ComputeBackend};
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::compute::{rs_decode_step, rs_decode_step_profiled, rs_prefill};
use super::q4k::{ensure_attn_tensors_dequantised, rs_decode_step_walk, rs_prefill_walk};
use super::store::RsStore;
use crate::engines::profiler::{DecodeStageSummary, EngineProfiler};
use crate::engines::{EngineInfo, KvEngine};
use crate::model::ModelWeights;

pub struct MarkovResidualEngine {
    window_size: Option<usize>,
    store: Option<RsStore>,
    backend: Box<dyn ComputeBackend>,
    profiling: bool,
    profile: EngineProfiler,
    metal_prefill_done: bool,
}

impl MarkovResidualEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn ComputeBackend>) -> Self {
        Self {
            window_size,
            store: None,
            backend,
            profiling: false,
            profile: EngineProfiler::default(),
            metal_prefill_done: false,
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
    }
}

impl KvEngine for MarkovResidualEngine {
    fn name(&self) -> &str {
        "markov-rs"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w}"),
            None => "window=full".into(),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "markov-rs".into(),
            description: format!(
                "residual-stream KV replacement — K/V recomputed from stored residuals (mem={:.1}MB)",
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config,
        }
    }

    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        let result = rs_prefill(weights, token_ids, self.window_size, self.backend.as_ref());
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
        let rs = self.store.take()?;
        let (hidden, new_rs) = if self.profiling {
            rs_decode_step_profiled(
                weights,
                token_id,
                rs,
                self.backend.as_ref(),
                &mut self.profile,
            )?
        } else {
            rs_decode_step(weights, token_id, rs, self.backend.as_ref())?
        };
        self.store = Some(new_rs);
        Some(hidden)
    }

    fn memory_bytes(&self) -> usize {
        self.total_memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(self.profile.summary("markov-rs", self.backend.name()))
    }

    fn prefill_q4k(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        use crate::engines::unlimited_context::engine::q4k_prefill_metal;
        if let Some(h) = q4k_prefill_metal(weights, index, token_ids, backend) {
            self.metal_prefill_done = true;
            self.store = None;
            return Some(h);
        }
        self.metal_prefill_done = false;
        ensure_attn_tensors_dequantised(weights, index);
        let result = rs_prefill_walk(weights, index, token_ids, self.window_size, backend);
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step_q4k(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        use crate::engines::unlimited_context::engine::q4k_decode_token;
        if self.metal_prefill_done {
            if let Some(h) = q4k_decode_token(weights, index, token_id, backend) {
                return Some(h);
            }
        }
        ensure_attn_tensors_dequantised(weights, index);
        let rs = self.store.take()?;
        let (hidden, new_rs) = rs_decode_step_walk(weights, index, token_id, rs, backend)?;
        self.store = Some(new_rs);
        Some(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::make_test_weights;
    use crate::engines::KvEngine;
    use crate::forward::hidden_to_raw_logits;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn engine_name() {
        assert_eq!(MarkovResidualEngine::new(None).name(), "markov-rs");
    }

    #[test]
    fn engine_memory_zero_before_prefill() {
        let eng = MarkovResidualEngine::new(None);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn engine_info_full_window() {
        let eng = MarkovResidualEngine::new(None);
        let info = eng.info();
        assert!(
            info.config.contains("full"),
            "expected 'full' in config, got '{}'",
            info.config
        );
    }

    #[test]
    fn engine_info_fixed_window() {
        let eng = MarkovResidualEngine::new(Some(16));
        let info = eng.info();
        assert!(
            info.config.contains("16"),
            "expected window size in config, got '{}'",
            info.config
        );
    }

    // ── Prefill → decode cycle ────────────────────────────────────────────────

    #[test]
    fn prefill_stores_residuals_for_all_layers() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine.prefill(&weights, &[0u32, 1, 2]).expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > 0,
            "store should be non-empty after prefill"
        );
    }

    #[test]
    fn decode_step_produces_finite_logits() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(hidden_to_raw_logits(&weights, &h)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn memory_grows_with_each_decode_step() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &[0u32]).expect("prefill");
        let mem_after_prefill = engine.memory_bytes();
        engine.decode_step(&weights, 1).expect("decode 1");
        let mem_after_1 = engine.memory_bytes();
        engine.decode_step(&weights, 2).expect("decode 2");
        let mem_after_2 = engine.memory_bytes();
        assert!(
            mem_after_1 > mem_after_prefill,
            "memory should grow with decode steps"
        );
        assert!(mem_after_2 > mem_after_1);
    }

    #[test]
    fn window_clipping_limits_hot_store() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(Some(2)); // window=2 tokens
        engine
            .prefill(&weights, &[0u32, 1, 2, 3, 4])
            .expect("prefill 5 tokens");
        // After clipping, hot store ≤ window
        assert!(
            engine.window_tokens() <= 2,
            "window_tokens={} should be ≤ 2",
            engine.window_tokens()
        );
        // Cold bytes should now be non-zero (overflow clipped to cold)
        assert!(
            engine.cold_bytes() > 0,
            "cold tier should have bytes after clipping"
        );
    }

    #[test]
    fn multiple_decode_steps_produce_consistent_shapes() {
        let weights = make_test_weights();
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &[0u32]).expect("prefill");
        for step in 0..3 {
            let h = engine.decode_step(&weights, step as u32).expect("decode");
            assert_eq!(h.shape(), &[1, weights.hidden_size], "step {step}");
        }
    }
}
