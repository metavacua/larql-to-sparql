//! MarkovResidualEngine — KvEngine implementation.

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::prefill::rs_prefill;
use super::step::{rs_decode_step, rs_decode_step_profiled};
use super::store::RsStore;
use super::walk::{ensure_attn_tensors_dequantised, rs_decode_step_walk, rs_prefill_walk};
use crate::profiler::EngineProfiler;
use crate::{DecodeStageSummary, EngineInfo, KvEngine};
use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};

pub struct MarkovResidualEngine {
    pub(super) window_size: Option<usize>,
    pub(super) store: Option<RsStore>,
    pub(super) backend: Box<dyn EngineBackend>,
    pub(super) profiling: bool,
    pub(super) profile: EngineProfiler,
    /// W1-GPU: handle into the backend's internal K/V cache, populated
    /// when `prefill_quant` routes through `coarse_prefill_with_state`.
    /// `None` means the engine took the legacy per-layer walk path.
    pub(super) kv_handle: Option<larql_inference::KvHandle>,
    /// Position counter used by `coarse_decode_step_with_state` for RoPE.
    /// Tracks `prompt_len + steps_already_decoded`.
    pub(super) abs_position: usize,
    /// Engine-owned f32 dequant scratch for the per-layer walk fallback —
    /// `prefill_quant`/`decode_step_quant` populate it; the walk resolves
    /// attention through a `WeightsView::with_scratch` over it. Empty on the
    /// W1-GPU/coarse path. Keeps `weights` immutable (no `weights.tensors`
    /// mutation).
    pub(super) dequant_scratch: larql_inference::DequantScratch,
}

impl MarkovResidualEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            window_size,
            store: None,
            backend,
            profiling: false,
            profile: EngineProfiler::default(),
            kv_handle: None,
            abs_position: 0,
            dequant_scratch: larql_inference::DequantScratch::new(),
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    /// Residual store + any K/V held on this engine's behalf elsewhere.
    ///
    /// On the W1-GPU path the store stays empty and the K/V lives outside
    /// the engine, so the store alone reports 0 — which reads as "costs
    /// nothing" rather than "measured somewhere else". It can be in
    /// either of two places depending on backend, and they are mutually
    /// exclusive: inside the `kv_handle` (CPU's whole-model Q4K cache) or
    /// inside the backend itself (Metal, whose handle is a sentinel and
    /// whose `backend_resident_kv_bytes` is the only way to see it).
    pub fn total_memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
            + self.kv_handle.as_ref().map_or(0, |h| h.resident_bytes())
            + self.backend.backend_resident_kv_bytes()
    }
}

// W1-GPU dispatch methods (`try_prefill_via_dispatch` /
// `decode_step_via_dispatch`) live in [`super::dispatch`] as an
// additional `impl MarkovResidualEngine` block. They mutate the
// `pub(super)` fields above.

/// §4 architecture-precondition gate
/// (`crates/larql-inference/docs/specs/markov-residual-engine.md`): the
/// exactness contract holds only when per-layer K/V is a pure function
/// of `(residual, weights, absolute position)` — stateless norms,
/// RoPE-only position encoding, direct K/V projection. Constructors
/// don't take weights, so the check runs at the earliest point the
/// engine sees the model (every prefill entry) and fails closed with a
/// typed error. Queries the structural
/// `ModelArchitecture::kv_recomputable_from_residuals` predicate — no
/// architecture-name matching. Shared with the codec twin.
pub(crate) fn check_residual_recompute_preconditions(
    engine_name: &str,
    weights: &ModelWeights,
) -> Result<(), EngineError> {
    if weights.arch.kv_recomputable_from_residuals() {
        Ok(())
    } else {
        Err(EngineError::InvariantViolation {
            what: format!(
                "{engine_name}: architecture '{}' violates the residual-recompute \
                 preconditions (markov-residual-engine.md §4); refusing to run non-exact",
                weights.arch.family()
            ),
        })
    }
}

impl MarkovResidualEngine {
    /// Shared body for `decode_step` / `decode_step_resident` — `index`
    /// reaches the attention step's Q4K-direct route when present.
    /// The store is borrowed, never taken. A failed step therefore costs the
    /// engine nothing but its droppable `hot_kv` derivative — see
    /// [`super::step`]'s failure invariant — so a refusal is reported as
    /// itself (`EngineError::Execution`, retryable) rather than as a dead
    /// engine wearing the words "called before prefill".
    fn decode_step_impl(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let Self {
            store,
            backend,
            profile,
            profiling,
            ..
        } = self;
        let rs = store
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (store missing)".into(),
            })?;
        if *profiling {
            rs_decode_step_profiled(
                larql_inference::WeightsView::dense(weights),
                token_id,
                rs,
                backend.as_ref(),
                profile,
                Some(ffn),
                index,
            )
        } else {
            rs_decode_step(
                larql_inference::WeightsView::dense(weights),
                token_id,
                rs,
                backend.as_ref(),
                Some(ffn),
                index,
            )
        }
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

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // `?` before the assignment: a refused prefill must not replace the
        // store an earlier one built. See `prefill`'s transactional contract.
        let result = rs_prefill(
            larql_inference::WeightsView::dense(weights),
            token_ids,
            self.window_size,
            self.backend.as_ref(),
            Some(ffn),
        )?;
        self.store = Some(result.store);
        Ok(result.hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_impl(weights, ffn, token_id, None)
    }

    /// Resident-path decode: threads `index` down to the walk's attention
    /// step so the Q4K-direct projections (`LARQL_Q4K_DIRECT_ATTN` family)
    /// can fire — the structural gap that left non-standard engines on the
    /// f32 path after the standard engine got the fast attention.
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_impl(weights, ffn, token_id, Some(index))
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

    fn dispatch_path(&self) -> Option<larql_inference::kv_engine::DispatchPath> {
        use larql_inference::kv_engine::DispatchPath;
        // `kv_handle` is stashed only by the W1-GPU coarse prefill, and
        // cleared when that path is abandoned; `store` exists after any
        // successful prefill. Neither set = no prefill yet.
        match (self.kv_handle.is_some(), self.store.is_some()) {
            (true, _) => Some(DispatchPath::Coarse),
            (false, true) => Some(DispatchPath::PerLayer),
            (false, false) => None,
        }
    }

    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(self.profile.summary("markov-rs", self.backend.name()))
    }

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // W1-GPU path: route through KvDispatch's coarse_prefill_with_state
        // when the engine's stored EngineBackend supports it. State capture
        // gives us per-layer h_in (= the residual we'd store) and per-layer
        // K/V (= the hot K/V tier from W2) in a single backend call —
        // backend can run on GPU; engine's state policy reads the dump.
        // Legacy per-layer walk remains as the fallback so unmigrated
        // backends keep working.
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Ok(hidden);
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let result = rs_prefill_walk(view, index, token_ids, self.window_size, backend);
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        self.kv_handle = None; // ensure dispatch path is not used for subsequent decode
        self.abs_position = token_ids.len();
        Ok(hidden)
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        // W1-GPU path: if prefill went through coarse_prefill_with_state
        // and stashed `kv_handle`, continue on that path. State capture
        // gives us per-layer h_in / K_new / V_new to update engine state.
        if self.kv_handle.is_some() {
            return self
                .decode_step_via_dispatch(weights, index, token_id)
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "decode_step_via_dispatch returned None".into(),
                });
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let rs = self
            .store
            .take()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step_quant called before prefill (store missing)".into(),
            })?;
        let prof = self.profiling.then_some(&mut self.profile);
        let (hidden, new_rs) = rs_decode_step_walk(view, index, token_id, rs, backend, prof)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "rs_decode_step_walk returned None".into(),
            })?;
        self.store = Some(new_rs);
        self.abs_position += 1;
        Ok(hidden)
    }

    // ── Executor-aware migration (Phase 2 of engine-state-vs-execution spec) ──
    //
    // The methods below override the trait defaults to drive the layer
    // loop through a caller-supplied `LayerExecutor` and honor the
    // caller-supplied `FfnBackend`. Old `prefill_quant` /
    // `decode_step_quant` stay above for backward compat; they construct
    // their own WalkFfn and ignore the FFN parameter. The new methods
    // are what remote-FFN deployments and per-layer codec engines must
    // call to get the engine's contract.

    fn prefill_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        use crate::engines::markov_residual::recompute_kv;
        use larql_inference::attention::SharedKV;
        use larql_inference::forward::embed_tokens_pub;
        use larql_inference::layer_executor::ExecutorDispatchKind;
        use ndarray::Array2;
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Engines whose state policy requires per-layer dispatch (this
        // one) must refuse fused executors at construction. Until the
        // `requires_per_layer_dispatch()` trait hook lands (Phase 3),
        // degrade transparently to the legacy fused-or-walk path.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }

        // Q4K attn weights need dequant once before the per-layer
        // executor can drive f32 attention against them.
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);

        let backend = executor.backend();
        let num_layers = weights.num_layers;
        let seq_len = token_ids.len();
        let mut h = embed_tokens_pub(weights, token_ids);
        // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
        let ple_inputs =
            larql_inference::forward::ple::precompute_per_layer_inputs(weights, &h, token_ids);
        let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            stored.push(h.clone());
            // Executor drives attention + FFN; engine doesn't care which
            // backend or whether FFN is local/remote. Engine discards
            // the layer's K/V — residual-stream contract recomputes K/V
            // per decode step from the stored residuals.
            let (h_out, _kv) = executor
                .run_prefill_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h,
                    ffn,
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "executor.run_prefill_layer returned None".into(),
                })?;
            // `LayerExecutor::run_*_layer` returns attention + bare FFN only
            // (`LocalWalkExecutor`, the sole production impl, ends at
            // `run_ffn`); the PLE + layer_scalar tail is the driving loop's
            // responsibility, mirroring the legacy `kv_prefill_run` sequence.
            h = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_out,
                layer,
                ple_inputs.get(layer),
            );
        }

        // State management identical to `rs_prefill_walk`: build the
        // store, clip overflow into cold tier, precompute cold K/V via
        // `recompute_kv` (engine policy — the executor doesn't own this).
        let mut rs = RsStore {
            hot_len: stored.first().map_or(0, |s| s.shape()[0]),
            stored,
            cold_residuals: None,
            cold_kv: None,
            cold_len: 0,
            // Executor path doesn't yet capture K/V from the executor's
            // `run_prefill_layer` return; falls back to recompute-on-decode
            // for now (W2 follow-up: thread the captured K/V through
            // `LayerExecutor::run_prefill_layer`'s return tuple).
            hot_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: self.window_size,
        };
        let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            rs.clip_layer(layer, &mut cold);
        }
        rs.finalise_hot_len_after_clip();
        if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
            let cold_kv: Vec<SharedKV> = (0..num_layers)
                .map(|layer| {
                    recompute_kv(
                        larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                        &cold[layer],
                        layer,
                        0,
                        backend,
                        Some(index),
                    )
                    .expect("cold K/V pre-computation failed")
                })
                .collect();
            // 2026-05-19 audit fix: doubling-capacity append.
            rs.append_cold_overflow(cold, Some(cold_kv));
            rs.cold_abs_start = 0;
        }

        let hidden = {
            use ndarray::s;
            let last = h.shape()[0] - 1;
            h.slice(s![last..=last, ..]).to_owned()
        };
        self.store = Some(rs);
        Ok(hidden)
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        use crate::engines::markov_residual::recompute_kv;
        use larql_inference::attention::SharedKV;
        use larql_inference::forward::embed_tokens_pub;
        use larql_inference::layer_executor::ExecutorDispatchKind;
        use ndarray::{s, Array2};

        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }

        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);

        let backend = executor.backend();
        let rs = self
            .store
            .take()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step_quant_via_executor called before prefill (store missing)".into(),
            })?;
        let num_layers = weights.num_layers;
        let abs_position = rs.next_position;
        let mut h_new = embed_tokens_pub(weights, &[token_id]);
        // PLE inputs are per-token — recompute for this single-token decode
        // step, matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = larql_inference::forward::ple::precompute_per_layer_inputs(
            weights,
            &h_new,
            &[token_id],
        );
        let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            // Logical lengths only — `stored` and the cold buffers are
            // doubling-capacity (`hot_len` / `cold_len` are the row
            // counts, see RsStore docs); the zero rows past the logical
            // length must never reach attention.
            let s_hot = rs.hot_len;
            let h_hot = rs.stored[layer].slice(s![..s_hot, ..]);
            let hot_abs_start = abs_position.saturating_sub(s_hot);

            // Engine assembles the K/V to attend against from its store.
            // The executor doesn't own state — it just receives prior_kv
            // and runs the layer.
            let prior_kv: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold_buf, v_cold_buf) = &cold_kv[layer];
                let h_hot_owned = h_hot.to_owned();
                let (k_hot, v_hot) = recompute_kv(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    &h_hot_owned,
                    layer,
                    hot_abs_start,
                    backend,
                    Some(index),
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "recompute_kv (hot) returned None".into(),
                })?;
                let c = rs.cold_len;
                let kv_dim = k_cold_buf.shape()[1];
                let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                k_combined
                    .slice_mut(s![..c, ..])
                    .assign(&k_cold_buf.slice(s![..c, ..]));
                k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
                let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                v_combined
                    .slice_mut(s![..c, ..])
                    .assign(&v_cold_buf.slice(s![..c, ..]));
                v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
                (k_combined, v_combined)
            } else {
                let s_cold = if rs.cold_residuals.is_some() {
                    rs.cold_len
                } else {
                    0
                };
                let (h_full, full_abs_start) = match &rs.cold_residuals {
                    Some(cold) if s_cold > 0 => {
                        let h_cold = cold[layer].slice(s![..s_cold, ..]);
                        let hidden = h_hot.shape()[1];
                        let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                        combined.slice_mut(s![..s_cold, ..]).assign(&h_cold);
                        combined.slice_mut(s![s_cold.., ..]).assign(&h_hot);
                        (combined, rs.cold_abs_start)
                    }
                    _ => (h_hot.to_owned(), hot_abs_start),
                };
                recompute_kv(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    &h_full,
                    layer,
                    full_abs_start,
                    backend,
                    Some(index),
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "recompute_kv (full) returned None".into(),
                })?
            };

            new_stored.push(h_new.clone());
            // Run the layer through the executor.
            let (h_out, _new_kv) = executor
                .run_decode_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h_new,
                    &prior_kv,
                    abs_position,
                    ffn,
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "executor.run_decode_layer returned None".into(),
                })?;
            // Executor returns bare post-FFN hidden; PLE + layer_scalar tail
            // is the driving loop's responsibility (see prefill loop above).
            h_new = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_out,
                layer,
                ple_inputs.get(layer),
            );
        }

        // Append new row to store, clip overflow into cold. Note: this
        // is the executor (non-dispatch) decode path, which doesn't go
        // through the W8.2 hot-path optimisation — it still allocates
        // a fresh Array2 per step. The CPU/executor path is a fallback;
        // the dispatch hot path in `decode_step_via_dispatch` is the
        // one that matters for tok/s.
        let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
            let s_old_logical = rs.hot_len; // logical row count
            let hidden_dim = stored.shape()[1];
            let mut combined = Array2::<f32>::zeros((s_old_logical + 1, hidden_dim));
            if s_old_logical > 0 {
                combined
                    .slice_mut(s![..s_old_logical, ..])
                    .assign(&stored.slice(s![..s_old_logical, ..]));
            }
            combined.slice_mut(s![s_old_logical.., ..]).assign(new_row);
            updated_stored.push(combined);
        }

        let mut updated_rs = RsStore {
            hot_len: updated_stored.first().map_or(0, |s| s.shape()[0]),
            stored: updated_stored,
            cold_residuals: rs.cold_residuals,
            cold_kv: rs.cold_kv,
            cold_len: rs.cold_len,
            // `run_decode_layer`'s returned K/V row is not appended here,
            // so a carried-through hot_kv would be one row short of the
            // new `hot_len` — stale reads on the next walk step, or an
            // out-of-bounds clip below. Drop it; the next step recomputes
            // from `stored`.
            hot_kv: None,
            cold_abs_start: rs.cold_abs_start,
            next_position: abs_position + 1,
            max_window: rs.max_window,
        };

        let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            updated_rs.clip_layer(layer, &mut overflow);
        }
        updated_rs.finalise_hot_len_after_clip();
        // 2026-05-19 audit fix: doubling-capacity append. Via-executor
        // path doesn't carry evicted_hot_kv, so the helper invalidates
        // cold_kv (matches the prior `updated_rs.cold_kv = None` behaviour).
        updated_rs.append_cold_overflow(overflow, None);

        let last = h_new.shape()[0] - 1;
        let out = h_new.slice(s![last..=last, ..]).to_owned();
        self.store = Some(updated_rs);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvEngine;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

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
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > 0,
            "store should be non-empty after prefill"
        );
    }

    #[test]
    fn decode_step_produces_finite_logits() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(hidden_to_raw_logits(&weights, &h)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn memory_grows_with_each_decode_step() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let mem_after_prefill = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 1).expect("decode 1");
        let mem_after_1 = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 2).expect("decode 2");
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
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(Some(2)); // window=2 tokens
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3, 4])
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
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        for step in 0..3 {
            let h = engine
                .decode_step(&weights, &ffn, step as u32)
                .expect("decode");
            assert_eq!(h.shape(), &[1, weights.hidden_size], "step {step}");
        }
    }

    // ── Profiling ─────────────────────────────────────────────────────────────

    #[test]
    fn with_profiling_enables_profiling_branch() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None).with_profiling(true);
        // No decode yet → stage_summary returns None even with profiling on.
        assert!(engine.stage_summary().is_none());

        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        engine.decode_step(&weights, &ffn, 2).expect("decode");

        let summary = engine.stage_summary().expect("profiling summary");
        assert_eq!(summary.engine, "markov-rs");
        assert_eq!(summary.steps, 1);
        assert!(summary.avg_total_decode_us > 0.0);
    }

    #[test]
    fn stage_summary_none_without_profiling() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = MarkovResidualEngine::new(None); // profiling: false
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        engine.decode_step(&weights, &ffn, 1).expect("decode");
        assert!(
            engine.stage_summary().is_none(),
            "stage_summary must be None when profiling is disabled"
        );
    }

    #[test]
    fn profiling_decode_path_matches_unprofiled_shape() {
        // Two engines: one profiled, one not. Both should yield hidden states
        // of the same shape after the same prefill+decode sequence.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut profiled = MarkovResidualEngine::new(None).with_profiling(true);
        let mut plain = MarkovResidualEngine::new(None);
        profiled.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        plain.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
        let h_p = profiled.decode_step(&weights, &ffn, 2).unwrap();
        let h_n = plain.decode_step(&weights, &ffn, 2).unwrap();
        assert_eq!(h_p.shape(), h_n.shape());
    }

    // ── Q4K paths via CPU fallback ────────────────────────────────────────
    //
    // On a CPU backend, `fused_prefill` returns `None`, so the engine
    // falls through to `rs_prefill_walk` against the synthetic VectorIndex.
    // This exercises the prefill_quant / decode_step_quant branches that the
    // Metal-only happy path also takes (apart from the Metal early-return).

    #[test]
    fn prefill_q4k_cpu_fallback_runs_walk_path() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        // `NullFfn` satisfies the trait without borrowing `weights`, which is
        // `&mut` here. The engine ignores the FFN parameter on this path.
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_q4k_cpu_fallback_extends_store() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > mem_before,
            "store should grow after decode_step_quant"
        );
    }

    // ── Walk-path overflow branches (markov_residual/walk.rs) ────────────
    //
    // The two tests above use `window=None` so the cold tier never
    // populates — leaving the walk.rs cold-K/V precompute (lines 66-76)
    // and cold-residual decode branch (lines 162-186) uncovered. The
    // tests below drive them with a small window + multiple decode steps.

    #[test]
    fn prefill_quant_walk_with_window_populates_cold_kv() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill_quant with overflow");
        // window=2 + 4 prompt tokens → cold tier populated → walk.rs
        // lines 67-75 fire.
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
    }

    /// W2 parity: the cached-hot_kv decode path must produce the
    /// SAME hidden state as the legacy recompute-from-residuals path,
    /// bit-for-bit (or within fp rounding). Drives a few decode steps
    /// with caching enabled (default since W2) against a manually
    /// hot_kv-cleared store that forces the legacy fallback.
    #[test]
    fn decode_step_quant_w2_cached_matches_recompute_from_residuals() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;

        // Cached path (W2 default): prefill captures K/V, decode reuses.
        let mut cached = MarkovResidualEngine::new(None);
        cached
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill cached");
        let h_cached_1 = cached
            .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
            .expect("decode cached 1");
        let h_cached_2 = cached
            .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
            .expect("decode cached 2");

        // Recompute path: same engine, but force hot_kv = None after
        // prefill so the fallback recompute fires for every step.
        let mut recompute = MarkovResidualEngine::new(None);
        recompute
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill recompute");
        if let Some(s) = recompute.store.as_mut() {
            s.hot_kv = None;
        }
        let h_recompute_1 = recompute
            .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
            .expect("decode recompute 1");
        if let Some(s) = recompute.store.as_mut() {
            s.hot_kv = None;
        }
        let h_recompute_2 = recompute
            .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
            .expect("decode recompute 2");

        // Bit-equivalence: both paths run the same projection matmuls
        // at the same RoPE positions, so output must match within
        // f32 rounding. (Hidden states aren't normalised here; they
        // come straight from the layer stack.)
        for (a, b) in h_cached_1.iter().zip(h_recompute_1.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "step 1 diverged: cached={a}, recompute={b}"
            );
        }
        for (a, b) in h_cached_2.iter().zip(h_recompute_2.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "step 2 diverged: cached={a}, recompute={b}"
            );
        }
    }

    /// W2 fast path: both cold_kv AND hot_kv cached. Drives the
    /// triple-condition branch in `rs_decode_step_walk` that
    /// concatenates a cached cold tier with a cached hot tier
    /// (memcpy only, no projection). Achieved by prefilling past
    /// the window, then doing several decodes so cold_kv stays
    /// populated across steps.
    #[test]
    fn decode_step_quant_w2_cached_hot_and_cold_steady_state() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        // window=2, 4-token prompt → prefill overflows once,
        // populating cold_kv from the evicted hot_kv slice.
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill with overflow");
        let store = engine.store.as_ref().unwrap();
        assert!(store.hot_kv.is_some());
        assert!(store.cold_kv.is_some(), "prefill should populate cold_kv");

        // Multiple decodes — each appends a row to hot_kv (W2 fast
        // path with BOTH caches populated). Subsequent overflows
        // merge into cold_kv via the W2 evicted-K/V flow.
        for tok in 4u32..8 {
            let h = engine
                .decode_step_quant(&weights, &ffn, &index, tok, &*backend)
                .expect("decode");
            assert_eq!(h.shape(), &[1, weights.hidden_size]);
        }
        let store = engine.store.as_ref().unwrap();
        assert!(store.hot_kv.is_some());
        assert!(
            store.cold_kv.is_some(),
            "cold_kv stays populated across steps"
        );
        // Cold grew by ~3 rows (one per decode after the prefill cycle).
        let cold_rows = store.cold_kv.as_ref().unwrap()[0].0.shape()[0];
        assert!(
            cold_rows >= 3,
            "cold_kv should grow with successive overflows, got {cold_rows}"
        );
    }

    /// Drive the fallback path where `hot_kv` was dropped (legacy
    /// recompute-from-residuals). Covers the `if let Some(cold_kv) =
    /// &rs.cold_kv` branch with hot_kv=None — the pre-W2 behaviour
    /// that's still reachable via the via_executor path.
    #[test]
    fn decode_step_quant_w2_falls_back_when_hot_kv_dropped() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill");
        // Drop hot_kv — forces the recompute path that mirrors pre-W2.
        engine.store.as_mut().unwrap().hot_kv = None;
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
            .expect("decode via fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// W2 cache survives window-overflow: when stored is clipped, the
    /// evicted hot_kv rows merge into cold_kv (vs the legacy invalidation
    /// that cleared cold_kv and forced recompute on the next step).
    #[test]
    fn decode_step_quant_w2_overflow_merges_into_cold_kv() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;

        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill within window");
        // After prefill: hot_kv populated (2 rows), no cold_kv.
        assert!(engine.store.as_ref().unwrap().hot_kv.is_some());
        assert!(engine.store.as_ref().unwrap().cold_kv.is_none());
        // Decode a token → no overflow yet (still 2 rows after step
        // since window=2, the new row pushes the oldest out).
        let _ = engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode 1");
        // Overflow fired this step: oldest row evicted from hot_kv,
        // merged into cold_kv.
        let store = engine.store.as_ref().unwrap();
        assert!(
            store.cold_kv.is_some(),
            "post-overflow cold_kv should be populated from evicted hot_kv"
        );
        assert!(store.hot_kv.is_some(), "hot_kv stays alive");
    }

    /// Drive `rs_decode_step_walk`'s `Some(profiler)` branches — the
    /// non-profiled path is covered by `decode_step_q4k_cpu_fallback_*`;
    /// the profiled-arm branches are only reached when the engine is
    /// built with `with_profiling(true)`. Without this test the
    /// `if let (Some(prof), Some(t_step)) = ...` accumulation and the
    /// per-stage `if timing { ... }` arms stay uncovered.
    #[test]
    fn decode_step_q4k_walk_with_profiling_populates_summary() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2)).with_profiling(true);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill");
        // First decode: cold_kv branch (hot recompute timing arm).
        engine
            .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
            .expect("decode 1");
        // Second decode: cold_residuals branch (cold recompute timing arm).
        engine
            .decode_step_quant(&weights, &ffn, &index, 5, &*backend)
            .expect("decode 2");
        let summary = engine
            .stage_summary()
            .expect("Q4K walk profiler should populate summary");
        assert_eq!(summary.engine, "markov-rs");
        assert!(summary.steps >= 2);
        // The walk path accumulates into `recompute_*` (one of the two
        // branches will be non-zero depending on which fired); attention
        // and ffn always fire.
        assert!(summary.avg_attention_us > 0.0);
        assert!(summary.avg_ffn_us > 0.0);
        assert!(summary.avg_total_decode_us > 0.0);
    }

    #[test]
    fn decode_step_quant_walk_first_overflow_creates_cold_residuals() {
        // walk.rs lines 305-307: `None => updated_rs.cold_residuals =
        // Some(overflow)`. Fires when prefill didn't overflow (cold = None)
        // but the first decode does (window cap exceeded mid-decode).
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        // window=2, prefill=1 token → no overflow on prefill (cold=None).
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32], &*backend)
            .expect("prefill_quant");
        // Decode until hot exceeds window → first-time cold population.
        engine
            .decode_step_quant(&weights, &ffn, &index, 1, &*backend)
            .expect("decode 1");
        engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode 2 — triggers first-overflow None branch");
        // After overflow, cold tier is populated.
        assert!(engine.cold_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_walk_after_overflow_hits_cold_residuals_branch() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill_quant");
        // First decode: exercises walk.rs cold_kv branch (lines 132-161).
        engine
            .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
            .expect("first decode_step_quant");
        // Second decode: cold_kv was cleared by overflow at the first
        // decode (walk.rs line 309), so this hits the cold_residuals
        // recompute branch (lines 162-187).
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 5, &*backend)
            .expect("second decode_step_quant");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    // ── Phase 2: executor-driven path ─────────────────────────────────────

    #[test]
    fn prefill_quant_via_executor_runs_through_local_walk() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("executor prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_via_executor_extends_store() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > mem_before);
    }

    /// Counting FFN that records every `forward` call. Used to prove
    /// the executor path actually dispatches through the caller's
    /// `FfnBackend` instead of constructing a local `WalkFfn` (the
    /// legacy coupling that the migration removes).
    struct CountingFfn {
        calls: std::sync::atomic::AtomicUsize,
        hidden: usize,
    }
    impl larql_inference::ffn::FfnBackend for CountingFfn {
        fn forward(&self, _layer: usize, x: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ndarray::Array2::zeros((x.shape()[0], self.hidden))
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    #[test]
    fn executor_path_honors_ffn_parameter() {
        // Pass a counting stub. If the engine constructs its own
        // WalkFfn internally (the legacy bug we're fixing) the counter
        // stays at zero.
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);

        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let mut engine = MarkovResidualEngine::new(None);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill via executor");

        let call_count = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
        // Prefill runs FFN once per layer.
        assert_eq!(
            call_count, weights.num_layers,
            "executor path should dispatch FFN through the supplied backend \
             once per layer; got {call_count} for {} layers — engine is \
             likely constructing its own FFN internally",
            weights.num_layers
        );
    }

    #[test]
    fn prefill_quant_via_executor_with_window_populates_cold_tier() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill with overflow");
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
    }

    /// Drive `decode_step_quant_via_executor`'s `cold_kv` branch (lines
    /// 315-333): prefill with overflow so the engine pre-computes
    /// cold_kv during prefill, then run a single decode step that
    /// combines cold_kv + hot K/V for attention.
    #[test]
    fn decode_step_quant_via_executor_uses_cold_kv_branch() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill overflow → cold_kv populated");
        // First decode reads cold_kv branch (rs.cold_kv = Some(_)).
        let h = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 4)
            .expect("decode via cold_kv branch");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Drive the cold_residuals branch when cold_kv has been cleared
    /// (the second decode after overflow). At line ~399 the engine
    /// clears cold_kv when a new overflow happens, then subsequent
    /// decodes recompute K/V from cold_residuals.
    #[test]
    fn decode_step_quant_via_executor_hits_cold_residuals_branch() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill");
        // First decode clears cold_kv via overflow.
        engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 4)
            .expect("first decode");
        // Second decode: cold_kv is None, exercises the recompute_kv
        // from cold_residuals branch.
        let h = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 5)
            .expect("decode via cold_residuals recompute");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// `Fused`-executor fallback in `*_via_executor` (lines 219-221, 294-296):
    /// when the executor advertises fused dispatch, the engine routes back
    /// through the legacy `prefill_quant` / `decode_step_quant` path.
    struct FusedStubExecutor {
        backend: larql_compute::CpuBackend,
    }
    impl larql_inference::layer_executor::LayerExecutor for FusedStubExecutor {
        fn backend(&self) -> &dyn larql_compute::ComputeBackend {
            &self.backend
        }
        fn dispatch_kind(&self) -> larql_inference::layer_executor::ExecutorDispatchKind {
            larql_inference::layer_executor::ExecutorDispatchKind::Fused
        }
        fn name(&self) -> &str {
            "fused-stub"
        }
    }

    #[test]
    fn fused_executor_falls_back_to_legacy_quant_path() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let exec = FusedStubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill_quant_via_executor(&weights, &exec, &ffn, &index, &[0u32, 1])
            .expect("fused fallback prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let h2 = engine
            .decode_step_quant_via_executor(&weights, &exec, &ffn, &index, 2)
            .expect("fused fallback decode");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    }

    // ── Q4K dispatch path (try_prefill_via_dispatch + decode_step_via_dispatch) ──
    //
    // CpuBackend implements `coarse_prefill_with_state` /
    // `coarse_decode_step_with_state_masked` on Q4K-backed vindexes,
    // so the dispatch fast path fires on CPU when fed a Q4K fixture.
    // This is where the per-layer state-capture branches live —
    // including the `append_row` / `grow_capacity_2d` doubling-capacity
    // buffers in lines 17-55.

    #[test]
    fn prefill_via_dispatch_with_q4k_vindex_populates_store_and_handle() {
        if crate::engines::w10_enabled() {
            // W10 opt-in deliberately drops these shadows; test verifies the
            // default (Full mask) population, so skip when the env-gated
            // optimisation is active.
            return;
        }
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let h = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("dispatch-path prefill on Q4K vindex");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // Dispatch path populates kv_handle + abs_position.
        assert!(engine.kv_handle.is_some());
        assert_eq!(engine.abs_position, 3);
        assert!(engine.memory_bytes() > 0);
        let store = engine.store.as_ref().expect("store populated");
        assert_eq!(store.next_position, 3);
        // hot_kv shadow is populated by default (no LARQL_W10_HONLY).
        assert!(store.hot_kv.is_some());
        // No window → no overflow → cold tiers stay None.
        assert!(store.cold_residuals.is_none());
        assert!(store.cold_kv.is_none());
    }

    #[test]
    fn decode_via_dispatch_grows_buffers_in_place() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill dispatch");
        let mem_after_prefill = engine.memory_bytes();
        for tok in 2..6u32 {
            let h = engine
                .decode_step_quant(&weights, &ffn, &index, tok, &*backend)
                .expect("dispatch decode step");
            assert_eq!(h.shape(), &[1, weights.hidden_size]);
        }
        assert_eq!(engine.abs_position, 6);
        assert!(engine.memory_bytes() >= mem_after_prefill);
        assert!(engine.kv_handle.is_some());
    }

    #[test]
    fn dispatch_decode_with_profiling_records_w10_stages() {
        if crate::engines::w10_enabled() {
            // The mask cascade changes which timer slots fire (None mask
            // skips state_materialise/append); this test pins Full-mask
            // behaviour. Add a separate test for HOnly/None coverage.
            return;
        }
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None).with_profiling(true);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill");
        engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode 1");
        engine
            .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
            .expect("decode 2");
        // W10 instrumentation: dispatch path bumps state_capture +
        // state_materialise + state_append + decode_total on every step.
        let summary = engine
            .stage_summary()
            .expect("profiler should produce a summary on the dispatch path");
        assert_eq!(summary.engine, "markov-rs");
        assert!(summary.steps >= 2);
        assert!(summary.avg_state_capture_us > 0.0);
        assert!(summary.avg_state_materialise_us > 0.0);
        assert!(summary.avg_state_append_us > 0.0);
        assert!(summary.avg_total_decode_us > 0.0);
    }

    #[test]
    fn dispatch_prefill_with_window_evicts_to_cold_tier() {
        if crate::engines::w10_enabled() {
            // Window-driven cold-tier eviction depends on the Full-mode
            // shadow being populated; HOnly retains it (window != None
            // here so the HOnly branch keeps rs.stored).
            // Test pins Full-mask behaviour for clarity.
            return;
        }
        // window < prompt_len triggers the on-prefill eviction branch
        // inside try_prefill_via_dispatch (cold_residuals + cold_kv get
        // populated from snapshot_evicted_hot_kv).
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill with window");
        let store = engine.store.as_ref().expect("store");
        assert!(store.cold_residuals.is_some());
        assert!(store.cold_kv.is_some());
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
    }

    #[test]
    fn dispatch_decode_overflow_extends_cold_tier_from_evicted_hot_kv() {
        // Prefill at window cap; each decode step evicts one row into
        // the cold tier. Exercises the post-decode cold-merge branch
        // (lines ~423-462) including the `Some` arm where prior cold
        // K/V already exists and the eviction concatenates.
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(Some(2));
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill at window cap");
        for tok in 3..6u32 {
            engine
                .decode_step_quant(&weights, &ffn, &index, tok, &*backend)
                .expect("dispatch decode with overflow");
        }
        let store = engine.store.as_ref().expect("store");
        assert!(store.cold_residuals.is_some());
        // Each decode evicts → cold tier grew across steps.
        let cold = store.cold_residuals.as_ref().unwrap();
        assert!(cold[0].shape()[0] >= 3);
        assert!(engine.window_tokens() <= 2);
    }

    // ─── EngineError surface coverage ────────────────────────────────────────
    //
    // The Option → Result migration added typed-error paths at every
    // method entry. These tests pin the entry guards so the
    // `EmptyPrompt` / `InvariantViolation` variants stay constructible
    // (collapsing them back into a plain `BackendFailure` would
    // re-introduce the silent-failure problem the refactor exists to
    // fix). Backend-failure / dispatch-None paths are exercised
    // through the existing positive-path tests above when their
    // synthetic backends behave normally; only the entry guards have
    // dedicated negative tests here.

    #[test]
    fn prefill_returns_empty_prompt_error_on_empty_input() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let err = engine.prefill(&weights, &ffn, &[]).unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::EmptyPrompt
        ));
    }

    #[test]
    fn decode_step_returns_invariant_violation_before_prefill() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let err = engine.decode_step(&weights, &ffn, 0).unwrap_err();
        assert!(
            matches!(
                err,
                larql_inference::kv_engine::EngineError::InvariantViolation { .. }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn prefill_quant_returns_empty_prompt_error_on_empty_input() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let err = engine
            .prefill_quant(&weights, &ffn, &index, &[], &*backend)
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::EmptyPrompt
        ));
    }

    #[test]
    fn decode_step_quant_returns_invariant_violation_before_prefill() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let err = engine
            .decode_step_quant(&weights, &ffn, &index, 0, &*backend)
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::InvariantViolation { .. }
        ));
    }

    #[test]
    fn prefill_via_executor_returns_empty_prompt_error_on_empty_input() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let backend = larql_compute::CpuBackend;
        let executor = larql_inference::layer_executor::LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let err = engine
            .prefill_via_executor(&weights, &executor, &ffn, &[])
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::EmptyPrompt
        ));
    }

    #[test]
    fn prefill_quant_via_executor_returns_empty_prompt_error_on_empty_input() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::CpuBackend;
        let executor = larql_inference::layer_executor::LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let err = engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[])
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::EmptyPrompt
        ));
    }

    #[test]
    fn decode_step_quant_via_executor_returns_invariant_violation_before_prefill() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::CpuBackend;
        let executor = larql_inference::layer_executor::LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let err = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 0)
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::InvariantViolation { .. }
        ));
    }

    // ── Doubling-capacity + hot_kv-consistency regressions (executor path) ──

    /// Prompt/window pair that makes the prefill overflow so the cold
    /// buffers are capacity-padded (`cold_len = 2`, capacity 8).
    const EXEC_WINDOW: usize = 2;
    const EXEC_PROMPT: [u32; 4] = [0, 1, 2, 3];
    const EXEC_NEXT_TOKEN: u32 = 4;

    fn assert_bits_eq(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>, ctx: &str) {
        let a_bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
        let b_bits: Vec<u32> = b.iter().map(|v| v.to_bits()).collect();
        assert_eq!(a_bits, b_bits, "{ctx}");
    }

    #[test]
    fn decode_via_executor_padded_cold_buffers_match_trimmed_reference() {
        // The executor decode must read `cold_len` / `hot_len`, not the
        // buffers' capacity. Decode the same logical state over the
        // padded buffers and over buffers trimmed to exactly `cold_len`
        // (the layout the pre-migration code assumed) — outputs must be
        // bit-identical. The padded run previously attended six
        // zero-K/V phantom rows per layer.
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use ndarray::s;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;

        let run = |trim: bool| -> ndarray::Array2<f32> {
            let mut engine = MarkovResidualEngine::new(Some(EXEC_WINDOW));
            engine
                .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &EXEC_PROMPT)
                .expect("prefill with overflow");
            {
                let store = engine.store.as_mut().unwrap();
                let c = store.cold_len;
                assert!(
                    store.cold_kv.as_ref().unwrap()[0].0.shape()[0] > c,
                    "fixture must be capacity-padded"
                );
                if trim {
                    for layer in store.cold_residuals.as_mut().unwrap().iter_mut() {
                        *layer = layer.slice(s![..c, ..]).to_owned();
                    }
                    for (k, v) in store.cold_kv.as_mut().unwrap().iter_mut() {
                        *k = k.slice(s![..c, ..]).to_owned();
                        *v = v.slice(s![..c, ..]).to_owned();
                    }
                }
            }
            engine
                .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, EXEC_NEXT_TOKEN)
                .expect("executor decode")
        };

        let h_padded = run(false);
        let h_trimmed = run(true);
        assert_bits_eq(
            &h_padded,
            &h_trimmed,
            "executor decode attended capacity slack instead of cold_len rows",
        );
    }

    #[test]
    fn decode_via_executor_after_walk_prefill_drops_hot_kv_and_stays_consistent() {
        // The executor decode discards `run_decode_layer`'s returned K/V
        // while `hot_len` grows. Carrying the walk-prefill `hot_kv`
        // through unchanged violated the store invariant (kv[l] and
        // stored[l] agree on the first hot_len rows) and made the
        // post-step window clip slice out of bounds. The fixed path
        // drops the cache; attention never reads it here, so runs with
        // and without a prefill-seeded hot_kv must be bit-identical.
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;

        let run = |drop_hot_kv_after_prefill: bool| -> ndarray::Array2<f32> {
            let mut engine = MarkovResidualEngine::new(Some(EXEC_WINDOW));
            // Walk-path prefill captures hot_kv (2 rows, no overflow).
            engine
                .prefill_quant(
                    &weights,
                    &ffn,
                    &index,
                    &EXEC_PROMPT[..EXEC_WINDOW],
                    &*backend,
                )
                .expect("walk prefill");
            assert!(engine.store.as_ref().unwrap().hot_kv.is_some());
            if drop_hot_kv_after_prefill {
                engine.store.as_mut().unwrap().hot_kv = None;
            }
            // Two decode steps: the first overflows the window (the
            // out-of-bounds clip on the old code), the second reads
            // whatever state the first left behind.
            engine
                .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
                .expect("executor decode 1");
            let store = engine.store.as_ref().unwrap();
            assert!(
                store.hot_kv.is_none(),
                "executor decode must not carry a hot_kv it did not extend"
            );
            engine
                .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 3)
                .expect("executor decode 2")
        };

        let h_with_seeded_cache = run(false);
        let h_without_cache = run(true);
        assert_bits_eq(
            &h_with_seeded_cache,
            &h_without_cache,
            "stale hot_kv leaked into the executor decode path",
        );
    }

    // ── §4 architecture-precondition gate ─────────────────────────────────

    /// Stub violating the residual-recompute contract structurally
    /// (MLA: K/V routed through a decode-time latent). Everything else
    /// inherits the trait defaults from the test fixture's config.
    struct MlaStubArch {
        config: larql_inference::larql_models::ModelConfig,
    }
    impl larql_inference::larql_models::ModelArchitecture for MlaStubArch {
        fn family(&self) -> &str {
            "mla-stub"
        }
        fn config(&self) -> &larql_inference::larql_models::ModelConfig {
            &self.config
        }
        fn uses_mla(&self) -> bool {
            true
        }
    }

    #[test]
    fn prefill_refuses_arch_violating_residual_recompute_preconditions() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        weights.arch = Box::new(MlaStubArch {
            config: weights.arch.config().clone(),
        });
        assert!(!weights.arch.kv_recomputable_from_residuals());

        let ffn = NullFfn;
        let mut engine = MarkovResidualEngine::new(None);
        let err = engine.prefill(&weights, &ffn, &[0u32, 1]).unwrap_err();
        assert!(
            matches!(
                err,
                larql_inference::kv_engine::EngineError::InvariantViolation { .. }
            ),
            "precondition violation must fail closed, got {err:?}"
        );
        assert!(
            engine.store.is_none(),
            "no state may be built after refusal"
        );

        // Same gate on the quant entry point.
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let err = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .unwrap_err();
        assert!(matches!(
            err,
            larql_inference::kv_engine::EngineError::InvariantViolation { .. }
        ));
    }
}
