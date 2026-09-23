//! One residual-stream decode step.
//!
//! # Failure invariant
//!
//! The store is borrowed, not consumed, and every canonical field —
//! `stored`, `hot_len`, the cold tiers and `next_position` — is written only
//! in [`commit`], after the last thing that can fail. So an `Err` leaves the
//! store describing exactly the token sequence it described on entry, and the
//! caller may drive the same token through the same engine once the cause is
//! fixed.
//!
//! The one field that does change is `hot_kv`, taken up front and left `None`
//! on the error path. That is deliberate: it is a droppable derivative of
//! `stored` (see [`RsStore::hot_kv`]), a partially-appended in-place buffer
//! must not survive into a retry, and rebuilding it costs one step. This is
//! the same invariant `boundary_per_layer::walk::run_decode` documents — the
//! two engines are residual-canonical, so both can answer "rewind or
//! invalidate" with *rewind*, where a K/V-canonical engine cannot.

use larql_compute::ComputeBackend;
use larql_inference::attention::SharedKV;
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::kv_engine::EngineError;
use ndarray::Array2;

use super::compute::last_row;
use super::step_attention::{resolve_layer_attention, HotKv, StepTimings};
use super::store::RsStore;
use crate::profiler::EngineProfiler;

mod commit;
#[cfg(test)]
mod tests;

use commit::commit;

/// Advance `rs` by one token, returning the new last hidden row.
pub fn rs_decode_step(
    weights: larql_inference::WeightsView,
    new_token_id: u32,
    rs: &mut RsStore,
    backend: &dyn ComputeBackend,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Result<Array2<f32>, EngineError> {
    rs_decode_step_inner(weights, new_token_id, rs, backend, None, moe_ffn, index)
}

/// [`rs_decode_step`] with per-stage timings recorded into `profiler`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rs_decode_step_profiled(
    weights: larql_inference::WeightsView,
    new_token_id: u32,
    rs: &mut RsStore,
    backend: &dyn ComputeBackend,
    profiler: &mut EngineProfiler,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Result<Array2<f32>, EngineError> {
    rs_decode_step_inner(
        weights,
        new_token_id,
        rs,
        backend,
        Some(profiler),
        moe_ffn,
        index,
    )
}

#[allow(clippy::too_many_arguments)]
fn rs_decode_step_inner(
    weights: larql_inference::WeightsView,
    new_token_id: u32,
    rs: &mut RsStore,
    backend: &dyn ComputeBackend,
    mut profiler: Option<&mut EngineProfiler>,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Result<Array2<f32>, EngineError> {
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let timed = profiler.is_some();
    let t_step = timed.then(std::time::Instant::now);
    let mut timings = StepTimings::default();

    let mut h_new = embed_tokens_pub(&weights, &[new_token_id]);
    // PLE inputs are per-token — recompute for this single-token decode
    // step, matching the legacy `kv_decode_step_run` recipe exactly.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h_new, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    // W2 hot-K/V cache on the resident walk (2026-06-13). When there is no cold
    // tier (the common unbounded-window case), `hot_kv` holds the FULL K/V and
    // is read instead of re-deriving every position via `recompute_kv` (a
    // per-step O(N) matmul — the engine's bottleneck). Only for unbounded
    // windows (the default): then `clip_layer` is a no-op, so the cache never
    // has to track a window-eviction transition. Windowed configs keep the
    // recompute path unchanged.
    let cache_eligible =
        rs.max_window.is_none() && rs.cold_residuals.is_none() && rs.cold_kv.is_none();
    let mut step_new_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
    // Taken up front: see this module's failure invariant. Moving it out also
    // lets the steady state append into it mutably while `rs.stored` (a
    // disjoint field) is read immutably.
    let mut hot_kv_store = rs.hot_kv.take();
    let had_hot_kv = hot_kv_store.is_some();
    let idx_kv: Option<&dyn larql_compute::KvIndex> =
        index.map(|v| v as &dyn larql_compute::KvIndex);

    for layer in 0..num_layers {
        new_stored.push(h_new.clone());

        let hot_kv = match (cache_eligible && had_hot_kv, hot_kv_store.as_mut()) {
            (true, Some(bufs)) => HotKv::InPlace(bufs),
            _ => HotKv::Recompute,
        };
        let h_post_attn = resolve_layer_attention(
            weights,
            rs,
            layer,
            &h_new,
            abs_position,
            backend,
            idx_kv,
            hot_kv,
            &mut step_new_kv,
            cache_eligible,
            timed,
            &mut timings,
        )
        .ok_or_else(|| EngineError::BackendFailure {
            details: format!("attention returned None during MarkovRS decode at layer {layer}"),
        })?;

        let bffn = BackendFfn {
            weights: weights.canonical(),
            backend,
        };
        h_new = StepTimings::measure(timed, &mut timings.ffn_us, || {
            crate::engines::layer_ffn_or_moe(
                weights.canonical(),
                &h_post_attn,
                layer,
                &bffn,
                moe_ffn,
                ple_inputs.get(layer),
            )
        })
        .map_err(EngineError::Execution)?;
    }

    if let (Some(prof), Some(t_step)) = (profiler.as_mut(), t_step) {
        prof.recompute_cold.total_us += timings.recompute_cold_us;
        prof.recompute_cold.count += 1;
        prof.recompute_hot.total_us += timings.recompute_hot_us;
        prof.recompute_hot.count += 1;
        prof.attention.total_us += timings.attention_us;
        prof.attention.count += 1;
        prof.ffn.total_us += timings.ffn_us;
        prof.ffn.count += 1;
        prof.decode_total.record(t_step);
    }

    commit(
        rs,
        &new_stored,
        cache_eligible,
        had_hot_kv,
        hot_kv_store,
        step_new_kv,
        abs_position,
    );
    Ok(last_row(&h_new))
}
