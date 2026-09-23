//! One codec-cold-tier decode step.
//!
//! # Failure invariant
//!
//! Identical to [`crate::engines::markov_residual::step`]: the store is
//! borrowed rather than consumed, canonical state (`stored`, `hot_len`, the
//! cold tiers, `next_position`) is written only in [`commit`] after the last
//! fallible call, and `hot_kv` — a droppable derivative — is taken up front
//! and left `None` on the error path. A refused step therefore rewinds
//! exactly, and the same token can be retried on the same engine.

use larql_compute::ComputeBackend;
use larql_inference::attention::SharedKV;
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::kv_engine::EngineError;
use ndarray::Array2;

use super::helpers::last_row;
use super::step_attention::{resolve_layer_attention, HotKv};
use super::store::RsStoreCodec;

mod commit;
#[cfg(test)]
mod tests;

use commit::commit;

/// Advance `rs` by one token, returning the new last hidden row.
pub fn rs_decode_step_codec(
    weights: larql_inference::WeightsView,
    new_token_id: u32,
    rs: &mut RsStoreCodec,
    backend: &dyn ComputeBackend,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Result<Array2<f32>, EngineError> {
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(&weights, &[new_token_id]);
    // PLE inputs are per-token — recompute for this single-token decode
    // step, matching the legacy `kv_decode_step_run` recipe exactly.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h_new, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    // W2 hot-K/V cache on the resident walk (2026-06-13), twin of
    // markov_residual: with no cold tier, `hot_kv` holds the FULL K/V and is
    // read instead of re-deriving via `recompute_kv` each step. `stored`
    // remains the canonical re-derivable state. Only for unbounded windows
    // (the default): `clip_layer_overflow` is then a no-op, so the cache never
    // tracks a window-eviction transition.
    let cache_eligible =
        rs.max_window.is_none() && rs.cold_encoded.is_none() && rs.cold_kv.is_none();
    let mut step_new_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
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
        )
        .ok_or_else(|| EngineError::BackendFailure {
            details: format!("attention returned None during codec decode at layer {layer}"),
        })?;

        let bffn = BackendFfn {
            weights: weights.canonical(),
            backend,
        };
        h_new = crate::engines::layer_ffn_or_moe(
            weights.canonical(),
            &h_post_attn,
            layer,
            &bffn,
            moe_ffn,
            ple_inputs.get(layer),
        )
        .map_err(EngineError::Execution)?;
    }

    commit(
        weights.hidden_size,
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
