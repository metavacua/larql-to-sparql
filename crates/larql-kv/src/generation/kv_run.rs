//! The per-layer prefill and decode building blocks every `KvEngine` and the
//! dispatch ring are measured against.
//!
//! These two functions are the **oracle**: `dispatch_parity` compares
//! `kv_prefill_via_dispatch` / `kv_decode_step_via_dispatch` against them, and
//! `engine_ple_parity` compares the engines against them. That is why they
//! dispatch experts through [`crate::engines::layer_ffn_or_moe`] rather than
//! running a dense FFN — an oracle that skipped the expert half would make
//! every MoE parity comparison agree about the wrong answer.
//!
//! Both are transactional; see each function's contract and
//! [`transaction`] for the rewind that makes the decode one true.

use larql_inference::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend,
};
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::forward::hooks::LayerHook;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::kv_engine::EngineError;
use larql_inference::ModelWeights;
use ndarray::Array2;

use crate::cache::KvCache;
use crate::generation::last_row_as_2d;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

#[cfg(test)]
mod tests;
mod transaction;

use transaction::{cache_row_counts, decode_rewind_is_sound, rewind_cache};

/// Prefill phase as a reusable building block: runs a full forward over
/// `prompt_ids`, populates a fresh [`KvCache`] (bounded if `window` is
/// `Some`), and returns `(last_hidden_1xD, populated_cache)`.
///
/// This is the production K/V cache prefill loop, extracted so
/// `KvEngine::prefill` impls can call it directly, and the oracle the
/// dispatch ring is compared against — which is why it dispatches experts
/// through [`crate::engines::layer_ffn_or_moe`] rather than running a dense
/// FFN. An oracle that skipped the expert half would make every MoE parity
/// comparison agree about the wrong answer.
///
/// Transactional: the cache is built into a local and returned only on
/// success, so a refusal costs the caller nothing it already had.
///
/// The caller applies `final_norm + lm_head` to the returned hidden
/// state to get logits.
#[allow(clippy::too_many_arguments)]
pub fn kv_prefill_run(
    weights: larql_inference::WeightsView,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    window: Option<usize>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    hook: &mut dyn LayerHook,
) -> Result<(Array2<f32>, KvCache), EngineError> {
    if prompt_ids.is_empty() {
        return Err(EngineError::EmptyPrompt);
    }
    let num_layers = weights.num_layers;
    let mut cache = match window {
        Some(w) => KvCache::with_window(num_layers, w),
        None => KvCache::with_layers(num_layers),
    };

    let mut h = embed_tokens_pub(&weights, prompt_ids);
    // Per-Layer Embedding inputs for Gemma-4 archs. Returns empty Vec
    // for non-PLE archs (`ple_inputs.get(layer)` then yields `None` and
    // `apply_per_layer_embedding` is a no-op).
    let ple_inputs = precompute_per_layer_inputs(&weights, &h, prompt_ids);
    for layer in 0..num_layers {
        hook.on_pre_layer(layer, &h);

        let (mut h_post_attn, k_rope, v) =
            run_attention_with_kv_backend(weights, &h, layer, backend, None).ok_or_else(|| {
                EngineError::BackendFailure {
                    details: format!("attention returned None during prefill at layer {layer}"),
                }
            })?;
        cache.layers[layer] = Some((k_rope, v));
        cache.clip_layer(layer);

        hook.on_post_attention(layer, &mut h_post_attn);

        let mut h_out = crate::engines::layer_ffn_or_moe(
            weights.canonical(),
            &h_post_attn,
            layer,
            ffn,
            Some(ffn),
            ple_inputs.get(layer),
        )
        .map_err(EngineError::Execution)?;

        hook.on_post_layer(layer, &mut h_out);
        h = h_out;
    }
    cache.next_position = prompt_ids.len();

    Ok((last_row_as_2d(&h), cache))
}

/// Decode-step phase as a reusable building block: takes one new
/// `token_id`, runs the autoregressive attention against an existing
/// populated [`KvCache`], mutates the cache to append the new K/V (and
/// clip to window), and returns the new token's hidden state (shape
/// `[1, hidden_dim]`).
///
/// This is the production decode step extracted so `KvEngine::decode_step`
/// impls can call it directly, and — like [`kv_prefill_run`] — the oracle the
/// dispatch ring is compared against, so it dispatches experts.
///
/// **Transactional.** Each layer's attention appends the new token's K/V
/// before the FFN gets the chance to refuse, so a step that does not complete
/// truncates every layer back to the length it had on entry. When the cache is
/// windowed and already at its limit that truncation would be a lie —
/// append-then-evict leaves the row count unchanged while the oldest row is
/// gone — so the failure is reported as
/// [`EngineError::StateInvalidated`] instead, and the caller must rebuild the
/// cache rather than continue from it.
#[allow(clippy::too_many_arguments)]
pub fn kv_decode_step_run(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    cache: &mut KvCache,
    token_id: u32,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    hook: &mut dyn LayerHook,
) -> Result<Array2<f32>, EngineError> {
    let entry_rows = cache_row_counts(cache);
    let rewindable = decode_rewind_is_sound(cache, &entry_rows);
    match decode_step_appending(weights, ffn, cache, token_id, backend, hook) {
        Ok(hidden) => Ok(hidden),
        Err(failure) if rewindable => {
            rewind_cache(cache, &entry_rows);
            Err(failure)
        }
        Err(failure) => Err(failure.invalidating_engine_state()),
    }
}

/// The body of a decode step, which appends to `cache` as it goes.
#[allow(clippy::too_many_arguments)]
fn decode_step_appending(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    cache: &mut KvCache,
    token_id: u32,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    hook: &mut dyn LayerHook,
) -> Result<Array2<f32>, EngineError> {
    let num_layers = weights.num_layers;
    let h_new = embed_tokens_pub(weights, &[token_id]);
    let abs_position = cache.next_position;
    // PLE inputs are per-token. Recompute for this single-token decode
    // step rather than indexing a prefill-sized slab. Matches the
    // recipe used by `vindex::kquant_forward::cached` and the GPU
    // `layer_graph::generate` decode loop.
    let ple_inputs = precompute_per_layer_inputs(weights, &h_new, &[token_id]);
    let mut h_step = h_new;
    for layer in 0..num_layers {
        hook.on_pre_layer(layer, &h_step);

        let kv_entry = cache.layers[layer].as_ref();
        let (mut h_post_attn, new_kv) = run_attention_block_decode_step_backend(
            larql_inference::WeightsView::dense(weights),
            &h_step,
            layer,
            kv_entry,
            abs_position,
            backend,
        )
        .ok_or_else(|| EngineError::BackendFailure {
            details: format!("attention returned None during decode at layer {layer}"),
        })?;
        cache.layers[layer] = Some(new_kv);
        cache.clip_layer(layer);

        hook.on_post_attention(layer, &mut h_post_attn);

        let mut h_out = crate::engines::layer_ffn_or_moe(
            weights,
            &h_post_attn,
            layer,
            ffn,
            Some(ffn),
            ple_inputs.get(layer),
        )
        .map_err(EngineError::Execution)?;

        hook.on_post_layer(layer, &mut h_out);
        h_step = h_out;
    }
    cache.next_position += 1;
    Ok(h_step)
}
