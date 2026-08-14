//! KV-cached CPU Q4_K decode.
//!
//! `predict_kquant_hidden` (sibling module) reprocesses the entire
//! `token_ids` sequence at every decode step — O(N²) work where N
//! grows with each generated token. This module splits that into
//! prefill (full-sequence pass that captures K/V per layer) plus
//! per-step decode (single-row attention against the cache + 1-row
//! FFN). Speedup scales linearly with decode length.
//!
//! Prefill projects Q/K/V/O and gate/up/down straight from the vindex's
//! Q4_K/Q6_K bytes via the amortised q4k/q6k matmul — no per-layer f32
//! dequant — when the projection dims are 256-aligned (see
//! `predict_kquant_prefill_with_state`'s `use_q4k_attn` / `use_q4k_ffn`).
//! Per-step decode still dequantises via `insert_q4k_layer_tensors`
//! (a smaller follow-up).
//!
//! Scope: dense architectures only. Hybrid-MoE (Gemma 4 26B A4B)
//! and cross-layer KV sharing (Gemma 4 E2B) fall back to the slow
//! `predict_kquant_hidden` path — the caller decides via
//! [`supports_cached_decode`].

// `cache[layer]` indexing reads more naturally than the iterator
// equivalent and pairs cleanly with the explicit `layer` ID that's
// passed into `insert_q4k_layer_tensors` / `run_attention_block_*`.
// The `(Array2, (Array2, Array2))` return is the documented
// `(h_post_attn, (k_cache, v_cache))` shape used across the decode
// helpers; introducing a type alias would just spread the shape
// across two files.
#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use crate::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k_into, Q8KActivation};
use crate::ComputeBackend;
use larql_models::ModelWeights;
use ndarray::Array2;

use crate::attention::{
    decode::{gqa_attention_decode_step_windowed, run_attention_block_decode_step_backend},
    rope::apply_rope_partial_at_full,
    run_attention_with_kv_backend,
};
use crate::forward::embed_tokens_pub;
use crate::forward::layer::apply_layer_scalar;
use crate::forward::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use crate::forward::run_ffn;
use crate::forward::{add_bias, apply_norm};
use crate::residual::{rms_norm_heads_no_weight, rms_norm_qk_for_arch};

use super::tensors::{insert_q4k_attn_tensors, insert_q4k_layer_tensors, remove_layer_tensors};

#[cfg(test)]
mod tests;

/// Per-layer K/V captured during prefill. One entry per layer; matches
/// the [`crate::attention::decode::KvCache`] convention so future work
/// can swap in window clipping or surgery without churn here.
pub type CpuKvCache = Vec<Option<(Array2<f32>, Array2<f32>)>>;

/// Timing instrumentation for the cached CPU Q4K path. Times are
/// summed across all layers in a single call (prefill = one call;
/// decode = one call per generated token).
#[derive(Debug, Default, Clone, Copy)]
pub struct CachedTimings {
    pub dequant_ms: f64,
}

impl CachedTimings {
    fn merge(&mut self, other: CachedTimings) {
        self.dequant_ms += other.dequant_ms;
    }
}

/// True if the cached decode loop can handle this model. False for
/// hybrid-MoE (router/expert path runs through `run_moe_layer_cpu`)
/// and for architectures with cross-layer KV sharing (the decode-step
/// attention helper only knows the "this layer has its own K/V" case
/// today).
pub fn supports_cached_decode(weights: &ModelWeights) -> bool {
    // Pure MoE is exactly as unsupported here as hybrid — this loop's FFN
    // is dense-only.
    if weights.arch.is_moe() || weights.arch.is_hybrid_moe() {
        return false;
    }
    for layer in 0..weights.num_layers {
        if weights.arch.kv_shared_source_layer(layer).is_some() {
            return false;
        }
    }
    true
}

/// Prefill: run the full prompt through every layer once, capturing
/// each layer's post-RoPE K and final V into the returned cache.
/// Returns the `[seq_len, hidden]` hidden state and the populated
/// cache. Caller takes the last row for lm_head.
pub fn predict_kquant_prefill(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &dyn crate::KvIndex,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    predict_kquant_prefill_with_state(weights, token_ids, index, None)
}

/// Prefill with optional per-layer state capture (W1-GPU step 3
/// sibling of [`predict_kquant_decode_step_direct_with_state`]). When
/// `state` is `Some`, populates per-layer `h_in` ([seq_len, hidden]),
/// `k_new` ([seq_len, kv_dim]), `v_new` ([seq_len, kv_dim]) for every
/// position in the prompt — engines (markov_residual,
/// windowed_checkpoint, turbo_quant) use this to seed their state policy
/// from a single prefill pass without a follow-up CPU re-walk. When
/// `state` is `None`, bit-identical to [`predict_kquant_prefill`].
pub fn predict_kquant_prefill_with_state(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &dyn crate::KvIndex,
    mut state: Option<&mut crate::PerLayerDecodeState>,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    let num_layers = weights.num_layers;
    let mut cache: CpuKvCache = vec![None; num_layers];
    let mut timings = CachedTimings::default();
    // Forward-local dequant scratch — per-forward derived state; `weights`
    // stays immutable. Readers resolve via with_scratch (scratch ∪ canonical).
    let mut scratch = larql_models::DequantScratch::new();

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        // q4k-direct prefill: project straight from the vindex's Q4_K/Q6_K bytes
        // (no per-layer f32 dequant). The FFN gate/up/down (`use_q4k_ffn`) and
        // attention Q/K/V/O (`use_q4k_attn`) are gated independently — each needs
        // its interleaved bytes present and the relevant dims 256-aligned (the
        // matmul contraction is `hidden` for Q/K/V/gate/up and `q_dim` for O).
        // The FFN weights are ~4× the attention weights, so the FFN path is the
        // bulk of the saving; the attn path closes the rest.
        const BLK: usize = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
        let q_dim =
            weights.arch.num_q_heads_for_layer(layer) * weights.arch.head_dim_for_layer(layer);
        let use_q4k_ffn = weights.hidden_size.is_multiple_of(BLK)
            && index.interleaved_kquant_layer_data(layer).is_some();
        let use_q4k_attn = weights.hidden_size.is_multiple_of(BLK)
            && q_dim.is_multiple_of(BLK)
            && index.attn_kquant_layer_data(layer).is_some();

        let t0 = std::time::Instant::now();
        // Dequant only what the q4k-direct paths won't read straight from bytes.
        let inserted = match (use_q4k_attn, use_q4k_ffn) {
            (true, true) => Ok(Vec::new()),
            (false, true) => insert_q4k_attn_tensors(&mut scratch, weights, index, layer),
            _ => insert_q4k_layer_tensors(&mut scratch, weights, index, layer),
        }
        .unwrap_or_else(|err| panic!("{err}"));
        timings.dequant_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Snapshot pre-attention residual for this layer if engine wants it.
        if let Some(s) = state.as_deref_mut() {
            s.h_in_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(h.clone()));
        }

        // Attention with K/V capture. When `use_q4k_attn`, Q/K/V/O project
        // straight from the Q4_K/Q6_K bytes (empty scratch — norms still come
        // from canonical weights); else the dequantised f32 view path.
        let attn_index = if use_q4k_attn { Some(index) } else { None };
        let (h_post_attn, k_rope, v_final) = match run_attention_with_kv_backend(
            larql_models::WeightsView::with_scratch(weights, &scratch),
            &h,
            layer,
            None,
            attn_index,
        ) {
            Some(t) => t,
            None => {
                remove_layer_tensors(&mut scratch, inserted);
                return (h, cache, timings);
            }
        };

        if let Some(s) = state.as_deref_mut() {
            // Prefill K/V for THIS layer = full seq_len × kv_dim.
            s.k_new_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(k_rope.clone()));
            s.v_new_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(v_final.clone()));
        }

        let h_post_ffn = if use_q4k_ffn {
            let ffn = crate::ffn::Q4kMatmulFfn { weights, index };
            run_ffn(weights, &h_post_attn, layer, &ffn, false).0
        } else {
            let ffn = crate::ffn::ViewFfn {
                view: larql_models::WeightsView::with_scratch(weights, &scratch),
            };
            run_ffn(weights, &h_post_attn, layer, &ffn, false).0
        };
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);

        remove_layer_tensors(&mut scratch, inserted);

        cache[layer] = Some((k_rope, v_final));
        h = h_out;
    }

    (h, cache, timings)
}

/// Decode step: run a single new token through every layer using the
/// prefill cache. Each layer's cache entry is appended to in place.
/// Returns the new `[1, hidden]` hidden state for lm_head.
///
/// `abs_position` is the absolute RoPE position of the new token —
/// `prompt_len + steps_already_decoded`. The caller maintains this
/// counter (typical: `prompt_len + step_index` starting at 0).
pub fn predict_kquant_decode_step(
    weights: &ModelWeights,
    token_id: u32,
    index: &dyn crate::KvIndex,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<(Array2<f32>, CachedTimings)> {
    let num_layers = weights.num_layers;
    if cache.len() != num_layers {
        return None;
    }
    let mut timings = CachedTimings::default();
    let mut scratch = larql_models::DequantScratch::new();

    // 1-row embed + 1-row PLE for the new token.
    let mut h = embed_tokens_pub(weights, &[token_id]);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);

    for layer in 0..num_layers {
        let t0 = std::time::Instant::now();
        let inserted = insert_q4k_layer_tensors(&mut scratch, weights, index, layer)
            .unwrap_or_else(|err| panic!("{err}"));
        timings.dequant_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let kv_entry = cache[layer].as_ref();
        let (h_post_attn, new_kv) = match run_attention_block_decode_step_backend(
            larql_models::WeightsView::with_scratch(weights, &scratch),
            &h,
            layer,
            kv_entry,
            abs_position,
            None,
        ) {
            Some(t) => t,
            None => {
                remove_layer_tensors(&mut scratch, inserted);
                return None;
            }
        };
        cache[layer] = Some(new_kv);

        let ffn = crate::ffn::ViewFfn {
            view: larql_models::WeightsView::with_scratch(weights, &scratch),
        };
        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);

        remove_layer_tensors(&mut scratch, inserted);

        h = h_out;
    }

    Some((h, timings))
}

impl CachedTimings {
    /// Merge another timing block into self. Useful for accumulating
    /// per-step decode timings across a generation loop.
    pub fn add(&mut self, other: CachedTimings) {
        self.merge(other);
    }
}

// ── Phase 2: dequant-free decode step ───────────────────────────────────
//
// `predict_kquant_decode_step` (above) still pays the per-step Q4_K/Q6_K →
// f32 dequant cost via `insert_q4k_layer_tensors`. Profiling showed
// dequant is ~93% of CPU forward time even with the KV cache wired —
// gemm and attention are a small slice. This module routes Q/K/V/O and
// gate/up/down projections straight through `backend.quant_matvec`
// (CPU `q4k_matvec_into` / `q6k_matvec_into`), skipping the dequant
// staging entirely.

/// Format-aware Q*K × Q8_K matvec used by the production decode path.
/// Uses NEON `sdot` (Q4_K) or `vmlal_s8` (Q6_K) under the hood — ~2-3×
/// the f32-FMA throughput of `backend.quant_matvec`. Returns `None`
/// for any unsupported format (caller falls through to dequant).
fn matvec_q4k_or_q6k_q8k(
    bytes: &[u8],
    format: &str,
    x_q8k: &Q8KActivation,
    rows: usize,
    cols: usize,
) -> Option<Vec<f32>> {
    if rows == 0 || cols == 0 {
        return Some(vec![0.0f32; rows]);
    }
    const ELEMS_PER_BLOCK: usize = 256;
    if !cols.is_multiple_of(ELEMS_PER_BLOCK) {
        return None;
    }
    // Pre-flight length check only (the actual matvec recomputes this stride
    // internally). Gate on the kernel-backed formats via the `FormatRoute`
    // registry and take the packed row length from the format helper instead
    // of re-spelling `(cols/256)*144`.
    let bytes_per_row = match crate::QuantFormat::from_registry_tag(format) {
        Some(f) if f.route().q8k_matvec.is_some() => f.packed_matrix_bytes(1, cols)?,
        _ => return None,
    };
    if bytes.len() < rows * bytes_per_row {
        return None;
    }

    // `q4k_q8k_matvec_into` (larql-compute) is a single-threaded
    // per-row loop. Wrap it with `par_chunks_mut(CHUNK_ROWS)` here so
    // every Q4_K/Q6_K × Q8_K matvec on the decode path scales across
    // the 11 perf cores on M3 Max — matching the rayon strategy of
    // `q4k_matvec_into` in `q4_common.rs`. Without this, decode runs
    // single-threaded and the sdot path actually regresses vs the
    // (rayon-parallel) f32 path despite each row being faster.
    let mut out = vec![0.0f32; rows];
    crate::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_parallel(
        &mut out, x_q8k, bytes, rows, cols, format,
    );
    Some(out)
}

/// True when every Q/K/V/O + gate/up/down slice for `layer` is in a
/// format the direct-matvec path knows how to handle. Used to gate
/// per-layer routing: the cached decode step prefers the direct
/// matvec when this returns true and falls back to the dequant path
/// otherwise (e.g. Q4_KF layers, padded down projections).
fn layer_supports_direct_matvec(index: &dyn crate::KvIndex, layer: usize) -> bool {
    // "Direct-matvec-capable" = the tag resolves to a format with a Q8K
    // matvec kernel in the `FormatRoute` registry (Q4_K/Q6_K today).
    let has_q8k_kernel = |tag: &str| {
        crate::QuantFormat::from_registry_tag(tag).is_some_and(|f| f.route().q8k_matvec.is_some())
    };
    let attn = match index.attn_kquant_layer_data(layer) {
        Some(a) => a,
        None => return false,
    };
    for (_, fmt) in attn.iter() {
        if !has_q8k_kernel(fmt) {
            return false;
        }
    }
    let ffn = match index.interleaved_kquant_layer_data(layer) {
        Some(f) => f,
        None => return false,
    };
    for (_, fmt) in ffn.iter() {
        if !has_q8k_kernel(fmt) {
            return false;
        }
    }
    // The down projection in the FFN is sometimes stored with a padded
    // intermediate dim (rounded up to a 256-multiple). `q4k_matvec_into`
    // rejects non-multiple `cols`, which would silently zero the
    // output — refuse the direct path so the dequant fallback runs.
    let intermediate = index.num_features(layer);
    intermediate.is_multiple_of(larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
}

/// True when the whole model can run on the direct-matvec decode path.
/// Metal-fused multi-token prefill: run the prompt through all layers
/// via the backend's fused `prefill_kquant` kernel, populating the
/// backend's internal K/V cache for subsequent decode steps.
///
/// Returns `None` for CPU backends (no fused `prefill_kquant` impl) and
/// for vindex shapes the fused pipeline can't handle. Refactored to
/// take `&dyn KvIndex` (ADR-0022 Step 7).
pub fn fused_prefill(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_ids: &[u32],
    backend: &dyn crate::ComputeBackend,
) -> Option<Array2<f32>> {
    if !backend.supports_quant(crate::QuantFormat::Q4_K) {
        return None;
    }
    let (q4_ffn_mmap, ffn_is_q4k) = if let Some(m) = index.interleaved_kquant_mmap_ref() {
        (m, true)
    } else {
        (index.interleaved_q4_mmap_ref()?, false)
    };
    index.attn_kquant_layer_data(0)?;

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = index.num_features(0);
    if intermediate == 0 {
        return None;
    }

    let ffn_format = if ffn_is_q4k {
        crate::QuantFormat::Q4_K
    } else {
        crate::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format.packed_matrix_bytes(intermediate, hidden)?;

    let layers = crate::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let h_embed = crate::forward::embed_tokens_pub(weights, token_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();

    let seq_len = token_ids.len();
    let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm = arch.attn_q_norm_key(0).is_some();

    backend.reset_kv_cache();
    {
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(
            &kv_shapes,
            crate::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ,
        );
    }

    let h_vec =
        backend.prefill_kquant(&layers, &x, hidden, intermediate, seq_len, qk_norm, softcap)?;

    let h_2d = Array2::from_shape_vec((seq_len, hidden), h_vec).ok()?;
    let last = h_2d.shape()[0] - 1;
    Some(h_2d.slice(ndarray::s![last..=last, ..]).to_owned())
}

/// Metal-fused single-token decode: run one token through all layers
/// via the backend's fused `decode_token` kernel, using the K/V cache
/// populated by a prior [`fused_prefill`] call on the same backend.
pub fn fused_decode_step(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_id: u32,
    backend: &dyn crate::ComputeBackend,
) -> Option<Array2<f32>> {
    fused_decode_step_inner(
        weights,
        index,
        token_id,
        backend,
        None,
        crate::StateDumpMask::Full,
    )
}

/// Variant of [`fused_decode_step`] that also captures per-layer state
/// via the backend's `decode_token_with_state_dump`.
pub fn fused_decode_step_with_state(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_id: u32,
    backend: &dyn crate::ComputeBackend,
    state: &mut crate::DecodeStateDump,
) -> Option<Array2<f32>> {
    fused_decode_step_inner(
        weights,
        index,
        token_id,
        backend,
        Some(state),
        crate::StateDumpMask::Full,
    )
}

/// Mask-aware variant of [`fused_decode_step_with_state`]. Lets engines
/// that treat K/V as derivative state request
/// [`crate::StateDumpMask::HOnly`] to skip the K/V staging + readback.
pub fn fused_decode_step_with_state_masked(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_id: u32,
    backend: &dyn crate::ComputeBackend,
    state: &mut crate::DecodeStateDump,
    mask: crate::StateDumpMask,
) -> Option<Array2<f32>> {
    fused_decode_step_inner(weights, index, token_id, backend, Some(state), mask)
}

fn fused_decode_step_inner(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    token_id: u32,
    backend: &dyn crate::ComputeBackend,
    state: Option<&mut crate::DecodeStateDump>,
    mask: crate::StateDumpMask,
) -> Option<Array2<f32>> {
    let (q4_ffn_mmap, ffn_is_q4k) = if let Some(m) = index.interleaved_kquant_mmap_ref() {
        (m, true)
    } else {
        (index.interleaved_q4_mmap_ref()?, false)
    };

    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = index.num_features(0);

    let ffn_format = if ffn_is_q4k {
        crate::QuantFormat::Q4_K
    } else {
        crate::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format.packed_matrix_bytes(intermediate, hidden)?;

    let layers = crate::pipeline_layer::build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let h_tok = crate::forward::embed_tokens_pub(weights, &[token_id]);
    let x_dec: Vec<f32> = h_tok.row(0).to_vec();

    let h_vec = backend.decode_token_with_state_dump_masked(
        &layers,
        &x_dec,
        hidden,
        intermediate,
        state,
        mask,
    )?;
    Array2::from_shape_vec((1, hidden), h_vec).ok()
}

/// Same gating as [`supports_cached_decode`] plus a per-layer format
/// check. Used by the bench labeler and as the cpu.rs routing key.
pub fn supports_direct_matvec_decode(weights: &ModelWeights, index: &dyn crate::KvIndex) -> bool {
    if !supports_cached_decode(weights) {
        return false;
    }
    for layer in 0..weights.num_layers {
        if !layer_supports_direct_matvec(index, layer) {
            return false;
        }
    }
    true
}

fn vec_to_2d_row(v: Vec<f32>) -> Array2<f32> {
    let n = v.len();
    Array2::from_shape_vec((1, n), v).expect("matvec output shape")
}

/// One-row attention block using direct Q4_K/Q6_K matvec on the
/// quantised attention slices. Mirrors
/// [`crate::attention::decode::run_attention_block_decode_step_backend`]
/// but reads weights from `index.attn_kquant_layer_data(layer)` instead of
/// dequantised f32 in `weights.tensors`.
#[allow(clippy::too_many_arguments)]
/// Production-path attention decode step reading **quantised** weights
/// from the vindex (not f32 dequantised tensors). Same input/output
/// shape as
/// [`crate::attention::run_attention_block_decode_step_backend`], but
/// reads `index.attn_kquant_layer_data(layer)` directly and dispatches
/// the Q/K/V/O projections to the backend's native quantised matvec
/// (today Q4K / Q4_KF / Q6K via `q4k_matvec_q8_input`). Extending to
/// new quantised formats is internal to this function — the public
/// signature stays format-agnostic.
///
/// Used by `StandardEngine`'s coarse path and by research engines
/// (`MarkovResidual`, `WindowedCheckpoint`, `TurboQuant`) that want the
/// production decode kernel without inheriting the per-layer dispatch
/// trait's cached-K/V shape.
///
/// `h_new` must be a single-row residual (1 × hidden). Multi-row
/// prefill is handled by `predict_kquant_prefill` (separate shape; the
/// `q4k_` in that name is pre-existing debt — see ROADMAP U8/U9 for
/// the broader quant-agnostic rename of the kquant_forward module).
///
/// Returns `None` if the layer has no quantised attention data in the
/// index or if the backend's matvec for the format is unavailable.
pub fn attention_decode_step_native(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    // Kept on the helper signature for parity with the outer
    // `predict_kquant_decode_step_direct` API and any future asm dispatch
    // that wants runtime feature detection.
    _backend: &dyn ComputeBackend,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&(Array2<f32>, Array2<f32>)>,
    abs_position: usize,
) -> Option<(Array2<f32>, (Array2<f32>, Array2<f32>))> {
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let norm_offset = arch.norm_weight_offset();

    let h_norm = apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );
    let h_norm_row: &[f32] = h_norm.row(0).to_slice().or_else(|| h_norm.as_slice())?;

    let attn = index.attn_kquant_layer_data(layer)?;
    let (q_bytes, q_fmt) = attn[0];
    let (k_bytes, k_fmt) = attn[1];
    let (v_bytes, v_fmt) = attn[2];
    let (o_bytes, o_fmt) = attn[3];

    // Q8_K-quantise `h_norm` once and reuse for Q / K / V projections.
    // sdot int8 dot is ~2-3× the f32 FMA throughput of the
    // `q4k_matvec_into` path; the quantisation step itself is O(hidden)
    // and amortises across the three projections (and O after attn).
    let mut h_norm_q8k = Q8KActivation::with_capacity(hidden);
    quantize_x_to_q8k_into(&mut h_norm_q8k, h_norm_row);

    let q_vec = matvec_q4k_or_q6k_q8k(q_bytes, q_fmt, &h_norm_q8k, q_dim, hidden)?;
    let mut q_full = vec_to_2d_row(q_vec);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_qk_for_arch(&q_full, norm_w, num_q, head_dim, qk_norm_off, arch),
        None => q_full,
    };
    // RoPE must match the staged path / prefill exactly: override-aware
    // base, the per-layer position divisor (Gemma 3 linear rope_scaling
    // applies ÷factor on GLOBAL layers only), and llama3 frequency
    // scaling. The unscaled `apply_rope_partial_at` here was the direct-
    // path divergence on gemma3-4b (global-layer K/Q rope'd at 8× the
    // position the prefill cache used).
    let layer_rope_base = crate::forward_overrides::effective_rope_base_for_layer(arch, layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let pos_divisor =
        crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, layer);
    let rope_scaling = crate::forward_overrides::effective_rope_freq_scaling(arch);
    let q_rope = apply_rope_partial_at_full(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        abs_position,
        pos_divisor,
        rope_scaling,
    );

    let k_vec = matvec_q4k_or_q6k_q8k(k_bytes, k_fmt, &h_norm_q8k, kv_dim, hidden)?;
    let v_vec = matvec_q4k_or_q6k_q8k(v_bytes, v_fmt, &h_norm_q8k, kv_dim, hidden)?;
    let mut k_full_new = vec_to_2d_row(k_vec);
    let mut v_full_new = vec_to_2d_row(v_vec);
    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full_new, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full_new, bias);
    }
    if arch.has_v_norm() {
        v_full_new = rms_norm_heads_no_weight(&v_full_new, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => {
            rms_norm_qk_for_arch(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off, arch)
        }
        None => k_full_new,
    };
    let k_new_rope = apply_rope_partial_at_full(
        &k_normed,
        num_kv,
        head_dim,
        layer_rope_base,
        rotary_frac,
        abs_position,
        pos_divisor,
        rope_scaling,
    );

    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
            let total = k_cached.shape()[0] + 1;
            let mut k_out = Array2::<f32>::zeros((total, kv_dim));
            let mut v_out = Array2::<f32>::zeros((total, kv_dim));
            k_out
                .slice_mut(ndarray::s![..k_cached.shape()[0], ..])
                .assign(k_cached);
            v_out
                .slice_mut(ndarray::s![..v_cached.shape()[0], ..])
                .assign(v_cached);
            k_out
                .slice_mut(ndarray::s![k_cached.shape()[0].., ..])
                .assign(&k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    // Per-layer sliding window, same shared rule the Metal spec uses.
    // This is the CPU coarse Q4K decode — the path `standard` takes on
    // CpuBackend — so without it a Gemma-class model attends full history
    // here while Metal masks.
    let window = crate::forward_overrides::effective_attention_window_for_layer(arch, layer);
    let attn_out = gqa_attention_decode_step_windowed(
        &q_rope,
        &k_concat,
        &v_concat,
        num_q,
        head_dim,
        reps,
        scale,
        softcap,
        crate::attention::sinks::resolve(
            arch.attn_sinks_key(layer),
            &weights.vectors,
            num_q,
            layer,
        ),
        window,
    );
    let attn_out_row: &[f32] = attn_out.row(0).to_slice().or_else(|| attn_out.as_slice())?;

    // Re-quantise the attention output for the O projection. Different
    // input from Q/K/V (attn_out vs h_norm), so we need a fresh Q8_K.
    let mut attn_out_q8k = Q8KActivation::with_capacity(q_dim);
    quantize_x_to_q8k_into(&mut attn_out_q8k, attn_out_row);
    let o_vec = matvec_q4k_or_q6k_q8k(o_bytes, o_fmt, &attn_out_q8k, hidden, q_dim)?;
    let mut attn_projected = vec_to_2d_row(o_vec);
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = apply_norm(
            weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h_new + &(&normed * res_mult)
        } else {
            h_new + &normed
        }
    } else if res_mult != 1.0 {
        h_new + &(&attn_projected * res_mult)
    } else {
        h_new + &attn_projected
    };

    Some((h_post_attn, (k_concat, v_concat)))
}

/// One-row gated FFN block using direct native-quantised matvec on
/// the vindex's compact bytes (Q4K / Q6K today). Mirrors
/// [`crate::ffn::weight::dense_ffn_forward_backend`] but reads gate/up/
/// down from the vindex slices and avoids the f32 staging — same
/// production path that powers `larql run` / `larql bench --cpu` at
/// ~24 tok/s on Gemma 3 4B Q4K (M3 Max, 8 threads).
///
/// Returns `None` if the vindex layer lacks compact FFN bytes or the
/// architecture isn't supported by the direct-matvec path. Engines
/// that get `None` fall back to whichever `FfnBackend` they have.
///
/// `h_post_attn` must be a single-row residual (1 × hidden). Public
/// counterpart to [`attention_decode_step_native`] for the FFN side.
pub fn ffn_decode_step_native(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    backend: &dyn ComputeBackend,
    h_post_attn: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    run_ffn_decode_step_q4k_direct(weights, index, backend, h_post_attn, layer)
}

/// One-row gated FFN block using direct Q4_K/Q6_K matvec. Mirrors
/// [`crate::ffn::weight::dense_ffn_forward_backend`] but reads gate/up/
/// down from the vindex slices and avoids the f32 staging.
fn run_ffn_decode_step_q4k_direct(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    _backend: &dyn ComputeBackend,
    h_post_attn: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let intermediate = index.num_features(layer);
    let norm_offset = arch.norm_weight_offset();

    // Pre-FFN norm: same selection logic as `run_ffn` — when the arch
    // uses post_norms, the pre-FFN key is `pre_feedforward_layernorm`;
    // otherwise it reuses `post_attention_layernorm` as the FFN input
    // norm. Falls back to weightless RMS when no key is set.
    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_in = match pre_ffn_key {
        Some(key) => apply_norm(weights, h_post_attn, &key, norm_offset),
        None => crate::residual::rms_norm(h_post_attn, None, norm_offset),
    };
    let h_in_row: &[f32] = h_in.row(0).to_slice().or_else(|| h_in.as_slice())?;

    let ffn = index.interleaved_kquant_layer_data(layer)?;
    let (gate_bytes, gate_fmt) = ffn[0];
    let (up_bytes, up_fmt) = ffn[1];
    let (down_bytes, down_fmt) = ffn[2];

    // Only Gated FFNs reach this path today (it's what predict_kquant_hidden
    // currently dequantises). Non-gated archs route through the dequant
    // fallback via the per-layer gate at the caller.
    if arch.ffn_type() != larql_models::FfnType::Gated {
        return None;
    }

    // Q8_K-quantise `h_in` once and feed it to both gate and up via the
    // sdot-based fused matvec. This is the int8-dot Q4_K × Q8_K path
    // that closes the bandwidth gap to llama.cpp on M3 Max.
    let mut h_in_q8k = Q8KActivation::with_capacity(hidden);
    quantize_x_to_q8k_into(&mut h_in_q8k, h_in_row);

    // Two separate matvecs, each rayon-parallel inside
    // `matvec_q4k_or_q6k_q8k`. The "fused gate+up" variant in
    // `larql-compute` (`q4k_q8k_gate_up_into`) is single-threaded;
    // the input vector (10 KB) stays in L1 across two sequential
    // calls anyway, so we don't need explicit fusion to keep `x`
    // hot. Splitting lets both matvecs run row-parallel.
    let gate_vec = matvec_q4k_or_q6k_q8k(gate_bytes, gate_fmt, &h_in_q8k, intermediate, hidden)?;
    let up_vec = matvec_q4k_or_q6k_q8k(up_bytes, up_fmt, &h_in_q8k, intermediate, hidden)?;

    // Element-wise activation: activation(gate) * up. Rayon-chunked — the
    // per-element math (libm tanh/exp included) is unchanged, so the output
    // is bit-identical to the serial loop; the decode sample showed this
    // scalar pass serial on the main thread while the workers slept.
    let mut activated = vec![0.0f32; intermediate];
    {
        let gelu = arch.activation().uses_gelu_tanh_gate_up();
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let gate_ref = &gate_vec[..];
        let up_ref = &up_vec[..];
        crate::cpu::spin_pool::par_chunks_mut(&mut activated, 256, |ci, a_c| {
            let start = ci * 256;
            let g_c = &gate_ref[start..start + a_c.len()];
            let u_c = &up_ref[start..start + a_c.len()];
            if gelu {
                for ((a, &x), &u) in a_c.iter_mut().zip(g_c.iter()).zip(u_c.iter()) {
                    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                    *a = 0.5 * x * (1.0 + inner.tanh()) * u;
                }
            } else {
                // SiLU = x * sigmoid(x). Same shape as dense_ffn_forward_backend.
                for ((a, &x), &u) in a_c.iter_mut().zip(g_c.iter()).zip(u_c.iter()) {
                    let sig = 1.0 / (1.0 + (-x).exp());
                    *a = x * sig * u;
                }
            }
        });
    }

    // down projection: out = activated @ W_down.T → [hidden].
    // Re-quantise the post-activation vector (`intermediate`-wide) for
    // the down matvec — different input from gate/up.
    //
    // The stored down row width may be PADDED up to a 256-multiple when
    // `intermediate` isn't one (e.g. the 26B-A4B hybrid-MoE dense slab:
    // intermediate 2112 stored as 2304-col Q6_K rows). Derive the stored
    // width from the byte length and zero-pad the activation to match —
    // pad columns multiply zero activations, so the result is exact.
    // (Twin of the same handling in larql-inference's cached.rs — keep in
    // lockstep, see the consolidation hazard in q4k-direct-attention.md.)
    let down_sb_bytes = match crate::QuantFormat::from_registry_tag(down_fmt)
        .filter(|f| f.route().q8k_matvec.is_some())
        .and_then(|f| f.packed_block_layout())
    {
        Some((_, block_bytes)) => block_bytes,
        None => return None,
    };
    let down_bytes_per_row = down_bytes.len() / hidden;
    if down_bytes_per_row == 0 || !down_bytes_per_row.is_multiple_of(down_sb_bytes) {
        return None;
    }
    let stored_cols =
        down_bytes_per_row / down_sb_bytes * larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    if stored_cols < intermediate {
        return None;
    }
    let activated_padded: Vec<f32>;
    let act_slice: &[f32] = if stored_cols != intermediate {
        let mut p = vec![0.0f32; stored_cols];
        p[..intermediate].copy_from_slice(&activated);
        activated_padded = p;
        &activated_padded
    } else {
        &activated
    };
    let mut activated_q8k = Q8KActivation::with_capacity(stored_cols);
    quantize_x_to_q8k_into(&mut activated_q8k, act_slice);
    let down_vec =
        matvec_q4k_or_q6k_q8k(down_bytes, down_fmt, &activated_q8k, hidden, stored_cols)?;
    let mut out = vec_to_2d_row(down_vec);
    if let Some(bias) = arch
        .ffn_down_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut out, bias);
    }

    // Post-FFN residual + optional post-FFN layernorm. Same selection
    // logic as `run_ffn`: only fire when has_post_norms() AND the arch
    // exposes a post-FFN norm key.
    let res_mult = arch.residual_multiplier();
    let h_post_ffn = if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => apply_norm(weights, &out, &key, norm_offset),
            None => crate::residual::rms_norm(&out, None, norm_offset),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&out * res_mult)
    } else {
        h_post_attn + &out
    };

    Some(h_post_ffn)
}

/// Dequant-free decode step. Same shape contract as
/// [`predict_kquant_decode_step`] but routes every projection through
/// `backend.quant_matvec` instead of the per-layer
/// `insert_q4k_layer_tensors` → dense f32 staging dance. Returns `None`
/// if any layer has a format the direct-matvec path doesn't handle
/// (caller falls back to [`predict_kquant_decode_step`]).
pub fn predict_kquant_decode_step_direct(
    weights: &ModelWeights,
    token_id: u32,
    index: &dyn crate::KvIndex,
    backend: &dyn ComputeBackend,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<Array2<f32>> {
    predict_kquant_decode_step_direct_with_state(
        weights,
        token_id,
        index,
        backend,
        cache,
        abs_position,
        None,
    )
}

/// Decode step with optional per-layer state capture (`Some(state)`
/// populates `h_in` / `k_new` / `v_new` per layer at near-zero cost
/// since this CPU path already walks the layers serially). Engines
/// that need per-layer state — `markov_residual` for residual storage,
/// `markov_residual_codec` ditto, `turbo_quant` for per-layer K/V
/// compression — call through here via `KvDispatch::
/// coarse_decode_step_with_state`. When `state` is `None` this is
/// bit-identical to [`predict_kquant_decode_step_direct`].
pub fn predict_kquant_decode_step_direct_with_state(
    weights: &ModelWeights,
    token_id: u32,
    index: &dyn crate::KvIndex,
    backend: &dyn ComputeBackend,
    cache: &mut CpuKvCache,
    abs_position: usize,
    mut state: Option<&mut crate::PerLayerDecodeState>,
) -> Option<Array2<f32>> {
    use ndarray::s;
    let num_layers = weights.num_layers;
    if cache.len() != num_layers {
        return None;
    }

    let mut h = embed_tokens_pub(weights, &[token_id]);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);

    for layer in 0..num_layers {
        if let Some(s) = state.as_deref_mut() {
            s.h_in_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(h.clone()));
        }
        let kv_entry = cache[layer].as_ref();
        let (h_post_attn, new_kv) = attention_decode_step_native(
            weights,
            index,
            backend,
            &h,
            layer,
            kv_entry,
            abs_position,
        )?;
        if let Some(s) = state.as_deref_mut() {
            // new_kv is the full prior+new K/V; the new row is the
            // last row. Engines that cache per-layer K/V (markov_rs
            // hot_kv, turbo_quant compressed) consume this row.
            let n = new_kv.0.shape()[0];
            s.k_new_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(
                    new_kv.0.slice(s![n - 1..n, ..]).to_owned(),
                ));
            s.v_new_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(
                    new_kv.1.slice(s![n - 1..n, ..]).to_owned(),
                ));
        }
        cache[layer] = Some(new_kv);

        let h_post_ffn =
            run_ffn_decode_step_q4k_direct(weights, index, backend, &h_post_attn, layer)?;
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);
        h = h_out;
    }

    Some(h)
}
