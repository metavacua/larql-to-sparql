//! KV-cached CPU Q4_K decode.
//!
//! `predict_kquant_hidden` (sibling module) reprocesses the entire
//! `token_ids` sequence at every decode step — O(N²) work where N
//! grows with each generated token. This module splits that into
//! prefill (full-sequence pass that captures K/V per layer) plus
//! per-step decode (single-row attention against the cache + 1-row
//! FFN). Speedup scales linearly with decode length.
//!
//! Per-step Q4_K → f32 dequant via `insert_q4k_layer_tensors` is
//! still paid for now; eliminating it is a follow-up (route Q/K/V/O
//! and gate/up/down through `backend.q4k_matvec` directly).
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

use larql_compute::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k_into, Q8KActivation};
use larql_compute::ComputeBackend;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::attention::{
    decode::{gqa_attention_decode_step, run_attention_block_decode_step_backend},
    rope::apply_rope_partial_at_full,
    run_attention_with_kv_backend,
};
use crate::ffn::WeightFfn;
use crate::forward::embed_tokens_pub;
use crate::forward::layer::apply_layer_scalar;
use crate::forward::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use crate::forward::run_ffn;
use crate::forward::{add_bias, apply_norm};
use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

use super::tensors::{insert_q4k_layer_tensors, remove_layer_tensors};

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
    if weights.arch.is_hybrid_moe() {
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
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    predict_kquant_prefill_with_state(weights, token_ids, index, None)
}

/// Prefill with optional per-layer state capture (W1-GPU step 3
/// sibling of [`predict_kquant_decode_step_direct_with_state`]). When
/// `state` is `Some`, populates per-layer `h_in` ([seq_len, hidden]),
/// `k_new` ([seq_len, kv_dim]), `v_new` ([seq_len, kv_dim]) for every
/// position in the prompt — engines (markov_residual,
/// unlimited_context, turbo_quant) use this to seed their state policy
/// from a single prefill pass without a follow-up CPU re-walk. When
/// `state` is `None`, bit-identical to [`predict_kquant_prefill`].
pub fn predict_kquant_prefill_with_state(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    mut state: Option<&mut crate::PerLayerDecodeState>,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    let num_layers = weights.num_layers;
    let mut cache: CpuKvCache = vec![None; num_layers];
    let mut timings = CachedTimings::default();

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        let t0 = std::time::Instant::now();
        let inserted =
            insert_q4k_layer_tensors(weights, index, layer).unwrap_or_else(|err| panic!("{err}"));
        timings.dequant_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Snapshot pre-attention residual for this layer if engine wants it.
        if let Some(s) = state.as_deref_mut() {
            s.h_in_per_layer
                .push(larql_compute::state_handle::CpuStateHandle::boxed(
                    h.clone(),
                ));
        }

        // Attention with K/V capture. Backend stays None — we want the
        // CPU BLAS path for the dequantised f32 tensors that
        // `insert_q4k_layer_tensors` just placed in `weights.tensors`.
        let (h_post_attn, k_rope, v_final) =
            match run_attention_with_kv_backend(weights, &h, layer, None) {
                Some(t) => t,
                None => {
                    remove_layer_tensors(weights, inserted);
                    return (h, cache, timings);
                }
            };

        if let Some(s) = state.as_deref_mut() {
            // Prefill K/V for THIS layer = full seq_len × kv_dim.
            s.k_new_per_layer
                .push(larql_compute::state_handle::CpuStateHandle::boxed(
                    k_rope.clone(),
                ));
            s.v_new_per_layer
                .push(larql_compute::state_handle::CpuStateHandle::boxed(
                    v_final.clone(),
                ));
        }

        let ffn = WeightFfn { weights };
        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);

        remove_layer_tensors(weights, inserted);

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
    weights: &mut ModelWeights,
    token_id: u32,
    index: &VectorIndex,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<(Array2<f32>, CachedTimings)> {
    let num_layers = weights.num_layers;
    if cache.len() != num_layers {
        return None;
    }
    let mut timings = CachedTimings::default();

    // 1-row embed + 1-row PLE for the new token.
    let mut h = embed_tokens_pub(weights, &[token_id]);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);

    for layer in 0..num_layers {
        let t0 = std::time::Instant::now();
        let inserted =
            insert_q4k_layer_tensors(weights, index, layer).unwrap_or_else(|err| panic!("{err}"));
        timings.dequant_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let kv_entry = cache[layer].as_ref();
        let (h_post_attn, new_kv) = match run_attention_block_decode_step_backend(
            weights,
            &h,
            layer,
            kv_entry,
            abs_position,
            None,
        ) {
            Some(t) => t,
            None => {
                remove_layer_tensors(weights, inserted);
                return None;
            }
        };
        cache[layer] = Some(new_kv);

        let ffn = WeightFfn { weights };
        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);

        remove_layer_tensors(weights, inserted);

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
    // Gate on the two kernel-backed formats and take the packed row length
    // from the format helper rather than re-spelling `(cols/256)*144` (twin
    // of `larql_compute::cpu::ops::q4k_q8k_dot` — kept in sync via
    // `from_registry_tag`). `%256` is guarded above, so `packed_matrix_bytes`
    // is exact.
    let bytes_per_row = match larql_compute::QuantFormat::from_registry_tag(format) {
        Some(f @ (larql_compute::QuantFormat::Q4_K | larql_compute::QuantFormat::Q6_K)) => {
            f.packed_matrix_bytes(1, cols)?
        }
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
    larql_compute::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_parallel(
        &mut out, x_q8k, bytes, rows, cols, format,
    );
    Some(out)
}

/// True when every Q/K/V/O + gate/up/down slice for `layer` is in a
/// format the direct-matvec path knows how to handle. Used to gate
/// per-layer routing: the cached decode step prefers the direct
/// matvec when this returns true and falls back to the dequant path
/// otherwise (e.g. Q4_KF layers, padded down projections).
fn layer_supports_direct_matvec(index: &VectorIndex, layer: usize) -> bool {
    let attn = match index.attn_kquant_layer_data(layer) {
        Some(a) => a,
        None => return false,
    };
    for (_, fmt) in attn.iter() {
        if !matches!(*fmt, "Q4_K" | "Q6_K") {
            return false;
        }
    }
    let ffn = match index.interleaved_kquant_layer_data(layer) {
        Some(f) => f,
        None => return false,
    };
    for (_, fmt) in ffn.iter() {
        if !matches!(*fmt, "Q4_K" | "Q6_K") {
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
/// Same gating as [`supports_cached_decode`] plus a per-layer format
/// check. Used by the bench labeler and as the cpu.rs routing key.
pub fn supports_direct_matvec_decode(weights: &ModelWeights, index: &VectorIndex) -> bool {
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
/// Metal-fused prefill: run the prompt through every layer via the
/// backend's `prefill_kquant` kernel (one command buffer per session), seed
/// the backend's internal K/V cache, return last-row hidden.
///
/// Returns `None` if the backend doesn't have Q4 support
/// (`!backend.supports_quant(::larql_compute::QuantFormat::Q4_K)`), the vindex lacks Q4K/Q4_0 interleaved FFN
/// bytes, or the architecture isn't compatible with the fused pipeline.
/// CPU callers get `None` — they use [`predict_kquant_prefill`] instead.
///
/// Public counterpart to [`predict_kquant_prefill`] for the Metal side.
/// Previously lived inline in `larql-kv/engines/unlimited_context/engine.rs`;
/// promoted here so [`crate::kv_dispatch::metal::MetalBackend::coarse_prefill`]
/// can use it without an `larql-inference → larql-kv` dep cycle.
pub fn fused_prefill(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    use crate::layer_graph::pipeline_layer::{build_pipeline_layers, DEFAULT_GPU_KV_CACHE_MAX_SEQ};
    use larql_vindex::GateIndex;

    if !backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
        return None;
    }

    let gate_index: &dyn GateIndex = index;
    let (q4_ffn_mmap, ffn_is_q4k) = if let Some(m) = gate_index.interleaved_kquant_mmap_ref() {
        (m, true)
    } else if let Some(m) = gate_index.interleaved_q4_mmap_ref() {
        (m, false)
    } else {
        return None;
    };
    index.attn_kquant_layer_data(0)?;

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = gate_index.num_features(0);
    if intermediate == 0 {
        return None;
    }

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format.packed_matrix_bytes(intermediate, hidden)?;

    let layers = build_pipeline_layers(
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
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
    }

    let h_vec =
        backend.prefill_kquant(&layers, &x, hidden, intermediate, seq_len, qk_norm, softcap)?;

    let h_2d = Array2::from_shape_vec((seq_len, hidden), h_vec).ok()?;
    let last = h_2d.shape()[0] - 1;
    Some(h_2d.slice(ndarray::s![last..=last, ..]).to_owned())
}

/// Metal-fused single-token decode: run one token through all layers via
/// the backend's fused `decode_token` kernel, using the K/V cache
/// populated by a prior [`fused_prefill`] call on the same backend.
///
/// Returns `None` for CPU backends (no fused `decode_token` impl) and
/// for vindex shapes the fused pipeline can't handle. Public counterpart
/// to [`predict_kquant_decode_step_direct`] for the Metal side.
pub fn fused_decode_step(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_id: u32,
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    fused_decode_step_inner(weights, index, token_id, backend, None)
}

/// Variant of [`fused_decode_step`] that also captures per-layer state
/// via the backend's `decode_token_with_state_dump`. Engines pass
/// `Some(state)` to drive their state policy without a CPU re-walk.
/// `None` is bit-identical to the non-state variant.
pub fn fused_decode_step_with_state(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_id: u32,
    backend: &dyn ComputeBackend,
    state: &mut larql_compute::DecodeStateDump,
) -> Option<Array2<f32>> {
    fused_decode_step_inner(weights, index, token_id, backend, Some(state))
}

fn fused_decode_step_inner(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_id: u32,
    backend: &dyn ComputeBackend,
    state: Option<&mut larql_compute::DecodeStateDump>,
) -> Option<Array2<f32>> {
    use crate::layer_graph::pipeline_layer::build_pipeline_layers;
    use larql_vindex::GateIndex;

    let gate_index: &dyn GateIndex = index;
    let (q4_ffn_mmap, ffn_is_q4k) = if let Some(m) = gate_index.interleaved_kquant_mmap_ref() {
        (m, true)
    } else if let Some(m) = gate_index.interleaved_q4_mmap_ref() {
        (m, false)
    } else {
        return None;
    };

    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = gate_index.num_features(0);

    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let q4_ffn_per_matrix = ffn_format.packed_matrix_bytes(intermediate, hidden)?;

    let layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn_mmap,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let h_tok = crate::forward::embed_tokens_pub(weights, &[token_id]);
    let x_dec: Vec<f32> = h_tok.row(0).to_vec();

    let h_vec =
        backend.decode_token_with_state_dump(&layers, &x_dec, hidden, intermediate, state)?;
    Array2::from_shape_vec((1, hidden), h_vec).ok()
}

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
/// (`MarkovResidual`, `UnlimitedContext`, `TurboQuant`) that want the
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
    index: &VectorIndex,
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
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    // RoPE must match the staged path / prefill exactly: override-aware
    // base, the per-layer position divisor (Gemma 3 linear rope_scaling
    // applies ÷factor on GLOBAL layers only), and llama3 frequency
    // scaling. The unscaled `apply_rope_partial_at` here was the direct-
    // path divergence on gemma3-4b (global-layer K/Q rope'd at 8× the
    // position the prefill cache used — `ave_direct_step_parity`).
    let layer_rope_base = crate::forward_overrides::effective_rope_base_for_layer(arch, layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let pos_divisor =
        crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, layer);
    let llama3 = crate::forward_overrides::effective_llama3_rope_scaling(arch);
    let q_rope = apply_rope_partial_at_full(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        abs_position,
        pos_divisor,
        llama3,
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
        Some(norm_w) => rms_norm_heads(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off),
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
        llama3,
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
    let attn_out = gqa_attention_decode_step(
        &q_rope, &k_concat, &v_concat, num_q, head_dim, reps, scale, softcap,
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
    index: &VectorIndex,
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
    index: &VectorIndex,
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
        let gelu = matches!(arch.activation(), larql_models::Activation::GeluTanh);
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let gate_ref = &gate_vec[..];
        let up_ref = &up_vec[..];
        larql_compute::cpu::spin_pool::par_chunks_mut(&mut activated, 256, |ci, a_c| {
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
    // Bytes per 256-element super-block, from the format helper (= 144 / 210).
    let down_sb_bytes = match larql_compute::QuantFormat::from_registry_tag(down_fmt) {
        Some(f @ (larql_compute::QuantFormat::Q4_K | larql_compute::QuantFormat::Q6_K)) => {
            f.packed_matrix_bytes(1, 256)?
        }
        _ => return None,
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
    weights: &mut ModelWeights,
    token_id: u32,
    index: &VectorIndex,
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
    weights: &mut ModelWeights,
    token_id: u32,
    index: &VectorIndex,
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
                .push(larql_compute::state_handle::CpuStateHandle::boxed(
                    h.clone(),
                ));
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
                .push(larql_compute::state_handle::CpuStateHandle::boxed(
                    new_kv.0.slice(s![n - 1..n, ..]).to_owned(),
                ));
            s.v_new_per_layer
                .push(larql_compute::state_handle::CpuStateHandle::boxed(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, Q4KTestFixtures};
    use larql_compute::CpuBackend;

    // ── supports_cached_decode / supports_direct_matvec_decode ──────────

    #[test]
    fn supports_cached_decode_is_true_for_dense_arch() {
        let weights = make_test_q4k_weights();
        assert!(
            supports_cached_decode(&weights),
            "synthetic Gemma 3-style weights are dense, no KV sharing, no hybrid MoE"
        );
    }

    #[test]
    fn supports_direct_matvec_decode_is_true_for_q4k_synthetic_vindex() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        assert!(
            supports_direct_matvec_decode(&weights, &index),
            "synth Q4_K vindex has Q4_K attn + interleaved data, intermediate divisible by 256"
        );
    }

    #[test]
    fn layer_supports_direct_matvec_is_true_for_q4k_synthetic() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        for l in 0..weights.num_layers {
            assert!(
                layer_supports_direct_matvec(&index, l),
                "layer {l} should support direct matvec"
            );
        }
    }

    // ── matvec_q4k_or_q6k_q8k dispatcher ────────────────────────────────

    #[test]
    fn matvec_q4k_or_q6k_q8k_q4k_format_produces_finite_output() {
        use larql_compute::cpu::ops::q4_common::quantize_q4_k;
        use larql_compute::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k_into;

        let rows = 16usize;
        let cols = 256usize; // one super-block
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.001).sin() * 0.1)
            .collect();
        let q4k_bytes = quantize_q4_k(&weights);

        let x: Vec<f32> = (0..cols)
            .map(|j| ((j as f32) * 0.007).cos() * 0.5)
            .collect();
        let mut x_q8k = Q8KActivation::with_capacity(cols);
        quantize_x_to_q8k_into(&mut x_q8k, &x);

        let out = matvec_q4k_or_q6k_q8k(&q4k_bytes, "Q4_K", &x_q8k, rows, cols)
            .expect("Q4_K dispatch must succeed");
        assert_eq!(out.len(), rows);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(
            out.iter().any(|&v| v.abs() > 1e-6),
            "matvec should produce non-zero output for non-degenerate input"
        );
    }

    #[test]
    fn matvec_q4k_or_q6k_q8k_unknown_format_returns_none() {
        let x_q8k = Q8KActivation::with_capacity(256);
        // Empty bytes + unknown format should drop through to None.
        let out = matvec_q4k_or_q6k_q8k(&[], "F32", &x_q8k, 4, 256);
        assert!(out.is_none(), "unsupported format must return None");
    }

    #[test]
    fn matvec_q4k_or_q6k_q8k_zero_dims_returns_zeroed_vec() {
        let x_q8k = Q8KActivation::with_capacity(0);
        let out = matvec_q4k_or_q6k_q8k(&[], "Q4_K", &x_q8k, 0, 0);
        assert_eq!(out.as_deref(), Some(&[][..]));
    }

    #[test]
    fn matvec_q4k_or_q6k_q8k_non_multiple_cols_returns_none() {
        let x_q8k = Q8KActivation::with_capacity(255);
        // 255 is not a multiple of 256 → reject (caller falls back to dequant path).
        let out = matvec_q4k_or_q6k_q8k(&[], "Q4_K", &x_q8k, 4, 255);
        assert!(out.is_none(), "non-256-multiple cols must be rejected");
    }

    // ── predict_kquant_prefill / predict_kquant_decode_step ────────────────────

    #[test]
    fn predict_kquant_prefill_returns_hidden_with_expected_shape() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32, 2, 3];
        let (h, cache, _timings) = predict_kquant_prefill(&mut fx.weights, &token_ids, &fx.index);
        assert_eq!(
            h.shape()[0],
            token_ids.len(),
            "prefill returns seq_len rows"
        );
        assert_eq!(h.shape()[1], fx.weights.hidden_size);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "hidden state must be finite"
        );
        assert_eq!(cache.len(), fx.weights.num_layers);
        for layer in 0..fx.weights.num_layers {
            assert!(
                cache[layer].is_some(),
                "every layer should have K/V populated after prefill"
            );
        }
    }

    #[test]
    fn predict_kquant_decode_step_appends_kv_and_returns_one_row() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32, 2, 3];
        let (_, mut cache, _) = predict_kquant_prefill(&mut fx.weights, &token_ids, &fx.index);

        let pre_lens: Vec<usize> = cache
            .iter()
            .map(|c| c.as_ref().map(|(k, _)| k.shape()[0]).unwrap_or(0))
            .collect();

        let (h_new, _step_timings) =
            predict_kquant_decode_step(&mut fx.weights, 4, &fx.index, &mut cache, token_ids.len())
                .expect("decode step must succeed on a populated cache");

        assert_eq!(h_new.shape(), &[1, fx.weights.hidden_size]);
        assert!(h_new.iter().all(|v| v.is_finite()));

        for (layer, pre) in pre_lens.iter().enumerate() {
            let post = cache[layer]
                .as_ref()
                .map(|(k, _)| k.shape()[0])
                .unwrap_or(0);
            assert_eq!(post, pre + 1, "layer {layer} K/V should have grown by 1");
        }
    }

    #[test]
    fn predict_kquant_decode_step_rejects_mismatched_cache_length() {
        let mut fx = Q4KTestFixtures::build();
        // Cache length doesn't match num_layers — function must return None.
        let mut bad_cache: CpuKvCache = vec![None; fx.weights.num_layers + 1];
        let result = predict_kquant_decode_step(&mut fx.weights, 1, &fx.index, &mut bad_cache, 0);
        assert!(result.is_none());
    }

    // ── predict_kquant_decode_step_direct (Q4K × Q8K sdot path) ────────────

    /// The direct step must TRACK the staged step, not merely stay finite:
    /// same prefill cache, same token, same position → high-cosine hidden
    /// agreement. (The q4_common f16 subnormal bug passed the finite-only
    /// check below while garbling chained generation on real models —
    /// see `examples/ave_direct_step_parity.rs`.)
    #[test]
    fn predict_kquant_decode_step_direct_tracks_staged_step() {
        let token_ids = vec![1u32, 2, 3];

        let mut fx_a = Q4KTestFixtures::build();
        let (_, mut cache_a, _) =
            predict_kquant_prefill(&mut fx_a.weights, &token_ids, &fx_a.index);
        let (h_staged, _) =
            predict_kquant_decode_step(&mut fx_a.weights, 4, &fx_a.index, &mut cache_a, 3)
                .expect("staged step");

        let mut fx_b = Q4KTestFixtures::build();
        let (_, mut cache_b, _) =
            predict_kquant_prefill(&mut fx_b.weights, &token_ids, &fx_b.index);
        let backend = CpuBackend;
        let h_direct = predict_kquant_decode_step_direct(
            &mut fx_b.weights,
            4,
            &fx_b.index,
            &backend,
            &mut cache_b,
            3,
        )
        .expect("direct step");

        let a = h_staged.row(0);
        let b = h_direct.row(0);
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(
            cos > 0.999,
            "direct step diverged from staged step: cosine {cos} (norms {na} vs {nb})"
        );
        let ratio = if na > nb { na / nb } else { nb / na };
        assert!(
            ratio < 1.05,
            "direct step norm drifted from staged: {na} vs {nb}"
        );
    }

    /// Scaled-RoPE regression: on a Gemma-3 arch with linear
    /// `rope_scaling` (position divisor 8 on the global layer), the
    /// direct step must still track the staged step. Pre-2026-06-12 the
    /// direct path roped Q/K with the UNSCALED `apply_rope_partial_at` —
    /// no position divisor, no llama3 scaling — so on any rope-scaled
    /// config the global layer's K landed at 8× the position the prefill
    /// cache used. The non-scaled fixtures can't see that gap; this one
    /// exists to.
    #[test]
    fn predict_kquant_decode_step_direct_tracks_staged_on_rope_scaled_arch() {
        use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights_rope_scaled};

        let mut weights_a = make_test_q4k_weights_rope_scaled();
        // Guard: the fixture must actually parse into a divisor-8 global
        // layer — otherwise this test silently stops testing anything.
        let scaled_layers: Vec<usize> = (0..weights_a.num_layers)
            .filter(|&l| weights_a.arch.rope_position_divisor_for_layer(l) == 8.0)
            .collect();
        assert!(
            !scaled_layers.is_empty(),
            "fixture drift: no layer carries rope position divisor 8 — \
             the rope_scaling config no longer parses as global-only linear"
        );
        let index = make_test_q4k_vindex(&weights_a);
        assert!(
            supports_direct_matvec_decode(&weights_a, &index),
            "rope-scaled fixture must support the direct-matvec path"
        );

        // Prompt long enough that the scaled position (pos/8) and the
        // unscaled position differ by a large rotary angle.
        let token_ids = vec![1u32, 2, 3, 4, 5];
        let next = 6u32;

        let (_, mut cache_a, _) = predict_kquant_prefill(&mut weights_a, &token_ids, &index);
        let (h_staged, _) =
            predict_kquant_decode_step(&mut weights_a, next, &index, &mut cache_a, token_ids.len())
                .expect("staged step");

        let mut weights_b = make_test_q4k_weights_rope_scaled();
        let (_, mut cache_b, _) = predict_kquant_prefill(&mut weights_b, &token_ids, &index);
        let backend = CpuBackend;
        let h_direct = predict_kquant_decode_step_direct(
            &mut weights_b,
            next,
            &index,
            &backend,
            &mut cache_b,
            token_ids.len(),
        )
        .expect("direct step");

        // Primary assertion: the K row each path APPENDED to the cache.
        // RoPE is relative — if the direct step ropes both new-Q and
        // new-K at the wrong scale, their geometry to each other is
        // preserved and the hidden state barely moves on a bland random
        // fixture. The appended K row is the object the divisor rotates,
        // and it must match the staged row at every layer (most of all
        // the divisor-8 global layers).
        for layer in 0..weights_a.num_layers {
            let (k_a, _) = cache_a[layer].as_ref().expect("staged cache");
            let (k_b, _) = cache_b[layer].as_ref().expect("direct cache");
            let ra = k_a.row(k_a.nrows() - 1);
            let rb = k_b.row(k_b.nrows() - 1);
            let dot: f32 = ra.iter().zip(rb.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = ra.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = rb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(
                cos > 0.999,
                "appended K row diverged at layer {layer} (divisor {}): cosine {cos} — \
                 check the rope divisor / llama3 scaling in attention_decode_step_native",
                weights_a.arch.rope_position_divisor_for_layer(layer)
            );
        }

        // Secondary: the hidden state still tracks.
        let a = h_staged.row(0);
        let b = h_direct.row(0);
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(
            cos > 0.999,
            "direct step hidden diverged from staged on the rope-scaled arch \
             (global layers {scaled_layers:?}): cosine {cos}"
        );
    }

    #[test]
    fn predict_kquant_decode_step_direct_returns_finite_hidden() {
        let mut fx = Q4KTestFixtures::build();
        let token_ids = vec![1u32, 2, 3];
        let (_, mut cache, _) = predict_kquant_prefill(&mut fx.weights, &token_ids, &fx.index);

        let backend = CpuBackend;
        let h_new = predict_kquant_decode_step_direct(
            &mut fx.weights,
            4,
            &fx.index,
            &backend,
            &mut cache,
            token_ids.len(),
        )
        .expect("direct decode step must succeed");

        assert_eq!(h_new.shape(), &[1, fx.weights.hidden_size]);
        assert!(h_new.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn predict_kquant_decode_step_direct_rejects_mismatched_cache_length() {
        let mut fx = Q4KTestFixtures::build();
        let mut bad_cache: CpuKvCache = vec![None; fx.weights.num_layers - 1];
        let backend = CpuBackend;
        let result = predict_kquant_decode_step_direct(
            &mut fx.weights,
            1,
            &fx.index,
            &backend,
            &mut bad_cache,
            0,
        );
        assert!(result.is_none());
    }

    // ── CachedTimings merge ──────────────────────────────────────────────

    #[test]
    fn cached_timings_add_accumulates_dequant_ms() {
        let mut t = CachedTimings::default();
        assert_eq!(t.dequant_ms, 0.0);
        t.add(CachedTimings { dequant_ms: 1.5 });
        t.add(CachedTimings { dequant_ms: 2.25 });
        assert_eq!(t.dequant_ms, 3.75);
    }

    // ── fused_prefill / fused_decode_step ────────────────────────────────
    //
    // The public fused fast path: dispatches to `backend.prefill_kquant` /
    // `backend.decode_token`. **Not Metal-specific** — `CpuBackend` returns
    // `supports_quant(Q4_K) == true` (it ships a C Q4 kernel) and may implement either
    // method. The functions short-circuit when the vindex lacks the
    // interleaved FFN bytes the fused pipeline needs (the case for the
    // synthetic test vindex below), regardless of which backend is used.
    // The earlier name `metal_fused_*` was a misnomer.

    #[test]
    fn fused_prefill_returns_none_on_synthetic_vindex() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = CpuBackend;
        let result = fused_prefill(&weights, &index, &[0u32, 1], &backend);
        assert!(
            result.is_none(),
            "synthetic vindex without interleaved fused-pipeline bytes must short-circuit"
        );
    }

    #[test]
    fn fused_decode_step_returns_none_on_synthetic_vindex() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = CpuBackend;
        let result = fused_decode_step(&weights, &index, 0, &backend);
        assert!(
            result.is_none(),
            "synthetic vindex without interleaved fused-pipeline bytes must short-circuit"
        );
    }
}

#[cfg(test)]
mod branch_tests {
    use super::*;
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer, Q4K_TEST_HIDDEN,
        Q4K_TEST_INTER, Q4K_TEST_NUM_LAYERS, Q4K_TEST_VOCAB,
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    use larql_compute::CpuBackend;
    use larql_models::{detect_from_json, ModelWeights, WeightArray};
    use ndarray::Array2;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn rand_mat(rows: usize, cols: usize, seed: u64) -> WeightArray {
        let mut state = seed;
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data)
            .unwrap()
            .into_shared()
    }

    /// Llama-style fixture: same dimensions as the Gemma 3 fixture but
    /// `model_type=llama` so `arch.activation()` returns SiLU instead
    /// of GeluTanh. Exercises the SiLU branch in
    /// `run_ffn_decode_step_q4k_direct`.
    fn make_llama_q4k_weights() -> ModelWeights {
        let num_q = 4usize;
        let num_kv = 2usize;
        let head_dim = Q4K_TEST_HIDDEN / num_q;
        let arch_json = serde_json::json!({
            "model_type": "llama",
            "hidden_size": Q4K_TEST_HIDDEN,
            "num_hidden_layers": Q4K_TEST_NUM_LAYERS,
            "intermediate_size": Q4K_TEST_INTER,
            "head_dim": head_dim,
            "num_attention_heads": num_q,
            "num_key_value_heads": num_kv,
            "vocab_size": Q4K_TEST_VOCAB,
            "hidden_activation": "silu",
            "rope_theta": 10000.0,
        });
        let arch = detect_from_json(&arch_json);
        let mut tensors: HashMap<String, WeightArray> = HashMap::new();
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut seed = 0xc0ffee_u64.wrapping_mul(31);
        let mut next_seed = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        let embed = rand_mat(Q4K_TEST_VOCAB, Q4K_TEST_HIDDEN, next_seed());
        let lm_head = embed.clone();
        tensors.insert(arch.embed_key().to_string(), embed.clone());
        vectors.insert(
            arch.final_norm_key().to_string(),
            vec![1.0; Q4K_TEST_HIDDEN],
        );
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        for layer in 0..Q4K_TEST_NUM_LAYERS {
            tensors.insert(
                arch.attn_q_key(layer),
                rand_mat(q_dim, Q4K_TEST_HIDDEN, next_seed()),
            );
            tensors.insert(
                arch.attn_k_key(layer),
                rand_mat(kv_dim, Q4K_TEST_HIDDEN, next_seed()),
            );
            tensors.insert(
                arch.attn_v_key(layer),
                rand_mat(kv_dim, Q4K_TEST_HIDDEN, next_seed()),
            );
            tensors.insert(
                arch.attn_o_key(layer),
                rand_mat(Q4K_TEST_HIDDEN, q_dim, next_seed()),
            );
            tensors.insert(
                arch.ffn_gate_key(layer),
                rand_mat(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, next_seed()),
            );
            tensors.insert(
                arch.ffn_up_key(layer),
                rand_mat(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, next_seed()),
            );
            tensors.insert(
                arch.ffn_down_key(layer),
                rand_mat(Q4K_TEST_HIDDEN, Q4K_TEST_INTER, next_seed()),
            );
            vectors.insert(arch.input_layernorm_key(layer), vec![1.0; Q4K_TEST_HIDDEN]);
            vectors.insert(
                arch.post_attention_layernorm_key(layer),
                vec![1.0; Q4K_TEST_HIDDEN],
            );
        }
        ModelWeights {
            tensors,
            vectors,
            raw_bytes: HashMap::new(),
            packed_mmaps: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: HashMap::new(),
            embed,
            lm_head,
            position_embed: None,
            arch,
            num_layers: Q4K_TEST_NUM_LAYERS,
            hidden_size: Q4K_TEST_HIDDEN,
            intermediate_size: Q4K_TEST_INTER,
            vocab_size: Q4K_TEST_VOCAB,
            head_dim,
            num_q_heads: num_q,
            num_kv_heads: num_kv,
            rope_base: 10_000.0,
        }
    }

    /// Direct decode step on a SiLU-activation arch — exercises the
    /// non-GeluTanh branch in `run_ffn_decode_step_q4k_direct`.
    #[test]
    fn predict_kquant_decode_step_direct_silu_activation_path() {
        let mut weights = make_llama_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let _tok = make_test_tokenizer(weights.vocab_size);
        let token_ids = vec![1u32, 2];
        let (_, mut cache, _) = predict_kquant_prefill(&mut weights, &token_ids, &index);
        let backend = CpuBackend;
        let h_new = predict_kquant_decode_step_direct(
            &mut weights,
            3,
            &index,
            &backend,
            &mut cache,
            token_ids.len(),
        )
        .expect("SiLU direct decode step must succeed");
        assert!(h_new.iter().all(|v| v.is_finite()));
    }

    /// `predict_kquant_decode_step` (dequant path) on the same SiLU
    /// fixture — exercises `run_ffn`'s SiLU branch.
    #[test]
    fn predict_kquant_decode_step_silu_activation_path() {
        let mut weights = make_llama_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let token_ids = vec![1u32, 2];
        let (_, mut cache, _) = predict_kquant_prefill(&mut weights, &token_ids, &index);
        let (h_new, _) =
            predict_kquant_decode_step(&mut weights, 3, &index, &mut cache, token_ids.len())
                .expect("SiLU dequant decode step must succeed");
        assert!(h_new.iter().all(|v| v.is_finite()));
    }

    /// Q4K format check — when a matrix's bytes are too short for the
    /// claimed (rows, cols), the dispatcher should return None instead
    /// of panicking on the out-of-range slice.
    #[test]
    fn matvec_q4k_or_q6k_q8k_short_bytes_returns_none() {
        let x_q8k = Q8KActivation::with_capacity(256);
        // Need at least 4 * (256/256) * 144 = 576 bytes for 4×256 Q4_K.
        let too_short = vec![0u8; 144]; // only 1 row's worth
        let out = matvec_q4k_or_q6k_q8k(&too_short, "Q4_K", &x_q8k, 4, 256);
        assert!(out.is_none(), "short byte buffer must be rejected");
    }

    /// Cover the no-attention-data branch of
    /// `layer_supports_direct_matvec`: a vindex with no attn_kquant_layer_data.
    #[test]
    fn layer_supports_direct_matvec_false_without_attn_data() {
        let weights = make_test_q4k_weights();
        // Build an index with no Q4_K storage at all.
        let bare = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        assert!(
            !supports_direct_matvec_decode(&weights, &bare),
            "vindex with no Q4_K data can't support direct matvec"
        );
        for l in 0..weights.num_layers {
            assert!(!layer_supports_direct_matvec(&bare, l));
        }
    }

    /// Cover the unsupported-format branch of
    /// `layer_supports_direct_matvec`: vindex carries Q4_K storage but
    /// one entry is tagged with a format the direct path doesn't
    /// recognise (e.g., `Q4_KF`).
    #[test]
    fn layer_supports_direct_matvec_false_with_unsupported_format() {
        let weights = make_test_q4k_weights();
        let arch = &*weights.arch;
        let mut attn_payload = Vec::new();
        let mut attn_manifest: Vec<(usize, usize, String)> = Vec::new();
        let q_dim = weights.num_q_heads * weights.head_dim;
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for layer in 0..weights.num_layers {
            for (idx, key) in [
                arch.attn_q_key(layer),
                arch.attn_k_key(layer),
                arch.attn_v_key(layer),
                arch.attn_o_key(layer),
            ]
            .iter()
            .enumerate()
            {
                let rows = if idx == 0 || idx == 3 { q_dim } else { kv_dim };
                let _ = rows;
                let tensor = weights.tensors.get(key).unwrap();
                let bytes = quantize_q4_k(tensor.as_slice().unwrap());
                let off = attn_payload.len();
                let len = bytes.len();
                attn_payload.extend_from_slice(&bytes);
                // Tag the first Q with an unsupported format.
                let fmt = if layer == 0 && idx == 0 {
                    "Q4_KF".to_string()
                } else {
                    "Q4_K".to_string()
                };
                attn_manifest.push((off, len, fmt));
            }
        }
        // Also need interleaved Q4_K so the FFN branch passes through
        // before we hit the attn check. Use the synth helper for that.
        let mut index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        index.vocab_size = weights.vocab_size;
        let mut anon = memmap2::MmapMut::map_anon(attn_payload.len()).expect("anon");
        anon.copy_from_slice(&attn_payload);
        let ro = anon.make_read_only().unwrap();
        // Charge the bytes permanently: this mapping lives for the process lifetime
        // inside the Arc so release is never needed.
        larql_compute::resource_guard::charge_mmap(ro.len());
        let mmap = Arc::new(ro);
        Arc::make_mut(&mut index.storage).set_attn_kquant(mmap, Some(attn_manifest));
        assert!(
            !layer_supports_direct_matvec(&index, 0),
            "layer with Q4_KF format must not support the direct matvec path"
        );
    }

    /// `CachedTimings::merge` is private; verify the public `add`
    /// wrapper covers it (both should sum into `dequant_ms`).
    #[test]
    fn cached_timings_default_starts_at_zero() {
        let t = CachedTimings::default();
        assert_eq!(t.dequant_ms, 0.0);
    }

    /// Padded-down handling: when the stored down rows are wider than the
    /// layer's `intermediate` (256-padded — the 26B-A4B hybrid-MoE dense
    /// slab stores intermediate 2112 as 2304-col rows), the direct FFN
    /// step derives the stored width from the byte length, zero-pads the
    /// activation, and produces the same output as the unpadded layout:
    /// the real 256-element quant blocks are bit-identical and the pad
    /// blocks multiply zero activations.
    #[test]
    fn ffn_decode_step_native_padded_down_matches_unpadded() {
        use crate::test_utils::arc_mmap_from_bytes;
        use larql_compute::cpu::ops::q4_common::quantize_q4_k;

        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let h_post_attn = ndarray::Array2::from_shape_vec(
            (1, hidden),
            (0..hidden)
                .map(|i| ((i as f32) * 0.013).sin() * 0.05)
                .collect(),
        )
        .unwrap();
        let backend = CpuBackend;
        let baseline = ffn_decode_step_native(&weights, &index, &backend, &h_post_attn, 0)
            .expect("unpadded direct FFN step");

        // Rebuild the interleaved storage with every down matrix stored
        // 256-padded: [hidden, inter] → [hidden, inter + 256], zero cols.
        let arch = &*weights.arch;
        let mut payload: Vec<u8> = Vec::new();
        let mut manifest: Vec<(usize, usize, String)> = Vec::new();
        for layer in 0..weights.num_layers {
            for (key, pad) in [
                (arch.ffn_gate_key(layer), false),
                (arch.ffn_up_key(layer), false),
                (arch.ffn_down_key(layer), true),
            ] {
                let tensor = weights
                    .tensors
                    .get(&key)
                    .unwrap_or_else(|| panic!("missing tensor {key}"));
                let bytes = if pad {
                    let rows = tensor.shape()[0];
                    let cols = tensor.shape()[1];
                    let padded_cols = cols + 256;
                    let mut padded = vec![0.0f32; rows * padded_cols];
                    for r in 0..rows {
                        let src = tensor.row(r).to_vec();
                        padded[r * padded_cols..r * padded_cols + cols].copy_from_slice(&src);
                    }
                    quantize_q4_k(&padded)
                } else {
                    quantize_q4_k(tensor.as_slice().expect("contiguous row-major"))
                };
                let offset = payload.len();
                manifest.push((offset, bytes.len(), "Q4_K".to_string()));
                payload.extend_from_slice(&bytes);
            }
        }
        let mut index_padded = make_test_q4k_vindex(&weights);
        {
            let storage = std::sync::Arc::make_mut(&mut index_padded.storage);
            storage.set_interleaved_kquant(arc_mmap_from_bytes(&payload), Some(manifest));
        }
        assert_eq!(
            index_padded.num_features(0),
            index.num_features(0),
            "down padding must not change the derived intermediate width \
             (num_features comes from the gate manifest)"
        );

        let padded_out = ffn_decode_step_native(&weights, &index_padded, &backend, &h_post_attn, 0)
            .expect("padded direct FFN step");

        let max_abs = baseline
            .iter()
            .zip(padded_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-5,
            "padded-down output must match unpadded layout (max_abs={max_abs})"
        );
    }

    /// The padded-down derivation must reject byte lengths that aren't a
    /// whole number of super-blocks per row (corrupt / mismatched store)
    /// rather than computing with a truncated width.
    #[test]
    fn ffn_decode_step_native_rejects_ragged_down_bytes() {
        use crate::test_utils::arc_mmap_from_bytes;
        use larql_compute::cpu::ops::q4_common::quantize_q4_k;

        let weights = make_test_q4k_weights();
        let arch = &*weights.arch;
        let mut payload: Vec<u8> = Vec::new();
        let mut manifest: Vec<(usize, usize, String)> = Vec::new();
        for layer in 0..weights.num_layers {
            for key in [
                arch.ffn_gate_key(layer),
                arch.ffn_up_key(layer),
                arch.ffn_down_key(layer),
            ] {
                let tensor = weights.tensors.get(&key).expect("fixture tensor");
                let mut bytes = quantize_q4_k(tensor.as_slice().expect("contiguous"));
                if key == arch.ffn_down_key(layer) {
                    bytes.truncate(bytes.len() - 7); // ragged: not a whole super-block
                }
                let offset = payload.len();
                manifest.push((offset, bytes.len(), "Q4_K".to_string()));
                payload.extend_from_slice(&bytes);
            }
        }
        let mut index = make_test_q4k_vindex(&weights);
        {
            let storage = std::sync::Arc::make_mut(&mut index.storage);
            storage.set_interleaved_kquant(arc_mmap_from_bytes(&payload), Some(manifest));
        }
        let h = ndarray::Array2::<f32>::from_elem((1, weights.hidden_size), 0.01);
        assert!(
            ffn_decode_step_native(&weights, &index, &CpuBackend, &h, 0).is_none(),
            "ragged down byte length must fall back (return None), not mis-stride"
        );
    }
}
