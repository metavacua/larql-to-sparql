//! KV-cached CPU Q4_K decode — the `VectorIndex`-typed adapter over the
//! substrate forward in [`larql_compute::kquant_forward`].
//!
//! Every function here coerces `&VectorIndex` to `&dyn larql_compute::KvIndex`
//! and delegates: larql-compute owns the single implementation of the Q4_K CPU
//! prefill/decode (ADR-0022), including the q4k-direct FFN prefill. This module
//! exists only to keep a stable, `VectorIndex`-typed API and the local
//! `CachedTimings` type for callers that hold a concrete `VectorIndex`.
//!
//! The unique, non-delegated Q4_K paths — `predict_kquant_hidden`, the OV/RD
//! interventions, Metal, MoE, and streaming generation — live in the sibling
//! modules of `vindex::kquant_forward`, not here.

#![allow(clippy::type_complexity)]

use larql_compute::ComputeBackend;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

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
    larql_compute::kquant_forward::supports_cached_decode(weights)
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
/// windowed_checkpoint, turbo_quant) use this to seed their state policy
/// from a single prefill pass without a follow-up CPU re-walk. When
/// `state` is `None`, bit-identical to [`predict_kquant_prefill`].
pub fn predict_kquant_prefill_with_state(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    state: Option<&mut crate::PerLayerDecodeState>,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    // Delegate to the substrate copy in larql-compute — the single source of
    // truth for the Q4_K CPU forward (ADR-0022). `VectorIndex` satisfies
    // `larql_compute::KvIndex`; the only impedance is `CachedTimings`, a
    // same-shape struct defined in each crate.
    let (h, cache, timings) = larql_compute::kquant_forward::predict_kquant_prefill_with_state(
        weights,
        token_ids,
        index as &dyn larql_compute::KvIndex,
        state,
    );
    (
        h,
        cache,
        CachedTimings {
            dequant_ms: timings.dequant_ms,
        },
    )
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
    index: &VectorIndex,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<(Array2<f32>, CachedTimings)> {
    // Delegate to the substrate copy in larql-compute (ADR-0022); bridge the
    // per-crate `CachedTimings`.
    larql_compute::kquant_forward::predict_kquant_decode_step(
        weights,
        token_id,
        index as &dyn larql_compute::KvIndex,
        cache,
        abs_position,
    )
    .map(|(h, timings)| {
        (
            h,
            CachedTimings {
                dequant_ms: timings.dequant_ms,
            },
        )
    })
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

/// True when the whole model can run on the direct-matvec decode path.
/// Same gating as [`supports_cached_decode`] plus a per-layer format
/// check. Used by the bench labeler and as the cpu.rs routing key.
pub fn supports_direct_matvec_decode(weights: &ModelWeights, index: &VectorIndex) -> bool {
    larql_compute::kquant_forward::supports_direct_matvec_decode(
        weights,
        index as &dyn larql_compute::KvIndex,
    )
}

/// Fused Q4_K prefill via the compute backend (Metal fast path).
/// Delegates to the larql-compute substrate copy (ADR-0022).
pub fn fused_prefill(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    larql_compute::kquant_forward::fused_prefill(
        weights,
        index as &dyn larql_compute::KvIndex,
        token_ids,
        backend,
    )
}

/// Fused Q4_K decode step via the compute backend. Delegates to larql-compute.
pub fn fused_decode_step(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_id: u32,
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    larql_compute::kquant_forward::fused_decode_step(
        weights,
        index as &dyn larql_compute::KvIndex,
        token_id,
        backend,
    )
}

/// Fused Q4_K decode step with a per-layer state dump. Delegates to larql-compute.
pub fn fused_decode_step_with_state(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_id: u32,
    backend: &dyn ComputeBackend,
    state: &mut larql_compute::DecodeStateDump,
) -> Option<Array2<f32>> {
    larql_compute::kquant_forward::fused_decode_step_with_state(
        weights,
        index as &dyn larql_compute::KvIndex,
        token_id,
        backend,
        state,
    )
}

/// Dequant-free attention decode step (Q4_K/Q6_K x Q8_K direct matvec).
/// Delegates to the larql-compute substrate copy (ADR-0022).
#[allow(clippy::type_complexity)]
pub fn attention_decode_step_native(
    weights: &ModelWeights,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&(Array2<f32>, Array2<f32>)>,
    abs_position: usize,
) -> Option<(Array2<f32>, (Array2<f32>, Array2<f32>))> {
    larql_compute::kquant_forward::attention_decode_step_native(
        weights,
        index as &dyn larql_compute::KvIndex,
        backend,
        h_new,
        layer,
        kv_entry,
        abs_position,
    )
}

/// Dequant-free FFN decode step (gate/up/down via direct Q4_K matvec).
/// Delegates to the larql-compute substrate copy (ADR-0022).
pub fn ffn_decode_step_native(
    weights: &ModelWeights,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    h_post_attn: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    larql_compute::kquant_forward::ffn_decode_step_native(
        weights,
        index as &dyn larql_compute::KvIndex,
        backend,
        h_post_attn,
        layer,
    )
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
    state: Option<&mut crate::PerLayerDecodeState>,
) -> Option<Array2<f32>> {
    // Delegate to the substrate copy in larql-compute (ADR-0022).
    larql_compute::kquant_forward::predict_kquant_decode_step_direct_with_state(
        weights,
        token_id,
        index as &dyn larql_compute::KvIndex,
        backend,
        cache,
        abs_position,
        state,
    )
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

    // ── matvec_q4k_or_q6k_q8k dispatcher ────────────────────────────────

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
        for entry in &cache {
            assert!(
                entry.is_some(),
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
            predict_kquant_decode_step(&fx.weights, 4, &fx.index, &mut cache, token_ids.len())
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
        let fx = Q4KTestFixtures::build();
        // Cache length doesn't match num_layers — function must return None.
        let mut bad_cache: CpuKvCache = vec![None; fx.weights.num_layers + 1];
        let result = predict_kquant_decode_step(&fx.weights, 1, &fx.index, &mut bad_cache, 0);
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
            predict_kquant_decode_step(&fx_a.weights, 4, &fx_a.index, &mut cache_a, 3)
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
            predict_kquant_decode_step(&weights_a, next, &index, &mut cache_a, token_ids.len())
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
    use larql_compute::CpuBackend;
    use larql_models::{detect_from_json, ModelWeights, WeightArray};
    use ndarray::Array2;
    use std::collections::HashMap;

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
            per_layer_ffn_format: Default::default(),
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
            predict_kquant_decode_step(&weights, 3, &index, &mut cache, token_ids.len())
                .expect("SiLU dequant decode step must succeed");
        assert!(h_new.iter().all(|v| v.is_finite()));
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
