//! Executor-driven path for `BoundaryPerLayerEngine` (Phase 2
//! migration of the per-layer engines onto `LayerExecutor`).
//!
//! Drives the per-layer dispatch loop through a caller-supplied
//! [`LayerExecutor`] so the caller's FFN backend is honoured (e.g.
//! `--ffn http://shard:8080` routes FFN through a remote shard).
//!
//! Per-layer codec policy state is the engine's responsibility — the
//! executor handles attention + FFN compute only. On fused-kind
//! executors the engine glue falls back to the dense walk via
//! `super::walk::run_prefill` / `run_decode` since per-layer state
//! capture isn't possible.

use larql_inference::attention::SharedKV;
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::layer_executor::LayerExecutor;
use ndarray::{s, Array2};

use crate::engines::boundary_per_layer::cold_tier::{
    extend_cold_kv_with_overflow, last_row, roundtrip,
};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;

/// Executor-driven prefill. Caller MUST have already checked that
/// `executor.dispatch_kind() != Fused` (engine glue falls back to
/// `walk::run_prefill` in that case).
pub(super) fn run_prefill(
    weights: larql_inference::WeightsView,
    executor: &dyn LayerExecutor,
    ffn: &dyn FfnBackend,
    policy: &BoundaryLayerPolicy,
    window_size: Option<usize>,
    token_ids: &[u32],
) -> Option<(Array2<f32>, RsStorePerLayer)> {
    let backend = executor.backend();
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(&weights, token_ids);
    // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
    let ple_inputs =
        larql_inference::forward::ple::precompute_per_layer_inputs(&weights, &h, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_out, _kv) = executor.run_prefill_layer(weights, layer, &h, ffn)?;
        // `LayerExecutor::run_*_layer` returns attention + bare FFN only
        // (`LocalWalkExecutor`, the sole production impl, ends at
        // `run_ffn`); the PLE + layer_scalar tail is the driving loop's
        // responsibility, mirroring the legacy `kv_prefill_run` sequence.
        h = crate::engines::apply_ple_and_layer_scalar(
            weights.canonical(),
            &h_out,
            layer,
            ple_inputs.get(layer),
        );
    }

    let mut rs = RsStorePerLayer {
        stored,
        cold_encoded: None,
        cold_kv: None,
        hot_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window: window_size,
        policy_codecs: policy.entries.clone(),
    };

    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let mut encoded_layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
        let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
        for (layer, overflow) in overflow_per_layer.iter().enumerate() {
            let codec = policy.codec_for(layer);
            let decoded_overflow = roundtrip(overflow, codec);
            let (k, v) = recompute_kv(weights, &decoded_overflow, layer, 0, backend, None)
                .expect("cold K/V pre-computation failed");
            cold_kv.push((k, v));
            let mut enc = PerLayerEncodedColdLayer::empty(codec, weights.hidden_size);
            enc.append(overflow);
            encoded_layers.push(enc);
        }
        rs.cold_encoded = Some(encoded_layers);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }

    Some((last_row(&h), rs))
}

/// Executor-driven decode step, mutating `rs` in place. Caller MUST have
/// already checked that `executor.dispatch_kind() != Fused`.
///
/// Failure invariant: on any `None` return (transient executor / backend
/// failure), the canonical state — `stored`, the cold tiers, and
/// `next_position` — is untouched, so the engine can retry or fall back
/// through another path with the same store.
pub(super) fn run_decode(
    weights: larql_inference::WeightsView,
    executor: &dyn LayerExecutor,
    ffn: &dyn FfnBackend,
    policy: &BoundaryLayerPolicy,
    rs: &mut RsStorePerLayer,
    token_id: u32,
) -> Option<Array2<f32>> {
    let backend = executor.backend();
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    // The executor path extends `stored` without maintaining the walk
    // path's hot-K/V cache; drop the (droppable-derivative) cache so a
    // later walk-path decode cannot attend over a stale buffer.
    rs.hot_kv = None;
    let mut h_new = embed_tokens_pub(&weights, &[token_id]);
    // PLE inputs are per-token — recompute for this single-token decode
    // step, matching the legacy `kv_decode_step_run` recipe exactly.
    let ple_inputs =
        larql_inference::forward::ple::precompute_per_layer_inputs(&weights, &h_new, &[token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let prior_kv: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold, v_cold) = &cold_kv[layer];
            let (k_hot, v_hot) = recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)?;
            let c = k_cold.shape()[0];
            let kv_dim = k_cold.shape()[1];
            let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            k_combined.slice_mut(s![..c, ..]).assign(k_cold);
            k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
            let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            v_combined.slice_mut(s![..c, ..]).assign(v_cold);
            v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
            (k_combined, v_combined)
        } else {
            let (h_full, full_abs_start) = match &rs.cold_encoded {
                Some(cold_layers) if cold_layers[layer].n_positions > 0 => {
                    let decoded = cold_layers[layer].decode();
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                    combined
                        .slice_mut(s![..decoded.shape()[0], ..])
                        .assign(&decoded);
                    combined
                        .slice_mut(s![decoded.shape()[0].., ..])
                        .assign(h_hot);
                    (combined, rs.cold_abs_start)
                }
                _ => (h_hot.clone(), hot_abs_start),
            };
            recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?
        };

        new_stored.push(h_new.clone());
        let (h_out, _new_kv) =
            executor.run_decode_layer(weights, layer, &h_new, &prior_kv, abs_position, ffn)?;
        // Executor returns bare post-FFN hidden; PLE + layer_scalar tail
        // is the driving loop's responsibility (see prefill loop above).
        h_new = crate::engines::apply_ple_and_layer_scalar(
            weights.canonical(),
            &h_out,
            layer,
            ple_inputs.get(layer),
        );
    }

    for (slab, new_row) in rs.stored.iter_mut().zip(new_stored.iter()) {
        slab.push_row(new_row.row(0))
            .expect("push_row shape mismatch");
    }
    rs.next_position = abs_position + 1;

    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_abs_pos =
            rs.cold_abs_start + rs.cold_encoded.as_ref().map_or(0, |l| l[0].n_positions);
        match rs.cold_encoded.as_mut() {
            Some(layers) => {
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    layers[layer].append(overflow);
                }
            }
            None => {
                let hidden = weights.hidden_size;
                let mut layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    let codec = policy.codec_for(layer);
                    let mut enc = PerLayerEncodedColdLayer::empty(codec, hidden);
                    enc.append(overflow);
                    layers.push(enc);
                }
                rs.cold_encoded = Some(layers);
            }
        }
        if extend_cold_kv_with_overflow(
            weights,
            backend,
            policy,
            rs,
            &overflow_per_layer,
            cold_abs_pos,
        )
        .is_none()
        {
            // Per-layer K/V recompute failed: the helper dropped `cold_kv`
            // wholesale (atomicity — no layer desync possible), so the next
            // decode step rebuilds cold K/V from `cold_encoded`.
            debug_assert!(rs.cold_kv.is_none(), "failed extend must drop cold_kv");
        }
    }

    Some(last_row(&h_new))
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn run_prefill_no_window_returns_state_with_no_cold_tier() {
        // window_size = None → no overflow → no cold_encoded, no cold_kv.
        let weights = make_test_weights();
        let backend = CpuBackend;
        let executor = LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let token_ids: Vec<u32> = vec![0, 1, 2];
        let (hidden, rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &executor,
            &ffn,
            &policy,
            None,
            &token_ids,
        )
        .expect("prefill should succeed with synthetic weights");
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
        assert_eq!(rs.next_position, 3);
        assert!(rs.cold_encoded.is_none(), "no overflow → no cold_encoded");
        assert!(rs.cold_kv.is_none(), "no overflow → no cold_kv");
        // Each layer's stored slab has all 3 tokens.
        assert_eq!(rs.stored.len(), weights.num_layers);
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 3, "each layer slab carries all 3 tokens");
        }
    }

    #[test]
    fn run_prefill_with_small_window_evicts_to_cold_tier() {
        // window_size = 2 with 3-token prefill → 1 row of overflow per
        // layer, populates cold_encoded + cold_kv.
        let weights = make_test_weights();
        let backend = CpuBackend;
        let executor = LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let token_ids: Vec<u32> = vec![0, 1, 2];
        let (_hidden, rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &executor,
            &ffn,
            &policy,
            Some(2),
            &token_ids,
        )
        .expect("prefill should succeed");
        assert!(
            rs.cold_encoded.is_some(),
            "overflow path must populate cold_encoded"
        );
        assert!(
            rs.cold_kv.is_some(),
            "overflow path must pre-compute cold_kv"
        );
        let cold_kv = rs.cold_kv.as_ref().unwrap();
        for (k, _v) in cold_kv {
            assert_eq!(k.shape()[0], 1, "1 row of overflow per layer");
        }
    }

    #[test]
    fn run_decode_extends_hot_tier_when_below_window() {
        // After a 1-token prefill with window=4, decode 1 token →
        // both fit in hot, no overflow, no cold-tier mutation.
        let weights = make_test_weights();
        let backend = CpuBackend;
        let executor = LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, mut rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &executor,
            &ffn,
            &policy,
            Some(4),
            &[0],
        )
        .unwrap();
        assert!(
            rs.cold_encoded.is_none(),
            "no overflow expected after prefill"
        );

        let hidden = run_decode(
            larql_inference::WeightsView::dense(&weights),
            &executor,
            &ffn,
            &policy,
            &mut rs,
            1,
        )
        .expect("decode should succeed");
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
        assert_eq!(rs.next_position, 2);
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 2, "hot slab grew to 2 rows");
        }
        // Still no overflow at this scale.
        assert!(rs.cold_encoded.is_none());
    }

    /// Executor-driven loops must run the same per-layer sequence as the
    /// legacy `kv_prefill_run` / `kv_decode_step_run` oracle: attention →
    /// FFN → PLE → layer_scalar. `LayerExecutor::run_*_layer` stops at the
    /// bare FFN, so the PLE + layer_scalar tail lives in this module's
    /// loops — the E2B fixture (non-zero PLE tensors, layer_scalar 0.75)
    /// diverges bit-visibly if the tail is dropped. Representative for all
    /// `LayerExecutor`-driven engine loops (markov_residual{,_codec},
    /// turbo_quant, windowed_checkpoint share the same pattern).
    #[cfg(not(windows))]
    #[test]
    fn executor_prefill_and_decode_match_legacy_on_ple_arch() {
        use crate::generation::{kv_decode_step_run, kv_prefill_run};
        use larql_inference::ffn::WeightFfn;
        use larql_inference::forward::NoopHook;
        use larql_inference::test_utils::make_synthetic_e2b_like_weights;

        const DECODE_STEPS: usize = 3;
        const FIRST_DECODE_TOKEN: u32 = 3;
        // Decode steps 1+ recompute the prior K/V from the stored
        // residuals in one batched matmul, where the legacy cache carries
        // rows projected incrementally (one single-row matmul per step) —
        // BLAS accumulation order legitimately differs by a few ulp
        // (observed ≤ 1e-6 relative L2). Dropping PLE / layer_scalar
        // deviates by ≥ 0.73 relative L2, so the tolerance still pins the
        // bug class. Prefill and decode step 0 attend against
        // identically-produced K/V and stay bit-exact.
        const RECOMPUTE_ACCUM_ORDER_REL_TOL: f32 = 1e-5;

        let weights = make_synthetic_e2b_like_weights();
        let backend = CpuBackend;
        let executor = LocalWalkExecutor::new(&backend);
        let ffn = WeightFfn { weights: &weights };
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let prompt = [0u32, 1, 2];

        let (h_exec, mut rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &executor,
            &ffn,
            &policy,
            None,
            &prompt,
        )
        .expect("executor PLE prefill");
        let (h_ref, mut cache) = kv_prefill_run(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            Some(&backend),
            &mut NoopHook,
        )
        .expect("legacy PLE prefill");
        let bits = |h: &Array2<f32>| h.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        assert_eq!(
            bits(&h_exec),
            bits(&h_ref),
            "executor PLE prefill must match legacy bit-for-bit"
        );

        for step in 0..DECODE_STEPS {
            let token = FIRST_DECODE_TOKEN + step as u32;
            let h_exec = run_decode(
                larql_inference::WeightsView::dense(&weights),
                &executor,
                &ffn,
                &policy,
                &mut rs,
                token,
            )
            .expect("executor PLE decode");
            let h_ref = kv_decode_step_run(
                &weights,
                &ffn,
                &mut cache,
                token,
                Some(&backend),
                &mut NoopHook,
            )
            .expect("legacy PLE decode");
            if step == 0 {
                assert_eq!(
                    bits(&h_exec),
                    bits(&h_ref),
                    "executor PLE decode step 0 must match legacy bit-for-bit"
                );
            } else {
                let ref_norm = h_ref.iter().map(|v| v * v).sum::<f32>().sqrt();
                let err_norm = h_exec
                    .iter()
                    .zip(h_ref.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                let rel = err_norm / ref_norm.max(f32::MIN_POSITIVE);
                assert!(
                    rel <= RECOMPUTE_ACCUM_ORDER_REL_TOL,
                    "executor PLE decode step {step}: relative L2 deviation {rel} \
                     exceeds accumulation-order tolerance"
                );
            }
        }
    }

    #[test]
    fn run_decode_promotes_to_cold_tier_on_overflow() {
        // Prefill 3 tokens with window=2 → 1 row already in cold.
        // Decode 1 more token → 2 rows in cold after eviction.
        // Exercises the Some(layers) arm of cold_encoded.as_mut().
        let weights = make_test_weights();
        let backend = CpuBackend;
        let executor = LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, mut rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &executor,
            &ffn,
            &policy,
            Some(2),
            &[0, 1, 2],
        )
        .unwrap();
        assert!(
            rs.cold_encoded.is_some(),
            "prefill should have populated cold_encoded"
        );
        let initial_cold_rows = rs
            .cold_encoded
            .as_ref()
            .map(|l| l[0].n_positions)
            .unwrap_or(0);
        assert_eq!(initial_cold_rows, 1, "1 row in cold after prefill");

        run_decode(
            larql_inference::WeightsView::dense(&weights),
            &executor,
            &ffn,
            &policy,
            &mut rs,
            3,
        )
        .unwrap();
        let after_cold_rows = rs
            .cold_encoded
            .as_ref()
            .map(|l| l[0].n_positions)
            .unwrap_or(0);
        assert_eq!(after_cold_rows, 2, "decode evicted 1 more row to cold");
        assert_eq!(rs.next_position, 4);
        // cold_kv must track cold_encoded row-for-row (atomic extend).
        let cold_kv = rs.cold_kv.as_ref().expect("overflow keeps cold_kv");
        for (k, v) in cold_kv {
            assert_eq!(k.shape()[0], 2, "cold K rows must match cold_encoded");
            assert_eq!(v.shape()[0], 2, "cold V rows must match cold_encoded");
        }
    }
}
