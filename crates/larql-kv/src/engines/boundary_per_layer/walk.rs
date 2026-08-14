//! CPU walk path for `BoundaryPerLayerEngine`.
//!
//! Mirrors `markov_residual_codec/walk.rs`'s shape: free functions
//! that take all inputs explicitly and return `(hidden, new_store)`
//! — the engine glue (in `engine.rs`) handles store ownership and
//! `KvHandle` lifecycle.
//!
//! The dense path is used when the W1-GPU dispatch path
//! (`dispatch::try_prefill_via_dispatch`) returns `None` —
//! typically on backends/vindexes lacking direct-matvec decode.

use larql_compute::ComputeBackend;
use larql_inference::attention::{run_attention_with_kv_backend, SharedKV};
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::kv_engine::EngineError;
use ndarray::{s, Array2};

use crate::engines::boundary_per_layer::cold_tier::{
    extend_cold_kv_with_overflow, last_row, roundtrip,
};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;

/// Run a full prefill through the dense walk. Returns
/// `(last_hidden, new_store)` — caller owns the store.
///
/// Transactional: the store is built into locals and handed back only on
/// success, so a refusal leaves an engine's existing store untouched.
pub(super) fn run_prefill(
    weights: larql_inference::WeightsView,
    ffn: &dyn FfnBackend,
    backend: &dyn ComputeBackend,
    policy: &BoundaryLayerPolicy,
    window_size: Option<usize>,
    token_ids: &[u32],
) -> Result<(Array2<f32>, RsStorePerLayer), EngineError> {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(&weights, token_ids);
    // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be, None)
            .ok_or_else(|| EngineError::BackendFailure {
                details: format!(
                    "attention returned None during boundary-per-layer prefill at layer {layer}"
                ),
            })?;
        h = crate::engines::layer_ffn_or_moe(
            weights.canonical(),
            &h_post_attn,
            layer,
            ffn,
            Some(ffn),
            ple_inputs.get(layer),
        )
        .map_err(EngineError::Execution)?;
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
                .ok_or_else(|| EngineError::BackendFailure {
                    details: format!("cold K/V pre-computation returned None at layer {layer}"),
                })?;
            cold_kv.push((k, v));
            let mut enc = PerLayerEncodedColdLayer::empty(codec, weights.hidden_size);
            enc.append(overflow);
            encoded_layers.push(enc);
        }
        rs.cold_encoded = Some(encoded_layers);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }

    Ok((last_row(&h), rs))
}

/// A backend stage that declined, as a typed engine failure.
///
/// Named once so every decline in the decode walk reads the same and points
/// at the layer — a bare `None` here used to reach the engine as one
/// undifferentiated "run_decode returned None".
fn declined(stage: &str, layer: usize) -> EngineError {
    EngineError::BackendFailure {
        details: format!("{stage} returned None during boundary-per-layer decode at layer {layer}"),
    }
}

/// Run one decode step through the dense walk, mutating `rs` in place.
///
/// Failure invariant (the reason `rs` is `&mut` and not by-value): on any
/// `Err` return, the canonical state — `stored`, the cold tiers, and
/// `next_position` — is untouched, and `rs.hot_kv` is left `None`. The
/// hot-K/V cache is a droppable derivative of `stored` (see
/// `RsStorePerLayer::hot_kv`), so a transient backend failure costs one
/// rebuild on the next successful step; it never bricks the session.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_decode(
    weights: larql_inference::WeightsView,
    ffn: &dyn FfnBackend,
    backend: &dyn ComputeBackend,
    policy: &BoundaryLayerPolicy,
    rs: &mut RsStorePerLayer,
    token_id: u32,
    index: Option<&larql_vindex::VectorIndex>,
) -> Result<Array2<f32>, EngineError> {
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(&weights, &[token_id]);
    // PLE inputs are per-token — recompute for this single-token decode
    // step, matching the legacy `kv_decode_step_run` recipe exactly.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h_new, &[token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    // W2 hot-K/V cache (twin of markov_residual_codec). When unbounded with no
    // cold tier, `hot_kv` holds the full K/V and the steady state (step 2+)
    // appends the new row IN PLACE + attends over views — instead of
    // `recompute_kv`-ing the whole hot tier AND rebuilding an owned `[ctx+1]`
    // concat every layer every step (the pre-W2 cost this engine carried). The
    // canonical state is still `stored` (the per-layer residuals); `hot_kv` is a
    // droppable derivative. The in-place / owned-concat choice is gated by the
    // shared `LARQL_MARKOV_INPLACE_KV` toggle (default on); both are
    // bit-identical (engine-level A/B test). Windowed/cold configs (the engine's
    // primary purpose) are NOT cache_eligible and keep the recompute path.
    let cache_eligible =
        rs.max_window.is_none() && rs.cold_encoded.is_none() && rs.cold_kv.is_none();
    let mut step_new_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
    // Taken up front: every fallible early return below leaves `rs.hot_kv`
    // as `None` (dropped derivative — see the function docs), because a
    // partially-appended in-place buffer must not survive into a retry.
    let mut hot_kv_store = rs.hot_kv.take();
    let had_hot_kv = hot_kv_store.is_some();
    let idx_kv: Option<&dyn larql_compute::KvIndex> =
        index.map(|v| v as &dyn larql_compute::KvIndex);
    let inplace_enabled = crate::engines::markov_residual::compute::markov_inplace_kv_enabled();

    for layer in 0..num_layers {
        // `stored` is push_row-grown, so `shape()[0]` IS the logical hot length.
        let s_hot = rs.stored[layer].shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        new_stored.push(h_new.clone());

        let h_post_attn = if cache_eligible && had_hot_kv {
            // STEADY STATE (step 2+): append in place into the doubling-capacity
            // `hot_kv` buffer and attend over the `[..s_hot+1]` views.
            let bufs = hot_kv_store.as_mut().expect("had_hot_kv");
            #[cfg(debug_assertions)]
            {
                // f32-path parity gate (the Q4K route's projections differ from
                // f32 `recompute_kv` by >1e-2; it has its own A/B oracle).
                if !larql_compute::options::q4k_direct_attn_enabled() {
                    let (k_buf, v_buf) = &bufs[layer];
                    if let Some((rk, rv)) = recompute_kv(
                        weights,
                        &rs.stored[layer],
                        layer,
                        hot_abs_start,
                        backend,
                        None,
                    ) {
                        let kd = k_buf
                            .slice(s![..s_hot, ..])
                            .iter()
                            .zip(rk.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        let vd = v_buf
                            .slice(s![..s_hot, ..])
                            .iter()
                            .zip(rv.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        debug_assert!(kd < 1e-2, "boundary-per-layer hot_kv K diverged: {kd}");
                        debug_assert!(vd < 1e-2, "boundary-per-layer hot_kv V diverged: {vd}");
                    }
                }
            }
            let (k_buf, v_buf) = &mut bufs[layer];
            let inplace = if inplace_enabled {
                larql_inference::attention::run_attention_block_decode_step_auto_inplace(
                    weights,
                    &h_new,
                    layer,
                    k_buf,
                    v_buf,
                    s_hot,
                    abs_position,
                    Some(backend),
                    idx_kv,
                )
            } else {
                None
            };
            match inplace {
                Some(h) => h,
                None => {
                    // Q4K-direct off (flags-off parity) or no attn bytes: owned
                    // concat over the buffer view, then replace.
                    let prior: SharedKV = (
                        k_buf.slice(s![..s_hot, ..]).to_owned(),
                        v_buf.slice(s![..s_hot, ..]).to_owned(),
                    );
                    let (h, new_kv) =
                        larql_inference::attention::run_attention_block_decode_step_auto(
                            weights,
                            &h_new,
                            layer,
                            Some(&prior),
                            abs_position,
                            Some(backend),
                            idx_kv,
                        )
                        .ok_or_else(|| declined("attention", layer))?;
                    *k_buf = new_kv.0;
                    *v_buf = new_kv.1;
                    h
                }
            }
        } else {
            // FIRST STEP (cache None → seed) or windowed/cold tier: recompute the
            // prior K/V, let attention concat the new row, collect it (the
            // cache_eligible first step seeds `hot_kv`).
            let h_hot = &rs.stored[layer];
            let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold, v_cold) = &cold_kv[layer];
                let (k_hot, v_hot) =
                    recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)
                        .ok_or_else(|| declined("hot K/V recompute", layer))?;
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
                let (h_full, full_abs_start) = if let Some(cold_layers) = &rs.cold_encoded {
                    let enc = &cold_layers[layer];
                    if enc.n_positions > 0 {
                        let decoded = enc.decode();
                        let hidden = h_hot.shape()[1];
                        let mut combined =
                            Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                        combined
                            .slice_mut(s![..decoded.shape()[0], ..])
                            .assign(&decoded);
                        combined
                            .slice_mut(s![decoded.shape()[0].., ..])
                            .assign(h_hot);
                        (combined, rs.cold_abs_start)
                    } else {
                        (h_hot.clone(), hot_abs_start)
                    }
                } else {
                    (h_hot.clone(), hot_abs_start)
                };
                recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)
                    .ok_or_else(|| declined("cold K/V recompute", layer))?
            };

            let (h_post_attn, new_kv) =
                larql_inference::attention::run_attention_block_decode_step_auto(
                    weights,
                    &h_new,
                    layer,
                    Some(&(k_full, v_full)),
                    abs_position,
                    Some(backend),
                    idx_kv,
                )
                .ok_or_else(|| declined("attention", layer))?;
            if cache_eligible {
                step_new_kv.push(new_kv);
            }
            h_post_attn
        };

        h_new = crate::engines::layer_ffn_or_moe(
            weights.canonical(),
            &h_post_attn,
            layer,
            ffn,
            Some(ffn),
            ple_inputs.get(layer),
        )
        .map_err(EngineError::Execution)?;
    }

    // Amortised O(m) per-row append via ndarray::Array2::push_row.
    // Replaces the O(N²) per-step "Array2::zeros + .assign" rebuild
    // (bug A; see `engines/boundary_per_layer/mod.rs`).
    for (slab, new_row) in rs.stored.iter_mut().zip(new_stored.iter()) {
        slab.push_row(new_row.row(0))
            .expect("push_row shape mismatch");
    }
    rs.next_position = abs_position + 1;
    // Step 2+ mutated `hot_kv_store` in place; the first step seeds it. Cleared
    // for windowed/cold configs (the recompute path stays canonical there).
    rs.hot_kv = if cache_eligible {
        if had_hot_kv {
            hot_kv_store
        } else {
            Some(step_new_kv)
        }
    } else {
        None
    };

    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        // Snapshot the absolute position at which the new overflow rows land
        // BEFORE appending to cold_encoded. Used by extend_cold_kv for RoPE.
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
            // decode step rebuilds cold K/V from `cold_encoded`. The decode
            // output itself is already computed and stays valid.
            debug_assert!(rs.cold_kv.is_none(), "failed extend must drop cold_kv");
        }
    }

    Ok(last_row(&h_new))
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn run_prefill_no_window_returns_state_with_no_cold_tier() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (hidden, rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            None,
            &[0, 1, 2],
        )
        .expect("prefill should succeed");
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
        assert_eq!(rs.next_position, 3);
        assert!(rs.cold_encoded.is_none());
        assert!(rs.cold_kv.is_none());
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 3);
        }
    }

    #[test]
    fn run_prefill_with_small_window_evicts_to_cold_tier() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            Some(2),
            &[0, 1, 2],
        )
        .unwrap();
        assert!(rs.cold_encoded.is_some(), "overflow → cold_encoded");
        assert!(rs.cold_kv.is_some(), "overflow → cold_kv pre-computed");
        let cold_kv = rs.cold_kv.as_ref().unwrap();
        for (k, _v) in cold_kv {
            assert_eq!(k.shape()[0], 1);
        }
    }

    #[test]
    fn run_decode_extends_hot_tier_when_below_window() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, mut rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            Some(4),
            &[0],
        )
        .unwrap();
        assert!(rs.cold_encoded.is_none());

        let hidden = run_decode(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            &mut rs,
            1,
            None,
        )
        .unwrap();
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
        assert_eq!(rs.next_position, 2);
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 2);
        }
        assert!(rs.cold_encoded.is_none());
    }

    #[test]
    fn run_decode_uses_cold_encoded_when_cold_kv_is_none() {
        // Defensive branch: cold_kv is None but cold_encoded is Some
        // and non-empty. In practice run_prefill always builds both
        // together, but the code carries a fallback path for the
        // case where they get desynchronised. Hand-construct that
        // state to exercise the decode path.
        use crate::engines::markov_residual_codec::codec::ColdResidualCodec;

        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);

        // First build a normal state with overflow.
        let (_, mut rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            Some(2),
            &[0, 1, 2],
        )
        .unwrap();
        // Now wipe the pre-computed cold_kv. cold_encoded stays
        // populated. Decode should recompute K/V from the decoded
        // cold residuals.
        rs.cold_kv = None;
        // Sanity: the cold_encoded still carries the evicted row.
        assert!(rs.cold_encoded.as_ref().unwrap()[0].n_positions > 0);

        let _ = ColdResidualCodec::Bf16; // keep import live
        let hidden = run_decode(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            &mut rs,
            3,
            None,
        )
        .expect("decode should succeed without cold_kv");
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
    }

    /// Flags-ON parity gate for the W2 in-place hot-K/V fast path: an A/B of the
    /// in-place steady state vs the owned-concat reference, both with Q4K-direct
    /// attention live. Twin of the markov/codec tests — the two paths must
    /// produce bit-identical hidden states every step. Serialised on
    /// `Q4K_FLAG_ENV_LOCK`; the path is selected via the shared
    /// `LARQL_MARKOV_INPLACE_KV` thread-local override.
    #[test]
    fn run_decode_inplace_matches_owned_concat_flags_on() {
        use crate::engines::markov_residual::compute::set_markov_env_override;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

        let _q4k = crate::engines::Q4kFlagGuard::set(&[
            (larql_compute::options::ENV_Q4K_DIRECT_ATTN, true),
            (larql_compute::options::ENV_Q4K_ATTN_INT8, false),
        ]);

        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);

        let run = |inplace: bool| -> (Vec<Vec<u32>>, usize) {
            set_markov_env_override(
                "LARQL_MARKOV_INPLACE_KV",
                Some(if inplace { "1" } else { "0" }),
            );
            let (_, mut rs) = run_prefill(
                larql_inference::WeightsView::dense(&weights),
                &ffn,
                &backend,
                &policy,
                None,
                &[0u32, 1, 2],
            )
            .unwrap();
            let mut hiddens = Vec::new();
            for tok in 3u32..=12 {
                let h = run_decode(
                    larql_inference::WeightsView::dense(&weights),
                    &ffn,
                    &backend,
                    &policy,
                    &mut rs,
                    tok,
                    Some(&index),
                )
                .unwrap();
                assert!(h.iter().all(|v| v.is_finite()));
                hiddens.push(h.iter().map(|v| v.to_bits()).collect());
            }
            (hiddens, rs.next_position)
        };

        let (a, a_pos) = run(true);
        let (b, b_pos) = run(false);
        assert_eq!(a_pos, 13, "3 prompt + 10 decode");
        assert_eq!(a_pos, b_pos);
        assert_eq!(
            a, b,
            "boundary-per-layer in-place vs owned-concat hidden states diverged"
        );
    }

    #[test]
    fn run_decode_promotes_to_cold_tier_on_overflow() {
        // Prefill 3 with window=2 → 1 in cold. Decode 1 → 2 in cold.
        // Exercises Some(layers) arm of cold_encoded match.
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, mut rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            Some(2),
            &[0, 1, 2],
        )
        .unwrap();
        let initial = rs
            .cold_encoded
            .as_ref()
            .map(|l| l[0].n_positions)
            .unwrap_or(0);
        assert_eq!(initial, 1);

        run_decode(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            &mut rs,
            3,
            None,
        )
        .unwrap();
        let after = rs
            .cold_encoded
            .as_ref()
            .map(|l| l[0].n_positions)
            .unwrap_or(0);
        assert_eq!(after, 2);
        assert_eq!(rs.next_position, 4);
        // The pre-computed cold K/V cache must track cold_encoded row-for-row
        // on every layer — a desync here is exactly the state the atomic
        // extend contract forbids.
        let cold_kv = rs.cold_kv.as_ref().expect("overflow keeps cold_kv");
        for (k, v) in cold_kv {
            assert_eq!(k.shape()[0], 2, "cold K rows must match cold_encoded");
            assert_eq!(v.shape()[0], 2, "cold V rows must match cold_encoded");
        }
    }

    /// First-overflow cold-tier initialisation. A prefill that fills the
    /// window exactly leaves `cold_encoded = None`; the first decode
    /// pushes past the window, so its overflow row must *initialise*
    /// `cold_encoded` (the `None` match arm) rather than append to an
    /// existing one.
    #[test]
    fn run_decode_initialises_cold_tier_on_first_overflow() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);

        // window=2, exactly 2 prompt tokens → fills the window, no overflow.
        let (_, mut rs) = run_prefill(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            Some(2),
            &[0, 1],
        )
        .unwrap();
        assert!(
            rs.cold_encoded.is_none(),
            "a window-filling prefill must not overflow yet"
        );

        // First decode overflows by one row → initialises cold_encoded.
        run_decode(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &backend,
            &policy,
            &mut rs,
            2,
            None,
        )
        .unwrap();
        assert!(
            rs.cold_encoded.is_some(),
            "first decode overflow must initialise the cold tier"
        );
        assert_eq!(
            rs.cold_encoded.as_ref().unwrap()[0].n_positions,
            1,
            "exactly one row evicted to cold"
        );
        // First overflow also initialises cold_kv (extend's None arm), in
        // lockstep with cold_encoded.
        let cold_kv = rs.cold_kv.as_ref().expect("first overflow inits cold_kv");
        for (k, v) in cold_kv {
            assert_eq!(k.shape()[0], 1);
            assert_eq!(v.shape()[0], 1);
        }
    }
}
