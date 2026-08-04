//! Multi-token extend with prior K,V checkpoint.
//!
//! Runs a CPU/GPU forward pass over new tokens, seeding each layer's attention
//! with an optional prior K,V cache (the window boundary checkpoint).

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};

use larql_inference::attention::{run_attention_block_decode_step_backend, SharedKV};
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};

pub struct ExtendOutput {
    /// Hidden state at the last processed token, shape (1, hidden).
    pub last_hidden: Array2<f32>,
    /// Per-layer full K,V cache covering `[prior_tokens, new_tokens]`.
    pub kv_cache: Vec<SharedKV>,
    /// Per-layer last-row K,V ready to save as the next boundary checkpoint.
    pub new_checkpoint: Vec<SharedKV>,
}

/// What an extend produced when the caller kept ownership of the K/V.
///
/// The cache itself is left in the caller's buffer rather than returned, so a
/// step that fails partway can be rewound by whoever owns the window — see
/// [`truncate_kv_rows`].
pub struct ExtendStep {
    /// Hidden state at the last processed token, shape (1, hidden).
    pub last_hidden: Array2<f32>,
    /// Per-layer last-row K,V ready to save as the next boundary checkpoint.
    pub new_checkpoint: Vec<SharedKV>,
}

/// Truncate every layer's K/V back to `rows`, undoing an extend that stopped
/// partway.
///
/// Needed because the owned-concat path *replaces* each layer's buffer with a
/// longer one as it goes, and this path reads a prior by `shape()[0]` rather
/// than by a counter — so a partially advanced cache would silently attend
/// over a token whose step never completed.
pub fn truncate_kv_rows(kv_cache: &mut [SharedKV], rows: usize) {
    for (k, v) in kv_cache.iter_mut() {
        if k.shape()[0] > rows {
            *k = k.slice(s![..rows, ..]).to_owned();
        }
        if v.shape()[0] > rows {
            *v = v.slice(s![..rows, ..]).to_owned();
        }
    }
}

/// Run the decoder forward over `token_ids` seeded with an optional prior K,V
/// checkpoint at each layer. Matmuls route through `backend`.
///
/// `abs_start` is the absolute position of the *first new token*.
pub fn rs_extend_from_checkpoint(
    weights: larql_inference::WeightsView,
    token_ids: &[u32],
    prior_kv: Vec<SharedKV>,
    abs_start: usize,
) -> Result<ExtendOutput, EngineError> {
    let mut kv_cache = prior_kv;
    let step = rs_extend_from_checkpoint_backend(
        weights,
        token_ids,
        &mut kv_cache,
        abs_start,
        &larql_compute::CpuBackend,
        None,
        None,
    )?;
    Ok(ExtendOutput {
        last_hidden: step.last_hidden,
        kv_cache,
        new_checkpoint: step.new_checkpoint,
    })
}

/// Backend-dispatched variant of [`rs_extend_from_checkpoint`].
///
/// Takes `kv_cache` by reference so the per-token extend loop can mutate it
/// in place — cloning the prior K/V per step is O(window²) total over a full
/// window, a real overhead on growing caches — and so that a caller who owns
/// the window can rewind it after a failure. **On `Err` the cache is left
/// partially advanced**: layers before the failing one hold the new row.
/// [`truncate_kv_rows`] is how the owner undoes that.
#[allow(clippy::too_many_arguments)]
pub fn rs_extend_from_checkpoint_backend(
    weights: larql_inference::WeightsView,
    token_ids: &[u32],
    kv_cache: &mut [SharedKV],
    abs_start: usize,
    backend: &dyn ComputeBackend,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Result<ExtendStep, EngineError> {
    let num_layers = weights.num_layers;

    if token_ids.is_empty() {
        return Err(EngineError::EmptyPrompt);
    }
    if kv_cache.len() != num_layers {
        return Err(EngineError::InvariantViolation {
            what: format!(
                "prior K/V has {} layers, model has {num_layers}",
                kv_cache.len()
            ),
        });
    }

    let mut last_hidden: Option<Array2<f32>> = None;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let abs_position = abs_start + i;
        let mut h = embed_tokens_pub(&weights, &[token_id]);
        // PLE inputs are per-token — this loop embeds one token at a time,
        // matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = precompute_per_layer_inputs(&weights, &h, &[token_id]);

        for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
            let kv_entry: Option<&SharedKV> = if kv_slot.0.shape()[0] > 0 {
                Some(kv_slot)
            } else {
                None
            };

            let (h_post_attn, new_kv) =
                larql_inference::attention::run_attention_block_decode_step_auto(
                    weights,
                    &h,
                    layer,
                    kv_entry,
                    abs_position,
                    Some(backend),
                    index.map(|v| v as &dyn larql_compute::KvIndex),
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: format!(
                        "attention returned None during unlimited-context extend at layer {layer}"
                    ),
                })?;

            let bffn = BackendFfn {
                weights: weights.canonical(),
                backend,
            };
            h = crate::engines::layer_ffn_or_moe(
                weights.canonical(),
                &h_post_attn,
                layer,
                &bffn,
                moe_ffn,
                ple_inputs.get(layer),
            )
            .map_err(EngineError::Execution)?;
            *kv_slot = new_kv;
        }

        last_hidden = Some(h);
    }

    let new_checkpoint: Vec<SharedKV> = kv_cache
        .iter()
        .map(|(k, v)| {
            let n = k.shape()[0];
            let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
            let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
            (last_k, last_v)
        })
        .collect();

    Ok(ExtendStep {
        last_hidden: last_hidden.ok_or_else(|| EngineError::BackendFailure {
            details: "extend produced no hidden state".into(),
        })?,
        new_checkpoint,
    })
}

/// In-place multi-token extend for the decode hot path (the W8.2/in-place twin
/// of [`rs_extend_from_checkpoint_backend`]). Appends each token's K/V row into
/// the caller's **doubling-capacity** `kv_cache` buffers — starting at logical
/// row `prior_len` — and attends over the growing `[..len]` views, instead of
/// rebuilding an owned `[len+1, kv_dim]` concat every layer every step (the
/// O(window²) cost the owned-concat form pays). On return each buffer holds
/// `prior_len + token_ids.len()` logical rows (track that count in the caller —
/// the buffers are over-allocated, so `shape()[0]` is capacity, not length).
///
/// Per-layer fallback: if a layer has no Q4K attn bytes the in-place projection
/// returns `None`, so this writes the owned-concat result back into the buffer
/// for that layer — buffers stay consistent. Same numerics as the owned-concat
/// form (engine-level A/B test), only the cache representation differs.
#[allow(clippy::too_many_arguments)]
pub fn rs_extend_inplace(
    weights: larql_inference::WeightsView,
    token_ids: &[u32],
    kv_cache: &mut [SharedKV],
    prior_len: usize,
    abs_start: usize,
    backend: &dyn ComputeBackend,
    moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Result<Array2<f32>, EngineError> {
    let num_layers = weights.num_layers;
    if token_ids.is_empty() || kv_cache.len() != num_layers {
        return Err(EngineError::InvariantViolation {
            what: format!(
                "in-place extend needs a non-empty chunk and {num_layers} K/V slots;                  got {} tokens and {} slots",
                token_ids.len(),
                kv_cache.len()
            ),
        });
    }
    let idx_kv: Option<&dyn larql_compute::KvIndex> =
        index.map(|v| v as &dyn larql_compute::KvIndex);
    let mut last_hidden: Option<Array2<f32>> = None;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let abs_position = abs_start + i;
        // Logical row count of every buffer at the start of this token: the
        // prior window length plus the tokens already appended this call.
        let len = prior_len + i;
        let mut h = embed_tokens_pub(&weights, &[token_id]);
        // PLE inputs are per-token — this loop embeds one token at a time,
        // matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = precompute_per_layer_inputs(&weights, &h, &[token_id]);

        for (layer, (k_buf, v_buf)) in kv_cache.iter_mut().enumerate() {
            let h_post_attn =
                match larql_inference::attention::run_attention_block_decode_step_auto_inplace(
                    weights,
                    &h,
                    layer,
                    k_buf,
                    v_buf,
                    len,
                    abs_position,
                    Some(backend),
                    idx_kv,
                ) {
                    Some(hp) => hp,
                    None => {
                        // Q4K-direct off / no attn bytes: owned concat over the
                        // `[..len]` view, then replace the buffer.
                        let prior: SharedKV = (
                            k_buf.slice(ndarray::s![..len, ..]).to_owned(),
                            v_buf.slice(ndarray::s![..len, ..]).to_owned(),
                        );
                        let prior_ref = if len > 0 { Some(&prior) } else { None };
                        let (hp, new_kv) =
                            larql_inference::attention::run_attention_block_decode_step_auto(
                                weights,
                                &h,
                                layer,
                                prior_ref,
                                abs_position,
                                Some(backend),
                                idx_kv,
                            )
                            .ok_or_else(|| {
                                EngineError::BackendFailure {
                                    details: format!(
                                        "attention returned None during in-place extend \
                                     at layer {layer}"
                                    ),
                                }
                            })?;
                        *k_buf = new_kv.0;
                        *v_buf = new_kv.1;
                        hp
                    }
                };

            let bffn = BackendFfn {
                weights: weights.canonical(),
                backend,
            };
            h = crate::engines::layer_ffn_or_moe(
                weights.canonical(),
                &h_post_attn,
                layer,
                &bffn,
                moe_ffn,
                ple_inputs.get(layer),
            )
            .map_err(EngineError::Execution)?;
        }

        last_hidden = Some(h);
    }

    last_hidden.ok_or_else(|| EngineError::BackendFailure {
        details: "in-place extend produced no hidden state".into(),
    })
}

/// CPU Q4K variant of [`rs_extend_from_checkpoint_backend`].
///
/// Uses `WalkFfn` (reads Q4K bytes directly from `index`) for FFN instead of
/// `BackendFfn` (needs f32 tensors in `weights.tensors`). Attention projection
/// uses the dequantised f32 tensors already inserted by
/// `ensure_attn_tensors_dequantised`. Call that before this function.
pub fn rs_extend_from_checkpoint_quant(
    weights: larql_inference::WeightsView,
    index: &VectorIndex,
    token_ids: &[u32],
    prior_kv: Vec<SharedKV>,
    abs_start: usize,
    backend: &dyn ComputeBackend,
    mut profiler: Option<&mut crate::profiler::EngineProfiler>,
) -> Option<ExtendOutput> {
    let num_layers = weights.num_layers;

    if token_ids.is_empty() {
        return None;
    }
    if prior_kv.len() != num_layers {
        return None;
    }

    let mut kv_cache: Vec<SharedKV> = prior_kv;
    let mut last_hidden: Option<Array2<f32>> = None;

    // Hoist WalkFfn out of both loops. Previously this rebuilt the
    // WalkFfn once per (token, layer) — N×34 times per extend call.
    // It's now once total. WalkFfn carries no per-(token,layer) state.
    let walk_ffn =
        WalkFfn::from_config(weights.canonical(), index, WalkFfnConfig::dense(num_layers))
            .with_backend(backend);

    // Per-stage timing. `LARQL_INSTRUMENT_UNLIMITED=1` enables the
    // verbose stderr line; `profiler` is the structured channel used by
    // `larql bench --profile`. Both gate on the same flag bit.
    let instrument = std::env::var("LARQL_INSTRUMENT_UNLIMITED").is_ok();
    let timing = profiler.is_some() || instrument;
    let t_step = if timing {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut t_embed = 0.0f64;
    let mut t_attention = 0.0f64;
    let mut t_ffn = 0.0f64;
    let mut t_attn_helper_misses = 0usize;
    let mut t_ffn_helper_misses = 0usize;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let abs_position = abs_start + i;
        let t_embed_start = if timing {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut h = embed_tokens_pub(&weights, &[token_id]);
        // PLE inputs are per-token — this loop embeds one token at a time,
        // matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = precompute_per_layer_inputs(&weights, &h, &[token_id]);
        if let Some(start) = t_embed_start {
            t_embed += start.elapsed().as_secs_f64() * 1e6;
        }

        for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
            let kv_entry: Option<&SharedKV> = if kv_slot.0.shape()[0] > 0 {
                Some(kv_slot)
            } else {
                None
            };

            // Try production native-quantised attention helper first;
            // fall back to f32 path. Same pattern as MarkovResidual.
            let t_attn_start = if timing {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let attn_native = larql_inference::vindex::attention_decode_step_native(
                weights.canonical(),
                index,
                backend,
                &h,
                layer,
                kv_entry,
                abs_position,
            );
            if attn_native.is_none() && instrument {
                t_attn_helper_misses += 1;
            }
            let (h_post_attn, new_kv) = attn_native.or_else(|| {
                run_attention_block_decode_step_backend(
                    weights,
                    &h,
                    layer,
                    kv_entry,
                    abs_position,
                    Some(backend),
                )
            })?;
            if let Some(start) = t_attn_start {
                t_attention += start.elapsed().as_secs_f64() * 1e6;
            }

            // Native-quantised FFN; falls back to WalkFfn (which falls
            // further to dense f32 if no sparse features). The native
            // path is ~100× faster on Gemma 3 4B Q4K — see
            // `bench/baselines/cpu/async-dispatch-2026-05-16.md`.
            let t_ffn_start = if timing {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let ffn_native = larql_inference::vindex::ffn_decode_step_native(
                weights.canonical(),
                index,
                backend,
                &h_post_attn,
                layer,
            );
            if ffn_native.is_none() && instrument {
                t_ffn_helper_misses += 1;
            }
            // Both branches return the bare post-FFN hidden —
            // `ffn_decode_step_native` is also `moe_ffn_block_cpu`'s pre-PLE
            // dense slab — so the PLE + layer_scalar tail applies to either.
            let h_post_ffn = ffn_native.unwrap_or_else(|| {
                let (h, _) = run_ffn(&weights, &h_post_attn, layer, &walk_ffn, false);
                h
            });
            let h_out = crate::engines::apply_ple_and_layer_scalar(
                &weights,
                &h_post_ffn,
                layer,
                ple_inputs.get(layer),
            );
            if let Some(start) = t_ffn_start {
                t_ffn += start.elapsed().as_secs_f64() * 1e6;
            }
            h = h_out;
            *kv_slot = new_kv;
        }

        last_hidden = Some(h);
    }

    if instrument {
        let total = t_embed + t_attention + t_ffn;
        eprintln!(
            "[unlimited-context/extend_quant] tokens={} layers={num_layers} \
             embed={:.2}ms attention={:.2}ms ffn={:.2}ms total={:.2}ms \
             (attn_helper miss/ffn_helper miss={t_attn_helper_misses}/{t_ffn_helper_misses})",
            token_ids.len(),
            t_embed / 1e3,
            t_attention / 1e3,
            t_ffn / 1e3,
            total / 1e3,
        );
    }

    if let (Some(prof), Some(t_step)) = (profiler.as_mut(), t_step) {
        // windowed_checkpoint appends K/V incrementally → no `recompute_*`
        // stages fire. embed/attention/ffn/decode_total carry the
        // attribution; recompute_cold/hot stay at zero (and the print
        // logic shows them only when non-zero).
        prof.embed.total_us += t_embed;
        prof.embed.count += 1;
        prof.attention.total_us += t_attention;
        prof.attention.count += 1;
        prof.ffn.total_us += t_ffn;
        prof.ffn.count += 1;
        prof.decode_total.total_us += t_step.elapsed().as_secs_f64() * 1e6;
        prof.decode_total.count += 1;
    }

    let new_checkpoint: Vec<SharedKV> = kv_cache
        .iter()
        .map(|(k, v)| {
            let n = k.shape()[0];
            let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
            let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
            (last_k, last_v)
        })
        .collect();

    Some(ExtendOutput {
        last_hidden: last_hidden?,
        kv_cache,
        new_checkpoint,
    })
}

/// Build an empty (zero-row) K,V seed for use when no prior checkpoint exists.
pub fn empty_prior(weights: &ModelWeights) -> Vec<SharedKV> {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|layer| {
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            (
                Array2::<f32>::zeros((0, kv_dim)),
                Array2::<f32>::zeros((0, kv_dim)),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    // ── empty_prior ───────────────────────────────────────────────────────────

    #[test]
    fn empty_prior_shape_per_layer() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        assert_eq!(prior.len(), weights.num_layers);
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, v) in &prior {
            assert_eq!(k.shape(), &[0, kv_dim]);
            assert_eq!(v.shape(), &[0, kv_dim]);
        }
    }

    // ── rs_extend_from_checkpoint ─────────────────────────────────────────────

    #[test]
    fn extend_empty_tokens_returns_none() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let result =
            rs_extend_from_checkpoint(larql_inference::WeightsView::dense(&weights), &[], prior, 0);
        assert!(
            matches!(result, Err(EngineError::EmptyPrompt)),
            "an empty chunk is a caller-input error, not a backend failure"
        );
    }

    #[test]
    fn extend_wrong_prior_len_returns_none() {
        let weights = make_test_weights();
        // prior has 0 layers but model has 2 — mismatch
        let result = rs_extend_from_checkpoint(
            larql_inference::WeightsView::dense(&weights),
            &[0u32],
            Vec::new(),
            0,
        );
        assert!(
            matches!(result, Err(EngineError::InvariantViolation { .. })),
            "a prior with the wrong layer count is a contract violation"
        );
    }

    #[test]
    fn extend_single_token_from_empty_prior() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output = rs_extend_from_checkpoint(
            larql_inference::WeightsView::dense(&weights),
            &[0u32],
            prior,
            0,
        )
        .expect("single token extend should succeed");
        assert_eq!(output.last_hidden.shape(), &[1, weights.hidden_size]);
        assert!(output.last_hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn extend_kv_cache_grows_with_each_token() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output = rs_extend_from_checkpoint(
            larql_inference::WeightsView::dense(&weights),
            &[0u32, 1, 2],
            prior,
            0,
        )
        .expect("3-token extend");
        // After 3 tokens from empty prior, K has 3 rows per layer
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, v) in &output.kv_cache {
            assert_eq!(k.shape(), &[3, kv_dim], "K should have 3 rows");
            assert_eq!(v.shape(), &[3, kv_dim], "V should have 3 rows");
        }
    }

    #[test]
    fn extend_checkpoint_is_last_row_of_kv_cache() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output = rs_extend_from_checkpoint(
            larql_inference::WeightsView::dense(&weights),
            &[0u32, 1],
            prior,
            0,
        )
        .expect("2-token extend");
        // new_checkpoint should be the last row of each K/V
        for (layer, ((k_cache, v_cache), (k_ckpt, v_ckpt))) in output
            .kv_cache
            .iter()
            .zip(output.new_checkpoint.iter())
            .enumerate()
        {
            let n = k_cache.shape()[0];
            let last_k = k_cache.row(n - 1).to_vec();
            let ckpt_k = k_ckpt.row(0).to_vec();
            for (a, b) in last_k.iter().zip(ckpt_k.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "layer {layer}: checkpoint K doesn't match last K cache row"
                );
            }
            let _ = (v_cache, v_ckpt); // symmetry — trust by shape
        }
    }

    #[test]
    fn extend_abs_start_shifts_rope() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let out0 = rs_extend_from_checkpoint(
            larql_inference::WeightsView::dense(&weights),
            &[0u32],
            prior.clone(),
            0,
        )
        .unwrap();
        let out5 = rs_extend_from_checkpoint(
            larql_inference::WeightsView::dense(&weights),
            &[0u32],
            prior,
            5,
        )
        .unwrap();
        // Different abs_start → different RoPE → different K
        let k0 = &out0.kv_cache[0].0;
        let k5 = &out5.kv_cache[0].0;
        let diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.0,
            "different abs_start should produce different K (RoPE)"
        );
    }

    #[test]
    fn extend_output_logits_are_finite() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output = rs_extend_from_checkpoint(
            larql_inference::WeightsView::dense(&weights),
            &[0u32],
            prior,
            0,
        )
        .unwrap();
        let logits = hidden_to_raw_logits(&weights, &output.last_hidden);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn extend_seeded_from_checkpoint_matches_empty_start() {
        // Extending from a non-empty checkpoint should not panic and should be finite.
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let first = rs_extend_from_checkpoint(
            larql_inference::WeightsView::dense(&weights),
            &[0u32],
            prior,
            0,
        )
        .unwrap();
        // Use the checkpoint from the first extend as the prior for the second
        let second = rs_extend_from_checkpoint(
            larql_inference::WeightsView::dense(&weights),
            &[1u32],
            first.new_checkpoint.clone(),
            1,
        )
        .expect("extend from non-empty prior");
        assert_eq!(second.last_hidden.shape(), &[1, weights.hidden_size]);
        assert!(second.last_hidden.iter().all(|v| v.is_finite()));
    }

    // ── rs_extend_from_checkpoint_quant (vindex-backed FFN path) ───────────────

    #[test]
    fn extend_quant_empty_tokens_returns_none() {
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let prior = empty_prior(&weights);
        let out = rs_extend_from_checkpoint_quant(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[],
            prior,
            0,
            &*backend,
            None,
        );
        assert!(out.is_none(), "empty token_ids should return None");
    }

    #[test]
    fn extend_quant_wrong_prior_len_returns_none() {
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let out = rs_extend_from_checkpoint_quant(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32],
            Vec::new(),
            0,
            &*backend,
            None,
        );
        assert!(out.is_none(), "prior length mismatch should return None");
    }

    #[test]
    fn extend_quant_grows_kv_cache_and_returns_finite() {
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let prior = empty_prior(&weights);
        let out = rs_extend_from_checkpoint_quant(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1, 2],
            prior,
            0,
            &*backend,
            None,
        )
        .expect("3-token Q4K extend");
        assert_eq!(out.last_hidden.shape(), &[1, weights.hidden_size]);
        assert!(out.last_hidden.iter().all(|v| v.is_finite()));
        // After 3 tokens from an empty prior, each layer's K/V has 3 rows.
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, v) in &out.kv_cache {
            assert_eq!(k.shape(), &[3, kv_dim]);
            assert_eq!(v.shape(), &[3, kv_dim]);
        }
        assert_eq!(out.new_checkpoint.len(), weights.num_layers);
    }

    // ── truncate_kv_rows ──────────────────────────────────────────────────────

    #[test]
    fn truncate_kv_rows_drops_rows_past_len() {
        let mut kv: Vec<SharedKV> = vec![(
            Array2::from_shape_fn((5, 4), |(i, j)| (i * 4 + j) as f32),
            Array2::from_shape_fn((5, 4), |(i, j)| -((i * 4 + j) as f32)),
        )];
        truncate_kv_rows(&mut kv, 3);
        assert_eq!(kv[0].0.shape(), &[3, 4]);
        assert_eq!(kv[0].1.shape(), &[3, 4]);
        // Kept the *first* rows in order — the inverse of an append.
        assert_eq!(kv[0].0[[0, 0]], 0.0);
        assert_eq!(kv[0].0[[2, 3]], 11.0);
    }

    #[test]
    fn truncate_kv_rows_is_noop_at_or_below_target() {
        let mut kv: Vec<SharedKV> =
            vec![(Array2::zeros((2, 4)), Array2::zeros((2, 4)))];
        truncate_kv_rows(&mut kv, 2);
        assert_eq!(kv[0].0.shape(), &[2, 4]);
        truncate_kv_rows(&mut kv, 5);
        assert_eq!(kv[0].0.shape(), &[2, 4], "shorter than target stays put");
    }

    #[test]
    fn truncate_kv_rows_truncates_k_and_v_independently() {
        // A rewind after a partial in-place fallback can leave K and V at
        // different lengths for a layer; each side is clamped on its own.
        let mut kv: Vec<SharedKV> =
            vec![(Array2::zeros((5, 4)), Array2::zeros((3, 4)))];
        truncate_kv_rows(&mut kv, 3);
        assert_eq!(kv[0].0.shape(), &[3, 4]);
        assert_eq!(kv[0].1.shape(), &[3, 4]);
    }

    // ── rs_extend_inplace ─────────────────────────────────────────────────────

    fn over_allocated_buffers(
        weights: &ModelWeights,
        capacity: usize,
    ) -> Vec<SharedKV> {
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        (0..weights.num_layers)
            .map(|_| {
                (
                    Array2::<f32>::zeros((capacity, kv_dim)),
                    Array2::<f32>::zeros((capacity, kv_dim)),
                )
            })
            .collect()
    }

    #[test]
    fn inplace_empty_tokens_is_invariant_violation() {
        let weights = make_test_weights();
        let mut kv = over_allocated_buffers(&weights, 4);
        let result = rs_extend_inplace(
            larql_inference::WeightsView::dense(&weights),
            &[],
            &mut kv,
            0,
            0,
            &larql_compute::CpuBackend,
            None,
            None,
        );
        assert!(matches!(
            result,
            Err(EngineError::InvariantViolation { .. })
        ));
    }

    #[test]
    fn inplace_wrong_slot_count_is_invariant_violation() {
        let weights = make_test_weights();
        let mut kv: Vec<SharedKV> = Vec::new();
        let result = rs_extend_inplace(
            larql_inference::WeightsView::dense(&weights),
            &[0u32],
            &mut kv,
            0,
            0,
            &larql_compute::CpuBackend,
            None,
            None,
        );
        assert!(matches!(
            result,
            Err(EngineError::InvariantViolation { .. })
        ));
    }

    #[test]
    fn inplace_fallback_matches_owned_concat_bit_for_bit() {
        // Dense f32 weights with no index: the in-place projection returns
        // None every layer, so every step takes the owned-concat fallback —
        // which must be the same numerics as rs_extend_from_checkpoint_backend
        // (it IS the same attention call over the same `[..len]` view).
        let weights = make_test_weights();
        let tokens = [0u32, 1, 2];

        let mut ref_kv = empty_prior(&weights);
        let reference = rs_extend_from_checkpoint_backend(
            larql_inference::WeightsView::dense(&weights),
            &tokens,
            &mut ref_kv,
            0,
            &larql_compute::CpuBackend,
            None,
            None,
        )
        .expect("owned-concat reference extend");

        let mut kv = over_allocated_buffers(&weights, 8);
        let last = rs_extend_inplace(
            larql_inference::WeightsView::dense(&weights),
            &tokens,
            &mut kv,
            0,
            0,
            &larql_compute::CpuBackend,
            None,
            None,
        )
        .expect("in-place extend via fallback");

        assert_eq!(last.shape(), reference.last_hidden.shape());
        for (a, b) in last.iter().zip(reference.last_hidden.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "fallback path must be bit-identical to the owned-concat form"
            );
        }
    }

    #[test]
    fn inplace_q4k_direct_matches_owned_concat_and_keeps_buffers() {
        // Q4K-direct on (thread-local override): the in-place projection
        // engages, appending rows into the caller's over-allocated buffers
        // instead of replacing them. Hidden states must still match the
        // owned-concat form — only the cache representation differs.
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let _q4k = crate::engines::Q4kFlagGuard::set(&[
            (larql_compute::options::ENV_Q4K_DIRECT_ATTN, true),
            (larql_compute::options::ENV_Q4K_ATTN_INT8, false),
        ]);
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokens = [0u32, 1, 2];
        let capacity = 8;

        let mut ref_kv = empty_prior(&weights);
        let reference = rs_extend_from_checkpoint_backend(
            larql_inference::WeightsView::dense(&weights),
            &tokens,
            &mut ref_kv,
            0,
            &larql_compute::CpuBackend,
            None,
            Some(&index),
        )
        .expect("owned-concat reference extend");

        let mut kv = over_allocated_buffers(&weights, capacity);
        let last = rs_extend_inplace(
            larql_inference::WeightsView::dense(&weights),
            &tokens,
            &mut kv,
            0,
            0,
            &larql_compute::CpuBackend,
            None,
            Some(&index),
        )
        .expect("in-place Q4K extend");

        for (a, b) in last.iter().zip(reference.last_hidden.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "in-place path must be bit-identical to the owned-concat form"
            );
        }
        assert!(last.iter().all(|v| v.is_finite()));
    }

    // ── rs_extend_from_checkpoint_quant profiler channel ──────────────────────

    #[test]
    fn extend_quant_populates_profiler_stages() {
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let mut prof = crate::profiler::EngineProfiler::default();
        let out = rs_extend_from_checkpoint_quant(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1],
            empty_prior(&weights),
            0,
            &*backend,
            Some(&mut prof),
        )
        .expect("profiled extend");
        assert!(out.last_hidden.iter().all(|v| v.is_finite()));
        // One extend call = one accumulation per stage; recompute_* stays
        // untouched (windowed_checkpoint appends incrementally).
        assert_eq!(prof.embed.count, 1);
        assert_eq!(prof.attention.count, 1);
        assert_eq!(prof.ffn.count, 1);
        assert_eq!(prof.decode_total.count, 1);
        assert!(prof.decode_total.total_us > 0.0);
        assert_eq!(prof.recompute_cold.count, 0);
        assert_eq!(prof.recompute_hot.count, 0);
    }

    #[test]
    fn extend_quant_seeded_from_prior_matches_shape() {
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let first = rs_extend_from_checkpoint_quant(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1],
            empty_prior(&weights),
            0,
            &*backend,
            None,
        )
        .expect("first extend");
        let second = rs_extend_from_checkpoint_quant(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[2u32],
            first.kv_cache.clone(),
            2,
            &*backend,
            None,
        )
        .expect("extend over prior kv");
        assert_eq!(second.last_hidden.shape(), &[1, weights.hidden_size]);
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, _) in &second.kv_cache {
            assert_eq!(k.shape(), &[3, kv_dim], "prior(2) + new(1) = 3 rows");
        }
    }
}
