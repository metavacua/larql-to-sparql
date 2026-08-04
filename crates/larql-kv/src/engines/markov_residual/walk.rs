//! Q4K helpers — attention dequantisation re-export + WalkFfn-backed
//! forward paths.
//!
//! `ensure_attn_tensors_dequantised` moved to
//! [`larql_inference::vindex::dequant`] (2026-05-16) so the
//! `KvDispatch` trait impls in `larql-inference::kv_dispatch::*` can
//! call it without a `larql-kv → larql-inference → larql-kv` cycle.
//! Re-exported here to keep existing call sites compiling.

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::compute::{last_row, recompute_kv};
use super::prefill::RsPrefillResult;
use super::store::RsStore;
use crate::profiler::EngineProfiler;
use larql_inference::attention::run_attention_with_kv_backend;
use larql_inference::attention::SharedKV;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};

/// Re-export — see [`larql_inference::vindex::dequant::ensure_attn_tensors_dequantised`].
pub use larql_inference::vindex::ensure_attn_tensors_dequantised;

/// Prefill using `WalkFfn` (Q4K FFN) instead of `BackendFfn` (f32 FFN).
pub(super) fn rs_prefill_walk(
    weights: larql_inference::WeightsView,
    index: &VectorIndex,
    token_ids: &[u32],
    max_window: Option<usize>,
    backend: &dyn ComputeBackend,
) -> RsPrefillResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(&weights, token_ids);
    // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    // Hoist WalkFfn construction out of the per-layer loop. Previously
    // this rebuilt the WalkFfn 34 times per prefill (once per layer);
    // now once total. WalkFfn carries no per-layer state — it's the
    // gate-index + backend pair, both stable across the loop.
    let walk_ffn =
        WalkFfn::from_config(weights.canonical(), index, WalkFfnConfig::dense(num_layers))
            .with_backend(backend);

    // Capture per-layer K/V from each layer's attention block. These
    // are *already computed* by the forward pass; previously discarded
    // and re-derived from residuals on every decode step (W2 measured
    // 80% of decode time spent on this redundant recompute). Stashing
    // them here means decode_step appends one row per layer instead
    // of recomputing the entire hot tier.
    let mut hot_kv_captured: Vec<SharedKV> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, k, v) = run_attention_with_kv_backend(weights, &h, layer, be, None)
            .expect("attention failed during MarkovRS Q4K prefill");
        hot_kv_captured.push((k, v));
        let (h_post_ffn, _) = run_ffn(&weights, &h_post_attn, layer, &walk_ffn, false);
        h = crate::engines::apply_ple_and_layer_scalar(
            &weights,
            &h_post_ffn,
            layer,
            ple_inputs.get(layer),
        );
    }

    let mut rs = RsStore {
        hot_len: stored.first().map_or(0, |s| s.shape()[0]),
        stored,
        cold_residuals: None,
        cold_kv: None,
        hot_kv: Some(hot_kv_captured),
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
        cold_len: 0,
    };

    // Build pre-clip evicted-rows lookup so we can move the evicted
    // top of `hot_kv` directly into `cold_kv` without calling
    // `recompute_kv` (the K/V is already correct — we just need to
    // slice it).
    let pre_clip_hot_rows: Vec<usize> = if rs.hot_kv.is_some() {
        let window = max_window.unwrap_or(usize::MAX);
        rs.stored
            .iter()
            .map(|s| s.shape()[0].saturating_sub(window))
            .collect()
    } else {
        Vec::new()
    };
    let evicted_hot_kv = rs
        .hot_kv
        .as_ref()
        .filter(|_| pre_clip_hot_rows.iter().any(|&n| n > 0))
        .and_then(|hot_kv| RsStore::snapshot_evicted_hot_kv(hot_kv, &pre_clip_hot_rows));

    let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut cold);
    }
    rs.finalise_hot_len_after_clip();
    if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_kv: Vec<SharedKV> = if let Some(evicted) = evicted_hot_kv {
            // Fast path: reuse the K/V we just computed during the
            // forward pass. No `recompute_kv` call needed — the
            // evicted slices ARE the cold K/V.
            evicted
        } else {
            // Fallback: forward pass didn't capture K/V (shouldn't
            // happen under current code, kept for safety).
            (0..num_layers)
                .map(|layer| {
                    recompute_kv(weights, &cold[layer], layer, 0, backend, Some(index))
                        .expect("cold K/V pre-computation failed")
                })
                .collect()
        };
        // 2026-05-19 audit fix: route through the doubling-capacity
        // helper so cold_len is correctly initialised and subsequent
        // overflows append in amortised O(1) instead of O(N).
        rs.append_cold_overflow(cold, Some(cold_kv));
        rs.cold_abs_start = 0;
    }
    let window_tokens = rs.window_tokens();
    let memory_bytes = rs.memory_bytes();
    RsPrefillResult {
        hidden: last_row(&h),
        store: rs,
        memory_bytes,
        window_tokens,
    }
}

/// Decode step using `WalkFfn` (Q4K FFN). Pass `Some(profiler)` to
/// accumulate per-stage wall-clock; pass `None` for the unprofiled
/// path. Sibling of [`super::compute::rs_decode_step_inner`] for the
/// Q4K side.
pub(super) fn rs_decode_step_walk(
    weights: larql_inference::WeightsView,
    index: &VectorIndex,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
    mut profiler: Option<&mut EngineProfiler>,
) -> Option<(Array2<f32>, RsStore)> {
    use ndarray::s;
    use std::time::Instant;

    // Verbose env-var instrumentation is kept as an ad-hoc debug
    // channel (prints per-step lines to stderr). The structured
    // `profiler` accumulator is the supported path for
    // `larql bench --profile`.
    let instrument = std::env::var("LARQL_INSTRUMENT_MARKOV").is_ok();
    let timing = profiler.is_some() || instrument;

    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let t_step = if timing { Some(Instant::now()) } else { None };
    let t_embed_start = t_step;
    let mut h_new = embed_tokens_pub(&weights, &[new_token_id]);
    let embed_us = t_embed_start
        .map(|t| t.elapsed().as_secs_f64() * 1e6)
        .unwrap_or(0.0);
    // PLE inputs are per-token — recompute for this single-token decode
    // step, matching the legacy `kv_decode_step_run` recipe exactly.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h_new, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    // Hoist WalkFfn out of the per-layer loop — see note in
    // `rs_prefill_walk`. Was 34× construction per decode step.
    let walk_ffn =
        WalkFfn::from_config(weights.canonical(), index, WalkFfnConfig::dense(num_layers))
            .with_backend(backend);

    // Per-stage accumulators. With W2 caching landed, both
    // `recompute_*` timings should be near zero for cached-path decode
    // steps — they only fire on the fallback when hot_kv was dropped
    // (e.g. via_executor path doesn't cache yet, or post-overflow
    // first-step before cache rebuild). `recompute_hot` now also
    // measures the cheap "concat cached cold + cached hot" path.
    let mut recompute_cold_us = 0.0f64;
    let mut recompute_hot_us = 0.0f64;
    let mut attention_us = 0.0f64;
    let mut ffn_us = 0.0f64;
    let mut concat_us = 0.0f64;
    let mut attn_helper_hits = 0usize;
    let mut attn_helper_misses = 0usize;
    let mut s_hot_first_layer = 0usize;

    // Per-layer new hot_kv slices (post-attention), built up across
    // the layer loop and committed to the store at the end.
    let mut new_hot_kvs: Vec<SharedKV> = Vec::with_capacity(num_layers);

    // The cold tier must be representable in K/V form before the
    // cached-hot_kv fast path may fire: `cold_kv` can be legitimately
    // absent while `cold_residuals` holds rows (dispatch fallback;
    // eviction into a cold tier whose K/V could not be seeded — see
    // `append_cold_overflow`). Branch 1 below reads only `cold_kv`, so
    // without a re-seed it would attend hot-only and silently drop the
    // cold context from every subsequent step. Re-derive the cold K/V
    // from the residuals at `cold_abs_start` once; later steps pay a
    // memcpy.
    let mut rs = rs;
    if rs.hot_kv.is_some() && rs.cold_kv.is_none() && rs.cold_len > 0 {
        if let Some(cold) = &rs.cold_residuals {
            let t_cold = if timing { Some(Instant::now()) } else { None };
            let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
            for (layer, buf) in cold.iter().enumerate() {
                let cold_logical = buf.slice(s![..rs.cold_len, ..]).to_owned();
                cold_kv.push(recompute_kv(
                    weights,
                    &cold_logical,
                    layer,
                    rs.cold_abs_start,
                    backend,
                    Some(index),
                )?);
            }
            if let Some(t) = t_cold {
                recompute_cold_us += t.elapsed().as_secs_f64() * 1e6;
            }
            rs.cold_kv = Some(cold_kv);
        }
    }

    for layer in 0..num_layers {
        // `stored` / `hot_kv` / cold buffers are doubling-capacity (W8.2 /
        // 2026-05-19 store migration): logical row counts are `hot_len` /
        // `cold_len`, never `shape()[0]`. Capacity rows past the logical
        // length are zeros — attending them poisons the softmax (score-0
        // phantom rows) and shifts the hot rows' RoPE offsets.
        let s_hot = rs.hot_len;
        let h_hot = rs.stored[layer].slice(s![..s_hot, ..]);
        if layer == 0 {
            s_hot_first_layer = s_hot;
        }
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        // Number of cold rows included in this layer's attention prior —
        // the offset the post-attention hot_kv capture must slice at.
        // Set per branch below (path 3 includes `cold_len` cold rows even
        // though `cold_kv` is None).
        let cold_rows_in_prior;

        // Build prior_kv. Three paths, in order of preference:
        //   1. Cached hot_kv (+ optional cached cold_kv) — concat is
        //      a memcpy; no projection work. **W2 fast path.**
        //   2. Cached cold_kv only — recompute hot from h_hot, concat
        //      with cold. (Hot K/V wasn't captured; falls back to the
        //      pre-W2 behaviour.)
        //   3. Neither cached — recompute everything from residuals.
        //      Slowest path; fires on first decode after overflow
        //      eviction (cache rebuilds during this step's tail
        //      processing) or on the via_executor path which doesn't
        //      capture K/V yet.
        let (k_full, v_full) =
            if let (Some(hot_kv), maybe_cold) = (rs.hot_kv.as_ref(), rs.cold_kv.as_ref()) {
                let (k_hot_buf, v_hot_buf) = &hot_kv[layer];
                let k_hot = k_hot_buf.slice(s![..s_hot, ..]);
                let v_hot = v_hot_buf.slice(s![..s_hot, ..]);
                let t_concat = if timing { Some(Instant::now()) } else { None };
                let pair = if let Some(cold_kv) = maybe_cold {
                    let (k_cold_buf, v_cold_buf) = &cold_kv[layer];
                    let c = rs.cold_len;
                    let kv_dim = k_cold_buf.shape()[1];
                    cold_rows_in_prior = c;
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
                    cold_rows_in_prior = 0;
                    (k_hot.to_owned(), v_hot.to_owned())
                };
                if let Some(t) = t_concat {
                    concat_us += t.elapsed().as_secs_f64() * 1e6;
                }
                pair
            } else if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold_buf, v_cold_buf) = &cold_kv[layer];
                let t_hot = if timing { Some(Instant::now()) } else { None };
                let h_hot_owned = h_hot.to_owned();
                let (k_hot, v_hot) = recompute_kv(
                    weights,
                    &h_hot_owned,
                    layer,
                    hot_abs_start,
                    backend,
                    Some(index),
                )?;
                if let Some(t) = t_hot {
                    recompute_hot_us += t.elapsed().as_secs_f64() * 1e6;
                }
                let t_concat = if timing { Some(Instant::now()) } else { None };
                let c = rs.cold_len;
                let kv_dim = k_cold_buf.shape()[1];
                cold_rows_in_prior = c;
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
                if let Some(t) = t_concat {
                    concat_us += t.elapsed().as_secs_f64() * 1e6;
                }
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
                cold_rows_in_prior = s_cold;
                let t_cold = if timing { Some(Instant::now()) } else { None };
                let pair = recompute_kv(
                    weights,
                    &h_full,
                    layer,
                    full_abs_start,
                    backend,
                    Some(index),
                )?;
                if let Some(t) = t_cold {
                    recompute_cold_us += t.elapsed().as_secs_f64() * 1e6;
                }
                pair
            };

        new_stored.push(h_new.clone());

        let t_attn = if timing { Some(Instant::now()) } else { None };
        let kv_pair = (k_full, v_full);
        let native_result = larql_inference::vindex::attention_decode_step_native(
            weights.canonical(),
            index,
            backend,
            &h_new,
            layer,
            Some(&kv_pair),
            abs_position,
        );
        if instrument {
            if native_result.is_some() {
                attn_helper_hits += 1;
            } else {
                attn_helper_misses += 1;
            }
        }
        let (h_post_attn, new_kv_full) = native_result.or_else(|| {
            larql_inference::attention::run_attention_block_decode_step_auto(
                weights,
                &h_new,
                layer,
                Some(&kv_pair),
                abs_position,
                Some(backend),
                Some(index as &dyn larql_compute::KvIndex),
            )
        })?;
        if let Some(t) = t_attn {
            attention_us += t.elapsed().as_secs_f64() * 1e6;
        }

        // Capture the new hot_kv slice for this layer.
        // `new_kv_full` = (cold_prior ++ hot ++ new_row). Slicing off
        // `cold_rows_in_prior` (the cold portion, unchanged this step)
        // leaves `hot ++ new_row`, which is exactly the new hot K/V to
        // cache for the next step. The offset must count the cold rows
        // actually in the prior — path 3 folds `cold_len` recomputed
        // cold rows in even though `cold_kv` is None; slicing from 0
        // there would cache cold rows as hot, desyncing `hot_kv` from
        // `stored`/`hot_len` (the store invariant, store.rs).
        let new_hot_kv = (
            new_kv_full.0.slice(s![cold_rows_in_prior.., ..]).to_owned(),
            new_kv_full.1.slice(s![cold_rows_in_prior.., ..]).to_owned(),
        );
        new_hot_kvs.push(new_hot_kv);

        let t_ffn = if timing { Some(Instant::now()) } else { None };
        // Try the production-path native-quantised FFN helper first —
        // direct Q4K/Q6K matvec on the vindex's compact gate/up/down
        // bytes. Falls back to WalkFfn (and then dense WeightFfn) when
        // the backend doesn't have native quant support or the layer
        // isn't direct-matvec-eligible.
        // Both branches return the bare post-FFN hidden —
        // `ffn_decode_step_native` is also `moe_ffn_block_cpu`'s pre-PLE
        // dense slab — so the PLE + layer_scalar tail applies to either.
        let h_post_ffn = larql_inference::vindex::ffn_decode_step_native(
            weights.canonical(),
            index,
            backend,
            &h_post_attn,
            layer,
        )
        .unwrap_or_else(|| {
            let (h, _) = run_ffn(&weights, &h_post_attn, layer, &walk_ffn, false);
            h
        });
        let h_out = crate::engines::apply_ple_and_layer_scalar(
            &weights,
            &h_post_ffn,
            layer,
            ple_inputs.get(layer),
        );
        if let Some(t) = t_ffn {
            ffn_us += t.elapsed().as_secs_f64() * 1e6;
        }
        h_new = h_out;
    }

    if instrument {
        let total_ms =
            (embed_us + recompute_cold_us + recompute_hot_us + concat_us + attention_us + ffn_us)
                / 1e3;
        eprintln!(
            "[markov-rs/decode] s_hot={s_hot_first_layer} embed={:.2}ms \
             recompute_cold={:.2}ms recompute_hot={:.2}ms concat={:.2}ms \
             attention={:.2}ms ffn={:.2}ms total={:.2}ms \
             (attn_helper hits/miss={attn_helper_hits}/{attn_helper_misses})",
            embed_us / 1e3,
            recompute_cold_us / 1e3,
            recompute_hot_us / 1e3,
            concat_us / 1e3,
            attention_us / 1e3,
            ffn_us / 1e3,
            total_ms,
        );
    }

    if let (Some(prof), Some(t_step)) = (profiler.as_mut(), t_step) {
        prof.embed.total_us += embed_us;
        prof.embed.count += 1;
        prof.recompute_cold.total_us += recompute_cold_us;
        prof.recompute_cold.count += 1;
        prof.recompute_hot.total_us += recompute_hot_us;
        prof.recompute_hot.count += 1;
        prof.attention.total_us += attention_us;
        prof.attention.count += 1;
        prof.ffn.total_us += ffn_us;
        prof.ffn.count += 1;
        prof.decode_total.total_us += t_step.elapsed().as_secs_f64() * 1e6;
        prof.decode_total.count += 1;
    }

    let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
        // Logical rows only — `stored` may be a capacity-padded buffer
        // (dispatch-prefill fallback).
        let s_old = rs.hot_len;
        let hidden_dim = stored.shape()[1];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
        combined
            .slice_mut(s![..s_old, ..])
            .assign(&stored.slice(s![..s_old, ..]));
        combined.slice_mut(s![s_old.., ..]).assign(new_row);
        updated_stored.push(combined);
    }

    let mut updated_rs = RsStore {
        hot_len: updated_stored.first().map_or(0, |s| s.shape()[0]),
        stored: updated_stored,
        cold_residuals: rs.cold_residuals,
        cold_kv: rs.cold_kv,
        cold_len: rs.cold_len,
        // Commit the new hot K/V slices (one row appended per layer
        // vs the pre-decode cache). Becomes the prior K/V for the
        // next step's fast path. Will be sliced by `clip_layer`
        // below if the window cap is exceeded.
        hot_kv: Some(new_hot_kvs),
        cold_abs_start: rs.cold_abs_start,
        next_position: abs_position + 1,
        max_window: rs.max_window,
    };

    // Pre-clip snapshot of how many hot_kv rows each layer would
    // evict — used below to move evicted K/V directly into cold_kv
    // (vs the pre-W2 behaviour of clearing cold_kv and recomputing
    // from cold_residuals on the next step).
    let pre_clip_evicted_rows: Vec<usize> = if updated_rs.hot_kv.is_some() {
        let window = updated_rs.max_window.unwrap_or(usize::MAX);
        updated_rs
            .stored
            .iter()
            .map(|s| s.shape()[0].saturating_sub(window))
            .collect()
    } else {
        Vec::new()
    };
    let evicted_hot_kv = updated_rs
        .hot_kv
        .as_ref()
        .filter(|_| pre_clip_evicted_rows.iter().any(|&n| n > 0))
        .and_then(|hot_kv| RsStore::snapshot_evicted_hot_kv(hot_kv, &pre_clip_evicted_rows));

    let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        updated_rs.clip_layer(layer, &mut overflow);
    }
    updated_rs.finalise_hot_len_after_clip();
    // 2026-05-19 audit fix: geometric-capacity cold append. The
    // evicted K/V is the K/V projection of the evicted residuals at
    // their original RoPE positions, so we hand it straight to
    // `append_cold_overflow` without a recompute_kv call (W2 fast
    // path). When `evicted_hot_kv` is `None` (via_executor or pre-W2
    // codepaths), the helper invalidates cold_kv so the next step
    // rebuilds K/V from cold_residuals.
    updated_rs.append_cold_overflow(overflow, evicted_hot_kv);

    Some((last_row(&h_new), updated_rs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::test_utils::{make_test_vindex, make_test_weights};

    #[test]
    fn prefill_walk_returns_finite_hidden_and_full_window_store() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let result = rs_prefill_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1, 2],
            None,
            &CpuBackend,
        );
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
        assert!(result.store.cold_residuals.is_none());
        assert!(result.store.cold_kv.is_none());
        assert!(result.store.hot_kv.is_some());
        assert_eq!(result.window_tokens, 3);
        assert!(result.memory_bytes > 0);
    }

    #[test]
    fn prefill_walk_with_overflow_populates_cold_tier_from_evicted_hot_kv() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let result = rs_prefill_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            &CpuBackend,
        );
        assert!(result.store.cold_residuals.is_some());
        assert!(result.store.cold_kv.is_some());
        // Window-clipped, but cold tier captured the two evicted rows.
        assert_eq!(result.window_tokens, 2);
        // 2026-05-19 audit fix: cold_residuals[l].shape()[0] is now the
        // doubling capacity, not the logical row count. Use `cold_len`.
        assert_eq!(result.store.cold_len, 2);
    }

    #[test]
    fn decode_walk_extends_position_and_returns_finite() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1],
            None,
            &CpuBackend,
        );
        assert_eq!(prefill.store.next_position, 2);
        let (h, rs2) = rs_decode_step_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            2,
            prefill.store,
            &CpuBackend,
            None,
        )
        .unwrap();
        assert_eq!(rs2.next_position, 3);
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_walk_with_profiler_accumulates_all_stage_timings() {
        // Window=2 with 4-token prompt → cold_kv populated. The decode
        // step exercises the "hot_kv + cold_kv" fast-path concat, plus
        // the timing accumulators on every per-stage block.
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            &CpuBackend,
        );
        let mut prof = EngineProfiler::default();
        let (h, _) = rs_decode_step_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            4,
            prefill.store,
            &CpuBackend,
            Some(&mut prof),
        )
        .unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(prof.decode_total.count, 1);
        assert_eq!(prof.attention.count, 1);
        assert_eq!(prof.ffn.count, 1);
        assert_eq!(prof.embed.count, 1);
        assert_eq!(prof.recompute_cold.count, 1);
        assert_eq!(prof.recompute_hot.count, 1);
    }

    #[test]
    fn decode_walk_recomputes_hot_when_hot_kv_dropped_with_cold_kv_present() {
        // Force the "cached cold_kv only" middle path: drop the hot_kv
        // cache but keep cold_kv. Decode must recompute the hot K/V
        // from h_hot and concat with the cached cold K/V.
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            &CpuBackend,
        );
        let mut store = prefill.store;
        store.hot_kv = None;
        let mut prof = EngineProfiler::default();
        let (h, rs2) = rs_decode_step_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            4,
            store,
            &CpuBackend,
            Some(&mut prof),
        )
        .unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // hot_kv is repopulated on every decode step.
        assert!(rs2.hot_kv.is_some());
    }

    #[test]
    fn decode_walk_recomputes_full_when_no_caches_and_cold_residuals_present() {
        // Drop both hot_kv and cold_kv, leaving only the raw
        // cold_residuals behind. Drives the "neither cached" else arm
        // that concatenates cold residuals with h_hot before
        // recomputing the K/V.
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            &CpuBackend,
        );
        let mut store = prefill.store;
        store.hot_kv = None;
        store.cold_kv = None;
        let (h, _) = rs_decode_step_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            4,
            store,
            &CpuBackend,
            None,
        )
        .unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    // ── Doubling-capacity regression tests (2026-05-19 store migration) ──
    //
    // `cold_kv` / `cold_residuals` are capacity-padded buffers whose
    // logical row count is `cold_len` (`store.rs`). A decode that reads
    // `shape()[0]` instead attends zero-padded phantom K/V rows and
    // recomputes hot K at shifted RoPE offsets. Each test decodes the
    // same logical state twice — once over the padded buffers, once over
    // buffers trimmed to exactly `cold_len` (the layout the
    // pre-migration code assumed) — and requires bit-identical output.

    /// Window / prompt sized so the prefill overflows: `cold_len = 2`
    /// while the cold buffers are allocated at the doubling minimum
    /// (8 rows), i.e. capacity strictly exceeds the logical length.
    const PADDED_WINDOW: usize = 2;
    const PADDED_PROMPT: [u32; 4] = [0, 1, 2, 3];
    const PADDED_NEXT_TOKEN: u32 = 4;
    /// Hidden-state tolerance between the cached-hot-K/V path and the
    /// recompute-from-residuals path (same bound as the W2 parity test
    /// in engine.rs — the two paths project K/V through different code
    /// but must agree to fp rounding).
    const CACHED_VS_RECOMPUTE_TOL: f32 = 1e-4;

    fn prefill_padded_store(
        weights: &larql_inference::ModelWeights,
        index: &larql_vindex::VectorIndex,
    ) -> RsStore {
        let store = rs_prefill_walk(
            larql_inference::WeightsView::dense(weights),
            index,
            &PADDED_PROMPT,
            Some(PADDED_WINDOW),
            &CpuBackend,
        )
        .store;
        // Self-check: the fixture must actually be capacity-padded,
        // otherwise these tests stop pinning the bug.
        assert!(store.cold_kv.as_ref().unwrap()[0].0.shape()[0] > store.cold_len);
        assert!(store.cold_residuals.as_ref().unwrap()[0].shape()[0] > store.cold_len);
        store
    }

    /// Rebuild the cold buffers at exactly `cold_len` rows. Decode over
    /// this store is correct under both the old and the fixed code, so
    /// it is the parity reference for the capacity-padded twin.
    fn with_trimmed_cold(mut store: RsStore) -> RsStore {
        use ndarray::s;
        let c = store.cold_len;
        if let Some(cold) = store.cold_residuals.as_mut() {
            for layer in cold.iter_mut() {
                *layer = layer.slice(s![..c, ..]).to_owned();
            }
        }
        if let Some(kv) = store.cold_kv.as_mut() {
            for (k, v) in kv.iter_mut() {
                *k = k.slice(s![..c, ..]).to_owned();
                *v = v.slice(s![..c, ..]).to_owned();
            }
        }
        store
    }

    fn decode_once(
        weights: &larql_inference::ModelWeights,
        index: &larql_vindex::VectorIndex,
        store: RsStore,
    ) -> (Array2<f32>, RsStore) {
        rs_decode_step_walk(
            larql_inference::WeightsView::dense(weights),
            index,
            PADDED_NEXT_TOKEN,
            store,
            &CpuBackend,
            None,
        )
        .expect("decode")
    }

    fn assert_bits_eq(a: &Array2<f32>, b: &Array2<f32>, ctx: &str) {
        let a_bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
        let b_bits: Vec<u32> = b.iter().map(|v| v.to_bits()).collect();
        assert_eq!(a_bits, b_bits, "{ctx}");
    }

    #[test]
    fn decode_walk_padded_cold_kv_matches_trimmed_reference_cached_path() {
        // Fast path (hot_kv + cold_kv cached): the cold concat must use
        // `cold_len`, not the buffer capacity — otherwise 6 phantom
        // zero-K/V rows join the softmax.
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let padded = prefill_padded_store(&weights, &index);
        let trimmed = with_trimmed_cold(prefill_padded_store(&weights, &index));
        let (h_padded, _) = decode_once(&weights, &index, padded);
        let (h_trimmed, _) = decode_once(&weights, &index, trimmed);
        assert_bits_eq(
            &h_padded,
            &h_trimmed,
            "cached path attended capacity slack instead of cold_len rows",
        );
    }

    #[test]
    fn decode_walk_padded_cold_kv_matches_trimmed_reference_hot_dropped() {
        // Middle path (cold_kv only): same cold_len contract when the
        // hot K/V is recomputed.
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let mut padded = prefill_padded_store(&weights, &index);
        padded.hot_kv = None;
        let mut trimmed = with_trimmed_cold(prefill_padded_store(&weights, &index));
        trimmed.hot_kv = None;
        let (h_padded, _) = decode_once(&weights, &index, padded);
        let (h_trimmed, _) = decode_once(&weights, &index, trimmed);
        assert_bits_eq(
            &h_padded,
            &h_trimmed,
            "cold_kv-only path attended capacity slack instead of cold_len rows",
        );
    }

    #[test]
    fn decode_walk_padded_cold_residuals_match_trimmed_reference_no_caches() {
        // Slow path (neither cache): reading `shape()[0]` off the padded
        // cold_residuals both injects phantom rows AND shifts the hot
        // rows' RoPE offsets (`full_abs_start + padded_cold_rows`).
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let mut padded = prefill_padded_store(&weights, &index);
        padded.hot_kv = None;
        padded.cold_kv = None;
        let mut trimmed = with_trimmed_cold(prefill_padded_store(&weights, &index));
        trimmed.hot_kv = None;
        trimmed.cold_kv = None;
        let (h_padded, _) = decode_once(&weights, &index, padded);
        let (h_trimmed, _) = decode_once(&weights, &index, trimmed);
        assert_bits_eq(
            &h_padded,
            &h_trimmed,
            "no-cache path recomputed over capacity slack / wrong RoPE offsets",
        );
    }

    #[test]
    fn decode_walk_no_cache_step_leaves_hot_kv_usable_by_next_cached_step() {
        // Path-3 hot_kv capture: with `s_cold` cold rows folded into the
        // attention prior (cold_kv = None), the captured hot K/V must
        // slice past them. Caching cold+hot+new as "hot" desyncs
        // `hot_kv` from `stored`/`hot_len`, so the NEXT step's cached
        // decode attends the wrong rows. Pin by value: step 2 cached
        // must match step 2 recomputed-from-residuals.
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);

        let run_step2 = |drop_caches_before_step2: bool| -> Array2<f32> {
            let mut store = prefill_padded_store(&weights, &index);
            store.hot_kv = None;
            store.cold_kv = None;
            // Step 1 takes the no-cache path in both runs → identical
            // state going into step 2.
            let (_, mut rs2) = decode_once(&weights, &index, store);
            if drop_caches_before_step2 {
                rs2.hot_kv = None;
                rs2.cold_kv = None;
            }
            let (h2, _) = rs_decode_step_walk(
                larql_inference::WeightsView::dense(&weights),
                &index,
                PADDED_NEXT_TOKEN + 1,
                rs2,
                &CpuBackend,
                None,
            )
            .expect("step 2");
            h2
        };

        let h2_cached = run_step2(false);
        let h2_recompute = run_step2(true);
        for (a, b) in h2_cached.iter().zip(h2_recompute.iter()) {
            assert!(
                (a - b).abs() < CACHED_VS_RECOMPUTE_TOL,
                "step-2 cached path diverged from recompute reference: \
                 cached={a}, recompute={b} — hot_kv captured cold rows as hot"
            );
        }
    }

    #[test]
    fn decode_walk_first_overflow_initializes_cold_residuals() {
        // Prefill without overflow, then decode past the window cap.
        // First overflow exercises the `None` arm of
        // `updated_rs.cold_residuals.as_mut()` — fresh cold tier
        // initialised from the evicted block.
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            &[0u32, 1],
            Some(2),
            &CpuBackend,
        );
        assert!(prefill.store.cold_residuals.is_none());
        let (_, rs2) = rs_decode_step_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            2,
            prefill.store,
            &CpuBackend,
            None,
        )
        .unwrap();
        assert!(rs2.cold_residuals.is_some());
        // 2026-05-19 audit fix: shape()[0] is doubling capacity. Use cold_len.
        assert_eq!(rs2.cold_len, 1);
        assert!(rs2.cold_kv.is_some());
    }
}
