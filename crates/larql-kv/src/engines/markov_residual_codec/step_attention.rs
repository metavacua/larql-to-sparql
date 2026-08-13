//! Resolving one layer's attention output from the codec store's tiers.
//!
//! Twin of [`crate::engines::markov_residual::step_attention`], differing only
//! in the cold tier: this one decodes `cold_encoded` where the sibling slices
//! `cold_residuals`.

use larql_compute::ComputeBackend;
use larql_inference::attention::SharedKV;
use ndarray::{s, Array2};

use crate::engines::markov_residual::recompute_kv;
use crate::engines::markov_residual_codec::store::RsStoreCodec;

/// The hot-K/V cache's state for this step, which is what decides how
/// attention gets its prior.
pub(super) enum HotKv<'a> {
    /// Steady state (step 2+) on an unbounded window: the buffers hold the
    /// full prior K/V and this layer's row is appended into them in place.
    InPlace(&'a mut Vec<SharedKV>),
    /// First step, or a windowed/cold configuration: the prior is recomputed
    /// from the canonical residuals.
    Recompute,
}

/// Derive this layer's `h_post_attn`.
///
/// Returns `None` when the backend declines. Every mutation this makes is to
/// `hot_kv`, a droppable derivative of `rs.stored`, never to canonical state.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_layer_attention(
    weights: larql_inference::WeightsView,
    rs: &RsStoreCodec,
    layer: usize,
    h_new: &Array2<f32>,
    abs_position: usize,
    backend: &dyn ComputeBackend,
    idx_kv: Option<&dyn larql_compute::KvIndex>,
    hot_kv: HotKv<'_>,
    step_new_kv: &mut Vec<SharedKV>,
    cache_eligible: bool,
) -> Option<Array2<f32>> {
    // `stored` is a doubling-capacity buffer (W8.2): logical row count is
    // `hot_len`, not `shape()[0]`.
    let s_hot = rs.hot_len;
    let hot_abs_start = abs_position.saturating_sub(s_hot);

    match hot_kv {
        HotKv::InPlace(bufs) => {
            #[cfg(debug_assertions)]
            debug_assert_hot_kv_parity(weights, rs, layer, bufs, s_hot, hot_abs_start, backend);
            attend_in_place(
                weights,
                &mut bufs[layer],
                layer,
                h_new,
                s_hot,
                abs_position,
                backend,
                idx_kv,
            )
        }
        HotKv::Recompute => {
            let kv_arg = recompute_prior_kv(weights, rs, layer, s_hot, hot_abs_start, backend)?;
            let (h_post_attn, new_kv) =
                larql_inference::attention::run_attention_block_decode_step_auto(
                    weights,
                    h_new,
                    layer,
                    Some(&kv_arg),
                    abs_position,
                    Some(backend),
                    idx_kv,
                )?;
            if cache_eligible {
                step_new_kv.push(new_kv);
            }
            Some(h_post_attn)
        }
    }
}

/// Steady state: append this token's projected+RoPE'd K/V row IN PLACE into
/// the doubling-capacity buffer and attend over the `[..s_hot+1]` views — no
/// per-step O(ctx) owned concat (O(L) total cache copy vs O(L²)). See the
/// markov twin for the rationale.
#[allow(clippy::too_many_arguments)]
fn attend_in_place(
    weights: larql_inference::WeightsView,
    buf: &mut SharedKV,
    layer: usize,
    h_new: &Array2<f32>,
    s_hot: usize,
    abs_position: usize,
    backend: &dyn ComputeBackend,
    idx_kv: Option<&dyn larql_compute::KvIndex>,
) -> Option<Array2<f32>> {
    let (k_buf, v_buf) = buf;
    let inplace = if crate::engines::markov_residual::compute::markov_inplace_kv_enabled() {
        larql_inference::attention::run_attention_block_decode_step_auto_inplace(
            weights,
            h_new,
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
        Some(h) => Some(h),
        None => {
            // Q4K-direct off (flags-off parity) or no attn bytes: owned concat
            // over the buffer view, then replace. Bit-identical to the legacy
            // borrow path.
            let prior: SharedKV = (
                k_buf.slice(s![..s_hot, ..]).to_owned(),
                v_buf.slice(s![..s_hot, ..]).to_owned(),
            );
            let (h, new_kv) = larql_inference::attention::run_attention_block_decode_step_auto(
                weights,
                h_new,
                layer,
                Some(&prior),
                abs_position,
                Some(backend),
                idx_kv,
            )?;
            *k_buf = new_kv.0;
            *v_buf = new_kv.1;
            Some(h)
        }
    }
}

/// First step (cache `None` → seed) or windowed/cold tier.
fn recompute_prior_kv(
    weights: larql_inference::WeightsView,
    rs: &RsStoreCodec,
    layer: usize,
    s_hot: usize,
    hot_abs_start: usize,
    backend: &dyn ComputeBackend,
) -> Option<SharedKV> {
    let h_hot = &rs.stored[layer];
    if let Some(cold_kv) = &rs.cold_kv {
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
        return Some((k_combined, v_combined));
    }

    let (h_full, full_abs_start) = match &rs.cold_encoded {
        Some(cold_layers) if cold_layers[layer].n_positions > 0 => {
            let decoded = cold_layers[layer].decode(rs.codec);
            let n_cold = decoded.shape()[0];
            let hidden = h_hot.shape()[1];
            let mut combined = Array2::<f32>::zeros((n_cold + s_hot, hidden));
            combined.slice_mut(s![..n_cold, ..]).assign(&decoded);
            combined.slice_mut(s![n_cold.., ..]).assign(h_hot);
            (combined, rs.cold_abs_start)
        }
        _ => (h_hot.clone(), hot_abs_start),
    };
    recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)
}

/// f32-path parity gate only — the Q4K-direct route has its own oracles (the
/// compute-level bit-identity test + the engine A/B).
#[cfg(debug_assertions)]
fn debug_assert_hot_kv_parity(
    weights: larql_inference::WeightsView,
    rs: &RsStoreCodec,
    layer: usize,
    bufs: &[SharedKV],
    s_hot: usize,
    hot_abs_start: usize,
    backend: &dyn ComputeBackend,
) {
    /// Largest per-element f32 gap tolerated between the cached prior K/V and
    /// a fresh recompute.
    const MAX_CACHE_DRIFT: f32 = 1e-2;

    if larql_compute::options::q4k_direct_attn_enabled() {
        return;
    }
    let (k_buf, v_buf) = &bufs[layer];
    let h_logical = rs.stored[layer].slice(s![..s_hot, ..]).to_owned();
    let Some((rk, rv)) = recompute_kv(weights, &h_logical, layer, hot_abs_start, backend, None)
    else {
        return;
    };
    let max_gap = |buf: &Array2<f32>, fresh: &Array2<f32>| {
        buf.slice(s![..s_hot, ..])
            .iter()
            .zip(fresh.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let kd = max_gap(k_buf, &rk);
    let vd = max_gap(v_buf, &rv);
    debug_assert!(kd < MAX_CACHE_DRIFT, "codec hot_kv K cache diverged: {kd}");
    debug_assert!(vd < MAX_CACHE_DRIFT, "codec hot_kv V cache diverged: {vd}");
}
