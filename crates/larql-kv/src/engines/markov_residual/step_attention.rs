//! Resolving one layer's attention output from the residual store's tiers.
//!
//! The decode step's per-layer work splits in two: derive `h_post_attn` from
//! whatever K/V the store can offer (this file), then run the FFN over it
//! (see [`super::step`]). Only the first half has to know about the store's
//! three shapes — a warm in-place hot cache, a cold tier that must be
//! recombined, and the first step, which has neither.

use larql_compute::ComputeBackend;
use larql_inference::attention::SharedKV;
use ndarray::{s, Array2};

use super::compute::recompute_kv;
use super::store::RsStore;

/// Per-stage microsecond accumulators for one decode step.
///
/// Grouped rather than passed as four `&mut f64` so a new stage does not
/// widen every signature between here and the profiler.
#[derive(Default)]
pub(super) struct StepTimings {
    pub(super) recompute_cold_us: f64,
    pub(super) recompute_hot_us: f64,
    pub(super) attention_us: f64,
    pub(super) ffn_us: f64,
}

impl StepTimings {
    /// Time `f` into `slot` when `enabled`, else just run it.
    ///
    /// The timing branch is written once here because the alternative — an
    /// `Instant::now()` pair around every stage — is what made the original
    /// loop unreadable, and a mistimed stage silently misattributes the
    /// engine's cost.
    pub(super) fn measure<T>(enabled: bool, slot: &mut f64, f: impl FnOnce() -> T) -> T {
        /// The profiler reports microseconds; `elapsed` gives seconds.
        const MICROS_PER_SECOND: f64 = 1e6;

        if !enabled {
            return f();
        }
        let start = std::time::Instant::now();
        let out = f();
        *slot += start.elapsed().as_secs_f64() * MICROS_PER_SECOND;
        out
    }
}

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
/// Returns `None` when the backend declines — the caller turns that into a
/// typed failure. Every mutation this makes is to `hot_kv`, a droppable
/// derivative of `rs.stored`, and never to the store's canonical state.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_layer_attention(
    weights: larql_inference::WeightsView,
    rs: &RsStore,
    layer: usize,
    h_new: &Array2<f32>,
    abs_position: usize,
    backend: &dyn ComputeBackend,
    idx_kv: Option<&dyn larql_compute::KvIndex>,
    hot_kv: HotKv<'_>,
    step_new_kv: &mut Vec<SharedKV>,
    cache_eligible: bool,
    timed: bool,
    timings: &mut StepTimings,
) -> Option<Array2<f32>> {
    // `stored` is a doubling-capacity buffer (W8.2): the logical row count is
    // `hot_len`, not `shape()[0]` (see RsStore docs).
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
                timed,
                timings,
            )
        }
        HotKv::Recompute => {
            let kv_arg = recompute_prior_kv(
                weights,
                rs,
                layer,
                s_hot,
                hot_abs_start,
                backend,
                timed,
                timings,
            )?;
            let (h_post_attn, new_kv) =
                StepTimings::measure(timed, &mut timings.attention_us, || {
                    larql_inference::attention::run_attention_block_decode_step_auto(
                        weights,
                        h_new,
                        layer,
                        Some(&kv_arg),
                        abs_position,
                        Some(backend),
                        idx_kv,
                    )
                })?;
            // The attention step already projected the new token's K/V
            // (RoPE'd) — free; collect it to seed `hot_kv` for the in-place
            // steady state.
            if cache_eligible {
                step_new_kv.push(new_kv);
            }
            Some(h_post_attn)
        }
    }
}

/// Steady state: append this token's projected+RoPE'd row into the layer's
/// doubling-capacity buffer and attend over the `[..s_hot+1]` views.
///
/// No per-step O(ctx) owned concat — the previous `_auto` path rebuilt the
/// whole K/V every layer every step, i.e. O(L²) copy over a generation; this
/// is O(L), matching `standard`'s in-place handle. The residual `stored` stays
/// the canonical re-derivable state; the K/V is a droppable derivative.
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
    timed: bool,
    timings: &mut StepTimings,
) -> Option<Array2<f32>> {
    let (k_buf, v_buf) = buf;
    StepTimings::measure(timed, &mut timings.attention_us, || {
        let inplace = if super::compute::markov_inplace_kv_enabled() {
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
                // Q4K-direct disabled (the flags-off parity baseline) or no
                // attn bytes for this layer: fall back to the owned concat
                // over the buffer's logical view, then replace the buffer with
                // the exact-length result. Bit-identical to the legacy borrow
                // path; only the non-default flags-off case pays this copy.
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
    })
}

/// First step (cache `None` → seed) or windowed/cold tier: recompute the prior
/// K/V so attention can concat the new row onto it.
#[allow(clippy::too_many_arguments)]
fn recompute_prior_kv(
    weights: larql_inference::WeightsView,
    rs: &RsStore,
    layer: usize,
    s_hot: usize,
    hot_abs_start: usize,
    backend: &dyn ComputeBackend,
    timed: bool,
    timings: &mut StepTimings,
) -> Option<SharedKV> {
    let h_hot = &rs.stored[layer];
    if let Some(cold_kv) = &rs.cold_kv {
        let (k_cold_buf, v_cold_buf) = &cold_kv[layer];
        // 2026-05-19 audit fix: slice to cold_len, not shape()[0].
        // cold_kv now uses doubling-capacity (see append_cold_overflow).
        let c = rs.cold_len;
        let k_cold = k_cold_buf.slice(s![..c, ..]);
        let v_cold = v_cold_buf.slice(s![..c, ..]);
        let (k_hot, v_hot) = StepTimings::measure(timed, &mut timings.recompute_hot_us, || {
            recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)
        })?;
        let kv_dim = k_cold_buf.shape()[1];
        let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
        k_combined.slice_mut(s![..c, ..]).assign(&k_cold);
        k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
        let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
        v_combined.slice_mut(s![..c, ..]).assign(&v_cold);
        v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
        return Some((k_combined, v_combined));
    }

    let (h_full, full_abs_start) = match &rs.cold_residuals {
        // 2026-05-19 audit fix: slice to cold_len, not shape()[0].
        Some(cold) if rs.cold_len > 0 => {
            let s_cold = rs.cold_len;
            let h_cold = cold[layer].slice(s![..s_cold, ..]);
            let hidden = h_hot.shape()[1];
            let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
            combined.slice_mut(s![..s_cold, ..]).assign(&h_cold);
            combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
            (combined, rs.cold_abs_start)
        }
        _ => (h_hot.clone(), hot_abs_start),
    };
    StepTimings::measure(timed, &mut timings.recompute_cold_us, || {
        recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)
    })
}

/// Parity gate for the f32 path: the cached prior K/V must match a fresh f32
/// `recompute_kv`.
///
/// Only meaningful when attention is NOT on the Q4K-direct route — that
/// route's projections differ from `recompute_kv` by more than the bound even
/// in f32-activation (different kernels/byte sources), so it has its own
/// oracles: the compute-level bit-identity test (`run_..._inplace` ≡ the
/// concat form) and the engine-level in-place-vs-owned-concat A/B test.
#[cfg(debug_assertions)]
fn debug_assert_hot_kv_parity(
    weights: larql_inference::WeightsView,
    rs: &RsStore,
    layer: usize,
    bufs: &[SharedKV],
    s_hot: usize,
    hot_abs_start: usize,
    backend: &dyn ComputeBackend,
) {
    /// Largest per-element f32 gap tolerated between the cached prior K/V and
    /// a fresh recompute. Loose enough for accumulation order, tight enough
    /// that a genuinely stale cache trips it.
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
    debug_assert!(kd < MAX_CACHE_DRIFT, "markov hot_kv K cache diverged: {kd}");
    debug_assert!(vd < MAX_CACHE_DRIFT, "markov hot_kv V cache diverged: {vd}");
}
