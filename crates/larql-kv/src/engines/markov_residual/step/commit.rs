//! Writing a completed decode step into the residual store.
//!
//! Every canonical mutation the step makes lives here, and nothing here can
//! fail — which is what makes [`super`]'s failure invariant true by
//! construction rather than by inspection.

use larql_inference::attention::SharedKV;
use ndarray::{s, Array2};

use crate::engines::markov_residual::helpers::append_row;
use crate::engines::markov_residual::store::RsStore;

/// Write the completed step into the store.
///
/// Everything that mutates canonical state lives here, after the last
/// fallible call — which is what makes this module's failure invariant true
/// by construction rather than by inspection.
pub(super) fn commit(
    rs: &mut RsStore,
    new_stored: &[Array2<f32>],
    cache_eligible: bool,
    had_hot_kv: bool,
    hot_kv_store: Option<Vec<SharedKV>>,
    step_new_kv: Vec<SharedKV>,
    abs_position: usize,
) {
    let num_layers = new_stored.len();

    // W8.2: in the cache_eligible path `stored` is a doubling-capacity buffer
    // (no window → never clips), so append the new row in place rather than
    // allocating + bzeroing a fresh `[s_old+1, hidden]` array every step. That
    // rebuild was the resident walk's dominant per-step malloc — `__bzero` +
    // `szone_malloc` were ~32% of the driver's serial work, idling the worker
    // pool (see helpers::append_row, mirrors the dispatch path). The
    // windowed/cold path keeps the rebuild: it clips and is not cache_eligible.
    if cache_eligible {
        let hot_len = rs.hot_len;
        for (layer, new_row) in new_stored.iter().enumerate() {
            append_row(&mut rs.stored[layer], new_row, hot_len);
        }
        rs.hot_len = hot_len + 1;
    } else {
        let mut rebuilt: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
            let s_old = stored.shape()[0];
            let hidden_dim = stored.shape()[1];
            let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
            combined.slice_mut(s![..s_old, ..]).assign(stored);
            combined.slice_mut(s![s_old.., ..]).assign(new_row);
            rebuilt.push(combined);
        }
        rs.hot_len = rebuilt.first().map_or(0, |s| s.shape()[0]);
        rs.stored = rebuilt;
    }

    // Cache the full K/V (returned by attention) for next step when there's no
    // cold tier; else None (the cold/windowed path recomputes). The clip loop
    // below clips `hot_kv` in lockstep with `stored` when a window is set.
    // Step 2+ mutated `hot_kv_store` in place (the in-place fast path); the
    // first step seeds it from the freshly-collected `step_new_kv`.
    rs.hot_kv = match (cache_eligible, had_hot_kv) {
        (true, true) => hot_kv_store,
        (true, false) => Some(step_new_kv),
        (false, _) => None,
    };
    rs.next_position = abs_position + 1;

    let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut overflow);
    }
    rs.finalise_hot_len_after_clip();
    // 2026-05-19 audit fix: geometric-capacity cold append.
    // CPU walk path passes `evicted_kv = None` (cold_kv is rebuilt
    // from residuals on the next step), mirroring the prior behaviour
    // that invalidated cold_kv. See RsStore::append_cold_overflow.
    rs.append_cold_overflow(overflow, None);
}
