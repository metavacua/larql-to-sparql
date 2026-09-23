//! Writing a completed decode step into the codec store.
//!
//! Twin of [`crate::engines::markov_residual::step::commit`], with the cold
//! tier encoded rather than stored as f32. Nothing here can fail, which is
//! what makes [`super`]'s failure invariant structural.

use larql_inference::attention::SharedKV;
use ndarray::{s, Array2};

use crate::engines::markov_residual_codec::helpers::append_row;
use crate::engines::markov_residual_codec::store::{EncodedColdLayer, RsStoreCodec};

/// Write the completed step into the store — every canonical mutation, after
/// the last fallible call.
#[allow(clippy::too_many_arguments)]
pub(super) fn commit(
    hidden_size: usize,
    rs: &mut RsStoreCodec,
    new_stored: &[Array2<f32>],
    cache_eligible: bool,
    had_hot_kv: bool,
    hot_kv_store: Option<Vec<SharedKV>>,
    step_new_kv: Vec<SharedKV>,
    abs_position: usize,
) {
    let num_layers = new_stored.len();

    // Append the new row to each layer's hot tier. W8.2: in the cache_eligible
    // path `stored` is a doubling-capacity buffer (no window → never clips), so
    // append in place rather than allocating + bzeroing a fresh `[s_old+1,
    // hidden]` array every step (the resident walk's dominant per-step malloc;
    // see helpers::append_row). The windowed/cold path keeps the rebuild.
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

    // Cache the full K/V for next step when there's no cold tier; else None
    // (cold/windowed recomputes). Step 2+ mutated `hot_kv_store` in place; the
    // first step seeds it.
    rs.hot_kv = match (cache_eligible, had_hot_kv) {
        (true, true) => hot_kv_store,
        (true, false) => Some(step_new_kv),
        (false, _) => None,
    };
    rs.next_position = abs_position + 1;

    // Clip overflow into the encoded cold tier; clear cold_kv to force
    // recompute, because the codec round-trip is lossy.
    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    rs.finalise_hot_len_after_clip();
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) == 0 {
        return;
    }
    match rs.cold_encoded.as_mut() {
        Some(layers) => {
            for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                layers[layer].append(rs.codec, overflow);
            }
        }
        None => {
            let mut layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
            for overflow in overflow_per_layer.iter() {
                let mut enc = EncodedColdLayer::empty(hidden_size);
                enc.append(rs.codec, overflow);
                layers.push(enc);
            }
            rs.cold_encoded = Some(layers);
        }
    }
    rs.cold_kv = None;
}
