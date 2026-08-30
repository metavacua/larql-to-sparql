use ndarray::Array2;

use super::dispatch::q4k_direct_attn_enabled;
use super::q4k_direct::{decode_step_attend_q4k_direct, decode_step_project_q4k_direct};

/// Append one `[1, cols]` row to a doubling-capacity `[cap, cols]` buffer at
/// logical row `len`, growing (doubling) the buffer first if it is full. Mirror
/// of `larql-kv`'s `helpers::append_row` — kept here because larql-compute can't
/// depend on larql-kv. The caller increments its logical length after.
fn append_kv_row(buf: &mut Array2<f32>, row: &Array2<f32>, len: usize) {
    let cap = buf.shape()[0];
    if len == cap {
        let cols = buf.shape()[1];
        let new_cap = (cap * 2).max(8);
        let mut grown = Array2::<f32>::zeros((new_cap, cols));
        grown
            .slice_mut(ndarray::s![..len, ..])
            .assign(&buf.slice(ndarray::s![..len, ..]));
        *buf = grown;
    }
    buf.slice_mut(ndarray::s![len..len + 1, ..]).assign(row);
}

/// In-place Q4K-direct decode-step attention for walk engines that hold their
/// hot K/V as **doubling-capacity** buffers (markov_residual / _codec). It
/// projects the new token's K/V, appends the RoPE'd row into `k_cache`/`v_cache`
/// at logical row `cache_len` (growing the buffer if full), then attends over
/// the `[..cache_len + 1]` views — eliminating the per-step O(ctx) owned concat
/// that [`run_attention_block_decode_step_q4k_direct`] pays. Over an L-token
/// generation that turns the cache copy from O(L²) total into O(L).
///
/// On return the caller's buffers hold `cache_len + 1` logical rows and the
/// function yields `h_post_attn`. Returns `None` — leaving the buffers
/// **untouched** (the projection runs before any mutation) — when the index has
/// no Q4K attention bytes for this layer, so the caller can fall back to the
/// owned-concat path. Bit-identical to the concat form: same data attended, same
/// kernels; only the cache representation (in-place views vs fresh owned concat)
/// differs.
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block_decode_step_q4k_direct_inplace(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    k_cache: &mut Array2<f32>,
    v_cache: &mut Array2<f32>,
    cache_len: usize,
    abs_position: usize,
    backend: &dyn crate::ComputeBackend,
    index: &dyn crate::KvIndex,
) -> Option<Array2<f32>> {
    let proj = decode_step_project_q4k_direct(weights, h_new, layer, abs_position, backend, index)?;
    append_kv_row(k_cache, &proj.k_new_rope, cache_len);
    append_kv_row(v_cache, &proj.v_new, cache_len);
    let total = cache_len + 1;
    decode_step_attend_q4k_direct(
        weights,
        h_new,
        layer,
        &proj.q_rope,
        k_cache.slice(ndarray::s![..total, ..]),
        v_cache.slice(ndarray::s![..total, ..]),
        backend,
        index,
    )
}

/// Best-available in-place decode-step attention for walk engines that own a
/// doubling-capacity K/V buffer: the Q4K-direct in-place path when the flag is
/// on and an index with attention bytes is supplied, else `None` so the caller
/// uses the owned-concat [`run_attention_block_decode_step_auto`]. The SAME
/// per-layer Q4K-vs-f32 choice the dispatch path makes — see
/// [`run_attention_block_decode_step_auto`].
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block_decode_step_auto_inplace(
    weights: larql_models::WeightsView,
    h_new: &Array2<f32>,
    layer: usize,
    k_cache: &mut Array2<f32>,
    v_cache: &mut Array2<f32>,
    cache_len: usize,
    abs_position: usize,
    backend: Option<&dyn crate::ComputeBackend>,
    index: Option<&dyn crate::KvIndex>,
) -> Option<Array2<f32>> {
    if q4k_direct_attn_enabled() {
        if let (Some(be), Some(idx)) = (backend, index) {
            return run_attention_block_decode_step_q4k_direct_inplace(
                weights.canonical(),
                h_new,
                layer,
                k_cache,
                v_cache,
                cache_len,
                abs_position,
                be,
                idx,
            );
        }
    }
    None
}
