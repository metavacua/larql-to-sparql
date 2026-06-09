//! DeepSeek V4 attention block — `compress_ratio == 4`, indexer path.
//!
//! Sibling to:
//! - [`super::dsv4_attn_block::dsv4_attn_block_no_compress`] (compress_ratio=0)
//! - [`super::dsv4_attn_block_compress::dsv4_attn_block_compress_no_indexer`] (compress_ratio!=4)
//!
//! For `compress_ratio == 4`, DSv4 adds a **second** smaller compressor
//! (the "indexer") whose output drives a top-k sparse-attention mask
//! over the main compressed positions. Per-token, only the top-k
//! compressed positions ranked by the indexer survive the mask; the
//! rest are -∞.
//!
//! Pipeline:
//!
//! 1. Pre-norm → cur
//! 2. Q low-rank → qr (shared with the indexer), then up + per-head
//!    norm + tail-RoPE
//! 3. Raw KV low-rank + tail-RoPE + FP8
//! 4. Main compressor → main_kv_comp (head_dim) + FP8
//! 5. **Indexer compressor** → index_kv (indexer_head_size, smaller)
//! 6. **Indexer scores** via [`dsv4_build_indexer_scores_prefill`] using
//!    `(x, qr, index_kv)` and the indexer's per-head wq_b/wproj weights.
//!    Mask the scores with a causal "chunk c visible iff fully past"
//!    pattern.
//! 7. **Argsort top-k** on the indexer scores
//! 8. **Sparse compressed mask** via
//!    [`dsv4_build_compressed_mask_from_topk`]
//! 9. **Raw-window mask** over the raw KV portion (causal + sliding-window)
//! 10. Concat raw-window mask + sparse compressed mask along columns
//!     → `(n_tokens, n_tokens + n_comp)` full attention mask
//! 11. Concat raw KV + main compressed KV → combined K=V
//! 12. Masked MQA + grouped o-proj
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:1230-1308`
//! (the `if (compress_ratio == 4)` indexer branch).

use ndarray::{s, Array2, ArrayView2};

use larql_compute::{dot_proj_gpu, ComputeBackend};

use super::dsv4_attn_block::{DsV4AttnBlockParams, DsV4AttnBlockWeights};
use super::dsv4_compressor_prefill::{
    build_compressor_prefill, dsv4_compressor_step_coff2, CompressorParams, CompressorWeights,
};
use super::dsv4_fp8_kv::fp8_kv_quantize;
use super::dsv4_grouped_o_proj::grouped_o_proj;
use super::dsv4_indexer::{
    argsort_top_k, build_compressed_mask_from_topk, build_indexer_scores_prefill, IndexerParams,
    IndexerWeights,
};
use super::dsv4_kv_cache::DsV4LayerHcaCache;
use super::dsv4_masked_attn::dsv4_masked_attn;
use super::dsv4_rope_tail_yarn::rope_tail_dispatch;

/// Combined weights for the HCA-with-indexer block.
#[derive(Clone, Copy)]
pub struct DsV4AttnBlockIndexerWeights<'a> {
    /// Standard attention weights (Wq_a, Wq_b, Wkv, Wo_a, Wo_b, etc.).
    pub attn: DsV4AttnBlockWeights<'a>,
    /// Main compressor (full head_dim, compress_ratio=4, coff=2).
    pub compressor: CompressorWeights<'a>,
    /// Indexer compressor (indexer_head_size, compress_ratio=4, coff=2).
    pub indexer_compressor: CompressorWeights<'a>,
    /// Indexer score weights (wq_b, wproj).
    pub indexer: IndexerWeights<'a>,
}

/// Combined params for the HCA-with-indexer block.
#[derive(Clone, Copy, Debug)]
pub struct DsV4AttnBlockIndexerParams {
    pub attn: DsV4AttnBlockParams,
    pub compressor: CompressorParams,
    pub indexer_compressor: CompressorParams,
    pub indexer: IndexerParams,
    pub top_k: usize,
}

/// Run the HCA-with-indexer attention block. Requires
/// `compress_ratio == 4` on both compressors and `top_k > 0`.
pub fn dsv4_attn_block_compress_with_indexer(
    x: ArrayView2<f32>,
    w: &DsV4AttnBlockIndexerWeights,
    p: &DsV4AttnBlockIndexerParams,
    position_offset: usize,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    assert_eq!(x.shape(), &[n_tokens, p.attn.n_embd]);
    assert_eq!(
        p.compressor.compress_ratio, 4,
        "indexer path requires compress_ratio = 4 on main compressor"
    );
    assert_eq!(
        p.indexer_compressor.compress_ratio, 4,
        "indexer path requires compress_ratio = 4 on indexer compressor"
    );
    assert_eq!(
        p.attn.head_dim, p.compressor.head_dim,
        "attn and main-compressor head_dim must match"
    );
    assert_eq!(
        p.attn.n_embd, p.compressor.n_embd,
        "attn and main-compressor n_embd must match"
    );
    assert_eq!(
        p.indexer_compressor.head_dim, p.indexer.n_index_head_size,
        "indexer_compressor head_dim must equal indexer.n_index_head_size"
    );
    assert!(p.top_k > 0, "top_k must be > 0");

    // 1. attn_norm.
    let cur = rms_norm_2d(x, w.attn.attn_norm, p.attn.norm_eps);

    // 2. Q low-rank: qr is shared with the indexer.
    let qr = w.attn.proj_wq_a(&cur, backend);
    let qr = rms_norm_2d(qr.view(), w.attn.q_a_norm, p.attn.norm_eps);
    let q = w.attn.proj_wq_b(&qr, backend);
    let q = rms_norm_per_head(q.view(), p.attn.n_head, p.attn.head_dim, p.attn.norm_eps);
    let q = rope_tail_dispatch(
        &q,
        p.attn.n_head,
        p.attn.head_dim,
        p.attn.n_rot,
        p.attn.rope_base,
        p.attn.rope_mode,
        p.attn.yarn.as_ref(),
        false,
        position_offset,
    );

    // 3. Raw KV low-rank.
    let kv_raw = w.attn.proj_wkv(&cur, backend);
    let kv_raw = rms_norm_2d(kv_raw.view(), w.attn.kv_a_norm, p.attn.norm_eps);
    let kv_raw = rope_tail_dispatch(
        &kv_raw,
        1,
        p.attn.head_dim,
        p.attn.n_rot,
        p.attn.rope_base,
        p.attn.rope_mode,
        p.attn.yarn.as_ref(),
        false,
        position_offset,
    );
    let kv_raw = fp8_kv_quantize(kv_raw.view(), p.attn.n_rot);

    // 4. Main compressed KV.
    let kv_comp_pre = build_compressor_prefill(cur.view(), &w.compressor, &p.compressor, backend);
    let kv_comp = fp8_kv_quantize(kv_comp_pre.view(), p.attn.n_rot);
    let n_comp = kv_comp.shape()[0];

    // 5. Indexer compressed KV (smaller head_dim = indexer_head_size).
    //    No FP8 quant — the indexer uses lower-precision math anyway via
    //    the scaled per-head weights, and the reference doesn't quant
    //    the indexer's KV.
    let index_kv = build_compressor_prefill(
        cur.view(),
        &w.indexer_compressor,
        &p.indexer_compressor,
        backend,
    );
    assert_eq!(
        index_kv.shape(),
        &[n_comp, p.indexer.n_index_head_size],
        "indexer_compressor should produce n_comp rows of indexer_head_size dim"
    );

    // 6. Indexer scores with a per-(token, comp) causal mask.
    let indexer_causal_mask = build_indexer_causal_mask(
        n_tokens,
        n_comp,
        position_offset,
        p.compressor.compress_ratio,
    );
    let index_scores = build_indexer_scores_prefill(
        x,
        qr.view(),
        index_kv.view(),
        indexer_causal_mask.view(),
        &w.indexer,
        &p.indexer,
        backend,
    );

    // 7. Argsort top-k.
    let top_k = p.top_k.min(n_comp);
    let topk = argsort_top_k(index_scores.view(), top_k);

    // 8. Sparse compressed mask `(n_tokens, n_comp)`.
    let comp_mask = build_compressed_mask_from_topk(index_scores.view(), topk.view());

    // 9. Raw-window mask `(n_tokens, n_tokens)`.
    let raw_mask = build_raw_window_mask(n_tokens, position_offset, p.attn.window_size);

    // 10. Concat: `(n_tokens, n_tokens + n_comp)`.
    let mut full_mask = Array2::<f32>::zeros((n_tokens, n_tokens + n_comp));
    full_mask.slice_mut(s![.., ..n_tokens]).assign(&raw_mask);
    full_mask.slice_mut(s![.., n_tokens..]).assign(&comp_mask);

    // 11. Concat raw + compressed K/V along the n_kv axis.
    let mut kv_cat = Array2::<f32>::zeros((n_tokens + n_comp, p.attn.head_dim));
    kv_cat.slice_mut(s![..n_tokens, ..]).assign(&kv_raw);
    kv_cat.slice_mut(s![n_tokens.., ..]).assign(&kv_comp);

    // 12. Masked MQA.
    let scale = 1.0 / (p.attn.head_dim as f32).sqrt();
    let o = dsv4_masked_attn(
        q.view(),
        kv_cat.view(),
        kv_cat.view(),
        full_mask.view(),
        p.attn.n_head,
        p.attn.head_dim,
        w.attn.attn_sinks,
        scale,
        backend,
    );

    grouped_o_proj(
        o.view(),
        w.attn.wo_a,
        w.attn.wo_b,
        p.attn.n_head,
        p.attn.head_dim,
        p.attn.n_groups,
        p.attn.o_lora_rank,
        backend,
        w.attn.quant.map(|q| q.wo_a),
        w.attn.quant.map(|q| q.wo_b),
    )
}

/// Build the `(n_tokens, n_comp)` causal mask used by the indexer:
/// query at absolute position `p_t` sees compressed chunk c iff
/// `(c + 1) * ratio <= p_t + 1`.
pub fn build_indexer_causal_mask(
    n_tokens: usize,
    n_comp: usize,
    position_offset: usize,
    compress_ratio: usize,
) -> Array2<f32> {
    let mut m = Array2::<f32>::from_elem((n_tokens, n_comp), f32::NEG_INFINITY);
    for t in 0..n_tokens {
        let abs_pos = position_offset + t;
        let c_visible = (abs_pos + 1) / compress_ratio;
        for c in 0..c_visible.min(n_comp) {
            m[[t, c]] = 0.0;
        }
    }
    m
}

/// Build the `(n_tokens, n_tokens)` causal sliding-window mask over the
/// raw KV portion of the attention.
pub fn build_raw_window_mask(
    n_tokens: usize,
    position_offset: usize,
    window_size: usize,
) -> Array2<f32> {
    let mut m = Array2::<f32>::from_elem((n_tokens, n_tokens), f32::NEG_INFINITY);
    for t in 0..n_tokens {
        let abs_pos = position_offset + t;
        let j_lo = abs_pos.saturating_sub(window_size - 1);
        let j_hi = abs_pos;
        for j in j_lo..=j_hi.min(n_tokens - 1) {
            m[[t, j]] = 0.0;
        }
    }
    m
}

/// KV-cached variant of [`dsv4_attn_block_compress_with_indexer`].
///
/// Composes everything we built for KV caching into a complete
/// cache-aware HCA-with-indexer attention block:
/// - `dsv4_compressor_step_coff2` (#329) is called twice per
///   completed chunk — once on the main compressor weights/state and
///   once on the indexer compressor weights/state.
/// - Both compressors share the `cache.pending_cur` buffer and share
///   chunk-completion timing (one chunk fills → both compressors run).
/// - `cache.overlap_state` threads the main compressor's prev halves;
///   `cache.indexer_overlap_state` threads the indexer compressor's
///   prev halves.
/// - Indexer scores are computed on the new tokens' x + qr against
///   the cached indexer compressed KV; top-k + sparse mask is built
///   over the full cached compressed positions.
///
/// `cache.index_compressed` must be `Some(_)` — use
/// [`DsV4LayerHcaCache::with_capacity_and_indexer`] or
/// [`DsV4LayerHcaCache::enable_indexer`] to initialize it. Panics
/// otherwise.
///
/// Bit-exact-equivalent to [`dsv4_attn_block_compress_with_indexer`]
/// on the same input when the cache starts empty and
/// `position_offset == 0`.
pub fn dsv4_attn_block_compress_with_indexer_cached(
    x: ArrayView2<f32>,
    w: &DsV4AttnBlockIndexerWeights,
    p: &DsV4AttnBlockIndexerParams,
    position_offset: usize,
    cache: &mut DsV4LayerHcaCache,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let n_new = x.shape()[0];
    assert_eq!(x.shape(), &[n_new, p.attn.n_embd]);
    assert_eq!(
        p.compressor.compress_ratio, 4,
        "indexer cached path requires compress_ratio = 4"
    );
    assert_eq!(
        p.indexer_compressor.compress_ratio, 4,
        "indexer cached path requires indexer_compressor.compress_ratio = 4"
    );
    assert_eq!(
        cache.compress_ratio, 4,
        "cache.compress_ratio must equal 4 for the indexer path"
    );
    assert_eq!(
        cache.raw.current_len(),
        position_offset,
        "position_offset ({position_offset}) must equal cache.raw.current_len ({})",
        cache.raw.current_len()
    );
    assert!(p.top_k > 0, "top_k must be > 0");
    assert!(
        cache.index_compressed.is_some(),
        "cache.index_compressed must be initialized; call enable_indexer"
    );

    // 1. attn_norm.
    let cur = rms_norm_2d(x, w.attn.attn_norm, p.attn.norm_eps);

    // 2. Q low-rank: qr shared with the indexer scores; routed via backend.
    let qr = w.attn.proj_wq_a(&cur, backend);
    let qr = rms_norm_2d(qr.view(), w.attn.q_a_norm, p.attn.norm_eps);
    let q = w.attn.proj_wq_b(&qr, backend);
    let q = rms_norm_per_head(q.view(), p.attn.n_head, p.attn.head_dim, p.attn.norm_eps);
    let q = rope_tail_dispatch(
        &q,
        p.attn.n_head,
        p.attn.head_dim,
        p.attn.n_rot,
        p.attn.rope_base,
        p.attn.rope_mode,
        p.attn.yarn.as_ref(),
        false,
        position_offset,
    );

    // 3. Raw KV (new tokens only) → append to cache.raw. Backend-routed.
    let kv_raw = w.attn.proj_wkv(&cur, backend);
    let kv_raw = rms_norm_2d(kv_raw.view(), w.attn.kv_a_norm, p.attn.norm_eps);
    let kv_raw = rope_tail_dispatch(
        &kv_raw,
        1,
        p.attn.head_dim,
        p.attn.n_rot,
        p.attn.rope_base,
        p.attn.rope_mode,
        p.attn.yarn.as_ref(),
        false,
        position_offset,
    );
    let kv_raw = fp8_kv_quantize(kv_raw.view(), p.attn.n_rot);
    cache.raw.append(kv_raw.view());

    // 4. Drain new cur rows into pending; per completed chunk run
    //    BOTH compressor steps (main + indexer) and append their
    //    outputs to the respective caches.
    for t in 0..n_new {
        cache.pending_cur.push(cur.row(t).to_owned());
        if cache.pending_cur.len() >= p.compressor.compress_ratio {
            // Build chunk array (compress_ratio, n_embd).
            let mut chunk = Array2::<f32>::zeros((p.compressor.compress_ratio, p.attn.n_embd));
            let drained: Vec<_> = cache
                .pending_cur
                .drain(..p.compressor.compress_ratio)
                .collect();
            for (r, row) in drained.iter().enumerate() {
                for d in 0..p.attn.n_embd {
                    chunk[[r, d]] = row[d];
                }
            }

            // Main compressor step (coff=2 with cache.overlap_state).
            let chunk_idx_main = cache.compressed.current_len();
            let new_main = dsv4_compressor_step_coff2(
                chunk.view(),
                &w.compressor,
                &p.compressor,
                chunk_idx_main,
                &mut cache.overlap_state,
                backend,
            );
            let new_main_2d =
                Array2::<f32>::from_shape_vec((1, p.attn.head_dim), new_main.to_vec())
                    .expect("step coff2 returned head_dim values");
            let new_main_2d = fp8_kv_quantize(new_main_2d.view(), p.attn.n_rot);
            cache.compressed.append(new_main_2d.view());

            // Indexer compressor step (coff=2 with cache.indexer_overlap_state).
            // No FP8 quant — the prefill function doesn't quantize the
            // indexer KV either.
            let chunk_idx_idx = cache
                .index_compressed
                .as_ref()
                .expect("indexer enabled")
                .current_len();
            let new_idx = dsv4_compressor_step_coff2(
                chunk.view(),
                &w.indexer_compressor,
                &p.indexer_compressor,
                chunk_idx_idx,
                &mut cache.indexer_overlap_state,
                backend,
            );
            let new_idx_2d =
                Array2::<f32>::from_shape_vec((1, p.indexer.n_index_head_size), new_idx.to_vec())
                    .expect("step coff2 returned indexer_head_size values");
            cache
                .index_compressed
                .as_mut()
                .expect("indexer enabled")
                .append(new_idx_2d.view());
        }
    }

    // 5. Indexer scores: new tokens' x/qr × cached full indexer KV.
    let raw_total = cache.raw.current_len();
    let n_comp = cache.compressed.current_len();
    let idx_kv_view = cache.index_compressed.as_ref().unwrap().view_current();
    let idx_causal = build_indexer_causal_mask_cached(
        n_new,
        n_comp,
        position_offset,
        p.compressor.compress_ratio,
    );
    let index_scores = build_indexer_scores_prefill(
        x,
        qr.view(),
        idx_kv_view,
        idx_causal.view(),
        &w.indexer,
        &p.indexer,
        backend,
    );

    // 6. Argsort top-k and sparse mask.
    let top_k = p.top_k.min(n_comp);
    let topk = argsort_top_k(index_scores.view(), top_k);
    let comp_mask = build_compressed_mask_from_topk(index_scores.view(), topk.view());

    // 7. Raw-window mask over the full cached raw KV.
    let raw_mask =
        build_raw_window_mask_cached(n_new, raw_total, position_offset, p.attn.window_size);

    // 8. Concat masks: (n_new, raw_total + n_comp).
    let mut full_mask = Array2::<f32>::zeros((n_new, raw_total + n_comp));
    full_mask.slice_mut(s![.., ..raw_total]).assign(&raw_mask);
    full_mask.slice_mut(s![.., raw_total..]).assign(&comp_mask);

    // 9. Concat raw + main compressed KV.
    let mut kv_cat = Array2::<f32>::zeros((raw_total + n_comp, p.attn.head_dim));
    kv_cat
        .slice_mut(s![..raw_total, ..])
        .assign(&cache.raw.view_current());
    kv_cat
        .slice_mut(s![raw_total.., ..])
        .assign(&cache.compressed.view_current());

    // 10. Masked MQA.
    let scale = 1.0 / (p.attn.head_dim as f32).sqrt();
    let o = dsv4_masked_attn(
        q.view(),
        kv_cat.view(),
        kv_cat.view(),
        full_mask.view(),
        p.attn.n_head,
        p.attn.head_dim,
        w.attn.attn_sinks,
        scale,
        backend,
    );

    // 11. Grouped o-proj.
    grouped_o_proj(
        o.view(),
        w.attn.wo_a,
        w.attn.wo_b,
        p.attn.n_head,
        p.attn.head_dim,
        p.attn.n_groups,
        p.attn.o_lora_rank,
        backend,
        w.attn.quant.map(|q| q.wo_a),
        w.attn.quant.map(|q| q.wo_b),
    )
}

/// Cached version of [`build_indexer_causal_mask`]. Mask shape
/// `(n_new, n_comp)`; query at absolute position `p_t = position_offset
/// + t` sees compressed chunk c iff `(c+1)*ratio <= p_t + 1`.
pub fn build_indexer_causal_mask_cached(
    n_new: usize,
    n_comp: usize,
    position_offset: usize,
    compress_ratio: usize,
) -> Array2<f32> {
    let mut m = Array2::<f32>::from_elem((n_new, n_comp), f32::NEG_INFINITY);
    for t in 0..n_new {
        let abs_pos = position_offset + t;
        let c_visible = (abs_pos + 1) / compress_ratio;
        for c in 0..c_visible.min(n_comp) {
            m[[t, c]] = 0.0;
        }
    }
    m
}

/// Cached version of [`build_raw_window_mask`] over the FULL cached
/// raw KV (raw_total rows, indexed by absolute position).
pub fn build_raw_window_mask_cached(
    n_new: usize,
    raw_total: usize,
    position_offset: usize,
    window_size: usize,
) -> Array2<f32> {
    let mut m = Array2::<f32>::from_elem((n_new, raw_total), f32::NEG_INFINITY);
    for t in 0..n_new {
        let abs_pos = position_offset + t;
        let j_lo = abs_pos.saturating_sub(window_size - 1);
        let j_hi = abs_pos.min(raw_total.saturating_sub(1));
        if abs_pos < raw_total {
            for j in j_lo..=j_hi {
                m[[t, j]] = 0.0;
            }
        }
    }
    m
}

// ── Shared helpers (mirrors `dsv4_attn_block.rs`) ────────────────────

fn rms_norm_2d(x: ArrayView2<f32>, weight: &[f32], eps: f32) -> Array2<f32> {
    let n_rows = x.shape()[0];
    let n_dims = x.shape()[1];
    assert_eq!(weight.len(), n_dims);
    let mut out = Array2::<f32>::zeros((n_rows, n_dims));
    for t in 0..n_rows {
        let mut sumsq = 0.0_f32;
        for d in 0..n_dims {
            let v = x[[t, d]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n_dims as f32 + eps).sqrt();
        for d in 0..n_dims {
            out[[t, d]] = x[[t, d]] * inv * weight[d];
        }
    }
    out
}

fn rms_norm_per_head(x: ArrayView2<f32>, n_head: usize, head_dim: usize, eps: f32) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    assert_eq!(x.shape(), &[n_tokens, n_head * head_dim]);
    let mut out = Array2::<f32>::zeros((n_tokens, n_head * head_dim));
    for t in 0..n_tokens {
        for h in 0..n_head {
            let off = h * head_dim;
            let mut sumsq = 0.0_f32;
            for d in 0..head_dim {
                let v = x[[t, off + d]];
                sumsq += v * v;
            }
            let inv = 1.0 / (sumsq / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                out[[t, off + d]] = x[[t, off + d]] * inv;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_rope_tail::DsV4RopeMode;
    use super::*;
    use ndarray::{Array1, Array3};

    /// Smoke: full indexer-path block with compress_ratio=4 produces
    /// finite, non-trivial output. Verifies the entire 12-step pipeline
    /// composes without shape/type errors.
    #[test]
    fn block_with_indexer_smoke() {
        let n_tokens = 16; // n_comp = 4 with ratio=4
        let n_embd = 64;
        let n_head = 4;
        let head_dim = 64; // n_nope = 64, n_rot = 0
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let indexer_head_size = 16;
        let n_index_head = 2;
        let top_k = 2;

        let attn_p = DsV4AttnBlockParams {
            n_embd,
            n_head,
            head_dim,
            q_lora_rank,
            n_groups,
            o_lora_rank,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            yarn: None,
        };
        let comp_p = CompressorParams {
            head_dim,
            n_embd,
            compress_ratio: 4,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let idx_comp_p = CompressorParams {
            head_dim: indexer_head_size,
            n_embd,
            compress_ratio: 4,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let idx_p = IndexerParams {
            n_embd,
            q_lora_rank,
            n_index_head,
            n_index_head_size: indexer_head_size,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
        };
        let p = DsV4AttnBlockIndexerParams {
            attn: attn_p,
            compressor: comp_p,
            indexer_compressor: idx_comp_p,
            indexer: idx_p,
            top_k,
        };

        // ── Weights ──
        let group_heads = n_head / n_groups;
        let group_dim = head_dim * group_heads;
        let low_dim = o_lora_rank * n_groups;

        let attn_norm = vec![1.0_f32; n_embd];
        let wq_a = Array2::<f32>::from_shape_fn((q_lora_rank, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.01).sin()
        });
        let q_a_norm = vec![1.0_f32; q_lora_rank];
        let wq_b = Array2::<f32>::from_shape_fn((n_head * head_dim, q_lora_rank), |(i, j)| {
            ((i + j) as f32 * 0.013).cos() * 0.1
        });
        let wkv = Array2::<f32>::from_shape_fn((head_dim, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.007).sin() * 0.05
        });
        let kv_a_norm = vec![1.0_f32; head_dim];
        let wo_a = Array3::<f32>::from_shape_fn((n_groups, o_lora_rank, group_dim), |(g, r, j)| {
            ((g + r + j) as f32 * 0.005).cos() * 0.05
        });
        let wo_b = Array2::<f32>::from_shape_fn((n_embd, low_dim), |(i, j)| {
            ((i + j) as f32 * 0.011).sin() * 0.1
        });
        let attn_sinks = Array1::<f32>::from_elem(n_head, -1.0);

        // Main compressor (coff=2 for compress_ratio=4): n_kv = 2 * head_dim
        let main_n_kv = 2 * head_dim;
        let comp_wkv = Array2::<f32>::from_shape_fn((main_n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.013).cos() * 0.05
        });
        let comp_wgate = Array2::<f32>::from_shape_fn((main_n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.017).sin() * 0.05
        });
        let comp_ape = Array2::<f32>::from_shape_fn((4, main_n_kv), |(r, k)| {
            ((r + k) as f32 * 0.03).sin() * 0.05
        });
        let comp_norm = vec![1.0_f32; head_dim];

        // Indexer compressor (coff=2 for compress_ratio=4): n_kv = 2 * indexer_head_size
        let idx_n_kv = 2 * indexer_head_size;
        let idx_comp_wkv = Array2::<f32>::from_shape_fn((idx_n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.014).cos() * 0.05
        });
        let idx_comp_wgate = Array2::<f32>::from_shape_fn((idx_n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.016).sin() * 0.05
        });
        let idx_comp_ape = Array2::<f32>::from_shape_fn((4, idx_n_kv), |(r, k)| {
            ((r + k) as f32 * 0.025).sin() * 0.05
        });
        let idx_comp_norm = vec![1.0_f32; indexer_head_size];

        // Indexer score weights
        let idx_wq_b = Array2::<f32>::from_shape_fn(
            (n_index_head * indexer_head_size, q_lora_rank),
            |(i, j)| ((i + j) as f32 * 0.011).sin() * 0.1,
        );
        let idx_wproj = Array2::<f32>::from_shape_fn((n_index_head, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.012).cos() * 0.1
        });

        let w = DsV4AttnBlockIndexerWeights {
            attn: DsV4AttnBlockWeights {
                quant: None,
                attn_norm: &attn_norm,
                wq_a: wq_a.view(),
                q_a_norm: &q_a_norm,
                wq_b: wq_b.view(),
                wkv: wkv.view(),
                kv_a_norm: &kv_a_norm,
                wo_a: wo_a.view(),
                wo_b: wo_b.view(),
                attn_sinks: Some(attn_sinks.view()),
            },
            compressor: CompressorWeights {
                wkv: comp_wkv.view(),
                wgate: comp_wgate.view(),
                ape: comp_ape.view(),
                norm: &comp_norm,
                quant: None,
            },
            indexer_compressor: CompressorWeights {
                wkv: idx_comp_wkv.view(),
                wgate: idx_comp_wgate.view(),
                ape: idx_comp_ape.view(),
                norm: &idx_comp_norm,
                quant: None,
            },
            indexer: IndexerWeights {
                wq_b: idx_wq_b.view(),
                wproj: idx_wproj.view(),
                quant: None,
            },
        };

        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 7 + d) as f32 * 0.013).sin()
        });
        let out = dsv4_attn_block_compress_with_indexer(x.view(), &w, &p, 0, None);
        assert_eq!(out.shape(), &[n_tokens, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(total > 1e-3, "output collapsed (sum={total})");
    }

    /// Indexer causal mask: token t sees compressed chunk c iff
    /// `(c+1)*ratio <= t+1`.
    #[test]
    fn indexer_causal_mask_structure() {
        let m = build_indexer_causal_mask(8, 4, 0, 4);
        assert_eq!(m.shape(), &[8, 4]);
        // Token 0..2 (abs_pos 0..2): c_visible = (abs_pos+1)/4 = 0 →
        // no compressed visible.
        for t in 0..3 {
            for c in 0..4 {
                assert_eq!(m[[t, c]], f32::NEG_INFINITY, "t={t} c={c} should be -∞");
            }
        }
        // Token 3..6 (abs_pos 3..6): c_visible = 1 → chunk 0 visible.
        for t in 3..7 {
            assert_eq!(m[[t, 0]], 0.0, "t={t} c=0 should be 0");
            for c in 1..4 {
                assert_eq!(m[[t, c]], f32::NEG_INFINITY, "t={t} c={c} should be -∞");
            }
        }
        // Token 7 (abs_pos 7): c_visible = 8/4 = 2 → chunks 0, 1 visible.
        assert_eq!(m[[7, 0]], 0.0);
        assert_eq!(m[[7, 1]], 0.0);
        for c in 2..4 {
            assert_eq!(m[[7, c]], f32::NEG_INFINITY);
        }
    }

    /// Raw-window mask: standard causal + sliding window over the
    /// raw KV positions.
    #[test]
    fn raw_window_mask_structure() {
        let m = build_raw_window_mask(6, 0, 3);
        assert_eq!(m.shape(), &[6, 6]);
        // Token 0: only j=0 allowed.
        assert_eq!(m[[0, 0]], 0.0);
        for j in 1..6 {
            assert_eq!(m[[0, j]], f32::NEG_INFINITY);
        }
        // Token 5: window [3, 4, 5] allowed; [0, 1, 2] disallowed.
        for j in 0..3 {
            assert_eq!(m[[5, j]], f32::NEG_INFINITY);
        }
        for j in 3..=5 {
            assert_eq!(m[[5, j]], 0.0);
        }
    }

    // ── Cached Indexer attention equivalence tests ──

    /// Helper: build the same setup as `block_with_indexer_smoke` and
    /// return everything needed for both prefill and cached calls.
    /// Long return type but keeps the two equivalence tests symmetric.
    #[allow(clippy::type_complexity)]
    fn build_indexer_test_setup(
        n_tokens: usize,
    ) -> (
        Array2<f32>,
        DsV4AttnBlockIndexerParams,
        Vec<f32>,
        Array2<f32>,
        Vec<f32>,
        Array2<f32>,
        Array2<f32>,
        Vec<f32>,
        Array3<f32>,
        Array2<f32>,
        Array1<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Vec<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Vec<f32>,
        Array2<f32>,
        Array2<f32>,
    ) {
        let n_embd = 64;
        let n_head = 4;
        let head_dim = 64;
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let indexer_head_size = 16;
        let n_index_head = 2;
        let top_k = 2;

        let attn_p = DsV4AttnBlockParams {
            n_embd,
            n_head,
            head_dim,
            q_lora_rank,
            n_groups,
            o_lora_rank,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            yarn: None,
        };
        let comp_p = CompressorParams {
            head_dim,
            n_embd,
            compress_ratio: 4,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let idx_comp_p = CompressorParams {
            head_dim: indexer_head_size,
            n_embd,
            compress_ratio: 4,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let idx_p = IndexerParams {
            n_embd,
            q_lora_rank,
            n_index_head,
            n_index_head_size: indexer_head_size,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
        };
        let p = DsV4AttnBlockIndexerParams {
            attn: attn_p,
            compressor: comp_p,
            indexer_compressor: idx_comp_p,
            indexer: idx_p,
            top_k,
        };

        let group_heads = n_head / n_groups;
        let group_dim = head_dim * group_heads;
        let low_dim = o_lora_rank * n_groups;

        let attn_norm = vec![1.0_f32; n_embd];
        let wq_a = Array2::<f32>::from_shape_fn((q_lora_rank, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.01).sin()
        });
        let q_a_norm = vec![1.0_f32; q_lora_rank];
        let wq_b = Array2::<f32>::from_shape_fn((n_head * head_dim, q_lora_rank), |(i, j)| {
            ((i + j) as f32 * 0.013).cos() * 0.1
        });
        let wkv = Array2::<f32>::from_shape_fn((head_dim, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.007).sin() * 0.05
        });
        let kv_a_norm = vec![1.0_f32; head_dim];
        let wo_a = Array3::<f32>::from_shape_fn((n_groups, o_lora_rank, group_dim), |(g, r, j)| {
            ((g + r + j) as f32 * 0.005).cos() * 0.05
        });
        let wo_b = Array2::<f32>::from_shape_fn((n_embd, low_dim), |(i, j)| {
            ((i + j) as f32 * 0.011).sin() * 0.1
        });
        let attn_sinks = Array1::<f32>::from_elem(n_head, -1.0);

        let main_n_kv = 2 * head_dim;
        let comp_wkv = Array2::<f32>::from_shape_fn((main_n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.013).cos() * 0.05
        });
        let comp_wgate = Array2::<f32>::from_shape_fn((main_n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.017).sin() * 0.05
        });
        let comp_ape = Array2::<f32>::from_shape_fn((4, main_n_kv), |(r, k)| {
            ((r + k) as f32 * 0.03).sin() * 0.05
        });
        let comp_norm = vec![1.0_f32; head_dim];

        let idx_n_kv = 2 * indexer_head_size;
        let idx_comp_wkv = Array2::<f32>::from_shape_fn((idx_n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.014).cos() * 0.05
        });
        let idx_comp_wgate = Array2::<f32>::from_shape_fn((idx_n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.016).sin() * 0.05
        });
        let idx_comp_ape = Array2::<f32>::from_shape_fn((4, idx_n_kv), |(r, k)| {
            ((r + k) as f32 * 0.025).sin() * 0.05
        });
        let idx_comp_norm = vec![1.0_f32; indexer_head_size];

        let idx_wq_b = Array2::<f32>::from_shape_fn(
            (n_index_head * indexer_head_size, q_lora_rank),
            |(i, j)| ((i + j) as f32 * 0.011).sin() * 0.1,
        );
        let idx_wproj = Array2::<f32>::from_shape_fn((n_index_head, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.012).cos() * 0.1
        });

        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 7 + d) as f32 * 0.013).sin()
        });

        (
            x,
            p,
            attn_norm,
            wq_a,
            q_a_norm,
            wq_b,
            wkv,
            kv_a_norm,
            wo_a,
            wo_b,
            attn_sinks,
            comp_wkv,
            comp_wgate,
            comp_ape,
            comp_norm,
            idx_comp_wkv,
            idx_comp_wgate,
            idx_comp_ape,
            idx_comp_norm,
            idx_wq_b,
            idx_wproj,
        )
    }

    /// Cornerstone bit-exact equivalence: cached path with empty cache
    /// + 16-token input matches prefill function. n_comp = 4 chunks.
    #[test]
    fn cached_indexer_empty_cache_equals_prefill() {
        let n_tokens = 16;
        let (
            x,
            p,
            attn_norm,
            wq_a,
            q_a_norm,
            wq_b,
            wkv,
            kv_a_norm,
            wo_a,
            wo_b,
            attn_sinks,
            comp_wkv,
            comp_wgate,
            comp_ape,
            comp_norm,
            idx_comp_wkv,
            idx_comp_wgate,
            idx_comp_ape,
            idx_comp_norm,
            idx_wq_b,
            idx_wproj,
        ) = build_indexer_test_setup(n_tokens);
        let w = DsV4AttnBlockIndexerWeights {
            attn: DsV4AttnBlockWeights {
                quant: None,
                attn_norm: &attn_norm,
                wq_a: wq_a.view(),
                q_a_norm: &q_a_norm,
                wq_b: wq_b.view(),
                wkv: wkv.view(),
                kv_a_norm: &kv_a_norm,
                wo_a: wo_a.view(),
                wo_b: wo_b.view(),
                attn_sinks: Some(attn_sinks.view()),
            },
            compressor: CompressorWeights {
                wkv: comp_wkv.view(),
                wgate: comp_wgate.view(),
                ape: comp_ape.view(),
                norm: &comp_norm,
                quant: None,
            },
            indexer_compressor: CompressorWeights {
                wkv: idx_comp_wkv.view(),
                wgate: idx_comp_wgate.view(),
                ape: idx_comp_ape.view(),
                norm: &idx_comp_norm,
                quant: None,
            },
            indexer: IndexerWeights {
                wq_b: idx_wq_b.view(),
                wproj: idx_wproj.view(),
                quant: None,
            },
        };

        let out_prefill = dsv4_attn_block_compress_with_indexer(x.view(), &w, &p, 0, None);

        let mut cache = DsV4LayerHcaCache::with_capacity_and_indexer(
            32,
            p.attn.head_dim,
            4,
            p.indexer.n_index_head_size,
        );
        let out_cached =
            dsv4_attn_block_compress_with_indexer_cached(x.view(), &w, &p, 0, &mut cache, None);

        assert_eq!(out_prefill.shape(), out_cached.shape());
        let mut max_diff = 0.0_f32;
        for (a, b) in out_prefill.iter().zip(out_cached.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-5,
            "cached indexer vs prefill max diff = {max_diff}"
        );
        assert_eq!(cache.raw.current_len(), n_tokens);
        assert_eq!(cache.compressed.current_len(), n_tokens / 4);
        assert_eq!(
            cache.index_compressed.as_ref().unwrap().current_len(),
            n_tokens / 4
        );
        assert!(cache.pending_cur.is_empty());
    }

    /// CpuBackend equivalence on the Indexer prefill block: routing
    /// Q/KV/O matmuls through `Some(&CpuBackend)` must match the
    /// `None` fallback to within BLAS reduction tolerance. Locks in
    /// the GPU-4c backend-threading wiring.
    #[test]
    fn indexer_prefill_cpu_backend_matches_none_backend() {
        let n_tokens = 16;
        let (
            x,
            p,
            attn_norm,
            wq_a,
            q_a_norm,
            wq_b,
            wkv,
            kv_a_norm,
            wo_a,
            wo_b,
            attn_sinks,
            comp_wkv,
            comp_wgate,
            comp_ape,
            comp_norm,
            idx_comp_wkv,
            idx_comp_wgate,
            idx_comp_ape,
            idx_comp_norm,
            idx_wq_b,
            idx_wproj,
        ) = build_indexer_test_setup(n_tokens);
        let w = DsV4AttnBlockIndexerWeights {
            attn: DsV4AttnBlockWeights {
                quant: None,
                attn_norm: &attn_norm,
                wq_a: wq_a.view(),
                q_a_norm: &q_a_norm,
                wq_b: wq_b.view(),
                wkv: wkv.view(),
                kv_a_norm: &kv_a_norm,
                wo_a: wo_a.view(),
                wo_b: wo_b.view(),
                attn_sinks: Some(attn_sinks.view()),
            },
            compressor: CompressorWeights {
                wkv: comp_wkv.view(),
                wgate: comp_wgate.view(),
                ape: comp_ape.view(),
                norm: &comp_norm,
                quant: None,
            },
            indexer_compressor: CompressorWeights {
                wkv: idx_comp_wkv.view(),
                wgate: idx_comp_wgate.view(),
                ape: idx_comp_ape.view(),
                norm: &idx_comp_norm,
                quant: None,
            },
            indexer: IndexerWeights {
                wq_b: idx_wq_b.view(),
                wproj: idx_wproj.view(),
                quant: None,
            },
        };

        let out_none = dsv4_attn_block_compress_with_indexer(x.view(), &w, &p, 0, None);
        let cpu = larql_compute::CpuBackend;
        let out_cpu = dsv4_attn_block_compress_with_indexer(
            x.view(),
            &w,
            &p,
            0,
            Some(&cpu as &dyn ComputeBackend),
        );

        assert_eq!(out_none.shape(), out_cpu.shape());
        let max_diff = out_none
            .iter()
            .zip(out_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "CpuBackend vs None mismatch on Indexer prefill: max_diff={max_diff}"
        );
    }

    /// Incremental: 16-token forward split as 8-token prefill + 8-token
    /// decode matches one-shot 16-token cached.
    #[test]
    fn cached_indexer_split_prefill_decode_equals_oneshot() {
        let n_total = 16;
        let (
            x,
            p,
            attn_norm,
            wq_a,
            q_a_norm,
            wq_b,
            wkv,
            kv_a_norm,
            wo_a,
            wo_b,
            attn_sinks,
            comp_wkv,
            comp_wgate,
            comp_ape,
            comp_norm,
            idx_comp_wkv,
            idx_comp_wgate,
            idx_comp_ape,
            idx_comp_norm,
            idx_wq_b,
            idx_wproj,
        ) = build_indexer_test_setup(n_total);
        let w = DsV4AttnBlockIndexerWeights {
            attn: DsV4AttnBlockWeights {
                quant: None,
                attn_norm: &attn_norm,
                wq_a: wq_a.view(),
                q_a_norm: &q_a_norm,
                wq_b: wq_b.view(),
                wkv: wkv.view(),
                kv_a_norm: &kv_a_norm,
                wo_a: wo_a.view(),
                wo_b: wo_b.view(),
                attn_sinks: Some(attn_sinks.view()),
            },
            compressor: CompressorWeights {
                wkv: comp_wkv.view(),
                wgate: comp_wgate.view(),
                ape: comp_ape.view(),
                norm: &comp_norm,
                quant: None,
            },
            indexer_compressor: CompressorWeights {
                wkv: idx_comp_wkv.view(),
                wgate: idx_comp_wgate.view(),
                ape: idx_comp_ape.view(),
                norm: &idx_comp_norm,
                quant: None,
            },
            indexer: IndexerWeights {
                wq_b: idx_wq_b.view(),
                wproj: idx_wproj.view(),
                quant: None,
            },
        };

        // Path A: one-shot.
        let mut cache_a = DsV4LayerHcaCache::with_capacity_and_indexer(
            32,
            p.attn.head_dim,
            4,
            p.indexer.n_index_head_size,
        );
        let out_one =
            dsv4_attn_block_compress_with_indexer_cached(x.view(), &w, &p, 0, &mut cache_a, None);

        // Path B: 8-token prefill + 8-token decode.
        let mut cache_b = DsV4LayerHcaCache::with_capacity_and_indexer(
            32,
            p.attn.head_dim,
            4,
            p.indexer.n_index_head_size,
        );
        let x_pre = x.slice(s![..8, ..]);
        let x_dec = x.slice(s![8.., ..]);
        let out_pre =
            dsv4_attn_block_compress_with_indexer_cached(x_pre, &w, &p, 0, &mut cache_b, None);
        let out_dec =
            dsv4_attn_block_compress_with_indexer_cached(x_dec, &w, &p, 8, &mut cache_b, None);

        let mut max_diff = 0.0_f32;
        for t in 0..8 {
            for d in 0..p.attn.n_embd {
                max_diff = max_diff.max((out_one[[t, d]] - out_pre[[t, d]]).abs());
            }
        }
        for t in 0..8 {
            for d in 0..p.attn.n_embd {
                max_diff = max_diff.max((out_one[[8 + t, d]] - out_dec[[t, d]]).abs());
            }
        }
        assert!(
            max_diff < 1e-5,
            "split-call vs one-shot cached indexer: max diff = {max_diff}"
        );
        assert_eq!(cache_a.raw.current_len(), n_total);
        assert_eq!(cache_b.raw.current_len(), n_total);
        assert_eq!(cache_a.compressed.current_len(), n_total / 4);
        assert_eq!(cache_b.compressed.current_len(), n_total / 4);
    }

    /// Missing indexer cache panics with diagnostic.
    #[test]
    #[should_panic(expected = "cache.index_compressed must be initialized")]
    fn cached_indexer_uninitialized_indexer_panics() {
        let (
            x,
            p,
            attn_norm,
            wq_a,
            q_a_norm,
            wq_b,
            wkv,
            kv_a_norm,
            wo_a,
            wo_b,
            attn_sinks,
            comp_wkv,
            comp_wgate,
            comp_ape,
            comp_norm,
            idx_comp_wkv,
            idx_comp_wgate,
            idx_comp_ape,
            idx_comp_norm,
            idx_wq_b,
            idx_wproj,
        ) = build_indexer_test_setup(4);
        let w = DsV4AttnBlockIndexerWeights {
            attn: DsV4AttnBlockWeights {
                quant: None,
                attn_norm: &attn_norm,
                wq_a: wq_a.view(),
                q_a_norm: &q_a_norm,
                wq_b: wq_b.view(),
                wkv: wkv.view(),
                kv_a_norm: &kv_a_norm,
                wo_a: wo_a.view(),
                wo_b: wo_b.view(),
                attn_sinks: Some(attn_sinks.view()),
            },
            compressor: CompressorWeights {
                wkv: comp_wkv.view(),
                wgate: comp_wgate.view(),
                ape: comp_ape.view(),
                norm: &comp_norm,
                quant: None,
            },
            indexer_compressor: CompressorWeights {
                wkv: idx_comp_wkv.view(),
                wgate: idx_comp_wgate.view(),
                ape: idx_comp_ape.view(),
                norm: &idx_comp_norm,
                quant: None,
            },
            indexer: IndexerWeights {
                wq_b: idx_wq_b.view(),
                wproj: idx_wproj.view(),
                quant: None,
            },
        };
        // Cache without enable_indexer → index_compressed is None → panic.
        let mut cache = DsV4LayerHcaCache::with_capacity(16, p.attn.head_dim, 4);
        let _ = dsv4_attn_block_compress_with_indexer_cached(x.view(), &w, &p, 0, &mut cache, None);
    }
}
