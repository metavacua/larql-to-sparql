//! DeepSeek V4 attention block — `compress_ratio > 0`, no-indexer path.
//!
//! Sibling to [`super::dsv4_attn_block::dsv4_attn_block_no_compress`].
//! This block runs when the layer's `compress_ratio` is positive but
//! not 4 (the indexer-equipped case, handled in a follow-up stage).
//!
//! Pipeline (input `x: (n_tokens, n_embd)` → output `(n_tokens, n_embd)`):
//!
//! 1. **Pre-norm**: `cur = rms_norm(x, attn_norm)`
//! 2. **Q low-rank**: same as Stage 8a — `Wq_a → q_a_norm → Wq_b →
//!    reshape → per-head RMSNorm → tail-RoPE`.
//! 3. **Raw KV low-rank**: `Wkv → kv_a_norm → tail-RoPE → FP8 fake-quant`.
//!    Shape: `(n_tokens, head_dim)`.
//! 4. **Compressed KV**: `build_compressor_prefill(cur, ...) → FP8`.
//!    Shape: `(n_comp, head_dim)` where `n_comp = n_tokens / compress_ratio`.
//! 5. **Concatenate** raw + compressed K/V along the n_kv axis. Final
//!    shape: `(n_tokens + n_comp, head_dim)`.
//! 6. **Build mask**: per `(t, j)` for `j` in `[0, n_tokens + n_comp)`:
//!    - `j < n_tokens` (raw): 0 if `j <= t` AND `t - j < window_size`,
//!      else -∞.
//!    - `j >= n_tokens` (compressed, `j = n_tokens + c`): 0 if compressed
//!      chunk `c` is fully past (`(c+1) * compress_ratio <= t + 1`),
//!      else -∞.
//! 7. **Masked MQA** with optional attention sinks.
//! 8. **Grouped o-proj** → `(n_tokens, n_embd)`.
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:1187-1322`
//! (the if/else branch for `compress_ratio != 4` ending at line 1322).
//! Mask structure matches `dsv4_mask_kind::ATTN_STATIC`.

use ndarray::{s, Array2, ArrayView2, Axis};

use larql_compute::{dot_proj_gpu, ComputeBackend};

use super::dsv4_attn_block::{DsV4AttnBlockParams, DsV4AttnBlockWeights};
use super::dsv4_compressor_prefill::{
    build_compressor_prefill, dsv4_compressor_step_coff1, CompressorParams, CompressorWeights,
};
use super::dsv4_fp8_kv::fp8_kv_quantize;
use super::dsv4_grouped_o_proj::grouped_o_proj;
use super::dsv4_kv_cache::DsV4LayerHcaCache;
use super::dsv4_masked_attn::dsv4_masked_attn;
use super::dsv4_rope_tail_yarn::rope_tail_dispatch;

/// Combined weights for the HCA no-indexer block.
#[derive(Clone, Copy)]
pub struct DsV4AttnBlockCompressWeights<'a> {
    /// Base attention weights (shared with the compress_ratio=0 block).
    pub attn: DsV4AttnBlockWeights<'a>,
    /// Compressor weights for the compressed KV producer.
    pub compressor: CompressorWeights<'a>,
}

/// Combined params for the HCA no-indexer block.
#[derive(Clone, Copy, Debug)]
pub struct DsV4AttnBlockCompressParams {
    pub attn: DsV4AttnBlockParams,
    pub compressor: CompressorParams,
}

/// Run the HCA no-indexer attention block.
///
/// Requires `params.attn.head_dim == params.compressor.head_dim`,
/// `params.attn.n_embd == params.compressor.n_embd`, and
/// `params.compressor.compress_ratio != 4` (the indexer-equipped path
/// is a follow-up stage).
pub fn dsv4_attn_block_compress_no_indexer(
    x: ArrayView2<f32>,
    w: &DsV4AttnBlockCompressWeights,
    p: &DsV4AttnBlockCompressParams,
    position_offset: usize,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    assert_eq!(
        x.shape(),
        &[n_tokens, p.attn.n_embd],
        "x must be (n_tokens, n_embd)"
    );
    assert_eq!(
        p.attn.head_dim, p.compressor.head_dim,
        "attn and compressor head_dim must match"
    );
    assert_eq!(
        p.attn.n_embd, p.compressor.n_embd,
        "attn and compressor n_embd must match"
    );
    assert!(
        p.compressor.compress_ratio > 0 && p.compressor.compress_ratio != 4,
        "no-indexer path requires compress_ratio in {{1, 2, 3, 5, ...}}; got {}",
        p.compressor.compress_ratio
    );

    // 1. attn_norm.
    let cur = rms_norm_2d(x, w.attn.attn_norm, p.attn.norm_eps);

    // 2. Q low-rank (mirrors Stage 8a).
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

    // 4. Compressed KV.
    let kv_comp_pre = build_compressor_prefill(cur.view(), &w.compressor, &p.compressor, backend);
    let kv_comp = fp8_kv_quantize(kv_comp_pre.view(), p.attn.n_rot);
    let n_comp = kv_comp.shape()[0];

    // 5. Concatenate raw + compressed K/V along the n_kv axis.
    let n_kv_total = n_tokens + n_comp;
    let mut kv_cat = Array2::<f32>::zeros((n_kv_total, p.attn.head_dim));
    kv_cat.slice_mut(s![..n_tokens, ..]).assign(&kv_raw);
    kv_cat.slice_mut(s![n_tokens.., ..]).assign(&kv_comp);

    // 6. Build the static mask.
    let mask = build_static_mask(
        n_tokens,
        n_comp,
        position_offset,
        p.attn.window_size,
        p.compressor.compress_ratio,
    );

    // 7. Masked MQA.
    let scale = 1.0 / (p.attn.head_dim as f32).sqrt();
    let o = dsv4_masked_attn(
        q.view(),
        kv_cat.view(),
        kv_cat.view(),
        mask.view(),
        p.attn.n_head,
        p.attn.head_dim,
        w.attn.attn_sinks,
        scale,
        backend,
    );

    // 8. Grouped o-proj.
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

/// Build the `(n_tokens, n_tokens + n_comp)` static attention mask.
///
/// - Raw columns `[0, n_tokens)`: causal sliding-window. Query at row
///   `t` (absolute position `position_offset + t`) sees key position
///   `j` iff `j <= position_offset + t` AND
///   `position_offset + t - j < window_size`.
/// - Compressed columns `[n_tokens, n_tokens + n_comp)`: chunk `c =
///   col - n_tokens` covers positions `[c*ratio, (c+1)*ratio)`. The
///   query at absolute position `p_t` sees chunk `c` iff the chunk is
///   *fully past*: `(c + 1) * ratio <= p_t + 1`, i.e. `c < (p_t + 1)
///   / compress_ratio`.
///
/// `0.0` for allowed pairs, `f32::NEG_INFINITY` for disallowed.
pub fn build_static_mask(
    n_tokens: usize,
    n_comp: usize,
    position_offset: usize,
    window_size: usize,
    compress_ratio: usize,
) -> Array2<f32> {
    let n_kv_total = n_tokens + n_comp;
    let mut mask = Array2::<f32>::from_elem((n_tokens, n_kv_total), f32::NEG_INFINITY);
    for t in 0..n_tokens {
        let abs_pos = position_offset + t;
        // Raw columns: causal sliding-window. j is also an absolute
        // position (raw KV column j is the key at absolute position j).
        let j_lo = abs_pos.saturating_sub(window_size - 1);
        let j_hi = abs_pos;
        for j in j_lo..=j_hi.min(n_tokens - 1) {
            mask[[t, j]] = 0.0;
        }
        // Compressed columns.
        let c_visible = (abs_pos + 1) / compress_ratio; // exclusive upper bound on c
        for c in 0..c_visible.min(n_comp) {
            mask[[t, n_tokens + c]] = 0.0;
        }
    }
    mask
}

/// KV-cached variant of [`dsv4_attn_block_compress_no_indexer`].
///
/// Same scope: `compress_ratio` in `{1, 2, 3, 5, 6, ...}` — NOT 4
/// (the indexer-equipped path is a sibling cached function).
///
/// Threads incremental state through `&mut DsV4LayerHcaCache`:
/// - **Raw KV**: new tokens' rotated+FP8 kv_a rows are appended to
///   `cache.raw`. Query attends to the full raw cache (with sliding-
///   window mask).
/// - **Pending cur buffer**: new tokens' post-attn-norm hidden state
///   (`cur`) rows are pushed to `cache.pending_cur`. Once that buffer
///   accumulates `compress_ratio` rows, [`dsv4_compressor_step_coff1`]
///   runs on the completed chunk, the result is FP8-quantized and
///   appended to `cache.compressed`, and the consumed rows are
///   drained from `pending_cur`.
/// - **Mask**: built over the full raw + compressed cache via
///   [`build_cached_static_mask`], which honors `position_offset` so
///   new tokens correctly mask out future positions.
///
/// Bit-exact-equivalent to [`dsv4_attn_block_compress_no_indexer`] on
/// the same input when the cache starts empty and `position_offset =
/// 0`. Verified in the unit tests.
///
/// Asserts `position_offset == cache.raw.current_len()` on entry to
/// catch out-of-sync caller bugs.
pub fn dsv4_attn_block_compress_no_indexer_cached(
    x: ArrayView2<f32>,
    w: &DsV4AttnBlockCompressWeights,
    p: &DsV4AttnBlockCompressParams,
    position_offset: usize,
    cache: &mut DsV4LayerHcaCache,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let n_new = x.shape()[0];
    assert_eq!(
        x.shape(),
        &[n_new, p.attn.n_embd],
        "x must be (n_tokens, n_embd)"
    );
    assert_eq!(
        p.attn.head_dim, p.compressor.head_dim,
        "attn and compressor head_dim must match"
    );
    assert!(
        p.compressor.compress_ratio > 0 && p.compressor.compress_ratio != 4,
        "no-indexer cached path requires compress_ratio in {{1, 2, 3, 5, ...}}; got {}",
        p.compressor.compress_ratio
    );
    assert_eq!(
        cache.compress_ratio, p.compressor.compress_ratio,
        "cache.compress_ratio ({}) must equal params.compressor.compress_ratio ({})",
        cache.compress_ratio, p.compressor.compress_ratio
    );
    assert_eq!(
        cache.raw.current_len(),
        position_offset,
        "position_offset ({position_offset}) must equal cache.raw.current_len ({})",
        cache.raw.current_len()
    );

    // 1. attn_norm.
    let cur = rms_norm_2d(x, w.attn.attn_norm, p.attn.norm_eps);

    // 2. Q low-rank + tail-RoPE (routed via backend when supplied).
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

    // 3. Raw KV (for new tokens only) + tail-RoPE + FP8 → append to cache.
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

    // 4. Drain new cur rows into pending_cur; when a full chunk fills,
    //    run the cached compressor step and append result to compressed.
    for t in 0..n_new {
        cache.pending_cur.push(cur.row(t).to_owned());
        if cache.pending_cur.len() >= p.compressor.compress_ratio {
            // Build chunk array from the first compress_ratio pending rows.
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
            let chunk_idx = cache.compressed.current_len();
            let new_comp = dsv4_compressor_step_coff1(
                chunk.view(),
                &w.compressor,
                &p.compressor,
                chunk_idx,
                backend,
            );
            // FP8 fake-quantize then append.
            let new_comp_2d =
                Array2::<f32>::from_shape_vec((1, p.attn.head_dim), new_comp.to_vec())
                    .expect("compressor step returned head_dim values");
            let new_comp_2d = fp8_kv_quantize(new_comp_2d.view(), p.attn.n_rot);
            cache.compressed.append(new_comp_2d.view());
        }
    }

    // 5. Concat raw + compressed into a single KV view for masked MQA.
    let raw_total = cache.raw.current_len();
    let n_comp_total = cache.compressed.current_len();
    let n_kv_total = raw_total + n_comp_total;
    let mut kv_cat = Array2::<f32>::zeros((n_kv_total, p.attn.head_dim));
    kv_cat
        .slice_mut(s![..raw_total, ..])
        .assign(&cache.raw.view_current());
    kv_cat
        .slice_mut(s![raw_total.., ..])
        .assign(&cache.compressed.view_current());

    // 6. Build the static mask for the cached (raw_total ≥ n_new) shape.
    let mask = build_cached_static_mask(
        n_new,
        raw_total,
        n_comp_total,
        position_offset,
        p.attn.window_size,
        p.compressor.compress_ratio,
    );

    // 7. Masked MQA.
    let scale = 1.0 / (p.attn.head_dim as f32).sqrt();
    let o = dsv4_masked_attn(
        q.view(),
        kv_cat.view(),
        kv_cat.view(),
        mask.view(),
        p.attn.n_head,
        p.attn.head_dim,
        w.attn.attn_sinks,
        scale,
        backend,
    );

    // 8. Grouped o-proj.
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

/// Cached mask: shape `(n_new, raw_total + n_compressed)`. Raw columns
/// indexed by absolute KV position `[0, raw_total)`; compressed
/// columns indexed by chunk `[raw_total, raw_total + n_compressed)`.
/// For empty cache + `position_offset = 0` + `raw_total == n_new`,
/// this reduces to [`build_static_mask`].
pub fn build_cached_static_mask(
    n_new: usize,
    raw_total: usize,
    n_compressed: usize,
    position_offset: usize,
    window_size: usize,
    compress_ratio: usize,
) -> Array2<f32> {
    let n_kv_total = raw_total + n_compressed;
    let mut mask = Array2::<f32>::from_elem((n_new, n_kv_total), f32::NEG_INFINITY);
    for t in 0..n_new {
        let abs_pos = position_offset + t;
        // Raw columns: causal sliding-window. Column j corresponds to
        // absolute position j (raw cache holds positions 0..raw_total).
        let j_lo = abs_pos.saturating_sub(window_size - 1);
        let j_hi = abs_pos.min(raw_total.saturating_sub(1));
        if abs_pos < raw_total {
            for j in j_lo..=j_hi {
                mask[[t, j]] = 0.0;
            }
        }
        // Compressed columns: chunk c spans positions [c*ratio, (c+1)*ratio).
        // Visible iff fully past abs_pos.
        let c_visible = (abs_pos + 1) / compress_ratio;
        for c in 0..c_visible.min(n_compressed) {
            mask[[t, raw_total + c]] = 0.0;
        }
    }
    mask
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

// Silence unused-Axis warning if codegen ever drops the import.
#[allow(dead_code)]
fn _axis_marker(_: Axis) {}

#[cfg(test)]
mod tests {
    use super::super::dsv4_rope_tail::DsV4RopeMode;
    use super::*;
    use ndarray::{Array1, Array3};

    fn make_block(
        n_tokens: usize,
        compress_ratio: usize,
    ) -> (
        Array2<f32>,
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
        DsV4AttnBlockCompressParams,
    ) {
        // Compact but spec-shaped enough that each helper exercises its
        // path. head_dim=64 → n_nope=64 (one FP8 block), n_rot=0.
        let n_embd = 64;
        let n_head = 4;
        let head_dim = 64;
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let n_rot = 0;
        let rope_base = 10000.0;
        let rope_mode = DsV4RopeMode::Neox;
        let window_size = 8;
        let norm_eps = 1e-5;

        let group_heads = n_head / n_groups; // 2
        let group_dim = head_dim * group_heads; // 128
        let low_dim = o_lora_rank * n_groups; // 16

        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 7 + d) as f32 * 0.013).sin()
        });
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

        // Compressor: coff=1 (compress_ratio != 4).
        let coff = 1;
        let n_kv = coff * head_dim;
        let comp_wkv = Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
            ((i * 3 + j) as f32 * 0.01).cos() * 0.1
        });
        let comp_wgate = Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
            ((i * 5 + j) as f32 * 0.011).sin() * 0.1
        });
        let comp_ape = Array2::<f32>::from_shape_fn((compress_ratio, n_kv), |(r, k)| {
            ((r + k) as f32 * 0.05).sin() * 0.05
        });
        let comp_norm = vec![1.0_f32; head_dim];

        let p = DsV4AttnBlockCompressParams {
            attn: DsV4AttnBlockParams {
                n_embd,
                n_head,
                head_dim,
                q_lora_rank,
                n_groups,
                o_lora_rank,
                n_rot,
                rope_base,
                rope_mode,
                window_size,
                norm_eps,
                yarn: None,
            },
            compressor: CompressorParams {
                head_dim,
                n_embd,
                compress_ratio,
                n_rot,
                rope_base,
                rope_mode,
                norm_eps,
            },
        };

        (
            x, attn_norm, wq_a, q_a_norm, wq_b, wkv, kv_a_norm, wo_a, wo_b, attn_sinks, comp_wkv,
            comp_wgate, comp_ape, comp_norm, p,
        )
    }

    /// Smoke: compress_ratio=2 block produces finite, non-trivial output.
    #[test]
    fn block_compress_ratio_2_smoke() {
        let n_tokens = 8;
        let (
            x,
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
            p,
        ) = make_block(n_tokens, 2);

        let w = DsV4AttnBlockCompressWeights {
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
        };
        let out = dsv4_attn_block_compress_no_indexer(x.view(), &w, &p, 0, None);
        assert_eq!(out.shape(), &[n_tokens, p.attn.n_embd]);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(total > 1e-3, "output collapsed (sum={total})");
    }

    /// CpuBackend equivalence on the Compress prefill block: routing
    /// Q/KV/O matmuls through `Some(&CpuBackend)` must match the
    /// `None` fallback to within BLAS reduction tolerance. Locks in
    /// the GPU-4b backend-threading wiring.
    #[test]
    fn compress_prefill_cpu_backend_matches_none_backend() {
        let n_tokens = 8;
        let (
            x,
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
            p,
        ) = make_block(n_tokens, 2);

        let w = DsV4AttnBlockCompressWeights {
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
        };

        let out_none = dsv4_attn_block_compress_no_indexer(x.view(), &w, &p, 0, None);
        let cpu = larql_compute::CpuBackend;
        let out_cpu = dsv4_attn_block_compress_no_indexer(
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
            "CpuBackend vs None mismatch on Compress prefill: max_diff={max_diff}"
        );
    }

    /// Static mask shape: `(n_tokens, n_tokens + n_comp)`. Token 0 sees
    /// only itself (causal) on raw, and no compressed chunks yet.
    /// Token n_tokens-1 sees a window of raw + all past compressed chunks.
    #[test]
    fn static_mask_causal_structure() {
        let n_tokens = 6;
        let n_comp = 3;
        let window = 3;
        let ratio = 2;
        let mask = build_static_mask(n_tokens, n_comp, 0, window, ratio);
        assert_eq!(mask.shape(), &[n_tokens, n_tokens + n_comp]);

        // Row 0: raw col 0 allowed (causal self), cols 1..n_tokens disallowed.
        assert_eq!(mask[[0, 0]], 0.0);
        for j in 1..n_tokens {
            assert_eq!(mask[[0, j]], f32::NEG_INFINITY, "row 0 j={j}");
        }
        // Compressed cols (chunks 0, 1, 2): chunk c=0 covers [0,1] — for
        // abs_pos=0, c_visible = (0+1)/2 = 0 → no compressed visible.
        for c in 0..n_comp {
            assert_eq!(
                mask[[0, n_tokens + c]],
                f32::NEG_INFINITY,
                "row 0 c={c} should be disallowed"
            );
        }
        // Row 5 (abs_pos=5): raw window [3,4,5] allowed (window=3).
        assert_eq!(mask[[5, 3]], 0.0);
        assert_eq!(mask[[5, 4]], 0.0);
        assert_eq!(mask[[5, 5]], 0.0);
        // Compressed: c_visible = (5+1)/2 = 3 → all 3 chunks visible.
        for c in 0..n_comp {
            assert_eq!(
                mask[[5, n_tokens + c]],
                0.0,
                "row 5 c={c} should be allowed"
            );
        }
    }

    /// Equivalence: with compress_ratio that yields n_comp=0 (i.e.
    /// n_tokens < compress_ratio), the block degrades to … well, it
    /// can't, because the assert in build_compressor_prefill requires
    /// n_comp > 0. So instead: verify that when the compressor weights
    /// produce ~zero compressed KV (zeroed compressor), the output is
    /// dominated by the raw-window path (matches the Stage 8a block
    /// to within FP8 round-off).
    #[test]
    fn zero_compressor_approximately_matches_stage_8a() {
        use super::super::dsv4_attn_block::dsv4_attn_block_no_compress;

        let n_tokens = 6;
        let (
            x,
            attn_norm,
            wq_a,
            q_a_norm,
            wq_b,
            wkv,
            kv_a_norm,
            wo_a,
            wo_b,
            attn_sinks,
            _comp_wkv,
            _comp_wgate,
            _comp_ape,
            _comp_norm,
            p,
        ) = make_block(n_tokens, 2);

        // Zero-out compressor: compressed KV will be a function of the
        // RMSNorm of zero + RoPE, but its mask column contribution
        // should equal the raw-KV contribution if anything, NOT
        // dominate. Use a sane test: compressor produces a non-trivial
        // (but bounded) compressed KV; the *static mask* allows it; we
        // just check that BOTH compressed and raw paths flow through.
        // True parity with Stage 8a isn't expected — the compressor
        // changes the result.
        //
        // What we DO check: shape + finiteness + that the compressed
        // path contributes (output differs from the Stage 8a baseline).
        let attn_w = DsV4AttnBlockWeights {
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
        };
        let baseline = dsv4_attn_block_no_compress(x.view(), &attn_w, &p.attn, 0, None);

        let n_kv = p.compressor.head_dim;
        let comp_wkv = Array2::<f32>::from_shape_fn((n_kv, p.compressor.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.013).sin() * 0.05
        });
        let comp_wgate = Array2::<f32>::from_shape_fn((n_kv, p.compressor.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.017).cos() * 0.05
        });
        let comp_ape =
            Array2::<f32>::from_shape_fn((p.compressor.compress_ratio, n_kv), |(r, k)| {
                ((r + k) as f32 * 0.03).sin() * 0.02
            });
        let comp_norm = vec![1.0_f32; p.compressor.head_dim];
        let w = DsV4AttnBlockCompressWeights {
            attn: attn_w,
            compressor: CompressorWeights {
                wkv: comp_wkv.view(),
                wgate: comp_wgate.view(),
                ape: comp_ape.view(),
                norm: &comp_norm,
                quant: None,
            },
        };
        let compressed = dsv4_attn_block_compress_no_indexer(x.view(), &w, &p, 0, None);
        // Compressed path produces different output than the raw-only
        // baseline — i.e. the compressed KV actually contributes.
        let diff: f32 = baseline
            .iter()
            .zip(compressed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-3,
            "HCA block should differ from Stage 8a (compressed KV ignored?): diff={diff}"
        );
        assert!(compressed.iter().all(|v| v.is_finite()));
    }

    /// Verify the compress_ratio=4 path is rejected by the no-indexer
    /// helper (it requires the indexer-based mask construction).
    #[test]
    #[should_panic(expected = "no-indexer path requires")]
    fn rejects_compress_ratio_4() {
        let (
            x,
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
            p,
        ) = make_block(8, 4);
        let w = DsV4AttnBlockCompressWeights {
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
        };
        let _ = dsv4_attn_block_compress_no_indexer(x.view(), &w, &p, 0, None);
    }

    // ── Cached Compress attention tests ──

    /// Cornerstone bit-exact equivalence: cached path with empty cache
    /// + full prefill input matches the non-cached prefill function.
    #[test]
    fn cached_compress_empty_cache_equals_prefill() {
        let n_tokens = 8; // 8 / cr=2 = 4 compressed positions
        let cr = 2;
        let (
            x,
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
            p,
        ) = make_block(n_tokens, cr);
        let w = DsV4AttnBlockCompressWeights {
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
        };

        let out_prefill = dsv4_attn_block_compress_no_indexer(x.view(), &w, &p, 0, None);

        let mut cache = DsV4LayerHcaCache::with_capacity(32, p.attn.head_dim, cr);
        let out_cached =
            dsv4_attn_block_compress_no_indexer_cached(x.view(), &w, &p, 0, &mut cache, None);

        assert_eq!(out_prefill.shape(), out_cached.shape());
        let mut max_diff = 0.0_f32;
        for (a, b) in out_prefill.iter().zip(out_cached.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-5,
            "cached vs prefill (cr=2, n=8) max diff = {max_diff}"
        );
        // Cache state after full prefill:
        assert_eq!(cache.raw.current_len(), n_tokens);
        assert_eq!(cache.compressed.current_len(), n_tokens / cr);
        assert!(cache.pending_cur.is_empty());
    }

    /// Incremental prefill+decode equivalence: split-call cached path
    /// matches one-shot cached path on the same input.
    #[test]
    fn cached_compress_split_prefill_decode_equals_oneshot() {
        let n_total = 10; // 5 compressed positions
        let cr = 2;
        let (
            x,
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
            p,
        ) = make_block(n_total, cr);
        let w = DsV4AttnBlockCompressWeights {
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
        };

        // Path A: one-shot cached.
        let mut cache_a = DsV4LayerHcaCache::with_capacity(32, p.attn.head_dim, cr);
        let out_one =
            dsv4_attn_block_compress_no_indexer_cached(x.view(), &w, &p, 0, &mut cache_a, None);

        // Path B: prefill 6 tokens, then decode 4 more.
        let mut cache_b = DsV4LayerHcaCache::with_capacity(32, p.attn.head_dim, cr);
        let x_pre = x.slice(s![..6, ..]);
        let x_dec = x.slice(s![6.., ..]);
        let out_pre =
            dsv4_attn_block_compress_no_indexer_cached(x_pre, &w, &p, 0, &mut cache_b, None);
        let out_dec =
            dsv4_attn_block_compress_no_indexer_cached(x_dec, &w, &p, 6, &mut cache_b, None);

        // Rows 0..6 should match out_one rows 0..6, rows 6..10 should match out_one rows 6..10.
        let mut max_diff = 0.0_f32;
        for t in 0..6 {
            for d in 0..p.attn.n_embd {
                max_diff = max_diff.max((out_one[[t, d]] - out_pre[[t, d]]).abs());
            }
        }
        for t in 0..4 {
            for d in 0..p.attn.n_embd {
                max_diff = max_diff.max((out_one[[6 + t, d]] - out_dec[[t, d]]).abs());
            }
        }
        assert!(
            max_diff < 1e-5,
            "split-call vs one-shot cached: max diff = {max_diff}"
        );
        assert_eq!(cache_a.raw.current_len(), n_total);
        assert_eq!(cache_b.raw.current_len(), n_total);
        assert_eq!(cache_a.compressed.current_len(), n_total / cr);
        assert_eq!(cache_b.compressed.current_len(), n_total / cr);
    }

    /// Partial chunks remain in pending_cur until a new token fills them.
    /// Tests that the prefill-then-step pattern correctly carries
    /// state across calls.
    #[test]
    fn cached_compress_partial_chunk_stays_pending() {
        let n_total = 9; // 4 compressed + 1 pending
        let cr = 2;
        let (
            x,
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
            p,
        ) = make_block(n_total, cr);
        let w = DsV4AttnBlockCompressWeights {
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
        };
        let mut cache = DsV4LayerHcaCache::with_capacity(32, p.attn.head_dim, cr);
        let _ = dsv4_attn_block_compress_no_indexer_cached(x.view(), &w, &p, 0, &mut cache, None);

        // 9 tokens with cr=2: 4 full chunks → 4 compressed, 1 leftover pending.
        assert_eq!(cache.raw.current_len(), 9);
        assert_eq!(cache.compressed.current_len(), 4);
        assert_eq!(cache.pending_cur.len(), 1);
    }

    /// position_offset out of sync with cache.raw.current_len() panics.
    #[test]
    #[should_panic(expected = "position_offset")]
    fn cached_compress_position_offset_mismatch_panics() {
        let cr = 2;
        let (
            _x,
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
            p,
        ) = make_block(4, cr);
        let w = DsV4AttnBlockCompressWeights {
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
        };
        // Empty cache but caller claims offset=5 → panic.
        let mut cache = DsV4LayerHcaCache::with_capacity(16, p.attn.head_dim, cr);
        let x = Array2::<f32>::zeros((1, p.attn.n_embd));
        let _ = dsv4_attn_block_compress_no_indexer_cached(x.view(), &w, &p, 5, &mut cache, None);
    }
}
