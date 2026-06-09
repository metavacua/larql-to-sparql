//! DeepSeek V4 compressor prefill builder — produces compressed KV rows
//! for the HCA (compress_ratio>0) attention path.
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:627-703`
//! (`dsv4_build_compressor_prefill`).
//!
//! Pipeline:
//!
//! 1. Project the residual to `kv` and `score`, each shape
//!    `(n_tokens, coff * head_dim)`.
//! 2. Take the first `n_comp * compress_ratio` tokens (drop the
//!    remainder) and view as `(n_comp, compress_ratio, coff * head_dim)`.
//! 3. Add per-chunk-position APE bias to the score.
//! 4. Pool over the chunk dimension:
//!    - **coff = 1** (compress_ratio ≠ 4): permute to
//!      `(n_comp, head_dim, ratio)` and run [`softmax_pool_ratio`].
//!    - **coff = 2** (compress_ratio = 4): split the head-dim axis into
//!      `prev` and `curr` halves. Apply [`shift_overlap_state`] to the
//!      prev halves (kv padded with 0.0, score with -∞). Concat
//!      prev+curr along the ratio axis → `(n_comp, head_dim, 2*ratio)`,
//!      then [`softmax_pool_ratio`].
//! 5. Apply RMSNorm (with learned `norm` gain) along the head_dim axis.
//! 6. Apply tail-RoPE at positions `[0, ratio, 2*ratio, ...,
//!    (n_comp-1)*ratio]` — the absolute token positions of each
//!    compressed-chunk anchor.
//!
//! Output shape: `(n_comp, head_dim)` — one compressed KV row per chunk.

use ndarray::{s, Array2, Array3, ArrayView2};

use larql_compute::{dot_proj_gpu, ComputeBackend};
use larql_models::quant::lazy::QuantTensor;

use super::dsv4_compressor::{shift_overlap_state, softmax_pool_ratio};
use super::dsv4_rope_tail::{dsv4_rope_tail, DsV4RopeMode};

/// Resident-Q4_K companions for the compressor projection weights (P8).
///
/// `wkv`/`wgate` (`attn_compress_kv/gate`, and the indexer's
/// `indexer.compress_kv/gate`) are Q4_K in the GGUF. After P1–P7 they were
/// the last large weights still dequantized to f32 on the HCA decode path
/// (~16.8 MB f32/layer for the main compressor × the ~41 Compress+Indexer
/// layers ≈ ~688 MB/decode-step, plus the indexer sub-compressor). Holding
/// them resident (mirrors P5's `DsV4AttnQuant` / P7's `IndexerQuant`) and
/// running the lazy Q4_K×Q8_K matmul cuts that read ~3.5×.
#[derive(Clone, Copy)]
pub struct CompressorQuant<'a> {
    /// `(coff*head_dim, n_embd)` KV projection.
    pub wkv: &'a QuantTensor,
    /// `(coff*head_dim, n_embd)` gate/score projection.
    pub wgate: &'a QuantTensor,
}

/// Per-layer compressor weight refs.
#[derive(Clone, Copy)]
pub struct CompressorWeights<'a> {
    /// `(coff * head_dim, n_embd)` — KV projection.
    pub wkv: ArrayView2<'a, f32>,
    /// `(coff * head_dim, n_embd)` — gate / score projection.
    pub wgate: ArrayView2<'a, f32>,
    /// `(compress_ratio, coff * head_dim)` — absolute position embedding
    /// added to the score before softmax pooling.
    pub ape: ArrayView2<'a, f32>,
    /// `(head_dim,)` — RMSNorm gain applied to the pooled output.
    pub norm: &'a [f32],
    /// Resident-Q4_K companions for `wkv`/`wgate` (P8). `None` = use the
    /// f32 views above (streaming path).
    pub quant: Option<CompressorQuant<'a>>,
}

impl<'a> CompressorWeights<'a> {
    /// `x @ wkv^T`: resident-Q4_K lazy matmul when `quant` is set, else
    /// the f32 `dot_proj_gpu` path.
    pub fn proj_wkv(
        &self,
        x: ArrayView2<f32>,
        backend: Option<&dyn ComputeBackend>,
    ) -> Array2<f32> {
        match &self.quant {
            Some(q) => q
                .wkv
                .matmul(&x.to_owned())
                .expect("compressor wkv quant matmul"),
            None => dot_proj_gpu(&x, &self.wkv, backend),
        }
    }
    /// `x @ wgate^T` (resident-Q4_K when available).
    pub fn proj_wgate(
        &self,
        x: ArrayView2<f32>,
        backend: Option<&dyn ComputeBackend>,
    ) -> Array2<f32> {
        match &self.quant {
            Some(q) => q
                .wgate
                .matmul(&x.to_owned())
                .expect("compressor wgate quant matmul"),
            None => dot_proj_gpu(&x, &self.wgate, backend),
        }
    }
}

/// Compressor scalar config.
#[derive(Clone, Copy, Debug)]
pub struct CompressorParams {
    pub head_dim: usize,
    pub n_embd: usize,
    pub compress_ratio: usize,
    pub n_rot: usize,
    pub rope_base: f64,
    pub rope_mode: DsV4RopeMode,
    pub norm_eps: f32,
}

/// Build the compressed KV table for a single layer (prefill).
///
/// `x`: `(n_tokens, n_embd)`. Returns `(n_comp, head_dim)` where
/// `n_comp = n_tokens / compress_ratio` (any remainder is dropped, per
/// llama.cpp reference).
pub fn build_compressor_prefill(
    x: ArrayView2<f32>,
    w: &CompressorWeights,
    p: &CompressorParams,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    assert!(p.compress_ratio > 0, "compress_ratio must be > 0");
    let n_tokens = x.shape()[0];
    assert_eq!(x.shape()[1], p.n_embd, "x feature dim must equal n_embd");
    let n_comp = n_tokens / p.compress_ratio;
    assert!(
        n_comp > 0,
        "need n_tokens >= compress_ratio (got {n_tokens})"
    );

    let coff: usize = if p.compress_ratio == 4 { 2 } else { 1 };
    let n_kv = coff * p.head_dim;
    if w.quant.is_none() {
        assert_eq!(w.wkv.shape(), &[n_kv, p.n_embd], "wkv shape");
        assert_eq!(w.wgate.shape(), &[n_kv, p.n_embd], "wgate shape");
    }
    assert_eq!(
        w.ape.shape(),
        &[p.compress_ratio, n_kv],
        "ape shape (ratio, coff*head_dim)"
    );
    assert_eq!(w.norm.len(), p.head_dim, "norm length must equal head_dim");

    // 1. Project to kv and score, each (n_tokens, n_kv).
    let kv_full = w.proj_wkv(x, backend);
    let score_full = w.proj_wgate(x, backend);

    // 2. Reshape the first cutoff = n_comp * compress_ratio tokens to
    //    (n_comp, ratio, n_kv).
    let cutoff = n_comp * p.compress_ratio;
    let mut kv_chunks = Array3::<f32>::zeros((n_comp, p.compress_ratio, n_kv));
    let mut score_chunks = Array3::<f32>::zeros((n_comp, p.compress_ratio, n_kv));
    for t in 0..cutoff {
        let c = t / p.compress_ratio;
        let r = t % p.compress_ratio;
        for k in 0..n_kv {
            kv_chunks[[c, r, k]] = kv_full[[t, k]];
            score_chunks[[c, r, k]] = score_full[[t, k]];
        }
    }

    // 3. Add APE to score (broadcast over n_comp).
    for c in 0..n_comp {
        for r in 0..p.compress_ratio {
            for k in 0..n_kv {
                score_chunks[[c, r, k]] += w.ape[[r, k]];
            }
        }
    }

    // 4. Pool over the chunk dim.
    let pooled = if coff == 1 {
        // (n_comp, ratio, head_dim) → permute to (n_comp, head_dim, ratio).
        let kv_p = kv_chunks
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        let score_p = score_chunks
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        softmax_pool_ratio(kv_p.view(), score_p.view())
    } else {
        // coff == 2: split head-dim axis into prev (first head_dim) and
        // curr (next head_dim).
        let kv_prev = kv_chunks.slice(s![.., .., 0..p.head_dim]).to_owned();
        let kv_curr = kv_chunks
            .slice(s![.., .., p.head_dim..2 * p.head_dim])
            .to_owned();
        let score_prev = score_chunks.slice(s![.., .., 0..p.head_dim]).to_owned();
        let score_curr = score_chunks
            .slice(s![.., .., p.head_dim..2 * p.head_dim])
            .to_owned();

        // Shift the prev halves back by one along n_comp (pad 0.0 / -INF).
        let kv_prev_shifted = shift_overlap_state(kv_prev.view(), 0.0);
        let score_prev_shifted = shift_overlap_state(score_prev.view(), f32::NEG_INFINITY);

        // Permute each to (n_comp, head_dim, ratio).
        let kv_prev_p = kv_prev_shifted
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        let kv_curr_p = kv_curr
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        let score_prev_p = score_prev_shifted
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        let score_curr_p = score_curr
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();

        // Concat along the ratio axis → (n_comp, head_dim, 2*ratio).
        let mut kv_cat = Array3::<f32>::zeros((n_comp, p.head_dim, 2 * p.compress_ratio));
        let mut score_cat = Array3::<f32>::zeros((n_comp, p.head_dim, 2 * p.compress_ratio));
        for c in 0..n_comp {
            for d in 0..p.head_dim {
                for r in 0..p.compress_ratio {
                    kv_cat[[c, d, r]] = kv_prev_p[[c, d, r]];
                    kv_cat[[c, d, p.compress_ratio + r]] = kv_curr_p[[c, d, r]];
                    score_cat[[c, d, r]] = score_prev_p[[c, d, r]];
                    score_cat[[c, d, p.compress_ratio + r]] = score_curr_p[[c, d, r]];
                }
            }
        }
        softmax_pool_ratio(kv_cat.view(), score_cat.view())
    };
    // `pooled` is (n_comp, head_dim).

    // 5. RMSNorm with learned gain along head_dim.
    let mut normed = Array2::<f32>::zeros((n_comp, p.head_dim));
    for c in 0..n_comp {
        let mut sumsq = 0.0_f32;
        for d in 0..p.head_dim {
            let v = pooled[[c, d]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / p.head_dim as f32 + p.norm_eps).sqrt();
        for d in 0..p.head_dim {
            normed[[c, d]] = pooled[[c, d]] * inv * w.norm[d];
        }
    }

    // 6. Tail-RoPE at positions [0, ratio, 2*ratio, ...]. The CPU
    //    reference applies RoPE row-by-row using the position_offset
    //    argument (each row treated as a single-token slice at its
    //    absolute compressed-chunk position).
    let mut roped = Array2::<f32>::zeros((n_comp, p.head_dim));
    for c in 0..n_comp {
        let row = normed.slice(s![c..c + 1, ..]).to_owned();
        let row_roped = dsv4_rope_tail(
            &row,
            1,
            p.head_dim,
            p.n_rot,
            p.rope_base,
            p.rope_mode,
            false,
            c * p.compress_ratio,
        );
        roped.row_mut(c).assign(&row_roped.row(0));
    }
    roped
}

/// Cached / incremental compressor step for the `coff = 1` case.
///
/// Processes a single chunk of `compress_ratio` cur rows and returns
/// one compressed kv row of shape `(head_dim,)`. Used by the cached
/// HCA Compress attention path to produce one new compressed position
/// each time a chunk completes during streaming decode.
///
/// This is bit-exact equivalent to taking row `compressed_chunk_index`
/// of `build_compressor_prefill(cur_full, ...)` where
/// `cur_full[c*ratio..(c+1)*ratio] == cur_chunk` for c =
/// compressed_chunk_index. The two are equivalent **because for coff=1
/// the compressor's chunks are independent** — no overlap state
/// flows between chunks. The cr=4 path (coff=2) needs overlap state
/// and is handled in a follow-up PR.
///
/// Requires `cur_chunk.shape() == [p.compress_ratio, p.n_embd]` and
/// `p.compress_ratio != 4`.
pub fn dsv4_compressor_step_coff1(
    cur_chunk: ArrayView2<f32>,
    w: &CompressorWeights,
    p: &CompressorParams,
    compressed_chunk_index: usize,
    backend: Option<&dyn ComputeBackend>,
) -> ndarray::Array1<f32> {
    assert!(p.compress_ratio > 0, "compress_ratio must be > 0");
    assert_ne!(
        p.compress_ratio, 4,
        "coff=1 path requires compress_ratio != 4; use the coff=2 cached step for cr=4"
    );
    assert_eq!(
        cur_chunk.shape(),
        &[p.compress_ratio, p.n_embd],
        "cur_chunk must be (compress_ratio, n_embd)"
    );
    let n_kv = p.head_dim; // coff = 1 → n_kv == head_dim
    if w.quant.is_none() {
        assert_eq!(w.wkv.shape(), &[n_kv, p.n_embd], "wkv shape");
        assert_eq!(w.wgate.shape(), &[n_kv, p.n_embd], "wgate shape");
    }
    assert_eq!(
        w.ape.shape(),
        &[p.compress_ratio, n_kv],
        "ape shape (compress_ratio, head_dim)"
    );
    assert_eq!(w.norm.len(), p.head_dim, "norm length");

    // 1. Project chunk to kv and score: each (compress_ratio, head_dim).
    let kv_chunk = w.proj_wkv(cur_chunk, backend);
    let mut score_chunk = w.proj_wgate(cur_chunk, backend);

    // 2. Add APE row-wise.
    for r in 0..p.compress_ratio {
        for k in 0..n_kv {
            score_chunk[[r, k]] += w.ape[[r, k]];
        }
    }

    // 3. Permute (compress_ratio, head_dim) → (1, head_dim, compress_ratio)
    //    so softmax_pool_ratio can pool over the chunk dim.
    let mut kv_3d = Array3::<f32>::zeros((1, p.head_dim, p.compress_ratio));
    let mut score_3d = Array3::<f32>::zeros((1, p.head_dim, p.compress_ratio));
    for d in 0..p.head_dim {
        for r in 0..p.compress_ratio {
            kv_3d[[0, d, r]] = kv_chunk[[r, d]];
            score_3d[[0, d, r]] = score_chunk[[r, d]];
        }
    }

    // 4. Pool → (1, head_dim).
    let pooled = softmax_pool_ratio(kv_3d.view(), score_3d.view());

    // 5. RMSNorm with learned gain on the single row.
    let mut sumsq = 0.0_f32;
    for d in 0..p.head_dim {
        let v = pooled[[0, d]];
        sumsq += v * v;
    }
    let inv = 1.0 / (sumsq / p.head_dim as f32 + p.norm_eps).sqrt();
    let mut normed = Array2::<f32>::zeros((1, p.head_dim));
    for d in 0..p.head_dim {
        normed[[0, d]] = pooled[[0, d]] * inv * w.norm[d];
    }

    // 6. Tail-RoPE at absolute position `compressed_chunk_index * compress_ratio`.
    let roped = dsv4_rope_tail(
        &normed,
        1,
        p.head_dim,
        p.n_rot,
        p.rope_base,
        p.rope_mode,
        false,
        compressed_chunk_index * p.compress_ratio,
    );

    // Return as 1D (head_dim,).
    roped.row(0).to_owned()
}

/// Threaded overlap state for the [`dsv4_compressor_step_coff2`]
/// chunk-by-chunk compressor (compress_ratio == 4 path).
///
/// Holds the previous chunk's `kv_prev` and `score_prev` halves so
/// the next chunk's call has the same "shifted-by-1" inputs as
/// [`build_compressor_prefill`]. For chunk index 0 both fields are
/// `None` and the step uses the same zero / -∞ pad the prefill does.
///
/// Per-layer footprint at DSv4-Flash (head_dim=512, compress_ratio=4):
/// 2 × 4 × 512 × 4 = 16 KB. Trivial.
#[derive(Clone, Debug, Default)]
pub struct CompressorOverlapState {
    /// Last processed chunk's `kv_prev` half. Shape `(compress_ratio,
    /// head_dim)`. `None` before the first chunk.
    pub kv_prev_last: Option<Array2<f32>>,
    /// Last processed chunk's `score_prev` half. Same shape.
    pub score_prev_last: Option<Array2<f32>>,
}

impl CompressorOverlapState {
    /// Empty state — equivalent to "we haven't processed any chunks yet".
    pub fn empty() -> Self {
        Self::default()
    }

    /// Reset state, e.g. to start a fresh sequence.
    pub fn clear(&mut self) {
        self.kv_prev_last = None;
        self.score_prev_last = None;
    }
}

/// Cached / incremental compressor step for the `coff = 2` case
/// (compress_ratio == 4).
///
/// Same role as [`dsv4_compressor_step_coff1`] but the cr=4 path
/// couples chunks via the prefill compressor's `shift_overlap_state`
/// call. To stay bit-exact-equivalent with prefill, this step needs
/// to thread the previous chunk's `kv_prev` / `score_prev` halves
/// through `&mut overlap_state`.
///
/// - First call (chunk 0): pass `CompressorOverlapState::empty()`.
///   The step pads the "shifted prev" with 0.0 / -∞, matching the
///   prefill's chunk-0 behavior. After the call, the state holds
///   chunk 0's kv_prev / score_prev for the next call.
/// - Subsequent calls: pass the same `&mut overlap_state`. The step
///   reads the cached `kv_prev_last` / `score_prev_last` as the
///   shifted prev (equivalent to what the prefill's shift_overlap_state
///   would produce for this chunk index).
///
/// Bit-exact-equivalent to taking row `compressed_chunk_index` of
/// `build_compressor_prefill(cur_full, ...)` where the full cur
/// contains the same per-chunk data.
pub fn dsv4_compressor_step_coff2(
    cur_chunk: ArrayView2<f32>,
    w: &CompressorWeights,
    p: &CompressorParams,
    compressed_chunk_index: usize,
    overlap_state: &mut CompressorOverlapState,
    backend: Option<&dyn ComputeBackend>,
) -> ndarray::Array1<f32> {
    assert_eq!(
        p.compress_ratio, 4,
        "coff=2 path requires compress_ratio == 4; use the coff=1 cached step otherwise"
    );
    assert_eq!(
        cur_chunk.shape(),
        &[p.compress_ratio, p.n_embd],
        "cur_chunk must be (compress_ratio, n_embd)"
    );
    let n_kv = 2 * p.head_dim; // coff = 2
                               // Shape checks apply to the f32 view; in resident-Q4_K mode (P8) the
                               // f32 wkv/wgate are empty and the shape lives in the QuantTensor.
    if w.quant.is_none() {
        assert_eq!(w.wkv.shape(), &[n_kv, p.n_embd], "wkv shape");
        assert_eq!(w.wgate.shape(), &[n_kv, p.n_embd], "wgate shape");
    }
    assert_eq!(
        w.ape.shape(),
        &[p.compress_ratio, n_kv],
        "ape shape (compress_ratio, 2*head_dim)"
    );
    assert_eq!(w.norm.len(), p.head_dim, "norm length");

    // 1. Project chunk to kv and score: each (compress_ratio, 2*head_dim).
    //    Resident-Q4_K lazy matmul when present, else f32 dot_proj_gpu.
    let kv_chunk = w.proj_wkv(cur_chunk, backend);
    let mut score_chunk = w.proj_wgate(cur_chunk, backend);

    // 2. Add APE row-wise.
    for r in 0..p.compress_ratio {
        for k in 0..n_kv {
            score_chunk[[r, k]] += w.ape[[r, k]];
        }
    }

    // 3. Split into prev (first head_dim) and curr (next head_dim) halves.
    //    Each is (compress_ratio, head_dim).
    let kv_curr = kv_chunk
        .slice(s![.., p.head_dim..2 * p.head_dim])
        .to_owned();
    let score_curr = score_chunk
        .slice(s![.., p.head_dim..2 * p.head_dim])
        .to_owned();
    let kv_prev_this_chunk = kv_chunk.slice(s![.., 0..p.head_dim]).to_owned();
    let score_prev_this_chunk = score_chunk.slice(s![.., 0..p.head_dim]).to_owned();

    // 4. The "shifted prev" for this chunk is what the prefill's
    //    shift_overlap_state would produce at this chunk index — i.e.,
    //    the previous chunk's prev half, or zero/-∞ pad for chunk 0.
    let kv_prev_shifted = overlap_state
        .kv_prev_last
        .clone()
        .unwrap_or_else(|| Array2::<f32>::zeros((p.compress_ratio, p.head_dim)));
    let score_prev_shifted = overlap_state.score_prev_last.clone().unwrap_or_else(|| {
        Array2::<f32>::from_elem((p.compress_ratio, p.head_dim), f32::NEG_INFINITY)
    });

    // 5. Concat [shifted_prev, curr] along the ratio axis → (head_dim, 2*ratio).
    //    Then add the n_comp=1 outer axis for softmax_pool_ratio.
    let mut kv_cat = Array3::<f32>::zeros((1, p.head_dim, 2 * p.compress_ratio));
    let mut score_cat = Array3::<f32>::zeros((1, p.head_dim, 2 * p.compress_ratio));
    for d in 0..p.head_dim {
        for r in 0..p.compress_ratio {
            kv_cat[[0, d, r]] = kv_prev_shifted[[r, d]];
            kv_cat[[0, d, p.compress_ratio + r]] = kv_curr[[r, d]];
            score_cat[[0, d, r]] = score_prev_shifted[[r, d]];
            score_cat[[0, d, p.compress_ratio + r]] = score_curr[[r, d]];
        }
    }

    // 6. Update overlap state for the NEXT chunk: store this chunk's
    //    prev halves. (Must be saved AFTER step 5 so the read above
    //    sees the previous chunk's data, not this one's.)
    overlap_state.kv_prev_last = Some(kv_prev_this_chunk);
    overlap_state.score_prev_last = Some(score_prev_this_chunk);

    // 7. Pool → (1, head_dim).
    let pooled = softmax_pool_ratio(kv_cat.view(), score_cat.view());

    // 8. RMSNorm + RoPE — same as the coff=1 path.
    let mut sumsq = 0.0_f32;
    for d in 0..p.head_dim {
        let v = pooled[[0, d]];
        sumsq += v * v;
    }
    let inv = 1.0 / (sumsq / p.head_dim as f32 + p.norm_eps).sqrt();
    let mut normed = Array2::<f32>::zeros((1, p.head_dim));
    for d in 0..p.head_dim {
        normed[[0, d]] = pooled[[0, d]] * inv * w.norm[d];
    }

    let roped = dsv4_rope_tail(
        &normed,
        1,
        p.head_dim,
        p.n_rot,
        p.rope_base,
        p.rope_mode,
        false,
        compressed_chunk_index * p.compress_ratio,
    );

    roped.row(0).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(n_tokens: usize, n_embd: usize) -> Array2<f32> {
        Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 17 + d) as f32 * 0.013).sin()
        })
    }

    /// Smoke: compress_ratio=2 (coff=1) path produces finite, non-zero
    /// output with the expected shape.
    #[test]
    fn coff1_compress_ratio_2_shape_and_finite() {
        let p = CompressorParams {
            head_dim: 16,
            n_embd: 32,
            compress_ratio: 2,
            n_rot: 8,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let coff = 1;
        let n_kv = coff * p.head_dim;
        let n_tokens = 10;

        let wkv = Array2::<f32>::from_shape_fn((n_kv, p.n_embd), |(i, j)| {
            ((i * 3 + j) as f32 * 0.01).cos() * 0.1
        });
        let wgate = Array2::<f32>::from_shape_fn((n_kv, p.n_embd), |(i, j)| {
            ((i * 5 + j) as f32 * 0.011).sin() * 0.1
        });
        let ape = Array2::<f32>::from_shape_fn((p.compress_ratio, n_kv), |(r, k)| {
            ((r + k) as f32 * 0.05).sin() * 0.05
        });
        let norm = vec![1.0_f32; p.head_dim];

        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let x = make_input(n_tokens, p.n_embd);
        let out = build_compressor_prefill(x.view(), &w, &p, None);

        let n_comp = n_tokens / p.compress_ratio;
        assert_eq!(out.shape(), &[n_comp, p.head_dim]);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite");
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(total > 1e-3, "output collapsed (sum={total})");
    }

    /// CpuBackend equivalence: build_compressor_prefill with
    /// `Some(&CpuBackend)` must match the `None` fallback to within
    /// BLAS reduction tolerance. Locks in the GPU-7b backend-routing
    /// wiring on the wkv/wgate matmuls.
    #[test]
    fn compressor_prefill_cpu_backend_matches_none_backend() {
        let p = CompressorParams {
            head_dim: 16,
            n_embd: 32,
            compress_ratio: 2,
            n_rot: 8,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let coff = 1;
        let n_kv = coff * p.head_dim;
        let n_tokens = 12;

        let wkv = Array2::<f32>::from_shape_fn((n_kv, p.n_embd), |(i, j)| {
            ((i * 3 + j) as f32 * 0.01).cos() * 0.1
        });
        let wgate = Array2::<f32>::from_shape_fn((n_kv, p.n_embd), |(i, j)| {
            ((i * 5 + j) as f32 * 0.011).sin() * 0.1
        });
        let ape = Array2::<f32>::from_shape_fn((p.compress_ratio, n_kv), |(r, k)| {
            ((r + k) as f32 * 0.05).sin() * 0.05
        });
        let norm = vec![1.0_f32; p.head_dim];

        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let x = make_input(n_tokens, p.n_embd);

        let out_none = build_compressor_prefill(x.view(), &w, &p, None);
        let cpu = larql_compute::CpuBackend;
        let out_cpu = build_compressor_prefill(x.view(), &w, &p, Some(&cpu as &dyn ComputeBackend));

        assert_eq!(out_none.shape(), out_cpu.shape());
        let max_diff = out_none
            .iter()
            .zip(out_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "CpuBackend vs None mismatch on compressor prefill: max_diff={max_diff}"
        );
    }

    /// coff=2 path (compress_ratio=4) — verify shape, finiteness, and
    /// that the shifted-prev padding doesn't blow up the first row.
    #[test]
    fn coff2_compress_ratio_4_shape_and_finite() {
        let p = CompressorParams {
            head_dim: 16,
            n_embd: 32,
            compress_ratio: 4,
            n_rot: 8,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let coff = 2;
        let n_kv = coff * p.head_dim;
        let n_tokens = 16;

        let wkv = Array2::<f32>::from_shape_fn((n_kv, p.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.013).cos() * 0.1
        });
        let wgate = Array2::<f32>::from_shape_fn((n_kv, p.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.017).sin() * 0.1
        });
        let ape = Array2::<f32>::from_shape_fn((p.compress_ratio, n_kv), |(r, k)| {
            ((r + k) as f32 * 0.03).sin() * 0.05
        });
        let norm = vec![1.0_f32; p.head_dim];

        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let x = make_input(n_tokens, p.n_embd);
        let out = build_compressor_prefill(x.view(), &w, &p, None);

        let n_comp = n_tokens / p.compress_ratio;
        assert_eq!(out.shape(), &[n_comp, p.head_dim]);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite");
        // First row uses the all-pad shifted-prev → ends up driven by
        // curr alone. Should still be finite and non-zero (RoPE + norm
        // never produce all-zero from a non-zero input).
        let row0_norm: f32 = out.row(0).iter().map(|v| v.abs()).sum();
        assert!(row0_norm > 1e-3, "row 0 collapsed (sum={row0_norm})");
    }

    /// Truncation: with n_tokens not divisible by compress_ratio, the
    /// remainder is dropped. Result n_comp = n_tokens / compress_ratio
    /// (floor).
    #[test]
    fn truncates_remainder() {
        let p = CompressorParams {
            head_dim: 8,
            n_embd: 16,
            compress_ratio: 3,
            n_rot: 4,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let coff = 1;
        let n_kv = coff * p.head_dim;
        let n_tokens = 10; // → cutoff=9, n_comp=3, drop 1 token

        let wkv = Array2::<f32>::from_elem((n_kv, p.n_embd), 0.05);
        let wgate = Array2::<f32>::from_elem((n_kv, p.n_embd), 0.03);
        let ape = Array2::<f32>::zeros((p.compress_ratio, n_kv));
        let norm = vec![1.0; p.head_dim];

        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let x = make_input(n_tokens, p.n_embd);
        let out = build_compressor_prefill(x.view(), &w, &p, None);
        assert_eq!(out.shape(), &[3, p.head_dim]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Position differentiation: with translation-invariant weights
    /// (zero APE, identical x rows), the only thing that distinguishes
    /// the compressed rows is tail-RoPE applied at positions
    /// 0, ratio, 2*ratio, ... → consecutive rows should differ.
    #[test]
    fn position_distinguishes_compressed_rows() {
        let p = CompressorParams {
            head_dim: 16,
            n_embd: 32,
            compress_ratio: 2,
            n_rot: 16, // full rotation so position differences are visible
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let coff = 1;
        let n_kv = coff * p.head_dim;
        let n_tokens = 8; // n_comp = 4

        let wkv = Array2::<f32>::from_elem((n_kv, p.n_embd), 0.05);
        let wgate = Array2::<f32>::from_elem((n_kv, p.n_embd), 0.03);
        let ape = Array2::<f32>::zeros((p.compress_ratio, n_kv));
        let norm = vec![1.0; p.head_dim];

        // Identical x rows → same projections per token.
        let x = Array2::<f32>::from_elem((n_tokens, p.n_embd), 0.7);

        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let out = build_compressor_prefill(x.view(), &w, &p, None);
        // Consecutive rows differ purely due to per-row tail-RoPE
        // (rows are at positions 0, 2, 4, 6).
        let diff_01: f32 = (0..p.head_dim)
            .map(|d| (out[[0, d]] - out[[1, d]]).abs())
            .sum();
        let diff_12: f32 = (0..p.head_dim)
            .map(|d| (out[[1, d]] - out[[2, d]]).abs())
            .sum();
        assert!(diff_01 > 1e-3, "rows 0/1 should differ (diff={diff_01})");
        assert!(diff_12 > 1e-3, "rows 1/2 should differ (diff={diff_12})");
    }

    // ── dsv4_compressor_step_coff1 tests ──

    /// Helper: build CompressorParams + weights for coff=1 (compress_ratio != 4).
    fn make_coff1_weights(
        compress_ratio: usize,
        head_dim: usize,
        n_embd: usize,
    ) -> (
        CompressorParams,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Vec<f32>,
    ) {
        assert_ne!(compress_ratio, 4);
        let p = CompressorParams {
            head_dim,
            n_embd,
            compress_ratio,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let n_kv = head_dim; // coff=1
        let wkv = Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
            ((i * 11 + j) as f32 * 0.013).cos() * 0.05
        });
        let wgate = Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
            ((i * 7 + j) as f32 * 0.017).sin() * 0.05
        });
        let ape = Array2::<f32>::from_shape_fn((compress_ratio, n_kv), |(r, k)| {
            ((r * 5 + k) as f32 * 0.03).sin() * 0.05
        });
        let norm = vec![1.0_f32; head_dim];
        (p, wkv, wgate, ape, norm)
    }

    /// Single chunk via the cached step equals row 0 of the prefill
    /// compressor on the same chunk.
    #[test]
    fn coff1_step_single_chunk_matches_prefill() {
        let (p, wkv, wgate, ape, norm) = make_coff1_weights(2, 8, 16);
        let chunk = make_input(p.compress_ratio, p.n_embd);
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let prefill = build_compressor_prefill(chunk.view(), &w, &p, None);
        let step = dsv4_compressor_step_coff1(chunk.view(), &w, &p, 0, None);

        assert_eq!(prefill.shape(), &[1, p.head_dim]);
        assert_eq!(step.shape(), &[p.head_dim]);
        let mut max_diff = 0.0_f32;
        for d in 0..p.head_dim {
            max_diff = max_diff.max((prefill[[0, d]] - step[d]).abs());
        }
        assert!(max_diff < 1e-6, "single-chunk diff = {max_diff}");
    }

    /// Two-chunk prefill via cached step (chunk-by-chunk) equals the
    /// prefill compressor on the concatenated 2-chunk input. This is
    /// the bit-exact incremental equivalence — the cornerstone for
    /// cached HCA decode.
    #[test]
    fn coff1_step_two_chunks_match_prefill() {
        let (p, wkv, wgate, ape, norm) = make_coff1_weights(2, 8, 16);
        let full = make_input(2 * p.compress_ratio, p.n_embd);
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let prefill = build_compressor_prefill(full.view(), &w, &p, None);
        assert_eq!(prefill.shape(), &[2, p.head_dim]);

        let chunk0 = full.slice(s![..p.compress_ratio, ..]);
        let chunk1 = full.slice(s![p.compress_ratio..2 * p.compress_ratio, ..]);
        let step0 = dsv4_compressor_step_coff1(chunk0, &w, &p, 0, None);
        let step1 = dsv4_compressor_step_coff1(chunk1, &w, &p, 1, None);

        let mut max_diff = 0.0_f32;
        for d in 0..p.head_dim {
            max_diff = max_diff.max((prefill[[0, d]] - step0[d]).abs());
            max_diff = max_diff.max((prefill[[1, d]] - step1[d]).abs());
        }
        assert!(max_diff < 1e-6, "two-chunk diff = {max_diff}");
    }

    /// Larger compress_ratio (16) also matches prefill chunk-by-chunk.
    /// Verifies the helper isn't accidentally specialized to small ratios.
    #[test]
    fn coff1_step_ratio_16_matches_prefill() {
        let (p, wkv, wgate, ape, norm) = make_coff1_weights(16, 8, 16);
        let full = make_input(2 * p.compress_ratio, p.n_embd);
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let prefill = build_compressor_prefill(full.view(), &w, &p, None);
        let chunk0 = full.slice(s![..p.compress_ratio, ..]);
        let chunk1 = full.slice(s![p.compress_ratio..2 * p.compress_ratio, ..]);
        let step0 = dsv4_compressor_step_coff1(chunk0, &w, &p, 0, None);
        let step1 = dsv4_compressor_step_coff1(chunk1, &w, &p, 1, None);

        let mut max_diff = 0.0_f32;
        for d in 0..p.head_dim {
            max_diff = max_diff.max((prefill[[0, d]] - step0[d]).abs());
            max_diff = max_diff.max((prefill[[1, d]] - step1[d]).abs());
        }
        assert!(max_diff < 1e-6, "ratio=16 diff = {max_diff}");
    }

    /// compress_ratio=4 → panic (use the coff=2 cached step instead).
    #[test]
    #[should_panic(expected = "coff=1 path requires compress_ratio != 4")]
    fn coff1_step_panics_on_ratio_4() {
        let (p, wkv, wgate, ape, norm) = {
            let p = CompressorParams {
                head_dim: 8,
                n_embd: 16,
                compress_ratio: 4, // <-- the disallowed value
                n_rot: 0,
                rope_base: 10000.0,
                rope_mode: DsV4RopeMode::Neox,
                norm_eps: 1e-5,
            };
            let n_kv = p.head_dim;
            let wkv = Array2::<f32>::zeros((n_kv, p.n_embd));
            let wgate = Array2::<f32>::zeros((n_kv, p.n_embd));
            let ape = Array2::<f32>::zeros((p.compress_ratio, n_kv));
            let norm = vec![1.0_f32; p.head_dim];
            (p, wkv, wgate, ape, norm)
        };
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let chunk = Array2::<f32>::zeros((p.compress_ratio, p.n_embd));
        let _ = dsv4_compressor_step_coff1(chunk.view(), &w, &p, 0, None);
    }

    /// Wrong chunk shape → panic.
    #[test]
    #[should_panic(expected = "cur_chunk must be")]
    fn coff1_step_wrong_chunk_shape_panics() {
        let (p, wkv, wgate, ape, norm) = make_coff1_weights(2, 8, 16);
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        // Half-size chunk.
        let chunk = Array2::<f32>::zeros((1, p.n_embd));
        let _ = dsv4_compressor_step_coff1(chunk.view(), &w, &p, 0, None);
    }

    // ── dsv4_compressor_step_coff2 tests (compress_ratio == 4) ──

    /// Helper: build CompressorParams + weights for coff=2 (compress_ratio = 4).
    fn make_coff2_weights(
        head_dim: usize,
        n_embd: usize,
    ) -> (
        CompressorParams,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Vec<f32>,
    ) {
        let p = CompressorParams {
            head_dim,
            n_embd,
            compress_ratio: 4, // coff=2
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let n_kv = 2 * head_dim;
        let wkv = Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
            ((i * 11 + j) as f32 * 0.014).cos() * 0.05
        });
        let wgate = Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
            ((i * 7 + j) as f32 * 0.016).sin() * 0.05
        });
        let ape = Array2::<f32>::from_shape_fn((p.compress_ratio, n_kv), |(r, k)| {
            ((r * 5 + k) as f32 * 0.025).sin() * 0.05
        });
        let norm = vec![1.0_f32; head_dim];
        (p, wkv, wgate, ape, norm)
    }

    /// Empty overlap state + 4-row chunk → matches row 0 of prefill on
    /// the same chunk. Verifies the zero/-∞ pad equivalence with
    /// `shift_overlap_state` at chunk 0.
    #[test]
    fn coff2_step_first_chunk_matches_prefill() {
        let (p, wkv, wgate, ape, norm) = make_coff2_weights(8, 16);
        let chunk = make_input(p.compress_ratio, p.n_embd);
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let prefill = build_compressor_prefill(chunk.view(), &w, &p, None);
        let mut state = CompressorOverlapState::empty();
        let step = dsv4_compressor_step_coff2(chunk.view(), &w, &p, 0, &mut state, None);
        let mut max_diff = 0.0_f32;
        for d in 0..p.head_dim {
            max_diff = max_diff.max((prefill[[0, d]] - step[d]).abs());
        }
        assert!(max_diff < 1e-6, "first-chunk diff = {max_diff}");
        // State is now populated for the next chunk.
        assert!(state.kv_prev_last.is_some());
        assert!(state.score_prev_last.is_some());
    }

    /// Two-chunk threading: step(chunk0) + step(chunk1, state) ==
    /// prefill(chunk0 + chunk1). This is the **cornerstone** bit-exact
    /// test — overlap state must carry the previous chunk's prev-half
    /// across calls or the second chunk's pool diverges.
    #[test]
    fn coff2_step_two_chunks_threaded_match_prefill() {
        let (p, wkv, wgate, ape, norm) = make_coff2_weights(8, 16);
        let full = make_input(2 * p.compress_ratio, p.n_embd);
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let prefill = build_compressor_prefill(full.view(), &w, &p, None);
        assert_eq!(prefill.shape(), &[2, p.head_dim]);

        let chunk0 = full.slice(s![..p.compress_ratio, ..]);
        let chunk1 = full.slice(s![p.compress_ratio..2 * p.compress_ratio, ..]);
        let mut state = CompressorOverlapState::empty();
        let step0 = dsv4_compressor_step_coff2(chunk0, &w, &p, 0, &mut state, None);
        let step1 = dsv4_compressor_step_coff2(chunk1, &w, &p, 1, &mut state, None);

        let mut max_diff = 0.0_f32;
        for d in 0..p.head_dim {
            max_diff = max_diff.max((prefill[[0, d]] - step0[d]).abs());
            max_diff = max_diff.max((prefill[[1, d]] - step1[d]).abs());
        }
        assert!(max_diff < 1e-6, "two-chunk threaded diff = {max_diff}");
    }

    /// Three-chunk threading: confirms state correctly threads beyond
    /// the first transition.
    #[test]
    fn coff2_step_three_chunks_threaded_match_prefill() {
        let (p, wkv, wgate, ape, norm) = make_coff2_weights(8, 16);
        let full = make_input(3 * p.compress_ratio, p.n_embd);
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let prefill = build_compressor_prefill(full.view(), &w, &p, None);
        let mut state = CompressorOverlapState::empty();
        let chunks: Vec<_> = (0..3)
            .map(|c| {
                let lo = c * p.compress_ratio;
                let hi = (c + 1) * p.compress_ratio;
                full.slice(s![lo..hi, ..]).to_owned()
            })
            .collect();
        let steps: Vec<_> = chunks
            .iter()
            .enumerate()
            .map(|(c, chunk)| dsv4_compressor_step_coff2(chunk.view(), &w, &p, c, &mut state, None))
            .collect();

        let mut max_diff = 0.0_f32;
        for c in 0..3 {
            for d in 0..p.head_dim {
                max_diff = max_diff.max((prefill[[c, d]] - steps[c][d]).abs());
            }
        }
        assert!(max_diff < 1e-6, "three-chunk threaded diff = {max_diff}");
    }

    /// State clear: after processing chunks 0..N, clear, then process
    /// chunk 0 again — should match a fresh first-chunk run.
    #[test]
    fn coff2_overlap_state_clear_resets_behavior() {
        let (p, wkv, wgate, ape, norm) = make_coff2_weights(8, 16);
        let chunk = make_input(p.compress_ratio, p.n_embd);
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };

        // Run once with empty state.
        let mut state_a = CompressorOverlapState::empty();
        let out_a = dsv4_compressor_step_coff2(chunk.view(), &w, &p, 0, &mut state_a, None);

        // Run twice with a state that gets cleared.
        let mut state_b = CompressorOverlapState::empty();
        let _ = dsv4_compressor_step_coff2(chunk.view(), &w, &p, 0, &mut state_b, None);
        state_b.clear();
        let out_b = dsv4_compressor_step_coff2(chunk.view(), &w, &p, 0, &mut state_b, None);

        let mut max_diff = 0.0_f32;
        for d in 0..p.head_dim {
            max_diff = max_diff.max((out_a[d] - out_b[d]).abs());
        }
        assert!(max_diff < 1e-6, "clear-then-rerun diff = {max_diff}");
    }

    /// compress_ratio != 4 → panic.
    #[test]
    #[should_panic(expected = "coff=2 path requires compress_ratio == 4")]
    fn coff2_step_panics_on_non_4_ratio() {
        let p = CompressorParams {
            head_dim: 8,
            n_embd: 16,
            compress_ratio: 2, // <-- not 4
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: 1e-5,
        };
        let n_kv = 2 * p.head_dim;
        let wkv = Array2::<f32>::zeros((n_kv, p.n_embd));
        let wgate = Array2::<f32>::zeros((n_kv, p.n_embd));
        let ape = Array2::<f32>::zeros((p.compress_ratio, n_kv));
        let norm = vec![1.0_f32; p.head_dim];
        let w = CompressorWeights {
            wkv: wkv.view(),
            wgate: wgate.view(),
            ape: ape.view(),
            norm: &norm,
            quant: None,
        };
        let chunk = Array2::<f32>::zeros((p.compress_ratio, p.n_embd));
        let mut state = CompressorOverlapState::empty();
        let _ = dsv4_compressor_step_coff2(chunk.view(), &w, &p, 0, &mut state, None);
    }
}
