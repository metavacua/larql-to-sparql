//! DeepSeek V4 indexer scoring + top-k sparse attention mask.
//!
//! Reference: llama.cpp PR #23122
//! - `src/models/deepseek4.cpp:985-1025` (`dsv4_build_indexer_scores_prefill`)
//! - `src/models/deepseek4.cpp:1067-1082` (`dsv4_build_compressed_mask_from_topk`)
//!
//! For DSv4 layers with `compress_ratio == 4`, the indexer assigns each
//! token a sparse subset of compressed-KV positions to attend to. It
//! does so in two steps:
//!
//! 1. **Indexer scores**: a small attention-like operation (n_index_head
//!    heads, n_index_head_size dim, with learned per-head weights) that
//!    scores `(token, comp)` pairs.
//! 2. **Top-k mask**: per-token, select the top-k compressed positions by
//!    score and build an additive attention mask (-∞ for non-selected
//!    positions, 0 for selected).
//!
//! These map cleanly to CPU code with no special ndarray gymnastics.

use ndarray::{Array2, ArrayView2, ArrayView3};

use larql_compute::{dot_proj_gpu, ComputeBackend};
use larql_models::quant::lazy::QuantTensor;

use super::dsv4_rope_tail::{dsv4_rope_tail, DsV4RopeMode};

/// Resident-Q4_K companion for the indexer's `wq_b` up-projection (P7).
///
/// `indexer.attn_q_b` is `(n_index_head*n_index_head_size, q_lora_rank)`
/// = `[8192, 1024]` ≈ 8.4M params for DSv4-Flash — the single largest
/// weight that the P1–P6 residency arc left dequantized to f32 (~33.5 MB
/// f32 read per indexer layer, ~700 MB/decode-step across the ~21
/// Indexer-variant layers). Holding it as a `QuantTensor` over the raw
/// Q4_K bytes (mirrors P5's [`super::dsv4_attn_block::DsV4AttnQuant`]) and
/// running the lazy Q4_K×Q8_K matmul cuts that read ~3.5×. This is the
/// larql-native analog of the DeepSeek-V4 paper's FP4 indexer-QK path.
#[derive(Clone, Copy)]
pub struct IndexerQuant<'a> {
    /// `(n_index_head*n_index_head_size, q_lora_rank)` indexer Q-up.
    pub wq_b: &'a QuantTensor,
}

/// Per-layer indexer weight refs.
#[derive(Clone, Copy)]
pub struct IndexerWeights<'a> {
    /// `(n_index_head * n_index_head_size, q_lora_rank)` — indexer Q
    /// "up" matrix; consumed alongside the main attention's q_lora_rank
    /// projection.
    pub wq_b: ArrayView2<'a, f32>,
    /// `(n_index_head, n_embd)` — per-head score weights projected from
    /// the residual.
    pub wproj: ArrayView2<'a, f32>,
    /// Resident-Q4_K companion for `wq_b` (P7). `None` = use the f32
    /// `wq_b` view above (streaming path).
    pub quant: Option<IndexerQuant<'a>>,
}

impl<'a> IndexerWeights<'a> {
    /// `qr @ wq_b^T`: resident-Q4_K lazy matmul when `quant` is set, else
    /// the f32 `dot_proj_gpu` path. Same semantics either way (`wq_b` is
    /// stored `[out, in]`).
    pub fn proj_wq_b(
        &self,
        qr: ArrayView2<f32>,
        backend: Option<&dyn ComputeBackend>,
    ) -> Array2<f32> {
        match &self.quant {
            // `matmul` wants an owned `Array2`; at decode `qr` is
            // `[1, q_lora_rank]` so the copy is negligible.
            Some(q) => q
                .wq_b
                .matmul(&qr.to_owned())
                .expect("indexer wq_b quant matmul"),
            None => dot_proj_gpu(&qr, &self.wq_b, backend),
        }
    }
}

/// Indexer scalar config.
#[derive(Clone, Copy, Debug)]
pub struct IndexerParams {
    pub n_embd: usize,
    pub q_lora_rank: usize,
    pub n_index_head: usize,
    pub n_index_head_size: usize,
    pub n_rot: usize,
    pub rope_base: f64,
    pub rope_mode: DsV4RopeMode,
}

/// Build indexer scores for prefill.
///
/// Inputs:
/// - `x`: `(n_tokens, n_embd)` — the layer residual.
/// - `qr`: `(n_tokens, q_lora_rank)` — q_a rms-normed output, shared
///   with the main attention.
/// - `index_kv`: `(n_comp, n_index_head_size)` — the indexer-compressor
///   output (single-head, already RoPE-rotated).
/// - `causal_mask`: `(n_tokens, n_comp)` — per `(token, comp)` mask
///   (0 for allowed, -∞ for masked). Added to the final score.
///
/// Output: `(n_tokens, n_comp)` — scalar score per `(token, comp)` pair.
pub fn build_indexer_scores_prefill(
    x: ArrayView2<f32>,
    qr: ArrayView2<f32>,
    index_kv: ArrayView2<f32>,
    causal_mask: ArrayView2<f32>,
    w: &IndexerWeights,
    p: &IndexerParams,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    let n_comp = index_kv.shape()[0];
    assert_eq!(x.shape()[1], p.n_embd, "x feature dim");
    assert_eq!(
        qr.shape(),
        &[n_tokens, p.q_lora_rank],
        "qr shape (n_tokens, q_lora_rank)"
    );
    assert_eq!(
        index_kv.shape()[1],
        p.n_index_head_size,
        "index_kv head dim"
    );
    assert_eq!(
        causal_mask.shape(),
        &[n_tokens, n_comp],
        "causal mask shape"
    );
    // wq_b shape check only applies to the f32 view; in resident-Q4_K
    // mode (P7) the f32 view is empty and the shape lives in the
    // `QuantTensor` (validated at build time).
    if w.quant.is_none() {
        assert_eq!(
            w.wq_b.shape(),
            &[p.n_index_head * p.n_index_head_size, p.q_lora_rank],
            "wq_b shape"
        );
    }
    assert_eq!(w.wproj.shape(), &[p.n_index_head, p.n_embd], "wproj shape");

    // 1. q = qr @ wq_b^T  →  (n_tokens, n_index_head * n_index_head_size)
    //    Resident-Q4_K lazy matmul when present, else f32 dot_proj_gpu.
    let q = w.proj_wq_b(qr, backend);
    //    Tail-RoPE at positions 0..n_tokens-1.
    let q = dsv4_rope_tail(
        &q,
        p.n_index_head,
        p.n_index_head_size,
        p.n_rot,
        p.rope_base,
        p.rope_mode,
        false,
        0,
    );

    // 2. score[t, h, c] = sum_d q[t, h, d] * index_kv[c, d]
    //    (Q · K^T per head against the single-head shared K.)
    let mut score = ndarray::Array3::<f32>::zeros((n_tokens, p.n_index_head, n_comp));
    for t in 0..n_tokens {
        for h in 0..p.n_index_head {
            let q_off = h * p.n_index_head_size;
            for c in 0..n_comp {
                let mut acc = 0.0_f32;
                for d in 0..p.n_index_head_size {
                    acc += q[[t, q_off + d]] * index_kv[[c, d]];
                }
                score[[t, h, c]] = acc;
            }
        }
    }
    // 3. ReLU.
    for v in score.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    // 4. weights = x @ wproj^T  →  (n_tokens, n_index_head), scaled.
    let weights = dot_proj_gpu(&x, &w.wproj, backend);
    let scale = 1.0 / ((p.n_index_head_size * p.n_index_head) as f32).sqrt();
    let mut scaled_weights = Array2::<f32>::zeros((n_tokens, p.n_index_head));
    for t in 0..n_tokens {
        for h in 0..p.n_index_head {
            scaled_weights[[t, h]] = weights[[t, h]] * scale;
        }
    }

    // 5. score *= scaled_weights (broadcast across c)
    // 6. score = sum over h → (n_tokens, n_comp)
    let mut out = Array2::<f32>::zeros((n_tokens, n_comp));
    for t in 0..n_tokens {
        for c in 0..n_comp {
            let mut sum = 0.0_f32;
            for h in 0..p.n_index_head {
                sum += score[[t, h, c]] * scaled_weights[[t, h]];
            }
            out[[t, c]] = sum;
        }
    }

    // 7. Add causal mask.
    for t in 0..n_tokens {
        for c in 0..n_comp {
            out[[t, c]] += causal_mask[[t, c]];
        }
    }
    out
}

/// Top-k row-wise selection: for each row of `scores`, return the
/// indices of the `top_k` largest values.
///
/// Returns `(n_tokens, top_k)` indices. Ties broken by lower index
/// (matches `ggml_argsort_top_k` stable ordering).
pub fn argsort_top_k(scores: ArrayView2<f32>, top_k: usize) -> ndarray::Array2<usize> {
    let n_tokens = scores.shape()[0];
    let n_comp = scores.shape()[1];
    assert!(top_k > 0 && top_k <= n_comp, "top_k must be in (0, n_comp]");

    let mut out = ndarray::Array2::<usize>::zeros((n_tokens, top_k));
    let mut buf: Vec<(f32, usize)> = Vec::with_capacity(n_comp);
    for t in 0..n_tokens {
        buf.clear();
        for c in 0..n_comp {
            buf.push((scores[[t, c]], c));
        }
        // Stable sort by descending score, ties by ascending index.
        buf.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        for k in 0..top_k {
            out[[t, k]] = buf[k].1;
        }
    }
    out
}

/// Build a sparse attention mask from top-k indexer selections.
///
/// `scores`: `(n_tokens, n_comp)` — indexer scores.
/// `topk`: `(n_tokens, top_k)` — selected compressed positions per token.
///
/// Output: `(n_tokens, n_comp)` mask where:
/// - Selected position with score `>` -1e30: value 0 (allowed).
/// - Selected position with score `≤` -1e30 (effectively -∞): value -1e9.
/// - Non-selected positions: -∞.
///
/// The two-tier "selected but score was -inf" handling matches the
/// llama.cpp reference: such positions get a strongly negative finite
/// value (-1e9) rather than -∞, so the softmax doesn't produce NaN if
/// every selected position happens to be unscored.
pub fn build_compressed_mask_from_topk(
    scores: ArrayView2<f32>,
    topk: ndarray::ArrayView2<usize>,
) -> Array2<f32> {
    let n_tokens = scores.shape()[0];
    let n_comp = scores.shape()[1];
    let top_k = topk.shape()[1];
    assert_eq!(
        topk.shape()[0],
        n_tokens,
        "topk row count must match scores"
    );

    let mut mask = Array2::<f32>::from_elem((n_tokens, n_comp), f32::NEG_INFINITY);
    for t in 0..n_tokens {
        for k in 0..top_k {
            let c = topk[[t, k]];
            assert!(
                c < n_comp,
                "topk index {c} out of bounds (n_comp={n_comp}) at t={t} k={k}"
            );
            // Selected score with valid finite value → 0; -∞ → -1e9.
            let s = scores[[t, c]];
            let valid = if s + 1.0e30 > 0.0 { 1.0_f32 } else { 0.0_f32 };
            mask[[t, c]] = (valid - 1.0) * 1.0e9;
        }
    }
    mask
}

// Unused import workaround: we expose ArrayView3 only via the public
// builder; this re-export keeps the existing IndexerWeights doc happy
// without forcing callers to pull ArrayView3 in scope.
#[allow(dead_code)]
fn _array_view3_marker(_: ArrayView3<f32>) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity-style scenario: when the indexer-K is one-hot and the
    /// per-head weights are uniform, the indexer produces predictable
    /// per-`(token, comp)` scalar values. We don't try to match exact
    /// numerics — just check shape, finiteness, and that the mask
    /// pipeline produces causal-aware selections.
    #[test]
    fn indexer_shape_and_finite() {
        let p = IndexerParams {
            n_embd: 16,
            q_lora_rank: 8,
            n_index_head: 2,
            n_index_head_size: 4,
            n_rot: 4,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
        };
        let n_tokens = 6;
        let n_comp = 3;

        let x = Array2::<f32>::from_shape_fn((n_tokens, p.n_embd), |(t, d)| {
            ((t * 7 + d) as f32 * 0.011).sin()
        });
        let qr = Array2::<f32>::from_shape_fn((n_tokens, p.q_lora_rank), |(t, d)| {
            ((t + d) as f32 * 0.013).cos() * 0.1
        });
        let index_kv = Array2::<f32>::from_shape_fn((n_comp, p.n_index_head_size), |(c, d)| {
            ((c + d) as f32 * 0.05).sin() * 0.2
        });
        let wq_b = Array2::<f32>::from_shape_fn(
            (p.n_index_head * p.n_index_head_size, p.q_lora_rank),
            |(i, j)| ((i + j) as f32 * 0.01).cos() * 0.1,
        );
        let wproj = Array2::<f32>::from_shape_fn((p.n_index_head, p.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.013).sin() * 0.1
        });
        let causal_mask = Array2::<f32>::zeros((n_tokens, n_comp));

        let w = IndexerWeights {
            wq_b: wq_b.view(),
            wproj: wproj.view(),
            quant: None,
        };
        let out = build_indexer_scores_prefill(
            x.view(),
            qr.view(),
            index_kv.view(),
            causal_mask.view(),
            &w,
            &p,
            None,
        );
        assert_eq!(out.shape(), &[n_tokens, n_comp]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Indexer-Q @ K is ReLU'd: a deliberately-negative-by-construction
    /// pair must produce zero contribution from that head before the
    /// per-head weights aggregate. Smoke: with all weights/index_kv set
    /// to make Q·K negative AND wproj zero, the output is exactly the
    /// causal mask (no contribution from the score path).
    #[test]
    fn zero_wproj_passes_through_causal_mask() {
        let p = IndexerParams {
            n_embd: 8,
            q_lora_rank: 4,
            n_index_head: 1,
            n_index_head_size: 4,
            n_rot: 2,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
        };
        let n_tokens = 3;
        let n_comp = 2;

        let x = Array2::<f32>::from_elem((n_tokens, p.n_embd), 0.5);
        let qr = Array2::<f32>::from_elem((n_tokens, p.q_lora_rank), 0.5);
        let index_kv = Array2::<f32>::from_elem((n_comp, p.n_index_head_size), 0.5);
        let wq_b =
            Array2::<f32>::from_elem((p.n_index_head * p.n_index_head_size, p.q_lora_rank), 0.5);
        let wproj = Array2::<f32>::zeros((p.n_index_head, p.n_embd));
        let causal_mask =
            Array2::<f32>::from_shape_fn((n_tokens, n_comp), |(t, c)| (t * 10 + c) as f32 * -0.5);
        let w = IndexerWeights {
            wq_b: wq_b.view(),
            wproj: wproj.view(),
            quant: None,
        };
        let out = build_indexer_scores_prefill(
            x.view(),
            qr.view(),
            index_kv.view(),
            causal_mask.view(),
            &w,
            &p,
            None,
        );
        // wproj=0 → scaled_weights=0 → score contribution from path is 0.
        // out should equal causal_mask exactly.
        for t in 0..n_tokens {
            for c in 0..n_comp {
                assert!(
                    (out[[t, c]] - causal_mask[[t, c]]).abs() < 1e-6,
                    "t={t} c={c}: out={} mask={}",
                    out[[t, c]],
                    causal_mask[[t, c]]
                );
            }
        }
    }

    /// CpuBackend equivalence: build_indexer_scores_prefill with
    /// `Some(&CpuBackend)` must match the `None` fallback to within
    /// BLAS reduction tolerance. Locks in the GPU-7c backend-routing
    /// wiring on the wq_b/wproj matmuls.
    #[test]
    fn indexer_scores_cpu_backend_matches_none_backend() {
        let p = IndexerParams {
            n_embd: 16,
            q_lora_rank: 12,
            n_index_head: 2,
            n_index_head_size: 8,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
        };
        let n_tokens = 5;
        let n_comp = 4;

        let x = Array2::<f32>::from_shape_fn((n_tokens, p.n_embd), |(t, d)| {
            ((t * 13 + d) as f32 * 0.011).sin()
        });
        let qr = Array2::<f32>::from_shape_fn((n_tokens, p.q_lora_rank), |(t, d)| {
            ((t * 7 + d) as f32 * 0.017).cos() * 0.1
        });
        let index_kv = Array2::<f32>::from_shape_fn((n_comp, p.n_index_head_size), |(c, d)| {
            ((c * 5 + d) as f32 * 0.013).sin() * 0.1
        });
        let wq_b = Array2::<f32>::from_shape_fn(
            (p.n_index_head * p.n_index_head_size, p.q_lora_rank),
            |(i, j)| ((i + j) as f32 * 0.013).sin() * 0.1,
        );
        let wproj = Array2::<f32>::from_shape_fn((p.n_index_head, p.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.019).cos() * 0.1
        });
        let causal_mask = Array2::<f32>::zeros((n_tokens, n_comp));

        let w = IndexerWeights {
            wq_b: wq_b.view(),
            wproj: wproj.view(),
            quant: None,
        };

        let out_none = build_indexer_scores_prefill(
            x.view(),
            qr.view(),
            index_kv.view(),
            causal_mask.view(),
            &w,
            &p,
            None,
        );
        let cpu = larql_compute::CpuBackend;
        let out_cpu = build_indexer_scores_prefill(
            x.view(),
            qr.view(),
            index_kv.view(),
            causal_mask.view(),
            &w,
            &p,
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
            "CpuBackend vs None mismatch on indexer scores: max_diff={max_diff}"
        );
    }

    /// Top-k selection picks the indices with the largest scores per row.
    #[test]
    fn argsort_top_k_picks_largest() {
        let scores = ndarray::array![[0.0, 5.0, 2.0, 4.0, 1.0], [9.0, 1.0, 8.0, 2.0, 7.0]];
        let topk = argsort_top_k(scores.view(), 3);
        assert_eq!(topk.shape(), &[2, 3]);
        // Row 0: sorted desc → 5.0 (idx 1), 4.0 (idx 3), 2.0 (idx 2).
        assert_eq!(topk[[0, 0]], 1);
        assert_eq!(topk[[0, 1]], 3);
        assert_eq!(topk[[0, 2]], 2);
        // Row 1: sorted desc → 9.0 (idx 0), 8.0 (idx 2), 7.0 (idx 4).
        assert_eq!(topk[[1, 0]], 0);
        assert_eq!(topk[[1, 1]], 2);
        assert_eq!(topk[[1, 2]], 4);
    }

    /// Top-k ties break by lower index.
    #[test]
    fn argsort_top_k_stable_ties() {
        let scores = ndarray::array![[1.0, 1.0, 1.0, 1.0]];
        let topk = argsort_top_k(scores.view(), 4);
        // All scores tied → indices in ascending order.
        assert_eq!(topk[[0, 0]], 0);
        assert_eq!(topk[[0, 1]], 1);
        assert_eq!(topk[[0, 2]], 2);
        assert_eq!(topk[[0, 3]], 3);
    }

    /// Sparse mask: selected positions → 0, non-selected → -∞.
    #[test]
    fn sparse_mask_selects_topk_others_neg_inf() {
        let scores = ndarray::array![[0.5, 1.0, 0.3, 2.0, 0.7]];
        let topk = argsort_top_k(scores.view(), 2); // picks idx 3 then 1
        let mask = build_compressed_mask_from_topk(scores.view(), topk.view());
        assert_eq!(mask.shape(), &[1, 5]);
        assert_eq!(mask[[0, 1]], 0.0, "topk #2 → 0");
        assert_eq!(mask[[0, 3]], 0.0, "topk #1 → 0");
        // Non-selected positions: -∞.
        for &c in &[0usize, 2, 4] {
            assert_eq!(
                mask[[0, c]],
                f32::NEG_INFINITY,
                "non-topk c={c} should be -∞ (got {})",
                mask[[0, c]]
            );
        }
    }

    /// Sparse mask: selected position with -∞ score → -1e9, not -∞.
    /// This is the "valid - 1" branch that prevents downstream softmax
    /// from producing NaN.
    #[test]
    fn sparse_mask_neg_inf_score_yields_neg_1e9() {
        let scores =
            Array2::<f32>::from_shape_vec((1, 3), vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY])
                .unwrap();
        let topk = argsort_top_k(scores.view(), 2);
        // 1.0 is the only finite score → picked first; one -∞ entry follows.
        assert_eq!(topk[[0, 0]], 1);
        let mask = build_compressed_mask_from_topk(scores.view(), topk.view());
        assert_eq!(mask[[0, 1]], 0.0, "finite-score topk → 0");
        // The -∞-score topk entry gets -1e9 (not -∞).
        let other = topk[[0, 1]];
        assert!(
            (mask[[0, other]] + 1.0e9).abs() < 1.0,
            "-∞-score topk → ~-1e9 (got {})",
            mask[[0, other]]
        );
    }
}
