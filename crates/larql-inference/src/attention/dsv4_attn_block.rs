//! DeepSeek V4 attention block — `compress_ratio == 0` path.
//!
//! Composes the per-op helpers built in Stages 2-7 into a single
//! end-to-end attention block. Drives the simpler of the two DSv4
//! attention paths (the one without compressed KV). The compressed
//! branch (HCA, `compress_ratio > 0`) is handled separately in
//! Stage 8b.
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:1135-1192`
//! (per-layer attention construction up to and including the
//! `compress_ratio == 0` `build_attn_mha` call).
//!
//! Flow (input `x: (n_tokens, n_embd)` → output `(n_tokens, n_embd)`):
//!
//! 1. **Pre-norm**: `cur = rms_norm(x, attn_norm)`
//! 2. **Q low-rank**: `qr = rms_norm(cur @ Wq_a, q_a_norm)`,
//!    `q = qr @ Wq_b`, reshape to `[n_tokens, n_head, head_dim]`,
//!    per-head RMSNorm (no learned weight), then tail-RoPE.
//! 3. **KV low-rank**: `kv = rms_norm(cur @ Wkv, kv_a_norm)`,
//!    reshape to `[n_tokens, 1, head_dim]`, tail-RoPE.
//! 4. **FP8 KV fake-quant** on the rotated KV.
//! 5. **Sliding-window attention** (MQA, optional sinks).
//! 6. **Grouped low-rank o-proj**: per-group `Wo_a` then shared `Wo_b`.
//!
//! mHC pre/post (the 4-stream residual bookends) and FFN are NOT in this
//! block — they're applied by the caller. This keeps Stage 8a focused on
//! attention.

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayView3};

use larql_compute::{dot_proj_gpu, ComputeBackend};
use larql_models::quant::lazy::QuantTensor;

use super::dsv4_fp8_kv::fp8_kv_quantize;
use super::dsv4_grouped_o_proj::grouped_o_proj;
use super::dsv4_kv_cache::DsV4LayerKvCache;
use super::dsv4_rope_tail::DsV4RopeMode;
use super::dsv4_rope_tail_yarn::rope_tail_dispatch;
use super::dsv4_swa::dsv4_sliding_window_attn;

/// Resident-Q4_K companions for the attention projection weights (P5).
///
/// When present on [`DsV4AttnBlockWeights::quant`], the projection matmuls
/// run against these [`QuantTensor`]s (lazy Q4_K×Q8_K dot, no f32 dequant)
/// instead of the f32 views — cutting the per-token cold-RAM read ~3.5× on
/// the attention projections, the 74% decode hot spot (see `dsv4_profile`).
/// All-or-nothing per layer: resident mode sets `Some(_)` with every field
/// populated and leaves the f32 views empty; streaming leaves it `None`.
#[derive(Clone, Copy)]
pub struct DsV4AttnQuant<'a> {
    /// `(q_lora_rank, n_embd)` Q-down.
    pub wq_a: &'a QuantTensor,
    /// `(n_head*head_dim, q_lora_rank)` Q-up.
    pub wq_b: &'a QuantTensor,
    /// `(head_dim, n_embd)` KV-down.
    pub wkv: &'a QuantTensor,
    /// `(n_groups*o_lora_rank, group_dim)` grouped O-a (per-group via
    /// [`QuantTensor::expert_slice`]).
    pub wo_a: &'a QuantTensor,
    /// `(n_embd, o_lora_rank*n_groups)` shared O-b.
    pub wo_b: &'a QuantTensor,
}

/// Per-layer DSv4 attention weights — references-only, so callers can
/// hold the weight buffers however they like (mmap'd GGUF, owned Vec,
/// etc.).
#[derive(Clone, Copy)]
pub struct DsV4AttnBlockWeights<'a> {
    /// `(n_embd,)` — input RMSNorm weight.
    pub attn_norm: &'a [f32],
    /// `(q_lora_rank, n_embd)` — Q low-rank "down" matrix.
    pub wq_a: ArrayView2<'a, f32>,
    /// `(q_lora_rank,)` — RMSNorm weight on the q_a output.
    pub q_a_norm: &'a [f32],
    /// `(n_head * head_dim, q_lora_rank)` — Q low-rank "up" matrix.
    pub wq_b: ArrayView2<'a, f32>,
    /// `(head_dim, n_embd)` — KV "down" matrix (single shared KV head).
    pub wkv: ArrayView2<'a, f32>,
    /// `(head_dim,)` — RMSNorm weight on the kv output.
    pub kv_a_norm: &'a [f32],
    /// `(n_groups, o_lora_rank, group_dim)` — grouped output A matrix.
    pub wo_a: ArrayView3<'a, f32>,
    /// `(n_embd, o_lora_rank * n_groups)` — shared output B matrix.
    pub wo_b: ArrayView2<'a, f32>,
    /// Optional per-head attention sinks `(n_head,)`.
    pub attn_sinks: Option<ArrayView1<'a, f32>>,
    /// Resident-Q4_K companions (P5). `None` = use the f32 views above.
    pub quant: Option<DsV4AttnQuant<'a>>,
}

impl<'a> DsV4AttnBlockWeights<'a> {
    /// `x @ wq_a^T`: resident-Q4_K lazy matmul when `quant` is set, else
    /// the f32 `dot_proj_gpu` path.
    pub fn proj_wq_a(&self, x: &Array2<f32>, backend: Option<&dyn ComputeBackend>) -> Array2<f32> {
        match &self.quant {
            Some(q) => q.wq_a.matmul(x).expect("wq_a quant matmul"),
            None => dot_proj_gpu(x, &self.wq_a, backend),
        }
    }
    /// `x @ wq_b^T` (resident-Q4_K when available).
    pub fn proj_wq_b(&self, x: &Array2<f32>, backend: Option<&dyn ComputeBackend>) -> Array2<f32> {
        match &self.quant {
            Some(q) => q.wq_b.matmul(x).expect("wq_b quant matmul"),
            None => dot_proj_gpu(x, &self.wq_b, backend),
        }
    }
    /// `x @ wkv^T` (resident-Q4_K when available).
    pub fn proj_wkv(&self, x: &Array2<f32>, backend: Option<&dyn ComputeBackend>) -> Array2<f32> {
        match &self.quant {
            Some(q) => q.wkv.matmul(x).expect("wkv quant matmul"),
            None => dot_proj_gpu(x, &self.wkv, backend),
        }
    }
}

/// Per-layer DSv4 attention dimensions and scalar config.
#[derive(Clone, Copy, Debug)]
pub struct DsV4AttnBlockParams {
    pub n_embd: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub n_groups: usize,
    pub o_lora_rank: usize,
    /// Number of rotated tail dims (the rest are nope).
    pub n_rot: usize,
    /// RoPE frequency base. Used as-is when `yarn` is `None`.
    pub rope_base: f64,
    pub rope_mode: DsV4RopeMode,
    pub window_size: usize,
    pub norm_eps: f32,
    /// Optional YARN scaling config. When `Some(_)`, the rope-tail
    /// calls dispatch to [`super::dsv4_rope_tail_yarn::dsv4_rope_tail_yarn`]
    /// and the `rope_base` field is shadowed by `yarn.freq_base`.
    /// Dispatch wiring lands in a follow-up; this field is plumbing
    /// for now.
    pub yarn: Option<super::dsv4_yarn_config::DsV4RopeYarnConfig>,
}

/// Run the DSv4 `compress_ratio == 0` attention block on a single-stream
/// residual. `position_offset` is the absolute position of `x[0]` (0
/// for prefill from scratch; cache length for decode).
pub fn dsv4_attn_block_no_compress(
    x: ArrayView2<f32>,
    w: &DsV4AttnBlockWeights,
    p: &DsV4AttnBlockParams,
    position_offset: usize,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    assert_eq!(
        x.shape(),
        &[n_tokens, p.n_embd],
        "x must be (n_tokens, n_embd)"
    );

    // 1. attn_norm — RMSNorm on the input with learned per-feature gain.
    let cur = rms_norm_2d(x, w.attn_norm, p.norm_eps);

    // 2. Q low-rank.
    //    qr = (cur @ Wq_a^T) — (n_tokens, q_lora_rank)
    let qr = w.proj_wq_a(&cur, backend);
    let qr = rms_norm_2d(qr.view(), w.q_a_norm, p.norm_eps);
    //    q = qr @ Wq_b^T — (n_tokens, n_head * head_dim)
    let q = w.proj_wq_b(&qr, backend);
    //    per-head RMSNorm (no learned weight) on each (token, head)
    //    n_embd_head_k vector — apply BEFORE tail-RoPE.
    let q = rms_norm_per_head(q.view(), p.n_head, p.head_dim, p.norm_eps);
    let q = rope_tail_dispatch(
        &q,
        p.n_head,
        p.head_dim,
        p.n_rot,
        p.rope_base,
        p.rope_mode,
        p.yarn.as_ref(),
        false,
        position_offset,
    );

    // 3. KV low-rank — single shared head.
    //    kv = (cur @ Wkv^T) — (n_tokens, head_dim)
    let kv = w.proj_wkv(&cur, backend);
    let kv = rms_norm_2d(kv.view(), w.kv_a_norm, p.norm_eps);
    //    tail-RoPE — reshape view (n_tokens, head_dim) treated as 1 head.
    let kv = rope_tail_dispatch(
        &kv,
        1,
        p.head_dim,
        p.n_rot,
        p.rope_base,
        p.rope_mode,
        p.yarn.as_ref(),
        false,
        position_offset,
    );

    // 4. FP8 fake-quantize the KV.
    let kv = fp8_kv_quantize(kv.view(), p.n_rot);

    // 5. Sliding-window MQA with optional sinks. For the no-cache prefill
    //    path KV == K == V (single-head shared).
    let scale = 1.0 / (p.head_dim as f32).sqrt();
    let o = dsv4_sliding_window_attn(
        q.view(),
        kv.view(),
        kv.view(),
        p.n_head,
        p.head_dim,
        p.window_size,
        w.attn_sinks,
        scale,
        position_offset,
        backend,
    );

    // 6. Grouped low-rank output projection → (n_tokens, n_embd).
    grouped_o_proj(
        o.view(),
        w.wo_a,
        w.wo_b,
        p.n_head,
        p.head_dim,
        p.n_groups,
        p.o_lora_rank,
        backend,
        w.quant.map(|q| q.wo_a),
        w.quant.map(|q| q.wo_b),
    )
}

/// RMSNorm along axis 1 (per-row) with a learned per-feature gain weight.
/// `out[t, d] = x[t, d] / sqrt(mean(x[t, :]^2) + eps) * weight[d]`.
fn rms_norm_2d(x: ArrayView2<f32>, weight: &[f32], eps: f32) -> Array2<f32> {
    let n_rows = x.shape()[0];
    let n_dims = x.shape()[1];
    assert_eq!(weight.len(), n_dims, "weight length must equal feature dim");
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

/// Per-head RMSNorm with NO learned weight — applied to Q after the
/// `Wq_b` projection but before tail-RoPE. For each `(token, head)` the
/// `head_dim` vector is normalized: `out[t, h*head_dim + d] = x / rms`.
fn rms_norm_per_head(x: ArrayView2<f32>, n_head: usize, head_dim: usize, eps: f32) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    assert_eq!(
        x.shape(),
        &[n_tokens, n_head * head_dim],
        "x must be (n_tokens, n_head * head_dim)"
    );
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

/// KV-cached variant of [`dsv4_attn_block_no_compress`].
///
/// Computes the same attention as the non-cached prefill, but the
/// caller supplies a [`DsV4LayerKvCache`] that persists across calls.
/// Each call appends the newly-computed `kv_a` rows for `x` to the
/// cache, then attends `x`'s queries against the *full* cached K/V
/// (prior tokens + new). This enables incremental decode: a 1-token
/// `x` after a 128-token prefill runs in O(window_size) attention
/// time instead of O(prefill_len).
///
/// `position_offset` MUST equal `cache.current_len()` on entry — the
/// absolute position of `x[0]` in the running sequence. The function
/// asserts this invariant.
///
/// Equivalence with prefill: running this with an empty cache and a
/// full sequence produces the same output (within FP8 rounding noise)
/// as [`dsv4_attn_block_no_compress`] on the same sequence — verified
/// in the unit tests.
pub fn dsv4_attn_block_no_compress_cached(
    x: ArrayView2<f32>,
    w: &DsV4AttnBlockWeights,
    p: &DsV4AttnBlockParams,
    position_offset: usize,
    cache: &mut DsV4LayerKvCache,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    let n_tokens = x.shape()[0];
    assert_eq!(
        x.shape(),
        &[n_tokens, p.n_embd],
        "x must be (n_tokens, n_embd)"
    );
    assert_eq!(
        cache.head_dim(),
        p.head_dim,
        "cache.head_dim ({}) must match params.head_dim ({})",
        cache.head_dim(),
        p.head_dim
    );
    assert_eq!(
        cache.current_len(),
        position_offset,
        "position_offset ({position_offset}) must equal cache.current_len ({}) — cache and \
         offset are out of sync",
        cache.current_len()
    );

    // 1. attn_norm.
    let cur = rms_norm_2d(x, w.attn_norm, p.norm_eps);

    // 2. Q low-rank + tail-RoPE. Q matmuls route through backend when
    //    supplied — dot_proj_gpu(a, b, backend) = a @ b^T.
    let qr = w.proj_wq_a(&cur, backend);
    let qr = rms_norm_2d(qr.view(), w.q_a_norm, p.norm_eps);
    let q = w.proj_wq_b(&qr, backend);
    let q = rms_norm_per_head(q.view(), p.n_head, p.head_dim, p.norm_eps);
    let q = rope_tail_dispatch(
        &q,
        p.n_head,
        p.head_dim,
        p.n_rot,
        p.rope_base,
        p.rope_mode,
        p.yarn.as_ref(),
        false,
        position_offset,
    );

    // 3. KV low-rank + tail-RoPE (for the NEW tokens only). Backend-routed.
    let kv_new = w.proj_wkv(&cur, backend);
    let kv_new = rms_norm_2d(kv_new.view(), w.kv_a_norm, p.norm_eps);
    let kv_new = rope_tail_dispatch(
        &kv_new,
        1,
        p.head_dim,
        p.n_rot,
        p.rope_base,
        p.rope_mode,
        p.yarn.as_ref(),
        false,
        position_offset,
    );

    // 4. FP8 fake-quantize.
    let kv_new = fp8_kv_quantize(kv_new.view(), p.n_rot);

    // 5. Append to cache → cache now has (position_offset + n_tokens, head_dim).
    cache.append(kv_new.view());

    // 6. SWA MQA against the full cached K/V; Q is just the new positions.
    let kv_full = cache.view_current();
    let scale = 1.0 / (p.head_dim as f32).sqrt();
    let o = dsv4_sliding_window_attn(
        q.view(),
        kv_full,
        kv_full,
        p.n_head,
        p.head_dim,
        p.window_size,
        w.attn_sinks,
        scale,
        position_offset,
        backend,
    );

    // 7. Grouped low-rank o-proj.
    grouped_o_proj(
        o.view(),
        w.wo_a,
        w.wo_b,
        p.n_head,
        p.head_dim,
        p.n_groups,
        p.o_lora_rank,
        backend,
        w.quant.map(|q| q.wo_a),
        w.quant.map(|q| q.wo_b),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};

    /// Build a small but plausible-shape DSv4 attention block, run it on
    /// synthetic input, and check that:
    /// - output shape matches input shape
    /// - all values are finite (no NaN/Inf from accidental div-by-zero)
    /// - output is non-trivial (not all zero — confirms data flowed
    ///   through every stage)
    #[test]
    fn block_finite_and_nontrivial_smoke() {
        // Dims chosen to satisfy every helper's constraints simultaneously:
        //   - head_dim = 128 (n_nope=64 mod-64 for FP8; n_rot=64 for RoPE)
        //   - n_head = 8, n_groups = 4 (group_heads = 2)
        //   - n_embd = 256, q_lora_rank = 32, o_lora_rank = 16
        let p = DsV4AttnBlockParams {
            n_embd: 256,
            n_head: 8,
            head_dim: 128,
            q_lora_rank: 32,
            n_groups: 4,
            o_lora_rank: 16,
            n_rot: 64, // → n_nope = 64 = 1 FP8 block
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 16,
            norm_eps: 1e-5,
            yarn: None,
        };

        let n_tokens = 5;
        let group_heads = p.n_head / p.n_groups; // 2
        let group_dim = p.head_dim * group_heads; // 256
        let low_dim = p.o_lora_rank * p.n_groups; // 64

        let attn_norm = vec![1.0_f32; p.n_embd];
        let wq_a = Array2::<f32>::from_shape_fn((p.q_lora_rank, p.n_embd), |(i, j)| {
            ((i * 31 + j) as f32 * 0.001).sin()
        });
        let q_a_norm = vec![1.0_f32; p.q_lora_rank];
        let wq_b =
            Array2::<f32>::from_shape_fn((p.n_head * p.head_dim, p.q_lora_rank), |(i, j)| {
                ((i * 13 + j) as f32 * 0.0013).cos() * 0.1
            });
        let wkv = Array2::<f32>::from_shape_fn((p.head_dim, p.n_embd), |(i, j)| {
            ((i * 7 + j) as f32 * 0.0007).sin() * 0.05
        });
        let kv_a_norm = vec![1.0_f32; p.head_dim];
        let wo_a =
            Array3::<f32>::from_shape_fn((p.n_groups, p.o_lora_rank, group_dim), |(g, r, j)| {
                ((g * 100 + r * 10 + j) as f32 * 0.0005).cos() * 0.05
            });
        let wo_b = Array2::<f32>::from_shape_fn((p.n_embd, low_dim), |(i, j)| {
            ((i * 5 + j * 3) as f32 * 0.0011).sin() * 0.1
        });
        let attn_sinks = Array1::<f32>::from_elem(p.n_head, -1.0);

        let w = DsV4AttnBlockWeights {
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

        let x = Array2::<f32>::from_shape_fn((n_tokens, p.n_embd), |(t, d)| {
            ((t * 17 + d) as f32 * 0.013).sin()
        });
        let out = dsv4_attn_block_no_compress(x.view(), &w, &p, 0, None);

        assert_eq!(out.shape(), &[n_tokens, p.n_embd]);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
        let total_abs: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(
            total_abs > 1e-3,
            "output collapsed to zero (sum={total_abs})"
        );
    }

    /// Sink-free vs sink-equipped block paths both produce finite output
    /// and differ — confirms the optional sink wiring is plumbed through.
    #[test]
    fn block_with_and_without_sinks_differs() {
        let p = DsV4AttnBlockParams {
            n_embd: 64,
            n_head: 4,
            head_dim: 64, // n_rot=0 → n_nope=64 (one full FP8 block)
            q_lora_rank: 16,
            n_groups: 2,
            o_lora_rank: 8,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            yarn: None,
        };

        let n_tokens = 3;
        let group_heads = p.n_head / p.n_groups; // 2
        let group_dim = p.head_dim * group_heads; // 128
        let low_dim = p.o_lora_rank * p.n_groups; // 16

        let attn_norm = vec![1.0_f32; p.n_embd];
        let wq_a = Array2::<f32>::from_shape_fn((p.q_lora_rank, p.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.01).sin()
        });
        let q_a_norm = vec![1.0_f32; p.q_lora_rank];
        let wq_b =
            Array2::<f32>::from_shape_fn((p.n_head * p.head_dim, p.q_lora_rank), |(i, j)| {
                ((i + j) as f32 * 0.013).cos() * 0.1
            });
        let wkv = Array2::<f32>::from_shape_fn((p.head_dim, p.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.007).sin() * 0.05
        });
        let kv_a_norm = vec![1.0_f32; p.head_dim];
        let wo_a =
            Array3::<f32>::from_shape_fn((p.n_groups, p.o_lora_rank, group_dim), |(g, r, j)| {
                ((g + r + j) as f32 * 0.005).cos() * 0.05
            });
        let wo_b = Array2::<f32>::from_shape_fn((p.n_embd, low_dim), |(i, j)| {
            ((i + j) as f32 * 0.011).sin() * 0.1
        });

        let x = Array2::<f32>::from_shape_fn((n_tokens, p.n_embd), |(t, d)| {
            ((t + d) as f32 * 0.013).sin()
        });

        let w_nosink = DsV4AttnBlockWeights {
            quant: None,
            attn_norm: &attn_norm,
            wq_a: wq_a.view(),
            q_a_norm: &q_a_norm,
            wq_b: wq_b.view(),
            wkv: wkv.view(),
            kv_a_norm: &kv_a_norm,
            wo_a: wo_a.view(),
            wo_b: wo_b.view(),
            attn_sinks: None,
        };
        let sinks = Array1::<f32>::from_elem(p.n_head, 5.0);
        let w_sink = DsV4AttnBlockWeights {
            quant: None,
            attn_sinks: Some(sinks.view()),
            ..w_nosink
        };

        let out_nosink = dsv4_attn_block_no_compress(x.view(), &w_nosink, &p, 0, None);
        let out_sink = dsv4_attn_block_no_compress(x.view(), &w_sink, &p, 0, None);

        assert!(out_nosink.iter().all(|v| v.is_finite()));
        assert!(out_sink.iter().all(|v| v.is_finite()));
        let diff: f32 = out_nosink
            .iter()
            .zip(out_sink.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-4, "sink toggle had no effect (diff={diff})");
    }

    /// Position offset should match sequential prefill — the row at
    /// `position_offset = k` from a single-token call should equal row k
    /// from a full prefill. Sanity check on RoPE-tail / SWA composition.
    ///
    /// Note: because SWA looks at past keys but FP8-quantizes them, the
    /// single-token-with-offset path doesn't have the same KV cache as
    /// the prefill. So we only check the FIRST prefill row (where SWA
    /// has no past to look at): position 0 with no offset.
    #[test]
    fn block_first_row_position_zero_smoke() {
        let p = DsV4AttnBlockParams {
            n_embd: 64,
            n_head: 4,
            head_dim: 64,
            q_lora_rank: 16,
            n_groups: 2,
            o_lora_rank: 8,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            yarn: None,
        };
        let group_heads = p.n_head / p.n_groups;
        let group_dim = p.head_dim * group_heads;
        let low_dim = p.o_lora_rank * p.n_groups;

        let attn_norm = vec![1.0_f32; p.n_embd];
        let wq_a = Array2::<f32>::from_shape_fn((p.q_lora_rank, p.n_embd), |_| 0.01);
        let q_a_norm = vec![1.0_f32; p.q_lora_rank];
        let wq_b = Array2::<f32>::from_shape_fn((p.n_head * p.head_dim, p.q_lora_rank), |_| 0.01);
        let wkv = Array2::<f32>::from_shape_fn((p.head_dim, p.n_embd), |_| 0.01);
        let kv_a_norm = vec![1.0_f32; p.head_dim];
        let wo_a = Array3::<f32>::from_shape_fn((p.n_groups, p.o_lora_rank, group_dim), |_| 0.01);
        let wo_b = Array2::<f32>::from_shape_fn((p.n_embd, low_dim), |_| 0.01);
        let w = DsV4AttnBlockWeights {
            quant: None,
            attn_norm: &attn_norm,
            wq_a: wq_a.view(),
            q_a_norm: &q_a_norm,
            wq_b: wq_b.view(),
            wkv: wkv.view(),
            kv_a_norm: &kv_a_norm,
            wo_a: wo_a.view(),
            wo_b: wo_b.view(),
            attn_sinks: None,
        };
        let x = Array2::<f32>::from_elem((1, p.n_embd), 0.5);
        let out = dsv4_attn_block_no_compress(x.view(), &w, &p, 0, None);
        assert_eq!(out.shape(), &[1, p.n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// CpuBackend equivalence on the prefill block: routing Q/KV/O
    /// matmuls through `Some(&CpuBackend)` must match the `None`
    /// fallback to within BLAS reduction tolerance. Locks in the
    /// backend-threading wiring at the prefill entry point.
    #[test]
    fn prefill_cpu_backend_matches_none_backend() {
        let p = DsV4AttnBlockParams {
            n_embd: 64,
            n_head: 4,
            head_dim: 64,
            q_lora_rank: 16,
            n_groups: 2,
            o_lora_rank: 8,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            yarn: None,
        };
        let group_heads = p.n_head / p.n_groups;
        let group_dim = p.head_dim * group_heads;
        let low_dim = p.o_lora_rank * p.n_groups;
        let attn_norm = vec![1.0_f32; p.n_embd];
        let wq_a = Array2::<f32>::from_shape_fn((p.q_lora_rank, p.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.01).sin()
        });
        let q_a_norm = vec![1.0_f32; p.q_lora_rank];
        let wq_b =
            Array2::<f32>::from_shape_fn((p.n_head * p.head_dim, p.q_lora_rank), |(i, j)| {
                ((i + j) as f32 * 0.013).cos() * 0.1
            });
        let wkv = Array2::<f32>::from_shape_fn((p.head_dim, p.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.007).sin() * 0.05
        });
        let kv_a_norm = vec![1.0_f32; p.head_dim];
        let wo_a =
            Array3::<f32>::from_shape_fn((p.n_groups, p.o_lora_rank, group_dim), |(g, r, j)| {
                ((g + r + j) as f32 * 0.005).cos() * 0.05
            });
        let wo_b = Array2::<f32>::from_shape_fn((p.n_embd, low_dim), |(i, j)| {
            ((i + j) as f32 * 0.011).sin() * 0.1
        });
        let attn_sinks = Array1::<f32>::from_elem(p.n_head, -1.0);
        let w = DsV4AttnBlockWeights {
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

        let n_tokens = 6;
        let x = Array2::<f32>::from_shape_fn((n_tokens, p.n_embd), |(t, d)| {
            ((t * 19 + d) as f32 * 0.012).sin()
        });

        let out_none = dsv4_attn_block_no_compress(x.view(), &w, &p, 0, None);
        let cpu = larql_compute::CpuBackend;
        let out_cpu =
            dsv4_attn_block_no_compress(x.view(), &w, &p, 0, Some(&cpu as &dyn ComputeBackend));

        assert_eq!(out_none.shape(), out_cpu.shape());
        let max_diff = out_none
            .iter()
            .zip(out_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "CpuBackend vs None mismatch on prefill: max_diff={max_diff}"
        );
    }

    // ── Cached-attention tests ──

    /// Helper to build a deterministic weight set + params for cache
    /// equivalence tests. Same dimensions across all cached-attn tests.
    fn build_cached_test_weights() -> (
        DsV4AttnBlockParams,
        Vec<f32>, // attn_norm
        Array2<f32>,
        Vec<f32>,
        Array2<f32>,
        Array2<f32>,
        Vec<f32>,
        Array3<f32>,
        Array2<f32>,
        Array1<f32>,
    ) {
        let p = DsV4AttnBlockParams {
            n_embd: 64,
            n_head: 4,
            head_dim: 64,
            q_lora_rank: 16,
            n_groups: 2,
            o_lora_rank: 8,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 8,
            norm_eps: 1e-5,
            yarn: None,
        };
        let group_heads = p.n_head / p.n_groups;
        let group_dim = p.head_dim * group_heads;
        let low_dim = p.o_lora_rank * p.n_groups;
        let attn_norm = vec![1.0_f32; p.n_embd];
        let wq_a = Array2::<f32>::from_shape_fn((p.q_lora_rank, p.n_embd), |(i, j)| {
            ((i * 17 + j) as f32 * 0.013).sin()
        });
        let q_a_norm = vec![1.0_f32; p.q_lora_rank];
        let wq_b =
            Array2::<f32>::from_shape_fn((p.n_head * p.head_dim, p.q_lora_rank), |(i, j)| {
                ((i * 11 + j) as f32 * 0.017).cos() * 0.1
            });
        let wkv = Array2::<f32>::from_shape_fn((p.head_dim, p.n_embd), |(i, j)| {
            ((i + j) as f32 * 0.007).sin() * 0.05
        });
        let kv_a_norm = vec![1.0_f32; p.head_dim];
        let wo_a =
            Array3::<f32>::from_shape_fn((p.n_groups, p.o_lora_rank, group_dim), |(g, r, j)| {
                ((g * 100 + r * 10 + j) as f32 * 0.0005).cos() * 0.05
            });
        let wo_b = Array2::<f32>::from_shape_fn((p.n_embd, low_dim), |(i, j)| {
            ((i * 5 + j * 3) as f32 * 0.0011).sin() * 0.1
        });
        let attn_sinks = Array1::<f32>::from_elem(p.n_head, -1.0);
        (
            p, attn_norm, wq_a, q_a_norm, wq_b, wkv, kv_a_norm, wo_a, wo_b, attn_sinks,
        )
    }

    /// Cached path with an empty initial cache producing the same N-token
    /// output as the non-cached prefill path. The cache absorbs all the
    /// new kv_a rows and SWA sees the same K/V either way.
    #[test]
    fn cached_full_prefill_equals_uncached() {
        let (p, attn_norm, wq_a, q_a_norm, wq_b, wkv, kv_a_norm, wo_a, wo_b, attn_sinks) =
            build_cached_test_weights();
        let w = DsV4AttnBlockWeights {
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
        let n_tokens = 5;
        let x = Array2::<f32>::from_shape_fn((n_tokens, p.n_embd), |(t, d)| {
            ((t * 13 + d) as f32 * 0.011).sin()
        });

        // Path A: non-cached prefill.
        let out_prefill = dsv4_attn_block_no_compress(x.view(), &w, &p, 0, None);

        // Path B: cached path with empty cache.
        let mut cache = DsV4LayerKvCache::with_capacity(64, p.head_dim);
        let out_cached = dsv4_attn_block_no_compress_cached(x.view(), &w, &p, 0, &mut cache, None);

        assert_eq!(out_prefill.shape(), out_cached.shape());
        let mut max_diff = 0.0_f32;
        for (a, b) in out_prefill.iter().zip(out_cached.iter()) {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        // Same computation deterministically — exact match expected
        // (no batch reduction order changes).
        assert!(max_diff < 1e-5, "cached vs prefill max diff = {max_diff}");
        assert_eq!(cache.current_len(), n_tokens);
    }

    /// Incremental: prefill 4 tokens via cached path, then decode 1
    /// token via cached path. The accumulated 5-row output should
    /// match running all 5 tokens through cached path with an empty
    /// cache (since attention is causal — token 4's output depends
    /// only on tokens 0..4).
    #[test]
    fn cached_prefill_then_decode_equals_full_cached() {
        let (p, attn_norm, wq_a, q_a_norm, wq_b, wkv, kv_a_norm, wo_a, wo_b, attn_sinks) =
            build_cached_test_weights();
        let w = DsV4AttnBlockWeights {
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
        let n_total = 5;
        let x_full = Array2::<f32>::from_shape_fn((n_total, p.n_embd), |(t, d)| {
            ((t * 13 + d) as f32 * 0.011).sin()
        });

        // Path A: one-shot cached call.
        let mut cache_a = DsV4LayerKvCache::with_capacity(32, p.head_dim);
        let out_full =
            dsv4_attn_block_no_compress_cached(x_full.view(), &w, &p, 0, &mut cache_a, None);

        // Path B: split prefill (4) + decode (1) via the cached path.
        let mut cache_b = DsV4LayerKvCache::with_capacity(32, p.head_dim);
        let x_prefill = x_full.slice(ndarray::s![..4, ..]);
        let x_decode = x_full.slice(ndarray::s![4..5, ..]);
        let out_prefill =
            dsv4_attn_block_no_compress_cached(x_prefill, &w, &p, 0, &mut cache_b, None);
        let out_decode =
            dsv4_attn_block_no_compress_cached(x_decode, &w, &p, 4, &mut cache_b, None);

        // Compare: out_prefill (rows 0..4) + out_decode (row 4) == out_full.
        for t in 0..4 {
            for d in 0..p.n_embd {
                let a = out_full[[t, d]];
                let b = out_prefill[[t, d]];
                assert!(
                    (a - b).abs() < 1e-5,
                    "prefill row {t} dim {d}: full {a} vs split {b}"
                );
            }
        }
        for d in 0..p.n_embd {
            let a = out_full[[4, d]];
            let b = out_decode[[0, d]];
            assert!(
                (a - b).abs() < 1e-5,
                "decode row 4 dim {d}: full {a} vs split {b}"
            );
        }
        // Both caches end with the same length.
        assert_eq!(cache_a.current_len(), n_total);
        assert_eq!(cache_b.current_len(), n_total);
    }

    /// position_offset mismatch with cache.current_len() panics.
    #[test]
    #[should_panic(expected = "position_offset")]
    fn cached_position_offset_mismatch_panics() {
        let (p, attn_norm, wq_a, q_a_norm, wq_b, wkv, kv_a_norm, wo_a, wo_b, attn_sinks) =
            build_cached_test_weights();
        let w = DsV4AttnBlockWeights {
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
        // Cache is empty (current_len=0) but caller claims offset=5 → panic.
        let mut cache = DsV4LayerKvCache::with_capacity(16, p.head_dim);
        let x = Array2::<f32>::zeros((1, p.n_embd));
        let _ = dsv4_attn_block_no_compress_cached(x.view(), &w, &p, 5, &mut cache, None);
    }

    /// head_dim mismatch between cache and params panics.
    #[test]
    #[should_panic(expected = "cache.head_dim")]
    fn cached_head_dim_mismatch_panics() {
        let (p, attn_norm, wq_a, q_a_norm, wq_b, wkv, kv_a_norm, wo_a, wo_b, attn_sinks) =
            build_cached_test_weights();
        let w = DsV4AttnBlockWeights {
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
        // Cache head_dim = 32 but params head_dim = 64 → panic.
        let mut cache = DsV4LayerKvCache::with_capacity(16, 32);
        let x = Array2::<f32>::zeros((1, p.n_embd));
        let _ = dsv4_attn_block_no_compress_cached(x.view(), &w, &p, 0, &mut cache, None);
    }
}
