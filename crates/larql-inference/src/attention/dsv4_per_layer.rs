//! DSv4 per-layer forward — the full composition of every prior stage.
//!
//! Takes a 4-stream residual `(n_tokens, n_hc, n_embd)`, runs both
//! halves of the layer (attention + FFN), each bookended by an mHC
//! pre/post pair, and returns the next 4-stream residual.
//!
//! Pipeline:
//! ```text
//! 1. mhc_pre  (attn bookend) → cur_attn_in, post_attn, comb_attn
//! 2. attn_out = dsv4_attn_layer(cur_attn_in, attn_variant, position_offset)
//! 3. residual = mhc_post (attn_out, residual, post_attn, comb_attn)
//! 4. mhc_pre  (ffn  bookend) → cur_ffn_in, post_ffn, comb_ffn
//! 5. ffn_out  = dsv4_ffn_block(cur_ffn_in, ffn_weights, ffn_params, token_ids)
//! 6. residual = mhc_post (ffn_out, residual, post_ffn, comb_ffn)
//! ```
//!
//! Reference: llama.cpp PR #23122 `src/models/deepseek4.cpp:1140-1561`
//! (the per-layer body of the DSv4 forward graph).

use ndarray::{Array3, ArrayView3};

use larql_compute::ComputeBackend;

use super::dsv4_attn_block::dsv4_attn_block_no_compress_cached;
use super::dsv4_attn_block_compress::dsv4_attn_block_compress_no_indexer_cached;
use super::dsv4_attn_block_indexer::dsv4_attn_block_compress_with_indexer_cached;
use super::dsv4_attn_dispatch::{dsv4_attn_layer, DsV4AttnLayer};
use super::dsv4_ffn_block::{dsv4_ffn_block, Dsv4FfnParams, Dsv4FfnWeights};
use super::dsv4_kv_cache::DsV4LayerCache;
use super::dsv4_mhc_bookend::{dsv4_mhc_post, dsv4_mhc_pre, MhcParams, MhcWeights};

/// All weights for a single DSv4 layer: attention variant + its mHC
/// bookend + FFN block + its mHC bookend.
pub struct DsV4LayerWeights<'a, 'p> {
    /// mHC weights for the attention bookend.
    pub mhc_attn: MhcWeights<'a>,
    /// Attention dispatch variant (NoCompress / Compress / Indexer).
    pub attn: DsV4AttnLayer<'a, 'p>,
    /// mHC weights for the FFN bookend.
    pub mhc_ffn: MhcWeights<'a>,
    /// FFN block weights (MoE routing + dispatch + shared expert).
    pub ffn: Dsv4FfnWeights<'a>,
}

/// Per-layer scalar config.
#[derive(Clone, Copy, Debug)]
pub struct DsV4LayerParams {
    /// mHC params (shared between attn and FFN bookends — both use the
    /// same `n_hc`, `sinkhorn_iters`, etc.).
    pub mhc: MhcParams,
    pub ffn: Dsv4FfnParams,
}

/// Run a single DSv4 layer's forward pass.
///
/// `residual`: `(n_tokens, n_hc, n_embd)` — the 4-stream residual.
/// `token_ids`: required when any FFN layer uses hash routing (i.e.,
/// `ffn.gate_tid2eid.is_some()`).
/// `position_offset`: absolute position of `residual` row 0 (0 for
/// prefill from scratch; cache length for decode).
pub fn dsv4_per_layer_forward(
    residual: ArrayView3<f32>,
    w: &DsV4LayerWeights,
    p: &DsV4LayerParams,
    token_ids: Option<&[u32]>,
    position_offset: usize,
    backend: Option<&dyn ComputeBackend>,
) -> Array3<f32> {
    // ── 1. Attention bookend pre ──
    let pre_attn = dsv4_mhc_pre(residual, &w.mhc_attn, &p.mhc, backend);

    // ── 2. Attention block ──
    let attn_out = dsv4_attn_layer(pre_attn.cur.view(), &w.attn, position_offset, backend);

    // ── 3. Attention bookend post → updated residual ──
    let residual_mid = dsv4_mhc_post(
        attn_out.view(),
        residual,
        pre_attn.post.view(),
        pre_attn.comb.view(),
    );

    // ── 4. FFN bookend pre ──
    let pre_ffn = dsv4_mhc_pre(residual_mid.view(), &w.mhc_ffn, &p.mhc, backend);

    // ── 5. FFN block ──
    let ffn_out = dsv4_ffn_block(pre_ffn.cur.view(), &w.ffn, &p.ffn, token_ids, backend);

    // ── 6. FFN bookend post → final residual ──
    dsv4_mhc_post(
        ffn_out.view(),
        residual_mid.view(),
        pre_ffn.post.view(),
        pre_ffn.comb.view(),
    )
}

/// KV-cached variant of [`dsv4_per_layer_forward`].
///
/// Dispatches on `(attn variant, cache variant)`:
/// - `(NoCompress, DsV4LayerCache::NoCompress)` →
///   [`dsv4_attn_block_no_compress_cached`]
/// - `(Compress, DsV4LayerCache::Hca)` →
///   [`dsv4_attn_block_compress_no_indexer_cached`]
/// - `(Indexer, DsV4LayerCache::Hca)` (with indexer enabled) →
///   [`dsv4_attn_block_compress_with_indexer_cached`]
/// - `layer_cache = None` OR mismatched variant pair → fall back to
///   non-cached [`dsv4_attn_layer`] (correctness preserved, no decode
///   speedup)
///
/// `position_offset` is the absolute position of `residual[0]`. When a
/// cached path is selected it must equal the cache's current sequence
/// position (`DsV4LayerCache::position()`); this is asserted inside
/// the per-variant cached attention functions.
pub fn dsv4_per_layer_forward_cached(
    residual: ArrayView3<f32>,
    w: &DsV4LayerWeights,
    p: &DsV4LayerParams,
    token_ids: Option<&[u32]>,
    position_offset: usize,
    layer_cache: Option<&mut DsV4LayerCache>,
    backend: Option<&dyn ComputeBackend>,
) -> Array3<f32> {
    use super::dsv4_profile::timed;

    // ── 1. Attention bookend pre ──
    let pre_attn = timed("mhc", || {
        dsv4_mhc_pre(residual, &w.mhc_attn, &p.mhc, backend)
    });

    // ── 2. Attention block (cached if variants align, fallback otherwise) ──
    let attn_out = timed("attn", || match (&w.attn, layer_cache) {
        (DsV4AttnLayer::NoCompress { weights, params }, Some(DsV4LayerCache::NoCompress(c))) => {
            dsv4_attn_block_no_compress_cached(
                pre_attn.cur.view(),
                weights,
                params,
                position_offset,
                c,
                backend,
            )
        }
        (DsV4AttnLayer::Compress { weights, params }, Some(DsV4LayerCache::Hca(c))) => {
            dsv4_attn_block_compress_no_indexer_cached(
                pre_attn.cur.view(),
                weights,
                params,
                position_offset,
                c,
                backend,
            )
        }
        (DsV4AttnLayer::Indexer { weights, params }, Some(DsV4LayerCache::Hca(c)))
            if c.index_compressed.is_some() =>
        {
            dsv4_attn_block_compress_with_indexer_cached(
                pre_attn.cur.view(),
                weights,
                params,
                position_offset,
                c,
                backend,
            )
        }
        // No-cache or mismatched variant pair → non-cached path.
        _ => dsv4_attn_layer(pre_attn.cur.view(), &w.attn, position_offset, backend),
    });

    // ── 3. Attention bookend post → updated residual ──
    let residual_mid = timed("mhc", || {
        dsv4_mhc_post(
            attn_out.view(),
            residual,
            pre_attn.post.view(),
            pre_attn.comb.view(),
        )
    });

    // ── 4. FFN bookend pre ──
    let pre_ffn = timed("mhc", || {
        dsv4_mhc_pre(residual_mid.view(), &w.mhc_ffn, &p.mhc, backend)
    });

    // ── 5. FFN block ──
    let ffn_out = timed("ffn", || {
        dsv4_ffn_block(pre_ffn.cur.view(), &w.ffn, &p.ffn, token_ids, backend)
    });

    // ── 6. FFN bookend post → final residual ──
    timed("mhc", || {
        dsv4_mhc_post(
            ffn_out.view(),
            residual_mid.view(),
            pre_ffn.post.view(),
            pre_ffn.comb.view(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_attn_block::{DsV4AttnBlockParams, DsV4AttnBlockWeights};
    use super::super::dsv4_ffn_block::SharedExpertWeights;
    use super::super::dsv4_moe_dispatch::MoeExpertWeights;
    use super::super::dsv4_rope_tail::DsV4RopeMode;
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    /// Full per-layer forward: 4-stream residual → 4-stream residual.
    /// No-compress attention + non-hash FFN routing.
    #[test]
    fn per_layer_forward_smoke() {
        let n_tokens = 4;
        let n_hc = 4;
        let n_embd = 64;
        let n_head = 4;
        let head_dim = 64;
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let group_heads = n_head / n_groups;
        let group_dim = head_dim * group_heads;
        let low_dim = o_lora_rank * n_groups;
        let n_expert = 6;
        let n_expert_used = 2;
        let n_ff_exp = 8;
        let hidden_shared = 4;

        // Residual: small random 4-stream tensor.
        let residual = Array3::<f32>::from_shape_fn((n_tokens, n_hc, n_embd), |(t, h, d)| {
            ((t * 17 + h * 7 + d) as f32 * 0.013).sin()
        });

        // ── Attention weights ──
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
        let attn_params = DsV4AttnBlockParams {
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

        // ── mHC weights (separate for attn / FFN bookends) ──
        let hc_dim = n_hc * n_embd;
        let hc_mix = (2 + n_hc) * n_hc;
        let hc_fn_attn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m + k) as f32 * 0.003).sin() * 0.1
        });
        let hc_scale = [0.5_f32, 0.5, 0.5];
        let hc_base = vec![0.0_f32; hc_mix];
        let hc_fn_ffn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m + k) as f32 * 0.004).cos() * 0.1
        });

        // ── FFN weights ──
        let ffn_norm = vec![1.0_f32; n_embd];
        let gate_inp = Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| {
            ((e + d) as f32 * 0.01).cos() * 0.1
        });
        let gate_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.011).sin() * 0.1
        });
        let up_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.013).cos() * 0.1
        });
        let down_exps = Array3::<f32>::from_shape_fn((n_expert, n_embd, n_ff_exp), |(e, d, i)| {
            ((e + d + i) as f32 * 0.009).sin() * 0.1
        });
        let gate_shexp = Array2::<f32>::from_shape_fn((hidden_shared, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.015).sin() * 0.1
        });
        let up_shexp = Array2::<f32>::from_shape_fn((hidden_shared, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.017).cos() * 0.1
        });
        let down_shexp = Array2::<f32>::from_shape_fn((n_embd, hidden_shared), |(d, i)| {
            ((d + i) as f32 * 0.019).sin() * 0.1
        });

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
        let layer_w = DsV4LayerWeights {
            mhc_attn: MhcWeights {
                hc_fn: hc_fn_attn.view(),
                hc_scale: &hc_scale,
                hc_base: &hc_base,
            },
            attn: DsV4AttnLayer::NoCompress {
                weights: attn_w,
                params: &attn_params,
            },
            mhc_ffn: MhcWeights {
                hc_fn: hc_fn_ffn.view(),
                hc_scale: &hc_scale,
                hc_base: &hc_base,
            },
            ffn: Dsv4FfnWeights {
                ffn_norm: &ffn_norm,
                gate_inp: gate_inp.view(),
                exp_probs_b: None,
                gate_tid2eid: None,
                moe: MoeExpertWeights {
                    gate_exps: gate_exps.view(),
                    up_exps: up_exps.view(),
                    down_exps: down_exps.view(),
                    quant: None,
                },
                shared: SharedExpertWeights {
                    quant: None,
                    gate_shexp: gate_shexp.view(),
                    up_shexp: up_shexp.view(),
                    down_shexp: down_shexp.view(),
                },
            },
        };
        let layer_p = DsV4LayerParams {
            mhc: MhcParams {
                n_embd,
                n_hc,
                sinkhorn_iters: 2,
                hc_eps: 1e-5,
                norm_eps: 1e-5,
            },
            ffn: Dsv4FfnParams {
                n_expert,
                n_expert_used,
                norm_eps: 1e-5,
                routed_swiglu_limit: 7.0,
                shared_swiglu_limit: 0.0,
                expert_weights_norm: true,
                expert_weights_scale: 1.0,
            },
        };

        let out = dsv4_per_layer_forward(residual.view(), &layer_w, &layer_p, None, 0, None);
        assert_eq!(out.shape(), &[n_tokens, n_hc, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(total > 1e-3, "output collapsed (sum={total})");
    }

    /// Helper: build the per_layer_forward_smoke setup as an owned
    /// bundle so cached / non-cached paths share the same weights.
    /// Returns `(layer_w, layer_p, residual, n_tokens, n_hc, n_embd)`.
    fn build_no_compress_layer() -> (
        // Owned arrays we have to keep alive for the duration of the test.
        Vec<f32>,    // attn_norm
        Array2<f32>, // wq_a
        Vec<f32>,    // q_a_norm
        Array2<f32>, // wq_b
        Array2<f32>, // wkv
        Vec<f32>,    // kv_a_norm
        Array3<f32>, // wo_a
        Array2<f32>, // wo_b
        Array1<f32>, // attn_sinks
        Array2<f32>, // hc_fn_attn
        Array2<f32>, // hc_fn_ffn
        [f32; 3],    // hc_scale
        Vec<f32>,    // hc_base
        Vec<f32>,    // ffn_norm
        Array2<f32>, // gate_inp
        Array3<f32>, // gate_exps
        Array3<f32>, // up_exps
        Array3<f32>, // down_exps
        Array2<f32>, // gate_shexp
        Array2<f32>, // up_shexp
        Array2<f32>, // down_shexp
        DsV4AttnBlockParams,
        DsV4LayerParams,
        Array3<f32>, // residual
        usize,       // n_tokens
        usize,       // n_hc
        usize,       // n_embd
    ) {
        let n_tokens = 4;
        let n_hc = 4;
        let n_embd = 64;
        let n_head = 4;
        let head_dim = 64;
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let group_heads = n_head / n_groups;
        let group_dim = head_dim * group_heads;
        let low_dim = o_lora_rank * n_groups;
        let n_expert = 6;
        let n_expert_used = 2;
        let n_ff_exp = 8;
        let hidden_shared = 4;

        let residual = Array3::<f32>::from_shape_fn((n_tokens, n_hc, n_embd), |(t, h, d)| {
            ((t * 17 + h * 7 + d) as f32 * 0.013).sin()
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
        let attn_params = DsV4AttnBlockParams {
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

        let hc_dim = n_hc * n_embd;
        let hc_mix = (2 + n_hc) * n_hc;
        let hc_fn_attn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m + k) as f32 * 0.003).sin() * 0.1
        });
        let hc_scale = [0.5_f32, 0.5, 0.5];
        let hc_base = vec![0.0_f32; hc_mix];
        let hc_fn_ffn = Array2::<f32>::from_shape_fn((hc_mix, hc_dim), |(m, k)| {
            ((m + k) as f32 * 0.004).cos() * 0.1
        });

        let ffn_norm = vec![1.0_f32; n_embd];
        let gate_inp = Array2::<f32>::from_shape_fn((n_expert, n_embd), |(e, d)| {
            ((e + d) as f32 * 0.01).cos() * 0.1
        });
        let gate_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.011).sin() * 0.1
        });
        let up_exps = Array3::<f32>::from_shape_fn((n_expert, n_ff_exp, n_embd), |(e, i, j)| {
            ((e + i + j) as f32 * 0.013).cos() * 0.1
        });
        let down_exps = Array3::<f32>::from_shape_fn((n_expert, n_embd, n_ff_exp), |(e, d, i)| {
            ((e + d + i) as f32 * 0.009).sin() * 0.1
        });
        let gate_shexp = Array2::<f32>::from_shape_fn((hidden_shared, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.015).sin() * 0.1
        });
        let up_shexp = Array2::<f32>::from_shape_fn((hidden_shared, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.017).cos() * 0.1
        });
        let down_shexp = Array2::<f32>::from_shape_fn((n_embd, hidden_shared), |(d, i)| {
            ((d + i) as f32 * 0.019).sin() * 0.1
        });

        let layer_p = DsV4LayerParams {
            mhc: super::super::dsv4_mhc_bookend::MhcParams {
                n_embd,
                n_hc,
                sinkhorn_iters: 2,
                hc_eps: 1e-5,
                norm_eps: 1e-5,
            },
            ffn: super::super::dsv4_ffn_block::Dsv4FfnParams {
                n_expert,
                n_expert_used,
                norm_eps: 1e-5,
                routed_swiglu_limit: 7.0,
                shared_swiglu_limit: 0.0,
                expert_weights_norm: true,
                expert_weights_scale: 1.0,
            },
        };

        (
            attn_norm,
            wq_a,
            q_a_norm,
            wq_b,
            wkv,
            kv_a_norm,
            wo_a,
            wo_b,
            attn_sinks,
            hc_fn_attn,
            hc_fn_ffn,
            hc_scale,
            hc_base,
            ffn_norm,
            gate_inp,
            gate_exps,
            up_exps,
            down_exps,
            gate_shexp,
            up_shexp,
            down_shexp,
            attn_params,
            layer_p,
            residual,
            n_tokens,
            n_hc,
            n_embd,
        )
    }

    /// With `layer_cache = None`, the cached function produces the
    /// same output as the existing non-cached `dsv4_per_layer_forward`.
    #[test]
    fn cached_with_no_cache_equals_uncached() {
        use super::super::dsv4_attn_block::DsV4AttnBlockWeights;
        use super::super::dsv4_ffn_block::SharedExpertWeights;
        use super::super::dsv4_moe_dispatch::MoeExpertWeights;

        let (
            attn_norm,
            wq_a,
            q_a_norm,
            wq_b,
            wkv,
            kv_a_norm,
            wo_a,
            wo_b,
            attn_sinks,
            hc_fn_attn,
            hc_fn_ffn,
            hc_scale,
            hc_base,
            ffn_norm,
            gate_inp,
            gate_exps,
            up_exps,
            down_exps,
            gate_shexp,
            up_shexp,
            down_shexp,
            attn_params,
            layer_p,
            residual,
            _n_tokens,
            _n_hc,
            _n_embd,
        ) = build_no_compress_layer();

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
        let layer_w = DsV4LayerWeights {
            mhc_attn: MhcWeights {
                hc_fn: hc_fn_attn.view(),
                hc_scale: &hc_scale,
                hc_base: &hc_base,
            },
            attn: DsV4AttnLayer::NoCompress {
                weights: attn_w,
                params: &attn_params,
            },
            mhc_ffn: MhcWeights {
                hc_fn: hc_fn_ffn.view(),
                hc_scale: &hc_scale,
                hc_base: &hc_base,
            },
            ffn: Dsv4FfnWeights {
                ffn_norm: &ffn_norm,
                gate_inp: gate_inp.view(),
                exp_probs_b: None,
                gate_tid2eid: None,
                moe: MoeExpertWeights {
                    gate_exps: gate_exps.view(),
                    up_exps: up_exps.view(),
                    down_exps: down_exps.view(),
                    quant: None,
                },
                shared: SharedExpertWeights {
                    quant: None,
                    gate_shexp: gate_shexp.view(),
                    up_shexp: up_shexp.view(),
                    down_shexp: down_shexp.view(),
                },
            },
        };

        let out_nocache =
            dsv4_per_layer_forward(residual.view(), &layer_w, &layer_p, None, 0, None);
        let out_cached =
            dsv4_per_layer_forward_cached(residual.view(), &layer_w, &layer_p, None, 0, None, None);

        assert_eq!(out_nocache.shape(), out_cached.shape());
        let mut max_diff = 0.0_f32;
        for (a, b) in out_nocache.iter().zip(out_cached.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-5,
            "cached(cache=None) vs non-cached: max diff = {max_diff}"
        );
    }

    /// With `layer_cache = Some(_)` and NoCompress variant: full
    /// prefill via cached path yields same output as non-cached, AND
    /// the cache fills with n_tokens rows.
    #[test]
    fn cached_with_empty_cache_full_prefill_equals_uncached() {
        use super::super::dsv4_attn_block::DsV4AttnBlockWeights;
        use super::super::dsv4_ffn_block::SharedExpertWeights;
        use super::super::dsv4_moe_dispatch::MoeExpertWeights;

        let (
            attn_norm,
            wq_a,
            q_a_norm,
            wq_b,
            wkv,
            kv_a_norm,
            wo_a,
            wo_b,
            attn_sinks,
            hc_fn_attn,
            hc_fn_ffn,
            hc_scale,
            hc_base,
            ffn_norm,
            gate_inp,
            gate_exps,
            up_exps,
            down_exps,
            gate_shexp,
            up_shexp,
            down_shexp,
            attn_params,
            layer_p,
            residual,
            n_tokens,
            _n_hc,
            _n_embd,
        ) = build_no_compress_layer();

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
        let layer_w = DsV4LayerWeights {
            mhc_attn: MhcWeights {
                hc_fn: hc_fn_attn.view(),
                hc_scale: &hc_scale,
                hc_base: &hc_base,
            },
            attn: DsV4AttnLayer::NoCompress {
                weights: attn_w,
                params: &attn_params,
            },
            mhc_ffn: MhcWeights {
                hc_fn: hc_fn_ffn.view(),
                hc_scale: &hc_scale,
                hc_base: &hc_base,
            },
            ffn: Dsv4FfnWeights {
                ffn_norm: &ffn_norm,
                gate_inp: gate_inp.view(),
                exp_probs_b: None,
                gate_tid2eid: None,
                moe: MoeExpertWeights {
                    gate_exps: gate_exps.view(),
                    up_exps: up_exps.view(),
                    down_exps: down_exps.view(),
                    quant: None,
                },
                shared: SharedExpertWeights {
                    quant: None,
                    gate_shexp: gate_shexp.view(),
                    up_shexp: up_shexp.view(),
                    down_shexp: down_shexp.view(),
                },
            },
        };

        let out_nocache =
            dsv4_per_layer_forward(residual.view(), &layer_w, &layer_p, None, 0, None);

        let mut cache = DsV4LayerCache::new_no_compress(32, attn_params.head_dim);
        let out_cached = dsv4_per_layer_forward_cached(
            residual.view(),
            &layer_w,
            &layer_p,
            None,
            0,
            Some(&mut cache),
            None,
        );

        let mut max_diff = 0.0_f32;
        for (a, b) in out_nocache.iter().zip(out_cached.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-5,
            "cached(NoCompress, empty cache) vs non-cached: max diff = {max_diff}"
        );
        assert_eq!(cache.position(), n_tokens);
    }
}
