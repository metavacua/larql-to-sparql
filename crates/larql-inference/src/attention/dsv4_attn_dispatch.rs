//! DeepSeek V4 per-layer attention dispatcher.
//!
//! DSv4 layers come in three flavors depending on the per-layer
//! `compress_ratio`:
//! - `0`: standard sliding-window attention + sinks
//!   (Stage 8a, [`super::dsv4_attn_block::dsv4_attn_block_no_compress`])
//! - `1, 2, 3, 5, ...`: raw + compressed KV with a static causal mask
//!   (Stage 8f,
//!   [`super::dsv4_attn_block_compress::dsv4_attn_block_compress_no_indexer`])
//! - `4`: raw + compressed KV with indexer-driven top-k sparse mask
//!   (Stage 8g,
//!   [`super::dsv4_attn_block_indexer::dsv4_attn_block_compress_with_indexer`])
//!
//! This module wraps all three behind a single tagged-union API so the
//! per-layer forward loop just sees one entry point. GGUF loaders
//! (Stage 8h-2) construct the variant per-layer from the layer's
//! `compress_ratio` metadata.

use larql_compute::ComputeBackend;
use ndarray::{Array2, ArrayView2};

use super::dsv4_attn_block::{
    dsv4_attn_block_no_compress, DsV4AttnBlockParams, DsV4AttnBlockWeights,
};
use super::dsv4_attn_block_compress::{
    dsv4_attn_block_compress_no_indexer, DsV4AttnBlockCompressParams, DsV4AttnBlockCompressWeights,
};
use super::dsv4_attn_block_indexer::{
    dsv4_attn_block_compress_with_indexer, DsV4AttnBlockIndexerParams, DsV4AttnBlockIndexerWeights,
};

/// Per-layer attention variant. The variant determines which CPU
/// reference block runs; the per-layer GGUF loader populates the right
/// arm based on `compress_ratio` metadata.
pub enum DsV4AttnLayer<'a, 'p> {
    /// `compress_ratio == 0`: standard sliding-window attention + sinks.
    NoCompress {
        weights: DsV4AttnBlockWeights<'a>,
        params: &'p DsV4AttnBlockParams,
    },
    /// `compress_ratio` in `{1, 2, 3, 5, 6, ...}`: raw + compressed KV
    /// with a static causal mask.
    Compress {
        weights: DsV4AttnBlockCompressWeights<'a>,
        params: &'p DsV4AttnBlockCompressParams,
    },
    /// `compress_ratio == 4`: raw + compressed KV with indexer-driven
    /// top-k sparse mask.
    Indexer {
        weights: DsV4AttnBlockIndexerWeights<'a>,
        params: &'p DsV4AttnBlockIndexerParams,
    },
}

impl DsV4AttnLayer<'_, '_> {
    /// Identifying name for the variant — useful for logging and
    /// debugging which block runs at each layer.
    pub fn variant_name(&self) -> &'static str {
        match self {
            DsV4AttnLayer::NoCompress { .. } => "no_compress",
            DsV4AttnLayer::Compress { .. } => "compress",
            DsV4AttnLayer::Indexer { .. } => "indexer",
        }
    }

    /// Effective `compress_ratio` of this layer (`0` for the
    /// no-compress variant).
    pub fn compress_ratio(&self) -> usize {
        match self {
            DsV4AttnLayer::NoCompress { .. } => 0,
            DsV4AttnLayer::Compress { params, .. } => params.compressor.compress_ratio,
            DsV4AttnLayer::Indexer { params, .. } => params.compressor.compress_ratio,
        }
    }
}

/// Run the attention block for this layer. Dispatches to the variant
/// matching the layer's `compress_ratio`.
///
/// `backend` routes the per-block matmul shapes through the optional
/// compute backend (cuBLAS / Metal / CPU). All three prefill variants
/// (NoCompress GPU-4a, Compress GPU-4b, Indexer GPU-4c) thread it.
pub fn dsv4_attn_layer(
    x: ArrayView2<f32>,
    layer: &DsV4AttnLayer,
    position_offset: usize,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    match layer {
        DsV4AttnLayer::NoCompress { weights, params } => {
            dsv4_attn_block_no_compress(x, weights, params, position_offset, backend)
        }
        DsV4AttnLayer::Compress { weights, params } => {
            dsv4_attn_block_compress_no_indexer(x, weights, params, position_offset, backend)
        }
        DsV4AttnLayer::Indexer { weights, params } => {
            dsv4_attn_block_compress_with_indexer(x, weights, params, position_offset, backend)
        }
    }
}

/// Construct the variant tag corresponding to a `compress_ratio` value,
/// for caller-side validation that they're building the right arm.
/// `0` → NoCompress, `4` → Indexer, anything else > 0 → Compress.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DsV4AttnVariantTag {
    NoCompress,
    Compress,
    Indexer,
}

impl DsV4AttnVariantTag {
    pub fn for_compress_ratio(compress_ratio: usize) -> Self {
        match compress_ratio {
            0 => DsV4AttnVariantTag::NoCompress,
            4 => DsV4AttnVariantTag::Indexer,
            _ => DsV4AttnVariantTag::Compress,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_attn_block::DsV4AttnBlockParams;
    use super::super::dsv4_attn_block_compress::DsV4AttnBlockCompressParams;
    use super::super::dsv4_attn_block_indexer::DsV4AttnBlockIndexerParams;
    use super::super::dsv4_compressor_prefill::{CompressorParams, CompressorWeights};
    use super::super::dsv4_indexer::{IndexerParams, IndexerWeights};
    use super::super::dsv4_rope_tail::DsV4RopeMode;
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    /// Tag dispatch: 0 → NoCompress, 4 → Indexer, others → Compress.
    #[test]
    fn tag_for_compress_ratio() {
        assert_eq!(
            DsV4AttnVariantTag::for_compress_ratio(0),
            DsV4AttnVariantTag::NoCompress
        );
        assert_eq!(
            DsV4AttnVariantTag::for_compress_ratio(4),
            DsV4AttnVariantTag::Indexer
        );
        for r in [1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(
                DsV4AttnVariantTag::for_compress_ratio(r),
                DsV4AttnVariantTag::Compress,
                "compress_ratio = {r} should be Compress"
            );
        }
    }

    /// Variant accessors return the right values.
    #[test]
    fn variant_name_and_compress_ratio() {
        let attn_p = DsV4AttnBlockParams {
            n_embd: 16,
            n_head: 2,
            head_dim: 8,
            q_lora_rank: 4,
            n_groups: 1,
            o_lora_rank: 4,
            n_rot: 0,
            rope_base: 10000.0,
            rope_mode: DsV4RopeMode::Neox,
            window_size: 4,
            norm_eps: 1e-5,
            yarn: None,
        };
        let attn_norm = vec![1.0_f32; 16];
        let wq_a = Array2::<f32>::zeros((4, 16));
        let q_a_norm = vec![1.0_f32; 4];
        let wq_b = Array2::<f32>::zeros((16, 4));
        let wkv = Array2::<f32>::zeros((8, 16));
        let kv_a_norm = vec![1.0_f32; 8];
        let wo_a = Array3::<f32>::zeros((1, 4, 16));
        let wo_b = Array2::<f32>::zeros((16, 4));
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
            attn_sinks: None,
        };
        let nc = DsV4AttnLayer::NoCompress {
            weights: attn_w,
            params: &attn_p,
        };
        assert_eq!(nc.variant_name(), "no_compress");
        assert_eq!(nc.compress_ratio(), 0);
    }

    /// End-to-end dispatch: same input through each variant produces
    /// finite output (regression on the routing wiring). Different
    /// variants produce different outputs since each runs a different
    /// attention block.
    #[test]
    fn dispatch_routes_each_variant() {
        // ── Shared setup: dims small enough to keep test fast ──
        let n_tokens = 8;
        let n_embd = 64;
        let n_head = 4;
        let head_dim = 64; // n_rot = 0 → n_nope = 64 (one FP8 block)
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let group_heads = n_head / n_groups;
        let group_dim = head_dim * group_heads;
        let low_dim = o_lora_rank * n_groups;
        let n_rot = 0;
        let rope_base = 10000.0;
        let rope_mode = DsV4RopeMode::Neox;
        let window_size = 8;
        let norm_eps = 1e-5;

        let attn_p = DsV4AttnBlockParams {
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
        };

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

        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 7 + d) as f32 * 0.013).sin()
        });

        // ── NoCompress variant ──
        let layer_nc = DsV4AttnLayer::NoCompress {
            weights: attn_w,
            params: &attn_p,
        };
        let out_nc = dsv4_attn_layer(x.view(), &layer_nc, 0, None);
        assert_eq!(out_nc.shape(), &[n_tokens, n_embd]);
        assert!(out_nc.iter().all(|v| v.is_finite()));

        // ── Compress variant (compress_ratio = 2) ──
        let comp_p = CompressorParams {
            head_dim,
            n_embd,
            compress_ratio: 2,
            n_rot,
            rope_base,
            rope_mode,
            norm_eps,
        };
        let n_kv = head_dim;
        let comp_wkv = Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.013).cos() * 0.05
        });
        let comp_wgate = Array2::<f32>::from_shape_fn((n_kv, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.017).sin() * 0.05
        });
        let comp_ape =
            Array2::<f32>::from_shape_fn((2, n_kv), |(r, k)| ((r + k) as f32 * 0.03).sin() * 0.05);
        let comp_norm = vec![1.0_f32; head_dim];
        let comp_w = CompressorWeights {
            wkv: comp_wkv.view(),
            wgate: comp_wgate.view(),
            ape: comp_ape.view(),
            norm: &comp_norm,
            quant: None,
        };
        let block_compress_w = DsV4AttnBlockCompressWeights {
            attn: attn_w,
            compressor: comp_w,
        };
        let block_compress_p = DsV4AttnBlockCompressParams {
            attn: attn_p,
            compressor: comp_p,
        };
        let layer_c = DsV4AttnLayer::Compress {
            weights: block_compress_w,
            params: &block_compress_p,
        };
        let out_c = dsv4_attn_layer(x.view(), &layer_c, 0, None);
        assert_eq!(out_c.shape(), &[n_tokens, n_embd]);
        assert!(out_c.iter().all(|v| v.is_finite()));

        // Confirm dispatch actually picks different code paths: the
        // Compress variant's output must differ from NoCompress (since
        // the compressed KV is concatenated into the attention).
        let diff_c_nc: f32 = out_c
            .iter()
            .zip(out_nc.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff_c_nc > 1e-3,
            "Compress should differ from NoCompress (diff={diff_c_nc})"
        );
        assert_eq!(layer_c.variant_name(), "compress");
        assert_eq!(layer_c.compress_ratio(), 2);
        assert_eq!(layer_nc.variant_name(), "no_compress");
        assert_eq!(layer_nc.compress_ratio(), 0);
    }

    /// `compress_ratio = 4` routes to the Indexer variant.
    #[test]
    fn dispatch_routes_indexer_variant() {
        let n_tokens = 16;
        let n_embd = 64;
        let n_head = 4;
        let head_dim = 64;
        let q_lora_rank = 16;
        let n_groups = 2;
        let o_lora_rank = 8;
        let group_heads = n_head / n_groups;
        let group_dim = head_dim * group_heads;
        let low_dim = o_lora_rank * n_groups;
        let n_rot = 0;
        let rope_base = 10000.0;
        let rope_mode = DsV4RopeMode::Neox;
        let window_size = 8;
        let norm_eps = 1e-5;
        let indexer_head_size = 16;
        let n_index_head = 2;

        let attn_p = DsV4AttnBlockParams {
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
        };
        let comp_p = CompressorParams {
            head_dim,
            n_embd,
            compress_ratio: 4,
            n_rot,
            rope_base,
            rope_mode,
            norm_eps,
        };
        let idx_comp_p = CompressorParams {
            head_dim: indexer_head_size,
            n_embd,
            compress_ratio: 4,
            n_rot,
            rope_base,
            rope_mode,
            norm_eps,
        };
        let idx_p = IndexerParams {
            n_embd,
            q_lora_rank,
            n_index_head,
            n_index_head_size: indexer_head_size,
            n_rot,
            rope_base,
            rope_mode,
        };
        let block_indexer_p = DsV4AttnBlockIndexerParams {
            attn: attn_p,
            compressor: comp_p,
            indexer_compressor: idx_comp_p,
            indexer: idx_p,
            top_k: 2,
        };

        // Construct synthetic weights (same shape rules as Stage 8g test).
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
        let comp_w = CompressorWeights {
            wkv: comp_wkv.view(),
            wgate: comp_wgate.view(),
            ape: comp_ape.view(),
            norm: &comp_norm,
            quant: None,
        };

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
        let idx_comp_w = CompressorWeights {
            wkv: idx_comp_wkv.view(),
            wgate: idx_comp_wgate.view(),
            ape: idx_comp_ape.view(),
            norm: &idx_comp_norm,
            quant: None,
        };

        let idx_wq_b = Array2::<f32>::from_shape_fn(
            (n_index_head * indexer_head_size, q_lora_rank),
            |(i, j)| ((i + j) as f32 * 0.011).sin() * 0.1,
        );
        let idx_wproj = Array2::<f32>::from_shape_fn((n_index_head, n_embd), |(i, j)| {
            ((i + j) as f32 * 0.012).cos() * 0.1
        });
        let idx_w = IndexerWeights {
            wq_b: idx_wq_b.view(),
            wproj: idx_wproj.view(),
            quant: None,
        };

        let block_indexer_w = DsV4AttnBlockIndexerWeights {
            attn: attn_w,
            compressor: comp_w,
            indexer_compressor: idx_comp_w,
            indexer: idx_w,
        };
        let layer_i = DsV4AttnLayer::Indexer {
            weights: block_indexer_w,
            params: &block_indexer_p,
        };

        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 7 + d) as f32 * 0.013).sin()
        });
        let out = dsv4_attn_layer(x.view(), &layer_i, 0, None);
        assert_eq!(out.shape(), &[n_tokens, n_embd]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(layer_i.variant_name(), "indexer");
        assert_eq!(layer_i.compress_ratio(), 4);
    }
}
