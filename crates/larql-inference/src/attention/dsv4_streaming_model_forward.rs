//! Memory-bounded DSv4 model forward via per-layer streaming.
//!
//! The full 43-layer DSv4-Flash model holds ~1.1 TB of FFN weights in
//! aggregate (26 GB f32 per layer × 43), far beyond typical hardware
//! RAM. This module runs the model by loading one layer at a time,
//! running [`dsv4_per_layer_forward`] on the current residual, then
//! dropping the layer's storage before loading the next.
//!
//! Peak RAM stays at ~26 GB (one layer + the residual + the global
//! head) while the effective N-layer forward proceeds end-to-end.
//!
//! Use this as the production entry point for full-model forward on
//! real DSv4-Flash artifacts. The non-streaming
//! [`super::dsv4_model_forward::dsv4_model_forward`] is still useful
//! for small-N tests and synthetic-weight microbenchmarks where
//! holding all layers in memory at once is cheap.
//!
//! ## Per-layer dispatch params
//!
//! The attention dispatch arm (NoCompress / Compress / Indexer) is
//! chosen per-layer from the variant detected in the GGUF. This module
//! constructs the matching [`DsV4AttnBlockCompressParams`] /
//! [`DsV4AttnBlockIndexerParams`] from the model hyperparameters; the
//! caller doesn't have to pre-build per-variant params themselves.

use ndarray::{Array2, Array3};

use larql_compute::{dot_proj_gpu, ComputeBackend};
use larql_models::loading::gguf::GgufFile;

use super::dsv4_attn_block::DsV4AttnBlockParams;
use super::dsv4_attn_block_compress::DsV4AttnBlockCompressParams;
use super::dsv4_attn_block_indexer::DsV4AttnBlockIndexerParams;
use super::dsv4_compressor_prefill::CompressorParams;
use super::dsv4_ffn_block::Dsv4FfnParams;
use super::dsv4_full_loader::{load_dsv4_layer, DsV4LoadError};
use super::dsv4_head_storage::DsV4HeadStorage;
use super::dsv4_indexer::IndexerParams;
use super::dsv4_kv_cache::DsV4LayerCache;
use super::dsv4_layer_variants::DsV4LayerVariant;
use super::dsv4_mhc_bookend::MhcParams;
use super::dsv4_model_forward::{broadcast_to_streams, dsv4_hc_head, embed_tokens};
use super::dsv4_per_layer::{
    dsv4_per_layer_forward, dsv4_per_layer_forward_cached, DsV4LayerParams, DsV4LayerWeights,
};
use super::dsv4_rope_tail::DsV4RopeMode;
use super::dsv4_storage::DsV4LayerWeightStorage;
use super::dsv4_storage_build::DsV4Hyperparams;

/// Build the per-layer [`DsV4LayerParams`] (MoE + mHC scalar config)
/// from model-wide hyperparameters. The same params apply to every
/// layer in DSv4-Flash; per-layer variation is in the attention
/// dispatch arm only.
fn build_layer_params(hp: &DsV4Hyperparams) -> DsV4LayerParams {
    DsV4LayerParams {
        mhc: MhcParams {
            n_embd: hp.n_embd,
            n_hc: hp.n_hc,
            sinkhorn_iters: 2,
            hc_eps: 1e-5,
            norm_eps: hp.norm_eps,
        },
        ffn: Dsv4FfnParams {
            n_expert: hp.n_expert,
            n_expert_used: hp.n_expert_used,
            norm_eps: hp.norm_eps,
            routed_swiglu_limit: 7.0,
            shared_swiglu_limit: 0.0,
            expert_weights_norm: hp.expert_weights_norm,
            expert_weights_scale: hp.expert_weights_scale,
        },
    }
}

/// Build [`DsV4AttnBlockCompressParams`] for a given compress_ratio.
fn build_compress_params(
    hp: &DsV4Hyperparams,
    compress_ratio: usize,
) -> DsV4AttnBlockCompressParams {
    let attn = DsV4AttnBlockParams {
        n_embd: hp.n_embd,
        n_head: hp.n_head,
        head_dim: hp.head_dim,
        q_lora_rank: hp.q_lora_rank,
        n_groups: hp.n_groups,
        o_lora_rank: hp.o_lora_rank,
        n_rot: hp.n_rot,
        rope_base: hp.rope_base,
        rope_mode: DsV4RopeMode::Neox,
        window_size: hp.window_size,
        norm_eps: hp.norm_eps,
        yarn: hp.yarn,
    };
    DsV4AttnBlockCompressParams {
        attn,
        compressor: CompressorParams {
            head_dim: hp.head_dim,
            n_embd: hp.n_embd,
            compress_ratio,
            n_rot: hp.n_rot,
            rope_base: hp.rope_base,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: hp.norm_eps,
        },
    }
}

/// Build [`DsV4AttnBlockIndexerParams`] for a given compress_ratio.
/// Requires `hp.indexer_head_size`, `n_index_head`, `top_k` to be set.
fn build_indexer_params(hp: &DsV4Hyperparams, compress_ratio: usize) -> DsV4AttnBlockIndexerParams {
    let cp = build_compress_params(hp, compress_ratio);
    DsV4AttnBlockIndexerParams {
        attn: cp.attn,
        compressor: cp.compressor,
        indexer_compressor: CompressorParams {
            head_dim: hp.indexer_head_size.expect("indexer_head_size set"),
            n_embd: hp.n_embd,
            compress_ratio,
            n_rot: hp.n_rot,
            rope_base: hp.rope_base,
            rope_mode: DsV4RopeMode::Neox,
            norm_eps: hp.norm_eps,
        },
        indexer: IndexerParams {
            n_embd: hp.n_embd,
            q_lora_rank: hp.q_lora_rank,
            n_index_head: hp.n_index_head.expect("n_index_head set"),
            n_index_head_size: hp.indexer_head_size.expect("indexer_head_size set"),
            n_rot: hp.n_rot,
            rope_base: hp.rope_base,
            rope_mode: DsV4RopeMode::Neox,
        },
        top_k: hp.top_k.expect("top_k set"),
    }
}

/// Pre-built per-variant dispatch params, owned by the caller for the
/// duration of the streaming forward.
///
/// The streaming loop borrows from these per-layer based on the
/// detected [`DsV4LayerVariant`], so the structs need to outlive every
/// per-layer iteration.
struct DispatchParamsPool {
    no_cp: Option<DsV4AttnBlockCompressParams>,
    no_ip: Option<DsV4AttnBlockIndexerParams>,
    cp_4: Option<DsV4AttnBlockCompressParams>,
    ip_4: Option<DsV4AttnBlockIndexerParams>,
    cp_128: Option<DsV4AttnBlockCompressParams>,
}

impl DispatchParamsPool {
    fn new(hp: &DsV4Hyperparams) -> Self {
        let cp_4 = Some(build_compress_params(hp, 4));
        let ip_4 = if hp.indexer_head_size.is_some() {
            Some(build_indexer_params(hp, 4))
        } else {
            None
        };
        let cp_128 = Some(build_compress_params(hp, 128));
        Self {
            no_cp: None,
            no_ip: None,
            cp_4,
            ip_4,
            cp_128,
        }
    }

    fn pick(
        &self,
        variant: &DsV4LayerVariant,
    ) -> (
        &Option<DsV4AttnBlockCompressParams>,
        &Option<DsV4AttnBlockIndexerParams>,
    ) {
        match (variant.compress_ratio, variant.has_indexer) {
            (None, _) => (&self.no_cp, &self.no_ip),
            (Some(4), true) => (&self.cp_4, &self.ip_4),
            (Some(4), false) => (&self.cp_4, &self.no_ip),
            (Some(128), _) => (&self.cp_128, &self.no_ip),
            (Some(other), _) => panic!(
                "unsupported compress_ratio {other} (only 4 and 128 are known for DSv4-Flash)"
            ),
        }
    }
}

/// Run the full DSv4 model forward by streaming layers one at a time.
///
/// Loads each requested layer from the GGUF, runs the per-layer
/// forward against the running residual, then drops the layer storage
/// before loading the next. The global head (token embedding +
/// output_norm + lm_head + mHC head) is pre-loaded and reused.
///
/// `layer_indices` is the sequence of layers to run (typically `0..43`
/// for a full DSv4-Flash forward, or a prefix for partial forwards).
/// `position_offset` is 0 for prefill from scratch.
///
/// Returns logits of shape `(token_ids.len(), head.n_vocab)`.
pub fn dsv4_streaming_model_forward(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    head: &DsV4HeadStorage,
    token_ids: &[u32],
    layer_indices: &[usize],
    position_offset: usize,
    backend: Option<&dyn ComputeBackend>,
) -> Result<Array2<f32>, DsV4LoadError> {
    let layer_p = build_layer_params(hp);
    let pool = DispatchParamsPool::new(hp);

    let head_w = head.as_weights();
    let head_p = head.params(hp);

    // 1. Embed + broadcast to 4-stream initial residual.
    let x = embed_tokens(token_ids, head.token_embd_view());
    let mut residual: Array3<f32> = broadcast_to_streams(x.view(), hp.n_hc);

    // 2. Stream each layer.
    for &layer_idx in layer_indices {
        let (storage, variant) = load_dsv4_layer(gguf, hp, layer_idx)?;
        let (cp_ref, ip_ref) = pool.pick(&variant);
        let layer_w = DsV4LayerWeights {
            mhc_attn: storage
                .mhc_attn
                .as_ref()
                .expect("mhc_attn loaded by full loader")
                .as_weights(),
            attn: storage.dispatcher_layer(cp_ref, ip_ref),
            mhc_ffn: storage
                .mhc_ffn
                .as_ref()
                .expect("mhc_ffn loaded by full loader")
                .as_weights(),
            ffn: storage
                .ffn
                .as_ref()
                .expect("ffn loaded by full loader")
                .as_weights(),
        };
        residual = dsv4_per_layer_forward(
            residual.view(),
            &layer_w,
            &layer_p,
            Some(token_ids),
            position_offset,
            backend,
        );
        // storage drops here, freeing ~26 GB before the next iteration.
    }

    // 3. mHC head → 1-stream pooled.
    let mut pooled = dsv4_hc_head(residual.view(), &head_w, &head_p, backend);

    // 4. Final RMSNorm.
    let n_tokens = token_ids.len();
    let n_embd = hp.n_embd;
    for t in 0..n_tokens {
        let mut sumsq = 0.0_f32;
        for d in 0..n_embd {
            let v = pooled[[t, d]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n_embd as f32 + head_p.norm_eps).sqrt();
        for d in 0..n_embd {
            pooled[[t, d]] *= inv * head_w.output_norm[d];
        }
    }

    // 5. lm_head projection via ComputeBackend (GPU when available).
    let lm_head_owned = head_w.output_lm_head.to_owned();
    let logits = dot_proj_gpu(&pooled, &lm_head_owned, backend);
    Ok(logits)
}

/// KV-cached variant of [`dsv4_streaming_model_forward`].
///
/// Threads a per-layer `&mut DsV4LayerCache` through the streaming
/// loop. All 3 attention variants (NoCompress / Compress / Indexer)
/// dispatch to their cached counterparts when the cache variant
/// matches the layer variant; mismatched pairs fall back to the
/// non-cached attention path (correctness preserved, no speedup).
///
/// Cache allocation convention (matches DSv4-Flash):
/// - Layer 0, 1: `DsV4LayerCache::new_no_compress(...)`
/// - Odd layers 3..=41 (cr=128): `DsV4LayerCache::new_hca_compress(...)`
/// - Even layers 2..=42 (cr=4 + indexer): `DsV4LayerCache::new_hca_indexer(...)`
///
/// `layer_caches` must have length `>= layer_indices.len()`; indexing
/// is positional (caches[i] matches layer_indices[i]). If shorter,
/// later iterations get `None`. Passing `None` disables caching
/// entirely (behavior identical to `dsv4_streaming_model_forward`).
///
/// `position_offset` is the absolute position of `token_ids[0]` in
/// the running sequence. For a clean prefill it's 0; for a cached
/// decode step after an N-token prefill it's N.
pub fn dsv4_streaming_model_forward_cached(
    gguf: &GgufFile,
    hp: &DsV4Hyperparams,
    head: &DsV4HeadStorage,
    token_ids: &[u32],
    layer_indices: &[usize],
    position_offset: usize,
    mut layer_caches: Option<&mut [DsV4LayerCache]>,
    backend: Option<&dyn ComputeBackend>,
) -> Result<Array2<f32>, DsV4LoadError> {
    let layer_p = build_layer_params(hp);
    let pool = DispatchParamsPool::new(hp);

    let head_w = head.as_weights();
    let head_p = head.params(hp);

    let x = embed_tokens(token_ids, head.token_embd_view());
    let mut residual: Array3<f32> = broadcast_to_streams(x.view(), hp.n_hc);

    for (i, &layer_idx) in layer_indices.iter().enumerate() {
        let (storage, variant) = load_dsv4_layer(gguf, hp, layer_idx)?;
        let (cp_ref, ip_ref) = pool.pick(&variant);
        let layer_w = DsV4LayerWeights {
            mhc_attn: storage
                .mhc_attn
                .as_ref()
                .expect("mhc_attn loaded by full loader")
                .as_weights(),
            attn: storage.dispatcher_layer(cp_ref, ip_ref),
            mhc_ffn: storage
                .mhc_ffn
                .as_ref()
                .expect("mhc_ffn loaded by full loader")
                .as_weights(),
            ffn: storage
                .ffn
                .as_ref()
                .expect("ffn loaded by full loader")
                .as_weights(),
        };
        let cache_for_this_layer: Option<&mut DsV4LayerCache> = match &mut layer_caches {
            Some(caches) => caches.get_mut(i),
            None => None,
        };
        residual = dsv4_per_layer_forward_cached(
            residual.view(),
            &layer_w,
            &layer_p,
            Some(token_ids),
            position_offset,
            cache_for_this_layer,
            backend,
        );
    }

    let mut pooled = dsv4_hc_head(residual.view(), &head_w, &head_p, backend);

    let n_tokens = token_ids.len();
    let n_embd = hp.n_embd;
    for t in 0..n_tokens {
        let mut sumsq = 0.0_f32;
        for d in 0..n_embd {
            let v = pooled[[t, d]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n_embd as f32 + head_p.norm_eps).sqrt();
        for d in 0..n_embd {
            pooled[[t, d]] *= inv * head_w.output_norm[d];
        }
    }

    // 5. lm_head projection via ComputeBackend.
    let lm_head_owned = head_w.output_lm_head.to_owned();
    let logits = dot_proj_gpu(&pooled, &lm_head_owned, backend);
    Ok(logits)
}

/// Resident (non-streaming) DSv4 forward: runs the decode against
/// pre-built per-layer storages held in RAM rather than reloading +
/// re-dequantizing each layer from the GGUF per token.
///
/// `layers` is the full set of `(storage, variant)` pairs for the model
/// (built once — e.g. with resident `QuantTensor` experts via
/// `build_layer_storage_resident`). This is the `dsv4-quant-residency`
/// payoff path: with the quantized working set resident in RAM
/// (~161 GB Q4_K) the per-token streaming reload that dominates
/// [`dsv4_streaming_model_forward_cached`] (~90 s/step on the 2026-05-25
/// bench) is gone — each layer is loaded at most once for the whole
/// decode. The streaming forward stays available for the
/// model-exceeds-RAM case.
///
/// Output and semantics are identical to the streaming forward; the
/// only difference is the layer source (a borrowed slice vs a per-layer
/// GGUF load). `layer_caches`, when supplied, must have one entry per
/// layer in `layers` (same order).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_resident_model_forward_cached(
    layers: &[(DsV4LayerWeightStorage, DsV4LayerVariant)],
    hp: &DsV4Hyperparams,
    head: &DsV4HeadStorage,
    token_ids: &[u32],
    position_offset: usize,
    mut layer_caches: Option<&mut [DsV4LayerCache]>,
    backend: Option<&dyn ComputeBackend>,
) -> Result<Array2<f32>, DsV4LoadError> {
    let layer_p = build_layer_params(hp);
    let pool = DispatchParamsPool::new(hp);

    let head_w = head.as_weights();
    let head_p = head.params(hp);

    let x = embed_tokens(token_ids, head.token_embd_view());
    let mut residual: Array3<f32> = broadcast_to_streams(x.view(), hp.n_hc);

    for (i, (storage, variant)) in layers.iter().enumerate() {
        let (cp_ref, ip_ref) = pool.pick(variant);
        let layer_w = DsV4LayerWeights {
            mhc_attn: storage
                .mhc_attn
                .as_ref()
                .expect("mhc_attn loaded by full loader")
                .as_weights(),
            attn: storage.dispatcher_layer(cp_ref, ip_ref),
            mhc_ffn: storage
                .mhc_ffn
                .as_ref()
                .expect("mhc_ffn loaded by full loader")
                .as_weights(),
            ffn: storage
                .ffn
                .as_ref()
                .expect("ffn loaded by full loader")
                .as_weights(),
        };
        let cache_for_this_layer: Option<&mut DsV4LayerCache> = match &mut layer_caches {
            Some(caches) => caches.get_mut(i),
            None => None,
        };
        residual = dsv4_per_layer_forward_cached(
            residual.view(),
            &layer_w,
            &layer_p,
            Some(token_ids),
            position_offset,
            cache_for_this_layer,
            backend,
        );
    }

    let mut pooled = dsv4_hc_head(residual.view(), &head_w, &head_p, backend);

    let n_tokens = token_ids.len();
    let n_embd = hp.n_embd;
    for t in 0..n_tokens {
        let mut sumsq = 0.0_f32;
        for d in 0..n_embd {
            let v = pooled[[t, d]];
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n_embd as f32 + head_p.norm_eps).sqrt();
        for d in 0..n_embd {
            pooled[[t, d]] *= inv * head_w.output_norm[d];
        }
    }

    let lm_head_owned = head_w.output_lm_head.to_owned();
    let logits = dot_proj_gpu(&pooled, &lm_head_owned, backend);
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::dsv4_head_storage::load_dsv4_head;

    /// Build the dispatch pool, then exercise the pick logic against
    /// each known DSv4-Flash variant tag. No GGUF I/O — uses synthetic
    /// hyperparameters with the indexer fields populated.
    #[test]
    fn dispatch_pool_picks_correct_variant_arm() {
        let hp = DsV4Hyperparams {
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
            indexer_head_size: Some(16),
            n_index_head: Some(2),
            top_k: Some(2),
            n_hc: 4,
            n_expert: 4,
            n_expert_used: 2,
            n_ff_exp: 8,
            n_expert_shared: 1,
            expert_weights_norm: true,
            expert_weights_scale: 1.0,
            yarn: None,
            rope_base_swa: None,
        };
        let pool = DispatchParamsPool::new(&hp);

        // NoCompress: both None.
        let v = DsV4LayerVariant {
            uses_hash_routing: true,
            compress_ratio: None,
            has_indexer: false,
        };
        let (cp, ip) = pool.pick(&v);
        assert!(cp.is_none());
        assert!(ip.is_none());

        // Indexer cr=4: both Some.
        let v = DsV4LayerVariant {
            uses_hash_routing: false,
            compress_ratio: Some(4),
            has_indexer: true,
        };
        let (cp, ip) = pool.pick(&v);
        assert!(cp.is_some());
        assert!(ip.is_some());
        assert_eq!(cp.as_ref().unwrap().compressor.compress_ratio, 4);

        // Compress-only cr=128: cp Some, ip None.
        let v = DsV4LayerVariant {
            uses_hash_routing: false,
            compress_ratio: Some(128),
            has_indexer: false,
        };
        let (cp, ip) = pool.pick(&v);
        assert!(cp.is_some());
        assert!(ip.is_none());
        assert_eq!(cp.as_ref().unwrap().compressor.compress_ratio, 128);
    }

    /// End-to-end real-GGUF: streaming forward over layers 0..3.
    /// Same coverage scope as the per_layer / 2-layer model_forward
    /// smokes, but through the new reusable streaming helper.
    /// Expected wall: ~110 s release (3 × ~30 s loads + forward).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn streaming_helper_3_layer_smoke() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        // 16 tokens (compress_ratio=4 at layer 2 needs n_tokens >= 4).
        let n_tokens = 16;
        let token_ids: Vec<u32> = (0..n_tokens)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();

        let logits =
            dsv4_streaming_model_forward(&gguf, &hp, &head, &token_ids, &[0, 1, 2], 0, None)
                .expect("streaming forward");

        assert_eq!(logits.shape(), &[n_tokens, head.n_vocab]);
        let n_nonfinite = logits.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(n_nonfinite, 0, "{n_nonfinite} non-finite logits");
        let total: f32 = logits.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "logits all zero");
    }

    /// **The full 43-layer DSv4-Flash streaming forward.**
    ///
    /// Runs every layer of the real ~172 GB DSv4-Flash GGUF through
    /// the streaming helper, producing final (n_tokens, 129280) logits.
    /// Each layer's storage (~26 GB f32 of FFN expert tensors) is
    /// loaded, executed, and dropped before the next layer loads —
    /// peak RAM stays bounded at one layer + head + residual.
    ///
    /// Expected wall: **~22-25 min release mode**. Breakdown:
    /// - Head load: ~6 s
    /// - 43 × ~30 s layer load = ~22 min
    /// - Forward + lm_head matmul: ~1-2 min
    ///
    /// Need 128 tokens because the cr=128 layers require
    /// `n_tokens >= compress_ratio`. The 128 × 129280 lm_head matmul
    /// is ~67 GFLOPs — completes in ~1 min release.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk and ~25 min wall"]
    fn streaming_full_43_layer_forward() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        // 128 tokens (covers cr=128 layers' compressor n_tokens requirement).
        let n_tokens = 128;
        let token_ids: Vec<u32> = (0..n_tokens)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();
        let n_layer = 43;
        let layer_indices: Vec<usize> = (0..n_layer).collect();

        let logits =
            dsv4_streaming_model_forward(&gguf, &hp, &head, &token_ids, &layer_indices, 0, None)
                .expect("streaming forward over all 43 layers");

        assert_eq!(logits.shape(), &[n_tokens, head.n_vocab]);
        let n_nonfinite = logits.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            n_nonfinite, 0,
            "{n_nonfinite} non-finite values in full-43 logits"
        );
        let total: f32 = logits.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "full-43 logits all zero");

        // Per-token distinctness: different positions should produce
        // distinct logit rows after 43 layers of processing.
        let mut diff_01: f32 = 0.0;
        let mut diff_0last: f32 = 0.0;
        for v in 0..head.n_vocab {
            diff_01 += (logits[[0, v]] - logits[[1, v]]).abs();
            diff_0last += (logits[[0, v]] - logits[[n_tokens - 1, v]]).abs();
        }
        assert!(
            diff_01 > 0.0,
            "row 0 == row 1 (43-layer forward degenerate)"
        );
        assert!(
            diff_0last > 0.0,
            "row 0 == row {} (43-layer forward degenerate)",
            n_tokens - 1
        );
    }

    /// Real-GGUF: the cached streaming forward with `layer_caches=None`
    /// produces identical logits to the existing non-cached function.
    /// Same 3-layer scope as `streaming_helper_3_layer_smoke`.
    ///
    /// Expected wall: ~110 s release (3 × ~30 s loads + forward).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn streaming_cached_with_no_cache_equals_uncached() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        let n_tokens = 16;
        let token_ids: Vec<u32> = (0..n_tokens)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();
        let layers = [0, 1, 2];

        let logits_a =
            dsv4_streaming_model_forward(&gguf, &hp, &head, &token_ids, &layers, 0, None).unwrap();
        let logits_b = dsv4_streaming_model_forward_cached(
            &gguf, &hp, &head, &token_ids, &layers, 0, None, None,
        )
        .unwrap();
        assert_eq!(logits_a.shape(), logits_b.shape());

        let mut max_diff = 0.0_f32;
        for (a, b) in logits_a.iter().zip(logits_b.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        // Streaming + lm_head matmul gives wider reduction-order
        // sensitivity than per-op tests; a logit-magnitude-relative
        // threshold is more meaningful than an absolute one.
        let max_abs: f32 = logits_a.iter().fold(0.0, |m, &v| m.max(v.abs()));
        let rel = max_diff / max_abs.max(1.0);
        assert!(
            rel < 1e-4,
            "cached-vs-uncached relative diff {rel} (max_abs={max_abs}, max_diff={max_diff})"
        );
    }

    /// **Real-GGUF cached prefill + decode smoke**.
    ///
    /// Allocates per-variant caches for layers 0..3 of DSv4-Flash:
    /// - Layer 0, 1: `DsV4LayerCache::new_no_compress(...)`
    /// - Layer 2: `DsV4LayerCache::new_hca_indexer(...)` (cr=4 indexer)
    ///
    /// Runs the streaming_cached forward twice:
    /// 1. **Prefill** (16 tokens, position_offset=0) → logits + caches
    ///    populated.
    /// 2. **Decode** (1 token appended, position_offset=16) → new
    ///    logits row for the appended token.
    ///
    /// Verifies the entire cached arc (all 3 cached attention variants,
    /// per-layer dispatch via `DsV4LayerCache`, position_offset
    /// advancement) works end-to-end on real DSv4-Flash weights.
    ///
    /// Expected wall: ~220 s release (3-layer 16-tok cached prefill
    /// ~110 s + 3-layer 1-tok cached decode ~110 s, since the decode
    /// step still has to re-stream all 3 layers from the 172 GB GGUF).
    /// The KV-cache speedup is per-layer attention compute, not per-
    /// layer load — proving the cached pipeline works correctness-wise
    /// is the primary goal here.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn streaming_cached_real_gguf_prefill_then_decode() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        // Pre-allocate caches matching the layer variants:
        //   layer 0: NoCompress
        //   layer 1: NoCompress
        //   layer 2: Indexer (cr=4)
        let max_seq_len = 64;
        let mut layer_caches: Vec<DsV4LayerCache> = vec![
            DsV4LayerCache::new_no_compress(max_seq_len, hp.head_dim),
            DsV4LayerCache::new_no_compress(max_seq_len, hp.head_dim),
            DsV4LayerCache::new_hca_indexer(
                max_seq_len,
                hp.head_dim,
                4,
                hp.indexer_head_size.expect("indexer_head_size"),
            ),
        ];

        // 16-token prefill prompt.
        let n_prefill = 16;
        let prompt: Vec<u32> = (0..n_prefill)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();
        let layers = [0, 1, 2];

        let logits_prefill = dsv4_streaming_model_forward_cached(
            &gguf,
            &hp,
            &head,
            &prompt,
            &layers,
            0,
            Some(&mut layer_caches),
            None,
        )
        .expect("cached prefill");
        assert_eq!(logits_prefill.shape(), &[n_prefill, head.n_vocab]);
        let n_nan = logits_prefill.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(n_nan, 0, "prefill logits have {n_nan} non-finite values");

        // After prefill, each cache should hold n_prefill raw rows.
        for (i, c) in layer_caches.iter().enumerate() {
            assert_eq!(
                c.position(),
                n_prefill,
                "layer {i} cache position after prefill"
            );
        }

        // ── Decode step: feed one new token at position_offset=16. ──
        let next_tok: u32 = ((n_prefill * 257) % head.n_vocab) as u32;
        let decode_input = [next_tok];
        let logits_decode = dsv4_streaming_model_forward_cached(
            &gguf,
            &hp,
            &head,
            &decode_input,
            &layers,
            n_prefill, // position_offset
            Some(&mut layer_caches),
            None,
        )
        .expect("cached decode");
        assert_eq!(logits_decode.shape(), &[1, head.n_vocab]);
        let n_nan = logits_decode.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(n_nan, 0, "decode logits have {n_nan} non-finite values");
        let total: f32 = logits_decode.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "decode logits all zero");

        // Each cache should now hold n_prefill + 1 raw rows.
        for (i, c) in layer_caches.iter().enumerate() {
            assert_eq!(
                c.position(),
                n_prefill + 1,
                "layer {i} cache position after decode"
            );
        }
    }

    /// dsv4-quant-residency P3: the resident forward
    /// ([`dsv4_resident_model_forward_cached`]) produces **identical**
    /// logits to the streaming forward on the same layers. Both build
    /// f32 layers here (via `load_dsv4_layers` / per-call
    /// `load_dsv4_layer`), so the only difference is the layer source —
    /// the residual math, dispatch, and head are the same code path, so
    /// the result is bit-identical. This pins the resident loop's
    /// correctness; the quant-vs-f32 numeric tolerance is covered
    /// separately by the MoE-dispatch parity test.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn resident_forward_matches_streaming_forward() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        let n_tokens = 16;
        let token_ids: Vec<u32> = (0..n_tokens)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();

        // Resident: build all 3 layers once, forward against them.
        let layers = crate::attention::dsv4_full_loader::load_dsv4_layers(&gguf, &hp, 3)
            .expect("load resident layers");
        let resident =
            dsv4_resident_model_forward_cached(&layers, &hp, &head, &token_ids, 0, None, None)
                .expect("resident forward");

        // Streaming: reload each layer per call.
        let streaming = dsv4_streaming_model_forward_cached(
            &gguf,
            &hp,
            &head,
            &token_ids,
            &[0, 1, 2],
            0,
            None,
            None,
        )
        .expect("streaming forward");

        assert_eq!(resident.shape(), streaming.shape());
        let max_diff = resident
            .iter()
            .zip(streaming.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert_eq!(
            max_diff, 0.0,
            "resident vs streaming logits differ (max_diff={max_diff}) — same f32 layers \
             should be bit-identical"
        );
    }

    /// dsv4-quant-residency parity: the **quant-resident** forward
    /// (Q4_K experts via `expert_slice` + lazy-quant matmul) agrees with
    /// the **f32-streaming** forward (Q4_K dequant → f32 matmul) within
    /// tolerance, and predicts the **same greedy next token**. The two
    /// differ only by the Q8_K activation quantization in the lazy-quant
    /// kernel (the weights decode from the same Q4_K blocks), so logits
    /// are close but not bit-identical. Backs spec scenarios "Logits
    /// within tolerance" + "Greedy tokens match".
    ///
    /// The relative bound below is generous and **calibration-pending**
    /// on the first real-GGUF run; tighten it once observed.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn resident_quant_forward_within_tolerance_of_streaming_f32() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let head = load_dsv4_head(&gguf, &hp).expect("head");

        let n_tokens = 16;
        let token_ids: Vec<u32> = (0..n_tokens)
            .map(|t| (t * 257 % head.n_vocab) as u32)
            .collect();

        // Quant-resident forward (Q4_K experts).
        let resident_layers =
            crate::attention::dsv4_full_loader::load_dsv4_resident_layers(&gguf, &hp, 3)
                .expect("resident-quant layers");
        let quant = dsv4_resident_model_forward_cached(
            &resident_layers,
            &hp,
            &head,
            &token_ids,
            0,
            None,
            None,
        )
        .expect("quant resident forward");

        // f32-streaming forward (Q4_K → f32 dequant).
        let f32 = dsv4_streaming_model_forward_cached(
            &gguf,
            &hp,
            &head,
            &token_ids,
            &[0, 1, 2],
            0,
            None,
            None,
        )
        .expect("f32 streaming forward");

        assert_eq!(quant.shape(), f32.shape());

        // Relative logit agreement, scaled by the f32 logit magnitude.
        let mut max_rel = 0.0_f32;
        for (q, r) in quant.iter().zip(f32.iter()) {
            let denom = r.abs().max(1.0);
            max_rel = max_rel.max((q - r).abs() / denom);
        }
        eprintln!("quant vs f32 max relative logit diff: {max_rel:.4}");

        let argmax_row = |logits: &Array2<f32>, t: usize| -> usize {
            let row = logits.row(t);
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (j, &v) in row.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = j;
                }
            }
            best
        };
        let matches = (0..n_tokens)
            .filter(|&t| argmax_row(&quant, t) == argmax_row(&f32, t))
            .count();
        eprintln!("greedy argmax match: {matches}/{n_tokens} positions");

        // **Load-bearing assertion**: the greedy next-token (argmax of the
        // final position) is identical between the two paths. This is the
        // generation-relevant signal and the real correctness gate.
        let last = n_tokens - 1;
        assert_eq!(
            argmax_row(&quant, last),
            argmax_row(&f32, last),
            "greedy next-token differs between quant-resident and f32 forward"
        );

        // Loose logit-value guard (catastrophic-divergence only). Since P5
        // (resident-Q4_K *attention* weights) the quant path runs the Q4_K
        // ×Q8_K lazy matmul for the attention projections too, not just the
        // routed experts. The f32-streaming path dequantizes the same Q4_K
        // blocks to f32; the two then differ by the **Q8_K activation
        // quantization**, which RoPE + attention softmax amplify (and this
        // test stresses it with random token ids). Observed ~0.77 on the
        // 3-layer real-GGUF run with argmax stable (15-16/16 positions),
        // so the value bound is generous — argmax (above) is the contract;
        // this only catches NaN / sign-flip / order-of-magnitude blowups.
        assert!(
            max_rel < 1.5,
            "quant-resident logits diverge from f32 by {max_rel:.4} (>1.5) — \
             catastrophic; expected ~0.8 from Q8_K activation quant on the \
             fully-quant (P5) attention+expert path"
        );
    }
}
