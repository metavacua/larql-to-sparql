//! End-to-end smoke test: real-GGUF layer 0 attention forward.
//!
//! Loads a single layer's worth of real DSv4-Flash weights via the
//! full GGUF→storage pipeline (Stages 8h-4b-1..4a), gets the
//! borrowed dispatcher arm via Stage 8h-2b, and runs Stage 8h-1's
//! attention dispatcher on a synthetic 1-token input.
//!
//! Purpose: this is the first real test of the load→compute chain on
//! actual model weights. Synthetic-tensor tests verify each piece in
//! isolation; this is the first one that fires the full chain on real
//! data. Anything that goes wrong here (shape mismatches, NaN propagation,
//! wrong RoPE base, missing weights) surfaces here rather than mid-way
//! through a 22-minute full-model load.
//!
//! Layer 0 is the cheapest: hash-routed FFN (which we currently skip
//! since the int routing table isn't loaded yet), no compressor, no
//! indexer. Just the standard attention path (Stage 8a).

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use larql_models::loading::gguf::GgufFile;

    use crate::attention::dsv4_attn_block::DsV4AttnBlockParams;
    use crate::attention::dsv4_attn_block_compress::DsV4AttnBlockCompressParams;
    use crate::attention::dsv4_attn_block_indexer::DsV4AttnBlockIndexerParams;
    use crate::attention::dsv4_attn_dispatch::dsv4_attn_layer;
    use crate::attention::dsv4_compressor_prefill::CompressorParams;
    use crate::attention::dsv4_full_loader::load_dsv4_layer;
    use crate::attention::dsv4_hyperparams_load::DsV4MetadataError;
    use crate::attention::dsv4_indexer::IndexerParams;
    use crate::attention::dsv4_rope_tail::DsV4RopeMode;
    use crate::attention::dsv4_storage_build::DsV4Hyperparams;

    /// End-to-end: load layer 0 + run a 1-token forward through the
    /// attention dispatcher. Expect finite output of shape (1, n_embd).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn smoke_layer_0_attention_forward() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp: Result<DsV4Hyperparams, DsV4MetadataError> = DsV4Hyperparams::from_gguf(&gguf);
        let hp = hp.expect("hyperparams");
        let (storage, variant) = load_dsv4_layer(&gguf, &hp, 0).expect("load layer 0");

        // Layer 0 is NoCompress in the real DSv4-Flash model.
        assert_eq!(variant.compress_ratio, None);
        assert!(!variant.has_indexer);

        // Build a single-token input — synthetic but with the right
        // n_embd (4096). Use a small magnitude so SwiGLU/softmax don't
        // saturate immediately.
        let n_tokens = 1;
        let n_embd = hp.n_embd;
        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(_, d)| {
            ((d as f32 * 0.0013).sin()) * 0.1
        });

        // Build the no-compress dispatcher arm directly (no compress or
        // indexer params needed since variant is NoCompress).
        let layer = storage.dispatcher_layer(&None, &None);
        assert_eq!(layer.variant_name(), "no_compress");

        let out = dsv4_attn_layer(x.view(), &layer, 0, None);
        assert_eq!(out.shape(), &[n_tokens, n_embd]);
        // Finite-ness is the headline correctness check — if anything in
        // RoPE/SWA/grouped-o-proj goes off the rails we get NaN/Inf.
        let n_nonfinite = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            n_nonfinite, 0,
            "{n_nonfinite} non-finite values in layer-0 attention output"
        );
        // Non-trivial: at least one nonzero output.
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "layer-0 attention output is all zeros");
    }

    /// Layer 4 (Indexer + compress_ratio=4): heaviest attention path.
    /// If this passes, all three attention variants (NoCompress,
    /// Compress, Indexer) execute correctly on real DSv4-Flash data.
    ///
    /// Need 16+ tokens because compress_ratio=4 requires n_comp ≥ 1
    /// (build_compressor_prefill asserts `n_tokens >= compress_ratio`).
    /// Use 16 tokens → n_comp=4 compressed positions per the math.
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn smoke_layer_4_indexer_attention_forward() {
        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp: Result<DsV4Hyperparams, DsV4MetadataError> = DsV4Hyperparams::from_gguf(&gguf);
        let hp = hp.expect("hyperparams");
        let (storage, variant) = load_dsv4_layer(&gguf, &hp, 4).expect("load layer 4");

        // Layer 4 is Indexer + compress_ratio=4 in the real model.
        assert_eq!(variant.compress_ratio, Some(4));
        assert!(variant.has_indexer);

        // Build the Indexer dispatcher arm: needs both compress_params
        // and indexer_params constructed manually since the storage
        // holds the underlying weights but the params live alongside.
        let compress_params = Some(DsV4AttnBlockCompressParams {
            attn: DsV4AttnBlockParams {
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
            },
            compressor: CompressorParams {
                head_dim: hp.head_dim,
                n_embd: hp.n_embd,
                compress_ratio: 4,
                n_rot: hp.n_rot,
                rope_base: hp.rope_base,
                rope_mode: DsV4RopeMode::Neox,
                norm_eps: hp.norm_eps,
            },
        });
        let indexer_params = Some(DsV4AttnBlockIndexerParams {
            attn: compress_params.as_ref().unwrap().attn,
            compressor: compress_params.as_ref().unwrap().compressor,
            indexer_compressor: CompressorParams {
                head_dim: hp.indexer_head_size.expect("indexer_head_size"),
                n_embd: hp.n_embd,
                compress_ratio: 4,
                n_rot: hp.n_rot,
                rope_base: hp.rope_base,
                rope_mode: DsV4RopeMode::Neox,
                norm_eps: hp.norm_eps,
            },
            indexer: IndexerParams {
                n_embd: hp.n_embd,
                q_lora_rank: hp.q_lora_rank,
                n_index_head: hp.n_index_head.expect("n_index_head"),
                n_index_head_size: hp.indexer_head_size.expect("indexer_head_size"),
                n_rot: hp.n_rot,
                rope_base: hp.rope_base,
                rope_mode: DsV4RopeMode::Neox,
            },
            top_k: hp.top_k.expect("top_k"),
        });

        let layer = storage.dispatcher_layer(&compress_params, &indexer_params);
        assert_eq!(layer.variant_name(), "indexer");
        assert_eq!(layer.compress_ratio(), 4);

        // 16 tokens → n_comp=4 compressed positions (compress_ratio=4).
        let n_tokens = 16;
        let n_embd = hp.n_embd;
        let x = Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 17 + d) as f32 * 0.0013).sin() * 0.1
        });

        let out = dsv4_attn_layer(x.view(), &layer, 0, None);
        assert_eq!(out.shape(), &[n_tokens, n_embd]);
        let n_nonfinite = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            n_nonfinite, 0,
            "{n_nonfinite} non-finite values in layer-4 Indexer attention output"
        );
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "layer-4 Indexer attention output is all zeros");
    }

    /// mHC pre + attention + mHC post end-to-end on real layer 4.
    /// Validates Stage 8h-3d-1's bookend pair sandwiches the heaviest
    /// attention path (Indexer, compress_ratio=4) correctly when the
    /// mHC weights are loaded from the real GGUF.
    ///
    /// Pipeline:
    ///   residual_4stream (4, n_hc=4, n_embd=4096)
    ///   → mhc_pre  → cur (4, 4096), post, comb
    ///   → attn     → out (4, 4096)
    ///   → mhc_post → new_residual (4, 4, 4096)
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn smoke_layer_4_mhc_bookend_around_attention() {
        use ndarray::Array3;

        use crate::attention::dsv4_mhc_bookend::{dsv4_mhc_post, dsv4_mhc_pre, MhcParams};

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp: Result<DsV4Hyperparams, DsV4MetadataError> = DsV4Hyperparams::from_gguf(&gguf);
        let hp = hp.expect("hyperparams");
        let (storage, _variant) = load_dsv4_layer(&gguf, &hp, 4).expect("load layer 4");

        // Build attention dispatcher arm (Indexer variant).
        let compress_params = Some(DsV4AttnBlockCompressParams {
            attn: DsV4AttnBlockParams {
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
            },
            compressor: CompressorParams {
                head_dim: hp.head_dim,
                n_embd: hp.n_embd,
                compress_ratio: 4,
                n_rot: hp.n_rot,
                rope_base: hp.rope_base,
                rope_mode: DsV4RopeMode::Neox,
                norm_eps: hp.norm_eps,
            },
        });
        let indexer_params = Some(DsV4AttnBlockIndexerParams {
            attn: compress_params.as_ref().unwrap().attn,
            compressor: compress_params.as_ref().unwrap().compressor,
            indexer_compressor: CompressorParams {
                head_dim: hp.indexer_head_size.unwrap(),
                n_embd: hp.n_embd,
                compress_ratio: 4,
                n_rot: hp.n_rot,
                rope_base: hp.rope_base,
                rope_mode: DsV4RopeMode::Neox,
                norm_eps: hp.norm_eps,
            },
            indexer: IndexerParams {
                n_embd: hp.n_embd,
                q_lora_rank: hp.q_lora_rank,
                n_index_head: hp.n_index_head.unwrap(),
                n_index_head_size: hp.indexer_head_size.unwrap(),
                n_rot: hp.n_rot,
                rope_base: hp.rope_base,
                rope_mode: DsV4RopeMode::Neox,
            },
            top_k: hp.top_k.unwrap(),
        });
        let layer = storage.dispatcher_layer(&compress_params, &indexer_params);

        // 4-token input (compress_ratio=4 requires n_tokens >= 4).
        let n_tokens = 4;
        let n_embd = hp.n_embd;
        let n_hc = hp.n_hc;

        // Build 4-stream residual by broadcasting a 1-stream embedding.
        let one_stream = ndarray::Array2::<f32>::from_shape_fn((n_tokens, n_embd), |(t, d)| {
            ((t * 13 + d) as f32 * 0.0013).sin() * 0.1
        });
        let mut residual = Array3::<f32>::zeros((n_tokens, n_hc, n_embd));
        for t in 0..n_tokens {
            for h in 0..n_hc {
                for d in 0..n_embd {
                    residual[[t, h, d]] = one_stream[[t, d]];
                }
            }
        }

        // mHC params: sinkhorn_iters / hc_eps not in metadata; use the
        // canonical DSv4 defaults (2 iters, 1e-5 eps).
        let mhc_p = MhcParams {
            n_embd: hp.n_embd,
            n_hc: hp.n_hc,
            sinkhorn_iters: 2,
            hc_eps: 1e-5,
            norm_eps: hp.norm_eps,
        };
        let mhc_attn = storage
            .mhc_attn
            .as_ref()
            .expect("mhc_attn loaded for layer 4")
            .as_weights();

        // 1. mHC pre — collapse 4-stream → 1-stream cur.
        let pre = dsv4_mhc_pre(residual.view(), &mhc_attn, &mhc_p, None);
        assert_eq!(pre.cur.shape(), &[n_tokens, n_embd]);
        assert!(pre.cur.iter().all(|v| v.is_finite()), "pre.cur not finite");

        // 2. Attention block — Indexer path on the collapsed input.
        let attn_out =
            crate::attention::dsv4_attn_dispatch::dsv4_attn_layer(pre.cur.view(), &layer, 0, None);
        assert_eq!(attn_out.shape(), &[n_tokens, n_embd]);
        assert!(
            attn_out.iter().all(|v| v.is_finite()),
            "attn_out not finite"
        );

        // 3. mHC post — expand back to 4-stream residual.
        let new_residual = dsv4_mhc_post(
            attn_out.view(),
            residual.view(),
            pre.post.view(),
            pre.comb.view(),
        );
        assert_eq!(new_residual.shape(), &[n_tokens, n_hc, n_embd]);
        let n_nonfinite = new_residual.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            n_nonfinite, 0,
            "{n_nonfinite} non-finite values in new_residual"
        );
        let total: f32 = new_residual.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "new_residual is all zeros");
    }

    /// Full per-layer forward on real layer 0: mHC pre → attention
    /// (NoCompress variant) → mHC post → mHC pre → FFN (hash routing,
    /// since layer 0 ∈ [0, n_hash_layers=3)) → mHC post.
    ///
    /// This is the most ambitious DSv4 smoke yet: every stage from
    /// 8a through 8h-3d-2 runs on real GGUF weights, with the actual
    /// 256-expert MoE FFN dispatching through ~26 GB of f32 expert
    /// tensors. Expect ~30 s release-mode runtime (dominated by Q6_K
    /// dequant of the 3 expert tensors during load).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn smoke_layer_0_full_per_layer_forward() {
        use ndarray::Array3;

        use crate::attention::dsv4_ffn_block::Dsv4FfnParams;
        use crate::attention::dsv4_mhc_bookend::MhcParams;
        use crate::attention::dsv4_per_layer::{
            dsv4_per_layer_forward, DsV4LayerParams, DsV4LayerWeights,
        };

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let (storage, variant) = load_dsv4_layer(&gguf, &hp, 0).expect("load layer 0");
        assert!(variant.uses_hash_routing, "layer 0 should hash-route");

        // 4-token 4-stream residual (n_hc=4 for DSv4-Flash).
        let n_tokens = 4;
        let n_embd = hp.n_embd;
        let n_hc = hp.n_hc;
        let residual = Array3::<f32>::from_shape_fn((n_tokens, n_hc, n_embd), |(t, h, d)| {
            ((t * 13 + h * 7 + d) as f32 * 0.0013).sin() * 0.1
        });

        // Token IDs must be < n_vocab. Derive n_vocab from gate_tid2eid.
        let ffn = storage.ffn.as_ref().expect("ffn loaded");
        let n_vocab = ffn
            .gate_tid2eid
            .as_ref()
            .expect("hash routing table on layer 0")
            .shape()[0];
        let token_ids: Vec<u32> = (0..n_tokens).map(|t| (t * 257 % n_vocab) as u32).collect();

        // Wire layer storage views into per-layer weights.
        let attn_layer = storage.dispatcher_layer(&None, &None); // NoCompress
        let mhc_attn = storage
            .mhc_attn
            .as_ref()
            .expect("mhc_attn loaded")
            .as_weights();
        let mhc_ffn = storage
            .mhc_ffn
            .as_ref()
            .expect("mhc_ffn loaded")
            .as_weights();
        let ffn_weights = ffn.as_weights();
        let layer_w = DsV4LayerWeights {
            mhc_attn,
            attn: attn_layer,
            mhc_ffn,
            ffn: ffn_weights,
        };
        let layer_p = DsV4LayerParams {
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
        };

        let out = dsv4_per_layer_forward(
            residual.view(),
            &layer_w,
            &layer_p,
            Some(&token_ids),
            0,
            None,
        );
        assert_eq!(out.shape(), &[n_tokens, n_hc, n_embd]);
        let n_nonfinite = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            n_nonfinite, 0,
            "{n_nonfinite} non-finite values in full per-layer output"
        );
        let total: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(total > 0.0, "per-layer output is all zeros");
    }

    /// 2-layer chained forward on real layers 0 and 1 (both NoCompress
    /// variant, both hash-routed since 0,1 ∈ [0, n_hash_layers=3)).
    ///
    /// Validates layer-to-layer residual handoff on real GGUF weights:
    /// the residual emitted by layer 0 is fed directly into layer 1's
    /// `dsv4_per_layer_forward`. Cross-checks that layer 1's output
    /// differs from layer 0's (proves both layers actually transform
    /// the residual and the chain isn't degenerate).
    ///
    /// Pipeline:
    ///   residual_0 (4, n_hc=4, n_embd=4096)
    ///     → dsv4_per_layer_forward(layer_0) → residual_1
    ///     → dsv4_per_layer_forward(layer_1) → residual_2
    ///
    /// Expected wall: ~60 s release mode (2 × ~30 s load).
    #[test]
    #[ignore = "Requires the real ~172 GB DSv4-Flash GGUF on disk"]
    fn smoke_layers_0_to_1_chained_forward() {
        use ndarray::Array3;

        use crate::attention::dsv4_ffn_block::Dsv4FfnParams;
        use crate::attention::dsv4_mhc_bookend::MhcParams;
        use crate::attention::dsv4_per_layer::{
            dsv4_per_layer_forward, DsV4LayerParams, DsV4LayerWeights,
        };

        let path = std::path::Path::new(
            "/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present");
            return;
        }
        let gguf = GgufFile::open(path).expect("open DSv4 GGUF");
        let hp = DsV4Hyperparams::from_gguf(&gguf).expect("hyperparams");
        let (storage_0, variant_0) = load_dsv4_layer(&gguf, &hp, 0).expect("load layer 0");
        let (storage_1, variant_1) = load_dsv4_layer(&gguf, &hp, 1).expect("load layer 1");
        assert!(variant_0.uses_hash_routing, "layer 0 hash-routes");
        assert!(variant_1.uses_hash_routing, "layer 1 hash-routes");

        let n_tokens = 4;
        let n_embd = hp.n_embd;
        let n_hc = hp.n_hc;
        let residual_0 = Array3::<f32>::from_shape_fn((n_tokens, n_hc, n_embd), |(t, h, d)| {
            ((t * 13 + h * 7 + d) as f32 * 0.0013).sin() * 0.1
        });

        // Token IDs must be valid for layer 0's and layer 1's tid2eid.
        let n_vocab = storage_0
            .ffn
            .as_ref()
            .and_then(|f| f.gate_tid2eid.as_ref())
            .expect("layer 0 hash table")
            .shape()[0];
        let token_ids: Vec<u32> = (0..n_tokens).map(|t| (t * 257 % n_vocab) as u32).collect();

        let layer_p = DsV4LayerParams {
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
        };

        // Run layer 0.
        let layer_w_0 = DsV4LayerWeights {
            mhc_attn: storage_0.mhc_attn.as_ref().unwrap().as_weights(),
            attn: storage_0.dispatcher_layer(&None, &None),
            mhc_ffn: storage_0.mhc_ffn.as_ref().unwrap().as_weights(),
            ffn: storage_0.ffn.as_ref().unwrap().as_weights(),
        };
        let residual_1 = dsv4_per_layer_forward(
            residual_0.view(),
            &layer_w_0,
            &layer_p,
            Some(&token_ids),
            0,
            None,
        );
        assert_eq!(residual_1.shape(), &[n_tokens, n_hc, n_embd]);
        assert!(
            residual_1.iter().all(|v| v.is_finite()),
            "non-finite after layer 0"
        );

        // Run layer 1 fed by layer 0's output.
        let layer_w_1 = DsV4LayerWeights {
            mhc_attn: storage_1.mhc_attn.as_ref().unwrap().as_weights(),
            attn: storage_1.dispatcher_layer(&None, &None),
            mhc_ffn: storage_1.mhc_ffn.as_ref().unwrap().as_weights(),
            ffn: storage_1.ffn.as_ref().unwrap().as_weights(),
        };
        let residual_2 = dsv4_per_layer_forward(
            residual_1.view(),
            &layer_w_1,
            &layer_p,
            Some(&token_ids),
            0,
            None,
        );
        assert_eq!(residual_2.shape(), &[n_tokens, n_hc, n_embd]);
        let n_nonfinite = residual_2.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(n_nonfinite, 0, "{n_nonfinite} non-finite after layer 1");

        // Cross-check: residual_2 differs from residual_1 (layer 1
        // actually transformed something).
        let mut diff: f32 = 0.0;
        for (a, b) in residual_1.iter().zip(residual_2.iter()) {
            diff += (a - b).abs();
        }
        assert!(diff > 0.0, "residual_2 == residual_1 (layer 1 was a no-op)");

        // And residual_1 differs from residual_0 too (layer 0 also worked).
        let mut diff_0: f32 = 0.0;
        for (a, b) in residual_0.iter().zip(residual_1.iter()) {
            diff_0 += (a - b).abs();
        }
        assert!(
            diff_0 > 0.0,
            "residual_1 == residual_0 (layer 0 was a no-op)"
        );
    }
}
