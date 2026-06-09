use std::collections::HashMap;

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::attention::{KvCache, SharedKV};
use crate::forward::embed_tokens_pub;
use crate::forward::layer::run_layer_with_ffn_with_cache;
use crate::forward::ple::precompute_per_layer_inputs;
use crate::forward::run_layer_with_ffn;

use super::tensors::insert_q4k_layer_tensors;

/// Like [`predict_q4k_hidden`] but takes pre-embedded inputs and
/// returns the per-layer K/V cache alongside the final hidden state.
/// Used by the attention-service-routes prefill handler so it can
/// stash K/V into the session's `KvCache` without re-tokenising.
///
/// This variant SKIPS Per-Layer-Embeddings (Gemma 4 E2B). If the
/// model declares `has_per_layer_embeddings()`, the caller MUST
/// route through the token_ids-aware
/// [`predict_q4k_hidden`] path instead — otherwise the residual
/// stream loses the per-layer per-position contribution and the
/// output is gibberish.
///
/// Returns `(final_h, kv)` where:
/// - `final_h` is `[seq_len, hidden]` post-last-layer.
/// - `kv[layer]` is the layer's `(k_post_rope, v)` pair when the
///   layer participated, or `None` for skipped/KV-shared layers.
pub fn prefill_q4k_from_embeddings(
    weights: &mut ModelWeights,
    h0: Array2<f32>,
    index: &VectorIndex,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
) -> (Array2<f32>, Vec<Option<SharedKV>>) {
    let num_layers = weights.num_layers;
    let mut h = h0;
    let mut kvs: Vec<Option<SharedKV>> = vec![None; num_layers];
    let mut shared_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..num_layers {
        let inserted =
            insert_q4k_layer_tensors(weights, index, layer).unwrap_or_else(|err| panic!("{err}"));

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| shared_cache.get(&src));
        let is_moe_layer = weights.arch.is_hybrid_moe();
        let ffn_backend = crate::ffn::WeightFfn { weights };

        if is_moe_layer {
            if let Some((h_new, kv_out)) = run_moe_layer_cpu(
                weights,
                &h,
                layer,
                &ffn_backend,
                None, // PLE skipped for embedding-input variant
                shared_kv,
                moe_remote,
            ) {
                h = h_new;
                if let Some(kv) = kv_out.clone() {
                    shared_cache.insert(layer, kv.clone());
                    kvs[layer] = Some(kv);
                }
            }
        } else if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn_backend,
            false,
            None, // PLE skipped
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out.clone() {
                shared_cache.insert(layer, kv.clone());
                kvs[layer] = Some(kv);
            }
        }

        // Keep `inserted` resident in `weights.tensors` across calls:
        // `insert_q4k_layer_tensors` is idempotent, so the next call's
        // dequant is skipped. ~10 GB resident for Gemma 3 4B, ~3 GB for 1B.
        let _ = inserted;
    }

    (h, kvs)
}

/// Whether the persistent CPU Q4K KV cache (`KvCache` threaded through
/// [`predict_q4k_hidden_with_cache`]) supports the architecture of `weights`.
///
/// Returns `false` for:
/// - **Hybrid MoE** (`arch.is_hybrid_moe()` — Gemma 4 26B A4B): MoE layers
///   don't thread the cache.
/// - **Cross-layer K/V sharing** (`arch.kv_shared_source_layer(l).is_some()`
///   for any layer — Gemma 4 family): sharing layers need the donor's full
///   cache rather than just this-step's K/V.
///
/// Callers (e.g., `generate_via_cpu_q4k`) use this to decide whether to
/// allocate a `KvCache` and route through the decode-step path, or fall
/// back to the per-token full-replay forward.
pub fn cpu_q4k_cache_supported(weights: &ModelWeights) -> bool {
    let arch = &*weights.arch;
    if arch.is_hybrid_moe() {
        return false;
    }
    !(0..weights.num_layers).any(|l| arch.kv_shared_source_layer(l).is_some())
}

/// Compute the final hidden state for `token_ids` against a Q4_K/Q6_K
/// vindex, dequantising attn + FFN one layer at a time. Returns the
/// `[seq_len, hidden]` array; caller owns the lm_head step.
pub fn predict_q4k_hidden(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
) -> Array2<f32> {
    predict_q4k_hidden_with_cache(weights, token_ids, index, moe_remote, None)
}

/// Variant of [`predict_q4k_hidden`] that accepts an optional mutable
/// [`KvCache`] reference. When provided, the cache is read from and written to
/// per-layer by the CPU attention forward: an empty cache triggers a prefill
/// that snapshots K/V into the cache; a populated cache combined with a
/// single-row input triggers a decode-step that appends one position's K/V.
///
/// Two architectures bypass the cache (silently — correctness preserved by
/// running the uncached forward) because they need follow-up work to integrate
/// with a persistent per-layer cache:
///
/// - **Hybrid MoE** (`arch.is_hybrid_moe()` — Gemma 4 26B A4B): the MoE layer
///   path doesn't thread the cache parameter yet.
/// - **Cross-layer K/V sharing** (any layer with
///   `arch.kv_shared_source_layer(l).is_some()` — Gemma 4 family): the
///   sharing layer's attention needs to read the donor's full cache rather
///   than just this-step's K/V, which the current decode helper doesn't
///   model.
///
/// The bench-vs-llama-cpp target (Gemma 3 4B Q4_K) hits neither bypass.
pub fn predict_q4k_hidden_with_cache(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
    mut kv_cache: Option<&mut KvCache>,
) -> Array2<f32> {
    let num_layers = weights.num_layers;
    if kv_cache.is_some() {
        let arch = &*weights.arch;
        let unsupported = arch.is_hybrid_moe()
            || (0..num_layers).any(|l| arch.kv_shared_source_layer(l).is_some());
        if unsupported {
            return predict_q4k_hidden(weights, token_ids, index, moe_remote);
        }
    }
    let mut h = embed_tokens_pub(weights, token_ids);

    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut shared_kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    let dump_dir = crate::forward::dump_config::DumpConfig::get().layer_dir();
    if let Some(dir) = dump_dir {
        let slice = h.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/cpu_h_embed.f32"), &bytes);
    }

    // Per-layer dispatch: when the layer can fully run through the direct
    // Q4_K × Q8_K paths (Q4kDirectFfn for FFN, Q4kDirectAttention for the
    // attention prefill + decode), the f32 dequant cache from
    // `insert_q4k_layer_tensors` is dead weight — skipping the insert
    // saves ~10 GB resident on Gemma 3 4B (PR after #145).
    //
    // The cache is still needed for:
    // - non-Q8K-aligned hidden_size (Gemma 3 1B at hidden=1152): direct
    //   path's `quantize_x_to_q8k` requires `hidden % 256 == 0`;
    // - hybrid-MoE layers: the MoE expert path doesn't yet have a direct
    //   Q4_K variant;
    // - cross-layer K/V share donors (Gemma 4 family): attention pulls
    //   the donor's full K/V via `shared_kv`, not the direct path.
    let arch_pre = &*weights.arch;
    let q_dim_pre = arch_pre.num_q_heads_for_layer(0) * arch_pre.head_dim_for_layer(0);
    let direct_all_layers = !arch_pre.is_hybrid_moe()
        && h.shape()[1].is_multiple_of(256)
        && q_dim_pre.is_multiple_of(256)
        && !(0..num_layers).any(|l| arch_pre.kv_shared_source_layer(l).is_some());

    // When `direct_all_layers` is set we skip per-layer f32 dequant into
    // `weights.tensors`. The cached attention path
    // (`run_attention_block_with_kv_out_with_cache`) handles this by routing
    // through `q4k_prefill` / `q4k_direct`, which read straight from the
    // vindex. The *uncached* early-exit at the top of that function falls
    // back to `run_attention_block_with_kv_out`, which has no vindex handle
    // and reads from `weights.tensors` — empty under `direct_all_layers`,
    // producing a silent no-op forward. Allocate an internal cache so we
    // always take the vindex-aware path. The two bypass architectures
    // (hybrid-MoE, cross-layer K/V share) are already excluded by
    // `direct_all_layers`'s own guard.
    let mut owned_cache: Option<KvCache> =
        (kv_cache.is_none() && direct_all_layers).then(|| KvCache::with_layers(num_layers));
    let mut kv_cache: Option<&mut KvCache> = kv_cache.or(owned_cache.as_mut());

    for layer in 0..num_layers {
        let inserted = if direct_all_layers {
            Vec::new()
        } else {
            insert_q4k_layer_tensors(weights, index, layer).unwrap_or_else(|err| panic!("{err}"))
        };

        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| shared_kv_cache.get(&src));
        let is_moe_layer = weights.arch.is_hybrid_moe();
        // Decode-step FFN (seq == 1) uses the direct Q4_K × Q8_K matvec
        // path — skips f32 dequant entirely. Prefill (seq > 1) also uses
        // Q4kDirectFfn when the alignment guards hold; the function's
        // per-row matvec loop is slower than a BLAS GEMM on the dequant
        // cache but doesn't need that cache. RAM win > prefill speed loss
        // on the bench target (Gemma 3 4B end-to-end). Models with
        // non-Q8K-aligned hidden_size or hybrid-MoE fall back to WeightFfn
        // (which still requires the cache — handled by the
        // `direct_all_layers` gate above).
        let weight_ffn = crate::ffn::WeightFfn { weights };
        let direct_ffn = crate::ffn::Q4kDirectFfn {
            arch: &*weights.arch,
            index,
        };
        let hidden_q8k_aligned = h.shape()[1].is_multiple_of(256);
        let ffn_backend: &dyn crate::ffn::FfnBackend = if !is_moe_layer && hidden_q8k_aligned {
            &direct_ffn
        } else {
            &weight_ffn
        };
        if is_moe_layer {
            if let Some((h_new, kv_out)) = run_moe_layer_cpu(
                weights,
                &h,
                layer,
                ffn_backend,
                ple_inputs.get(layer),
                shared_kv,
                moe_remote,
            ) {
                h = h_new;
                if let Some(kv) = kv_out {
                    shared_kv_cache.insert(layer, kv);
                }
            }
        } else if let Some((h_new, _, kv_out)) = run_layer_with_ffn_with_cache(
            weights,
            &h,
            layer,
            ffn_backend,
            false,
            ple_inputs.get(layer),
            shared_kv,
            kv_cache.as_deref_mut(),
            Some(index),
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                shared_kv_cache.insert(layer, kv);
            }
        }

        // Keep `inserted` resident across decode steps so subsequent
        // tokens skip the per-layer Q4_K dequant. See
        // `insert_q4k_layer_tensors` doc for memory budget.
        let _ = inserted;

        if let Some(dir) = dump_dir {
            let slice = h.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let path = crate::forward::dump_config::cpu_layer_path(dir, layer);
            if let Err(e) = std::fs::write(&path, &bytes) {
                eprintln!("[dump] failed to write {path}: {e}");
            }
        }
    }

    if let Some(cache) = kv_cache {
        cache.next_position = cache.next_position.saturating_add(token_ids.len());
    }

    h
}

/// Build `MoeRouterWeights` for a single layer from the model's vector store.
fn build_moe_router_weights<'a>(
    weights: &'a larql_models::ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Option<crate::ffn::MoeRouterWeights<'a>> {
    let router_key = arch.moe_router_key(layer)?;
    let router_proj = weights.vectors.get(&router_key)?.as_slice();
    let sl = |k: Option<String>| -> &'a [f32] {
        k.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    Some(crate::ffn::MoeRouterWeights {
        router_proj,
        router_scale: sl(arch.moe_router_scale_key(layer)),
        router_per_expert_scale: sl(arch.moe_router_per_expert_scale_key(layer)),
        router_norm: sl(arch.moe_router_norm_key(layer)),
        router_norm_parameter_free: arch.moe_router_norm_parameter_free(),
        router_input_scalar: arch.moe_router_input_scalar().unwrap_or(1.0),
        pre_experts_norm: sl(arch.moe_pre_experts_norm_key(layer)),
        post_experts_norm: sl(arch.moe_post_experts_norm_key(layer)),
        num_experts: arch.num_experts(),
        top_k: arch.num_experts_per_token(),
    })
}

/// CPU forward for one hybrid-MoE layer (Gemma 4 26B A4B).
fn run_moe_layer_cpu(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn crate::ffn::FfnBackend,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
    moe_remote: Option<&crate::ffn::RemoteMoeBackend>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = h.ncols();

    let (h_post_attn, kv_out) = if let Some(shared) = shared_kv {
        let (h_pa, _, _) =
            crate::attention::run_attention_block_shared(weights, h, layer, false, Some(shared))?;
        (h_pa, None)
    } else {
        let (h_pa, _, _, k_rope, v_final) =
            crate::attention::run_attention_block_with_kv_out(weights, h, layer, false, None)?;
        (h_pa, Some((k_rope, v_final)))
    };

    if let Some(dir) = crate::forward::dump_config::DumpConfig::get().layer_dir() {
        let slice = h_post_attn.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = crate::forward::dump_config::cpu_layer_h_post_attn_path(dir, layer);
        let _ = std::fs::write(&path, &bytes);
    }

    let (h_post_ffn_dense, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, ffn, false);
    let h1 = &h_post_ffn_dense - &h_post_attn;

    let seq_len = h_post_attn.nrows();
    let mut h2 = Array2::<f32>::zeros((seq_len, hidden));

    if let Some(remote) = moe_remote {
        if let Some(router) = build_moe_router_weights(weights, arch, layer) {
            match remote.forward_moe_seq(layer, &h_post_attn, &router, norm_offset, eps) {
                Ok(out) => h2 = out,
                Err(e) => eprintln!("[run_moe_layer_cpu] remote dispatch error L{layer}: {e}"),
            }
        }
    } else {
        let moe_weights =
            crate::layer_graph::pipeline_layer::build_moe_weights(weights, arch, layer);
        if let Some(ref moe) = moe_weights {
            for pos in 0..seq_len {
                let row: Vec<f32> = h_post_attn.row(pos).to_vec();
                let moe_out =
                    larql_compute::cpu::ops::moe::cpu_moe_forward(&row, moe, norm_offset, eps);
                for (dst, src) in h2.row_mut(pos).iter_mut().zip(moe_out.iter()) {
                    *dst = *src;
                }
            }
        } else {
            let mut out = h_post_ffn_dense;
            let mut h_ple =
                crate::forward::ple::apply_per_layer_embedding(weights, &out, layer, ple_input);
            crate::forward::layer::apply_layer_scalar(weights, &mut h_ple, layer);
            out = h_ple;
            return Some((out, kv_out));
        }
    }

    let combined = &h1 + &h2;

    let l0_stage_dump = crate::forward::dump_config::DumpConfig::get().stage_dir(layer);
    let dump_l0_arr = |name: &str, arr: &Array2<f32>| {
        if let Some(dir) = l0_stage_dump {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(
                crate::forward::dump_config::cpu_stage_path(dir, name),
                &bytes,
            );
        }
    };
    dump_l0_arr("h1_dense_norm1", &h1);
    dump_l0_arr("h2_moe_norm2", &h2);
    dump_l0_arr("combined_h1_plus_h2", &combined);

    let outer_w_vec: Option<&Vec<f32>> = if arch.moe_has_combined_output_norm() {
        arch.moe_post_outer_norm_key(layer)
            .or_else(|| arch.post_feedforward_layernorm_key(layer))
            .and_then(|k| weights.vectors.get(&k))
    } else {
        None
    };

    let seq = combined.nrows();
    let mut out_buf = Array2::<f32>::zeros((seq, hidden));
    for pos in 0..seq {
        let h_post_attn_row = h_post_attn.row(pos);
        let combined_row = combined.row(pos);
        let combined_normed = larql_compute::cpu::ops::outer_combine::outer_post_norm_residual(
            h_post_attn_row.as_slice().expect("contiguous row"),
            combined_row.as_slice().expect("contiguous row"),
            outer_w_vec.map(|v| v.as_slice()),
            norm_offset,
            eps,
        );
        for (dst, src) in out_buf.row_mut(pos).iter_mut().zip(combined_normed.iter()) {
            *dst = *src;
        }
    }
    dump_l0_arr("h_out_pre_layer_scalar", &out_buf);

    let mut h_out =
        crate::forward::ple::apply_per_layer_embedding(weights, &out_buf, layer, ple_input);
    if let Some(scalar_key) = arch.layer_scalar_key(layer) {
        if let Some(scalars) = weights.vectors.get(&scalar_key) {
            if let Some(&scalar) = scalars.first() {
                let flat = h_out.as_slice_mut().expect("contiguous out_buf");
                larql_compute::cpu::ops::outer_combine::apply_layer_scalar_in_place(flat, scalar);
            }
        }
    }

    Some((h_out, kv_out))
}
