//! Autoregressive generation via a sharded expert grid.
//!
//! Uses the Metal pipeline for attention + dense FFN (same as normal `generate`),
//! but intercepts the MoE expert block per layer via a callback that dispatches
//! to remote shards over HTTP instead of calling `cpu_moe_forward` locally.
//!
//! The hook: `ComputeBackend::decode_token_with_moe(layers, x, ..., moe_fn)`
//! where `moe_fn(layer, h_post_attn) -> Vec<f32>` calls
//! `RemoteMoeBackend::forward_moe`.

use larql_compute::ComputeBackend;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use crate::ffn::RemoteMoeBackend;
use crate::ffn::moe_remote::{MoeRouterWeights, RemoteMoeError};
use crate::layer_graph::pipeline_layer::build_pipeline_layers;
use crate::layer_graph::generate::lm_head_topk as lm_topk;
use crate::forward::{apply_norm, embed_tokens_pub};

pub struct GridGenerateResult {
    pub tokens: Vec<String>,
    pub decode_ms: Vec<f64>,
}

/// Greedy autoregressive generation through a remote-expert grid.
///
/// Requires a Metal (or Q4-capable) backend — attention and dense FFN run on
/// the GPU exactly as in the normal `generate()` path.  Expert blocks are
/// dispatched to `remote` instead of running locally.
pub fn generate_with_remote_moe(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    remote: &RemoteMoeBackend,
    backend: &dyn ComputeBackend,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    let eos_id: u32 = 1;

    // ── Build pipeline layers (same as generate()) ────────────────────────────
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let q4_ffn = gate_index.interleaved_q4k_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .ok_or_else(|| RemoteMoeError::BadResponse(
            "no interleaved Q4 FFN mmap in vindex".into()))?;
    let ffn_is_q4k = gate_index.interleaved_q4k_mmap_ref().is_some();

    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 144
    } else {
        intermediate * hidden / 32 * 18
    };
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };

    let layers = build_pipeline_layers(weights, index, 0..num_layers,
                                       q4_ffn, q4_ffn_per_matrix, ffn_format);

    let q_dim  = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope   = arch.rope_base_for_layer(0) as f32;

    // ── Prefill ───────────────────────────────────────────────────────────────
    // GPU prefill builds the KV cache for prompt tokens.  We run the standard
    // prefill (which uses local experts) as an approximation — the prefill
    // residuals are slightly wrong but the KV cache is built correctly for
    // attention patterns.  Decode uses the remote experts from token 0.
    backend.reset_kv_cache();

    // Pre-allocate per-layer KV cache for asymmetric attention geometry (Gemma 4 26B).
    {
        let arch = &*weights.arch;
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, 4096);
    }

    let seq_len = prompt_ids.len();

    let h_embed = embed_tokens_pub(weights, &prompt_ids);
    let x: Vec<f32> = h_embed.as_slice().unwrap_or(&[]).to_vec();

    let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
    let qk_norm = arch.attn_q_norm_key(0).is_some();

    // Run GPU prefill (uses local experts for prefill positions).
    let h_prefill = backend.prefill_q4(
        &layers, &x, hidden, intermediate, q_dim, kv_dim,
        seq_len, weights.num_q_heads, weights.num_kv_heads, weights.head_dim,
        rope, qk_norm, softcap,
    ).ok_or_else(|| RemoteMoeError::BadResponse(
        "GPU prefill not available — need Metal backend".into()))?;

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut last_hidden_vec = h_prefill;
    let mut current_ids = prompt_ids;
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();

    // Get initial top-1 prediction from prefill output.
    let prefill_h_arr = ndarray::Array2::from_shape_vec(
        (seq_len, hidden), last_hidden_vec.clone()
    ).map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let last0 = h_norm0.row(seq_len - 1).to_owned();
    let first_id = lm_topk(index, weights, &last0, 1, backend)
        .into_iter().next().map(|(id, _)| id).unwrap_or(0);

    let first_tok = crate::tokenizer::decode_token(tokenizer, first_id)
        .unwrap_or_else(|| format!("<{first_id}>"));
    tokens.push(first_tok);
    current_ids.push(first_id);
    if first_id == eos_id || tokens.len() >= max_tokens {
        return Ok(GridGenerateResult { tokens, decode_ms: vec![0.0] });
    }

    for _step in 0..max_tokens.saturating_sub(1) {
        let t0 = std::time::Instant::now();
        let next_input_id = *current_ids.last().unwrap();

        // Embed next token.
        let tok_embed = embed_tokens_pub(weights, &[next_input_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        // Build the expert dispatch closure for this decode step.
        // Called once per MoE layer by `decode_token_with_moe`.
        let mut step_error: Option<RemoteMoeError> = None;
        // SKIP_MOE=1 zeroes out the expert block (diagnostic: checks if dense FFN alone is correct).
        let skip_moe = std::env::var("SKIP_MOE").is_ok();

        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            if skip_moe { return vec![0.0f32; hidden]; }
            if step_error.is_some() {
                return vec![0.0f32; hidden];
            }
            let arch = &*weights.arch;
            let router_proj_key = match arch.moe_router_key(layer) {
                Some(k) => k,
                None => return vec![0.0f32; hidden],
            };
            let router_proj = match weights.vectors.get(&router_proj_key) {
                Some(v) => v,
                None => return vec![0.0f32; hidden],
            };
            let router_scale = arch.moe_router_scale_key(layer)
                .and_then(|k| weights.vectors.get(&k))
                .map(|v| v.as_slice()).unwrap_or(&[]);
            let per_expert_scale = arch.moe_router_per_expert_scale_key(layer)
                .and_then(|k| weights.vectors.get(&k))
                .map(|v| v.as_slice()).unwrap_or(&[]);
            let pre_experts_norm = arch.moe_pre_experts_norm_key(layer)
                .and_then(|k| weights.vectors.get(&k))
                .map(|v| v.as_slice()).unwrap_or(&[]);
            let post_experts_norm = arch.moe_post_experts_norm_key(layer)
                .and_then(|k| weights.vectors.get(&k))
                .map(|v| v.as_slice()).unwrap_or(&[]);
            let router_norm = arch.moe_router_norm_key(layer)
                .and_then(|k| weights.vectors.get(&k))
                .map(|v| v.as_slice()).unwrap_or(&[]);
            let router_norm_parameter_free = arch.moe_router_norm_parameter_free();
            let router_input_scalar = arch.moe_router_input_scalar().unwrap_or(1.0);

            let router = MoeRouterWeights {
                router_proj: router_proj.as_slice(),
                router_scale,
                router_per_expert_scale: per_expert_scale,
                router_norm,
                router_norm_parameter_free,
                router_input_scalar,
                pre_experts_norm,
                post_experts_norm,
                num_experts: arch.num_experts(),
                top_k: arch.num_experts_per_token(),
            };

            match remote.forward_moe(layer, h_post_attn, &router, norm_offset, eps) {
                Ok(out) => out,
                Err(e) => {
                    step_error = Some(e);
                    vec![0.0f32; hidden]
                }
            }
        };

        let result = backend.decode_token_with_moe(
            &layers, &x_tok, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim,
            rope, &mut moe_fn,
        );

        if let Some(err) = step_error { return Err(err); }

        let h_vec = result.ok_or_else(|| RemoteMoeError::BadResponse(
            "decode_token_with_moe returned None".into()))?;

        last_hidden_vec = h_vec;

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let last_hidden = h_normed.row(0).to_owned();
        let next_id = lm_topk(index, weights, &last_hidden, 1, backend)
            .into_iter().next().map(|(id, _)| id).unwrap_or(0);

        decode_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let tok_str = crate::tokenizer::decode_token(tokenizer, next_id)
            .unwrap_or_else(|| format!("<{next_id}>"));
        tokens.push(tok_str);
        current_ids.push(next_id);

        if next_id == eos_id { break; }
    }

    Ok(GridGenerateResult { tokens, decode_ms })
}
