//! GGUF → `Qwen35Weights` bridge (Phase C.4d-bridge of
//! `inference-qwen35-deltanet`).
//!
//! Translates a loaded `ModelWeights` (HashMap of normalized tensor
//! keys → `ArcArray2<f32>` / `Vec<f32>`) into the typed
//! `Qwen35Weights` struct expected by `qwen35_forward_step`.
//!
//! Layer dispatch goes through `arch.is_linear_attention_layer(layer)`
//! so the same loader handles both Qwen 3.6 27B (dense) and
//! 35B-A3B (MoE) once an MoE-aware forward arrives. For now this
//! handles only the dense FFN suffix (Phase D will add the MoE
//! routed-expert path).
//!
//! All tensor look-ups are O(1) HashMap probes; the function does
//! `Arc` bumps and one `Arc::from(Vec<f32>)` per 1-D tensor (which
//! moves the Vec without copying its elements). No data copies on
//! the 2-D weights — `ArcArray2::clone()` only bumps the refcount.

use std::sync::Arc;

use larql_models::{ModelArchitecture, ModelWeights};

use super::deltanet_block::DeltaNetLayerWeights;
use super::qwen35_block::{Qwen35AttentionLayerWeights, Qwen35LayerWeights};
use super::qwen35_forward::{Qwen35FullLayerWeights, Qwen35MoeFfnWeights, Qwen35Weights};

/// Bridge errors — every variant names the missing tensor key so
/// the user can grep the GGUF for "did this load?" diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35LoadError {
    #[error("missing 2-D tensor: {0}")]
    MissingTensor(String),
    #[error("missing 1-D tensor (in `vectors` map): {0}")]
    MissingVector(String),
    #[error(
        "architecture method returned None for required key on layer {layer}: {method}; \
         is this layer really {kind}?"
    )]
    NoKey {
        layer: usize,
        method: &'static str,
        kind: &'static str,
    },
    #[error(
        "architecture is not hybrid Qwen 3.6 — {field} is 0. Did you pass a \
         Qwen35Arch or Qwen35MoeArch?"
    )]
    NotHybrid { field: &'static str },
}

/// Build a typed `Qwen35Weights` from a loaded `ModelWeights`.
///
/// Walks 0..num_layers, dispatching each to either the DeltaNet
/// branch (linear attention) or the full-attention branch via
/// `arch.is_linear_attention_layer(layer)`. Each branch pulls the
/// per-layer tensors using the architecture's key methods, falling
/// back to the trait defaults for the FFN trio + norms that follow
/// the standard `layers.N.<x>.weight` convention.
///
/// Returns `Qwen35Weights` with shared Arc-owned tensors that point
/// into the original `ModelWeights` storage (no copies).
pub fn load_qwen35_weights(
    weights: &ModelWeights,
    arch: &dyn ModelArchitecture,
) -> Result<Qwen35Weights, Qwen35LoadError> {
    if arch.full_attention_interval() == 0 {
        return Err(Qwen35LoadError::NotHybrid {
            field: "full_attention_interval",
        });
    }
    if arch.ssm_state_size() == 0 {
        return Err(Qwen35LoadError::NotHybrid {
            field: "ssm_state_size",
        });
    }

    let n_layers = weights.num_layers;
    let mut layers: Vec<Qwen35FullLayerWeights> = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let block = if arch.is_linear_attention_layer(layer) {
            Qwen35LayerWeights::Linear(load_deltanet_layer(weights, arch, layer)?)
        } else {
            Qwen35LayerWeights::Attention(load_attention_layer(weights, arch, layer)?)
        };
        // RMSNorm convention for Qwen 3.6: this GGUF stores `(1 + w)`
        // pre-applied (verified empirically via C.4y/C.5a diagnostic:
        // stored layers.0.input_layernorm.weight has mean=0.98, range
        // [0.88, 1.20] — i.e. ≈ 1.0, not the ≈0.05 you'd expect for a
        // raw zero-init delta).
        //
        // So the forward's `x * inv_rms * w` form just works.
        // No `add_one_to_vec` needed here despite Qwen3-Next using
        // `Qwen3NextRMSNorm(x) = x * inv_rms * (1 + w)` semantically.
        let attn_post_norm = get_vec(weights, &arch.post_attention_layernorm_key(layer))?;
        let empty_arr = ndarray::ArcArray2::from_shape_vec((0, 0), Vec::new())
            .expect("empty array is always valid");
        // MoE arch ships per-layer expert tensors instead of the
        // dense `mlp.{gate,up,down}_proj.weight` set. Branch up front
        // so we never try to pull the absent dense keys.
        let (ffn_gate, ffn_up, ffn_down, ffn_gate_quant, ffn_up_quant, ffn_down_quant, moe) =
            if arch.is_moe() {
                let moe = load_qwen35_moe_ffn(weights, arch, layer)?;
                (
                    empty_arr.clone(),
                    empty_arr.clone(),
                    empty_arr.clone(),
                    None,
                    None,
                    None,
                    Some(moe),
                )
            } else {
                let ffn_gate_key = arch.ffn_gate_key(layer);
                let ffn_up_key = arch.ffn_up_key(layer);
                let ffn_down_key = arch.ffn_down_key(layer);
                // Lazy-quant takes precedence per projection. The dense
                // field gets a 0×0 placeholder when the quant form is
                // available so we don't pay RAM twice.
                let ffn_gate_quant = weights.quant_tensors.get(&ffn_gate_key).cloned();
                let ffn_up_quant = weights.quant_tensors.get(&ffn_up_key).cloned();
                let ffn_down_quant = weights.quant_tensors.get(&ffn_down_key).cloned();
                let ffn_gate = if ffn_gate_quant.is_some() {
                    empty_arr.clone()
                } else {
                    get_tensor(weights, &ffn_gate_key)?
                };
                let ffn_up = if ffn_up_quant.is_some() {
                    empty_arr.clone()
                } else {
                    get_tensor(weights, &ffn_up_key)?
                };
                let ffn_down = if ffn_down_quant.is_some() {
                    empty_arr.clone()
                } else {
                    get_tensor(weights, &ffn_down_key)?
                };
                (
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    ffn_gate_quant,
                    ffn_up_quant,
                    ffn_down_quant,
                    None,
                )
            };
        layers.push(Qwen35FullLayerWeights {
            block,
            attn_post_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
            ffn_gate_quant,
            ffn_up_quant,
            ffn_down_quant,
            moe,
        });
    }

    let final_norm = get_vec(weights, arch.final_norm_key())?;

    Ok(Qwen35Weights {
        embed: weights.embed.as_array().into_owned(),
        embed_quant: weights.embed_quant.clone(),
        layers,
        final_norm,
        lm_head: weights.lm_head.clone(),
        lm_head_quant: weights.lm_head_quant.clone(),
        ffn_dim: arch.config().intermediate_size,
        // Backend is attached by the caller (typically the bench
        // harness reading `LARQL_QWEN35_GPU=1`). The bridge stays
        // backend-agnostic so synthetic-weight tests don't need
        // CUDA available.
        backend: None,
    })
}

// `add_one_to_vec` removed in C.5a: empirical inspection of the
// actual Qwen 3.6 27B Q4_K_S GGUF showed stored norm weights are
// already ≈ 1.0 (i.e. `(1 + w)` is pre-applied in the conversion
// path), so adding 1.0 again double-applies the offset.

/// GGUF-side tensor keys for the per-layer MoE FFN.
///
/// All keys live in `weights.quant_tensors` (the bench harness adds
/// them to the lazy-load set explicitly). Names follow llama.cpp's
/// GGUF convention rewritten through the loader's `blk.` → `layers.`
/// substitution — no other rewrite rule fires.
///
/// Layout (per F.0 probe of Qwen3.6-35B-A3B):
/// - `ffn_gate_exps.weight`  Q4_K, 3D `[hidden, ffn_dim, num_experts]`
///   → flattened by the lazy loader to `[num_experts * ffn_dim, hidden]`.
/// - `ffn_up_exps.weight`    Q4_K, same shape.
/// - `ffn_down_exps.weight`  Q5_K, 3D `[ffn_dim, hidden, num_experts]`
///   → flattened to `[num_experts * hidden, ffn_dim]`.
/// - `ffn_gate_inp.weight`   f32, 2D `[hidden, num_experts]`
///   → flattened to `[num_experts, hidden]` (the router matvec produces
///   one logit per expert from the token vector).
/// - `ffn_{gate,up,down}_shexp.weight` shared expert (optional). Same
///   shapes as a dense SwiGLU but with `expert_ffn_dim`.
fn load_qwen35_moe_ffn(
    weights: &ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Result<Qwen35MoeFfnWeights, Qwen35LoadError> {
    let prefix = format!("layers.{layer}.");
    let key = |suffix: &str| format!("{prefix}{suffix}");

    let pull = |suffix: &str| -> Result<larql_models::quant::lazy::QuantTensor, Qwen35LoadError> {
        let k = key(suffix);
        weights
            .quant_tensors
            .get(&k)
            .cloned()
            .ok_or(Qwen35LoadError::MissingTensor(k))
    };
    let pull_opt = |suffix: &str| weights.quant_tensors.get(&key(suffix)).cloned();

    let router = pull("ffn_gate_inp.weight")?;
    let gate_exps = pull("ffn_gate_exps.weight")?;
    let up_exps = pull("ffn_up_exps.weight")?;
    let down_exps = pull("ffn_down_exps.weight")?;

    // Shared expert (always-on). Qwen3.6-35B-A3B has one; some MoE
    // variants don't. Treat as a single unit — either all three are
    // present or none are.
    let shexp_gate = pull_opt("ffn_gate_shexp.weight");
    let shexp_up = pull_opt("ffn_up_shexp.weight");
    let shexp_down = pull_opt("ffn_down_shexp.weight");
    let (shexp_gate, shexp_up, shexp_down) = match (shexp_gate, shexp_up, shexp_down) {
        (Some(g), Some(u), Some(d)) => (Some(g), Some(u), Some(d)),
        (None, None, None) => (None, None, None),
        // Partial set is almost certainly a bug — fail loudly so
        // we don't silently drop a shared expert that's in the
        // GGUF.
        (g, u, d) => {
            return Err(Qwen35LoadError::MissingTensor(format!(
                "partial shared-expert tensors for layer {layer}: \
                     gate={present_g}, up={present_u}, down={present_d}",
                present_g = g.is_some(),
                present_u = u.is_some(),
                present_d = d.is_some(),
            )));
        }
    };

    let num_experts = arch.num_experts();
    let top_k = arch.num_experts_per_token();
    if num_experts == 0 {
        return Err(Qwen35LoadError::NotHybrid {
            field: "num_experts",
        });
    }
    if top_k == 0 {
        return Err(Qwen35LoadError::NotHybrid {
            field: "num_experts_per_token",
        });
    }

    // Shared-expert per-feature sigmoid gate, optional. Lives in
    // `weights.vectors` (1D F32 [hidden]). Both Qwen 3.6 35B-A3B and
    // Coder-Next ship `ffn_gate_inp_shexp.weight`; without applying it
    // as `sigmoid(x · gate_inp) * shared_output`, the shared expert
    // adds an over-amplified contribution that destabilises the
    // token distribution. arch.shared_expert_gate_inp_key() exposes
    // the key; the default impl returns None for arches without one.
    let shexp_gate_inp: Option<Arc<[f32]>> = weights
        .vectors
        .get(&key("ffn_gate_inp_shexp.weight"))
        .map(|v| Arc::from(v.clone()));

    Ok(Qwen35MoeFfnWeights {
        router,
        gate_exps,
        up_exps,
        down_exps,
        shexp_gate,
        shexp_up,
        shexp_down,
        shexp_gate_inp,
        num_experts,
        top_k,
    })
}

/// Tensor keys to lazify when loading a Qwen3.6 MoE GGUF (e.g.
/// 35B-A3B). The bench harness folds these into the regular
/// `lazy_keys` set before calling `load_gguf_lazy_tensors`. Returns
/// every per-layer MoE tensor name (`layers.{L}.ffn_*_exps.weight`,
/// `ffn_gate_inp.weight`, `ffn_*_shexp.weight`) — `lazy_tensors`
/// silently ignores keys that don't exist in the GGUF, so listing
/// the shared-expert tensors for archs without one is safe.
pub fn qwen35_moe_lazy_keys(n_layers: usize) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for l in 0..n_layers {
        let p = format!("layers.{l}.");
        out.insert(format!("{p}ffn_gate_inp.weight"));
        out.insert(format!("{p}ffn_gate_exps.weight"));
        out.insert(format!("{p}ffn_up_exps.weight"));
        out.insert(format!("{p}ffn_down_exps.weight"));
        out.insert(format!("{p}ffn_gate_shexp.weight"));
        out.insert(format!("{p}ffn_up_shexp.weight"));
        out.insert(format!("{p}ffn_down_shexp.weight"));
    }
    out
}

fn load_deltanet_layer(
    weights: &ModelWeights,
    arch: &dyn ModelArchitecture,
    layer: usize,
) -> Result<DeltaNetLayerWeights, Qwen35LoadError> {
    // GGUF stores `(1+w)` pre-applied for attn_norm — use raw.
    let attn_norm = get_vec(weights, &arch.input_layernorm_key(layer))?;
    let attn_qkv_key = require_key(arch.attn_qkv_key(layer), layer, "attn_qkv_key", "linear")?;
    let attn_qkv_quant = weights.quant_tensors.get(&attn_qkv_key).cloned();
    let attn_qkv = if attn_qkv_quant.is_some() {
        ndarray::ArcArray2::from_shape_vec((0, 0), Vec::new()).unwrap()
    } else {
        get_tensor(weights, &attn_qkv_key)?
    };
    let attn_gate_key = require_key(arch.attn_gate_key(layer), layer, "attn_gate_key", "linear")?;
    let attn_gate_quant = weights.quant_tensors.get(&attn_gate_key).cloned();
    let attn_gate = if attn_gate_quant.is_some() {
        ndarray::ArcArray2::from_shape_vec((0, 0), Vec::new()).unwrap()
    } else {
        get_tensor(weights, &attn_gate_key)?
    };
    let ssm_conv1d_key = require_key(
        arch.ssm_conv1d_key(layer),
        layer,
        "ssm_conv1d_key",
        "linear",
    )?;
    // GGUF stores ssm_conv1d as `[d_conv, conv_dim]` in GGML's
    // (cols, rows) order, which the loader's standard dim-swap
    // produces as `[conv_dim, d_conv]` in HF's `[rows, cols]`.
    // The forward path (causal_conv1d_step) expects rows = d_conv
    // and cols = conv_dim (per-tap-channel layout), so transpose
    // once at load time. Trivial cost: 48 layers × 4 × 10240 f32
    // ≈ 7.8 MB total for Qwen 3.6 27B, done once.
    //
    // `t().to_owned()` keeps the transposed strides (non-standard
    // layout), which silently disables every `as_slice()`-based GPU
    // dispatch downstream. `as_standard_layout().to_owned()` forces
    // a row-major copy so the resulting ArcArray's `as_slice()` is
    // `Some` — required by the Phase E.4 / E.6 CUDA hooks.
    let ssm_conv1d_raw = get_tensor(weights, &ssm_conv1d_key)?;
    let ssm_conv1d = ssm_conv1d_raw
        .t()
        .as_standard_layout()
        .to_owned()
        .into_shared();
    let ssm_dt_key = require_key(arch.ssm_dt_key(layer), layer, "ssm_dt_key", "linear")?;
    let ssm_dt = get_vec(weights, &ssm_dt_key)?;
    let ssm_a_key = require_key(arch.ssm_a_key(layer), layer, "ssm_a_key", "linear")?;
    let ssm_a = get_vec(weights, &ssm_a_key)?;
    let ssm_beta_key = require_key(arch.ssm_beta_key(layer), layer, "ssm_beta_key", "linear")?;
    let ssm_beta = get_tensor(weights, &ssm_beta_key)?;
    let ssm_alpha_key = require_key(arch.ssm_alpha_key(layer), layer, "ssm_alpha_key", "linear")?;
    let ssm_alpha = get_tensor(weights, &ssm_alpha_key)?;
    let ssm_norm_key = require_key(arch.ssm_norm_key(layer), layer, "ssm_norm_key", "linear")?;
    let ssm_norm = get_vec(weights, &ssm_norm_key)?;
    let ssm_out_key = require_key(arch.ssm_out_key(layer), layer, "ssm_out_key", "linear")?;
    let ssm_out_quant = weights.quant_tensors.get(&ssm_out_key).cloned();
    let ssm_out = if ssm_out_quant.is_some() {
        ndarray::ArcArray2::from_shape_vec((0, 0), Vec::new()).unwrap()
    } else {
        get_tensor(weights, &ssm_out_key)?
    };

    Ok(DeltaNetLayerWeights {
        attn_norm,
        attn_qkv,
        attn_gate,
        ssm_conv1d,
        ssm_dt,
        ssm_a,
        ssm_beta,
        ssm_alpha,
        ssm_norm,
        ssm_out,
        attn_qkv_quant,
        attn_gate_quant,
        ssm_out_quant,
    })
}

fn load_attention_layer(
    weights: &ModelWeights,
    arch: &dyn ModelArchitecture,
    layer: usize,
) -> Result<Qwen35AttentionLayerWeights, Qwen35LoadError> {
    // GGUF stores `(1+w)` pre-applied for all norms — use raw values.
    let attn_norm = get_vec(weights, &arch.input_layernorm_key(layer))?;
    let empty = || ndarray::ArcArray2::from_shape_vec((0, 0), Vec::new()).unwrap();
    let attn_q_key = arch.attn_q_key(layer);
    let attn_q_quant = weights.quant_tensors.get(&attn_q_key).cloned();
    let attn_q = if attn_q_quant.is_some() {
        empty()
    } else {
        get_tensor(weights, &attn_q_key)?
    };
    let attn_k_key = arch.attn_k_key(layer);
    let attn_k_quant = weights.quant_tensors.get(&attn_k_key).cloned();
    let attn_k = if attn_k_quant.is_some() {
        empty()
    } else {
        get_tensor(weights, &attn_k_key)?
    };
    let attn_v_key = arch.attn_v_key(layer);
    let attn_v_quant = weights.quant_tensors.get(&attn_v_key).cloned();
    let attn_v = if attn_v_quant.is_some() {
        empty()
    } else {
        get_tensor(weights, &attn_v_key)?
    };
    let attn_output_key = arch.attn_o_key(layer);
    let attn_output_quant = weights.quant_tensors.get(&attn_output_key).cloned();
    let attn_output = if attn_output_quant.is_some() {
        empty()
    } else {
        get_tensor(weights, &attn_output_key)?
    };
    let q_norm_key = require_key(
        arch.attn_q_per_head_norm_key(layer),
        layer,
        "attn_q_per_head_norm_key",
        "full-attn",
    )?;
    let attn_q_norm = get_vec(weights, &q_norm_key)?;
    let k_norm_key = require_key(
        arch.attn_k_per_head_norm_key(layer),
        layer,
        "attn_k_per_head_norm_key",
        "full-attn",
    )?;
    let attn_k_norm = get_vec(weights, &k_norm_key)?;

    Ok(Qwen35AttentionLayerWeights {
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_q_norm,
        attn_k_norm,
        attn_output,
        attn_q_quant,
        attn_k_quant,
        attn_v_quant,
        attn_output_quant,
    })
}

fn get_tensor(
    weights: &ModelWeights,
    key: &str,
) -> Result<ndarray::ArcArray2<f32>, Qwen35LoadError> {
    weights
        .tensors
        .get(key)
        .cloned()
        .ok_or_else(|| Qwen35LoadError::MissingTensor(key.to_string()))
}

fn get_vec(weights: &ModelWeights, key: &str) -> Result<Arc<[f32]>, Qwen35LoadError> {
    weights
        .vectors
        .get(key)
        .map(|v| Arc::from(v.as_slice()))
        .ok_or_else(|| Qwen35LoadError::MissingVector(key.to_string()))
}

fn require_key(
    opt: Option<String>,
    layer: usize,
    method: &'static str,
    kind: &'static str,
) -> Result<String, Qwen35LoadError> {
    opt.ok_or(Qwen35LoadError::NoKey {
        layer,
        method,
        kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::config::{ModelArchitecture as _, ModelConfig};
    use larql_models::{Qwen35Arch, WeightArray};
    use ndarray::Array2;
    use std::collections::HashMap;

    fn attach_cuda_backend_if_requested(w: &mut Qwen35Weights) {
        if std::env::var("LARQL_QWEN35_GPU").is_err() {
            return;
        }

        #[cfg(feature = "cuda")]
        {
            match larql_compute::cuda::CudaBackend::new() {
                Ok(b) => {
                    eprintln!("attached CudaBackend");
                    w.backend = Some(std::sync::Arc::new(b));
                }
                Err(e) => {
                    eprintln!("LARQL_QWEN35_GPU set but CudaBackend::new() failed: {e:?}");
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("LARQL_QWEN35_GPU set but larql-inference was built without --features cuda");
        }
    }

    /// Build a tiny synthetic Qwen3.6 config — 4 layers,
    /// `full_attention_interval=2` so layers 1 and 3 are full-attention
    /// and 0, 2 are linear.
    fn qwen35_tiny_config() -> ModelConfig {
        ModelConfig {
            model_type: "qwen35".into(),
            num_layers: 4,
            hidden_size: 8,
            intermediate_size: 16,
            head_dim: 4,
            num_q_heads: 2,
            num_kv_heads: 1,
            vocab_size: Some(32),
            rope_base: 10_000.0,
            rope_local_base: None,
            sliding_window: None,
            num_experts: None,
            num_experts_per_token: None,
            num_shared_experts: None,
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            embedding_multiplier: None,
            residual_multiplier: None,
            attention_multiplier: None,
            logits_scaling: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            per_layer_embed_dim: None,
            num_kv_shared_layers: None,
            full_attention_interval: Some(2),
            ssm_state_size: Some(2),
            ssm_inner_size: Some(4),
            ssm_group_count: Some(1),
            ssm_dt_rank: Some(2),
            ssm_conv_kernel: Some(2),
            rope_dimension_sections: None,
        }
    }

    fn non_hybrid_config() -> ModelConfig {
        let mut cfg = qwen35_tiny_config();
        cfg.full_attention_interval = None;
        cfg.ssm_state_size = None;
        cfg
    }

    fn make_2d(rows: usize, cols: usize, fill: f32) -> WeightArray {
        Array2::from_elem((rows, cols), fill).into_shared()
    }

    fn populate_tensors(
        tensors: &mut HashMap<String, WeightArray>,
        vectors: &mut HashMap<String, Vec<f32>>,
        arch: &Qwen35Arch,
    ) {
        let hidden = arch.config().hidden_size;
        let ffn_dim = arch.config().intermediate_size;
        let head_dim = arch.config().head_dim;
        let q_dim = arch.config().num_q_heads * head_dim;
        let kv_dim = arch.config().num_kv_heads * head_dim;
        let fused_q_dim = 2 * q_dim;

        let n_v_heads = arch.ssm_dt_rank();
        let head_v_dim = arch.ssm_state_size();
        let n_k_heads = arch.ssm_group_count();
        let key_dim = head_v_dim * n_k_heads;
        let value_dim = head_v_dim * n_v_heads;
        let conv_dim = 2 * key_dim + value_dim;

        vectors.insert("norm.weight".into(), vec![1.0; hidden]);

        for layer in 0..arch.config().num_layers {
            let prefix = format!("layers.{layer}.");
            vectors.insert(format!("{prefix}input_layernorm.weight"), vec![1.0; hidden]);
            // Qwen35Arch overrides post_attention_layernorm_key →
            // `post_attention_norm.weight` (GGUF-native naming).
            vectors.insert(
                format!("{prefix}post_attention_norm.weight"),
                vec![1.0; hidden],
            );
            tensors.insert(
                format!("{prefix}mlp.gate_proj.weight"),
                make_2d(ffn_dim, hidden, 0.1),
            );
            tensors.insert(
                format!("{prefix}mlp.up_proj.weight"),
                make_2d(ffn_dim, hidden, 0.1),
            );
            tensors.insert(
                format!("{prefix}mlp.down_proj.weight"),
                make_2d(hidden, ffn_dim, 0.1),
            );

            if arch.is_linear_attention_layer(layer) {
                tensors.insert(
                    format!("{prefix}self_attn.qkv_proj.weight"),
                    make_2d(conv_dim, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}attn_gate.weight"),
                    make_2d(value_dim, hidden, 0.1),
                );
                // Match the real GGUF layout `[conv_dim, d_conv]` —
                // the bridge transposes to `[d_conv, conv_dim]` for
                // the forward path.
                tensors.insert(
                    format!("{prefix}ssm_conv1d.weight"),
                    make_2d(conv_dim, arch.ssm_conv_kernel(), 0.5),
                );
                vectors.insert(format!("{prefix}ssm_dt.bias"), vec![0.0; n_v_heads]);
                vectors.insert(format!("{prefix}ssm_a"), vec![-1.0; n_v_heads]);
                tensors.insert(
                    format!("{prefix}ssm_beta.weight"),
                    make_2d(n_v_heads, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}ssm_alpha.weight"),
                    make_2d(n_v_heads, hidden, 0.1),
                );
                vectors.insert(format!("{prefix}ssm_norm.weight"), vec![1.0; head_v_dim]);
                tensors.insert(
                    format!("{prefix}ssm_out.weight"),
                    make_2d(hidden, value_dim, 0.5),
                );
            } else {
                tensors.insert(
                    format!("{prefix}self_attn.q_proj.weight"),
                    make_2d(fused_q_dim, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}self_attn.k_proj.weight"),
                    make_2d(kv_dim, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}self_attn.v_proj.weight"),
                    make_2d(kv_dim, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}self_attn.o_proj.weight"),
                    make_2d(hidden, q_dim, 0.5),
                );
                // GGUF-native naming (no `self_attn` infix).
                vectors.insert(format!("{prefix}attn_q_norm.weight"), vec![1.0; head_dim]);
                vectors.insert(format!("{prefix}attn_k_norm.weight"), vec![1.0; head_dim]);
            }
        }
    }

    fn build_model_weights(cfg: ModelConfig) -> (ModelWeights, Qwen35Arch) {
        let arch = Qwen35Arch::from_config(cfg.clone());
        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();
        populate_tensors(&mut tensors, &mut vectors, &arch);

        let vocab = cfg.vocab_size.unwrap();
        let hidden = cfg.hidden_size;
        let embed = make_2d(vocab, hidden, 0.5);
        let lm_head = make_2d(vocab, hidden, 0.5);

        let mw = ModelWeights {
            tensors,
            vectors,
            raw_bytes: HashMap::new(),
            packed_mmaps: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: HashMap::new(),
            embed: larql_models::embed::EmbedMatrix::from_array(embed),
            embed_quant: None,
            lm_head,
            lm_head_quant: None,
            position_embed: None,
            quant_tensors: HashMap::new(),
            arch: Box::new(Qwen35Arch::from_config(cfg.clone())),
            num_layers: cfg.num_layers,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            vocab_size: vocab,
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
        };
        (mw, arch)
    }

    #[test]
    fn bridge_loads_hybrid_layer_kinds_correctly() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load_qwen35_weights");
        assert_eq!(w.layers.len(), cfg.num_layers);
        // Layer 0,2 linear; 1,3 full-attn.
        for (i, layer) in w.layers.iter().enumerate() {
            let is_linear = (i + 1) % 2 != 0;
            match (&layer.block, is_linear) {
                (Qwen35LayerWeights::Linear(_), true) => {}
                (Qwen35LayerWeights::Attention(_), false) => {}
                _ => panic!("layer {} dispatched to wrong branch", i),
            }
        }
    }

    #[test]
    fn bridge_propagates_ffn_dim_and_embed_lm_head() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load");
        assert_eq!(w.ffn_dim, cfg.intermediate_size);
        assert_eq!(w.embed.shape(), &[cfg.vocab_size.unwrap(), cfg.hidden_size]);
        assert_eq!(
            w.lm_head.shape(),
            &[cfg.vocab_size.unwrap(), cfg.hidden_size]
        );
        assert_eq!(w.final_norm.len(), cfg.hidden_size);
    }

    /// Regression test for C.5a — verifies the bridge does NOT
    /// double-apply the `(1+w)` Gemma offset to norms. Empirical
    /// inspection (C.4y) confirmed the GGUF stores norms with
    /// `(1+w)` already baked in, so the bridge should pass the
    /// stored values through verbatim.
    ///
    /// Seeds zero-init delta weights (`0.0`) and verifies the bridge
    /// passes them through as `0.0`. Catches future re-application
    /// of `add_one_to_vec` or similar offset logic.
    #[test]
    fn bridge_passes_norm_weights_through_verbatim() {
        let cfg = qwen35_tiny_config();
        let arch = Qwen35Arch::from_config(cfg.clone());
        let mut tensors = std::collections::HashMap::new();
        let mut vectors = std::collections::HashMap::new();
        populate_tensors(&mut tensors, &mut vectors, &arch);

        // Overwrite every norm-weight vector with zeros so the bridge
        // pass-through is observable.
        for layer in 0..cfg.num_layers {
            let prefix = format!("layers.{layer}.");
            vectors.insert(
                format!("{prefix}input_layernorm.weight"),
                vec![0.0; cfg.hidden_size],
            );
            vectors.insert(
                format!("{prefix}post_attention_norm.weight"),
                vec![0.0; cfg.hidden_size],
            );
            if !arch.is_linear_attention_layer(layer) {
                vectors.insert(
                    format!("{prefix}attn_q_norm.weight"),
                    vec![0.0; cfg.head_dim],
                );
                vectors.insert(
                    format!("{prefix}attn_k_norm.weight"),
                    vec![0.0; cfg.head_dim],
                );
            } else {
                vectors.insert(
                    format!("{prefix}ssm_norm.weight"),
                    vec![0.0; cfg.ssm_state_size.unwrap()],
                );
            }
        }
        vectors.insert("norm.weight".into(), vec![0.0; cfg.hidden_size]);

        let mw = ModelWeights {
            tensors,
            vectors,
            raw_bytes: std::collections::HashMap::new(),
            packed_mmaps: std::collections::HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: std::collections::HashMap::new(),
            embed: larql_models::embed::EmbedMatrix::from_array(make_2d(
                cfg.vocab_size.unwrap(),
                cfg.hidden_size,
                0.5,
            )),
            embed_quant: None,
            lm_head: make_2d(cfg.vocab_size.unwrap(), cfg.hidden_size, 0.5),
            lm_head_quant: None,
            position_embed: None,
            quant_tensors: HashMap::new(),
            arch: Box::new(Qwen35Arch::from_config(cfg.clone())),
            num_layers: cfg.num_layers,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size.unwrap(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
        };

        let w = load_qwen35_weights(&mw, &arch).expect("bridge");

        // All norms should pass through verbatim — stored 0.0 → 0.0.
        for &v in w.final_norm.iter() {
            assert!(v.abs() < 1e-9, "final_norm should pass through; got {v}");
        }
        for (idx, layer) in w.layers.iter().enumerate() {
            for &v in layer.attn_post_norm.iter() {
                assert!(
                    v.abs() < 1e-9,
                    "layer {idx} attn_post_norm should pass through; got {v}"
                );
            }
            match &layer.block {
                Qwen35LayerWeights::Linear(dn) => {
                    for &v in dn.attn_norm.iter() {
                        assert!(
                            v.abs() < 1e-9,
                            "layer {idx} (linear) attn_norm should pass through; got {v}"
                        );
                    }
                    for &v in dn.ssm_norm.iter() {
                        assert!(
                            v.abs() < 1e-9,
                            "layer {idx} (linear) ssm_norm should pass through; got {v}"
                        );
                    }
                }
                Qwen35LayerWeights::Attention(at) => {
                    for &v in at.attn_norm.iter() {
                        assert!(
                            v.abs() < 1e-9,
                            "layer {idx} (attn) attn_norm should pass through; got {v}"
                        );
                    }
                    for &v in at.attn_q_norm.iter() {
                        assert!(
                            v.abs() < 1e-9,
                            "layer {idx} (attn) attn_q_norm should pass through; got {v}"
                        );
                    }
                    for &v in at.attn_k_norm.iter() {
                        assert!(
                            v.abs() < 1e-9,
                            "layer {idx} (attn) attn_k_norm should pass through; got {v}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bridge_missing_tensor_errors_with_key_name() {
        let cfg = qwen35_tiny_config();
        let (mut mw, arch) = build_model_weights(cfg);
        // Drop a known tensor from layer 1 (full-attn): self_attn.q_proj.weight
        mw.tensors.remove("layers.1.self_attn.q_proj.weight");
        match load_qwen35_weights(&mw, &arch) {
            Err(Qwen35LoadError::MissingTensor(k)) => {
                assert!(k.contains("layers.1.self_attn.q_proj"), "got {k}")
            }
            Err(other) => panic!("expected MissingTensor, got {other:?}"),
            Ok(_) => panic!("expected MissingTensor, got Ok"),
        }
    }

    #[test]
    fn bridge_missing_vector_errors_with_key_name() {
        let cfg = qwen35_tiny_config();
        let (mut mw, arch) = build_model_weights(cfg);
        mw.vectors.remove("layers.0.ssm_a");
        match load_qwen35_weights(&mw, &arch) {
            Err(Qwen35LoadError::MissingVector(k)) => {
                assert_eq!(k, "layers.0.ssm_a")
            }
            Err(other) => panic!("expected MissingVector, got {other:?}"),
            Ok(_) => panic!("expected MissingVector, got Ok"),
        }
    }

    #[test]
    fn bridge_rejects_non_hybrid_arch() {
        // A Qwen35Arch built without full_attention_interval / ssm_state_size
        // should be flagged before any tensor lookup happens.
        let cfg = non_hybrid_config();
        let arch = Qwen35Arch::from_config(cfg.clone());
        let mw = ModelWeights {
            tensors: HashMap::new(),
            vectors: HashMap::new(),
            raw_bytes: HashMap::new(),
            packed_mmaps: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: HashMap::new(),
            embed: larql_models::embed::EmbedMatrix::from_array(make_2d(1, 1, 0.0)),
            embed_quant: None,
            lm_head: make_2d(1, 1, 0.0),
            lm_head_quant: None,
            position_embed: None,
            quant_tensors: HashMap::new(),
            arch: Box::new(Qwen35Arch::from_config(cfg)),
            num_layers: 0,
            hidden_size: 0,
            intermediate_size: 0,
            vocab_size: 0,
            head_dim: 0,
            num_q_heads: 0,
            num_kv_heads: 0,
            rope_base: 0.0,
        };
        match load_qwen35_weights(&mw, &arch) {
            Err(Qwen35LoadError::NotHybrid { field }) => {
                assert_eq!(field, "full_attention_interval")
            }
            Err(other) => panic!("expected NotHybrid, got {other:?}"),
            Ok(_) => panic!("expected NotHybrid, got Ok"),
        }
    }

    // ── C.4e: end-to-end bridge → forward integration ──
    //
    // These tests feed the typed `Qwen35Weights` produced by
    // `load_qwen35_weights` into `qwen35_forward_step` to verify
    // the bridge produces structurally compatible output. They
    // exercise both layer kinds in one forward call (the tiny
    // config has 2 linear + 2 attention layers).

    fn integration_dn_dims(cfg: &ModelConfig) -> crate::attention::deltanet_block::DeltaNetDims {
        crate::attention::deltanet_block::DeltaNetDims {
            hidden: cfg.hidden_size,
            head_v_dim: cfg.ssm_state_size.unwrap(),
            n_v_heads: cfg.ssm_dt_rank.unwrap(),
            n_k_heads: cfg.ssm_group_count.unwrap(),
            d_conv: cfg.ssm_conv_kernel.unwrap(),
            eps: 1e-6,
            block_gqa: true,
        }
    }

    fn integration_attn_dims(
        cfg: &ModelConfig,
    ) -> crate::attention::qwen35_block::Qwen35AttentionDims {
        crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: cfg.hidden_size,
            n_head: cfg.num_q_heads,
            n_head_kv: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.head_dim,
            rope_base: cfg.rope_base,
            eps: 1e-6,
        }
    }

    fn layer_kinds(arch: &Qwen35Arch) -> Vec<bool> {
        (0..arch.config().num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect()
    }

    #[test]
    fn integration_bridge_forward_one_token_returns_finite_vocab_logits() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load");

        let dn_dims = integration_dn_dims(&cfg);
        let attn_dims = integration_attn_dims(&cfg);
        let kinds = layer_kinds(&arch);
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let logits = crate::attention::qwen35_forward::qwen35_forward_step(
            3, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        assert_eq!(logits.len(), cfg.vocab_size.unwrap());
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must all be finite"
        );
        assert_eq!(cache.next_position, 1);
    }

    #[test]
    fn integration_bridge_forward_grows_kv_and_advances_position_over_two_tokens() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load");

        let dn_dims = integration_dn_dims(&cfg);
        let attn_dims = integration_attn_dims(&cfg);
        let kinds = layer_kinds(&arch);
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let _ = crate::attention::qwen35_forward::qwen35_forward_step(
            1, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        let _ = crate::attention::qwen35_forward::qwen35_forward_step(
            2, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        assert_eq!(cache.next_position, 2);
        // Every full-attn layer's KV slab should have grown to 2 rows.
        for (layer_idx, kv) in cache.kv_layers.iter().enumerate() {
            let is_linear = (layer_idx + 1) % 2 != 0;
            match (kv, is_linear) {
                (None, true) => {} // linear layer — no KV slab
                (Some((k, v)), false) => {
                    assert_eq!(
                        k.shape()[0],
                        2,
                        "full-attn layer {layer_idx} K should have 2 rows"
                    );
                    assert_eq!(
                        v.shape()[0],
                        2,
                        "full-attn layer {layer_idx} V should have 2 rows"
                    );
                }
                _ => panic!("layer {layer_idx}: KV slot kind mismatch"),
            }
        }
    }

    #[test]
    fn integration_bridge_forward_dn_state_carries_across_two_tokens() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load");

        let dn_dims = integration_dn_dims(&cfg);
        let attn_dims = integration_attn_dims(&cfg);
        let kinds = layer_kinds(&arch);
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let _ = crate::attention::qwen35_forward::qwen35_forward_step(
            5, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        // The first linear layer's recurrent state should be off zero.
        let mass_layer0_after_1: f32 = cache.dn_state.layers[0]
            .as_ref()
            .expect("linear layer 0 has dn_state")
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_layer0_after_1 > 0.0,
            "layer 0 recurrent_state should be non-zero after one step"
        );

        // Layer 1 (full-attn) should have no dn_state slot.
        assert!(cache.dn_state.layers[1].is_none());

        let _ = crate::attention::qwen35_forward::qwen35_forward_step(
            7, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        let mass_layer0_after_2: f32 = cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_layer0_after_2 > 0.0,
            "layer 0 recurrent_state still non-zero after second step"
        );
        assert!(
            mass_layer0_after_2.is_finite(),
            "recurrent_state must remain finite"
        );
    }

    // ── C.4f: real-GGUF smoke test ──
    //
    // Gated on `LARQL_QWEN35_GGUF=/path/to/Qwen3.6-*.gguf`. If the
    // env var is unset, the test is a no-op. Loads the GGUF, builds
    // the Qwen35Arch / Qwen35MoeArch from its metadata, runs the
    // bridge, and asserts every layer's weights are present with
    // the expected shapes. Does NOT run the forward (which would
    // need ~50 GB working memory after Q*K dequant); just verifies
    // every key the bridge expects is in the loaded `ModelWeights`.
    #[test]
    fn real_gguf_qwen35_bridge_smoke() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping real-GGUF smoke test");
                return;
            }
        };
        let weights = larql_models::load_gguf(std::path::Path::new(&path))
            .expect("load_gguf failed on Qwen3.6 GGUF");
        let arch_family = weights.arch.family().to_string();
        assert!(
            matches!(arch_family.as_str(), "qwen35" | "qwen35moe"),
            "GGUF arch must be qwen35 or qwen35moe, got `{arch_family}`"
        );

        let w = load_qwen35_weights(&weights, &*weights.arch).expect("load_qwen35_weights failed");
        assert_eq!(w.layers.len(), weights.num_layers);
        assert_eq!(w.embed.shape()[1], weights.hidden_size);
        assert_eq!(w.lm_head.shape()[1], weights.hidden_size);
        assert_eq!(w.final_norm.len(), weights.hidden_size);

        // Spot-check the first linear and first full-attn layer's
        // tensor shapes.
        let conv_kernel = weights.arch.ssm_conv_kernel();
        let head_v_dim = weights.arch.ssm_state_size();
        let n_v_heads = weights.arch.ssm_dt_rank();
        let n_k_heads = weights.arch.ssm_group_count();
        let value_dim = head_v_dim * n_v_heads;
        let key_dim = head_v_dim * n_k_heads;
        let conv_dim = 2 * key_dim + value_dim;

        for (idx, layer) in w.layers.iter().enumerate() {
            match &layer.block {
                Qwen35LayerWeights::Linear(dn) => {
                    assert_eq!(
                        dn.ssm_conv1d.shape(),
                        &[conv_kernel, conv_dim],
                        "layer {idx} ssm_conv1d shape (after bridge transpose)"
                    );
                    assert_eq!(dn.ssm_a.len(), n_v_heads, "layer {idx} ssm_a");
                    assert_eq!(dn.ssm_dt.len(), n_v_heads, "layer {idx} ssm_dt");
                    assert_eq!(dn.ssm_norm.len(), head_v_dim, "layer {idx} ssm_norm");
                    assert_eq!(
                        dn.attn_qkv.shape(),
                        &[conv_dim, weights.hidden_size],
                        "layer {idx} attn_qkv"
                    );
                }
                Qwen35LayerWeights::Attention(at) => {
                    let head_dim = weights.head_dim;
                    assert_eq!(
                        at.attn_q_norm.len(),
                        head_dim,
                        "layer {idx} attn_q_norm len"
                    );
                    assert_eq!(
                        at.attn_k_norm.len(),
                        head_dim,
                        "layer {idx} attn_k_norm len"
                    );
                    assert_eq!(
                        at.attn_q.shape(),
                        &[2 * weights.num_q_heads * head_dim, weights.hidden_size],
                        "layer {idx} attn_q (fused Q+gate)"
                    );
                }
            }
            assert_eq!(
                layer.attn_post_norm.len(),
                weights.hidden_size,
                "layer {idx} attn_post_norm"
            );
        }
    }

    // ── C.4g: real-GGUF single-token forward smoke ──
    //
    // Gated on the same `LARQL_QWEN35_GGUF` env var as the bridge
    // smoke test. Loads the GGUF, builds the bridge, allocates a
    // hybrid cache, then runs `qwen35_forward_step` for one token.
    // Asserts the logits are finite, shape matches vocab, and that
    // the cache position advanced. Verifies the entire C.1 → C.4f
    // stack composes against real Q4_K_S weights without panicking,
    // NaN-ing, or shape-mismatching.
    //
    // Out-of-scope: matching llama.cpp's logits or argmax (that's
    // C.5 parity oracle). Here we only assert the forward runs to
    // completion and produces a finite distribution.
    #[test]
    fn real_gguf_qwen35_forward_one_token_smoke() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping real-GGUF forward smoke test");
                return;
            }
        };
        let weights = larql_models::load_gguf(std::path::Path::new(&path))
            .expect("load_gguf failed on Qwen3.6 GGUF");
        let w = load_qwen35_weights(&weights, &*weights.arch).expect("load_qwen35_weights failed");

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            block_gqa: matches!(arch.family(), "qwen3next"),
            eps: 1e-6,
        };
        // For Qwen3-Next / Qwen3.6 MoE both ship MRoPE with
        // `rope_dimension_sections`, but the section sum is in PAIR
        // units (number of (i, i+n_dims/2) pairs assigned to each
        // position channel). The total rotated dim count is
        // `rope.dimension_count` (== `partial_rotary_factor * head_dim`).
        // Prefer that explicit count so 35B-A3B (sections [11,11,10,0]
        // sum=32, but rotated dim=64) routes correctly. Falls back to
        // sum(sections) when no partial_rotary_factor is set (e.g.
        // pre-Qwen3-Next configs), then to head_dim for plain RoPE.
        let rotary_dim: usize = arch
            .config()
            .partial_rotary_factor
            .map(|f| ((weights.head_dim as f64) * f).round() as usize)
            .filter(|d| *d > 0)
            .or_else(|| {
                arch.rope_dimension_sections()
                    .map(|s| s.iter().sum::<usize>())
            })
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let token: u32 = 0; // any in-vocab token works for shape/finite check
        let t = std::time::Instant::now();
        let logits = crate::attention::qwen35_forward::qwen35_forward_step(
            token, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        eprintln!("forward(one token) took {:?}", t.elapsed());

        let vocab = weights.embed.shape()[0];
        assert_eq!(logits.len(), vocab, "logits vocab dim");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite (sample: {:?})",
            &logits.as_slice().unwrap()[..5.min(logits.len())]
        );
        assert_eq!(cache.next_position, 1);

        // Spot-check: at least one logit should be non-zero (a
        // uniform-zero output would indicate a silent zero-out
        // somewhere in the stack).
        let max_abs = logits.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        assert!(
            max_abs > 0.0,
            "max |logit| should be > 0; got {max_abs} (uniform zero output)"
        );
    }

    // ── C.4h: real-GGUF multi-token argmax decode ──
    //
    // Same gating as C.4f/C.4g. Generates `N_DECODE_STEPS` tokens
    // autoregressively by argmax sampling from token 0. Verifies
    // the cache / RoPE positions / DeltaNet state work across
    // iterations. Asserts:
    //
    // - Every forward returns finite logits with vocab shape.
    // - `cache.next_position` advances monotonically (1, 2, 3, …).
    // - The generated sequence has at least some variation
    //   (≥ 2 distinct tokens). A model collapsing to a fixed-point
    //   token would indicate a broken cache or numerical drift,
    //   not a meaningful inference path.
    //
    // Out of scope: any check vs llama.cpp's argmax (Phase C.5).
    #[test]
    fn real_gguf_qwen35_multi_token_argmax_decode() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping multi-token decode test");
                return;
            }
        };
        const N_DECODE_STEPS: usize = 4;

        let weights = larql_models::load_gguf(std::path::Path::new(&path))
            .expect("load_gguf failed on Qwen3.6 GGUF");
        let w = load_qwen35_weights(&weights, &*weights.arch).expect("load_qwen35_weights failed");

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            block_gqa: matches!(arch.family(), "qwen3next"),
            eps: 1e-6,
        };
        // For Qwen3-Next / Qwen3.6 MoE both ship MRoPE with
        // `rope_dimension_sections`, but the section sum is in PAIR
        // units (number of (i, i+n_dims/2) pairs assigned to each
        // position channel). The total rotated dim count is
        // `rope.dimension_count` (== `partial_rotary_factor * head_dim`).
        // Prefer that explicit count so 35B-A3B (sections [11,11,10,0]
        // sum=32, but rotated dim=64) routes correctly. Falls back to
        // sum(sections) when no partial_rotary_factor is set (e.g.
        // pre-Qwen3-Next configs), then to head_dim for plain RoPE.
        let rotary_dim: usize = arch
            .config()
            .partial_rotary_factor
            .map(|f| ((weights.head_dim as f64) * f).round() as usize)
            .filter(|d| *d > 0)
            .or_else(|| {
                arch.rope_dimension_sections()
                    .map(|s| s.iter().sum::<usize>())
            })
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let vocab = weights.embed.shape()[0];
        let mut token: u32 = 0;
        let mut generated: Vec<u32> = Vec::with_capacity(N_DECODE_STEPS);
        let mut step_logits: Vec<ndarray::Array1<f32>> = Vec::with_capacity(N_DECODE_STEPS);

        for step in 0..N_DECODE_STEPS {
            let t = std::time::Instant::now();
            let logits = crate::attention::qwen35_forward::qwen35_forward_step(
                token, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
            );
            eprintln!("step {step}: token={token} forward took {:?}", t.elapsed());
            assert_eq!(logits.len(), vocab, "step {step} logits vocab dim");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "step {step}: all logits must be finite"
            );
            assert_eq!(
                cache.next_position,
                step + 1,
                "position should advance monotonically"
            );

            // Diagnostic: print top-3 tokens by logit to see if the
            // distribution shifts even when argmax doesn't.
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("  top-3: {:?}", &indexed[..3.min(indexed.len())]);

            // argmax sampling.
            let next_id = indexed[0].0;
            generated.push(next_id as u32);
            step_logits.push(logits);
            token = next_id as u32;
        }

        eprintln!("generated tokens: {generated:?}");

        // The strong correctness signal: the full logit vectors at
        // different positions must NOT be (near-)identical. A 1.0
        // cosine similarity between step 0 and step N's logits
        // would mean the new K/V rows, new DeltaNet state, and
        // advanced RoPE phase produced no change in the output —
        // a smoking gun for cache or state-update bugs.
        //
        // We don't assert argmax variation here: a model can have
        // a clean fixed point at one boilerplate token under
        // degenerate input (token 0 is a common special id), but
        // the *distribution* across the rest of vocab should still
        // shift as context grows.
        if N_DECODE_STEPS >= 2 {
            let cos = cosine_similarity(&step_logits[0], &step_logits[N_DECODE_STEPS - 1]);
            eprintln!(
                "cosine(step 0 logits, step {} logits) = {cos:.6}",
                N_DECODE_STEPS - 1
            );
            assert!(
                cos < 0.9999,
                "logits at step 0 and step {} are too similar (cos={cos:.6}); state isn't propagating",
                N_DECODE_STEPS - 1
            );
        }
    }

    fn cosine_similarity(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na * nb)
    }

    // ── C.4i: real text round-trip via sibling tokenizer.json ──
    //
    // Same gating as C.4f–C.4h. Loads the tokenizer that ships
    // alongside the GGUF, encodes a short prompt, prefills the
    // hybrid cache token-by-token, then argmax-decodes a handful
    // of generations. Asserts:
    //
    // - Tokenizer loads.
    // - Encoding the prompt yields at least one token id within
    //   vocab range.
    // - Every prefill + decode step produces finite logits.
    // - The decoded continuation is a non-empty string.
    //
    // Out of scope: any semantic check on the generated text or
    // parity vs llama.cpp's output (Phase C.5). Here we only
    // assert the text→tokens→forward→tokens→text plumbing works
    // end to end against real Qwen3.6 27B Q4_K_S weights.
    #[test]
    fn real_gguf_qwen35_tokenizer_roundtrip() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping tokenizer round-trip");
                return;
            }
        };
        let gguf_path = std::path::PathBuf::from(&path);
        let model_dir = gguf_path
            .parent()
            .expect("LARQL_QWEN35_GGUF must point to a file in a directory");

        let tokenizer = match crate::tokenizer::load_tokenizer(model_dir) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("no tokenizer.json sibling next to GGUF ({e:?}) — skipping");
                return;
            }
        };

        let prompt = "Hello";
        let encoding = tokenizer.encode(prompt, true).expect("encode prompt");
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        eprintln!("prompt {prompt:?} → tokens {prompt_ids:?}");
        assert!(!prompt_ids.is_empty(), "empty prompt tokenization");

        let weights =
            larql_models::load_gguf(&gguf_path).expect("load_gguf failed on Qwen3.6 GGUF");
        let vocab = weights.embed.shape()[0];
        assert!(
            prompt_ids.iter().all(|&id| (id as usize) < vocab),
            "tokenizer produced an out-of-vocab id: {prompt_ids:?}, vocab={vocab}"
        );

        let w = load_qwen35_weights(&weights, &*weights.arch).expect("bridge load");

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            block_gqa: matches!(arch.family(), "qwen3next"),
            eps: 1e-6,
        };
        // For Qwen3-Next / Qwen3.6 MoE both ship MRoPE with
        // `rope_dimension_sections`, but the section sum is in PAIR
        // units (number of (i, i+n_dims/2) pairs assigned to each
        // position channel). The total rotated dim count is
        // `rope.dimension_count` (== `partial_rotary_factor * head_dim`).
        // Prefer that explicit count so 35B-A3B (sections [11,11,10,0]
        // sum=32, but rotated dim=64) routes correctly. Falls back to
        // sum(sections) when no partial_rotary_factor is set (e.g.
        // pre-Qwen3-Next configs), then to head_dim for plain RoPE.
        let rotary_dim: usize = arch
            .config()
            .partial_rotary_factor
            .map(|f| ((weights.head_dim as f64) * f).round() as usize)
            .filter(|d| *d > 0)
            .or_else(|| {
                arch.rope_dimension_sections()
                    .map(|s| s.iter().sum::<usize>())
            })
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        // 1. Prefill: feed every prompt token through the forward,
        //    keeping only the last token's logits.
        let mut last_logits: Option<ndarray::Array1<f32>> = None;
        let total = std::time::Instant::now();
        for (i, &tok) in prompt_ids.iter().enumerate() {
            let t = std::time::Instant::now();
            let logits = crate::attention::qwen35_forward::qwen35_forward_step(
                tok, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
            );
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "prefill step {i} (token={tok}): all logits must be finite"
            );
            eprintln!(
                "prefill step {i}: token={tok} forward took {:?}",
                t.elapsed()
            );
            last_logits = Some(logits);
        }

        // 2. Decode N tokens by argmax from the prefilled cache.
        const N_GEN: usize = 4;
        let mut generated: Vec<u32> = Vec::with_capacity(N_GEN);
        let mut logits = last_logits.expect("at least one prompt token");
        for step in 0..N_GEN {
            let (next_id, _) = logits.iter().enumerate().fold(
                (0_usize, f32::NEG_INFINITY),
                |(best_idx, best_v), (i, &v)| {
                    if v > best_v {
                        (i, v)
                    } else {
                        (best_idx, best_v)
                    }
                },
            );
            generated.push(next_id as u32);
            if step + 1 < N_GEN {
                let t = std::time::Instant::now();
                logits = crate::attention::qwen35_forward::qwen35_forward_step(
                    next_id as u32,
                    &w,
                    &dn_dims,
                    &attn_dims,
                    &mut cache,
                    1e-6,
                );
                eprintln!(
                    "decode step {step}: token={next_id} forward took {:?}",
                    t.elapsed()
                );
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "decode step {step}: all logits must be finite"
                );
            }
        }
        eprintln!("total round-trip time {:?}", total.elapsed());

        // 3. Decode generated tokens back to text.
        let continuation = tokenizer
            .decode(&generated, false)
            .expect("decode generated tokens");
        eprintln!("prompt {prompt:?} + generated {generated:?} → continuation {continuation:?}");

        // The strict end-to-end gate: the decoded continuation should
        // not be empty. Empty would indicate the generator produced
        // only special tokens that the decoder elided, or that
        // tokenizer / id mismatch left us decoding garbage.
        assert!(
            !continuation.is_empty(),
            "decoded continuation is empty; generated ids: {generated:?}"
        );
    }

    // ── C.4j: chat-template prompt forward ──
    //
    // Same gating as C.4f–C.4i. Runs the full pipeline with a
    // Qwen3-style chat-formatted prompt rather than raw text. The
    // expectation: an instruction-tuned model should produce
    // sensible text when given properly formatted input. If output
    // is gibberish for in-distribution input, the C.1–C.4 forward
    // path has a real correctness bug that needs investigation
    // BEFORE moving on to C.5 / Phase D.
    //
    // For Qwen 3.6 the standard chat format is:
    //
    //   <|im_start|>user\n<text><|im_end|>\n<|im_start|>assistant\n
    //
    // The tokenizer recognises `<|im_start|>` (id=248045) and
    // `<|im_end|>` (id=248046, also the EOS) as special tokens,
    // so a plain-string prompt encodes correctly.
    //
    // This test reports — it doesn't strictly assert the output is
    // meaningful (that requires parity vs llama.cpp). What it DOES
    // assert is the same plumbing gates as C.4i: tokens in range,
    // finite logits, non-empty decode.
    //
    // ── GROUND TRUTH (Phase C.4q, captured 2026-05-11) ──
    //
    // llama-cli with `--jinja --single-turn -p "Hi" --temp 0`
    // (greedy sampling, applies the GGUF's chat template) on the
    // same Qwen3.6 27B Q4_K_S GGUF produces:
    //
    //   |- [Start thinking]
    //   Here's a thinking process:
    //
    //   1.  **Analyze User Input**
    //      - User …
    //
    // i.e. real English with Qwen 3.6's `[Start thinking]` reasoning-
    // mode prefix. Our current chat-prompt forward (after PRs #60,
    // #63, #64) produces "和讯volu勇往直前ardeezoemann supposedBUFF"
    // — varied multi-script tokens but nothing resembling the
    // expected English output. That means there's at least one
    // remaining correctness bug after the three layout fixes
    // (DeltaNet reshape, Q+gate interleave, DeltaNet GQA cycle).
    //
    // Candidate areas to investigate next (none verified, all are
    // hypotheses ranked by likelihood):
    //
    // 1. **DeltaNet output flatten layout**: `delta_net_step` returns
    //    Array2 [s_v, h_v] with `o[d, h]` indexing. The naïve
    //    `o.into_iter()` flatten walks row-major → produces
    //    DIM-MAJOR flat (`flat[d * h_v + h]`). ggml's convention is
    //    HEAD-MAJOR (axis 0 inner = head_v_dim, so flat per head is
    //    contiguous). An attempted fix in a C.4p branch
    //    (`o.t().to_owned().into_iter().collect()`) produced
    //    head-major flat but EMPIRICALLY regressed output to a
    //    single Chinese attractor ("勤勤恳恳"×8). That's either
    //    (a) the right fix unmasking a downstream bug, or (b) the
    //    wrong direction. Needs token-level llama.cpp parity to
    //    disambiguate.
    //
    // 2. **`attn_qkv` post-conv split layout**: the Q/K/V slabs may
    //    have a different per-head interleaving than the simple
    //    head-major slice we use.
    //
    // 3. **`attn_post_norm` placement**: maybe it should be
    //    `ffn_in = post_norm(ffn_in)` rather than `ffn_in =
    //    post_norm(residual)`. design.md §6 says post-norm applies
    //    to the residual; double-check against llama.cpp.
    //
    // 4. **`final_norm` epsilon / weight offset**: Gemma-style models
    //    add 1.0 to weight; check whether Qwen 3.6 does too.
    //
    // 5. **Embedding scale / final-logit softcapping**: defaults for
    //    Qwen 3.6 should be 1.0 / None, but worth verifying.
    //
    // The Phase C.5 parity oracle is the right tool to bisect these
    // hypotheses one at a time against llama.cpp's per-layer dumps.
    #[test]
    fn real_gguf_qwen35_chat_prompt_forward() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping chat-prompt forward");
                return;
            }
        };
        let gguf_path = std::path::PathBuf::from(&path);
        let model_dir = gguf_path
            .parent()
            .expect("LARQL_QWEN35_GGUF must point to a file in a directory");
        let tokenizer = match crate::tokenizer::load_tokenizer(model_dir) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("no tokenizer.json sibling next to GGUF ({e:?}) — skipping");
                return;
            }
        };

        let prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, true).expect("encode prompt");
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        eprintln!("prompt has {} tokens: {prompt_ids:?}", prompt_ids.len());

        let weights = larql_models::load_gguf(&gguf_path).expect("load_gguf");
        let vocab = weights.embed.shape()[0];
        assert!(
            prompt_ids.iter().all(|&id| (id as usize) < vocab),
            "tokenizer produced an out-of-vocab id"
        );
        let w = load_qwen35_weights(&weights, &*weights.arch).expect("bridge load");

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            block_gqa: matches!(arch.family(), "qwen3next"),
            eps: 1e-6,
        };
        // For Qwen3-Next / Qwen3.6 MoE both ship MRoPE with
        // `rope_dimension_sections`, but the section sum is in PAIR
        // units (number of (i, i+n_dims/2) pairs assigned to each
        // position channel). The total rotated dim count is
        // `rope.dimension_count` (== `partial_rotary_factor * head_dim`).
        // Prefer that explicit count so 35B-A3B (sections [11,11,10,0]
        // sum=32, but rotated dim=64) routes correctly. Falls back to
        // sum(sections) when no partial_rotary_factor is set (e.g.
        // pre-Qwen3-Next configs), then to head_dim for plain RoPE.
        let rotary_dim: usize = arch
            .config()
            .partial_rotary_factor
            .map(|f| ((weights.head_dim as f64) * f).round() as usize)
            .filter(|d| *d > 0)
            .or_else(|| {
                arch.rope_dimension_sections()
                    .map(|s| s.iter().sum::<usize>())
            })
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        // Prefill.
        let mut last_logits: Option<ndarray::Array1<f32>> = None;
        for (i, &tok) in prompt_ids.iter().enumerate() {
            let t = std::time::Instant::now();
            let logits = crate::attention::qwen35_forward::qwen35_forward_step(
                tok, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
            );
            eprintln!("prefill {i:2}: token={tok:>6} forward {:?}", t.elapsed());
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "prefill {i} (token={tok}): finite logits"
            );
            last_logits = Some(logits);
        }

        // Decode up to N tokens or until EOS.
        const N_GEN: usize = 8;
        const EOS: u32 = 248046; // <|im_end|>
        let mut generated: Vec<u32> = Vec::with_capacity(N_GEN);
        let mut logits = last_logits.expect("at least one prompt token");
        for step in 0..N_GEN {
            let (next_id, top_v) = logits.iter().enumerate().fold(
                (0_usize, f32::NEG_INFINITY),
                |(best_idx, best_v), (i, &v)| {
                    if v > best_v {
                        (i, v)
                    } else {
                        (best_idx, best_v)
                    }
                },
            );
            let next_u32 = next_id as u32;
            generated.push(next_u32);
            // Diagnostic: top-3 at each decode step.
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!(
                "decode {step}: argmax={next_u32} ({top_v:.3}) top-3={:?}",
                &indexed[..3.min(indexed.len())]
            );
            if next_u32 == EOS {
                eprintln!("hit EOS at decode step {step}, stopping");
                break;
            }
            if step + 1 < N_GEN {
                let t = std::time::Instant::now();
                logits = crate::attention::qwen35_forward::qwen35_forward_step(
                    next_u32, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
                );
                eprintln!("decode {step} continuation forward {:?}", t.elapsed());
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "decode {step}: finite logits"
                );
            }
        }

        let continuation = tokenizer
            .decode(&generated, false)
            .expect("decode generated");
        eprintln!(
            "assistant generated {} tokens: {generated:?}",
            generated.len()
        );
        eprintln!("decoded continuation: {continuation:?}");

        // Plumbing gate (same as C.4i).
        assert!(
            !continuation.is_empty(),
            "decoded continuation is empty; generated ids: {generated:?}"
        );
    }

    // ── C.4k: diagnostic — lm_head row norms + embed row norms ──
    //
    // Bisects the C.4j bug. If `lm_head.row(185983)` has a much larger
    // L2 norm than typical rows, the attractor is a loader/quant
    // artifact (most likely a per-row outlier in the Q6_K
    // dequantization of `output.weight`). If row norms are uniform,
    // the bug is downstream in the forward path.
    //
    /// C.5j diagnostic — feed llama.cpp's `result_norm.bin` (token 8's
    /// x_final) into OUR `lm_head` and see if argmax matches theirs.
    /// Isolates the lm_head loading bug from any upstream parity issue.
    /// Expects `/tmp/llama_bin/result_norm.bin` and `result_output.bin`
    /// produced by `llama-eval-callback` with `LLAMA_DUMP_BIN_DIR`.
    #[test]
    fn real_gguf_qwen35_lm_head_vs_their_xfinal_diagnostic() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping");
                return;
            }
        };
        let xfinal_path = "/tmp/llama_bin/result_norm.bin";
        let logits_path = "/tmp/llama_bin/result_output.bin";
        if !std::path::Path::new(xfinal_path).exists() {
            eprintln!("No {xfinal_path} — run llama-eval-callback first");
            return;
        }
        let load_bin = |p: &str| -> Vec<f32> {
            use std::io::Read;
            let mut f = std::fs::File::open(p).expect("open");
            let mut header = [0u8; 32];
            f.read_exact(&mut header).expect("header");
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).expect("read");
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let their_x = load_bin(xfinal_path);
        let their_logits = load_bin(logits_path);
        let weights = larql_models::load_gguf(std::path::Path::new(&path)).expect("load_gguf");
        let lm_head = &weights.lm_head;
        let x = ndarray::Array1::from_vec(their_x);
        let our_logits_from_their_x: ndarray::Array1<f32> = lm_head.dot(&x);

        let our_argmax = our_logits_from_their_x
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let their_argmax = their_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        eprintln!(
            "argmax: (our_lm_head @ their_x) = {}  their_argmax = {}",
            our_argmax, their_argmax
        );
        eprintln!("their logit at our argmax = {}", their_logits[our_argmax]);
        eprintln!(
            "our (our_lm_head @ their_x) logit at their argmax = {}",
            our_logits_from_their_x[their_argmax]
        );
        // Pearson
        let n = their_logits.len();
        let mean_o = our_logits_from_their_x.iter().sum::<f32>() / n as f32;
        let mean_t = their_logits.iter().sum::<f32>() / n as f32;
        let mut num = 0.0f64;
        let mut do2 = 0.0f64;
        let mut dt2 = 0.0f64;
        for i in 0..n {
            let do_ = (our_logits_from_their_x[i] - mean_o) as f64;
            let dt = (their_logits[i] - mean_t) as f64;
            num += do_ * dt;
            do2 += do_ * do_;
            dt2 += dt * dt;
        }
        let pearson = num / (do2.sqrt() * dt2.sqrt());
        eprintln!(
            "pearson(our_lm_head @ their_x_final, their_logits) = {:.6}",
            pearson
        );
        // Check: is lm_head == embed (tied)?
        let embed_cow = weights.embed.as_array();
        let embed: &ndarray::ArcArray2<f32> = &embed_cow;
        eprintln!(
            "lm_head shape: {:?}  embed shape: {:?}",
            lm_head.shape(),
            embed.shape()
        );
        let row_diff = |row: usize| -> f32 {
            let lh = lm_head.row(row);
            let em = embed.row(row);
            lh.iter()
                .zip(em.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
        };
        for row in [0usize, 100, 1000, 145263, 248045, 248068] {
            if row < lm_head.shape()[0] && row < embed.shape()[0] {
                let lh = lm_head.row(row);
                let em = embed.row(row);
                let lh_l2: f32 = lh.iter().map(|v| v * v).sum::<f32>().sqrt();
                let em_l2: f32 = em.iter().map(|v| v * v).sum::<f32>().sqrt();
                eprintln!(
                    "  row {row:>6}: lm_head_l2={lh_l2:.4} embed_l2={em_l2:.4} sum|lh-em|={:.4} first-3: lm=[{:.4},{:.4},{:.4}] em=[{:.4},{:.4},{:.4}]",
                    row_diff(row), lh[0], lh[1], lh[2], em[0], em[1], em[2]
                );
            }
        }
        // Also try using `embed` as lm_head — does that give a better argmax?
        let logits_from_embed: ndarray::Array1<f32> = embed.dot(&x);
        let from_embed_argmax = logits_from_embed
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        eprintln!(
            "argmax (embed @ their_x_final) = {}  (theirs = {}; would be tied-embed case)",
            from_embed_argmax, their_argmax
        );
        // Pearson with embed
        let mean_e = logits_from_embed.iter().sum::<f32>() / n as f32;
        let mut nume = 0.0f64;
        let mut de2 = 0.0f64;
        for i in 0..n {
            let de = (logits_from_embed[i] - mean_e) as f64;
            let dt = (their_logits[i] - mean_t) as f64;
            nume += de * dt;
            de2 += de * de;
        }
        eprintln!(
            "pearson(embed @ their_x_final, their_logits) = {:.6}",
            nume / (de2.sqrt() * dt2.sqrt())
        );
        // Dump row 248068 (llama.cpp's argmax) of our lm_head to a binary file
        // so a Python script can elementwise diff it against gguf-py's dequant.
        if let Ok(p) = std::env::var("LARQL_QWEN35_DUMP_LM_HEAD_ROW") {
            let row_idx = p.parse::<usize>().unwrap_or(248068);
            let row = lm_head.row(row_idx);
            let out = "/tmp/larql_lm_head_row.bin";
            use std::io::Write;
            let mut f = std::fs::File::create(out).expect("create");
            let ne: [i64; 4] = [row.len() as i64, 1, 1, 1];
            for n in ne {
                f.write_all(&n.to_le_bytes()).ok();
            }
            for v in row.iter() {
                f.write_all(&v.to_le_bytes()).ok();
            }
            eprintln!(
                "Wrote our lm_head row {row_idx} ({} f32) to {out}",
                row.len()
            );
        }
    }

    // Same env-gating as the other real-GGUF tests.
    #[test]
    fn real_gguf_qwen35_lm_head_row_norms_diagnostic() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping lm_head row norm diagnostic");
                return;
            }
        };
        let weights = larql_models::load_gguf(std::path::Path::new(&path)).expect("load_gguf");
        let lm_head = &weights.lm_head;
        let embed_cow = weights.embed.as_array();
        let embed: &ndarray::ArcArray2<f32> = &embed_cow;
        let vocab = lm_head.shape()[0];
        let hidden = lm_head.shape()[1];
        eprintln!(
            "lm_head shape: {:?}, embed shape: {:?}",
            lm_head.shape(),
            embed.shape()
        );

        let row_norm = |arr: &ndarray::ArcArray2<f32>, row: usize| -> f32 {
            arr.row(row).iter().map(|&v| v * v).sum::<f32>().sqrt()
        };
        let row_mean_abs = |arr: &ndarray::ArcArray2<f32>, row: usize| -> f32 {
            arr.row(row).iter().map(|&v| v.abs()).sum::<f32>() / hidden as f32
        };

        // Sample some tokens: the C.4j attractor, the second-place
        // attractor, BOS, EOS, IM_START, IM_END, and uniformly spaced
        // indices to characterize the typical row norm.
        let probes: Vec<(u32, &str)> = vec![
            (185983, "C.4j argmax attractor"),
            (240193, "C.4j second-place"),
            (123894, "C.4j third-place"),
            (212496, "C.4i first decode"),
            (248044, "BOS"),
            (248045, "<|im_start|>"),
            (248046, "<|im_end|> / EOS"),
            (248055, "PAD"),
            (0, "token 0"),
            (1, "token 1"),
            (1000, "token 1000"),
            (10000, "token 10000"),
            (100000, "token 100000"),
            (200000, "token 200000"),
            (248319, "last in-vocab"),
        ];

        eprintln!(
            "\n{:<10} {:<30} {:>12} {:>12} {:>12} {:>12}",
            "tok_id", "note", "lm_head_norm", "lm_head_mean", "embed_norm", "embed_mean"
        );
        for (tok, note) in &probes {
            let t = *tok as usize;
            if t >= vocab {
                continue;
            }
            eprintln!(
                "{:<10} {:<30} {:>12.3} {:>12.6} {:>12.3} {:>12.6}",
                tok,
                note,
                row_norm(lm_head, t),
                row_mean_abs(lm_head, t),
                row_norm(embed, t),
                row_mean_abs(embed, t),
            );
        }

        // Population stats: sample every 1024th row of lm_head and
        // embed, compute mean, std, max norm.
        let stride = 1024;
        let lm_norms: Vec<f32> = (0..vocab)
            .step_by(stride)
            .map(|r| row_norm(lm_head, r))
            .collect();
        let em_norms: Vec<f32> = (0..vocab)
            .step_by(stride)
            .map(|r| row_norm(embed, r))
            .collect();
        let stats = |xs: &[f32]| -> (f32, f32, f32, f32) {
            let n = xs.len() as f32;
            let mean = xs.iter().sum::<f32>() / n;
            let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
            let max = xs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let min = xs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            (mean, var.sqrt(), min, max)
        };
        let (lm_mean, lm_std, lm_min, lm_max) = stats(&lm_norms);
        let (em_mean, em_std, em_min, em_max) = stats(&em_norms);
        eprintln!("\npopulation (every {stride}th row, n={}):", lm_norms.len());
        eprintln!(
            "  lm_head_norm  mean={lm_mean:.3} std={lm_std:.3} min={lm_min:.3} max={lm_max:.3}"
        );
        eprintln!(
            "  embed_norm    mean={em_mean:.3} std={em_std:.3} min={em_min:.3} max={em_max:.3}"
        );

        // If the C.4j attractor's lm_head row is > 3 sigma above mean,
        // we've found the smoking gun.
        let attr_norm = row_norm(lm_head, 185983);
        let attr_z = (attr_norm - lm_mean) / lm_std.max(1e-6);
        eprintln!("\n185983 lm_head_norm z-score vs population: {attr_z:.2}");
        if attr_z.abs() > 3.0 {
            eprintln!(
                "🔥 185983 IS A ROW-NORM OUTLIER. Hypothesis: \
                 Q6_K dequant produces a bad row, OR this token's \
                 row was naturally trained as an outlier 'sink'."
            );
        } else {
            eprintln!(
                "✓ 185983 row norm is within {attr_z:.1}σ of population mean. \
                 The attractor is NOT a row-magnitude artifact in lm_head — \
                 the bug is downstream in the forward path."
            );
        }

        // What does the lm_head naturally output if the residual
        // collapsed to a uniform / norm-weight-only direction?
        let final_norm = weights
            .vectors
            .get("norm.weight")
            .expect("final_norm tensor must exist");
        let final_norm_arr = ndarray::Array1::from(final_norm.clone());
        eprintln!(
            "\nfinal_norm shape={}, mean={:.4}, min={:.4}, max={:.4}",
            final_norm.len(),
            final_norm.iter().sum::<f32>() / final_norm.len() as f32,
            final_norm.iter().copied().fold(f32::INFINITY, f32::min),
            final_norm.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        );

        let argmax_top3 = |logits: &ndarray::Array1<f32>| -> Vec<(usize, f32)> {
            let mut idx: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx[..3].to_vec()
        };

        // Probe 1: lm_head @ ones_vector — what the model would output
        // if the residual stream collapsed to a uniform vector.
        let ones = ndarray::Array1::<f32>::ones(hidden);
        let logits_ones = lm_head.dot(&ones);
        eprintln!("lm_head @ ones top-3: {:?}", argmax_top3(&logits_ones));

        // Probe 2: lm_head @ final_norm_weight — the "no-info"
        // baseline. After RMSNorm of a degenerate residual, the output
        // is roughly (residual / |residual|) * final_norm_weight, so
        // a totally degenerate residual produces logits ≈
        // lm_head @ final_norm_weight (modulo sign).
        let logits_fn = lm_head.dot(&final_norm_arr);
        eprintln!(
            "lm_head @ final_norm_weight top-3: {:?}",
            argmax_top3(&logits_fn)
        );

        // Probe 3: lm_head @ embed_row(token=1) — what would the
        // model output if NO layers ran (residual = embedding only)?
        // If argmax matches 185983 → forward IS computing something
        // but it's converging to the same answer the embedding-only
        // path produces. If argmax differs from 185983 → forward
        // breaks the embedding's natural signal somewhere.
        let embed_row = embed.row(1).to_owned();
        let logits_embed_only = lm_head.dot(&embed_row);
        eprintln!(
            "lm_head @ embed(token=1) top-3: {:?}",
            argmax_top3(&logits_embed_only)
        );

        // Probe 4: argmax of (lm_head @ final_norm_weight) — does it
        // happen to be 185983? If yes, that means our forward path
        // is collapsing the residual to ~zero/uniform, and the model
        // is then outputting the "no-info" token, which IS 185983.
        let baseline_argmax = logits_fn
            .iter()
            .enumerate()
            .fold((0_usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        eprintln!("\nbaseline argmax (lm_head @ final_norm_weight) = {baseline_argmax}");
        if baseline_argmax == 185983 {
            eprintln!(
                "🔥 SMOKING GUN: 185983 is the model's 'no-information' baseline output. \
                 This means our forward path is collapsing the residual stream — \
                 the rest of the model (attention/DeltaNet/FFN) is being added to ~0 \
                 or otherwise normalized away to the bias-only output."
            );
        } else {
            eprintln!(
                "Baseline argmax differs from 185983 ({baseline_argmax} ≠ 185983). \
                 The attractor is NOT just the model's no-information default — \
                 something IS being computed by the forward, but it's converging \
                 to 185983 for our inputs."
            );
        }
    }

    // ── C.4m: temperature-sampled chat-prompt generation ──
    //
    // Same env-gating as the other real-GGUF tests. Replaces the
    // greedy-argmax sampling from C.4j with temperature-scaled
    // multinomial sampling at temperature=0.7, seeded from a fixed
    // u64 for reproducibility.
    //
    // If output diversifies into varied tokens (different argmax
    // choices, distinct decoded characters per step), the 107315
    // attractor from C.4j was a greedy-sampling artifact and the
    // forward path is doing real inference. If output remains
    // tightly repetitive even with temperature, there's likely
    // still a correctness bug.
    //
    // Plumbing-only assertions (same as C.4i/C.4j): tokens in
    // range, finite logits, non-empty decode.
    #[test]
    fn real_gguf_qwen35_chat_prompt_temperature_sampling() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping temperature sampling test");
                return;
            }
        };
        let gguf_path = std::path::PathBuf::from(&path);
        let model_dir = gguf_path
            .parent()
            .expect("LARQL_QWEN35_GGUF must point to a file in a directory");
        let tokenizer = match crate::tokenizer::load_tokenizer(model_dir) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("no tokenizer.json sibling next to GGUF ({e:?}) — skipping");
                return;
            }
        };

        let prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, true).expect("encode prompt");
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        eprintln!("prompt has {} tokens: {prompt_ids:?}", prompt_ids.len());

        let weights = larql_models::load_gguf(&gguf_path).expect("load_gguf");
        let vocab = weights.embed.shape()[0];
        let w = load_qwen35_weights(&weights, &*weights.arch).expect("bridge load");

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            block_gqa: matches!(arch.family(), "qwen3next"),
            eps: 1e-6,
        };
        // For Qwen3-Next / Qwen3.6 MoE both ship MRoPE with
        // `rope_dimension_sections`, but the section sum is in PAIR
        // units (number of (i, i+n_dims/2) pairs assigned to each
        // position channel). The total rotated dim count is
        // `rope.dimension_count` (== `partial_rotary_factor * head_dim`).
        // Prefer that explicit count so 35B-A3B (sections [11,11,10,0]
        // sum=32, but rotated dim=64) routes correctly. Falls back to
        // sum(sections) when no partial_rotary_factor is set (e.g.
        // pre-Qwen3-Next configs), then to head_dim for plain RoPE.
        let rotary_dim: usize = arch
            .config()
            .partial_rotary_factor
            .map(|f| ((weights.head_dim as f64) * f).round() as usize)
            .filter(|d| *d > 0)
            .or_else(|| {
                arch.rope_dimension_sections()
                    .map(|s| s.iter().sum::<usize>())
            })
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        // Prefill.
        let mut last_logits: Option<ndarray::Array1<f32>> = None;
        for (i, &tok) in prompt_ids.iter().enumerate() {
            let logits = crate::attention::qwen35_forward::qwen35_forward_step(
                tok, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
            );
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "prefill {i}: finite logits"
            );
            last_logits = Some(logits);
        }

        // Top-k temperature sampling with fixed seed.
        const N_GEN: usize = 12;
        const TEMPERATURE: f32 = 0.7;
        const TOP_K: usize = 50;
        const EOS: u32 = 248046;
        // Linear congruential PRNG seeded fixed for reproducibility.
        let mut rng_state: u64 = 0x_dead_beef_cafe_1234;
        let mut rand_unit = || -> f32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // High bits as f32 in [0, 1).
            ((rng_state >> 33) as u32 as f32) / (u32::MAX as f32 + 1.0)
        };

        let sample_top_k = |logits: &ndarray::Array1<f32>, rand: f32| -> u32 {
            // Indexed sort by logit desc, take TOP_K.
            let mut idx: Vec<(usize, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v / TEMPERATURE))
                .collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(TOP_K);
            let max_l = idx[0].1;
            let exps: Vec<f32> = idx.iter().map(|&(_, l)| (l - max_l).exp()).collect();
            let total: f32 = exps.iter().sum();
            let mut cum = 0.0_f32;
            let threshold = rand * total;
            for (i, e) in exps.iter().enumerate() {
                cum += e;
                if cum >= threshold {
                    return idx[i].0 as u32;
                }
            }
            idx[idx.len() - 1].0 as u32
        };

        let mut generated: Vec<u32> = Vec::with_capacity(N_GEN);
        let mut logits = last_logits.expect("at least one prompt token");
        for step in 0..N_GEN {
            let next = sample_top_k(&logits, rand_unit());
            eprintln!(
                "decode {step}: sampled={next} (token decode: {:?})",
                tokenizer.decode(&[next], false).unwrap_or_default()
            );
            generated.push(next);
            if next == EOS {
                break;
            }
            if (next as usize) >= vocab {
                panic!("sampled out-of-vocab id {next}");
            }
            if step + 1 < N_GEN {
                logits = crate::attention::qwen35_forward::qwen35_forward_step(
                    next, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
                );
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "decode {step}: finite logits"
                );
            }
        }

        let continuation = tokenizer
            .decode(&generated, false)
            .expect("decode generated");
        eprintln!("decoded continuation (T={TEMPERATURE}, top-{TOP_K}): {continuation:?}");

        // Cross-step uniqueness: temperature sampling should produce
        // ≥ 4 distinct tokens across 12 decoded steps. If it doesn't,
        // the distribution is too tight around one token even after
        // temperature scaling — indicating either Q4 quant collapse
        // or a remaining correctness bug.
        let mut uniq = generated.clone();
        uniq.sort();
        uniq.dedup();
        eprintln!(
            "generated {} tokens, {} distinct: {generated:?}",
            generated.len(),
            uniq.len()
        );
        assert!(
            !continuation.is_empty(),
            "decoded continuation must not be empty"
        );
        // Soft assertion (warn, don't fail): we want to SEE this number.
        // Strict fail at <2 unique tokens — at that point the
        // distribution is essentially a one-hot which would be a
        // separate, stronger bug signal.
        assert!(
            uniq.len() >= 2,
            "temperature sampling produced <2 distinct tokens: {generated:?}"
        );
    }

    // ── C.4r: token-level diff vs llama.cpp ground truth ──
    //
    // Greedily decodes 5 tokens from our forward, then for each
    // decoded position prints WHERE the llama.cpp ground-truth
    // first token lands in our logit distribution. Quantifies
    // "how broken is our forward" without needing a full parity
    // oracle.
    //
    // Ground truth from `llama-cli --jinja --single-turn -p "Hi"
    // --temp 0`: `|- [Start thinking]\nHere's a thinking…`. The
    // first generated character is `|`, so the first token is
    // tokenizer("|")[0]. Encode the prefix `|- [Start` and look up
    // where each of those tokens lands in our forward's logits at
    // each decode position.
    //
    // If our argmax matches → forward is correct at that position.
    // If our argmax differs but expected token is in top-10 →
    // forward is roughly right but slightly off (small numerical
    // bug). If expected token has small or negative logit →
    // forward is structurally broken.
    /// Phase A: probe the Qwen3.6-35B-A3B MoE GGUF to discover its
    /// actual tensor layout. The Qwen35MoeArch handler assumes
    /// per-expert `mlp.experts.N.{gate,up,down}_proj.weight` tensors,
    /// but llama.cpp's GGUF convention typically packs them as 3D
    /// `ffn_*_exps.weight` tensors. This test prints the relevant
    /// MoE metadata + expert/router tensor names + shapes so we can
    /// decide which layout to target. Env-gated by
    /// `LARQL_QWEN35_MOE_GGUF=/path/to/Qwen3.6-35B-A3B.gguf`.
    #[test]
    fn probe_qwen35_moe_gguf_layout() {
        let path = match std::env::var("LARQL_QWEN35_MOE_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_MOE_GGUF unset — skipping probe");
                return;
            }
        };
        let gguf = larql_models::loading::gguf::GgufFile::open(std::path::Path::new(&path))
            .expect("open MoE GGUF");
        eprintln!("=== Qwen3.6-35B-A3B GGUF probe ===");
        eprintln!("path: {path}");
        eprintln!("tensors: {}", gguf.tensor_infos.len());
        eprintln!("metadata entries: {}", gguf.metadata.len());

        // Print relevant MoE / arch metadata.
        eprintln!("\n-- metadata (MoE / arch keys) --");
        let mut keys: Vec<&String> = gguf.metadata.keys().collect();
        keys.sort();
        for k in &keys {
            let lower = k.to_lowercase();
            if lower.contains("expert")
                || lower.contains("moe")
                || lower.contains("arch")
                || lower.contains("ffn")
                || lower.contains("intermediate")
                || lower.contains("layer")
                || lower.contains("hidden")
                || lower.contains("vocab")
                || lower.contains("ssm")
            {
                let v = &gguf.metadata[*k];
                eprintln!("  {k} = {v:?}");
            }
        }

        // Print first-layer MoE-relevant tensor names + shapes + types.
        // Filter to layer 0 for brevity.
        eprintln!("\n-- layer 0 MoE / FFN tensors (names + shapes + type) --");
        let mut layer0_tensors: Vec<&larql_models::loading::gguf::GgufTensorInfo> = gguf
            .tensor_infos
            .iter()
            .filter(|t| {
                let n = t.name();
                n.contains("blk.0.") || n.contains("layers.0.")
            })
            .collect();
        layer0_tensors.sort_by_key(|t| t.name().to_string());
        for t in &layer0_tensors {
            eprintln!(
                "  {:60} dims={:?} type={}",
                t.name(),
                t.dims(),
                t.tensor_type()
            );
        }

        // Print any non-layer (top-level) tensor names for embed / lm_head / etc.
        eprintln!("\n-- top-level (non-blk) tensors --");
        for t in &gguf.tensor_infos {
            if !t.name().contains("blk.") && !t.name().contains("layers.") {
                eprintln!(
                    "  {:60} dims={:?} type={}",
                    t.name(),
                    t.dims(),
                    t.tensor_type()
                );
            }
        }
    }

    /// C.5l (task #30) — minimal tok/s bench against the same Qwen3.6
    /// GGUF used for parity. Times prefill and decode separately so the
    /// numbers can be compared with `llama-bench`'s `pp` / `tg` metrics.
    /// Env-gated like the other real-GGUF tests.
    #[test]
    fn real_gguf_qwen35_bench() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping bench");
                return;
            }
        };
        let gguf_path = std::path::PathBuf::from(&path);

        let n_prefill: usize = std::env::var("LARQL_QWEN35_BENCH_PREFILL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let n_decode: usize = std::env::var("LARQL_QWEN35_BENCH_DECODE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);

        let lazy_lm_head = std::env::var("LARQL_QWEN35_LAZY_LM_HEAD").is_ok();
        let lazy_ffn = std::env::var("LARQL_QWEN35_LAZY_FFN").is_ok();
        eprintln!("loading GGUF (lazy_lm_head={lazy_lm_head}, lazy_ffn={lazy_ffn})…");
        let load_t = std::time::Instant::now();
        let weights = if lazy_ffn {
            let probe = larql_models::load_gguf(&gguf_path).expect("probe load");
            let n_layers = probe.num_layers;
            let arch_ref = &*probe.arch;
            let mut lazy_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
            for l in 0..n_layers {
                lazy_keys.insert(arch_ref.ffn_gate_key(l));
                lazy_keys.insert(arch_ref.ffn_up_key(l));
                lazy_keys.insert(arch_ref.ffn_down_key(l));
                // DeltaNet linear-attention projections — the
                // remaining ~25 GiB of dense weights at Phase 2.
                if let Some(k) = arch_ref.attn_qkv_key(l) {
                    lazy_keys.insert(k);
                }
                if let Some(k) = arch_ref.attn_gate_key(l) {
                    lazy_keys.insert(k);
                }
                if let Some(k) = arch_ref.ssm_out_key(l) {
                    lazy_keys.insert(k);
                }
                // Full-attn projections (only populated for the
                // attention layers in the hybrid pattern; absent
                // keys are silently ignored by lazy_tensors).
                lazy_keys.insert(arch_ref.attn_q_key(l));
                lazy_keys.insert(arch_ref.attn_k_key(l));
                lazy_keys.insert(arch_ref.attn_v_key(l));
                lazy_keys.insert(arch_ref.attn_o_key(l));
            }
            if lazy_lm_head {
                lazy_keys.insert("lm_head.weight".to_string());
                lazy_keys.insert("output.weight".to_string());
            }
            // Embed: lazified when `LARQL_QWEN35_LAZY_FFN=1` is on,
            // since it's the biggest single remaining tensor and the
            // user opting into the lazy path almost always wants it.
            lazy_keys.insert(arch_ref.embed_key().to_string());
            // Phase F.3: MoE GGUFs ship router + 3D-packed expert
            // tensors + (optional) shared expert. None of these are
            // covered by `ffn_gate_key` / `ffn_up_key` / `ffn_down_key`
            // (those return the dense-FFN names), so add them
            // explicitly. For dense GGUFs this set stays empty.
            if arch_ref.is_moe() {
                lazy_keys.extend(qwen35_moe_lazy_keys(n_layers));
            }
            drop(probe);
            larql_models::load_gguf_lazy_tensors(&gguf_path, &lazy_keys)
                .expect("load_gguf_lazy_tensors")
        } else if lazy_lm_head {
            larql_models::load_gguf_lazy_lm_head(&gguf_path).expect("load_gguf_lazy_lm_head")
        } else {
            larql_models::load_gguf(&gguf_path).expect("load_gguf")
        };
        let mut w = load_qwen35_weights(&weights, &*weights.arch).expect("bridge load");
        eprintln!("loaded in {:.2}s", load_t.elapsed().as_secs_f64());

        // Phase E.1+: attach GPU backend when LARQL_QWEN35_GPU=1 is
        // set. CudaBackend::new() consumes ~no VRAM at construction;
        // the kernels lazy-compile on first dispatch.
        attach_cuda_backend_if_requested(&mut w);

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            block_gqa: matches!(arch.family(), "qwen3next"),
            eps: 1e-6,
        };
        // For Qwen3-Next / Qwen3.6 MoE both ship MRoPE with
        // `rope_dimension_sections`, but the section sum is in PAIR
        // units (number of (i, i+n_dims/2) pairs assigned to each
        // position channel). The total rotated dim count is
        // `rope.dimension_count` (== `partial_rotary_factor * head_dim`).
        // Prefer that explicit count so 35B-A3B (sections [11,11,10,0]
        // sum=32, but rotated dim=64) routes correctly. Falls back to
        // sum(sections) when no partial_rotary_factor is set (e.g.
        // pre-Qwen3-Next configs), then to head_dim for plain RoPE.
        let rotary_dim: usize = arch
            .config()
            .partial_rotary_factor
            .map(|f| ((weights.head_dim as f64) * f).round() as usize)
            .filter(|d| *d > 0)
            .or_else(|| {
                arch.rope_dimension_sections()
                    .map(|s| s.iter().sum::<usize>())
            })
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        // Use a simple sequence of tokens — content doesn't matter for
        // timing, only count.
        let toks: Vec<u32> = (0..n_prefill).map(|i| 1 + (i as u32 % 100)).collect();

        // Prefill timing.
        let prefill_t = std::time::Instant::now();
        let mut last_logits = ndarray::Array1::<f32>::zeros(1);
        for &tok in &toks {
            last_logits = crate::attention::qwen35_forward::qwen35_forward_step(
                tok, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
            );
        }
        let prefill_secs = prefill_t.elapsed().as_secs_f64();
        eprintln!(
            "prefill: {n_prefill} toks in {prefill_secs:.2}s → {:.2} tok/s",
            n_prefill as f64 / prefill_secs
        );

        // Decode timing: greedy from the prefill's last logits, N steps.
        let decode_t = std::time::Instant::now();
        for _ in 0..n_decode {
            let (next, _) =
                last_logits
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                        if v > acc.1 {
                            (i, v)
                        } else {
                            acc
                        }
                    });
            last_logits = crate::attention::qwen35_forward::qwen35_forward_step(
                next as u32,
                &w,
                &dn_dims,
                &attn_dims,
                &mut cache,
                1e-6,
            );
        }
        let decode_secs = decode_t.elapsed().as_secs_f64();
        eprintln!(
            "decode:  {n_decode} toks in {decode_secs:.2}s → {:.2} tok/s",
            n_decode as f64 / decode_secs
        );

        // Process RSS at this point (Linux only).
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            if let Some(line) = s.lines().find(|l| l.starts_with("VmRSS:")) {
                eprintln!("RSS:     {line}");
            }
        }

        // Phase E.7: VRAM use via nvidia-smi (process-scoped, in MiB).
        // Needed to make larql's value prop measurable against
        // llama.cpp on the VRAM axis. Falls through silently when
        // nvidia-smi isn't on PATH.
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let my_pid = std::process::id();
            for line in stdout.lines() {
                let mut parts = line.split(',').map(|s| s.trim());
                if let (Some(pid_s), Some(mem_s)) = (parts.next(), parts.next()) {
                    if let Ok(pid) = pid_s.parse::<u32>() {
                        if pid == my_pid {
                            eprintln!("VRAM:    {mem_s} MiB (this process, via nvidia-smi)");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn real_gguf_qwen35_token_diff_vs_llama_cpp() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping token-diff test");
                return;
            }
        };
        let gguf_path = std::path::PathBuf::from(&path);
        let model_dir = gguf_path
            .parent()
            .expect("LARQL_QWEN35_GGUF must point to a file in a directory");
        let tokenizer = match crate::tokenizer::load_tokenizer(model_dir) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("no tokenizer.json sibling next to GGUF ({e:?}) — skipping");
                return;
            }
        };

        // Same chat prompt as C.4j.
        let prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        let prompt_ids: Vec<u32> = tokenizer
            .encode(prompt, true)
            .expect("encode prompt")
            .get_ids()
            .to_vec();
        eprintln!("prompt has {} tokens: {prompt_ids:?}", prompt_ids.len());

        // Ground truth for the chat-template prompt above, captured
        // from llama-eval-callback against the same Qwen3.6-27B
        // Q4_K_S GGUF on 2026-05-11 (post C.5j Q6_K + decay-first
        // fixes). Each token is what llama.cpp's greedy decode emits
        // for the same residual stream. Decoded text:
        //   "<think>\n\n</think>\n\nHello"
        // Note: an earlier capture had a non-chat-template completion
        // (`|- [Start thinking]\nHere's a thinking`) — that was a raw
        // text-completion mode, not what a chat-tuned model produces
        // for the `<|im_start|>assistant\n` continuation. The current
        // text reflects the chat-tuned greedy continuation.
        let ground_truth_text = "<think>\n\n</think>\n\nHello";
        let gt_ids: Vec<u32> = tokenizer
            .encode(ground_truth_text, false)
            .expect("encode ground truth")
            .get_ids()
            .to_vec();

        // Optional lazy-quant paths. Both are env-gated and verify
        // that swapping in `QuantTensor::matvec` for FFN and/or
        // lm_head keeps argmax parity with llama.cpp.
        let lazy_lm_head = std::env::var("LARQL_QWEN35_LAZY_LM_HEAD").is_ok();
        let lazy_ffn = std::env::var("LARQL_QWEN35_LAZY_FFN").is_ok();
        eprintln!("ground truth {:?} → tokens {gt_ids:?}", ground_truth_text);

        let weights = if lazy_ffn {
            let probe = larql_models::load_gguf(&gguf_path).expect("probe load");
            let n_layers = probe.num_layers;
            let arch_ref = &*probe.arch;
            let mut lazy_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
            for l in 0..n_layers {
                lazy_keys.insert(arch_ref.ffn_gate_key(l));
                lazy_keys.insert(arch_ref.ffn_up_key(l));
                lazy_keys.insert(arch_ref.ffn_down_key(l));
                // DeltaNet linear-attention projections — the
                // remaining ~25 GiB of dense weights at Phase 2.
                if let Some(k) = arch_ref.attn_qkv_key(l) {
                    lazy_keys.insert(k);
                }
                if let Some(k) = arch_ref.attn_gate_key(l) {
                    lazy_keys.insert(k);
                }
                if let Some(k) = arch_ref.ssm_out_key(l) {
                    lazy_keys.insert(k);
                }
                // Full-attn projections (only populated for the
                // attention layers in the hybrid pattern; absent
                // keys are silently ignored by lazy_tensors).
                lazy_keys.insert(arch_ref.attn_q_key(l));
                lazy_keys.insert(arch_ref.attn_k_key(l));
                lazy_keys.insert(arch_ref.attn_v_key(l));
                lazy_keys.insert(arch_ref.attn_o_key(l));
            }
            if lazy_lm_head {
                lazy_keys.insert("lm_head.weight".to_string());
                lazy_keys.insert("output.weight".to_string());
            }
            // Embed: lazified when `LARQL_QWEN35_LAZY_FFN=1` is on,
            // since it's the biggest single remaining tensor and the
            // user opting into the lazy path almost always wants it.
            lazy_keys.insert(arch_ref.embed_key().to_string());
            // Phase F.3: MoE GGUFs ship router + 3D-packed expert
            // tensors + (optional) shared expert. None of these are
            // covered by `ffn_gate_key` / `ffn_up_key` / `ffn_down_key`
            // (those return the dense-FFN names), so add them
            // explicitly. For dense GGUFs this set stays empty.
            if arch_ref.is_moe() {
                lazy_keys.extend(qwen35_moe_lazy_keys(n_layers));
            }
            drop(probe);
            larql_models::load_gguf_lazy_tensors(&gguf_path, &lazy_keys)
                .expect("load_gguf_lazy_tensors")
        } else if lazy_lm_head {
            larql_models::load_gguf_lazy_lm_head(&gguf_path).expect("load_gguf_lazy_lm_head")
        } else {
            larql_models::load_gguf(&gguf_path).expect("load_gguf")
        };
        let mut w = load_qwen35_weights(&weights, &*weights.arch).expect("bridge load");
        attach_cuda_backend_if_requested(&mut w);

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            block_gqa: matches!(arch.family(), "qwen3next"),
            eps: 1e-6,
        };
        // For Qwen3-Next / Qwen3.6 MoE both ship MRoPE with
        // `rope_dimension_sections`, but the section sum is in PAIR
        // units (number of (i, i+n_dims/2) pairs assigned to each
        // position channel). The total rotated dim count is
        // `rope.dimension_count` (== `partial_rotary_factor * head_dim`).
        // Prefer that explicit count so 35B-A3B (sections [11,11,10,0]
        // sum=32, but rotated dim=64) routes correctly. Falls back to
        // sum(sections) when no partial_rotary_factor is set (e.g.
        // pre-Qwen3-Next configs), then to head_dim for plain RoPE.
        let rotary_dim: usize = arch
            .config()
            .partial_rotary_factor
            .map(|f| ((weights.head_dim as f64) * f).round() as usize)
            .filter(|d| *d > 0)
            .or_else(|| {
                arch.rope_dimension_sections()
                    .map(|s| s.iter().sum::<usize>())
            })
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        // Prefill (no greedy decoding during prefill).
        let mut last_logits: Option<ndarray::Array1<f32>> = None;
        for &tok in &prompt_ids {
            last_logits = Some(crate::attention::qwen35_forward::qwen35_forward_step(
                tok, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
            ));
        }

        // For each ground-truth position, compute argmax + where the
        // expected token lands in our logit distribution.
        let n_compare = gt_ids.len().min(5);
        let mut logits = last_logits.expect("at least one prompt token");
        eprintln!(
            "\n{:>4} {:>10} {:>10} {:>10} {:>10}",
            "step", "gt_id", "gt_logit", "gt_rank", "our_argmax"
        );
        for step in 0..n_compare {
            let expected = gt_ids[step];
            let expected_logit = logits[expected as usize];
            // Rank of expected: how many tokens have a strictly higher logit.
            let rank = logits.iter().filter(|&&v| v > expected_logit).count();
            let (argmax_id, argmax_v) = logits.iter().enumerate().fold(
                (0_usize, f32::NEG_INFINITY),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            );
            eprintln!(
                "{:>4} {:>10} {:>10.3} {:>10} {:>10}  (gt='{}', our='{}')",
                step,
                expected,
                expected_logit,
                rank,
                argmax_id,
                tokenizer
                    .decode(&[expected], false)
                    .unwrap_or_default()
                    .replace('\n', "\\n"),
                tokenizer
                    .decode(&[argmax_id as u32], false)
                    .unwrap_or_default()
                    .replace('\n', "\\n"),
            );
            // Feed argmax forward (NOT the ground-truth token — measures
            // our model's autoregressive trajectory).
            if step + 1 < n_compare {
                logits = crate::attention::qwen35_forward::qwen35_forward_step(
                    argmax_id as u32,
                    &w,
                    &dn_dims,
                    &attn_dims,
                    &mut cache,
                    1e-6,
                );
            }
        }
    }

    // ── C.4s: inspect ssm_a sign convention ──
    //
    // Per llama.cpp `qwen35.cpp:326` comment, the model expects
    // `gate = -A_log.exp() * softplus(alpha + dt)`. If the GGUF
    // stores ssm_a as already-negative (`-A_log.exp()`), my code
    // is correct. If stored as positive `A_log` or `A_log.exp()`,
    // my forward causes state EXPLOSION since
    // `g = exp(positive * positive softplus) > 1`.
    //
    // Just dumps the actual values from layer 0's ssm_a to settle
    // the question.
    #[test]
    fn real_gguf_qwen35_ssm_a_sign_diagnostic() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping ssm_a sign diagnostic");
                return;
            }
        };
        let weights = larql_models::load_gguf(std::path::Path::new(&path)).expect("load_gguf");
        // ssm_a for layer 0 is at key `layers.0.ssm_a` (1-D, n_v_heads = 48).
        let ssm_a = weights
            .vectors
            .get("layers.0.ssm_a")
            .expect("layers.0.ssm_a must exist");
        eprintln!(
            "layers.0.ssm_a (n_v_heads = {}) values: {:?}",
            ssm_a.len(),
            ssm_a
        );
        let min = ssm_a.iter().copied().fold(f32::INFINITY, f32::min);
        let max = ssm_a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = ssm_a.iter().sum::<f32>() / ssm_a.len() as f32;
        eprintln!("min={min:.4} max={max:.4} mean={mean:.4}");
        let n_neg = ssm_a.iter().filter(|&&v| v < 0.0).count();
        let n_pos = ssm_a.iter().filter(|&&v| v > 0.0).count();
        eprintln!(
            "negative: {n_neg}/{}, positive: {n_pos}/{}",
            ssm_a.len(),
            ssm_a.len()
        );
        if n_neg == ssm_a.len() {
            eprintln!(
                "✓ all ssm_a values are negative → matches my forward's assumption \
                 (log_g = ssm_a * softplus(...) ≤ 0 → decay factor in (0, 1])"
            );
        } else if n_pos == ssm_a.len() {
            eprintln!(
                "🔥 all ssm_a values are POSITIVE → my forward causes state explosion. \
                 Need to negate ssm_a OR the model stores `A_log` (positive) and \
                 llama.cpp computes `-A_log.exp()` from it (need to add the negation)."
            );
        } else {
            eprintln!(
                "Mixed signs ({} neg, {} pos) — unusual; investigate further",
                n_neg, n_pos
            );
        }
    }

    /// Diagnostic: dump the raw stored values of `input_layernorm.weight`
    /// for layer 0, and the magnitude of the FINAL value our bridge
    /// produces (after potential `(1+w)` offset).
    ///
    /// llama.cpp's eval-callback shows `attn_norm-0 = MUL(norm-0,
    /// blk.0.attn_norm.weight)` and the result is values ~1.0-2.0
    /// magnitude (similar to norm-0). That implies stored weight is
    /// ≈ 1.0 (not 0.05 as the agent suggested would be the case
    /// for raw zero-init delta storage).
    ///
    /// If our stored attn_norm.weight is ≈ 1.0 → my C.4u
    /// `add_one_to_vec` was WRONG (double-applies offset).
    /// If our stored attn_norm.weight is ≈ 0.05 → C.4u was correct.
    #[test]
    fn real_gguf_qwen35_attn_norm_weight_diagnostic() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping attn_norm weight diagnostic");
                return;
            }
        };
        let weights = larql_models::load_gguf(std::path::Path::new(&path)).expect("load_gguf");
        let attn_norm = weights
            .vectors
            .get("layers.0.input_layernorm.weight")
            .expect("layers.0.input_layernorm.weight must exist");
        eprintln!(
            "layers.0.input_layernorm.weight (n={}) first-10 values: {:?}",
            attn_norm.len(),
            &attn_norm[..10]
        );
        let min = attn_norm.iter().copied().fold(f32::INFINITY, f32::min);
        let max = attn_norm.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = attn_norm.iter().sum::<f32>() / attn_norm.len() as f32;
        let mean_abs = attn_norm.iter().map(|&v| v.abs()).sum::<f32>() / attn_norm.len() as f32;
        eprintln!("min={min:.4} max={max:.4} mean={mean:.4} mean_abs={mean_abs:.4}");
        if mean_abs > 0.5 {
            eprintln!(
                "✓ stored weight is ≈ 1.0 magnitude → GGUF already has (1+w) baked in. \
                 My C.4u `add_one_to_vec` is DOUBLE-APPLYING the offset and should be reverted."
            );
        } else if mean_abs < 0.1 {
            eprintln!(
                "✓ stored weight is ≈ delta magnitude (< 0.1) → GGUF stores raw `w`. \
                 C.4u's `add_one_to_vec` is correct."
            );
        } else {
            eprintln!(
                "Intermediate magnitude — neither cleanly raw-delta nor pre-applied. \
                 Investigate further."
            );
        }
    }
}
