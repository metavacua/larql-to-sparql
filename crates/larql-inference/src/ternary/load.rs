//! Loading a BitNet model from a vindex, and the errors that stop it.
//!
//! Split out of the former single-file `ternary.rs`; [`super`] shows
//! how the pieces compose.

// Siblings resolve through the parent's re-exports.
use super::*;
use larql_compute::cpu::ops::ternary_matvec::BitLinearWeight;

/// Errors surfaced by [`load_bitnet_model`].  Distinct from the
/// underlying VindexError so callers can pattern-match on
/// "tensor missing" cleanly.
#[derive(Debug)]
pub enum BitnetLoadError {
    Vindex(larql_vindex::VindexError),
    Model(larql_models::ModelError),
    NotBitnet(String),
    MissingTensor(String),
    Shape(String),
}

impl std::fmt::Display for BitnetLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitnetLoadError::Vindex(e) => write!(f, "vindex: {e}"),
            BitnetLoadError::Model(e) => write!(f, "model: {e}"),
            BitnetLoadError::NotBitnet(msg) => write!(f, "not a bitnet vindex: {msg}"),
            BitnetLoadError::MissingTensor(name) => {
                write!(f, "missing required tensor: {name}")
            }
            BitnetLoadError::Shape(msg) => write!(f, "shape mismatch: {msg}"),
        }
    }
}

impl std::error::Error for BitnetLoadError {}

impl From<larql_vindex::VindexError> for BitnetLoadError {
    fn from(e: larql_vindex::VindexError) -> Self {
        BitnetLoadError::Vindex(e)
    }
}

impl From<larql_models::ModelError> for BitnetLoadError {
    fn from(e: larql_models::ModelError) -> Self {
        BitnetLoadError::Model(e)
    }
}

/// Load a complete `BitnetModel` from a `--keep-quant` vindex
/// directory.  Reads:
///
///   index.json         — config + bitnet_layout
///   embeddings.bin     — embed table (F16 or F32; auto-detected)
///   norms.bin (etc.)   — per-layer attn_norm / sub_norm / ffn_norm /
///                         sub_norm + output_norm (via the standard
///                         dense loader, with `skip_attn` and
///                         `skip_ffn` so the BitLinear projections
///                         aren't dequantised into f32)
///   bitnet/<...>.i2s   — ternary BitLinear bytes
///   bitnet/scales.f32  — concatenated per-channel scales
///
/// LM head: BitNet b1.58 ties output to `token_embd.weight`
/// (verified on `microsoft/bitnet-b1.58-2B-4T-gguf`); we use the
/// same `embed` matrix as `lm_head` when the dense loader didn't
/// surface a separate `lm_head.weight` (the dense loader's tying
/// logic already handles this).
///
/// # Errors
/// `BitnetLoadError::NotBitnet` when index.json lacks
/// `bitnet_layout`.  Other variants surface tensor-shape /
/// I/O failures from the underlying loaders.
pub fn load_bitnet_model(vindex_path: &std::path::Path) -> Result<BitnetModel, BitnetLoadError> {
    let config = larql_vindex::load_vindex_config(vindex_path)?;
    let layout = config.bitnet_layout.clone().ok_or_else(|| {
        BitnetLoadError::NotBitnet(format!(
            "{} has no bitnet_layout in index.json (rebuild with `--keep-quant`)",
            vindex_path.display()
        ))
    })?;

    // 1. Dense parts: embed, lm_head, per-layer norms, output_norm.
    //
    //    skip_attn + skip_ffn keeps load_model_weights_with_opts from
    //    expanding the I2_S BitLinears into f32 (which would defeat
    //    the entire memory-savings story).  RMSnorm vectors don't
    //    match either pattern so they survive.
    let opts = larql_vindex::LoadWeightsOptions {
        skip_attn: true,
        skip_ffn: true,
        skip_lm_head: false,
        skip_embed: false,
    };
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let dense = larql_vindex::load_model_weights_with_opts(vindex_path, &mut callbacks, opts)?;

    // 2. Ternary BitLinears.
    let bitnet = larql_vindex::extract::bitnet_loader::load_bitnet_weights(vindex_path)?;

    // 3. Embed scale (vocabulary-aware) — the dense loader already
    //    yields `dense.embed` with `embed_scale` applied if relevant;
    //    keep ours at 1.0 because BitNet b1.58 doesn't scale embeddings.
    let embed_scale = 1.0;
    let embed = dense.embed.to_owned();
    let lm_head = dense.lm_head.to_owned();
    let hidden = config.hidden_size;
    let n_layers = config.num_layers;

    // 4. Per-layer norms + BitLinear projections.
    //
    //    Keys are HF-normalised — the GGUF loader rewrites every
    //    tensor name through GGUF_TO_HF_KEY_REPLACEMENTS before it
    //    reaches the vindex, and the keep-quant writer stamps the
    //    same HF names into `bitnet_layout`.  So:
    //      blk.N.attn_norm.weight   -> layers.N.input_layernorm.weight
    //      blk.N.ffn_norm.weight    -> layers.N.post_attention_layernorm.weight
    //      blk.N.attn_sub_norm      -> layers.N.attn_sub_norm.weight (no HF
    //                                  equivalent; only blk.->layers. applies)
    //      blk.N.ffn_sub_norm       -> layers.N.ffn_sub_norm.weight
    //      blk.N.attn_q.weight      -> layers.N.self_attn.q_proj.weight
    //      blk.N.ffn_gate.weight    -> layers.N.mlp.gate_proj.weight  (etc.)
    //    Using GGUF-native keys here was BUG-infer-deadlock bug #5
    //    (loader/vindex namespace mismatch) — it made every lookup miss.
    let inter = config.intermediate_size;
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let attn_norm = take_norm(
            &dense,
            &format!("layers.{i}.input_layernorm.weight"),
            hidden,
        )?;
        let attn_sub_norm = take_norm(&dense, &format!("layers.{i}.attn_sub_norm.weight"), hidden)?;
        let ffn_norm = take_norm(
            &dense,
            &format!("layers.{i}.post_attention_layernorm.weight"),
            hidden,
        )?;
        let ffn_sub_norm = take_norm(&dense, &format!("layers.{i}.ffn_sub_norm.weight"), inter)?;

        let get_bitlinear = |suffix: &str| -> Result<BitLinearWeight, BitnetLoadError> {
            let key = format!("layers.{i}.{suffix}.weight");
            bitnet
                .tensors
                .get(&key)
                .cloned()
                .ok_or(BitnetLoadError::MissingTensor(key))
        };
        let attn_q = get_bitlinear("self_attn.q_proj")?;
        let attn_k = get_bitlinear("self_attn.k_proj")?;
        let attn_v = get_bitlinear("self_attn.v_proj")?;
        let attn_o = get_bitlinear("self_attn.o_proj")?;
        let ffn_gate = get_bitlinear("mlp.gate_proj")?;
        let ffn_up = get_bitlinear("mlp.up_proj")?;
        let ffn_down = get_bitlinear("mlp.down_proj")?;

        let ffn = BitNetFfn {
            gate: ffn_gate,
            up: ffn_up,
            down: ffn_down,
            ffn_norm,
            ffn_sub_norm,
            eps: layout.rms_eps,
        };

        layers.push(BitnetLayer {
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_sub_norm,
            attn_o,
            ffn,
        });
    }

    // 5. output_norm.  GGUF `output_norm.weight` -> HF `norm.weight`.
    let output_norm = dense
        .vectors
        .get("norm.weight")
        .cloned()
        .ok_or_else(|| BitnetLoadError::MissingTensor("norm.weight".into()))?;

    Ok(BitnetModel {
        layers,
        embed,
        embed_scale,
        output_norm,
        lm_head,
        eps: layout.rms_eps,
        head_dim: layout.head_dim.max(1),
        n_q_heads: layout.n_q_heads.max(1),
        n_kv_heads: layout.n_kv_heads.max(1),
        rope_base: if layout.rope_base > 0.0 {
            layout.rope_base
        } else {
            10000.0
        },
    })
}

/// Pluck a 1D norm vector out of `dense.vectors` and validate its
/// length.
fn take_norm(
    dense: &larql_models::ModelWeights,
    key: &str,
    expected_len: usize,
) -> Result<Vec<f32>, BitnetLoadError> {
    let v = dense
        .vectors
        .get(key)
        .cloned()
        .ok_or_else(|| BitnetLoadError::MissingTensor(key.to_string()))?;
    if v.len() != expected_len {
        return Err(BitnetLoadError::Shape(format!(
            "{key}: len {} != expected {}",
            v.len(),
            expected_len
        )));
    }
    Ok(v)
}

// ─────────────────────────────────────────────────────────────────────────────
//  generate_streaming_bitnet — token-by-token callback with detok + EOS
// ─────────────────────────────────────────────────────────────────────────────
