//! Stage 4 — `norms.bin` (per-layer norms + MoE router/scales + final
//! model norm).
//!
//! All norm vectors stay f32 — they're small (~2 KB each) and the
//! forward path multiplies activations by them every token, so the
//! Q4_K-style super-block quantisation noise would compound. Returns
//! the running `Vec<WeightEntry>` (a single `weight_manifest.json`
//! references everything written here plus the PLE / lm_head stages).

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::super::write_f32::{kind, WeightEntry, WeightSource};

pub(super) fn write_norms_and_router(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
) -> Result<Vec<WeightEntry>, VindexError> {
    let arch = source.arch();
    let norms_path = dir.join(NORMS_BIN);
    let mut norms_file = BufWriter::new(std::fs::File::create(&norms_path)?);
    let norms_dtype = crate::config::dtype::StorageDtype::F32;
    let mut norms_offset: u64 = 0;
    let mut norm_entries: Vec<WeightEntry> = Vec::new();

    for layer in 0..num_layers {
        let keys: Vec<String> = [
            Some(arch.input_layernorm_key(layer)),
            Some(arch.post_attention_layernorm_key(layer)),
            arch.pre_feedforward_layernorm_key(layer),
            arch.post_feedforward_layernorm_key(layer),
            arch.attn_q_norm_key(layer),
            arch.attn_k_norm_key(layer),
            // Attention projection biases and sinks. Added 2026-07-29 —
            // previously unrequested here, so architectures that declare
            // biases (qwen, gpt2, starcoder2) lost them at extraction
            // while the attention kernels kept asking for them, and
            // GPT-OSS lost biases *and* sinks. See `docs/k3-funnel.md`
            // §4.6.1.
            arch.attn_q_bias_key(layer),
            arch.attn_k_bias_key(layer),
            arch.attn_v_bias_key(layer),
            arch.attn_o_bias_key(layer),
            arch.attn_sinks_key(layer),
            // LayerNorm biases. Added 2026-07-31 for the same reason as the
            // attention biases above: nothing named them, so GPT-2 and
            // StarCoder2 lost the `β` of every `γ·x̂ + β` at extraction while
            // the Metal `layer_norm` shader implemented `+ bias` and always
            // took its no-bias variant. `None` for RMSNorm architectures.
            arch.input_layernorm_bias_key(layer),
            arch.post_attention_layernorm_bias_key(layer),
            // Gemma 4 per-layer scalar multiplier. Stored as a 0-D scalar
            // in safetensors, surfaced through WeightSource as a 1-element
            // vector. The forward path multiplies h by this value after
            // FFN; omitting it silently produced garbage on 31B.
            arch.layer_scalar_key(layer),
            // Gemma 4 E2B per-layer embedding post-norm.
            if arch.has_per_layer_embeddings() {
                arch.post_per_layer_input_norm_key(layer)
            } else {
                None
            },
        ]
        .into_iter()
        .flatten()
        .collect();

        for key in keys {
            if let Some(data) = source.get_vector(&key) {
                let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
                norms_file.write_all(&bytes)?;
                norm_entries.push(WeightEntry {
                    key: key.clone(),
                    kind: kind::VECTOR.into(),
                    shape: vec![data.len()],
                    offset: norms_offset,
                    length: bytes.len() as u64,
                    file: NORMS_BIN.into(),
                });
                norms_offset += bytes.len() as u64;
            }
        }

        // MoE router + norms (hybrid MoE, e.g. Gemma 4 26B A4B).
        // router.proj.weight is 2D [num_experts, hidden] — flatten and store as "vector".
        // All other MoE keys are 1D vectors.
        // Pure MoE (GraniteMoE, OLMoE) needs its router in the vector store
        // too — the forward resolves it via `weights.vectors.get(router_key)`.
        // Gating on hybrid alone wrote router_weights.bin (which only DESCRIBE
        // reads) while leaving the forward's lookup empty, so the expert block
        // silently found no weights. The 1D keys below are Gemma-4-specific
        // and simply resolve to None elsewhere.
        if arch.is_moe() || arch.is_hybrid_moe() {
            // 2D router projection — flatten
            if let Some(key) = arch.moe_router_key(layer) {
                if let Some((data, _, _)) = source.get_tensor(&key) {
                    let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
                    norms_file.write_all(&bytes)?;
                    norm_entries.push(WeightEntry {
                        key: key.clone(),
                        kind: kind::VECTOR.into(),
                        shape: vec![data.len()],
                        offset: norms_offset,
                        length: bytes.len() as u64,
                        file: NORMS_BIN.into(),
                    });
                    norms_offset += bytes.len() as u64;
                }
            }
            // 1D MoE vectors
            let moe_vec_keys: Vec<String> = [
                arch.moe_router_scale_key(layer),
                arch.moe_router_per_expert_scale_key(layer),
                arch.moe_router_norm_key(layer),
                arch.moe_pre_experts_norm_key(layer),
                arch.moe_post_ffn1_norm_key(layer),
                arch.moe_post_experts_norm_key(layer),
                // Outer post-FFN norm used to re-normalise (h1 + h2) before
                // the residual add in hybrid MoE (HF Gemma 4). Distinct from
                // post_ffn1_norm, which is the dense-branch norm.
                arch.moe_post_outer_norm_key(layer),
            ]
            .into_iter()
            .flatten()
            .collect();
            for key in moe_vec_keys {
                if let Some(data) = source.get_vector(&key) {
                    let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
                    norms_file.write_all(&bytes)?;
                    norm_entries.push(WeightEntry {
                        key: key.clone(),
                        kind: kind::VECTOR.into(),
                        shape: vec![data.len()],
                        offset: norms_offset,
                        length: bytes.len() as u64,
                        file: NORMS_BIN.into(),
                    });
                    norms_offset += bytes.len() as u64;
                }
            }
        }
    }

    // Final model norm (after last layer).
    // Final norm, resolved through the accessor rather than a literal.
    // Every reader (`layer_graph::logits`, `predict`, `trace::vocab`,
    // `kquant_forward::cached`) uses `arch.final_norm_key()`, so a
    // hardcoded key here agrees only as long as no architecture
    // overrides it — at which point the writer would store under one
    // name and the reader look for another, and the final norm would
    // vanish silently. Same writer/reader split that lost the
    // attention biases (`docs/k3-funnel.md` §4.6.1).
    let final_norm_key = arch.final_norm_key().to_string();
    if let Some(data) = source.get_vector(&final_norm_key) {
        let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
        norms_file.write_all(&bytes)?;
        norm_entries.push(WeightEntry {
            key: final_norm_key.clone(),
            kind: kind::VECTOR.into(),
            shape: vec![data.len()],
            offset: norms_offset,
            length: bytes.len() as u64,
            file: NORMS_BIN.into(),
        });
        norms_offset += bytes.len() as u64;
    }

    // Final LayerNorm bias — `None` for RMSNorm architectures. Resolved
    // through the accessor rather than a literal, unlike the weight above:
    // an architecture whose final norm is not literally `norm.weight`
    // already loses the weight here, which is its own latent gap.
    if let Some(key) = arch.final_norm_bias_key() {
        if let Some(data) = source.get_vector(&key) {
            let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
            norms_file.write_all(&bytes)?;
            norm_entries.push(WeightEntry {
                key,
                kind: kind::VECTOR.into(),
                shape: vec![data.len()],
                offset: norms_offset,
                length: bytes.len() as u64,
                file: NORMS_BIN.into(),
            });
            norms_offset += bytes.len() as u64;
        }
    }

    // Gemma 4 E2B PLE global projection norm (small vector).
    if arch.has_per_layer_embeddings() {
        if let Some(data) = source.get_vector("per_layer_projection_norm.weight") {
            let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
            norms_file.write_all(&bytes)?;
            norm_entries.push(WeightEntry {
                key: "per_layer_projection_norm.weight".into(),
                kind: kind::VECTOR.into(),
                shape: vec![data.len()],
                offset: norms_offset,
                length: bytes.len() as u64,
                file: NORMS_BIN.into(),
            });
        }
    }
    norms_file.flush()?;
    drop(norms_file);

    Ok(norm_entries)
}
