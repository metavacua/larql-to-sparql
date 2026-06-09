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
        let mut keys: Vec<String> = [
            Some(arch.input_layernorm_key(layer)),
            Some(arch.post_attention_layernorm_key(layer)),
            arch.pre_feedforward_layernorm_key(layer),
            arch.post_feedforward_layernorm_key(layer),
            arch.attn_q_norm_key(layer),
            arch.attn_k_norm_key(layer),
            // Qwen 3.6 full-attention layers use per-head RMSNorm
            // weights stored under `attn_{q,k}_norm.weight` (without
            // a `_proj` infix). The arch returns `None` for layers
            // where they don't apply (e.g. linear-attention layers).
            arch.attn_q_per_head_norm_key(layer),
            arch.attn_k_per_head_norm_key(layer),
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
            // MoE shared-expert sigmoid gate (1-D F32 per layer) —
            // `ffn_gate_inp_shexp.weight` on Qwen 3.6 35B-A3B and
            // Qwen3-Coder-Next. The forward applies
            // `sigmoid(x · gate_inp) * shared_swiglu(x)`; without
            // this entry the shared-expert contribution is ungated
            // and over-amplifies the residual.
            arch.shared_expert_gate_inp_key(layer),
        ]
        .into_iter()
        .flatten()
        .collect();

        // DeltaNet linear-attention layers (Qwen 3.6) carry several small
        // tensors that aren't Q4_K candidates — RMSNorm weights and
        // per-head bias / decay vectors. They land here so the matmul-
        // class projections in [`super::deltanet`] stay single-purpose.
        // `ssm_conv1d` is 2-D and is handled separately below alongside
        // the moe_router 2-D case.
        if arch.is_linear_attention_layer(layer) {
            keys.extend(
                [
                    arch.ssm_norm_key(layer),
                    arch.ssm_dt_key(layer),
                    arch.ssm_a_key(layer),
                ]
                .into_iter()
                .flatten(),
            );
        }

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

        // DeltaNet 2-D `ssm_conv1d` kernel — depthwise Conv1D weight
        // [d_conv, conv_dim]. Flattened and stored as f32 since the
        // depthwise stride is too small (4) for Q4_K super-blocks to
        // be worthwhile. Shape recorded as a flat vector; the reader
        // reshapes back at load time using the per-arch `d_conv`.
        if arch.is_linear_attention_layer(layer) {
            if let Some(key) = arch.ssm_conv1d_key(layer) {
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
        }

        // MoE router (2-D `[num_experts, hidden]` projection — flattened
        // and stored as a "vector" entry in norms.bin). Covers both the
        // Gemma 4 26B A4B hybrid-MoE path (`is_hybrid_moe()`) and the
        // qwen35moe PerExpert path (`is_moe()` without hybrid-MoE — the
        // attention side is hybrid, FFN is pure MoE). PackedMxfp4 (e.g.
        // GPT-OSS) is excluded — its router layout differs and the
        // current writer can't emit it.
        if arch.is_moe() && arch.expert_format() != larql_models::ExpertFormat::PackedMxfp4 {
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
        }

        // 1-D MoE norm vectors (Gemma 4 26B A4B hybrid-MoE specific —
        // router_scale, per_expert_scale, router_norm, pre/post
        // experts/ffn norms). qwen35moe doesn't expose these (its arch
        // accessors return None), so the loop body simply skips on
        // None. Kept gated on `is_hybrid_moe` to make the Gemma scope
        // explicit at the call site.
        if arch.is_hybrid_moe() {
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

    // Final model norm (after last layer)
    if let Some(data) = source.get_vector("norm.weight") {
        let bytes = crate::config::dtype::encode_floats(&data, norms_dtype);
        norms_file.write_all(&bytes)?;
        norm_entries.push(WeightEntry {
            key: "norm.weight".into(),
            kind: kind::VECTOR.into(),
            shape: vec![data.len()],
            offset: norms_offset,
            length: bytes.len() as u64,
            file: NORMS_BIN.into(),
        });
        norms_offset += bytes.len() as u64;
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
