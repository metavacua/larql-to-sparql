//! Stage 1 — `attn_weights_q4k.bin` + manifest.

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::super::manifest::Q4kManifestEntry;
use super::super::write_f32::WeightSource;
use super::{
    pad_rows_to_block, resolve_v_tensor, try_preserve_quant_passthrough, try_q4k_passthrough,
    QuantBlockFormat,
};

/// Write Q/K/V/O attention projections to `attn_weights_q4k.bin`,
/// emitting a sidecar manifest with per-tensor offsets and formats.
///
/// Q/K/O are Q4_K; V is Q6_K. On layers where V reuses K (Gemma 4 31B
/// global layers), the K bytes go into the V slot so the 4-per-layer
/// indexing stays valid for downstream kernels reading V.
pub(super) fn write_attn_weights_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    let arch = source.arch();
    let attn_path = dir.join(ATTN_WEIGHTS_KQUANT_BIN);
    let mut attn_file = BufWriter::new(std::fs::File::create(&attn_path)?);
    let mut attn_offset: u64 = 0;
    let mut attn_manifest: Vec<Q4kManifestEntry> = Vec::with_capacity(num_layers * 4);

    for layer in 0..num_layers {
<<<<<<< HEAD:crates/larql-vindex/src/format/weights/write_kquant/attn.rs
        callbacks.on_layer_start(COMP_ATTN_KQUANT, layer, num_layers);
=======
        // Hybrid SSM/DeltaNet arches (Qwen 3.6) interleave linear-
        // attention layers between full-attention ones. The linear
        // layers don't expose Q/K/V/O — their weights live in
        // `deltanet_weights_q4k.bin` ([`super::deltanet`]). Skip here
        // so the manifest reflects only the full-attention layers
        // (e.g. 10 of 40 on Qwen 3.6 at `full_attention_interval=4`).
        if arch.is_linear_attention_layer(layer) {
            continue;
        }
        callbacks.on_layer_start(COMP_ATTN_Q4K, layer, num_layers);
>>>>>>> ianblenke/main:crates/larql-vindex/src/format/weights/write_q4k/attn.rs

        // Resolve each tensor. For V, fall back to K when v_shares_k=true or
        // v_proj simply isn't present (global layers on 31B).
        let q_key = arch.attn_q_key(layer);
        let k_key = arch.attn_k_key(layer);
        let v_key = arch.attn_v_key(layer);
        let o_key = arch.attn_o_key(layer);

        let q = source.get_tensor(&q_key);
        let k = source.get_tensor(&k_key);
        let v = resolve_v_tensor(source.get_tensor(&v_key), &k, arch.v_shares_k(layer));
        let o = source.get_tensor(&o_key);

        // Q, K, V, O in that order — use the same key string for V even when
        // the data is K's, so loaders that look up by position still work.
        #[allow(clippy::type_complexity)]
        let slots: [(&str, Option<(Vec<f32>, usize, usize)>); 4] = [
            (q_key.as_str(), q),
            (k_key.as_str(), k),
            (v_key.as_str(), v),
            (o_key.as_str(), o),
        ];

        for (i, (key, tensor)) in slots.iter().enumerate() {
            // V (index 2) gets Q6_K, others get Q4_K.
            let is_v = i == 2;
            let target_format = if is_v {
                QuantBlockFormat::Q6K
            } else {
                QuantBlockFormat::Q4K
            };

            // Tier 1: source already in the requested target format
            // (e.g. Q4_K GGUF → Q4_K vindex). Bytes go through
            // unchanged. Preserves imatrix-aware quantization (PR #195).
            if let Some((q_bytes, rows, padded_cols)) =
                try_q4k_passthrough(source, key, target_format)
            {
                attn_file.write_all(&q_bytes)?;
                let length = q_bytes.len() as u64;
                attn_manifest.push(Q4kManifestEntry {
                    key: key.to_string(),
                    shape: vec![rows, padded_cols],
                    format: target_format,
                    offset: attn_offset,
                    length,
                });
                attn_offset += length;
                continue;
            }

            // Tier 2: source carries the tensor at a higher-precision
            // quant than the requested target (e.g. Q8_0 / Q5_K in
            // Unsloth Q4_K_M GGUFs for attention weights). Preserve the
            // source format instead of downquantizing through f32 —
            // the latter loses ~0.1% per element, compounding through
            // the network. The loader / `QuantTensor::matvec` already
            // dispatch every K-quant family (Q4_K / Q5_K / Q6_K / Q8_0).
            if let Some((q_bytes, source_format, rows, padded_cols)) =
                try_preserve_quant_passthrough(source, key)
            {
                attn_file.write_all(&q_bytes)?;
                let length = q_bytes.len() as u64;
                attn_manifest.push(Q4kManifestEntry {
                    key: key.to_string(),
                    shape: vec![rows, padded_cols],
                    format: source_format,
                    offset: attn_offset,
                    length,
                });
                attn_offset += length;
                continue;
            }
            // Keep `format` available below for the dequant+requant
            // fallback (still Q4_K / Q6_K depending on slot index).
            let format = target_format;

            let (data, rows, cols) = match tensor {
                Some(t) => t.clone(),
                None => continue, // tensor genuinely absent — skip
            };

            // Row-pad to 256 so each row aligns to a super-block boundary.
            // Critical for models with non-256 inner dims (e.g. Gemma 4 26B A4B
            // where the dense intermediate is 2112). `padded_cols` is what the
            // matvec shader must use as `K`; callers also need to zero-pad the
            // input vector to the same width.
            let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
            let q_bytes = if is_v {
                quantize_q6_k(&padded)
            } else {
                quantize_q4_k(&padded)
            };

            attn_file.write_all(&q_bytes)?;
            let length = q_bytes.len() as u64;
            attn_manifest.push(Q4kManifestEntry {
                key: key.to_string(),
                shape: vec![rows, padded_cols],
                format,
                offset: attn_offset,
                length,
            });
            attn_offset += length;
        }

        callbacks.on_layer_done(COMP_ATTN_KQUANT, layer, 0.0);
    }
    attn_file.flush()?;
    drop(attn_file);

    let manifest_json = serde_json::to_string_pretty(&attn_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(ATTN_WEIGHTS_KQUANT_MANIFEST_JSON), manifest_json)?;
    Ok(())
}
