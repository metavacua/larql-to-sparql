//! Stage 1c — `shexp_weights_q4k.bin` + manifest.
//!
//! Per-layer shared-expert (always-on) gate / up / down projections
//! for MoE architectures that ship a shared expert alongside the
//! routed top-K. Qwen 3.6 35B-A3B and Qwen3-Coder-Next both have one
//! shared expert per layer (`ffn_{gate,up,down}_shexp.weight`), and
//! the qwen35 hybrid forward sums its output into the per-token MoE
//! FFN residual: `y_moe += swiglu(shexp_gate, shexp_up, shexp_down)(x)`.
//!
//! Why a dedicated writer: the file-level format invariant of
//! `attn_weights_q4k.bin` and `deltanet_weights_q4k.bin` already
//! supports mixed-format manifest entries (LYRW v2 / Q4kManifestEntry),
//! but those writers iterate fixed slot lists keyed off
//! `arch.attn_{q,k,v,o}_key` / `arch.attn_qkv_key` etc. — extending
//! their slot lists to add shexp would couple shexp emission to the
//! layer kind (linear-vs-full-attn) which doesn't actually gate
//! shexp's presence. Carving out a standalone writer keeps the
//! contract clean: one file = all layers × all 3 shexp projections,
//! independent of layer kind.
//!
//! On dense / non-shexp MoE arches (Gemma 4 26B A4B, Qwen 3.6 dense)
//! the writer is a no-op: `arch.shared_expert_gate_key(layer)` returns
//! `None` and nothing gets emitted. The downstream loader treats
//! file-absent as "no shexp" and the qwen35 forward keeps
//! `shexp_*: None` (drops the contribution, same as before this PR).

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::super::manifest::Q4kManifestEntry;
use super::super::write_f32::WeightSource;
use super::{pad_rows_to_block, try_preserve_quant_passthrough, QuantBlockFormat};

pub(super) fn write_shexp_weights_q4k(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
) -> Result<(), VindexError> {
    let arch = source.arch();

    // Quick probe: does this arch advertise shexp at all? If layer 0's
    // shexp keys are all None, skip the file entirely — the downstream
    // loader treats absence as "no shexp".
    if arch.shared_expert_gate_key(0).is_none()
        && arch.shared_expert_up_key(0).is_none()
        && arch.shared_expert_down_key(0).is_none()
    {
        return Ok(());
    }

    let path = dir.join(SHEXP_WEIGHTS_Q4K_BIN);
    let mut file = BufWriter::new(std::fs::File::create(&path)?);
    let mut offset: u64 = 0;
    let mut manifest: Vec<Q4kManifestEntry> = Vec::with_capacity(num_layers * 3);

    for layer in 0..num_layers {
        // Iterate exactly the three projection slots. Each can be
        // `None` (older arch revisions or selective overrides), in
        // which case we silently skip — the manifest reflects only
        // what's emitted, and the consumer keys lookups by name so
        // partial layers are tolerated as long as the loader's
        // `(Some, Some, Some) | (None, None, None)` invariant holds
        // upstream in `load_qwen35_moe_ffn`.
        let slots: [Option<String>; 3] = [
            arch.shared_expert_gate_key(layer),
            arch.shared_expert_up_key(layer),
            arch.shared_expert_down_key(layer),
        ];

        for key_opt in slots {
            let Some(key) = key_opt else { continue };

            // Tier 1: source-preserve passthrough. Picks Q4_K / Q5_K /
            // Q6_K / Q8_0 / MXFP4 from the source's native format and
            // records it in the manifest's per-entry `format` field —
            // no downquantization through f32.
            if let Some((q_bytes, source_format, rows, padded_cols)) =
                try_preserve_quant_passthrough(source, &key)
            {
                file.write_all(&q_bytes)?;
                let length = q_bytes.len() as u64;
                manifest.push(Q4kManifestEntry {
                    key,
                    shape: vec![rows, padded_cols],
                    format: source_format,
                    offset,
                    length,
                });
                offset += length;
                continue;
            }

            // Fallback: source not in `quant_tensors` or its format
            // isn't in the source-preserve allowlist. Dequant to f32,
            // pad to super-block, quantize to Q4_K. Lossy but keeps
            // the loader happy for arches whose shexp is f32 (rare).
            let Some((data, rows, cols)) = source.get_tensor(&key) else {
                continue;
            };
            let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
            let q_bytes = quantize_q4_k(&padded);
            file.write_all(&q_bytes)?;
            let length = q_bytes.len() as u64;
            manifest.push(Q4kManifestEntry {
                key,
                shape: vec![rows, padded_cols],
                format: QuantBlockFormat::Q4K,
                offset,
                length,
            });
            offset += length;
        }
    }

    file.flush()?;
    drop(file);

    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(SHEXP_WEIGHTS_Q4K_MANIFEST_JSON), manifest_json)?;
    Ok(())
}
