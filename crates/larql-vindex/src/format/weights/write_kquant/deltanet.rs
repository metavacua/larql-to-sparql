//! Stage 1b — `deltanet_weights_q4k.bin` + manifest.
//!
//! Hybrid SSM/DeltaNet architectures (Qwen 3.6 27B dense, Qwen 3.6 35B-A3B
//! MoE) mix full-attention layers with linear-attention (Gated DeltaNet)
//! layers per the `full_attention_interval`. The full-attention layers
//! emit Q/K/V/O via [`super::attn`]; the linear layers carry a different
//! tensor contract that lives here.
//!
//! Per linear-attention layer we emit the Q4_K-quantisable matmul
//! tensors:
//!
//! - `attn_qkv` — fused Q+K+V mixer (`wqkv @ x` in the DeltaNet block).
//! - `attn_gate` — Z-gate projection (`wqkv_gate @ x`).
//! - `ssm_alpha` — α projection (head-wise gating input).
//! - `ssm_beta`  — β projection (head-wise gating input).
//! - `ssm_out`   — output projection back to hidden.
//!
//! All five are 2-D matmul tensors; a tensor is emitted as Q4_K when
//! its column dimension is a multiple of 256 (the super-block size).
//! Anything that fails the alignment check is silently skipped at this
//! tier — those tensors live in `norms.bin` (vectors / scalars / conv1d
//! weights) and are emitted by [`super::norms`].
//!
//! The small / non-matmul DeltaNet tensors (`ssm_conv1d`, `ssm_dt`,
//! `ssm_a`, `ssm_norm`) belong in `norms.bin` and are handled there to
//! keep this writer single-purpose.

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::extract::stage_labels::*;
use crate::format::filenames::*;

use super::super::manifest::Q4kManifestEntry;
use super::super::write_f32::WeightSource;
use super::{
    pad_rows_to_block, try_preserve_quant_passthrough, try_q4k_passthrough, QuantBlockFormat,
};

/// Write per-linear-attention-layer DeltaNet matmul tensors to
/// `deltanet_weights_q4k.bin`. Emits one manifest entry per present
/// 256-aligned tensor; non-aligned or absent tensors are skipped.
///
/// Layers where `arch.is_linear_attention_layer(layer)` is `false` are
/// skipped entirely — those layers' attention weights go through
/// [`super::attn::write_attn_weights_q4k`] (Q/K/V/O).
///
/// No-op when `arch.full_attention_interval() == 0` (non-hybrid arch):
/// nothing is written and no file is created. This is the early-exit
/// the orchestrator relies on for every model that doesn't carry
/// DeltaNet layers.
pub(super) fn write_deltanet_weights_q4k(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    let arch = source.arch();
    if arch.full_attention_interval() == 0 {
        return Ok(());
    }

    let dn_path = dir.join(DELTANET_WEIGHTS_Q4K_BIN);
    let mut dn_file = BufWriter::new(std::fs::File::create(&dn_path)?);
    let mut dn_offset: u64 = 0;
    let mut dn_manifest: Vec<Q4kManifestEntry> = Vec::new();

    for layer in 0..num_layers {
        if !arch.is_linear_attention_layer(layer) {
            continue;
        }
        callbacks.on_layer_start(COMP_ATTN_Q4K, layer, num_layers);

        // Each accessor returns `None` for non-linear layers (filtered
        // above) AND for arches that don't expose a given DeltaNet
        // projection. We tolerate the latter by simply skipping the
        // slot; the manifest records only what's present.
        let slots: [Option<String>; 5] = [
            arch.attn_qkv_key(layer),
            arch.attn_gate_key(layer),
            arch.ssm_alpha_key(layer),
            arch.ssm_beta_key(layer),
            arch.ssm_out_key(layer),
        ];

        for key_opt in slots {
            let Some(key) = key_opt else { continue };

            // Tier 1: Q4_K source → Q4_K target (bit-exact, imatrix-
            // preserved). PR #195.
            if let Some((q_bytes, rows, padded_cols)) =
                try_q4k_passthrough(source, &key, QuantBlockFormat::Q4K)
            {
                dn_file.write_all(&q_bytes)?;
                let length = q_bytes.len() as u64;
                dn_manifest.push(Q4kManifestEntry {
                    key,
                    shape: vec![rows, padded_cols],
                    format: QuantBlockFormat::Q4K,
                    offset: dn_offset,
                    length,
                });
                dn_offset += length;
                continue;
            }

            // Tier 2: preserve a higher-precision source quant
            // (Q5_K, Q6_K, Q8_0) instead of downquantizing through f32
            // to Q4_K. Qwen3-Coder-Next ships attn_qkv / ssm_out as
            // Q5_K; Unsloth Q4_K_M ships them as Q8_0 — both round-trip
            // byte-for-byte through this path.
            if let Some((q_bytes, source_format, rows, padded_cols)) =
                try_preserve_quant_passthrough(source, &key)
            {
                dn_file.write_all(&q_bytes)?;
                let length = q_bytes.len() as u64;
                dn_manifest.push(Q4kManifestEntry {
                    key,
                    shape: vec![rows, padded_cols],
                    format: source_format,
                    offset: dn_offset,
                    length,
                });
                dn_offset += length;
                continue;
            }

            let Some((data, rows, cols)) = source.get_tensor(&key) else {
                continue;
            };

            // Q4_K super-blocks need col-dim divisible by 256. Row-pad
            // exactly the same way [`super::attn`] does for `cols % 256
            // != 0` rather than skipping the tensor — the matvec shader
            // reads each row as a fixed number of super-blocks.
            let (padded, padded_cols) = pad_rows_to_block(&data, rows, cols);
            let q_bytes = quantize_q4_k(&padded);

            dn_file.write_all(&q_bytes)?;
            let length = q_bytes.len() as u64;
            dn_manifest.push(Q4kManifestEntry {
                key,
                shape: vec![rows, padded_cols],
                format: QuantBlockFormat::Q4K,
                offset: dn_offset,
                length,
            });
            dn_offset += length;
        }

        callbacks.on_layer_done(COMP_ATTN_Q4K, layer, 0.0);
    }
    dn_file.flush()?;
    drop(dn_file);

    let manifest_json = serde_json::to_string_pretty(&dn_manifest)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join(DELTANET_WEIGHTS_Q4K_MANIFEST_JSON), manifest_json)?;
    Ok(())
}
