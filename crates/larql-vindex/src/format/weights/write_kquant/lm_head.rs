//! Stage 6 — `lm_head_q4.bin`.
//!
//! Q4_K of the output projection matrix. Falls back to embed_tokens
//! when the architecture ties the embed and lm_head weights (Gemma,
//! Qwen, etc.); the source layer surfaces that via `source.lm_head()`.
//! Manifest entry is appended to the running norms manifest so
//! `weight_manifest.json` references everything in one list.

use std::path::Path;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::super::write_f32::{kind, WeightEntry, WeightSource};
use super::{pad_rows_to_block, try_q4k_passthrough, QuantBlockFormat};

pub(super) fn write_lm_head_kquant(
    source: &dyn WeightSource,
    dir: &Path,
    norm_entries: &mut Vec<WeightEntry>,
) -> Result<(), VindexError> {
    // Bit-passthrough fast path: when the source carries lm_head as
    // raw GGUF Q4_K bytes with cols % 256 == 0, copy directly.
    // Falls back to dequant+requant otherwise. See
    // [`super::try_q4k_passthrough`].
    if let Some((q_bytes, rows, padded_cols)) =
        try_q4k_passthrough(source, "lm_head.weight", QuantBlockFormat::Q4K)
    {
        std::fs::write(dir.join(LM_HEAD_Q4_BIN), &q_bytes)?;
        norm_entries.push(WeightEntry {
            key: "lm_head.weight".into(),
            kind: kind::TENSOR_Q4K.into(),
            shape: vec![rows, padded_cols],
            offset: 0,
            length: q_bytes.len() as u64,
            file: LM_HEAD_Q4_BIN.into(),
        });
        return Ok(());
    }
    if let Some((data, rows, cols)) = source.lm_head() {
        // Preserve all rows. GGUFs commonly ship `token_embd` / lm_head
        // with padded vocab (extra rows for special tokens like
        // `<|im_start|>` on Qwen 3.6, or SIMD alignment on Gemma 3).
        // Truncating to the logical vocab makes the loader fall back
        // to a zero-pad workaround at the special-token IDs, which
        // corrupts the DeltaNet recurrent state on the first prompt
        // tokens (those are the chat-template special tokens). Match
        // `build::write_embeddings` and keep the full padded matrix.
        let (truncated_data, truncated_rows) = (data, rows);
        let (padded, padded_cols) = pad_rows_to_block(&truncated_data, truncated_rows, cols);
        let q_bytes = quantize_q4_k(&padded);
        std::fs::write(dir.join(LM_HEAD_KQUANT_BIN), &q_bytes)?;
        // Record in norms manifest so a single weight_manifest.json references
        // everything non-quantised-via-layout. Shape records the stored
        // `padded_cols` — callers route through the matvec dispatch which
        // uses shape[1] as `K`, so the padding stays invisible provided the
        // input activation buffer is zero-padded to match.
        norm_entries.push(WeightEntry {
            key: "lm_head.weight".into(),
            kind: kind::TENSOR_Q4K.into(),
            shape: vec![truncated_rows, padded_cols],
            offset: 0,
            length: q_bytes.len() as u64,
            file: LM_HEAD_KQUANT_BIN.into(),
        });
    }
    Ok(())
}
