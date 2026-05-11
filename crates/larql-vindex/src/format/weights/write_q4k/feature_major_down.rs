//! W2 feature-major down emit — transposes the down weights to
//! `[intermediate, hidden]` orientation and re-quantises at the same
//! precision the interleaved file uses, so per-feature decode at load
//! time can skip the `q4k_ffn_layer` cache and serve a single row.
//!
//! Lives only during the FFN write loop in
//! `super::write_model_weights_q4k_with_opts`. Each layer's down call
//! goes through `append_layer`; `finalize` flushes the bytes and emits
//! `down_features_q4k_manifest.json`. Both files are opt-in
//! (`Q4kWriteOptions::feature_major_down`).
//!
//! See `ROADMAP.md` § W2 for the perf rationale (2440× at K=100,
//! 25× at full K on Gemma 4B Q4_K).
//!
//! Carved out of the monolithic `write_q4k.rs` in the 2026-04-25
//! modularity pass.

use std::io::{BufWriter, Write};
use std::path::Path;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

use crate::error::VindexError;
use crate::format::weights::Q4kManifestEntry;

use super::{pad_rows_to_block, QuantBlockFormat};

/// In-flight state for the W2 feature-major down emission. Lives only
/// while the FFN write loop is running; collapsed into the manifest
/// JSON at end-of-loop. Each field has a name at the call sites
/// (replaces what used to be an anonymous 3-tuple inside the writer).
pub(crate) struct FeatureMajorDownState {
    file: BufWriter<std::fs::File>,
    next_offset: u64,
    manifest: Vec<Q4kManifestEntry>,
}

impl FeatureMajorDownState {
    pub(crate) fn new(path: &Path, capacity_layers: usize) -> Result<Self, VindexError> {
        Ok(Self {
            file: BufWriter::new(std::fs::File::create(path)?),
            next_offset: 0,
            manifest: Vec::with_capacity(capacity_layers),
        })
    }

    /// Transpose padded down (`[hidden, padded_intermediate]`) to
    /// feature-major (`[padded_intermediate, padded_hidden]`),
    /// re-pad rows to 256, and quantise at `format`. Mirrors the
    /// orientation used by `q4k_ffn_layer`'s in-memory transpose so
    /// the runtime decode path reads the same byte layout.
    pub(crate) fn append_layer(
        &mut self,
        key: String,
        padded_down: &[f32],
        rows_hidden: usize,
        cols_padded_intermediate: usize,
        format: QuantBlockFormat,
    ) -> Result<(), VindexError> {
        let n = rows_hidden * cols_padded_intermediate;
        debug_assert_eq!(padded_down.len(), n);
        let mut transposed = vec![0.0f32; n];
        for h in 0..rows_hidden {
            let src =
                &padded_down[h * cols_padded_intermediate..(h + 1) * cols_padded_intermediate];
            for (feat, &v) in src.iter().enumerate() {
                transposed[feat * rows_hidden + h] = v;
            }
        }
        let (fm_padded, fm_padded_cols) =
            pad_rows_to_block(&transposed, cols_padded_intermediate, rows_hidden);
        let bytes = match format {
            QuantBlockFormat::Q6K => quantize_q6_k(&fm_padded),
            QuantBlockFormat::Q4K => quantize_q4_k(&fm_padded),
        };
        self.file.write_all(&bytes)?;
        let length = bytes.len() as u64;
        self.manifest.push(Q4kManifestEntry {
            key,
            shape: vec![cols_padded_intermediate, fm_padded_cols],
            format,
            offset: self.next_offset,
            length,
        });
        self.next_offset += length;
        Ok(())
    }

    /// Flush the bytes and write the manifest JSON sidecar.
    pub(crate) fn finalize(mut self, manifest_path: &Path) -> Result<(), VindexError> {
        self.file.flush()?;
        drop(self.file);
        let json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(manifest_path, json)?;
        Ok(())
    }
}
