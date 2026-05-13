//! Shared writer for Gemma-4 Per-Layer Embedding (PLE) sidecars.
//!
//! PLE tensors are large (~4.7 GB for E2B's `embed_tokens_per_layer`)
//! but behave like embedding tables: each super-block of 256 values
//! spans a wide dynamic range with a handful of outliers. Q4_K's
//! per-super-block calibration zeros out the majority of cells to
//! accommodate those outliers, and the cell-level noise compounds
//! over 35+ layers of additive contribution — the observable result
//! was garbage tokens on Gemma 4 E2B / E4B. f16 halves the BF16
//! footprint and preserves enough precision for accurate per-token
//! retrieval, so PLE is stored as `kind::TENSOR_F16` in
//! `ple_weights.bin` regardless of the rest of the vindex's quant
//! mode.
//!
//! Both writers (`write_f32` and `write_q4k`) call into this helper
//! so the on-disk layout — and the manifest entries the loader
//! validates — stay byte-identical. Keeps Gemma-4 inference correct
//! across `--quant none`, `--quant q4k`, and any future quant modes.
//! Regression context: chrishayuk/larql#49.

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::write_f32::{kind, WeightEntry, WeightSource};

/// Write `ple_weights.bin` and append `tensor_f16` manifest entries
/// for every Gemma-4 PLE tensor. No-op when the architecture has no
/// PLE (i.e. `!arch.has_per_layer_embeddings()`).
///
/// `manifest_entries` is the running `Vec<WeightEntry>` that the
/// caller threads through every weight-writing stage; this function
/// appends to it in place.
pub(super) fn write_ple_weights(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
    manifest_entries: &mut Vec<WeightEntry>,
) -> Result<(), VindexError> {
    let arch = source.arch();
    if !arch.has_per_layer_embeddings() {
        return Ok(());
    }

    let ple_path = dir.join(PLE_WEIGHTS_BIN);
    let mut ple_file = BufWriter::new(std::fs::File::create(&ple_path)?);
    let mut ple_offset: u64 = 0;
    let ple_dtype = crate::config::dtype::StorageDtype::F16;

    let write_tensor = |file: &mut BufWriter<std::fs::File>,
                        manifest: &mut Vec<WeightEntry>,
                        offset: &mut u64,
                        key: String,
                        data: Option<(Vec<f32>, usize, usize)>|
     -> Result<(), VindexError> {
        if let Some((floats, rows, cols)) = data {
            let bytes = crate::config::dtype::encode_floats(&floats, ple_dtype);
            file.write_all(&bytes)?;
            manifest.push(WeightEntry {
                key,
                kind: kind::TENSOR_F16.into(),
                shape: vec![rows, cols],
                offset: *offset,
                length: bytes.len() as u64,
                file: PLE_WEIGHTS_BIN.into(),
            });
            *offset += bytes.len() as u64;
        }
        Ok(())
    };

    // Global: model projection [ple_dim·num_layers, hidden]
    write_tensor(
        &mut ple_file,
        manifest_entries,
        &mut ple_offset,
        "per_layer_model_projection.weight".into(),
        source.get_tensor("per_layer_model_projection.weight"),
    )?;

    // Global: big embedding table [vocab, ple_dim·num_layers]
    if let Some(key) = arch.per_layer_embed_key() {
        write_tensor(
            &mut ple_file,
            manifest_entries,
            &mut ple_offset,
            key.clone(),
            source.get_tensor(&key),
        )?;
    }

    // Per-layer: input_gate + projection
    for layer in 0..num_layers {
        if let Some(k) = arch.per_layer_input_gate_key(layer) {
            write_tensor(
                &mut ple_file,
                manifest_entries,
                &mut ple_offset,
                k.clone(),
                source.get_tensor(&k),
            )?;
        }
        if let Some(k) = arch.per_layer_projection_key(layer) {
            write_tensor(
                &mut ple_file,
                manifest_entries,
                &mut ple_offset,
                k.clone(),
                source.get_tensor(&k),
            )?;
        }
    }

    ple_file.flush()?;
    Ok(())
}
