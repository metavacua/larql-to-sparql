//! Stage 5 — `ple_weights.bin` (Gemma 4 E2B Per-Layer Embeddings).
//!
//! Stored as f16 — NOT Q4_K. The two globals
//! (`per_layer_model_projection`, `embed_tokens_per_layer`) and the
//! per-layer input_gate/projection matrices behave like embedding
//! tables: each super-block of 256 values spans a wide dynamic range
//! with a handful of outliers, and Q4_K's per-super-block (d, dmin)
//! calibration zeros out the majority of cells to accommodate those
//! outliers. PLE contributions are additive into every layer's
//! residual, so the cell-level noise compounds across 35 layers — the
//! observable result was "arrays" / "amphibians" instead of "Paris" on
//! Gemma 4 E2B. f16 halves the BF16 footprint (~4.7 GB for the big
//! lookup on E2B) and preserves enough precision for accurate
//! per-token PLE retrieval.
//!
//! Manifest entries are appended to the running norms manifest so
//! `weight_manifest.json` references everything in one list.

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::VindexError;
use crate::format::filenames::*;

use super::super::write_f32::{kind, WeightEntry, WeightSource};

pub(super) fn write_ple_weights(
    source: &dyn WeightSource,
    dir: &Path,
    num_layers: usize,
    norm_entries: &mut Vec<WeightEntry>,
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
        norm_entries,
        &mut ple_offset,
        "per_layer_model_projection.weight".into(),
        source.get_tensor("per_layer_model_projection.weight"),
    )?;

    // Global: big embedding table [vocab, ple_dim·num_layers]
    if let Some(key) = arch.per_layer_embed_key() {
        write_tensor(
            &mut ple_file,
            norm_entries,
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
                norm_entries,
                &mut ple_offset,
                k.clone(),
                source.get_tensor(&k),
            )?;
        }
        if let Some(k) = arch.per_layer_projection_key(layer) {
            write_tensor(
                &mut ple_file,
                norm_entries,
                &mut ple_offset,
                k.clone(),
                source.get_tensor(&k),
            )?;
        }
    }

    ple_file.flush()?;
    Ok(())
}
