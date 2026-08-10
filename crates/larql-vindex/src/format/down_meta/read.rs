//! Reading `down_meta.bin` — legacy heap reader + mmap-backed lazy reader.
//!
//! Both paths treat the header's `num_layers` / `top_k_count` /
//! per-layer `num_features` counts as **untrusted**: every count is
//! validated against the actual file size (with checked arithmetic)
//! before it sizes an allocation or an offset. A corrupt or truncated
//! file yields `VindexError::Parse`, never a panic and never a
//! multi-GB `Vec::with_capacity` abort.

use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::VindexError;
use crate::format::filenames::DOWN_META_BIN;
use crate::index::FeatureMeta;

use super::{
    FORMAT_VERSION, HEADER_BYTES, LEGACY_LITERAL_MAGIC, MAGIC, RECORD_FIXED_BYTES,
    TOP_K_RECORD_BYTES, U32_BYTES,
};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Bytes of one per-feature record for a given `top_k_count`, with
/// overflow checked — `top_k_count` comes straight from the file header.
fn record_byte_size(top_k_count: usize) -> Result<usize, VindexError> {
    top_k_count
        .checked_mul(TOP_K_RECORD_BYTES)
        .and_then(|n| n.checked_add(RECORD_FIXED_BYTES))
        .ok_or_else(|| VindexError::Parse("down_meta.bin record size overflow".into()))
}

/// Validate the magic + version fields shared by both readers.
fn check_magic_version(magic: u32, version: u32) -> Result<(), VindexError> {
    if magic != MAGIC && magic != LEGACY_LITERAL_MAGIC {
        return Err(VindexError::Parse(format!(
            "invalid down_meta.bin magic: expected 0x{MAGIC:08X}, got 0x{magic:08X}"
        )));
    }
    if version != FORMAT_VERSION {
        return Err(VindexError::Parse(format!(
            "unsupported down_meta.bin version: {version}"
        )));
    }
    Ok(())
}

/// Read down_meta from binary format (legacy heap path — the mmap path
/// is [`mmap_binary`]). Token strings are resolved via the tokenizer.
///
/// Header counts are attacker-controlled: `num_layers`, `top_k_count`
/// and each layer's `num_features` are validated against the real file
/// size before they size any allocation, mirroring [`mmap_binary`].
#[allow(clippy::type_complexity)]
pub fn read_binary(
    dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<(Vec<Option<Vec<Option<FeatureMeta>>>>, usize), VindexError> {
    let path = dir.join(DOWN_META_BIN);
    let file = std::fs::File::open(&path)?;
    let file_len = usize::try_from(file.metadata()?.len())
        .map_err(|_| VindexError::Parse("down_meta.bin larger than address space".into()))?;
    let mut r = BufReader::new(file);

    // Header
    let magic = read_u32(&mut r)?;
    let version = read_u32(&mut r)?;
    check_magic_version(magic, version)?;
    let num_layers = read_u32(&mut r)? as usize;
    let top_k_count = read_u32(&mut r)? as usize;

    let record_size = record_byte_size(top_k_count)?;

    // Every declared layer costs at least its u32 feature-count field,
    // so `num_layers` is bounded by the bytes actually on disk. This
    // caps the `with_capacity` below — a corrupt header can no longer
    // request a multi-GB allocation before the first read fails.
    let min_layer_bytes = num_layers
        .checked_mul(U32_BYTES)
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .ok_or_else(|| VindexError::Parse("down_meta.bin layer count overflow".into()))?;
    if min_layer_bytes > file_len {
        return Err(VindexError::Parse(format!(
            "truncated down_meta.bin: header declares {num_layers} layers \
             but the file is only {file_len} bytes"
        )));
    }

    let mut remaining = file_len - HEADER_BYTES;
    let mut down_meta: Vec<Option<Vec<Option<FeatureMeta>>>> = Vec::with_capacity(num_layers);
    let mut total = 0usize;

    for layer_idx in 0..num_layers {
        let num_features = read_u32(&mut r)? as usize;
        remaining = remaining.saturating_sub(U32_BYTES);
        if num_features == 0 {
            down_meta.push(None);
            continue;
        }

        // Bound this layer's declared records by the bytes left in the
        // file — same check the mmap reader's offset table performs.
        let layer_bytes = num_features
            .checked_mul(record_size)
            .ok_or_else(|| VindexError::Parse("down_meta.bin layer size overflow".into()))?;
        if layer_bytes > remaining {
            return Err(VindexError::Parse(format!(
                "truncated down_meta.bin records for layer {layer_idx}: \
                 {num_features} features need {layer_bytes} bytes, {remaining} remain"
            )));
        }
        remaining -= layer_bytes;

        let mut features: Vec<Option<FeatureMeta>> = Vec::with_capacity(num_features);
        for _ in 0..num_features {
            let top_token_id = read_u32(&mut r)?;
            let c_score = read_f32(&mut r)?;

            let mut top_k = Vec::with_capacity(top_k_count);
            for _ in 0..top_k_count {
                let token_id = read_u32(&mut r)?;
                let logit = read_f32(&mut r)?;
                if token_id > 0 || logit != 0.0 {
                    let token = tokenizer
                        .decode(&[token_id], true)
                        .unwrap_or_else(|_| format!("T{token_id}"))
                        .trim()
                        .to_string();
                    top_k.push(larql_models::TopKEntry {
                        token,
                        token_id,
                        logit,
                    });
                }
            }

            if top_token_id == 0 && c_score == 0.0 && top_k.is_empty() {
                features.push(None);
            } else {
                let top_token = tokenizer
                    .decode(&[top_token_id], true)
                    .unwrap_or_else(|_| format!("T{top_token_id}"))
                    .trim()
                    .to_string();
                features.push(Some(FeatureMeta {
                    top_token,
                    top_token_id,
                    c_score,
                    top_k,
                }));
                total += 1;
            }
        }

        down_meta.push(Some(features));
    }

    Ok((down_meta, total))
}

/// Mmap down_meta.bin and build a lazy reader (zero heap for feature data).
/// Only parses the header + per-layer feature counts to build the offset table.
pub fn mmap_binary(
    dir: &Path,
    tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
) -> Result<crate::index::core::DownMetaMmap, VindexError> {
    let path = dir.join(DOWN_META_BIN);
    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    if mmap.len() < HEADER_BYTES {
        return Err(VindexError::Parse("down_meta.bin too small".into()));
    }

    // Read header
    let magic = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
    let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
    check_magic_version(magic, version)?;
    let num_layers = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;
    let top_k_count = u32::from_le_bytes([mmap[12], mmap[13], mmap[14], mmap[15]]) as usize;

    let record_size = record_byte_size(top_k_count)?;

    // Same bound as `read_binary`: each declared layer costs at least
    // one u32 on disk, so the offset-table capacity can't outrun the file.
    let min_layer_bytes = num_layers
        .checked_mul(U32_BYTES)
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .ok_or_else(|| VindexError::Parse("down_meta.bin layer count overflow".into()))?;
    if min_layer_bytes > mmap.len() {
        return Err(VindexError::Parse(format!(
            "truncated down_meta.bin: header declares {num_layers} layers \
             but the file is only {} bytes",
            mmap.len()
        )));
    }

    // Build offset table by scanning per-layer num_features headers
    let mut layer_offsets = Vec::with_capacity(num_layers);
    let mut layer_num_features = Vec::with_capacity(num_layers);
    let mut pos = HEADER_BYTES;

    for _ in 0..num_layers {
        if pos + U32_BYTES > mmap.len() {
            return Err(VindexError::Parse(
                "truncated down_meta.bin layer header".into(),
            ));
        }
        let nf =
            u32::from_le_bytes([mmap[pos], mmap[pos + 1], mmap[pos + 2], mmap[pos + 3]]) as usize;
        pos += U32_BYTES;
        layer_offsets.push(pos); // records start here
        layer_num_features.push(nf);
        let layer_bytes = nf
            .checked_mul(record_size)
            .ok_or_else(|| VindexError::Parse("down_meta.bin layer size overflow".into()))?;
        let layer_end = pos
            .checked_add(layer_bytes)
            .ok_or_else(|| VindexError::Parse("down_meta.bin layer end overflow".into()))?;
        if layer_end > mmap.len() {
            return Err(VindexError::Parse(format!(
                "truncated down_meta.bin records for layer {}",
                layer_offsets.len() - 1
            )));
        }
        pos = layer_end; // skip all records
    }

    Ok(crate::index::core::DownMetaMmap {
        mmap: std::sync::Arc::new(mmap),
        layer_offsets,
        layer_num_features,
        top_k_count,
        tokenizer,
    })
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, VindexError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, VindexError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}
