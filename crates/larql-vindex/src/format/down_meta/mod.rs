#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

//! Binary down_meta format — compact storage for per-feature output metadata.
//!
//! Replaces down_meta.jsonl (~160 MB) with a binary format (~30 MB for top_k=10).
//! Token strings are resolved at read time via the tokenizer.
//!
//! File: down_meta.bin
//! Format (all integers/floats little-endian):
//!   Header (16 bytes): magic, version, num_layers, top_k
//!   Per layer: num_features (u32), then fixed-size records
//!   Per feature: top_token_id (u32), c_score (f32), top_k × (token_id u32, logit f32)
//!
//! Reading lives in [`read`] — both the legacy heap reader
//! ([`read_binary`]) and the mmap-backed lazy reader ([`mmap_binary`])
//! bound every allocation and offset by the actual file size, so a
//! corrupt header produces a parse error rather than a multi-GB
//! allocation or a panic.

mod read;

pub use read::{mmap_binary, read_binary};

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::index::FeatureMeta;

const MAGIC: u32 = 0x444D4554; // "DMET"
const LEGACY_LITERAL_MAGIC: u32 = 0x54454D44; // bytes written as b"DMET"
const FORMAT_VERSION: u32 = 1;
const U32_BYTES: usize = std::mem::size_of::<u32>();
const F32_BYTES: usize = std::mem::size_of::<f32>();
const HEADER_FIELDS: usize = 4;
const HEADER_BYTES: usize = HEADER_FIELDS * U32_BYTES;
const RECORD_FIXED_BYTES: usize = U32_BYTES + F32_BYTES;
const TOP_K_RECORD_BYTES: usize = U32_BYTES + F32_BYTES;

/// Write down_meta in binary format.
///
/// Writes to a sibling `.tmp` file and renames into place so an existing
/// `down_meta.bin` that is currently mmap'd by another part of the index
/// is not opened for write — Windows rejects `File::create` on a path
/// whose backing file has a user-mapped section open (`os error 1224`).
///
/// The tmp name is per-call unique (pid + monotonic counter) so two
/// concurrent writers in the same directory can't trample each other's
/// tmp file before the rename — `cargo test`'s parallel test harness
/// hits this whenever two tests' tempdirs happen to collide.
pub fn write_binary(
    dir: &Path,
    down_meta: &[Option<Vec<Option<FeatureMeta>>>],
    top_k_count: usize,
) -> Result<usize, VindexError> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(DOWN_META_BIN);
    let tmp_path = dir.join(format!(
        "{DOWN_META_BIN}.tmp.{}.{serial}",
        std::process::id()
    ));
    let file = std::fs::File::create(&tmp_path)?;
    let mut w = BufWriter::new(file);
    let mut total = 0usize;

    let num_layers = down_meta.len() as u32;

    // Header
    w.write_all(&MAGIC.to_le_bytes())?;
    w.write_all(&FORMAT_VERSION.to_le_bytes())?;
    w.write_all(&num_layers.to_le_bytes())?;
    w.write_all(&(top_k_count as u32).to_le_bytes())?;

    // Per layer
    for layer_meta in down_meta.iter() {
        match layer_meta {
            Some(features) => {
                let num_features = features.len() as u32;
                w.write_all(&num_features.to_le_bytes())?;

                for meta_opt in features {
                    match meta_opt {
                        Some(meta) => {
                            w.write_all(&meta.top_token_id.to_le_bytes())?;
                            w.write_all(&meta.c_score.to_le_bytes())?;

                            // Write exactly top_k_count entries (pad with zeros)
                            for i in 0..top_k_count {
                                if i < meta.top_k.len() {
                                    w.write_all(&meta.top_k[i].token_id.to_le_bytes())?;
                                    w.write_all(&meta.top_k[i].logit.to_le_bytes())?;
                                } else {
                                    w.write_all(&0u32.to_le_bytes())?;
                                    w.write_all(&0f32.to_le_bytes())?;
                                }
                            }
                            total += 1;
                        }
                        None => {
                            // Empty feature: token_id=0, c_score=0, all top_k zeros
                            w.write_all(&0u32.to_le_bytes())?;
                            w.write_all(&0f32.to_le_bytes())?;
                            for _ in 0..top_k_count {
                                w.write_all(&0u32.to_le_bytes())?;
                                w.write_all(&0f32.to_le_bytes())?;
                            }
                        }
                    }
                }
            }
            None => {
                w.write_all(&0u32.to_le_bytes())?; // 0 features
            }
        }
    }

    w.flush()?;
    drop(w); // close the file before rename
    std::fs::rename(&tmp_path, &path)?;
    Ok(total)
}

/// Check if a binary down_meta.bin exists in the directory.
pub fn has_binary(dir: &Path) -> bool {
    dir.join(DOWN_META_BIN).exists()
}
