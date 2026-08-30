//! Streaming access to source tensor payloads.
//!
//! The inventory records what tensors exist (name, dtype, shape, bytes,
//! shard) from safetensors *headers*; encoding additionally needs each
//! payload's absolute offset. This module re-reads the shard headers of one
//! artifact directory and answers seek+stream requests — payloads are never
//! held in memory whole, a 50 GB decoder stack streams through a fixed
//! buffer.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::VindexError;

/// Upper bound on a plausible safetensors header (mirrors the inventory
/// scanner's bound).
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

/// Key safetensors reserves for free-form metadata inside the header.
const HEADER_METADATA_KEY: &str = "__metadata__";

/// Where one tensor's payload lives on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadLocation {
    pub shard: PathBuf,
    /// Absolute file offset of the first payload byte.
    pub offset: u64,
    pub len: u64,
}

/// Payload locations for every tensor of one artifact directory.
pub struct ArtifactSource {
    locations: BTreeMap<String, PayloadLocation>,
}

impl ArtifactSource {
    /// Scan every `*.safetensors` shard in `dir` (via the HF index when
    /// present, else directory listing) and record payload locations.
    pub fn open(dir: &Path) -> Result<Self, VindexError> {
        let shards = discover_shards(dir)?;
        let mut locations = BTreeMap::new();
        for shard in shards {
            index_shard(&shard, &mut locations)?;
        }
        Ok(Self { locations })
    }

    /// Location of one tensor by its full source name.
    pub fn locate(&self, name: &str) -> Result<&PayloadLocation, VindexError> {
        self.locations.get(name).ok_or_else(|| {
            VindexError::Parse(format!(
                "tensor `{name}` is in the inventory but not in any shard header — \
                 the source directory changed since inspection"
            ))
        })
    }

    /// Stream one tensor's payload into `write`, feeding every byte through
    /// `observe` (hashing) on the way. Returns bytes copied. Dyn parameters
    /// because callers pass through `write_segment`'s dyn callback.
    pub fn stream_payload(
        &self,
        name: &str,
        write: &mut dyn std::io::Write,
        observe: &mut dyn FnMut(&[u8]),
    ) -> Result<u64, VindexError> {
        /// Fixed streaming buffer size.
        const STREAM_BUF: usize = 1 << 20;
        let location = self.locate(name)?;
        let mut file = std::fs::File::open(&location.shard)?;
        file.seek(SeekFrom::Start(location.offset))?;
        let mut remaining = location.len;
        let mut buffer = vec![0u8; STREAM_BUF];
        while remaining > 0 {
            let want = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..want]).map_err(|e| {
                VindexError::Parse(format!(
                    "short read streaming `{name}` from {}: {e}",
                    location.shard.display()
                ))
            })?;
            observe(&buffer[..want]);
            write.write_all(&buffer[..want])?;
            remaining -= want as u64;
        }
        Ok(location.len)
    }
}

/// Shard filenames for a checkpoint dir (HF index preferred, sorted).
fn discover_shards(dir: &Path) -> Result<Vec<PathBuf>, VindexError> {
    /// Filename of the HF shard index.
    const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";
    /// Shard extension for the fallback scan.
    const SAFETENSORS_EXT: &str = "safetensors";
    let index_path = dir.join(SAFETENSORS_INDEX_FILE);
    if index_path.exists() {
        let text = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| VindexError::Parse(format!("parse {SAFETENSORS_INDEX_FILE}: {e}")))?;
        let mut files: Vec<String> = index["weight_map"]
            .as_object()
            .map(|m| {
                m.values()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == SAFETENSORS_EXT))
        .collect();
    files.sort();
    Ok(files)
}

/// Record every tensor of one shard: absolute offset = 8-byte length
/// prefix + header + relative data offset.
fn index_shard(
    shard: &Path,
    out: &mut BTreeMap<String, PayloadLocation>,
) -> Result<(), VindexError> {
    let mut file = std::fs::File::open(shard)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_BYTES {
        return Err(VindexError::Parse(format!(
            "{}: safetensors header claims {header_len} bytes — corrupt",
            shard.display()
        )));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| VindexError::Parse(format!("{}: header: {e}", shard.display())))?;
    let payload_base = 8 + header_len;
    let entries = header.as_object().ok_or_else(|| {
        VindexError::Parse(format!("{}: header is not an object", shard.display()))
    })?;
    for (name, desc) in entries {
        if name == HEADER_METADATA_KEY {
            continue;
        }
        let offsets = desc["data_offsets"].as_array().and_then(|offs| {
            match (offs.first()?.as_u64(), offs.get(1)?.as_u64()) {
                (Some(start), Some(end)) if end >= start => Some((start, end)),
                _ => None,
            }
        });
        let Some((start, end)) = offsets else {
            return Err(VindexError::Parse(format!(
                "{}: tensor `{name}` has no valid data_offsets",
                shard.display()
            )));
        };
        out.insert(
            name.clone(),
            PayloadLocation {
                shard: shard.to_path_buf(),
                offset: payload_base + start,
                len: end - start,
            },
        );
    }
    Ok(())
}
