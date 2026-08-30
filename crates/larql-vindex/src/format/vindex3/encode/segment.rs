//! One representation's segment: self-describing framing + payloads.
//!
//! Framing mirrors the proven safetensors shape, with our own header:
//!
//! ```text
//! [u64 LE header length][header JSON][payload bytes…]
//! ```
//!
//! The header carries the representation id and a tensor table (name
//! relative to the object, dtype, shape, offset, length — offsets relative
//! to the payload region). Table order is the payload order and is
//! deterministic: sorted by relative name. Two hashes are produced in the
//! single writing pass: one over the payload region (the source-side
//! byte-equivalence anchor) and one over the whole file as written.

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::VindexError;

/// Current segment header schema. Bump on any breaking change.
pub const SEGMENT_HEADER_SCHEMA: u32 = 1;

/// The segment header, as serialised into the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentHeader {
    /// Always [`SEGMENT_HEADER_SCHEMA`].
    pub schema: u32,
    /// Representation id this segment materialises (`object@encoding`).
    pub representation: String,
    /// Tensor table; order is payload order.
    pub tensors: Vec<SegmentTensor>,
}

/// One tensor within a segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentTensor {
    /// Name relative to the logical object (never artifact-global).
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Offset within the payload region.
    pub offset: u64,
    pub len: u64,
}

/// What one planned tensor needs from the writer.
pub struct PlannedTensor {
    /// Object-relative name (table key).
    pub relative_name: String,
    /// Full source name (payload lookup key).
    pub source_name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub len: u64,
}

/// Result of writing one segment.
pub struct WrittenSegment {
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub segment_sha256: String,
}

/// Deterministic payload order: sorted by object-relative name. The single
/// definition of segment order, shared by the writer and the G4 source-side
/// re-hash so the two can never disagree about what "the same bytes" means.
pub fn sort_into_payload_order(tensors: &mut [PlannedTensor]) {
    tensors.sort_by(|a, b| a.relative_name.cmp(&b.relative_name));
}

/// Write one segment file: header first (offsets are known from the plan),
/// then every payload streamed through both hashers.
pub fn write_segment(
    path: &Path,
    representation: &str,
    mut tensors: Vec<PlannedTensor>,
    mut stream_payload: impl FnMut(
        &str,
        &mut dyn Write,
        &mut dyn FnMut(&[u8]),
    ) -> Result<u64, VindexError>,
) -> Result<WrittenSegment, VindexError> {
    sort_into_payload_order(&mut tensors);
    let mut offset = 0u64;
    let table: Vec<SegmentTensor> = tensors
        .iter()
        .map(|t| {
            let entry = SegmentTensor {
                name: t.relative_name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                offset,
                len: t.len,
            };
            offset += t.len;
            entry
        })
        .collect();
    let header = SegmentHeader {
        schema: SEGMENT_HEADER_SCHEMA,
        representation: representation.to_string(),
        tensors: table,
    };
    let header_bytes = serde_json::to_vec(&header)
        .map_err(|e| VindexError::Parse(format!("serialise segment header: {e}")))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    let mut file_hasher = Sha256::new();
    let len_prefix = (header_bytes.len() as u64).to_le_bytes();
    writer.write_all(&len_prefix)?;
    file_hasher.update(len_prefix);
    writer.write_all(&header_bytes)?;
    file_hasher.update(&header_bytes);

    let mut payload_hasher = Sha256::new();
    let mut payload_bytes = 0u64;
    for tensor in &tensors {
        let copied = stream_payload(&tensor.source_name, &mut writer, &mut |chunk| {
            payload_hasher.update(chunk);
            file_hasher.update(chunk);
        })?;
        if copied != tensor.len {
            return Err(VindexError::Parse(format!(
                "segment `{representation}`: tensor `{}` copied {copied} bytes, \
                 planned {} — source changed underneath the encode",
                tensor.source_name, tensor.len
            )));
        }
        payload_bytes += copied;
    }
    writer.flush()?;
    Ok(WrittenSegment {
        tensor_count: tensors.len(),
        payload_bytes,
        payload_sha256: format!("{:x}", payload_hasher.finalize()),
        segment_sha256: format!("{:x}", file_hasher.finalize()),
    })
}

/// Read a segment's header (framing only; no payload I/O).
pub fn read_segment_header(path: &Path) -> Result<(SegmentHeader, u64), VindexError> {
    use std::io::Read;
    /// Bound mirrors the writer's practical header sizes.
    const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_BYTES {
        return Err(VindexError::Parse(format!(
            "{}: segment header claims {header_len} bytes — corrupt",
            path.display()
        )));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let header: SegmentHeader = serde_json::from_slice(&header_bytes)
        .map_err(|e| VindexError::Parse(format!("{}: segment header: {e}", path.display())))?;
    Ok((header, 8 + header_len))
}
