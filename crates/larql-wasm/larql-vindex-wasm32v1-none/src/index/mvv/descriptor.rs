//! The MVV descriptor: the minimal subset of `index.json` (the canonical
//! source of truth) needed to interpret the gate-vector blob, plus the
//! validation invariants that make the reader total.
//!
//! Extra fields present in a full vindex `index.json` (model, family, fp4,
//! attn/lm_head layout, …) are ignored by serde — the MVV descriptor reads
//! only what a minimal gate-KNN query requires.
#[allow(unused_imports)]
use alloc::vec::Vec;
use serde::Deserialize;

use crate::config::dtype::{bytes_per_float, StorageDtype};

use super::error::MvvError;

/// The MVV descriptor version this reader accepts (matches `VindexConfig`).
pub const MVV_DESCRIPTOR_VERSION: u32 = 2;

/// Per-layer record: where this layer's gate matrix lives in the blob.
#[derive(Debug, Clone, Deserialize)]
pub struct MvvLayer {
    pub layer: usize,
    pub num_features: usize,
    /// Byte offset into the gate-vector blob.
    pub offset: u64,
    /// Byte length of this layer's gate matrix.
    pub length: u64,
}

/// Minimal descriptor parsed from `index.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct MvvDescriptor {
    pub version: u32,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub dtype: StorageDtype,
    pub layers: Vec<MvvLayer>,
}

impl MvvDescriptor {
    /// Parse the minimal descriptor from `index.json` bytes. Total: any parse
    /// failure (malformed JSON, missing field, bad dtype string) → `BadIndexJson`.
    pub fn from_index_json(bytes: &[u8]) -> Result<Self, MvvError> {
        serde_json::from_slice(bytes).map_err(|_| MvvError::BadIndexJson)
    }

    pub fn bytes_per_float(&self) -> usize {
        bytes_per_float(self.dtype)
    }

    /// Validate internal consistency and exact coverage of a `blob_len`-byte
    /// gate blob. After this returns `Ok`, every per-layer byte range is
    /// in-bounds and dimensionally consistent — the precondition the query
    /// kernel relies on to stay total without `unsafe`.
    pub fn validate(&self, blob_len: usize) -> Result<(), MvvError> {
        if self.version != MVV_DESCRIPTOR_VERSION {
            return Err(MvvError::UnsupportedVersion(self.version));
        }
        if self.layers.len() != self.num_layers {
            return Err(MvvError::LayerCountMismatch {
                declared: self.num_layers,
                actual: self.layers.len(),
            });
        }
        let bpf = self.bytes_per_float();
        let mut max_end: usize = 0;
        for (i, l) in self.layers.iter().enumerate() {
            let declared = l.length as usize;
            // expected = num_features * hidden_size * bytes_per_float, overflow-checked.
            let expected = l
                .num_features
                .checked_mul(self.hidden_size)
                .and_then(|v| v.checked_mul(bpf))
                .ok_or(MvvError::LayerLengthMismatch {
                    layer: i,
                    expected: usize::MAX,
                    declared,
                })?;
            if declared != expected {
                return Err(MvvError::LayerLengthMismatch {
                    layer: i,
                    expected,
                    declared,
                });
            }
            if bpf == 0 || declared % bpf != 0 {
                return Err(MvvError::Misaligned { layer: i });
            }
            let off = l.offset as usize;
            let end = off
                .checked_add(declared)
                .ok_or(MvvError::BlobLengthMismatch {
                    expected: usize::MAX,
                    actual: blob_len,
                })?;
            if end > blob_len {
                return Err(MvvError::BlobLengthMismatch {
                    expected: end,
                    actual: blob_len,
                });
            }
            if end > max_end {
                max_end = end;
            }
        }
        // Minimal blob = exact coverage, no trailing slack.
        if max_end != blob_len {
            return Err(MvvError::BlobLengthMismatch {
                expected: max_end,
                actual: blob_len,
            });
        }
        Ok(())
    }
}
