//! Patch file format — `.vlp` JSON diffs that overlay an immutable
//! base vindex without modifying its files on disk.
//!
//! This module owns the on-the-wire representation: `VindexPatch`,
//! `PatchOp` (Insert/Update/Delete + arch-B InsertKnn/DeleteKnn),
//! `PatchDownMeta`, save/load, and the base64 helpers used to embed
//! gate/key vectors inside the JSON.
//!
//! Runtime application of patches lives in `super::overlay`
//! (`PatchedVindex`).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VindexError;

// ═══════════════════════════════════════════════════════════════
// Patch data types
// ═══════════════════════════════════════════════════════════════

/// A vindex patch — a set of operations to apply to a base vindex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VindexPatch {
    pub version: u32,
    pub base_model: String,
    #[serde(default)]
    pub base_checksum: Option<String>,
    pub created_at: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub operations: Vec<PatchOp>,
}

/// A single patch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum PatchOp {
    Insert {
        layer: usize,
        feature: usize,
        #[serde(default)]
        relation: Option<String>,
        entity: String,
        target: String,
        #[serde(default)]
        confidence: Option<f32>,
        /// Base64-encoded f32 gate vector.
        #[serde(default)]
        gate_vector_b64: Option<String>,
        #[serde(default)]
        down_meta: Option<PatchDownMeta>,
    },
    Update {
        layer: usize,
        feature: usize,
        #[serde(default)]
        gate_vector_b64: Option<String>,
        #[serde(default)]
        down_meta: Option<PatchDownMeta>,
    },
    Delete {
        layer: usize,
        feature: usize,
        #[serde(default)]
        reason: Option<String>,
    },
    /// Architecture B: residual-key KNN insert.
    #[serde(rename = "insert_knn")]
    InsertKnn {
        layer: usize,
        entity: String,
        relation: String,
        target: String,
        target_id: u32,
        #[serde(default)]
        confidence: Option<f32>,
        /// Base64-encoded f32 residual key (L2-normalized).
        key_vector_b64: String,
    },
    /// Architecture B: remove all KNN entries for an entity.
    #[serde(rename = "delete_knn")]
    DeleteKnn {
        entity: String,
    },
}

/// Compact down_meta for a patch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchDownMeta {
    #[serde(rename = "t")]
    pub top_token: String,
    #[serde(rename = "i")]
    pub top_token_id: u32,
    #[serde(rename = "c")]
    pub c_score: f32,
}

impl PatchOp {
    /// The (layer, feature) this operation targets. KNN ops return None.
    pub fn key(&self) -> Option<(usize, usize)> {
        match self {
            PatchOp::Insert { layer, feature, .. } => Some((*layer, *feature)),
            PatchOp::Update { layer, feature, .. } => Some((*layer, *feature)),
            PatchOp::Delete { layer, feature, .. } => Some((*layer, *feature)),
            PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. } => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Patch file I/O
// ═══════════════════════════════════════════════════════════════

impl VindexPatch {
    /// Write patch to a .vlp file.
    pub fn save(&self, path: &Path) -> Result<(), VindexError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load patch from a .vlp file.
    pub fn load(path: &Path) -> Result<Self, VindexError> {
        let text = std::fs::read_to_string(path)?;
        let patch: VindexPatch = serde_json::from_str(&text)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        Ok(patch)
    }

    /// Number of operations in this patch.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Whether this patch has no operations.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Summary counts: (inserts, updates, deletes).
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut ins = 0;
        let mut upd = 0;
        let mut del = 0;
        for op in &self.operations {
            match op {
                PatchOp::Insert { .. } | PatchOp::InsertKnn { .. } => ins += 1,
                PatchOp::Update { .. } => upd += 1,
                PatchOp::Delete { .. } | PatchOp::DeleteKnn { .. } => del += 1,
            }
        }
        (ins, upd, del)
    }
}

// ═══════════════════════════════════════════════════════════════
// Base64 gate vector encoding
// ═══════════════════════════════════════════════════════════════

/// Encode a gate vector (f32 slice) as base64 string.
pub fn encode_gate_vector(vec: &[f32]) -> String {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4)
    };
    base64_encode(bytes)
}

/// Decode a base64 string back to f32 vector.
pub fn decode_gate_vector(b64: &str) -> Result<Vec<f32>, VindexError> {
    let bytes = base64_decode(b64)?;
    if bytes.len() % 4 != 0 {
        return Err(VindexError::Parse("gate vector bytes not aligned to f32".into()));
    }
    let floats: Vec<f32> = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
    }
    .to_vec();
    Ok(floats)
}

// Simple base64 (no external dependency). Used by `encode_gate_vector`
// and indirectly by patch save / DIFF INTO PATCH.
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 { result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char); } else { result.push('='); }
        if chunk.len() > 2 { result.push(CHARS[(triple & 0x3F) as usize] as char); } else { result.push('='); }
    }
    result
}

fn base64_decode(input: &str) -> Result<Vec<u8>, VindexError> {
    fn val(c: u8) -> Result<u32, VindexError> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0),
            _ => Err(VindexError::Parse(format!("invalid base64 char: {c}"))),
        }
    }
    let input = input.as_bytes();
    let mut result = Vec::with_capacity(input.len() * 3 / 4);
    for chunk in input.chunks(4) {
        if chunk.len() < 4 { break; }
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        let c = val(chunk[2])?;
        let d = val(chunk[3])?;
        let triple = (a << 18) | (b << 12) | (c << 6) | d;
        result.push(((triple >> 16) & 0xFF) as u8);
        if chunk[2] != b'=' { result.push(((triple >> 8) & 0xFF) as u8); }
        if chunk[3] != b'=' { result.push((triple & 0xFF) as u8); }
    }
    Ok(result)
}
