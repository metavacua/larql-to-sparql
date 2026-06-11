//! Manifest-driven reader for per-layer attention Q/K weight matrices.
//!
//! `attn_weights.bin` is a header-less concatenation of tensors; offsets and
//! shapes come from `weight_manifest.json` (a top-level JSON array). We read
//! only the `q_proj` / `k_proj` tensors — no tokenizer, no forward pass, and
//! no full-model load.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;
use serde::Deserialize;

use crate::config::dtype::{decode_floats, StorageDtype};
use crate::error::VindexError;
use crate::format::filenames::{ATTN_WEIGHTS_BIN, WEIGHT_MANIFEST_JSON};

#[derive(Deserialize)]
struct ManifestEntry {
    key: String,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
    file: String,
}

/// A `(W_Q, W_K)` pair for one transformer layer.
pub type QkPair = (Array2<f32>, Array2<f32>);

/// Load per-layer `(W_Q, W_K)` from `attn_weights.bin`.
/// `W_Q` is `[num_q_heads*head_dim, hidden]`, `W_K` is `[num_kv_heads*head_dim, hidden]`,
/// both as `f32` (decoded from on-disk f16/f32). Index in the returned Vec is the layer.
pub fn load_attention_qk(
    dir: &Path,
    num_layers: usize,
) -> Result<Vec<QkPair>, VindexError> {
    let manifest_bytes = std::fs::read(dir.join(WEIGHT_MANIFEST_JSON))?;
    let entries: Vec<ManifestEntry> = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| VindexError::Parse(format!("weight_manifest.json: {e}")))?;
    let by_key: HashMap<&str, &ManifestEntry> =
        entries.iter().map(|e| (e.key.as_str(), e)).collect();

    let bin = std::fs::read(dir.join(ATTN_WEIGHTS_BIN))?;

    let mut out = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let q_key = format!("layers.{layer}.self_attn.q_proj.weight");
        let k_key = format!("layers.{layer}.self_attn.k_proj.weight");
        let wq = read_entry(&bin, lookup(&by_key, &q_key)?)?;
        let wk = read_entry(&bin, lookup(&by_key, &k_key)?)?;
        out.push((wq, wk));
    }
    Ok(out)
}

fn lookup<'a>(
    by_key: &HashMap<&str, &'a ManifestEntry>,
    key: &str,
) -> Result<&'a ManifestEntry, VindexError> {
    by_key
        .get(key)
        .copied()
        .ok_or_else(|| VindexError::Parse(format!("weight_manifest.json missing key {key}")))
}

fn read_entry(bin: &[u8], e: &ManifestEntry) -> Result<Array2<f32>, VindexError> {
    if e.file != ATTN_WEIGHTS_BIN {
        return Err(VindexError::Parse(format!(
            "entry {} is in {}, expected {ATTN_WEIGHTS_BIN}",
            e.key, e.file
        )));
    }
    if e.shape.len() != 2 {
        return Err(VindexError::Parse(format!(
            "entry {} has non-2D shape {:?}",
            e.key, e.shape
        )));
    }
    let (rows, cols) = (e.shape[0], e.shape[1]);
    let n = rows * cols;
    if n == 0 {
        return Err(VindexError::Parse(format!("entry {} is empty", e.key)));
    }
    if !e.length.is_multiple_of(n) {
        return Err(VindexError::Parse(format!(
            "entry {} byte length {} is not a multiple of element count {n}",
            e.key, e.length
        )));
    }
    let dtype = match e.length / n {
        4 => StorageDtype::F32,
        2 => StorageDtype::F16,
        other => {
            return Err(VindexError::Parse(format!(
                "entry {} has {other} bytes/elem (expected 2 or 4)",
                e.key
            )))
        }
    };
    let end = e.offset.checked_add(e.length).ok_or_else(|| {
        VindexError::Parse(format!("entry {} offset+length overflows usize", e.key))
    })?;
    if end > bin.len() {
        return Err(VindexError::Parse(format!(
            "entry {} range {}..{end} exceeds {ATTN_WEIGHTS_BIN} ({} bytes)",
            e.key,
            e.offset,
            bin.len()
        )));
    }
    let floats = decode_floats(&bin[e.offset..end], dtype);
    Array2::from_shape_vec((rows, cols), floats)
        .map_err(|e| VindexError::Parse(format!("reshape attention weight: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_qk_from_a_synthetic_vindex() {
        let dir = tempfile::tempdir().unwrap();
        // Two f32 tensors: q_proj [2,2] then k_proj [2,2], concatenated.
        let q: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let k: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let mut bin = Vec::new();
        for v in q.iter().chain(k.iter()) {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.path().join(ATTN_WEIGHTS_BIN), &bin).unwrap();

        let manifest = serde_json::json!([
            {"key": "layers.0.self_attn.q_proj.weight", "shape": [2, 2],
             "offset": 0, "length": 16, "file": ATTN_WEIGHTS_BIN},
            {"key": "layers.0.self_attn.k_proj.weight", "shape": [2, 2],
             "offset": 16, "length": 16, "file": ATTN_WEIGHTS_BIN}
        ]);
        std::fs::write(
            dir.path().join(WEIGHT_MANIFEST_JSON),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let qk = load_attention_qk(dir.path(), 1).unwrap();
        assert_eq!(qk.len(), 1);
        let (wq, wk) = &qk[0];
        assert_eq!(wq.shape(), [2, 2]);
        assert_eq!(wk.shape(), [2, 2]);
        assert_eq!(wq[[0, 0]], 1.0);
        assert_eq!(wq[[1, 1]], 4.0);
        assert_eq!(wk[[0, 0]], 5.0);
        assert_eq!(wk[[1, 1]], 8.0);
    }

    #[test]
    fn missing_key_errors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(ATTN_WEIGHTS_BIN), [0u8; 16]).unwrap();
        std::fs::write(
            dir.path().join(WEIGHT_MANIFEST_JSON),
            serde_json::to_vec(&serde_json::json!([])).unwrap(),
        )
        .unwrap();
        assert!(load_attention_qk(dir.path(), 1).is_err());
    }
}
