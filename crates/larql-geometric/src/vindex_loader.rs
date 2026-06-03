// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
//
// Reads attention projection weights directly from larql vindex files.
// No dependency on larql-vindex crate — reads raw binary blobs directly.

use std::io;
use std::path::Path;

/// Weights for a single transformer layer's attention projections.
pub struct LayerAttnWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub hidden_dim: usize,
    pub kv_dim: usize,
}

fn read_f32_le(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "file length not divisible by 4"));
    }
    Ok(bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Load attention projection weights for one transformer layer from a larql vindex directory.
///
/// The vindex layout stores per-layer files under `layers/layer_NN/`.
/// Filenames: `attn_q.bin`, `attn_k.bin`, `attn_v.bin`, `attn_o.bin` (raw f32 LE).
///
/// Run `larql extract-index --level inference` to populate these files from a model checkpoint.
pub fn load_layer_attn(vindex_dir: &Path, layer: usize) -> io::Result<LayerAttnWeights> {
    let layer_dir = vindex_dir.join(format!("layers/layer_{layer:02}"));
    let q = read_f32_le(&layer_dir.join("attn_q.bin"))?;
    let k = read_f32_le(&layer_dir.join("attn_k.bin"))?;
    let v = read_f32_le(&layer_dir.join("attn_v.bin"))?;
    let o = read_f32_le(&layer_dir.join("attn_o.bin"))?;
    let hidden_dim = (q.len() as f64).sqrt() as usize;
    let kv_dim = if hidden_dim > 0 { k.len() / hidden_dim } else { 0 };
    Ok(LayerAttnWeights { q_proj: q, k_proj: k, v_proj: v, o_proj: o, hidden_dim, kv_dim })
}
