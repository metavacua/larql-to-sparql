//! Attention weight loaders + per-layer accessors.
//!
//! Loads the per-layer Q / K / V / O projection weights in Q8, Q4_K, or
//! Q4_0 format from `attn_weights_*.bin` files plus their JSON
//! manifests. Mirrors the FFN walk plumbing in `super::walk`; lives in
//! its own file so attention storage isn't tangled with FFN storage.

use std::sync::Arc;

use crate::error::VindexError;
use crate::mmap_util::mmap_optimized;

use super::core::VectorIndex;

impl VectorIndex {
    /// Load Q8 attention weights + manifest for GPU full pipeline.
    pub fn load_attn_q8(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("attn_weights_q8.bin");
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q8.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.attn_q8_mmap = Some(Arc::new(mmap));

        let manifest_path = dir.join("attn_weights_q8_manifest.json");
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?
            ).map_err(|e| VindexError::Parse(e.to_string()))?;

            let entries: Vec<(usize, usize, usize)> = json.iter()
                .map(|e| {
                    let offset = e["q8_offset"].as_u64().unwrap_or(0) as usize;
                    let vals_len = e["q8_vals_len"].as_u64().unwrap_or(0) as usize;
                    let scales_len = e["q8_scales_len"].as_u64().unwrap_or(0) as usize;
                    (offset, vals_len, scales_len)
                })
                .collect();
            self.attn_q8_manifest = Some(entries);
        }
        Ok(())
    }

    /// Get per-layer Q8 attention slices: (q_vals, q_scales, k_vals, k_scales, v_vals, v_scales, o_vals, o_scales)
    pub fn attn_q8_layer_data(&self, layer: usize) -> Option<[(&[u8], &[f32]); 4]> {
        let mmap = self.attn_q8_mmap.as_ref()?;
        let manifest = self.attn_q8_manifest.as_ref()?;

        let base = layer * 4;
        if base + 3 >= manifest.len() { return None; }

        let mut result = [(&[] as &[u8], &[] as &[f32]); 4];
        for i in 0..4 {
            let (offset, vals_len, scales_len) = manifest[base + i];
            let vals = &mmap[offset..offset + vals_len];
            let scales_start = offset + vals_len;
            let scales_data = &mmap[scales_start..scales_start + scales_len];
            let scales = unsafe {
                std::slice::from_raw_parts(
                    scales_data.as_ptr() as *const f32,
                    scales_len / 4,
                )
            };
            result[i] = (vals, scales);
        }
        Some(result)
    }

    /// Load Q4_K/Q6_K attention weights for Ollama-compatible GPU pipeline.
    pub fn load_attn_q4k(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("attn_weights_q4k.bin");
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q4k.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };

        let manifest_path = dir.join("attn_weights_q4k_manifest.json");
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?
            ).map_err(|e| VindexError::Parse(e.to_string()))?;

            // Each entry: {key, shape, format, offset, length}
            let entries: Vec<(usize, usize, String)> = json.iter()
                .map(|e| {
                    let offset = e["offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["length"].as_u64().unwrap_or(0) as usize;
                    let format = e["format"].as_str().unwrap_or("Q4_K").to_string();
                    (offset, length, format)
                })
                .collect();
            self.attn_q4k_manifest = Some(entries);
        }
        self.attn_q4k_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Get per-layer Q4_K/Q6_K attention slices: (data, format) for Q, K, V, O.
    pub fn attn_q4k_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        let mmap = self.attn_q4k_mmap.as_ref()?;
        let manifest = self.attn_q4k_manifest.as_ref()?;
        let base = layer * 4;
        if base + 3 >= manifest.len() { return None; }

        let mut result: [(&[u8], &str); 4] = [(&[], ""); 4];
        for i in 0..4 {
            let (offset, length, ref format) = manifest[base + i];
            result[i] = (&mmap[offset..offset + length], format.as_str());
        }
        Some(result)
    }

    /// Load Q4 attention weights + manifest for GPU full pipeline.
    pub fn load_attn_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("attn_weights_q4.bin");
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.attn_q4_mmap = Some(Arc::new(mmap));

        // Load manifest with per-matrix offsets
        let manifest_path = dir.join("attn_weights_q4_manifest.json");
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?
            ).map_err(|e| VindexError::Parse(e.to_string()))?;

            let entries: Vec<(usize, usize)> = json.iter()
                .map(|e| {
                    let offset = e["q4_offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["q4_length"].as_u64().unwrap_or(0) as usize;
                    (offset, length)
                })
                .collect();
            self.attn_q4_manifest = Some(entries);
        }
        Ok(())
    }

    /// Get raw Q4 attention weight bytes (all layers packed).
    pub fn attn_q4_data(&self) -> Option<&[u8]> {
        self.attn_q4_mmap.as_ref().map(|m| m.as_ref() as &[u8])
    }

    /// Get per-layer Q4 attention weight slices (Q, K, V, O) using the manifest.
    /// Returns None if manifest or Q4 attn data is not loaded.
    #[allow(clippy::type_complexity)]
    pub fn attn_q4_layer_slices(&self, layer: usize) -> Option<(&[u8], &[u8], &[u8], &[u8])> {
        let mmap = self.attn_q4_mmap.as_ref()?;
        let manifest = self.attn_q4_manifest.as_ref()?;

        // Each layer has 4 tensors: Q, K, V, O
        let base = layer * 4;
        if base + 3 >= manifest.len() { return None; }

        let q = &manifest[base];
        let k = &manifest[base + 1];
        let v = &manifest[base + 2];
        let o = &manifest[base + 3];

        let q_data = &mmap[q.0..q.0 + q.1];
        let k_data = &mmap[k.0..k.0 + k.1];
        let v_data = &mmap[v.0..v.0 + v.1];
        let o_data = &mmap[o.0..o.0 + o.1];

        Some((q_data, k_data, v_data, o_data))
    }

}
