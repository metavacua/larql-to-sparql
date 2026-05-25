//! Browser/Node.js WASM bindings for larql-vindex.
//!
//! Exposes a compressed subgraph view of a VIndex — gate-vector KNN,
//! feature metadata, and config inspection — without requiring OS-level
//! mmap or file I/O. Callers (JS/TS) are expected to:
//!
//! 1. Fetch `index.json` and pass the JSON string to `VindexSession::new`.
//! 2. Optionally fetch `gate_vectors.bin` and pass it to `VindexSession::load_gate_bytes`.
//! 3. Call `gate_knn`, `feature_meta`, and `config` as needed.
//!
//! This mirrors the native load path but replaces file mmap with caller-
//! supplied `Uint8Array` buffers, which JS can populate via `fetch()`.

use wasm_bindgen::prelude::*;

use larql_vindex::config::{dtype, VindexConfig};
use larql_vindex::index::VectorIndex;

/// An in-memory vindex session for browser / Node.js consumers.
///
/// Load `index.json`, optionally load `gate_vectors.bin`, then query.
#[wasm_bindgen]
pub struct VindexSession {
    inner: VectorIndex,
    config: VindexConfig,
}

#[wasm_bindgen]
impl VindexSession {
    /// Build a session from a serialised `index.json` string.
    ///
    /// Gate vectors are not loaded yet — call `load_gate_bytes` after
    /// fetching `gate_vectors.bin` to enable KNN queries.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<VindexSession, JsValue> {
        let config: VindexConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("config parse error: {e}")))?;
        let inner = VectorIndex::new(
            vec![None; config.num_layers],
            vec![None; config.num_layers],
            config.num_layers,
            config.hidden_size,
        );
        Ok(VindexSession { inner, config })
    }

    /// Load raw gate vectors from a `Uint8Array` (the content of `gate_vectors.bin`).
    ///
    /// Decodes f32/f16 bytes per the layer layout declared in the config and
    /// promotes the gate vectors onto the heap, enabling `gate_knn`. Idempotent
    /// — calling again replaces the previous vectors.
    pub fn load_gate_bytes(&mut self, gate_bytes: &[u8]) -> Result<(), JsValue> {
        let hidden = self.config.hidden_size;
        let mut gate_vectors = vec![None; self.config.num_layers];

        for (layer_idx, info) in self.config.layers.iter().enumerate() {
            if info.num_features == 0 || info.length == 0 {
                continue;
            }
            let start = info.offset as usize;
            let end = start + info.length as usize;
            if end > gate_bytes.len() {
                return Err(JsValue::from_str(&format!(
                    "gate_vectors.bin too short: layer {} needs bytes {start}..{end}, got {}",
                    info.layer,
                    gate_bytes.len()
                )));
            }
            let slice = &gate_bytes[start..end];
            let floats = dtype::decode_floats(slice, self.config.dtype);
            let num_feat = floats.len() / hidden;
            if num_feat == 0 || num_feat != info.num_features {
                continue;
            }
            let arr = ndarray::Array2::from_shape_vec((num_feat, hidden), floats)
                .map_err(|e| JsValue::from_str(&format!("layer {layer_idx} shape: {e}")))?;
            if layer_idx < gate_vectors.len() {
                gate_vectors[layer_idx] = Some(arr);
            }
        }

        self.inner = VectorIndex::new(
            gate_vectors,
            vec![None; self.config.num_layers],
            self.config.num_layers,
            self.config.hidden_size,
        );
        Ok(())
    }

    /// KNN gate routing for a single layer.
    ///
    /// `query_f32` must be a flat `Float32Array` of length `hidden_size`.
    /// Returns a JSON array `[{"feature": N, "score": F}, ...]` ordered
    /// by descending absolute score, capped at `k`.
    ///
    /// Returns an empty array when no gate vectors are loaded for the layer.
    pub fn gate_knn(&self, layer: u32, query_f32: &[f32], k: u32) -> String {
        let layer = layer as usize;
        let k = k as usize;
        let query = ndarray::Array1::from_vec(query_f32.to_vec());

        let results = self.inner.gate_knn(layer, &query, k);
        let items: Vec<serde_json::Value> = results
            .iter()
            .map(|(feat, score)| serde_json::json!({"feature": feat, "score": score}))
            .collect();
        serde_json::to_string(&items).unwrap_or_else(|_| "[]".to_string())
    }

    /// Return the feature metadata for `(layer, feature)` as JSON, or `"null"`.
    ///
    /// Schema: `{"top_token": "...", "top_token_id": N, "c_score": F,
    ///           "top_k": [{"token": "...", "token_id": N, "score": F}, ...]}`
    pub fn feature_meta(&self, layer: u32, feature: u32) -> String {
        let Some(m) = self.inner.feature_meta(layer as usize, feature as usize) else {
            return "null".to_string();
        };
        let top_k: Vec<serde_json::Value> = m
            .top_k
            .iter()
            .map(|e| {
                serde_json::json!({
                    "token": e.token,
                    "token_id": e.token_id,
                    "logit": e.logit,
                })
            })
            .collect();
        serde_json::json!({
            "top_token": m.top_token,
            "top_token_id": m.top_token_id,
            "c_score": m.c_score,
            "top_k": top_k,
        })
        .to_string()
    }

    /// Serialised `VindexConfig` JSON for inspection.
    pub fn config(&self) -> String {
        serde_json::to_string(&self.config).unwrap_or_else(|_| "{}".to_string())
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> u32 {
        self.config.num_layers as u32
    }

    /// Hidden dimension (embedding size).
    pub fn hidden_size(&self) -> u32 {
        self.config.hidden_size as u32
    }

    /// Number of gate features loaded for `layer` (0 if not loaded).
    pub fn num_features(&self, layer: u32) -> u32 {
        self.inner.num_features(layer as usize) as u32
    }

    /// Model name from the config (e.g. `"google/gemma-3-4b-it"`).
    pub fn model_name(&self) -> String {
        self.config.model.clone()
    }

    /// Whether gate vectors have been loaded (via `load_gate_bytes`).
    pub fn has_gate_vectors(&self) -> bool {
        self.inner.total_gate_vectors() > 0
    }
}
