//! Browser/Node.js WASM bindings for larql-vindex gate-KNN.
//!
//! This crate is **entirely standalone** — it does not depend on
//! `larql-vindex`. It owns the minimal subset of config types and
//! wasm-bindgen surface needed for the wasm32 tier:
//!
//! 1. A lightweight `VindexConfig` (mirrors the shape of `index.json`).
//! 2. A `wasm_bindgen` `VindexSession` that wraps the pure-compute
//!    `GateIndex` from `larql-wasm32v1-none-lib`.
//!
//! The native `larql-vindex` crate is not modified. All changes needed
//! to produce a valid wasm32 object live here or in larql-wasm32v1-none-lib.

use wasm_bindgen::prelude::*;
use larql_wasm32v1_none_lib::gate::{
    decode::StorageDtype,
    index::GateIndex,
    knn::gate_knn,
};

// ── Config types (subset of index.json) ──────────────────────────────────────

// StorageDtype is re-exported from larql-wasm32v1-none-lib (with serde enabled).
// Re-export it here so callers that import from this crate can use it.
pub use larql_wasm32v1_none_lib::gate::decode::StorageDtype as GateStorageDtype;

/// Per-layer layout entry from `index.json`.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize, Debug)]
pub struct VindexLayerInfo {
    pub layer: usize,
    pub num_features: usize,
    pub offset: u64,
    pub length: u64,
}

/// Minimal vindex config — the fields used by the wasm32 surface.
/// Serde ignores unknown fields so any `index.json` produced by the
/// native toolchain round-trips correctly.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize, Debug)]
pub struct VindexConfig {
    pub model: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub dtype: StorageDtype,
    #[serde(default)]
    pub layers: Vec<VindexLayerInfo>,
}

// ── wasm_bindgen surface ──────────────────────────────────────────────────────

/// An in-memory vindex session for browser / Node.js consumers.
///
/// 1. Parse `index.json` → `VindexSession::new(json_str)`.
/// 2. Fetch `gate_vectors.bin` and pass it to `load_gate_bytes`.
/// 3. Call `gate_knn`, `config`, `num_layers`, etc.
#[wasm_bindgen]
pub struct VindexSession {
    config: VindexConfig,
    gate: GateIndex,
}

#[wasm_bindgen]
impl VindexSession {
    /// Build a session from the serialised `index.json` string.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<VindexSession, JsValue> {
        let config: VindexConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("config parse error: {e}")))?;
        let gate = GateIndex::new(config.num_layers, config.hidden_size);
        Ok(VindexSession { config, gate })
    }

    /// Load raw gate vectors from a `Uint8Array` (the content of `gate_vectors.bin`).
    pub fn load_gate_bytes(&mut self, gate_bytes: &[u8]) -> Result<(), JsValue> {
        for info in &self.config.layers {
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
            self.gate
                .load_layer(info.layer, &gate_bytes[start..end], self.config.dtype);
        }
        Ok(())
    }

    /// KNN gate routing for a single layer.
    ///
    /// `query_f32` must be a flat `Float32Array` of length `hidden_size`.
    /// Returns a JSON array `[{"feature": N, "score": F}, ...]` ordered by
    /// descending absolute score, capped at `k`.
    pub fn gate_knn(&self, layer: u32, query_f32: &[f32], k: u32) -> String {
        let results = gate_knn(&self.gate, layer as usize, query_f32, k as usize);
        let items: Vec<serde_json::Value> = results
            .iter()
            .map(|(feat, score)| serde_json::json!({"feature": feat, "score": score}))
            .collect();
        serde_json::to_string(&items).unwrap_or_else(|_| "[]".to_string())
    }

    /// Serialised `VindexConfig` JSON for inspection.
    pub fn config(&self) -> String {
        serde_json::to_string(&self.config).unwrap_or_else(|_| "{}".to_string())
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> u32 {
        self.config.num_layers as u32
    }

    /// Hidden dimension.
    pub fn hidden_size(&self) -> u32 {
        self.config.hidden_size as u32
    }

    /// Number of gate features loaded for `layer` (0 if not loaded).
    pub fn num_features(&self, layer: u32) -> u32 {
        self.gate.layer_num_features(layer as usize) as u32
    }

    /// Model name from the config.
    pub fn model_name(&self) -> String {
        self.config.model.clone()
    }

    /// Whether gate vectors have been loaded via `load_gate_bytes`.
    pub fn has_gate_vectors(&self) -> bool {
        self.gate.total_gate_vectors() > 0
    }
}

// ── wasm32 smoke tests ────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_node_experimental);

    use super::*;

    fn minimal_config_json(num_layers: usize, hidden_size: usize) -> String {
        format!(
            r#"{{"model":"test","num_layers":{num_layers},"hidden_size":{hidden_size},"layers":[]}}"#
        )
    }

    #[wasm_bindgen_test]
    fn session_new_parses_config() {
        let s = VindexSession::new(&minimal_config_json(4, 8)).unwrap();
        assert_eq!(s.num_layers(), 4);
        assert_eq!(s.hidden_size(), 8);
        assert_eq!(s.model_name(), "test");
    }

    #[wasm_bindgen_test]
    fn session_empty_knn_returns_empty_array() {
        let s = VindexSession::new(&minimal_config_json(4, 8)).unwrap();
        let result = s.gate_knn(0, &[0.0f32; 8], 5);
        assert_eq!(result, "[]");
    }

    #[wasm_bindgen_test]
    fn session_config_roundtrips() {
        let json = minimal_config_json(4, 8);
        let s = VindexSession::new(&json).unwrap();
        let back: serde_json::Value = serde_json::from_str(&s.config()).unwrap();
        assert_eq!(back["num_layers"], 4);
        assert_eq!(back["hidden_size"], 8);
    }
}
