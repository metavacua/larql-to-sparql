//! Browser/Node.js WASM bindings for larql-vindex gate-KNN.
//!
//! This crate is **entirely standalone** — it does not depend on
//! `larql-vindex`. It owns the minimal subset of config types and
//! computation needed for the wasm32 surface:
//!
//! 1. A lightweight `VindexConfig` (mirrors the shape of `index.json`).
//! 2. A pure in-memory `GateIndex` that decodes f32/f16 bytes and runs
//!    brute-force dot-product KNN on pre-loaded gate vectors.
//! 3. A `wasm_bindgen` `VindexSession` that wraps both.
//!
//! The native `larql-vindex` crate is not modified. All changes needed
//! to produce a valid wasm32 object live here.

use wasm_bindgen::prelude::*;

// ── Config types (subset of index.json) ──────────────────────────────────────

/// Storage precision used in `gate_vectors.bin`.
#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StorageDtype {
    #[default]
    F32,
    F16,
}

/// Per-layer layout entry from `index.json`.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize, Debug)]
pub struct VindexLayerInfo {
    pub layer: usize,
    /// Number of gate features stored for this layer.
    pub num_features: usize,
    /// Byte offset into `gate_vectors.bin`.
    pub offset: u64,
    /// Byte length in `gate_vectors.bin`.
    pub length: u64,
}

/// Minimal vindex config — the fields used by the wasm32 surface.
/// Serde ignores unknown fields so any `index.json` produced by the
/// native toolchain round-trips correctly.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize, Debug)]
pub struct VindexConfig {
    /// Original model name.
    pub model: String,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Storage precision for gate vectors.
    #[serde(default)]
    pub dtype: StorageDtype,
    /// Per-layer layout.
    #[serde(default)]
    pub layers: Vec<VindexLayerInfo>,
}

// ── Pure gate KNN ─────────────────────────────────────────────────────────────

/// In-memory gate index. Holds one flat f32 array per layer.
struct GateIndex {
    hidden_size: usize,
    /// `data[layer]` = flat `num_features × hidden_size` f32 row-major.
    data: Vec<Option<Vec<f32>>>,
    /// `num_features[layer]` matches `data[layer].len() / hidden_size`.
    num_features: Vec<usize>,
}

impl GateIndex {
    fn new(num_layers: usize, hidden_size: usize) -> Self {
        Self {
            hidden_size,
            data: vec![None; num_layers],
            num_features: vec![0; num_layers],
        }
    }

    fn load_layer(&mut self, layer: usize, raw: &[u8], dtype: StorageDtype) {
        let floats = decode_floats(raw, dtype);
        if self.hidden_size == 0 || floats.len() < self.hidden_size {
            return;
        }
        let nf = floats.len() / self.hidden_size;
        self.num_features[layer] = nf;
        self.data[layer] = Some(floats);
    }

    /// Brute-force dot-product KNN. Returns `(feature_idx, score)` sorted
    /// by descending absolute score, capped at `k`.
    fn gate_knn(&self, layer: usize, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let Some(vecs) = self.data.get(layer).and_then(|v| v.as_ref()) else {
            return vec![];
        };
        let h = self.hidden_size;
        if h == 0 || query.len() < h {
            return vec![];
        }
        let nf = vecs.len() / h;
        let mut scores: Vec<(usize, f32)> = (0..nf)
            .map(|i| {
                let row = &vecs[i * h..(i + 1) * h];
                let dot: f32 = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();
        scores.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores
    }

    fn num_features(&self, layer: usize) -> usize {
        self.num_features.get(layer).copied().unwrap_or(0)
    }

    fn total_gate_vectors(&self) -> usize {
        self.num_features.iter().sum()
    }
}

// ── f32 / f16 decode ─────────────────────────────────────────────────────────

fn decode_floats(data: &[u8], dtype: StorageDtype) -> Vec<f32> {
    match dtype {
        StorageDtype::F32 => {
            let n = data.len() / 4;
            (0..n)
                .map(|i| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap()))
                .collect()
        }
        StorageDtype::F16 => {
            let n = data.len() / 2;
            (0..n)
                .map(|i| {
                    let bytes: [u8; 2] = data[i * 2..i * 2 + 2].try_into().unwrap();
                    half::f16::from_le_bytes(bytes).to_f32()
                })
                .collect()
        }
    }
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
        let results = self.gate.gate_knn(layer as usize, query_f32, k as usize);
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
        self.gate.num_features(layer as usize) as u32
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

    #[wasm_bindgen_test]
    fn decode_f32_roundtrip() {
        let data = vec![1.0f32, -2.5, 3.14];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let decoded = decode_floats(&bytes, StorageDtype::F32);
        assert_eq!(decoded.len(), 3);
        assert!((decoded[0] - 1.0).abs() < 1e-6);
        assert!((decoded[1] - (-2.5)).abs() < 1e-6);
    }

    #[wasm_bindgen_test]
    fn decode_f16_roundtrip() {
        let orig = 1.5f32;
        let half_bytes = half::f16::from_f32(orig).to_le_bytes();
        let decoded = decode_floats(&half_bytes, StorageDtype::F16);
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - orig).abs() < 0.01);
    }

    #[wasm_bindgen_test]
    fn gate_knn_returns_top_k() {
        let mut gate = GateIndex::new(1, 2);
        // Two features: [1,0] and [0,1]. Query [1,0] → feature 0 scores 1.0.
        let raw: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        gate.load_layer(0, &raw, StorageDtype::F32);
        let results = gate.gate_knn(0, &[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }
}
