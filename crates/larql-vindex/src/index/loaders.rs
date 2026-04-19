//! NDJSON loaders for `VectorIndex` — read gate vectors and down
//! metadata from `ffn_gate.vectors.jsonl` / `ffn_down.meta.jsonl`.
//!
//! These are the heap-mode constructors. The mmap-mode entry point
//! `VectorIndex::new_mmap` lives in `super::core` next to `new`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Mutex;

use ndarray::Array2;
use larql_models::TopKEntry;

use crate::error::VindexError;

use super::core::VectorIndex;
use super::types::*;

impl VectorIndex {
    pub fn load_gates(
        path: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, VindexError> {
        callbacks.on_file_start("ffn_gate", &path.display().to_string());
        let start = std::time::Instant::now();

        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(1 << 20, file);

        // First pass: collect all records to determine dimensions
        let mut records: Vec<(usize, usize, Vec<f32>, FeatureMeta)> = Vec::new();
        let mut hidden_size = 0;
        let mut max_layer = 0;
        let mut count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| VindexError::Parse(e.to_string()))?;

            if obj.get("_header").is_some() {
                if let Some(dim) = obj.get("dimension").and_then(|v| v.as_u64()) {
                    hidden_size = dim as usize;
                }
                continue;
            }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;

            let vector: Vec<f32> = obj["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();

            if hidden_size == 0 {
                hidden_size = vector.len();
            }

            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<TopKEntry> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|entry| {
                        Some(TopKEntry {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    })
                    .collect(),
                None => vec![],
            };

            let meta = FeatureMeta {
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            if layer > max_layer {
                max_layer = layer;
            }

            records.push((layer, feature, vector, meta));

            count += 1;
            if count % 10000 == 0 {
                callbacks.on_progress(count);
            }
        }

        let num_layers = max_layer + 1;

        // Group by layer, find max feature per layer
        let mut layer_sizes: HashMap<usize, usize> = HashMap::new();
        for &(layer, feature, _, _) in &records {
            let entry = layer_sizes.entry(layer).or_insert(0);
            if feature + 1 > *entry {
                *entry = feature + 1;
            }
        }

        // Build per-layer matrices
        let mut gate_vectors: Vec<Option<Array2<f32>>> = vec![None; num_layers];
        let mut gate_meta: Vec<Option<Vec<Option<FeatureMeta>>>> = vec![None; num_layers];

        // Pre-allocate
        for (&layer, &num_features) in &layer_sizes {
            gate_vectors[layer] = Some(Array2::zeros((num_features, hidden_size)));
            gate_meta[layer] = Some(vec![None; num_features]);
        }

        // Fill
        for (layer, feature, vector, meta) in records {
            if let Some(ref mut matrix) = gate_vectors[layer] {
                for (j, &val) in vector.iter().enumerate() {
                    matrix[[feature, j]] = val;
                }
            }
            if let Some(ref mut metas) = gate_meta[layer] {
                metas[feature] = Some(meta);
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_file_done("ffn_gate", count, elapsed_ms);

        Ok(VectorIndex {
            gate_vectors,
            gate_mmap_bytes: None,
            gate_mmap_dtype: crate::config::dtype::StorageDtype::F32,
            gate_mmap_slices: Vec::new(),
            down_meta: gate_meta,
            down_meta_mmap: None,
            down_overrides: HashMap::new(),
            up_overrides: HashMap::new(),
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            gate_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            gate_cache_max_layers: std::sync::atomic::AtomicUsize::new(0),
            warmed_gates: std::sync::RwLock::new(vec![None; num_layers]),
            down_features_mmap: None,
            up_features_mmap: None,
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
            lm_head_mmap: None,
            lm_head_f16_mmap: None,
            vocab_size: 0,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            interleaved_q4k_mmap: None,
            interleaved_q4k_manifest: None,
            q4k_ffn_cache: Mutex::new((0..num_layers).map(|_| [None, None, None]).collect()),
            gate_q4_mmap: None,
            gate_q4_slices: Vec::new(),
            lm_head_q4_mmap: None,
            lm_head_q4_synth: None,
            attn_q4k_mmap: None,
            attn_q4k_manifest: None,
            attn_q4_mmap: None,
            attn_q4_manifest: None,
            attn_q8_mmap: None,
            attn_q8_manifest: None,
            num_layers,
            hidden_size,
            layer_range: None,
        })
    }

    /// Load down-projection token metadata from an NDJSON file (ffn_down.vectors.jsonl).
    ///
    /// Only loads the metadata (top_token, top_k, c_score), NOT the full vectors.
    /// This replaces any gate-file metadata with the down-projection metadata,
    /// which tells you what each feature *outputs* rather than what it *responds to*.
    pub fn load_down_meta(
        &mut self,
        path: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<usize, VindexError> {
        callbacks.on_file_start("ffn_down", &path.display().to_string());
        let start = std::time::Instant::now();

        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(1 << 20, file);
        let mut count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| VindexError::Parse(e.to_string()))?;

            if obj.get("_header").is_some() {
                continue;
            }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;

            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<TopKEntry> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|entry| {
                        Some(TopKEntry {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    })
                    .collect(),
                None => vec![],
            };

            let meta = FeatureMeta {
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            if layer < self.num_layers {
                // Ensure layer slot exists
                while self.down_meta.len() <= layer {
                    self.down_meta.push(None);
                }
                if self.down_meta[layer].is_none() {
                    self.down_meta[layer] = Some(Vec::new());
                }
                if let Some(ref mut metas) = self.down_meta[layer] {
                    while metas.len() <= feature {
                        metas.push(None);
                    }
                    metas[feature] = Some(meta);
                }
            }

            count += 1;
            if count % 10000 == 0 {
                callbacks.on_progress(count);
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_file_done("ffn_down", count, elapsed_ms);

        Ok(count)
    }

}
