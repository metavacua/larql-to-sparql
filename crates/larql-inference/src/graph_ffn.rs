//! Graph-based FFN backend — replaces the gate matmul with a precomputed index.
//!
//! Offline: for each layer, compute gate activations for every embedding token,
//! record the top-K features per token. This is the "graph" — a token→features map.
//!
//! Runtime: project residual into embedding space (find nearest tokens),
//! look up their precomputed feature lists, run sparse up/down on those features.
//!
//! Eliminates the gate matmul entirely. One embedding projection + hash lookup
//! replaces 500ms of BLAS.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use ndarray::Array2;

use crate::error::InferenceError;
use crate::ffn::sigmoid;
use crate::model::ModelWeights;

/// Precomputed gate index: for each (layer, token_id), which features activate.
/// Built offline from the gate weight matrix and embedding matrix.
/// Serializable to disk for reuse across predict calls.
pub struct GateIndex {
    /// layer → per-token feature lists. index[layer][token_id] = [(feature_id, gate_act), ...]
    index: HashMap<usize, Vec<Vec<(usize, f32)>>>,
    /// How many top tokens to match the residual against at runtime.
    pub top_tokens: usize,
    /// How many features were indexed per token (for metadata).
    pub features_per_token: usize,
}

/// Callbacks for gate index build progress.
pub trait IndexBuildCallbacks {
    fn on_layer_start(&mut self, _layer: usize, _total_layers: usize) {}
    fn on_layer_done(&mut self, _layer: usize, _elapsed_ms: f64) {}
}

pub struct SilentIndexCallbacks;
impl IndexBuildCallbacks for SilentIndexCallbacks {}

impl GateIndex {
    /// Build the gate index from model weights.
    ///
    /// For each layer, for each token in the vocabulary:
    /// 1. Compute `gate_activation = SiLU(embedding[token] * embed_scale @ gate.T)`
    /// 2. Store top `features_per_token` features by magnitude
    ///
    /// This is the expensive offline step — one gate matmul per layer.
    pub fn build(
        weights: &ModelWeights,
        layers: &[usize],
        features_per_token: usize,
        top_tokens: usize,
        callbacks: &mut dyn IndexBuildCallbacks,
    ) -> Self {
        let vocab_size = weights.vocab_size;
        let embed_scale = weights.arch.embed_scale();
        let total = layers.len();

        // Scale embeddings once (Gemma convention)
        let scaled_embed = &weights.embed * embed_scale;

        let mut index = HashMap::new();

        for (idx, &layer) in layers.iter().enumerate() {
            callbacks.on_layer_start(layer, total);
            let start = std::time::Instant::now();

            let gate_key = weights.arch.ffn_gate_key(layer);
            let w_gate = match weights.tensors.get(&gate_key) {
                Some(w) => w,
                None => continue,
            };

            let intermediate = w_gate.shape()[0];
            let k = features_per_token.min(intermediate);

            // Process tokens in batches to avoid OOM on the (vocab × intermediate) matrix.
            // 8192 tokens × 10240 features × 4 bytes = 320MB per batch (vs 10GB for full vocab).
            let batch_size = 8192;
            let mut layer_index: Vec<Vec<(usize, f32)>> = Vec::with_capacity(vocab_size);

            for batch_start in (0..vocab_size).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(vocab_size);
                let embed_chunk = scaled_embed.slice(ndarray::s![batch_start..batch_end, ..]);
                let gate_proj = embed_chunk.dot(&w_gate.t());

                for row_idx in 0..(batch_end - batch_start) {
                    let mut features: Vec<(usize, f32)> = gate_proj
                        .row(row_idx)
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(i, v)| (i, v * sigmoid(v)))
                        .collect();

                    if k < intermediate {
                        features.select_nth_unstable_by(k, |a, b| {
                            b.1.abs().partial_cmp(&a.1.abs()).unwrap()
                        });
                        features.truncate(k);
                    }
                    features.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                    layer_index.push(features);
                }
            }

            index.insert(layer, layer_index);

            let _ = idx; // used for progress via callbacks
            callbacks.on_layer_done(layer, start.elapsed().as_secs_f64() * 1000.0);
        }

        GateIndex {
            index,
            top_tokens,
            features_per_token,
        }
    }

    /// Build the gate index and stream directly to disk — never holds more than
    /// one layer's worth of index data in memory at a time.
    pub fn build_streaming(
        weights: &ModelWeights,
        layers: &[usize],
        features_per_token: usize,
        top_tokens: usize,
        path: &Path,
        callbacks: &mut dyn IndexBuildCallbacks,
    ) -> Result<(), InferenceError> {
        let vocab_size = weights.vocab_size;
        let embed_scale = weights.arch.embed_scale();
        let total = layers.len();

        let scaled_embed = &weights.embed * embed_scale;

        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header
        let header = serde_json::json!({
            "_header": true,
            "type": "gate_index",
            "top_tokens": top_tokens,
            "features_per_token": features_per_token,
            "layers": layers,
        });
        serde_json::to_writer(&mut writer, &header)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        writer.write_all(b"\n")?;

        for (idx, &layer) in layers.iter().enumerate() {
            callbacks.on_layer_start(layer, total);
            let start = std::time::Instant::now();

            let gate_key = weights.arch.ffn_gate_key(layer);
            let w_gate = match weights.tensors.get(&gate_key) {
                Some(w) => w,
                None => continue,
            };

            let intermediate = w_gate.shape()[0];
            let k = features_per_token.min(intermediate);
            let batch_size = 8192;
            let mut tok_id = 0usize;

            for batch_start in (0..vocab_size).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(vocab_size);
                let embed_chunk = scaled_embed.slice(ndarray::s![batch_start..batch_end, ..]);
                let gate_proj = embed_chunk.dot(&w_gate.t());

                for row_idx in 0..(batch_end - batch_start) {
                    let mut features: Vec<(usize, f32)> = gate_proj
                        .row(row_idx)
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(i, v)| (i, v * sigmoid(v)))
                        .collect();

                    if k < intermediate {
                        features.select_nth_unstable_by(k, |a, b| {
                            b.1.abs().partial_cmp(&a.1.abs()).unwrap()
                        });
                        features.truncate(k);
                    }
                    features.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

                    if !features.is_empty() {
                        let flat: Vec<f32> =
                            features.iter().flat_map(|&(f, a)| [f as f32, a]).collect();
                        let record = serde_json::json!({
                            "l": layer,
                            "t": tok_id,
                            "f": flat,
                        });
                        serde_json::to_writer(&mut writer, &record)
                            .map_err(|e| InferenceError::Parse(e.to_string()))?;
                        writer.write_all(b"\n")?;
                    }

                    tok_id += 1;
                }
            }

            writer.flush()?;
            let _ = idx;
            callbacks.on_layer_done(layer, start.elapsed().as_secs_f64() * 1000.0);
        }

        writer.flush()?;
        Ok(())
    }

    /// Save the gate index to an NDJSON file.
    /// Format: header line, then one line per (layer, token) entry.
    pub fn save(&self, path: &Path) -> Result<(), InferenceError> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header
        let header = serde_json::json!({
            "_header": true,
            "type": "gate_index",
            "top_tokens": self.top_tokens,
            "features_per_token": self.features_per_token,
            "layers": self.index.keys().collect::<Vec<_>>(),
        });
        serde_json::to_writer(&mut writer, &header)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        writer.write_all(b"\n")?;

        // One line per (layer, token) with compact feature lists
        let mut layers: Vec<usize> = self.index.keys().copied().collect();
        layers.sort();
        for layer in layers {
            let layer_data = &self.index[&layer];
            for (tok_id, features) in layer_data.iter().enumerate() {
                if features.is_empty() {
                    continue;
                }
                // Compact format: [feat_id, gate_act, feat_id, gate_act, ...]
                let flat: Vec<f32> = features.iter().flat_map(|&(f, a)| [f as f32, a]).collect();
                let record = serde_json::json!({
                    "l": layer,
                    "t": tok_id,
                    "f": flat,
                });
                serde_json::to_writer(&mut writer, &record)
                    .map_err(|e| InferenceError::Parse(e.to_string()))?;
                writer.write_all(b"\n")?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Load a gate index from an NDJSON file.
    pub fn load(path: &Path, top_tokens: usize) -> Result<Self, InferenceError> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);

        let mut index: HashMap<usize, Vec<Vec<(usize, f32)>>> = HashMap::new();
        let mut features_per_token = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| InferenceError::Parse(e.to_string()))?;

            if obj.get("_header").is_some() {
                features_per_token = obj["features_per_token"].as_u64().unwrap_or(100) as usize;
                continue;
            }

            let layer = obj["l"].as_u64().unwrap() as usize;
            let tok_id = obj["t"].as_u64().unwrap() as usize;
            let flat: Vec<f32> = obj["f"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();

            // Decode flat pairs: [feat_id, gate_act, feat_id, gate_act, ...]
            let features: Vec<(usize, f32)> = flat
                .chunks_exact(2)
                .map(|pair| (pair[0] as usize, pair[1]))
                .collect();

            let layer_vec = index.entry(layer).or_default();
            // Extend to fit tok_id
            while layer_vec.len() <= tok_id {
                layer_vec.push(Vec::new());
            }
            layer_vec[tok_id] = features;
        }

        Ok(GateIndex {
            index,
            top_tokens,
            features_per_token,
        })
    }

    /// Number of layers indexed.
    pub fn num_layers(&self) -> usize {
        self.index.len()
    }

    /// Total entries across all layers.
    pub fn total_entries(&self) -> usize {
        self.index.values().map(|v| v.len()).sum()
    }

    /// Look up candidate features from pre-resolved token matches.
    /// Returns deduplicated feature IDs (no activation values — caller computes real gates).
    pub fn lookup_from_tokens(
        &self,
        token_scores: &[(usize, f32)],
        layer: usize,
        total_k: usize,
    ) -> Vec<usize> {
        let layer_index = match self.index.get(&layer) {
            Some(idx) => idx,
            None => return vec![],
        };

        // Union features from matched tokens, dedup, keep highest-magnitude precomputed score for ranking
        let mut feature_map: HashMap<usize, f32> = HashMap::new();
        for &(tok_id, _) in token_scores {
            if tok_id < layer_index.len() {
                for &(feat_id, gate_act) in &layer_index[tok_id] {
                    let entry = feature_map.entry(feat_id).or_insert(0.0);
                    if gate_act.abs() > entry.abs() {
                        *entry = gate_act;
                    }
                }
            }
        }

        // Select top-K by precomputed activation magnitude (used only for ranking/selection)
        let mut features: Vec<(usize, f32)> = feature_map.into_iter().collect();
        let k = total_k.min(features.len());
        if k > 0 && k < features.len() {
            features.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            features.truncate(k);
        }
        features.into_iter().map(|(id, _)| id).collect()
    }

    /// Precompute entity feature lists for all layers at once.
    /// Returns a vec indexed by layer number (sparse — unlisted layers are empty).
    /// Zero allocation at query time — just index into the vec.
    pub fn precompute_entity(
        &self,
        token_ids: &[u32],
        top_k: usize,
    ) -> Vec<Vec<usize>> {
        let token_scores: Vec<(usize, f32)> = token_ids.iter().map(|&t| (t as usize, 1.0)).collect();
        let max_layer = self.index.keys().copied().max().unwrap_or(0);
        let mut result = vec![Vec::new(); max_layer + 1];
        for &layer in self.index.keys() {
            result[layer] = self.lookup_from_tokens(&token_scores, layer, top_k);
        }
        result
    }

    /// Look up which features should activate for a given residual at a layer.
    ///
    /// Projects residual against the embedding matrix, finds the top-N nearest tokens,
    /// unions their precomputed feature lists, deduplicates, returns top-K by activation.
    #[allow(dead_code)]
    fn lookup(
        &self,
        layer: usize,
        residual: &ndarray::ArrayView1<f32>,
        embed: &Array2<f32>,
        total_k: usize,
    ) -> Vec<(usize, f32)> {
        let layer_index = match self.index.get(&layer) {
            Some(idx) => idx,
            None => return vec![],
        };

        let vocab_size = embed.shape()[0];

        // Project residual against embedding matrix → (vocab_size,) logits
        // This is hidden-dim dot products, same cost as one row of the embed matrix.
        let mut token_scores: Vec<(usize, f32)> = Vec::with_capacity(vocab_size);
        for tok in 0..vocab_size {
            let row = embed.row(tok);
            let dot: f32 = residual.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            token_scores.push((tok, dot));
        }

        // Top-N tokens by score
        let n = self.top_tokens.min(vocab_size);
        if n < vocab_size {
            token_scores.select_nth_unstable_by(n, |a, b| b.1.partial_cmp(&a.1).unwrap());
            token_scores.truncate(n);
        }

        // Union all features from top-N tokens, dedup by feature_id (keep max activation)
        let mut feature_map: HashMap<usize, f32> = HashMap::new();
        for &(tok_id, _token_score) in &token_scores {
            if tok_id < layer_index.len() {
                for &(feat_id, gate_act) in &layer_index[tok_id] {
                    let entry = feature_map.entry(feat_id).or_insert(0.0);
                    if gate_act.abs() > entry.abs() {
                        *entry = gate_act;
                    }
                }
            }
        }

        // Collect and select top-K overall
        let mut features: Vec<(usize, f32)> = feature_map.into_iter().collect();
        let k = total_k.min(features.len());
        if k > 0 && k < features.len() {
            features.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            features.truncate(k);
        }
        features
    }
}
