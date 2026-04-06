//! AppState: loaded vindex + config, shared across all handlers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use larql_models::ModelWeights;
use larql_vindex::{PatchedVindex, VindexConfig, ndarray::Array2, tokenizers};
use tokio::sync::RwLock;

use crate::cache::DescribeCache;
use crate::session::SessionManager;

/// A single loaded model.
pub struct LoadedModel {
    /// Model ID derived from config (e.g., "gemma-3-4b-it").
    pub id: String,
    /// Vindex directory on disk.
    pub path: PathBuf,
    /// Vindex config (index.json).
    pub config: VindexConfig,
    /// Base index with patch overlay (starts with no patches).
    pub patched: RwLock<PatchedVindex>,
    /// Embeddings matrix + scale factor, loaded once.
    pub embeddings: Array2<f32>,
    pub embed_scale: f32,
    /// Tokenizer for embedding lookups.
    pub tokenizer: tokenizers::Tokenizer,
    /// Whether inference is disabled (--no-infer).
    pub infer_disabled: bool,
    /// Model weights, lazy-loaded on first INFER request.
    pub weights: std::sync::OnceLock<ModelWeights>,
    /// Probe-confirmed feature labels: (layer, feature) → relation name.
    /// Loaded from feature_labels.json if present.
    pub probe_labels: HashMap<(usize, usize), String>,
}

impl LoadedModel {
    /// Get or lazy-load model weights for inference.
    pub fn get_or_load_weights(&self) -> Result<&ModelWeights, String> {
        if let Some(w) = self.weights.get() {
            return Ok(w);
        }
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(&self.path, &mut cb)
            .map_err(|e| format!("failed to load model weights: {e}"))?;
        let _ = self.weights.set(weights);
        Ok(self.weights.get().unwrap())
    }
}

/// Shared application state.
pub struct AppState {
    /// Loaded models, keyed by model ID.
    pub models: Vec<Arc<LoadedModel>>,
    /// Server start time for uptime reporting.
    pub started_at: std::time::Instant,
    /// Request counter.
    pub requests_served: std::sync::atomic::AtomicU64,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
    /// Per-session PatchedVindex manager.
    pub sessions: SessionManager,
    /// DESCRIBE result cache.
    pub describe_cache: DescribeCache,
}

impl AppState {
    /// Get model by ID, or the only model if single-model serving.
    pub fn model(&self, id: Option<&str>) -> Option<&Arc<LoadedModel>> {
        match id {
            Some(id) => self.models.iter().find(|m| m.id == id),
            None if self.models.len() == 1 => self.models.first(),
            None => None,
        }
    }

    /// Whether this is multi-model serving.
    pub fn is_multi_model(&self) -> bool {
        self.models.len() > 1
    }

    pub fn bump_requests(&self) {
        self.requests_served
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Load probe-confirmed feature labels from feature_labels.json.
/// Format: {"L{layer}_F{feature}": "relation_name", ...}
pub fn load_probe_labels(vindex_path: &std::path::Path) -> HashMap<(usize, usize), String> {
    let path = vindex_path.join("feature_labels.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return HashMap::new(),
    };
    let obj: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return HashMap::new(),
    };
    let map = match obj.as_object() {
        Some(m) => m,
        None => return HashMap::new(),
    };

    let mut labels = HashMap::new();
    for (key, value) in map {
        if let Some(rel) = value.as_str() {
            let parts: Vec<&str> = key.split('_').collect();
            if parts.len() == 2 {
                if let (Some(layer), Some(feat)) = (
                    parts[0].strip_prefix('L').and_then(|s| s.parse::<usize>().ok()),
                    parts[1].strip_prefix('F').and_then(|s| s.parse::<usize>().ok()),
                ) {
                    labels.insert((layer, feat), rel.to_string());
                }
            }
        }
    }
    labels
}

/// Derive a short model ID from the full model name.
/// "google/gemma-3-4b-it" → "gemma-3-4b-it"
pub fn model_id_from_name(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}
