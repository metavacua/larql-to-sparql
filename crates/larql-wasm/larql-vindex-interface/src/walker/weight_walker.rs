//! Walk FFN weight matrices — extract edges directly from model parameters.
//!
//! Every FFN feature is a potential edge. For each feature in a layer:
//! - What tokens activate it (gate projection) → c_in
//! - What tokens it produces (down projection) → c_out
//! - Confidence = c_in × c_out, normalized per-layer to [0, 1]
//!
//! Zero forward passes. Pure matrix multiplication.

use larql_core::core::edge::Edge;
use larql_core::core::enums::SourceType;
use larql_core::core::graph::Graph;
use larql_models::{load_model_dir_validated, resolve_model_path, ModelWeights};

use super::utils::{count_threshold, decode_token, partial_top_k_column, top_entities};
use crate::error::VindexError;
use crate::format::filenames::*;

/// Result of walking a single layer.
#[derive(Debug, Clone)]
pub struct LayerResult {
    pub layer: usize,
    pub features_scanned: usize,
    pub edges_found: usize,
    pub elapsed_ms: f64,
    pub stats: LayerStats,
}

/// Per-layer statistics for validation.
#[derive(Debug, Clone, Default)]
pub struct LayerStats {
    pub mean_confidence: f64,
    pub max_confidence: f64,
    pub min_confidence: f64,
    pub mean_c_in: f64,
    pub mean_c_out: f64,
    pub mean_selectivity: f64,
    pub max_selectivity: f64,
    pub self_loop_count: usize,
    pub self_loop_pct: f64,
    pub top_subjects: Vec<(String, usize, f64)>,
    pub top_objects: Vec<(String, usize, f64)>,
    /// Edges surviving at each confidence threshold.
    pub threshold_counts: ThresholdCounts,
    /// Edges surviving at each selectivity threshold.
    pub selectivity_threshold_counts: ThresholdCounts,
}

/// Edge counts at standard thresholds.
#[derive(Debug, Clone, Default)]
pub struct ThresholdCounts {
    pub t_01: usize,
    pub t_05: usize,
    pub t_10: usize,
    pub t_25: usize,
    pub t_50: usize,
    pub t_75: usize,
    pub t_90: usize,
}

/// Configuration for the weight walker.
pub struct WalkConfig {
    pub top_k: usize,
    pub min_score: f32,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.02,
        }
    }
}

/// Callbacks for walk progress.
pub trait WalkCallbacks {
    fn on_layer_start(&mut self, _layer: usize, _num_features: usize) {}
    fn on_layer_done(&mut self, _result: &LayerResult) {}
    fn on_checkpoint(&mut self, _graph: &Graph) {}
    fn on_progress(&mut self, _layer: usize, _features_done: usize, _total: usize) {}
}

pub struct SilentWalkCallbacks;
impl WalkCallbacks for SilentWalkCallbacks {}

// resolve_model_path is re-exported from crate::model via the import above.

/// A loaded model ready for weight walking.
pub struct WeightWalker {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
}

/// A raw edge before per-layer normalization.
struct RawEdge {
    subject: String,
    relation: String,
    object: String,
    c_in: f32,
    c_out: f32,
    layer: usize,
    feature: usize,
}

impl WeightWalker {
    pub fn load(model: &str) -> Result<Self, VindexError> {
        let model_path = resolve_model_path(model)?;
        let weights = load_model_dir_validated(&model_path)?;

        let tokenizer_path = model_path.join(TOKENIZER_JSON);
        if !tokenizer_path.exists() {
            return Err(VindexError::MissingTensor(
                "tokenizer.json not found".into(),
            ));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| VindexError::Parse(e.to_string()))?;

        Ok(Self { weights, tokenizer })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    /// Walk a single layer's FFN weights into the graph.
    ///
    /// Confidence scoring:
    /// - `c_in`: raw gate projection score (input selectivity)
    /// - `c_out`: raw down projection score (output strength)
    /// - `c`: normalized `c_in × c_out`, scaled to `[0,1]` per layer
    pub fn walk_layer(
        &self,
        layer: usize,
        config: &WalkConfig,
        graph: &mut Graph,
        callbacks: &mut dyn WalkCallbacks,
    ) -> Result<LayerResult, VindexError> {
        let start = std::time::Instant::now();

        let prefix = format!("layers.{layer}.mlp.");
        let w_gate = self
            .weights
            .tensors
            .get(&format!("{prefix}gate_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}gate_proj.weight")))?;
        let w_down = self
            .weights
            .tensors
            .get(&format!("{prefix}down_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}down_proj.weight")))?;

        let n_features = w_down.shape()[1];
        callbacks.on_layer_start(layer, n_features);

        // BLAS-accelerated matmuls
        let all_output = self.weights.embed.dot(w_down);
        let all_input = self.weights.embed.dot(&w_gate.t());

        let k = config.top_k.min(self.weights.vocab_size);
        let progress_interval = (n_features / 20).max(1);

        // Phase 1: collect raw edges with c_in / c_out
        let mut raw_edges: Vec<RawEdge> = Vec::new();

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(layer, feat_idx, n_features);
            }

            let top_in = partial_top_k_column(&all_input, feat_idx, k);
            let top_out = partial_top_k_column(&all_output, feat_idx, k);

            let mut subjects: Vec<(String, f32)> = Vec::new();
            for (idx, score) in &top_in {
                if *score >= config.min_score {
                    if let Some(tok) = decode_token(&self.tokenizer, *idx as u32) {
                        if !tok.is_empty() {
                            subjects.push((tok, *score));
                        }
                    }
                }
            }

            let mut objects: Vec<(String, f32)> = Vec::new();
            for (idx, score) in &top_out {
                if *score >= config.min_score {
                    if let Some(tok) = decode_token(&self.tokenizer, *idx as u32) {
                        if !tok.is_empty() {
                            objects.push((tok, *score));
                        }
                    }
                }
            }

            if subjects.is_empty() || objects.is_empty() {
                continue;
            }

            let relation = format!("L{layer}-F{feat_idx}");
            for (subj, c_in) in &subjects {
                for (obj, c_out) in &objects {
                    raw_edges.push(RawEdge {
                        subject: subj.clone(),
                        relation: relation.clone(),
                        object: obj.clone(),
                        c_in: *c_in,
                        c_out: *c_out,
                        layer,
                        feature: feat_idx,
                    });
                }
            }
        }

        // Phase 2: per-layer normalization
        // confidence = (c_in × c_out) / max(c_in × c_out)
        // selectivity = c_in / max(c_in) — factual signal
        let max_product = raw_edges
            .iter()
            .map(|e| e.c_in * e.c_out)
            .fold(f32::MIN, f32::max)
            .max(f32::EPSILON);

        let max_cin = raw_edges
            .iter()
            .map(|e| e.c_in)
            .fold(f32::MIN, f32::max)
            .max(f32::EPSILON);

        let mut sum_conf = 0.0f64;
        let mut sum_sel = 0.0f64;
        let mut sum_cin = 0.0f64;
        let mut sum_cout = 0.0f64;
        let mut max_conf = 0.0f64;
        let mut min_conf = 1.0f64;
        let mut max_sel = 0.0f64;
        let mut self_loops = 0usize;

        let mut conf_thresholds = ThresholdCounts::default();
        let mut sel_thresholds = ThresholdCounts::default();

        let mut subj_counts: std::collections::HashMap<String, (usize, f64)> =
            std::collections::HashMap::new();
        let mut obj_counts: std::collections::HashMap<String, (usize, f64)> =
            std::collections::HashMap::new();

        // Phase 3: add normalized edges to graph
        for raw in &raw_edges {
            let product = raw.c_in * raw.c_out;
            let confidence = (product / max_product) as f64;
            let selectivity = (raw.c_in / max_cin) as f64;

            sum_conf += confidence;
            sum_sel += selectivity;
            sum_cin += raw.c_in as f64;
            sum_cout += raw.c_out as f64;
            if confidence > max_conf {
                max_conf = confidence;
            }
            if confidence < min_conf {
                min_conf = confidence;
            }
            if selectivity > max_sel {
                max_sel = selectivity;
            }
            if raw.subject == raw.object {
                self_loops += 1;
            }

            count_threshold(&mut conf_thresholds, confidence);
            count_threshold(&mut sel_thresholds, selectivity);

            // Track entity frequencies
            let se = subj_counts.entry(raw.subject.clone()).or_insert((0, 0.0));
            se.0 += 1;
            se.1 += confidence;

            let oe = obj_counts.entry(raw.object.clone()).or_insert((0, 0.0));
            oe.0 += 1;
            oe.1 += confidence;

            let edge = Edge::new(&raw.subject, &raw.relation, &raw.object)
                .with_confidence(confidence)
                .with_source(SourceType::Parametric)
                .with_metadata("layer", serde_json::Value::from(raw.layer as u64))
                .with_metadata("feature", serde_json::Value::from(raw.feature as u64))
                .with_metadata("c_in", serde_json::Value::from(raw.c_in as f64))
                .with_metadata("c_out", serde_json::Value::from(raw.c_out as f64))
                .with_metadata("selectivity", serde_json::Value::from(selectivity));
            graph.add_edge(edge);
        }

        let n = raw_edges.len();

        // Build top entities by count
        let top_subjects = top_entities(&subj_counts, 10);
        let top_objects = top_entities(&obj_counts, 10);

        let stats = if n > 0 {
            LayerStats {
                mean_confidence: sum_conf / n as f64,
                max_confidence: max_conf,
                min_confidence: min_conf,
                mean_c_in: sum_cin / n as f64,
                mean_c_out: sum_cout / n as f64,
                mean_selectivity: sum_sel / n as f64,
                max_selectivity: max_sel,
                self_loop_count: self_loops,
                self_loop_pct: if n > 0 {
                    (self_loops as f64 / n as f64) * 100.0
                } else {
                    0.0
                },
                top_subjects,
                top_objects,
                threshold_counts: conf_thresholds,
                selectivity_threshold_counts: sel_thresholds,
            }
        } else {
            LayerStats::default()
        };

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let result = LayerResult {
            layer,
            features_scanned: n_features,
            edges_found: n,
            elapsed_ms: elapsed,
            stats,
        };
        callbacks.on_layer_done(&result);
        callbacks.on_checkpoint(graph);
        Ok(result)
    }
}

/// Convenience: load model and walk all (or selected) layers.
pub fn walk_model(
    model: &str,
    layers: Option<&[usize]>,
    config: &WalkConfig,
    graph: &mut Graph,
    callbacks: &mut dyn WalkCallbacks,
) -> Result<Vec<LayerResult>, VindexError> {
    let walker = WeightWalker::load(model)?;

    let layer_indices: Vec<usize> = match layers {
        Some(ls) => ls.to_vec(),
        None => (0..walker.num_layers()).collect(),
    };

    let mut results = Vec::new();
    for &layer in &layer_indices {
        let result = walker.walk_layer(layer, config, graph, callbacks)?;
        results.push(result);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::super::test_fixture::create_mock_model;
    use super::*;

    fn fixture(slug: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("larql_ww_inline_{slug}"));
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);
        dir
    }

    fn cleanup(dir: &std::path::Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    // ── ThresholdCounts ───────────────────────────────────────────────────────

    #[test]
    fn threshold_counts_default_all_zero() {
        let t = ThresholdCounts::default();
        assert_eq!(t.t_01, 0);
        assert_eq!(t.t_05, 0);
        assert_eq!(t.t_10, 0);
        assert_eq!(t.t_25, 0);
        assert_eq!(t.t_50, 0);
        assert_eq!(t.t_75, 0);
        assert_eq!(t.t_90, 0);
    }

    // ── WalkConfig ────────────────────────────────────────────────────────────

    #[test]
    fn walk_config_default_values() {
        let c = WalkConfig::default();
        assert_eq!(c.top_k, 5);
        assert!((c.min_score - 0.02).abs() < 1e-6);
    }

    // ── LayerStats ────────────────────────────────────────────────────────────

    #[test]
    fn layer_stats_default_zero() {
        let s = LayerStats::default();
        assert_eq!(s.self_loop_count, 0);
        assert_eq!(s.self_loop_pct, 0.0);
        assert!(s.top_subjects.is_empty());
        assert!(s.top_objects.is_empty());
    }

    // ── SilentWalkCallbacks (no-op trait impl) ───────────────────────────────

    #[test]
    fn silent_walk_callbacks_no_op() {
        let mut cb = SilentWalkCallbacks;
        cb.on_layer_start(0, 4);
        cb.on_progress(0, 1, 4);
        let dummy_result = LayerResult {
            layer: 0,
            features_scanned: 4,
            edges_found: 1,
            elapsed_ms: 0.0,
            stats: LayerStats::default(),
        };
        cb.on_layer_done(&dummy_result);
        let g = Graph::new();
        cb.on_checkpoint(&g);
    }

    // ── WeightWalker::load ────────────────────────────────────────────────────

    #[test]
    fn load_returns_expected_num_layers() {
        let dir = fixture("ww_load_layers");
        let walker = WeightWalker::load(dir.to_str().unwrap()).unwrap();
        assert_eq!(walker.num_layers(), 2);
        cleanup(&dir);
    }

    #[test]
    fn load_missing_directory_errors() {
        assert!(WeightWalker::load("/nonexistent/larql/ww/path").is_err());
    }

    #[test]
    fn load_missing_tokenizer_errors() {
        let dir = fixture("ww_missing_tok");
        std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
        match WeightWalker::load(dir.to_str().unwrap()) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("tokenizer"), "msg: {msg}");
            }
            Err(other) => panic!("expected MissingTensor; got {other:?}"),
            Ok(_) => panic!("expected MissingTensor; got Ok"),
        }
        cleanup(&dir);
    }

    // ── walk_layer ────────────────────────────────────────────────────────────

    #[test]
    fn walk_layer_extracts_edges_and_stats() {
        let dir = fixture("ww_edges");
        let walker = WeightWalker::load(dir.to_str().unwrap()).unwrap();
        let cfg = WalkConfig {
            top_k: 3,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        let r = walker.walk_layer(0, &cfg, &mut g, &mut cb).unwrap();

        assert_eq!(r.layer, 0);
        assert_eq!(r.features_scanned, 4);
        assert!(r.edges_found > 0);
        assert!(r.elapsed_ms >= 0.0);
        for edge in g.edges() {
            assert_eq!(edge.source, SourceType::Parametric);
            assert!(edge.confidence >= 0.0 && edge.confidence <= 1.0);
            let m = edge.metadata.as_ref().unwrap();
            assert!(m.contains_key("layer"));
            assert!(m.contains_key("feature"));
            assert!(m.contains_key("c_in"));
            assert!(m.contains_key("c_out"));
            assert!(m.contains_key("selectivity"));
        }
        cleanup(&dir);
    }

    #[test]
    fn walk_layer_min_score_drops_low_confidence_edges() {
        let dir = fixture("ww_min_score");
        let walker = WeightWalker::load(dir.to_str().unwrap()).unwrap();
        let permissive = WalkConfig {
            top_k: 3,
            min_score: 0.0,
        };
        let strict = WalkConfig {
            top_k: 3,
            min_score: f32::INFINITY,
        };
        let mut g_p = Graph::new();
        let mut g_s = Graph::new();
        let mut cb = SilentWalkCallbacks;
        let r_p = walker
            .walk_layer(0, &permissive, &mut g_p, &mut cb)
            .unwrap();
        let r_s = walker.walk_layer(0, &strict, &mut g_s, &mut cb).unwrap();
        assert!(r_p.edges_found > 0);
        assert_eq!(r_s.edges_found, 0);
        cleanup(&dir);
    }

    #[test]
    fn walk_layer_missing_gate_proj_errors() {
        let dir = fixture("ww_no_gate");
        let mut walker = WeightWalker::load(dir.to_str().unwrap()).unwrap();
        walker
            .weights
            .tensors
            .remove("layers.0.mlp.gate_proj.weight");
        let cfg = WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        match walker.walk_layer(0, &cfg, &mut g, &mut cb) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("gate_proj.weight"), "msg: {msg}");
            }
            Err(other) => panic!("expected MissingTensor; got {other:?}"),
            Ok(_) => panic!("expected MissingTensor; got Ok"),
        }
        cleanup(&dir);
    }

    #[test]
    fn walk_layer_missing_down_proj_errors() {
        let dir = fixture("ww_no_down");
        let mut walker = WeightWalker::load(dir.to_str().unwrap()).unwrap();
        walker
            .weights
            .tensors
            .remove("layers.0.mlp.down_proj.weight");
        let cfg = WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        match walker.walk_layer(0, &cfg, &mut g, &mut cb) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("down_proj.weight"), "msg: {msg}");
            }
            Err(other) => panic!("expected MissingTensor; got {other:?}"),
            Ok(_) => panic!("expected MissingTensor; got Ok"),
        }
        cleanup(&dir);
    }

    // ── walk_model ────────────────────────────────────────────────────────────

    #[test]
    fn walk_model_default_layers_walks_all() {
        let dir = fixture("ww_walk_all");
        let cfg = WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        let results = walk_model(dir.to_str().unwrap(), None, &cfg, &mut g, &mut cb).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].layer, 0);
        assert_eq!(results[1].layer, 1);
        cleanup(&dir);
    }

    #[test]
    fn walk_model_layer_filter_walks_only_specified() {
        let dir = fixture("ww_walk_filter");
        let cfg = WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        let results = walk_model(dir.to_str().unwrap(), Some(&[1]), &cfg, &mut g, &mut cb).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].layer, 1);
        cleanup(&dir);
    }

    #[test]
    fn walk_model_propagates_load_error() {
        let cfg = WalkConfig::default();
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        let r = walk_model(
            "/nonexistent/larql/ww/walk_model",
            None,
            &cfg,
            &mut g,
            &mut cb,
        );
        assert!(r.is_err());
    }
}
