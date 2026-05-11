//! Walk attention OV circuits — extract routing edges from attention heads.
//!
//! Each attention head is a routing rule. The OV circuit tells you:
//! when this head fires on input X, it copies output Y.
//! That's an edge: X → Y via this head.
//!
//! Confidence scoring:
//! - c_in: input amplification norm (how much this head amplifies the input)
//! - c_out: output logit magnitude (how strongly it projects to the output token)
//! - c: normalized (c_in × c_out) / max per layer
//!
//! Zero forward passes. Pure matrix multiplication.

use larql_core::core::edge::Edge;
use larql_core::core::enums::SourceType;
use larql_core::core::graph::Graph;
use larql_models::{load_model_dir_validated, resolve_model_path, ModelWeights};

use super::utils::{count_threshold, decode_token, partial_top_k, top_entities};
use super::weight_walker::{LayerResult, LayerStats, WalkCallbacks, WalkConfig};
use crate::error::VindexError;
use crate::format::filenames::*;

/// Result of walking attention heads at a single layer.
#[derive(Debug, Clone)]
pub struct AttentionLayerResult {
    pub layer: usize,
    pub heads_walked: usize,
    pub edges_found: usize,
    pub elapsed_ms: f64,
    pub stats: LayerStats,
}

/// A raw attention edge before per-layer normalization.
struct RawEdge {
    subject: String,
    relation: String,
    object: String,
    c_in: f32,
    c_out: f32,
    layer: usize,
    head: usize,
}

/// A loaded model ready for attention walking.
pub struct AttentionWalker {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
}

impl AttentionWalker {
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

    /// Walk all attention heads at a single layer.
    pub fn walk_layer(
        &self,
        layer: usize,
        config: &WalkConfig,
        graph: &mut Graph,
        callbacks: &mut dyn WalkCallbacks,
    ) -> Result<AttentionLayerResult, VindexError> {
        let start = std::time::Instant::now();

        let prefix = format!("layers.{layer}.self_attn.");
        let w_v = self
            .weights
            .tensors
            .get(&format!("{prefix}v_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}v_proj.weight")))?;
        let w_o = self
            .weights
            .tensors
            .get(&format!("{prefix}o_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}o_proj.weight")))?;

        let hd = self.weights.arch.head_dim_for_layer(layer);
        let num_kv_heads = w_v.shape()[0] / hd;
        callbacks.on_layer_start(layer, num_kv_heads);

        let k_inputs = (config.top_k * 10).min(self.weights.vocab_size);

        // Phase 1: collect raw edges with c_in / c_out
        let mut raw_edges: Vec<RawEdge> = Vec::new();

        for h in 0..num_kv_heads {
            callbacks.on_progress(layer, h, num_kv_heads);
            let v_h = w_v.slice(ndarray::s![h * hd..(h + 1) * hd, ..]);
            let o_h = w_o.slice(ndarray::s![.., h * hd..(h + 1) * hd]);

            // OV circuit: O_h @ V_h -> (hidden, hidden)
            let ov = o_h.dot(&v_h);

            // transformed = embed @ OV.T -> (vocab, hidden)
            let transformed = self.weights.embed.dot(&ov.t());

            // Norms: how much this head amplifies each input (c_in candidate)
            let norms: Vec<f32> = (0..self.weights.vocab_size)
                .map(|i| {
                    let row = transformed.row(i);
                    row.iter().map(|x| x * x).sum::<f32>().sqrt()
                })
                .collect();

            // Top-k inputs by amplification
            let top_inputs = partial_top_k(&norms, k_inputs);

            let relation = format!("L{layer}-H{h}");
            for &(inp_idx, c_in) in &top_inputs {
                let inp_vec = transformed.row(inp_idx);

                // out_logits = embed @ inp_vec -> (vocab,)
                let out_logits: Vec<f32> = (0..self.weights.vocab_size)
                    .map(|j| {
                        self.weights
                            .embed
                            .row(j)
                            .iter()
                            .zip(inp_vec.iter())
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();

                let top_out = partial_top_k(&out_logits, config.top_k);

                let inp_token = match decode_token(&self.tokenizer, inp_idx as u32) {
                    Some(t) if !t.is_empty() => t,
                    _ => continue,
                };

                for &(out_idx, c_out) in &top_out {
                    if c_out < config.min_score {
                        continue;
                    }
                    let out_token = match decode_token(&self.tokenizer, out_idx as u32) {
                        Some(t) if !t.is_empty() => t,
                        _ => continue,
                    };

                    raw_edges.push(RawEdge {
                        subject: inp_token.clone(),
                        relation: relation.clone(),
                        object: out_token,
                        c_in,
                        c_out,
                        layer,
                        head: h,
                    });
                }
            }
        }

        // Phase 2: per-layer normalization
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

        let mut conf_thresholds = super::weight_walker::ThresholdCounts::default();
        let mut sel_thresholds = super::weight_walker::ThresholdCounts::default();

        let mut subj_counts: std::collections::HashMap<String, (usize, f64)> =
            std::collections::HashMap::new();
        let mut obj_counts: std::collections::HashMap<String, (usize, f64)> =
            std::collections::HashMap::new();

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
                .with_metadata("head", serde_json::Value::from(raw.head as u64))
                .with_metadata("circuit", serde_json::Value::String("OV".to_string()))
                .with_metadata("c_in", serde_json::Value::from(raw.c_in as f64))
                .with_metadata("c_out", serde_json::Value::from(raw.c_out as f64))
                .with_metadata("selectivity", serde_json::Value::from(selectivity));
            graph.add_edge(edge);
        }

        let n = raw_edges.len();
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

        callbacks.on_layer_done(&LayerResult {
            layer,
            features_scanned: num_kv_heads,
            edges_found: n,
            elapsed_ms: elapsed,
            stats: stats.clone(),
        });
        callbacks.on_checkpoint(graph);

        Ok(AttentionLayerResult {
            layer,
            heads_walked: num_kv_heads,
            edges_found: n,
            elapsed_ms: elapsed,
            stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixture::create_mock_model;
    use super::super::weight_walker::{SilentWalkCallbacks, WalkConfig};
    use super::*;

    fn fixture(slug: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("larql_attn_inline_{slug}"));
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);
        dir
    }

    fn cleanup(dir: &std::path::Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn load_returns_expected_num_layers() {
        let dir = fixture("load_layers");
        let walker = AttentionWalker::load(dir.to_str().unwrap()).unwrap();
        assert_eq!(walker.num_layers(), 2);
        cleanup(&dir);
    }

    #[test]
    fn load_missing_directory_errors() {
        let result = AttentionWalker::load("/nonexistent/larql/attn/path");
        assert!(result.is_err(), "expected load error for missing dir");
    }

    #[test]
    fn load_missing_tokenizer_errors() {
        let dir = fixture("missing_tok");
        std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
        match AttentionWalker::load(dir.to_str().unwrap()) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("tokenizer"), "msg: {msg}");
            }
            Err(other) => panic!("expected MissingTensor; got {other:?}"),
            Ok(_) => panic!("expected MissingTensor; got Ok"),
        }
        cleanup(&dir);
    }

    #[test]
    fn walk_layer_extracts_edges_and_metadata() {
        let dir = fixture("edges_meta");
        let walker = AttentionWalker::load(dir.to_str().unwrap()).unwrap();
        let cfg = WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        let r = walker.walk_layer(0, &cfg, &mut g, &mut cb).unwrap();

        assert_eq!(r.layer, 0);
        assert_eq!(r.heads_walked, 2);
        assert!(r.edges_found > 0);
        assert!(g.edge_count() >= r.edges_found);
        for edge in g.edges() {
            let m = edge.metadata.as_ref().unwrap();
            assert_eq!(m["circuit"], "OV");
            assert!(m.contains_key("layer"));
            assert!(m.contains_key("head"));
            assert!(m.contains_key("c_in"));
            assert!(m.contains_key("c_out"));
            assert!(m.contains_key("selectivity"));
            assert!(edge.confidence >= 0.0 && edge.confidence <= 1.0);
        }
        cleanup(&dir);
    }

    #[test]
    fn walk_layer_layerstats_invariants() {
        let dir = fixture("stats_inv");
        let walker = AttentionWalker::load(dir.to_str().unwrap()).unwrap();
        let cfg = WalkConfig {
            top_k: 3,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        let r = walker.walk_layer(0, &cfg, &mut g, &mut cb).unwrap();
        let s = &r.stats;

        // Confidence is normalised so its max is 1.0 when n > 0.
        assert!(s.max_confidence <= 1.0 + 1e-6);
        assert!(s.min_confidence >= 0.0);
        assert!(s.mean_confidence <= s.max_confidence + 1e-6);
        assert!(s.mean_confidence >= s.min_confidence - 1e-6);
        // Selectivity max is 1.0 too; mean ≤ max.
        assert!(s.max_selectivity <= 1.0 + 1e-6);
        assert!(s.mean_selectivity <= s.max_selectivity + 1e-6);
        assert!(s.mean_c_in >= 0.0);
        assert!(s.mean_c_out >= 0.0);
        // Self-loop pct is bounded.
        assert!(s.self_loop_pct >= 0.0 && s.self_loop_pct <= 100.0 + 1e-6);
        cleanup(&dir);
    }

    #[test]
    fn walk_layer_min_score_filters_edges() {
        let dir = fixture("min_score");
        let walker = AttentionWalker::load(dir.to_str().unwrap()).unwrap();

        let permissive = WalkConfig {
            top_k: 3,
            min_score: 0.0,
        };
        let strict = WalkConfig {
            top_k: 3,
            min_score: f32::INFINITY, // filter everything
        };

        let mut g_p = Graph::new();
        let mut cb_p = SilentWalkCallbacks;
        let r_p = walker
            .walk_layer(0, &permissive, &mut g_p, &mut cb_p)
            .unwrap();

        let mut g_s = Graph::new();
        let mut cb_s = SilentWalkCallbacks;
        let r_s = walker.walk_layer(0, &strict, &mut g_s, &mut cb_s).unwrap();

        assert!(r_p.edges_found > 0);
        assert_eq!(
            r_s.edges_found, 0,
            "INFINITY min_score should drop every edge"
        );
        cleanup(&dir);
    }

    #[test]
    fn walk_layer_for_every_layer() {
        let dir = fixture("every_layer");
        let walker = AttentionWalker::load(dir.to_str().unwrap()).unwrap();
        let cfg = WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        for layer in 0..walker.num_layers() {
            let r = walker.walk_layer(layer, &cfg, &mut g, &mut cb).unwrap();
            assert_eq!(r.layer, layer);
            assert_eq!(r.heads_walked, 2);
        }
        cleanup(&dir);
    }

    #[test]
    fn walk_layer_missing_v_proj_errors() {
        let dir = fixture("missing_v");
        let walker = AttentionWalker::load(dir.to_str().unwrap()).unwrap();
        // Manually nuke the in-memory tensor before walking. This
        // covers the post-load MissingTensor error path.
        let mut walker = walker;
        walker
            .weights
            .tensors
            .remove("layers.0.self_attn.v_proj.weight");
        let cfg = WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        match walker.walk_layer(0, &cfg, &mut g, &mut cb) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("v_proj.weight"), "msg: {msg}");
            }
            other => panic!("expected MissingTensor for v_proj; got {other:?}"),
        }
        cleanup(&dir);
    }

    #[test]
    fn walk_layer_missing_o_proj_errors() {
        let dir = fixture("missing_o");
        let walker = AttentionWalker::load(dir.to_str().unwrap()).unwrap();
        let mut walker = walker;
        walker
            .weights
            .tensors
            .remove("layers.0.self_attn.o_proj.weight");
        let cfg = WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        match walker.walk_layer(0, &cfg, &mut g, &mut cb) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("o_proj.weight"), "msg: {msg}");
            }
            other => panic!("expected MissingTensor for o_proj; got {other:?}"),
        }
        cleanup(&dir);
    }

    #[test]
    fn attention_layer_result_fields_populated() {
        let dir = fixture("result_fields");
        let walker = AttentionWalker::load(dir.to_str().unwrap()).unwrap();
        let cfg = WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut g = Graph::new();
        let mut cb = SilentWalkCallbacks;
        let r = walker.walk_layer(1, &cfg, &mut g, &mut cb).unwrap();
        assert_eq!(r.layer, 1);
        assert_eq!(r.heads_walked, 2);
        assert!(r.edges_found > 0);
        assert!(r.elapsed_ms >= 0.0);
        cleanup(&dir);
    }
}
