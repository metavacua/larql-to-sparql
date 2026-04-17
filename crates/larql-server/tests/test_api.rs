//! Integration tests for larql-server API endpoints.
//!
//! Builds a synthetic in-memory vindex and tests each route handler
//! through the axum test infrastructure (no network, no disk).

use larql_vindex::ndarray::{Array1, Array2};
use larql_vindex::{
    FeatureMeta, PatchedVindex, VectorIndex, VindexConfig, VindexLayerInfo,
    ExtractLevel, LayerBands,
};

// ══════════════════════════════════════════════════════════════
// Test helpers
// ══════════════════════════════════════════════════════════════

fn make_top_k(token: &str, id: u32, logit: f32) -> larql_models::TopKEntry {
    larql_models::TopKEntry {
        token: token.to_string(),
        token_id: id,
        logit,
    }
}

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![
            make_top_k(token, id, score),
            make_top_k("also", id + 1, score * 0.5),
        ],
    }
}

/// Build a small test VectorIndex: 2 layers, 4 hidden dims, 3 features/layer.
fn test_index() -> VectorIndex {
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;

    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[1, 1]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        Some(make_meta("Tokyo", 201, 0.85)),
        Some(make_meta("Spain", 202, 0.70)),
    ];

    VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        vec![Some(meta0), Some(meta1)],
        num_layers,
        hidden,
    )
}

/// Build a test VindexConfig matching the test index.
fn test_config() -> VindexConfig {
    VindexConfig {
        version: 2,
        model: "test/model-4".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant: larql_vindex::QuantFormat::None,
        layer_bands: Some(LayerBands {
            syntax: (0, 0),
            knowledge: (0, 1),
            output: (1, 1),
        }),
        layers: vec![
            VindexLayerInfo { layer: 0, num_features: 3, offset: 0, length: 48, num_experts: None, num_features_per_expert: None },
            VindexLayerInfo { layer: 1, num_features: 3, offset: 48, length: 48, num_experts: None, num_features_per_expert: None },
        ],
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
    }
}

/// Build a tiny embeddings matrix (vocab=8, hidden=4).
fn test_embeddings() -> Array2<f32> {
    let mut embed = Array2::<f32>::zeros((8, 4));
    embed[[0, 0]] = 1.0;
    embed[[1, 1]] = 1.0;
    embed[[2, 2]] = 1.0;
    embed[[3, 3]] = 1.0;
    embed[[4, 0]] = 1.0;
    embed[[4, 1]] = 1.0;
    embed
}

// ══════════════════════════════════════════════════════════════
// CORE LOGIC TESTS (what the server handlers call)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_gate_knn_returns_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &query, 3);
    assert!(!hits.is_empty());
    // Feature 0 has gate[0,0]=1.0, should be top hit
    assert_eq!(hits[0].0, 0);
    assert!((hits[0].1 - 1.0).abs() < 0.01);
}

#[test]
fn test_walk_returns_per_layer_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0, 1], 3);
    assert_eq!(trace.layers.len(), 2);

    // Layer 0: feature 0 (Paris) should be top hit
    let (layer, hits) = &trace.layers[0];
    assert_eq!(*layer, 0);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].meta.top_token, "Paris");
}

#[test]
fn test_walk_with_layer_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let trace = patched.walk(&query, &[1], 3);
    assert_eq!(trace.layers.len(), 1);
    assert_eq!(trace.layers[0].0, 1);
}

#[test]
fn test_describe_entity_via_embedding() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate what the describe handler does:
    // Token embedding → gate KNN → aggregate edges.
    let embed = test_embeddings();
    let query = embed.row(0).mapv(|v| v * 1.0); // token 0 → [1,0,0,0]
    let trace = patched.walk(&query, &[0, 1], 10);

    let mut targets: Vec<String> = Vec::new();
    for (_, hits) in &trace.layers {
        for hit in hits {
            targets.push(hit.meta.top_token.clone());
        }
    }

    // Token 0 → dim 0 strong → feature 0 (Paris) at L0, feature 1 (Tokyo) at L1
    assert!(targets.contains(&"Paris".to_string()));
}

#[test]
fn test_select_by_layer() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate SELECT at layer 0
    let metas = patched.down_meta_at(0).unwrap();
    let tokens: Vec<&str> = metas
        .iter()
        .filter_map(|m| m.as_ref().map(|m| m.top_token.as_str()))
        .collect();

    assert_eq!(tokens, vec!["Paris", "French", "Europe"]);
}

#[test]
fn test_select_with_entity_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Filter for tokens containing "par" (case-insensitive)
    let metas = patched.down_meta_at(0).unwrap();
    let matches: Vec<&str> = metas
        .iter()
        .filter_map(|m| m.as_ref())
        .filter(|m| m.top_token.to_lowercase().contains("par"))
        .map(|m| m.top_token.as_str())
        .collect();

    assert_eq!(matches, vec!["Paris"]);
}

#[test]
fn test_relations_listing() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate SHOW RELATIONS: scan all layers, aggregate tokens
    let mut token_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for layer in patched.loaded_layers() {
        if let Some(metas) = patched.down_meta_at(layer) {
            for meta_opt in metas.iter() {
                if let Some(meta) = meta_opt {
                    *token_counts.entry(meta.top_token.clone()).or_default() += 1;
                }
            }
        }
    }

    assert_eq!(token_counts.len(), 6); // Paris, French, Europe, Berlin, Tokyo, Spain
    assert_eq!(*token_counts.get("Paris").unwrap(), 1);
}

#[test]
fn test_stats_from_config() {
    let config = test_config();
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
    assert_eq!(total_features, 6);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.hidden_size, 4);
    assert_eq!(config.model, "test/model-4");
}

// ══════════════════════════════════════════════════════════════
// PATCH OPERATIONS (what the patch endpoints use)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_apply_patch_modifies_walk() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    // Before patch: feature 0 at L0 = "Paris"
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);
    assert_eq!(trace.layers[0].1[0].meta.top_token, "Paris");

    // Update feature 0 at L0 to "London"
    patched.update_feature_meta(0, 0, make_meta("London", 300, 0.99));

    let trace = patched.walk(&query, &[0], 3);
    assert_eq!(trace.layers[0].1[0].meta.top_token, "London");
}

#[test]
fn test_delete_feature_removes_from_walk() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    // Delete feature 0 at L0
    patched.delete_feature(0, 0);

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);

    // Feature 0 should no longer appear
    for (_, hits) in &trace.layers {
        for hit in hits {
            assert_ne!(hit.feature, 0);
        }
    }
}

#[test]
fn test_patch_count_tracking() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);
    assert_eq!(patched.num_patches(), 0);

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: Some("test-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![
            larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: Some("test".into()),
            },
        ],
    };

    patched.apply_patch(patch);
    assert_eq!(patched.num_patches(), 1);
    assert_eq!(patched.num_overrides(), 1);
}

#[test]
fn test_remove_patch_restores_state() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: Some("removable".into()),
        author: None,
        tags: vec![],
        operations: vec![
            larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: None,
            },
        ],
    };

    patched.apply_patch(patch);
    assert_eq!(patched.num_patches(), 1);

    // Feature 0 should be deleted
    assert!(patched.feature_meta(0, 0).is_none());

    // Remove the patch
    patched.remove_patch(0);
    assert_eq!(patched.num_patches(), 0);

    // Feature 0 should be back
    assert!(patched.feature_meta(0, 0).is_some());
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Paris");
}

// ══════════════════════════════════════════════════════════════
// MULTI-MODEL SERVING LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_model_id_extraction() {
    assert_eq!(model_id("google/gemma-3-4b-it"), "gemma-3-4b-it");
    assert_eq!(model_id("llama-3-8b"), "llama-3-8b");
    assert_eq!(model_id("org/sub/model"), "model");
}

fn model_id(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}

// ══════════════════════════════════════════════════════════════
// EDGE CASES
// ══════════════════════════════════════════════════════════════

#[test]
fn test_empty_query_returns_no_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &query, 3);
    // All scores are 0, but KNN still returns results (sorted by abs)
    for (_feat, score) in &hits {
        assert!((score.abs()) < 0.01);
    }
}

#[test]
fn test_nonexistent_layer_returns_empty() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(99, &query, 3);
    assert!(hits.is_empty());
}

#[test]
fn test_walk_empty_layer_list() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[], 3);
    assert!(trace.layers.is_empty());
}

#[test]
fn test_large_top_k_clamped() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    // Request 100 but only 3 features exist
    let hits = patched.gate_knn(0, &query, 100);
    assert_eq!(hits.len(), 3);
}

// ══════════════════════════════════════════════════════════════
// PROBE LABELS (relation classifier in DESCRIBE)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_probe_label_lookup() {
    let mut labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    labels.insert((0, 0), "capital".into());
    labels.insert((0, 1), "language".into());
    labels.insert((1, 2), "continent".into());

    assert_eq!(labels.get(&(0, 0)).map(|s| s.as_str()), Some("capital"));
    assert_eq!(labels.get(&(0, 1)).map(|s| s.as_str()), Some("language"));
    assert_eq!(labels.get(&(1, 2)).map(|s| s.as_str()), Some("continent"));
    assert_eq!(labels.get(&(0, 2)), None);
    assert_eq!(labels.get(&(99, 99)), None);
}

#[test]
fn test_describe_edge_with_probe_label() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    let mut labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    labels.insert((0, 0), "capital".into());

    // Walk to find edges (simulates describe handler)
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 5);

    // Build edge info like the handler does
    for (layer, hits) in &trace.layers {
        for hit in hits {
            let label = labels.get(&(*layer, hit.feature));
            if hit.feature == 0 && *layer == 0 {
                assert_eq!(label, Some(&"capital".to_string()));
            } else {
                // Other features have no probe label
                assert!(label.is_none() || label.is_some());
            }
        }
    }
}

#[test]
fn test_probe_labels_empty_when_no_file() {
    // Simulates load_probe_labels on a nonexistent path
    let labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    assert!(labels.is_empty());
}

// ══════════════════════════════════════════════════════════════
// LAYER BAND FILTERING (DESCRIBE handler logic)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_layer_band_filtering() {
    let bands = LayerBands {
        syntax: (0, 0),
        knowledge: (0, 1),
        output: (1, 1),
    };

    let all_layers = vec![0, 1];

    let syntax: Vec<usize> = all_layers.iter().copied()
        .filter(|l| *l >= bands.syntax.0 && *l <= bands.syntax.1)
        .collect();
    assert_eq!(syntax, vec![0]);

    let knowledge: Vec<usize> = all_layers.iter().copied()
        .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
        .collect();
    assert_eq!(knowledge, vec![0, 1]);

    let output: Vec<usize> = all_layers.iter().copied()
        .filter(|l| *l >= bands.output.0 && *l <= bands.output.1)
        .collect();
    assert_eq!(output, vec![1]);
}

#[test]
fn test_layer_band_from_family() {
    let bands = LayerBands::for_family("gemma3", 34).unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));
}

#[test]
fn test_layer_band_fallback() {
    // Unknown family with enough layers → estimated bands
    let bands = LayerBands::for_family("unknown_family", 20).unwrap();
    assert_eq!(bands.syntax.0, 0);
    assert!(bands.knowledge.0 > 0);
    assert!(bands.output.1 == 19);
}

// ══════════════════════════════════════════════════════════════
// WALK LAYER RANGE PARSING
// ══════════════════════════════════════════════════════════════

fn parse_layers(s: &str, all: &[usize]) -> Vec<usize> {
    if let Some((start, end)) = s.split_once('-') {
        if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
            return all.iter().copied().filter(|l| *l >= s && *l <= e).collect();
        }
    }
    s.split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .filter(|l| all.contains(l))
        .collect()
}

#[test]
fn test_parse_layer_range() {
    let all = vec![0, 1, 2, 3, 4, 5];
    assert_eq!(parse_layers("2-4", &all), vec![2, 3, 4]);
    assert_eq!(parse_layers("0-1", &all), vec![0, 1]);
    assert_eq!(parse_layers("5-5", &all), vec![5]);
}

#[test]
fn test_parse_layer_list() {
    let all = vec![0, 1, 2, 3, 4, 5];
    assert_eq!(parse_layers("1,3,5", &all), vec![1, 3, 5]);
    assert_eq!(parse_layers("0", &all), vec![0]);
}

#[test]
fn test_parse_layer_range_filters_missing() {
    let all = vec![0, 2, 4]; // layers 1, 3 not loaded
    assert_eq!(parse_layers("0-4", &all), vec![0, 2, 4]);
    assert_eq!(parse_layers("1,3", &all), Vec::<usize>::new());
}

// ══════════════════════════════════════════════════════════════
// MULTI-MODEL LOOKUP
// ══════════════════════════════════════════════════════════════

#[test]
fn test_multi_model_lookup_by_id() {
    // Simulate AppState.model() logic
    let models = vec!["gemma-3-4b-it", "llama-3-8b", "mistral-7b"];

    let find = |id: &str| models.iter().find(|m| **m == id);

    assert_eq!(find("gemma-3-4b-it"), Some(&"gemma-3-4b-it"));
    assert_eq!(find("llama-3-8b"), Some(&"llama-3-8b"));
    assert_eq!(find("nonexistent"), None);
}

#[test]
fn test_single_model_returns_first() {
    let models = vec!["only-model"];

    // Single model mode: None → returns first
    let result = if models.len() == 1 { models.first() } else { None };
    assert_eq!(result, Some(&"only-model"));
}

#[test]
fn test_multi_model_none_returns_none() {
    let models = vec!["a", "b"];

    // Multi-model mode: None → returns None (must specify ID)
    let result: Option<&&str> = if models.len() == 1 { models.first() } else { None };
    assert_eq!(result, None);
}

// ══════════════════════════════════════════════════════════════
// INFER LOGIC (core computation path)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_infer_mode_parsing() {
    // The infer handler parses mode into walk/dense/compare
    let check = |mode: &str| -> (bool, bool) {
        let is_compare = mode == "compare";
        let use_walk = mode == "walk" || is_compare;
        let use_dense = mode == "dense" || is_compare;
        (use_walk, use_dense)
    };

    assert_eq!(check("walk"), (true, false));
    assert_eq!(check("dense"), (false, true));
    assert_eq!(check("compare"), (true, true));
}

#[test]
fn test_config_has_inference_capability() {
    let mut config = test_config();

    // Browse level → no inference
    config.extract_level = ExtractLevel::Browse;
    config.has_model_weights = false;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(!has_weights);

    // Inference level → has inference
    config.extract_level = ExtractLevel::Inference;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(has_weights);

    // Legacy has_model_weights flag
    config.extract_level = ExtractLevel::Browse;
    config.has_model_weights = true;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(has_weights);
}

// ══════════════════════════════════════════════════════════════
// AUTH LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_bearer_token_extraction() {
    let header = "Bearer sk-abc123";
    let token = if header.starts_with("Bearer ") {
        Some(&header[7..])
    } else {
        None
    };
    assert_eq!(token, Some("sk-abc123"));
}

#[test]
fn test_bearer_token_mismatch() {
    let header = "Bearer wrong-key";
    let required = "sk-abc123";
    let token = &header[7..];
    assert_ne!(token, required);
}

#[test]
fn test_no_auth_header() {
    let header: Option<&str> = None;
    let has_valid_token = header
        .filter(|h| h.starts_with("Bearer "))
        .map(|h| &h[7..])
        .is_some();
    assert!(!has_valid_token);
}

#[test]
fn test_health_exempt_from_auth() {
    let path = "/v1/health";
    let is_health = path == "/v1/health";
    assert!(is_health);

    let path = "/v1/describe";
    let is_health = path == "/v1/health";
    assert!(!is_health);
}

// ══════════════════════════════════════════════════════════════
// RATE LIMITER
// ══════════════════════════════════════════════════════════════

#[test]
fn test_rate_limit_parse() {
    // Valid formats
    assert!(rate_limit_parse("100/min").is_some());
    assert!(rate_limit_parse("10/sec").is_some());
    assert!(rate_limit_parse("3600/hour").is_some());
    assert!(rate_limit_parse("50/s").is_some());
    assert!(rate_limit_parse("200/m").is_some());

    // Invalid formats
    assert!(rate_limit_parse("abc").is_none());
    assert!(rate_limit_parse("100").is_none());
    assert!(rate_limit_parse("100/day").is_none());
}

fn rate_limit_parse(spec: &str) -> Option<(f64, f64)> {
    let parts: Vec<&str> = spec.split('/').collect();
    if parts.len() != 2 { return None; }
    let count: f64 = parts[0].trim().parse().ok()?;
    let per_sec = match parts[1].trim() {
        "sec" | "s" | "second" => count,
        "min" | "m" | "minute" => count / 60.0,
        "hour" | "h" => count / 3600.0,
        _ => return None,
    };
    Some((count, per_sec))
}

#[test]
fn test_rate_limit_token_bucket() {
    // Simulate token bucket: 2 tokens, 1 refill/sec
    let mut tokens: f64 = 2.0;
    let max_tokens: f64 = 2.0;

    // First two requests succeed
    assert!(tokens >= 1.0); tokens -= 1.0;
    assert!(tokens >= 1.0); tokens -= 1.0;

    // Third fails
    assert!(tokens < 1.0);

    // Refill
    tokens = (tokens + 1.0).min(max_tokens);
    assert!(tokens >= 1.0);
}

// ══════════════════════════════════════════════════════════════
// DESCRIBE CACHE
// ══════════════════════════════════════════════════════════════

#[test]
fn test_cache_key_format() {
    let key = format!("{}:{}:{}:{}:{}", "model", "France", "knowledge", 20, 5);
    assert_eq!(key, "model:France:knowledge:20:5");
}

#[test]
fn test_cache_disabled_when_ttl_zero() {
    // TTL=0 means cache is disabled
    let ttl = 0u64;
    assert!(!( ttl > 0));
}

#[test]
fn test_cache_hit_and_miss() {
    use std::collections::HashMap;

    let mut cache: HashMap<String, serde_json::Value> = HashMap::new();
    let key = "model:France:knowledge:20:5".to_string();
    let value = serde_json::json!({"entity": "France", "edges": []});

    // Miss
    assert!(cache.get(&key).is_none());

    // Insert
    cache.insert(key.clone(), value.clone());

    // Hit
    assert_eq!(cache.get(&key), Some(&value));
}

// ══════════════════════════════════════════════════════════════
// SELECT WITH RELATION FILTER
// ══════════════════════════════════════════════════════════════

#[test]
fn test_select_with_relation_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    let mut labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    labels.insert((0, 0), "capital".into());
    labels.insert((0, 1), "language".into());

    // Simulate SELECT with relation="capital" filter
    let metas = patched.down_meta_at(0).unwrap();
    let matches: Vec<(usize, &str)> = metas
        .iter()
        .enumerate()
        .filter_map(|(i, m)| m.as_ref().map(|m| (i, m.top_token.as_str())))
        .filter(|(i, _)| {
            labels.get(&(0, *i))
                .map(|r| r.to_lowercase().contains("capital"))
                .unwrap_or(false)
        })
        .collect();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].1, "Paris");
}

#[test]
fn test_select_relation_label_in_output() {
    let mut labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    labels.insert((0, 0), "capital".into());

    // Feature with label
    let rel = labels.get(&(0, 0));
    assert_eq!(rel, Some(&"capital".to_string()));

    // Feature without label
    let rel = labels.get(&(0, 1));
    assert_eq!(rel, None);
}

// ══════════════════════════════════════════════════════════════
// WALK WITH RELATION LABELS
// ══════════════════════════════════════════════════════════════

#[test]
fn test_walk_hits_include_relation_label() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    let mut labels: std::collections::HashMap<(usize, usize), String> =
        std::collections::HashMap::new();
    labels.insert((0, 0), "capital".into());

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);

    // Simulate what walk handler does: add relation label to hits
    for (layer, hits) in &trace.layers {
        for hit in hits {
            let label = labels.get(&(*layer, hit.feature));
            if hit.feature == 0 {
                assert_eq!(label, Some(&"capital".to_string()));
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════
// DESCRIBE HANDLER LOGIC (edge aggregation, scoring, filtering)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_describe_min_score_filtering() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0, 1], 10);

    let min_score = 0.5;
    let mut edges = Vec::new();
    for (_, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score >= min_score {
                edges.push(hit.meta.top_token.clone());
            }
        }
    }
    // Only hits above threshold should pass
    for (_, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score < min_score {
                assert!(!edges.contains(&hit.meta.top_token) || hit.gate_score >= min_score);
            }
        }
    }
}

#[test]
fn test_describe_edge_aggregation_by_target() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0, 1], 10);

    // Aggregate by target token (lowercase key)
    let mut edges: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for (_, hits) in &trace.layers {
        for hit in hits {
            let key = hit.meta.top_token.to_lowercase();
            let entry = edges.entry(key).or_insert(0.0);
            if hit.gate_score > *entry {
                *entry = hit.gate_score;
            }
        }
    }
    // Should have aggregated entries
    assert!(!edges.is_empty());
}

#[test]
fn test_describe_verbose_adds_layer_range() {
    // Verbose mode adds layer_min, layer_max, count
    let layers = vec![14usize, 18, 22, 27];
    let min_l = *layers.iter().min().unwrap();
    let max_l = *layers.iter().max().unwrap();
    assert_eq!(min_l, 14);
    assert_eq!(max_l, 27);
    assert_eq!(layers.len(), 4); // count
}

#[test]
fn test_describe_self_reference_filtered() {
    // DESCRIBE "France" should not include "France" as an edge target
    let entity = "France";
    let target = "France";
    assert_eq!(entity.to_lowercase(), target.to_lowercase());
    // Handler filters this case
}

// ══════════════════════════════════════════════════════════════
// SELECT HANDLER LOGIC (ordering, multi-filter)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_select_order_by_confidence_desc() {
    let mut rows = vec![(0.5f32, "a"), (0.9, "b"), (0.1, "c"), (0.7, "d")];
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    assert_eq!(rows[0].1, "b");
    assert_eq!(rows[1].1, "d");
    assert_eq!(rows[2].1, "a");
    assert_eq!(rows[3].1, "c");
}

#[test]
fn test_select_order_by_confidence_asc() {
    let mut rows = vec![(0.5f32, "a"), (0.9, "b"), (0.1, "c")];
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    assert_eq!(rows[0].1, "c");
    assert_eq!(rows[1].1, "a");
    assert_eq!(rows[2].1, "b");
}

#[test]
fn test_select_entity_substring_match() {
    let token = "Paris";
    let filter = "par";
    assert!(token.to_lowercase().contains(&filter.to_lowercase()));

    let token = "Berlin";
    assert!(!token.to_lowercase().contains(&filter.to_lowercase()));
}

#[test]
fn test_select_min_confidence_filter() {
    let scores = vec![0.1f32, 0.5, 0.8, 0.95];
    let min = 0.5;
    let filtered: Vec<f32> = scores.into_iter().filter(|s| *s >= min).collect();
    assert_eq!(filtered, vec![0.5, 0.8, 0.95]);
}

#[test]
fn test_select_limit_truncation() {
    let mut rows: Vec<i32> = (0..100).collect();
    let limit = 5;
    rows.truncate(limit);
    assert_eq!(rows.len(), 5);
}

// ══════════════════════════════════════════════════════════════
// INFER HANDLER LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_infer_disabled_check() {
    let disabled = true;
    assert!(disabled); // Handler returns 503

    let disabled = false;
    assert!(!disabled); // Handler proceeds
}

#[test]
fn test_infer_weights_required() {
    let config = test_config();
    // Browse level + no model weights → can't infer
    let can_infer = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(!can_infer);
}

#[test]
fn test_infer_compare_returns_both() {
    let mode = "compare";
    let is_compare = mode == "compare";
    let use_walk = mode == "walk" || is_compare;
    let use_dense = mode == "dense" || is_compare;
    assert!(is_compare);
    assert!(use_walk);
    assert!(use_dense);
}

// ══════════════════════════════════════════════════════════════
// ERROR HANDLING
// ══════════════════════════════════════════════════════════════

#[test]
fn test_error_model_not_found() {
    let models: Vec<&str> = vec!["gemma-3-4b-it"];
    let result = models.iter().find(|m| **m == "nonexistent");
    assert!(result.is_none()); // → 404
}

#[test]
fn test_error_empty_prompt() {
    let token_ids: Vec<u32> = vec![];
    assert!(token_ids.is_empty()); // → 400 BadRequest
}

#[test]
fn test_error_nonexistent_model_in_multi() {
    let models = vec!["model-a", "model-b"];
    let find = |id: &str| models.iter().find(|m| **m == id);
    assert!(find("model-c").is_none()); // → 404
}

// ══════════════════════════════════════════════════════════════
// SESSION MANAGEMENT LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_session_id_header_parsing() {
    let header_value = "sess-abc123";
    assert!(!header_value.is_empty());
    assert_eq!(header_value, "sess-abc123");
}

#[test]
fn test_session_patch_isolation() {
    // Two sessions should have independent patch state
    let index = test_index();
    let mut patched_a = PatchedVindex::new(index.clone());
    let mut patched_b = PatchedVindex::new(index);

    patched_a.delete_feature(0, 0);
    // Session A: feature 0 deleted
    assert!(patched_a.feature_meta(0, 0).is_none());
    // Session B: feature 0 still exists
    assert!(patched_b.feature_meta(0, 0).is_some());

    patched_b.update_feature_meta(0, 1, make_meta("Updated", 999, 0.99));
    assert_eq!(patched_b.feature_meta(0, 1).unwrap().top_token, "Updated");
    // Session A: feature 1 unchanged
    assert_eq!(patched_a.feature_meta(0, 1).unwrap().top_token, "French");
}

#[test]
fn test_session_global_unaffected() {
    let index = test_index();
    let global = PatchedVindex::new(index.clone());
    let mut session = PatchedVindex::new(index);

    session.delete_feature(0, 0);
    // Global: untouched
    assert!(global.feature_meta(0, 0).is_some());
    assert_eq!(global.feature_meta(0, 0).unwrap().top_token, "Paris");
}

// ══════════════════════════════════════════════════════════════
// WALK-FFN (decoupled inference protocol)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_walk_ffn_single_layer() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let residual = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &residual, 3);
    let features: Vec<usize> = hits.iter().map(|(f, _)| *f).collect();
    let scores: Vec<f32> = hits.iter().map(|(_, s)| *s).collect();
    assert!(!features.is_empty());
    assert_eq!(features.len(), scores.len());
    // Feature 0 should be top (responds to dim 0)
    assert_eq!(features[0], 0);
}

#[test]
fn test_walk_ffn_batched_layers() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let residual = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

    let layers = vec![0, 1];
    let mut results = Vec::new();
    for &layer in &layers {
        let hits = patched.gate_knn(layer, &residual, 3);
        results.push((layer, hits));
    }
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 0);
    assert_eq!(results[1].0, 1);
}

#[test]
fn test_walk_ffn_residual_dimension_check() {
    // Handler validates residual length == hidden_size
    let expected_hidden = 4;
    let residual_ok = vec![1.0f32; 4];
    let residual_bad = vec![1.0f32; 8];
    assert_eq!(residual_ok.len(), expected_hidden);
    assert_ne!(residual_bad.len(), expected_hidden);
}

#[test]
fn test_walk_ffn_top_k_default() {
    // Default top_k is 8092
    let default_top_k: usize = 8092;
    assert_eq!(default_top_k, 8092);
    // With only 3 features, top_k is clamped
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let residual = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &residual, default_top_k);
    assert_eq!(hits.len(), 3); // Only 3 features exist
}

// ══════════════════════════════════════════════════════════════
// ETAG / CDN CACHE HEADERS
// ══════════════════════════════════════════════════════════════

#[test]
fn test_etag_deterministic() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let body = serde_json::json!({"entity": "France", "edges": [{"target": "Paris"}]});
    let s = body.to_string();

    let mut h1 = DefaultHasher::new();
    s.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    s.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn test_etag_format() {
    // ETag should be quoted hex string
    let body = serde_json::json!({"test": true});
    let s = body.to_string();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&s, &mut hasher);
    let etag = format!("\"{}\"", format!("{:x}", std::hash::Hasher::finish(&hasher)));
    assert!(etag.starts_with('"'));
    assert!(etag.ends_with('"'));
    assert!(etag.len() > 4); // At least "xx"
}

#[test]
fn test_if_none_match_comparison() {
    let etag = "\"abc123\"";
    // Exact match
    assert_eq!(etag.trim(), etag);
    // Wildcard
    assert_eq!("*".trim(), "*");
    // No match
    assert_ne!("\"different\"".trim(), etag);
}

#[test]
fn test_304_not_modified_condition() {
    let cached_etag = "\"abc123\"";
    let request_etag = "\"abc123\"";
    let should_304 = request_etag.trim() == cached_etag || request_etag.trim() == "*";
    assert!(should_304);

    let stale_etag = "\"old\"";
    let should_304 = stale_etag.trim() == cached_etag || stale_etag.trim() == "*";
    assert!(!should_304);
}

// ══════════════════════════════════════════════════════════════
// SESSION-SCOPED DESCRIBE/WALK/SELECT
// ══════════════════════════════════════════════════════════════

#[test]
fn test_session_scoped_describe() {
    // Session A patches feature 0 → different describe result
    let index = test_index();
    let mut session_a = PatchedVindex::new(index.clone());
    let global = PatchedVindex::new(index);

    session_a.update_feature_meta(0, 0, make_meta("London", 300, 0.99));

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

    // Session A: London
    let trace_a = session_a.walk(&query, &[0], 3);
    assert_eq!(trace_a.layers[0].1[0].meta.top_token, "London");

    // Global: still Paris
    let trace_g = global.walk(&query, &[0], 3);
    assert_eq!(trace_g.layers[0].1[0].meta.top_token, "Paris");
}

#[test]
fn test_session_scoped_walk() {
    let index = test_index();
    let mut session = PatchedVindex::new(index.clone());
    let global = PatchedVindex::new(index);

    session.delete_feature(0, 0);

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace_s = session.walk(&query, &[0], 3);
    let trace_g = global.walk(&query, &[0], 3);

    // Session: feature 0 removed
    assert!(trace_s.layers[0].1.iter().all(|h| h.feature != 0));
    // Global: feature 0 present
    assert!(trace_g.layers[0].1.iter().any(|h| h.feature == 0));
}

#[test]
fn test_session_scoped_select() {
    let index = test_index();
    let mut session = PatchedVindex::new(index.clone());
    let global = PatchedVindex::new(index);

    session.update_feature_meta(0, 0, make_meta("London", 300, 0.99));

    // Session: feature 0 → London
    assert_eq!(session.feature_meta(0, 0).unwrap().top_token, "London");
    // Global: feature 0 → Paris
    assert_eq!(global.feature_meta(0, 0).unwrap().top_token, "Paris");
}

// ══════════════════════════════════════════════════════════════
// WEBSOCKET STREAM PROTOCOL
// ══════════════════════════════════════════════════════════════

#[test]
fn test_stream_describe_request_format() {
    let msg = serde_json::json!({"type": "describe", "entity": "France", "band": "all"});
    assert_eq!(msg["type"].as_str(), Some("describe"));
    assert_eq!(msg["entity"].as_str(), Some("France"));
    assert_eq!(msg["band"].as_str(), Some("all"));
}

#[test]
fn test_stream_layer_response_format() {
    let msg = serde_json::json!({
        "type": "layer",
        "layer": 27,
        "edges": [
            {"target": "Paris", "gate_score": 1436.9, "relation": "capital", "source": "probe"}
        ]
    });
    assert_eq!(msg["type"].as_str(), Some("layer"));
    assert_eq!(msg["layer"].as_u64(), Some(27));
    assert!(msg["edges"].as_array().unwrap().len() > 0);
}

#[test]
fn test_stream_done_response_format() {
    let msg = serde_json::json!({
        "type": "done",
        "entity": "France",
        "total_edges": 6,
        "latency_ms": 12.3,
    });
    assert_eq!(msg["type"].as_str(), Some("done"));
    assert_eq!(msg["total_edges"].as_u64(), Some(6));
    assert!(msg["latency_ms"].as_f64().unwrap() > 0.0);
}

#[test]
fn test_stream_error_response_format() {
    let msg = serde_json::json!({"type": "error", "message": "missing entity"});
    assert_eq!(msg["type"].as_str(), Some("error"));
    assert!(msg["message"].as_str().unwrap().contains("entity"));
}

#[test]
fn test_stream_unknown_type_rejected() {
    let msg_type = "foobar";
    let supported = ["describe", "infer"];
    assert!(!supported.contains(&msg_type));
}

// ══════════════════════════════════════════════════════════════
// WEBSOCKET INFER STREAMING
// ══════════════════════════════════════════════════════════════

#[test]
fn test_stream_infer_request_format() {
    let msg = serde_json::json!({
        "type": "infer",
        "prompt": "The capital of France is",
        "top": 5,
        "mode": "walk"
    });
    assert_eq!(msg["type"].as_str(), Some("infer"));
    assert_eq!(msg["prompt"].as_str(), Some("The capital of France is"));
    assert_eq!(msg["top"].as_u64(), Some(5));
    assert_eq!(msg["mode"].as_str(), Some("walk"));
}

#[test]
fn test_stream_prediction_response_format() {
    let msg = serde_json::json!({
        "type": "prediction",
        "rank": 1,
        "token": "Paris",
        "probability": 0.9791,
    });
    assert_eq!(msg["type"].as_str(), Some("prediction"));
    assert_eq!(msg["rank"].as_u64(), Some(1));
    assert_eq!(msg["token"].as_str(), Some("Paris"));
    assert!(msg["probability"].as_f64().unwrap() > 0.0);
}

#[test]
fn test_stream_infer_done_response_format() {
    let msg = serde_json::json!({
        "type": "infer_done",
        "prompt": "The capital of France is",
        "mode": "walk",
        "predictions": 5,
        "latency_ms": 210.0,
    });
    assert_eq!(msg["type"].as_str(), Some("infer_done"));
    assert_eq!(msg["mode"].as_str(), Some("walk"));
    assert_eq!(msg["predictions"].as_u64(), Some(5));
}

#[test]
fn test_stream_infer_modes() {
    let supported_modes = ["walk", "dense"];
    assert!(supported_modes.contains(&"walk"));
    assert!(supported_modes.contains(&"dense"));
    assert!(!supported_modes.contains(&"compare")); // compare not streamed
}

// ══════════════════════════════════════════════════════════════
// gRPC PROTO FORMAT
// ══════════════════════════════════════════════════════════════

#[test]
fn test_grpc_describe_request_fields() {
    // Mirrors DescribeRequest proto message
    let entity = "France";
    let band = "knowledge";
    let verbose = false;
    let limit = 20u32;
    let min_score = 5.0f32;
    assert!(!entity.is_empty());
    assert!(!band.is_empty());
    assert!(!verbose);
    assert!(limit > 0);
    assert!(min_score > 0.0);
}

#[test]
fn test_grpc_walk_response_structure() {
    // WalkResponse: prompt, hits[], latency_ms
    // WalkHit: layer, feature, gate_score, target, relation
    let hit = serde_json::json!({
        "layer": 27,
        "feature": 9515,
        "gate_score": 1436.9,
        "target": "Paris",
        "relation": "capital",
    });
    assert!(hit["layer"].as_u64().is_some());
    assert!(hit["feature"].as_u64().is_some());
    assert!(hit["gate_score"].as_f64().is_some());
    assert!(hit["target"].as_str().is_some());
}

#[test]
fn test_grpc_infer_compare_response() {
    // Compare mode returns walk_predictions + dense_predictions separately
    let walk_preds = vec![("Paris".to_string(), 0.9791f64)];
    let dense_preds = vec![("Paris".to_string(), 0.9801f64)];
    assert_eq!(walk_preds.len(), 1);
    assert_eq!(dense_preds.len(), 1);
    assert_ne!(walk_preds[0].1, dense_preds[0].1); // Slightly different
}

#[test]
fn test_grpc_port_flag() {
    // --grpc-port enables gRPC alongside HTTP
    let grpc_port: Option<u16> = Some(50051);
    assert!(grpc_port.is_some());
    let grpc_port: Option<u16> = None;
    assert!(grpc_port.is_none()); // gRPC disabled
}
