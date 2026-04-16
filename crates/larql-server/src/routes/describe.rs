//! GET /v1/describe — query all knowledge edges for an entity.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize)]
pub struct DescribeParams {
    pub entity: String,
    #[serde(default = "default_band")]
    pub band: String,
    #[serde(default)]
    pub verbose: bool,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_min_score")]
    pub min_score: f32,
}

fn default_band() -> String { "knowledge".into() }
fn default_limit() -> usize { 20 }
fn default_min_score() -> f32 { 5.0 }

fn describe_entity(
    model: &LoadedModel,
    params: &DescribeParams,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    let encoding = model
        .tokenizer
        .encode(params.entity.as_str(), false)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Ok(serde_json::json!({
            "entity": params.entity,
            "model": model.config.model,
            "edges": [],
            "latency_ms": 0.0,
        }));
    }

    let hidden = model.embeddings.shape()[1];
    let query = if token_ids.len() == 1 {
        model.embeddings.row(token_ids[0] as usize).mapv(|v| v * model.embed_scale)
    } else {
        let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            avg += &model.embeddings.row(tok as usize).mapv(|v| v * model.embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };

    let config = &model.config;
    let last = config.num_layers.saturating_sub(1);
    let bands = config
        .layer_bands
        .clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        });

    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();

    let scan_layers: Vec<usize> = match params.band.as_str() {
        "syntax" => all_layers.iter().copied()
            .filter(|l| *l >= bands.syntax.0 && *l <= bands.syntax.1)
            .collect(),
        "knowledge" => all_layers.iter().copied()
            .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
            .collect(),
        "output" => all_layers.iter().copied()
            .filter(|l| *l >= bands.output.0 && *l <= bands.output.1)
            .collect(),
        _ => all_layers,
    };

    let trace = patched.walk(&query, &scan_layers, params.limit);

    // Aggregate edges by target token (same logic as LQL DESCRIBE).
    struct EdgeInfo {
        gate: f32,
        layers: Vec<usize>,
        count: usize,
        original: String,
        also: Vec<String>,
        best_layer: usize,
        best_feature: usize,
    }

    let entity_lower = params.entity.to_lowercase();
    let mut edges: HashMap<String, EdgeInfo> = HashMap::new();

    for (layer_idx, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score < params.min_score {
                continue;
            }

            let tok = &hit.meta.top_token;
            let tok_trimmed = tok.trim();
            if tok_trimmed.is_empty() || tok_trimmed.len() < 2 {
                continue;
            }
            if tok_trimmed.to_lowercase() == entity_lower {
                continue;
            }

            let also: Vec<String> = hit
                .meta
                .top_k
                .iter()
                .filter(|t| {
                    let tt = t.token.trim();
                    tt.to_lowercase() != tok.to_lowercase()
                        && tt.to_lowercase() != entity_lower
                        && tt.len() >= 2
                        && t.logit > 0.0
                })
                .take(3)
                .map(|t| t.token.trim().to_string())
                .collect();

            let key = tok.to_lowercase();
            let entry = edges.entry(key).or_insert_with(|| EdgeInfo {
                gate: 0.0,
                layers: Vec::new(),
                best_feature: hit.feature,
                count: 0,
                original: tok_trimmed.to_string(),
                also,
                best_layer: *layer_idx,
            });

            if hit.gate_score > entry.gate {
                entry.gate = hit.gate_score;
                entry.best_layer = *layer_idx;
                entry.best_feature = hit.feature;
            }
            if !entry.layers.contains(layer_idx) {
                entry.layers.push(*layer_idx);
            }
            entry.count += 1;
        }
    }

    let mut ranked: Vec<&EdgeInfo> = edges.values().collect();
    ranked.sort_by(|a, b| b.gate.partial_cmp(&a.gate).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(params.limit);

    let edge_json: Vec<serde_json::Value> = ranked
        .iter()
        .map(|info| {
            let min_l = *info.layers.iter().min().unwrap_or(&0);
            let max_l = *info.layers.iter().max().unwrap_or(&0);

            let mut edge = serde_json::json!({
                "target": info.original,
                "gate_score": (info.gate * 10.0).round() / 10.0,
                "layer": info.best_layer,
            });

            // Probe-confirmed relation label.
            if let Some(label) = model.probe_labels.get(&(info.best_layer, info.best_feature)) {
                edge["relation"] = serde_json::json!(label);
                edge["source"] = serde_json::json!("probe");
            }

            if params.verbose {
                edge["layer_max"] = serde_json::json!(max_l);
                edge["layer_min"] = serde_json::json!(min_l);
                edge["count"] = serde_json::json!(info.count);
            }

            if !info.also.is_empty() {
                edge["also"] = serde_json::json!(info.also);
            }

            edge
        })
        .collect();

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(serde_json::json!({
        "entity": params.entity,
        "model": config.model,
        "edges": edge_json,
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
    }))
}

async fn describe_with_cache(
    state: &Arc<AppState>,
    model: &Arc<LoadedModel>,
    headers: &HeaderMap,
    params: DescribeParams,
) -> Result<Response, ServerError> {
    // Check cache.
    let cache_key = if state.describe_cache.is_enabled() {
        let key = crate::cache::DescribeCache::key(
            &model.id,
            &params.entity,
            &params.band,
            params.limit,
            params.min_score,
        );
        if let Some(cached) = state.describe_cache.get(&key) {
            let etag = crate::etag::compute_etag(&cached);
            let if_none_match = headers.get("if-none-match").and_then(|v| v.to_str().ok());
            if crate::etag::matches_etag(if_none_match, &etag) {
                return Ok((
                    axum::http::StatusCode::NOT_MODIFIED,
                    [("etag", etag)],
                ).into_response());
            }
            return Ok((
                [
                    ("etag", etag),
                    ("cache-control", "public, max-age=86400".into()),
                ],
                Json(cached),
            ).into_response());
        }
        Some(key)
    } else {
        None
    };

    let model = Arc::clone(model);
    let result = tokio::task::spawn_blocking(move || describe_entity(&model, &params))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;

    // Store in cache.
    if let Some(key) = cache_key {
        state.describe_cache.put(key, result.clone());
    }

    let etag = crate::etag::compute_etag(&result);
    Ok((
        [
            ("etag", etag),
            ("cache-control", "public, max-age=86400".into()),
        ],
        Json(result),
    ).into_response())
}

pub async fn handle_describe(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<DescribeParams>,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    describe_with_cache(&state, model, &headers, params).await
}

pub async fn handle_describe_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Query(params): Query<DescribeParams>,
) -> Result<Response, ServerError> {
    state.bump_requests();
    let model = state
        .model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    describe_with_cache(&state, model, &headers, params).await
}
