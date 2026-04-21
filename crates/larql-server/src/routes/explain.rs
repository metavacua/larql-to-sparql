//! POST /v1/explain-infer — walk inference with per-layer feature trace.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize)]
pub struct ExplainRequest {
    pub prompt: String,
    #[serde(default = "default_top")]
    pub top: usize,
    #[serde(default = "default_per_layer")]
    pub per_layer: usize,
    #[serde(default = "default_band")]
    pub band: String,
    #[serde(default)]
    pub relations_only: bool,
    #[serde(default)]
    pub with_attention: bool,
}

fn default_top() -> usize { 5 }
fn default_per_layer() -> usize { 3 }
fn default_band() -> String { "all".into() }

fn explain_infer(
    model: &LoadedModel,
    req: &ExplainRequest,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    let weights = model.get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let encoding = model.tokenizer.encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Decode tokens for attention display (None for special tokens like BOS/EOS)
    let token_strs: Vec<Option<String>> = if req.with_attention {
        token_ids.iter().map(|&id| {
            larql_inference::decode_token(&model.tokenizer, id)
        }).collect()
    } else {
        Vec::new()
    };

    let patched = model.patched.blocking_read();
    let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(weights, &*patched);

    let (predictions_raw, attention_captures, lens_residuals) = if req.with_attention {
        let r = larql_inference::predict_with_ffn_attention(
            weights, &model.tokenizer, &token_ids, req.top, &walk_ffn,
        );
        (r.predictions, r.attention, r.residuals)
    } else {
        let r = larql_inference::predict_with_ffn(
            weights, &model.tokenizer, &token_ids, req.top, &walk_ffn,
        );
        (r.predictions, Vec::new(), Vec::new())
    };
    let residuals = walk_ffn.take_residuals();
    let (predictions_raw, knn_override) = larql_inference::apply_knn_override(
        predictions_raw,
        &residuals,
        Some(&patched.knn_store),
        req.top,
    );
    let trace_layers = larql_inference::walk_trace_from_residuals(&residuals, &*patched);

    // Build logit lens: layer → (top_token, probability)
    let lens_map: std::collections::HashMap<usize, (String, f64)> = lens_residuals.iter()
        .filter_map(|(layer, residual)| {
            let pred = larql_inference::logit_lens_top1(weights, &model.tokenizer, residual)?;
            Some((*layer, pred))
        })
        .collect();

    // Build attention lookup: layer → top attended tokens
    let attention_map: std::collections::HashMap<usize, Vec<(String, f32)>> = {
        let mut map = std::collections::HashMap::new();
        for cap in &attention_captures {
            let n_heads = cap.weights.heads.len();
            if n_heads == 0 || token_strs.is_empty() { continue; }
            let seq_len = cap.weights.heads[0].len();
            let mut avg = vec![0.0f32; seq_len];
            for head in &cap.weights.heads {
                for (j, &w) in head.iter().enumerate() {
                    avg[j] += w;
                }
            }
            for v in avg.iter_mut() { *v /= n_heads as f32; }
            let mut pairs: Vec<(String, f32)> = avg.iter().copied().enumerate()
                .filter_map(|(j, w)| {
                    let tok = token_strs.get(j)?.as_ref()?;
                    Some((tok.trim().to_string(), w))
                })
                .collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            pairs.truncate(3);
            map.insert(cap.layer, pairs);
        }
        map
    };

    // Resolve band to layer range
    let last = model.config.num_layers.saturating_sub(1);
    let bands = model.config.layer_bands.clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&model.config.family, model.config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        });
    let layer_range: Option<(usize, usize)> = match req.band.as_str() {
        "syntax" => Some(bands.syntax),
        "knowledge" => Some(bands.knowledge),
        "output" => Some(bands.output),
        _ => None,
    };

    let predictions: Vec<serde_json::Value> = predictions_raw.iter()
        .map(|(tok, prob)| serde_json::json!({"token": tok, "probability": (*prob * 10000.0).round() / 10000.0}))
        .collect();

    let mut layers = Vec::new();
    for (layer, hits) in &trace_layers {
        if let Some((lo, hi)) = layer_range {
            if *layer < lo || *layer > hi {
                continue;
            }
        }
        // When relations_only, re-sort so positive gates rank first
        let ordered_hits: Vec<_> = if req.relations_only {
            let mut lh: Vec<_> = hits.iter()
                .filter(|hit| model.probe_labels.contains_key(&(*layer, hit.feature)))
                .collect();
            lh.sort_by(|a, b| {
                let a_pos = a.gate_score > 0.0;
                let b_pos = b.gate_score > 0.0;
                match (a_pos, b_pos) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => b.gate_score.abs().partial_cmp(&a.gate_score.abs())
                        .unwrap_or(std::cmp::Ordering::Equal),
                }
            });
            lh
        } else {
            hits.iter().collect()
        };

        let features: Vec<serde_json::Value> = ordered_hits.iter()
            .filter_map(|hit| {
                let relation = model.probe_labels.get(&(*layer, hit.feature)).cloned();
                if req.relations_only && relation.is_none() {
                    return None;
                }
                let top_tokens: Vec<String> = hit.meta.top_k.iter()
                    .take(3)
                    .map(|t| t.token.trim().to_string())
                    .collect();
                Some(serde_json::json!({
                    "feature": hit.feature,
                    "gate_score": (hit.gate_score * 10.0).round() / 10.0,
                    "top_token": hit.meta.top_token.trim(),
                    "top_tokens": top_tokens,
                    "relation": relation,
                }))
            })
            .take(req.per_layer)
            .collect();
        if !features.is_empty() {
            let mut layer_obj = serde_json::json!({
                "layer": layer,
                "features": features,
            });
            if let Some(attn) = attention_map.get(layer) {
                let attn_json: Vec<serde_json::Value> = attn.iter()
                    .map(|(tok, w)| serde_json::json!({"token": tok, "weight": (*w * 1000.0).round() / 1000.0}))
                    .collect();
                layer_obj["attention"] = serde_json::json!(attn_json);
            }
            if let Some((tok, prob)) = lens_map.get(layer) {
                layer_obj["lens"] = serde_json::json!({"token": tok, "probability": (*prob * 10000.0).round() / 10000.0});
            }
            layers.push(layer_obj);
        }
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut body = serde_json::json!({
        "prompt": req.prompt,
        "predictions": predictions,
        "trace": layers,
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
    });
    if let Some(ovr) = knn_override {
        body["knn_override"] = serde_json::json!({
            "token": ovr.token,
            "cosine": ovr.cosine,
            "layer": ovr.layer,
        });
    }
    Ok(body)
}

pub async fn handle_explain(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExplainRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let result = tokio::task::spawn_blocking(move || explain_infer(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_explain_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Json(req): Json<ExplainRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    let model = Arc::clone(model);
    let result = tokio::task::spawn_blocking(move || explain_infer(&model, &req))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
