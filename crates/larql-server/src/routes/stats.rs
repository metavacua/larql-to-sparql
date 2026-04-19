//! GET /v1/stats

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

fn build_stats(model: &LoadedModel) -> serde_json::Value {
    let config = &model.config;
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
    let features_per_layer = if !config.layers.is_empty() {
        config.layers[0].num_features
    } else {
        0
    };

    let layer_bands = config.layer_bands.as_ref().map(|b| {
        serde_json::json!({
            "syntax": [b.syntax.0, b.syntax.1],
            "knowledge": [b.knowledge.0, b.knowledge.1],
            "output": [b.output.0, b.output.1],
        })
    });

    let has_inference = config.extract_level == larql_vindex::ExtractLevel::Inference
        || config.extract_level == larql_vindex::ExtractLevel::All
        || config.has_model_weights;

    let mode = if model.embed_only {
        "embed-service"
    } else if model.ffn_only {
        "ffn-service"
    } else {
        "full"
    };

    serde_json::json!({
        "model": config.model,
        "family": config.family,
        "mode": mode,
        "layers": config.num_layers,
        "features": total_features,
        "features_per_layer": features_per_layer,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "extract_level": config.extract_level.to_string(),
        "dtype": config.dtype.to_string(),
        "layer_bands": layer_bands,
        "loaded": {
            "browse": !model.embed_only,
            "inference": has_inference && !model.infer_disabled,
            "ffn_service": !model.embed_only,
            "embed_service": true,
        },
    })
}

pub async fn handle_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    Ok(Json(build_stats(model)))
}

pub async fn handle_stats_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    Ok(Json(build_stats(model)))
}
