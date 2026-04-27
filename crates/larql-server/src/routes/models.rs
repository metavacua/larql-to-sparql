//! GET /v1/models

use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use crate::state::AppState;

pub async fn handle_models(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    state.bump_requests();

    let models: Vec<serde_json::Value> = state
        .models
        .iter()
        .map(|m| {
            let total_features: usize = m.config.layers.iter().map(|l| l.num_features).sum();
            serde_json::json!({
                "id": m.id,
                "path": if state.is_multi_model() {
                    format!("/v1/{}", m.id)
                } else {
                    "/v1".to_string()
                },
                "features": total_features,
                "loaded": true,
            })
        })
        .collect();

    Json(serde_json::json!({ "models": models }))
}
