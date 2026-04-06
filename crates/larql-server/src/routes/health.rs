//! GET /v1/health

use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use crate::state::AppState;

pub async fn handle_health(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    state.bump_requests();
    let uptime = state.started_at.elapsed().as_secs();
    let served = state
        .requests_served
        .load(std::sync::atomic::Ordering::Relaxed);

    Json(serde_json::json!({
        "status": "ok",
        "uptime_seconds": uptime,
        "requests_served": served,
    }))
}
