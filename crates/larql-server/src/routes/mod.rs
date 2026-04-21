//! Router setup — maps URL paths to handlers.

pub mod describe;
pub mod embed;
pub mod explain;
pub mod health;
pub mod infer;
pub mod insert;
pub mod models;
pub mod patches;
pub mod relations;
pub mod select;
pub mod stats;
pub mod stream;
pub mod walk;
pub mod walk_ffn;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post, delete};

use crate::state::AppState;

/// Build the router for single-model serving.
pub fn single_model_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/describe", get(describe::handle_describe))
        .route("/v1/walk", get(walk::handle_walk))
        .route("/v1/select", post(select::handle_select))
        .route("/v1/relations", get(relations::handle_relations))
        .route("/v1/stats", get(stats::handle_stats))
        .route("/v1/infer", post(infer::handle_infer))
        .route("/v1/patches/apply", post(patches::handle_apply_patch))
        .route("/v1/patches", get(patches::handle_list_patches))
        .route("/v1/patches/{name}", delete(patches::handle_remove_patch))
        .route("/v1/walk-ffn", post(walk_ffn::handle_walk_ffn))
        .route("/v1/explain-infer", post(explain::handle_explain))
        .route("/v1/insert", post(insert::handle_insert))
        .route("/v1/stream", get(stream::handle_stream))
        .route("/v1/health", get(health::handle_health))
        .route("/v1/models", get(models::handle_models))
        // Embed server endpoints (always available, required for --embed-only mode)
        .route("/v1/embed", post(embed::handle_embed))
        .route("/v1/embed/{token_id}", get(embed::handle_embed_single))
        .route("/v1/logits", post(embed::handle_logits))
        .route("/v1/token/encode", get(embed::handle_token_encode))
        .route("/v1/token/decode", get(embed::handle_token_decode))
        .with_state(state)
}

/// Build the router for multi-model serving.
pub fn multi_model_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(health::handle_health))
        .route("/v1/models", get(models::handle_models))
        .route("/v1/{model_id}/describe", get(describe::handle_describe_multi))
        .route("/v1/{model_id}/walk", get(walk::handle_walk_multi))
        .route("/v1/{model_id}/select", post(select::handle_select_multi))
        .route("/v1/{model_id}/relations", get(relations::handle_relations_multi))
        .route("/v1/{model_id}/stats", get(stats::handle_stats_multi))
        .route("/v1/{model_id}/infer", post(infer::handle_infer_multi))
        .route("/v1/{model_id}/patches/apply", post(patches::handle_apply_patch_multi))
        .route("/v1/{model_id}/patches", get(patches::handle_list_patches_multi))
        .route("/v1/{model_id}/patches/{name}", delete(patches::handle_remove_patch_multi))
        .route("/v1/{model_id}/explain-infer", post(explain::handle_explain_multi))
        .route("/v1/{model_id}/insert", post(insert::handle_insert_multi))
        // Embed server endpoints for multi-model mode
        .route("/v1/{model_id}/embed", post(embed::handle_embed_multi))
        .route("/v1/{model_id}/embed/{token_id}", get(embed::handle_embed_single_multi))
        .route("/v1/{model_id}/logits", post(embed::handle_logits_multi))
        .route("/v1/{model_id}/token/encode", get(embed::handle_token_encode_multi))
        .route("/v1/{model_id}/token/decode", get(embed::handle_token_decode_multi))
        .with_state(state)
}
