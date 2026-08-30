//! Error types → HTTP status codes.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use utoipa::ToSchema;

/// JSON body returned for every error response.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorBody {
    /// Human-readable error message.
    pub error: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("bad request: {0}")]
    BadRequest(String),

    /// The request is well-formed but the server's current state
    /// refuses it — a lifecycle mutation (`POST`/`DELETE
    /// /v1/runtime/model`) that conflicts with what's already
    /// happening (a load/unload in progress, a different model
    /// already bound) or with a static topology invariant
    /// (`docs/runtime-lifecycle-design.md` §7). Distinct from
    /// `BadRequest`: retrying the identical request later, once the
    /// conflicting state has changed, can succeed.
    #[error("conflict: {0}")]
    Conflict(String),

    #[error("inference not available: {0}")]
    #[allow(dead_code)]
    InferenceUnavailable(String),

    #[error("internal error: {0}")]
    Internal(String),

    /// Inference handler exceeded the server-side deadline.  We drop
    /// the in-flight `spawn_blocking` future, log the original
    /// elapsed time, and respond `504 Gateway Timeout` so the
    /// client can decide whether to retry.  The blocking thread
    /// keeps running to completion in the background — we don't
    /// have cooperative cancellation on the inference path — but it
    /// no longer holds up the HTTP handler or the next request.
    #[error("inference timed out: {0}")]
    Timeout(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ServerError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            ServerError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ServerError::Conflict(msg) => (StatusCode::CONFLICT, msg.clone()),
            ServerError::InferenceUnavailable(msg) => {
                (StatusCode::SERVICE_UNAVAILABLE, msg.clone())
            }
            ServerError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            ServerError::Timeout(msg) => (StatusCode::GATEWAY_TIMEOUT, msg.clone()),
        };

        (status, axum::Json(ErrorBody { error: message })).into_response()
    }
}
