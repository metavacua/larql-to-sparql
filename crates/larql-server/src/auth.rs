//! API key authentication middleware.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::Response;

use crate::state::AppState;

/// Middleware that validates the Authorization: Bearer <api_key> header.
/// If no api_key is configured, all requests pass through.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let required_key = match &state.api_key {
        Some(key) => key,
        None => return Ok(next.run(request).await),
    };

    // Allow health checks without auth.
    if request.uri().path() == "/v1/health" {
        return Ok(next.run(request).await);
    }

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header[7..];
            if token == required_key {
                Ok(next.run(request).await)
            } else {
                Err(StatusCode::UNAUTHORIZED)
            }
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
