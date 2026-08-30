//! N0-router — the OpenAI surface on the grid front door.
//!
//! Clients point an unmodified `openai` SDK at the *router* and the
//! router forwards each request to a grid server that announced
//! `serves_openai` (full model, inference enabled — see
//! `AnnounceMsg.serves_openai` in grid.proto). The router adds no
//! OpenAI semantics of its own:
//!
//! - `GET /v1/models` aggregates the distinct model ids across
//!   OpenAI-capable servers into the OpenAI list shape.
//! - `POST /v1/chat/completions` / `/v1/completions` /
//!   `/v1/embeddings` forward the body verbatim to the least-loaded
//!   capable server (matching the request's `model` field when set)
//!   and stream the response back unbuffered, so SSE passes through.
//!
//! Static `--shards` maps are layer slices with no capability signal,
//! so the OpenAI surface is grid-only: without a grid (or without any
//! capable server) the endpoints answer 503 in the OpenAI error shape.
//! The Responses API (`/v1/responses` + by-id retrieval) is proxied
//! with sticky routing in the [`responses`] submodule — stored
//! responses are process-local to the producing server.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::http::AppState;

pub mod responses;

pub use responses::ResponseRouteStore;

/// Path of the Responses API collection endpoint, shared by the route
/// table and the sticky by-id proxy.
pub(crate) const RESPONSES_PATH: &str = "/v1/responses";

/// `object` value on the models list envelope.
const LIST_OBJECT: &str = "list";
/// `object` value on each model entry.
const MODEL_OBJECT: &str = "model";
/// `owned_by` reported for every aggregated model (matches larql-server).
const OWNED_BY: &str = "larql";
/// OpenAI error `type` for client-side request problems.
const INVALID_REQUEST_ERROR: &str = "invalid_request_error";
/// OpenAI error `type` for backend unavailability.
const SERVER_ERROR: &str = "server_error";
/// OpenAI error `code` when the requested model is not on the grid.
const MODEL_NOT_FOUND: &str = "model_not_found";

/// One OpenAI-capable backend candidate, snapshotted out of the grid
/// lock so selection and dispatch never hold it.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenAIBackend {
    pub model_id: String,
    pub listen_url: String,
    pub requests_in_flight: u32,
}

/// Snapshot every server that announced `serves_openai`.
fn openai_backends(state: &AppState) -> Vec<OpenAIBackend> {
    let Some(grid) = &state.grid else {
        return Vec::new();
    };
    let guard = grid.read();
    guard
        .servers()
        .filter(|(_, e)| e.serves_openai)
        .map(|(_, e)| OpenAIBackend {
            model_id: e.model_id.clone(),
            listen_url: e.listen_url.clone(),
            requests_in_flight: e.requests_in_flight,
        })
        .collect()
}

/// Why no backend could be selected. Small by design (clippy
/// `result_large_err`) — rendered to an OpenAI error response at the
/// handler boundary via [`SelectError::into_response`].
#[derive(Debug, PartialEq, Eq)]
pub enum SelectError {
    /// The grid has no OpenAI-capable server at all (or no grid).
    NoCapableServer,
    /// Capable servers exist, but none serves the requested model.
    UnknownModel(String),
}

impl SelectError {
    fn into_response(self) -> Response {
        match self {
            SelectError::NoCapableServer => openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no OpenAI-capable server is registered on the grid",
                SERVER_ERROR,
                None,
            ),
            SelectError::UnknownModel(model) => openai_error(
                StatusCode::NOT_FOUND,
                &format!("model {model:?} is not served by any OpenAI-capable grid server"),
                INVALID_REQUEST_ERROR,
                Some(MODEL_NOT_FOUND),
            ),
        }
    }
}

/// Pick the least-loaded capable backend, restricted to `model` when
/// the request names one. Pure over the snapshot so it unit-tests
/// without a grid.
pub fn select_backend(
    backends: &[OpenAIBackend],
    model: Option<&str>,
) -> Result<OpenAIBackend, SelectError> {
    if backends.is_empty() {
        return Err(SelectError::NoCapableServer);
    }
    let candidates: Vec<&OpenAIBackend> = match model {
        Some(m) => backends.iter().filter(|b| b.model_id == m).collect(),
        None => backends.iter().collect(),
    };
    match candidates.into_iter().min_by_key(|b| b.requests_in_flight) {
        Some(b) => Ok(b.clone()),
        None => Err(SelectError::UnknownModel(
            model.unwrap_or_default().to_string(),
        )),
    }
}

/// OpenAI error envelope — the same shape larql-server emits, so a
/// client cannot tell router-originated errors from server ones.
fn openai_error(status: StatusCode, message: &str, kind: &str, code: Option<&str>) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": kind,
            "param": serde_json::Value::Null,
            "code": code,
        }
    });
    (status, axum::Json(body)).into_response()
}

/// Extract the `model` field from a request body, tolerating non-JSON
/// bodies (the backend will produce the authoritative 400).
pub fn model_of(body: &Bytes) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()?
        .get("model")?
        .as_str()
        .map(str::to_string)
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `GET /v1/models` — distinct model ids across OpenAI-capable servers.
pub async fn handle_models(State(state): State<Arc<AppState>>) -> Response {
    let mut ids: Vec<String> = openai_backends(&state)
        .into_iter()
        .map(|b| b.model_id)
        .collect();
    ids.sort();
    ids.dedup();
    let data: Vec<serde_json::Value> = ids
        .into_iter()
        .map(|id| {
            serde_json::json!({
                "id": id,
                "object": MODEL_OBJECT,
                // The router has no per-model load time; expose "unknown"
                // the way OpenAI's spec allows — a zero epoch.
                "created": 0,
                "owned_by": OWNED_BY,
            })
        })
        .collect();
    axum::Json(serde_json::json!({"object": LIST_OBJECT, "data": data})).into_response()
}

pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_openai(&state, "/v1/chat/completions", &headers, body).await
}

pub async fn handle_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_openai(&state, "/v1/completions", &headers, body).await
}

pub async fn handle_embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_openai(&state, "/v1/embeddings", &headers, body).await
}

/// Forward one OpenAI request to a capable backend and stream the
/// response back without buffering (SSE chat streams pass through
/// chunk-by-chunk; the shard client's request timeout still bounds the
/// whole exchange).
async fn proxy_openai(state: &AppState, path: &str, headers: &HeaderMap, body: Bytes) -> Response {
    let backend = match select_backend(&openai_backends(state), model_of(&body).as_deref()) {
        Ok(b) => b,
        Err(e) => return e.into_response(),
    };
    match send_to_backend(
        state,
        &backend,
        reqwest::Method::POST,
        path,
        headers,
        Some(body),
    )
    .await
    {
        Ok(upstream) => passthrough_response(upstream, Body::from_stream),
        Err(resp) => *resp,
    }
}

/// Send one request to a chosen backend, passing the client's bearer
/// token through (the backend may enforce its own `--api-key`).
///
/// Errors come back as a ready-to-return OpenAI 502, boxed: an
/// `axum::Response` is far larger than the success value, and an
/// unboxed `Err` would make every caller's `Result` carry that width on
/// the hot path.
pub(super) async fn send_to_backend(
    state: &AppState,
    backend: &OpenAIBackend,
    method: reqwest::Method,
    path: &str,
    headers: &HeaderMap,
    body: Option<Bytes>,
) -> Result<reqwest::Response, Box<Response>> {
    let url = format!("{}{}", backend.listen_url, path);
    let mut req = state.client.request(method, &url);
    if let Some(b) = body {
        req = req
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(b.to_vec());
    }
    if let Some(auth) = headers.get(header::AUTHORIZATION) {
        if let Ok(v) = auth.to_str() {
            req = req.header(reqwest::header::AUTHORIZATION, v);
        }
    }
    req.send().await.map_err(|e| {
        Box::new(openai_error(
            StatusCode::BAD_GATEWAY,
            &format!("backend {}: {e}", backend.listen_url),
            SERVER_ERROR,
            None,
        ))
    })
}

/// Re-emit an upstream response with its status + content type, with
/// the body assembled by `make_body` — identity streaming for the
/// plain proxy, an id-capturing wrapper for `/v1/responses`.
pub(super) fn passthrough_response<F>(upstream: reqwest::Response, make_body: F) -> Response
where
    F: FnOnce(futures::stream::BoxStream<'static, Result<Bytes, reqwest::Error>>) -> Body,
{
    use futures::StreamExt as _;
    let status = upstream.status().as_u16();
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type)
        .body(make_body(upstream.bytes_stream().boxed()))
        .unwrap_or_else(|e| {
            openai_error(
                StatusCode::BAD_GATEWAY,
                &format!("assemble proxied response: {e}"),
                SERVER_ERROR,
                None,
            )
        })
}

#[cfg(test)]
mod tests;
