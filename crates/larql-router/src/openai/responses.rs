//! N0-router — sticky proxying for the Responses API.
//!
//! `/v1/responses` differs from the stateless OpenAI POSTs: a stored
//! response lives in the *producing server's* process-local store, so
//! `previous_response_id` chaining and `GET`/`DELETE
//! /v1/responses/{id}` must land on that same server. The router
//! learns each response's id by observing the bytes it proxies — the
//! envelope (and the streaming `response.created` event) leads with
//! `"id":"resp_..."` — and keeps a bounded id → backend map:
//!
//! - `POST /v1/responses` routes via the map when the request carries
//!   a known `previous_response_id` (falling back to normal selection
//!   when the mapped backend has left the grid), and records the new
//!   response's id as the reply streams through.
//! - `GET`/`DELETE /v1/responses/{id}` route via the map; on a miss
//!   with exactly one capable server the request is proxied there (the
//!   single-server grid survives a router restart), otherwise the
//!   router answers 404 — it cannot know which server holds the id.
//!
//! The map is process-local and FIFO-bounded like the server-side
//! response store: an evicted route degrades to the miss behaviour
//! above, mirroring the server's own eviction contract.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;

use crate::http::AppState;

use super::{
    openai_backends, openai_error, passthrough_response, select_backend, send_to_backend,
    OpenAIBackend,
};

/// OpenAI error `type` for a response id the router cannot place
/// (matches larql-server's `OpenAIError::not_found`).
const NOT_FOUND_ERROR: &str = "not_found_error";

/// Maximum retained id → backend routes before FIFO eviction. Matches
/// the order of magnitude of the server-side response store so the
/// router does not forget routes the servers still remember.
pub const MAX_ROUTED_RESPONSES: usize = 4096;

/// The marker that opens a response id in everything we proxy: the
/// JSON envelope's first field, and the `response.created` payload.
const RESPONSE_ID_MARKER: &[u8] = br#""id":"resp_"#;

/// How many leading bytes of a proxied response to scan for the id.
/// The id sits in the first field of the first JSON object either way;
/// this bound just keeps a pathological upstream from growing the
/// capture buffer.
const ID_SCAN_LIMIT: usize = 4096;

// ── Sticky route store ───────────────────────────────────────────────────────

#[derive(Default)]
struct RouteInner {
    by_id: HashMap<String, String>,
    order: VecDeque<String>,
}

/// Bounded FIFO map: response id → the `listen_url` of the backend
/// that produced (and stored) it.
#[derive(Default)]
pub struct ResponseRouteStore {
    inner: Mutex<RouteInner>,
}

impl ResponseRouteStore {
    pub fn insert(&self, id: &str, listen_url: &str) {
        let mut inner = self.lock();
        if inner
            .by_id
            .insert(id.to_string(), listen_url.to_string())
            .is_none()
        {
            inner.order.push_back(id.to_string());
        }
        while inner.by_id.len() > MAX_ROUTED_RESPONSES {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            inner.by_id.remove(&oldest);
        }
    }

    pub fn get(&self, id: &str) -> Option<String> {
        self.lock().by_id.get(id).cloned()
    }

    /// Remove one route; returns whether it existed.
    pub fn remove(&self, id: &str) -> bool {
        let mut inner = self.lock();
        let existed = inner.by_id.remove(id).is_some();
        if existed {
            inner.order.retain(|entry| entry != id);
        }
        existed
    }

    pub fn len(&self) -> usize {
        self.lock().by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RouteInner> {
        // A poisoned lock only means a panic mid-insert; recover rather
        // than wedge every subsequent Responses request.
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

// ── Response-id capture ──────────────────────────────────────────────────────

/// Scans the leading bytes of a proxied response for `"id":"resp_..."`.
/// Feed chunks in arrival order; returns the id once, even when the
/// marker straddles a chunk boundary.
pub struct IdCapture {
    buf: Vec<u8>,
    done: bool,
}

impl IdCapture {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            done: false,
        }
    }

    /// Observe one chunk; returns `Some(id)` the first time the full
    /// id becomes visible.
    pub fn observe(&mut self, chunk: &[u8]) -> Option<String> {
        if self.done {
            return None;
        }
        self.buf.extend_from_slice(chunk);
        if let Some(id) = extract_response_id(&self.buf) {
            self.done = true;
            self.buf = Vec::new();
            return Some(id);
        }
        if self.buf.len() > ID_SCAN_LIMIT {
            self.done = true;
            self.buf = Vec::new();
        }
        None
    }
}

impl Default for IdCapture {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the first `"id":"resp_..."` in `buf` and return the full id
/// (`resp_` prefix included). Message/function-call item ids
/// (`msg_...`, `fc_...`) never match because the marker pins the
/// `resp_` prefix.
pub fn extract_response_id(buf: &[u8]) -> Option<String> {
    let start = buf
        .windows(RESPONSE_ID_MARKER.len())
        .position(|w| w == RESPONSE_ID_MARKER)?;
    // The id's value starts at the opening quote after `"id":`.
    let value_start = start + RESPONSE_ID_MARKER.len() - b"resp_".len();
    let rest = &buf[value_start..];
    let end = rest.iter().position(|b| *b == b'"')?;
    String::from_utf8(rest[..end].to_vec()).ok()
}

// ── Request-body fields ──────────────────────────────────────────────────────

/// `previous_response_id` from a request body, when present.
pub fn previous_response_id_of(body: &Bytes) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()?
        .get("previous_response_id")?
        .as_str()
        .map(str::to_string)
}

// ── Backend resolution ───────────────────────────────────────────────────────

/// Resolve the backend for a by-id request (`GET`/`DELETE`, or a
/// chained `POST`): the recorded producer when it is still on the
/// grid; else the only capable server, if there is exactly one (it is
/// the only place the response could live); else `None`.
pub fn resolve_sticky(backends: &[OpenAIBackend], route: Option<String>) -> Option<OpenAIBackend> {
    if let Some(url) = route {
        if let Some(b) = backends.iter().find(|b| b.listen_url == url) {
            return Some(b.clone());
        }
    }
    match backends {
        [only] => Some(only.clone()),
        _ => None,
    }
}

fn unknown_response_error(id: &str) -> Response {
    openai_error(
        StatusCode::NOT_FOUND,
        &format!(
            "response {id:?} is not routable: the router has no record of which \
             grid server produced it"
        ),
        NOT_FOUND_ERROR,
        None,
    )
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `POST /v1/responses` — sticky on `previous_response_id`, records
/// the produced response's id from the proxied bytes.
pub async fn handle_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let backends = openai_backends(&state);
    let chained = previous_response_id_of(&body)
        .and_then(|id| state.openai_responses.get(&id))
        .and_then(|url| backends.iter().find(|b| b.listen_url == url).cloned());
    let backend = match chained {
        Some(b) => b,
        None => match select_backend(&backends, super::model_of(&body).as_deref()) {
            Ok(b) => b,
            Err(e) => return e.into_response(),
        },
    };

    let upstream = match send_to_backend(
        &state,
        &backend,
        reqwest::Method::POST,
        super::RESPONSES_PATH,
        &headers,
        Some(body),
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    // Only a success can mint a stored response worth routing back to.
    let record = upstream.status().is_success();
    let state_cb = Arc::clone(&state);
    let backend_url = backend.listen_url.clone();
    passthrough_response(upstream, move |stream| {
        use futures::StreamExt as _;
        let mut capture = IdCapture::new();
        Body::from_stream(stream.map(move |chunk| {
            if record {
                if let Ok(bytes) = &chunk {
                    if let Some(id) = capture.observe(bytes) {
                        state_cb.openai_responses.insert(&id, &backend_url);
                    }
                }
            }
            chunk
        }))
    })
}

/// `GET /v1/responses/{id}` — proxy to the recorded producer.
pub async fn handle_get_response(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    proxy_by_id(&state, reqwest::Method::GET, &response_id, &headers).await
}

/// `DELETE /v1/responses/{id}` — proxy to the recorded producer and
/// drop the route once the backend confirms the delete.
pub async fn handle_delete_response(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let resp = proxy_by_id(&state, reqwest::Method::DELETE, &response_id, &headers).await;
    if resp.status().is_success() {
        state.openai_responses.remove(&response_id);
    }
    resp
}

async fn proxy_by_id(
    state: &Arc<AppState>,
    method: reqwest::Method,
    response_id: &str,
    headers: &HeaderMap,
) -> Response {
    let backends = openai_backends(state);
    let Some(backend) = resolve_sticky(&backends, state.openai_responses.get(response_id)) else {
        return unknown_response_error(response_id);
    };
    let path = format!("{}/{response_id}", super::RESPONSES_PATH);
    match send_to_backend(state, &backend, method, &path, headers, None).await {
        Ok(upstream) => passthrough_response(upstream, Body::from_stream),
        Err(resp) => *resp,
    }
}
