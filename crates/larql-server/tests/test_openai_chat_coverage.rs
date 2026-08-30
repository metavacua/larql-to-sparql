//! Coverage push for `routes/openai/chat/*` (handler, stream, tools).
//!
//! Uses the Q4K synthetic vindex so `generate_with_sampling` actually
//! runs without panicking on Q4K slices. Drives the chat completion
//! handler through validation branches, the chat-template rendering,
//! non-streaming and streaming responses, tool calls, and structured
//! output schemas.
//!
//! Fixture lifetime matters: the synthetic weights load lazily from the
//! fixture's temp dir on first generation, and for SSE responses that
//! happens while the *body* is being drained — after the handler has
//! already returned. Every helper here therefore keeps the fixture
//! alive until the body is fully consumed; dropping it earlier makes
//! generation fail with a weights-load error that still ends in a
//! well-formed `[DONE]` stream, silently gutting coverage.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

fn q4k_app() -> (
    axum::Router,
    common::synthetic_q4k_vindex::SyntheticQ4kVindex,
) {
    let (model, fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    (larql_server::routes::single_model_router(state), fixture)
}

async fn send_chat(app: &axum::Router, body: serde_json::Value) -> axum::http::Response<Body> {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
}

/// POST the request and fully drain the response body (fixture held
/// alive throughout); returns `(status, body_text)`.
async fn post_chat_drained(body: serde_json::Value) -> (StatusCode, String) {
    let (app, _fixture) = q4k_app();
    let resp = send_chat(&app, body).await;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn parse_json(body: &str) -> serde_json::Value {
    serde_json::from_str(body).unwrap_or_else(|_| serde_json::json!({"raw": body}))
}

// ── non-streaming ────────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_non_streaming_basic_returns_completion() {
    let (status, body) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "the capital of France is"}],
        "max_tokens": 4,
    }))
    .await;
    let v = parse_json(&body);
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["object"], "chat.completion", "{v}");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant", "{v}");
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0, "{v}");
}

#[tokio::test]
async fn chat_with_system_message_renders_template() {
    let (status, body) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "x"},
        ],
        "max_tokens": 2,
    }))
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn chat_with_sampling_params_runs_sampler() {
    let (status, body) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 2,
        "temperature": 0.5,
        "top_p": 0.9,
        "seed": 42,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    }))
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn chat_with_logprobs_returns_logprob_content() {
    let (status, body) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "the capital of France is"}],
        "max_tokens": 2,
        "logprobs": true,
    }))
    .await;
    let v = parse_json(&body);
    assert_eq!(status, StatusCode::OK, "{v}");
    assert!(
        v["choices"][0]["logprobs"].is_object(),
        "logprobs requested but missing: {v}"
    );
}

#[tokio::test]
async fn chat_with_stop_strings_runs_stop_branch() {
    // The Q4K fixture's CPU path can emit empty output, so the stop
    // can't be guaranteed to fire; asserting 200 + a finish reason
    // still drives the stop-checking code. The deterministic stop-trim
    // contract is pinned in `test_openai_responses_v3.rs` (V3 emits
    // real per-token text) and the byte-accounting helpers are unit
    // tested in `routes/openai/chat/tests`.
    let (status, body) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "the capital of France is"}],
        "max_tokens": 8,
        "stop": ["the", "capital", "of", "France", "is", "Paris", "a", "b", "c", "x", "y", "z"],
    }))
    .await;
    let v = parse_json(&body);
    assert_eq!(status, StatusCode::OK, "{v}");
    let reason = v["choices"][0]["finish_reason"].as_str().unwrap_or("");
    assert!(reason == "stop" || reason == "length", "{v}");
}

#[tokio::test]
async fn chat_with_response_format_json_object_runs_constrained_decode() {
    let (status, body) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 4,
        "response_format": {"type": "json_object"},
    }))
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

// ── validation ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_empty_messages_returns_400() {
    let (status, _) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [],
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_n_gt_1_returns_400() {
    let (status, _) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "x"}],
        "n": 2,
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_tool_message_without_content_returns_400() {
    let (status, body) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "tool", "tool_call_id": "call_1"},
        ],
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("requires content"), "{body}");
}

#[tokio::test]
async fn chat_user_message_without_content_returns_400() {
    let (status, body) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [{"role": "user"}],
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("requires content"), "{body}");
}

#[tokio::test]
async fn chat_invalid_json_returns_400() {
    let (app, _fixture) = q4k_app();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── streaming ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_streaming_basic_emits_role_content_and_done() {
    let (app, _fixture) = q4k_app();
    let resp = send_chat(
        &app,
        serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "the capital of France is"}],
            "max_tokens": 4,
            "stream": true,
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.contains("event-stream"), "expected SSE; got {ct}");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("chat.completion.chunk"), "{body}");
    assert!(
        body.contains(r#""role":"assistant""#),
        "first chunk must carry the role delta:\n{body}"
    );
    assert!(
        body.contains(r#""finish_reason":"stop""#) || body.contains(r#""finish_reason":"length""#),
        "final chunk must carry a finish reason:\n{body}"
    );
    assert!(body.contains("[DONE]"), "{body}");
    assert!(
        !body.contains("error"),
        "stream must not carry an error chunk:\n{body}"
    );
}

#[tokio::test]
async fn chat_streaming_with_stop_strings_stops_early() {
    let (app, _fixture) = q4k_app();
    let resp = send_chat(
        &app,
        serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "the capital of France is"}],
            "max_tokens": 8,
            "stream": true,
            "stop": ["the", "capital", "of", "France", "is", "Paris", "a", "b", "c", "x", "y", "z"],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(
        body.contains(r#""finish_reason":"stop""#) || body.contains(r#""finish_reason":"length""#),
        "{body}"
    );
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn chat_streaming_with_json_object_runs_constrained_stream() {
    let (app, _fixture) = q4k_app();
    let resp = send_chat(
        &app,
        serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 4,
            "stream": true,
            "response_format": {"type": "json_object"},
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn chat_streaming_with_tools_ends_in_tool_calls_or_parse_error() {
    // The synthetic vocab cannot spell the tool-call JSON scaffolding,
    // so this drives the `tools_active` streaming arm into either the
    // tool_calls chunk (if constrained decode produced parseable JSON)
    // or the parse-error chunk — both are valid exercised outcomes;
    // rendering may also 400 when the tool template tokenises to empty.
    let (app, _fixture) = q4k_app();
    let resp = send_chat(
        &app,
        serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 4,
            "stream": true,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }],
        }),
    )
    .await;
    let status = resp.status();
    if status != StatusCode::OK {
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "unexpected status {status}"
        );
        return;
    }
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("[DONE]"), "{body}");
    assert!(
        body.contains("tool_calls") || body.contains("tool_call output failed to parse"),
        "tools_active stream must end in a tool_calls chunk or a parse error:\n{body}"
    );
}

// ── tool template + multi-model ──────────────────────────────────────────────

#[tokio::test]
async fn chat_with_tools_renders_tool_template() {
    let (status, body) = post_chat_drained(serde_json::json!({
        "model": "synthetic",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 2,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
            }
        }],
    }))
    .await;
    // Accept 400 too: the synthetic vocab (12 entries) cannot represent
    // the tool-template scaffolding tokens, so the rendered prompt can
    // tokenise to empty and trip the empty-prompt guard. Exercising the
    // tool-template render path is the coverage goal here — generation
    // doesn't have to succeed.
    assert!(
        status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
        "unexpected status {status}: {body}"
    );
}

#[tokio::test]
async fn chat_multi_model_dispatches_by_model_field() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::multi_model_router(state);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    br#"{"model":"synthetic","messages":[{"role":"user","content":"x"}]}"#.to_vec(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::NOT_FOUND);
    let _ = axum::body::to_bytes(resp.into_body(), usize::MAX).await;

    let r404 = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    br#"{"model":"missing","messages":[{"role":"user","content":"x"}]}"#.to_vec(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        r404.status() == StatusCode::NOT_FOUND || r404.status() == StatusCode::BAD_REQUEST,
        "expected 404/400 for unknown model; got {:?}",
        r404.status()
    );
}

// ── server-side timeout arms ─────────────────────────────────────────────────

async fn post_chat_with_timeout(
    timeout: std::time::Duration,
    body: serde_json::Value,
) -> (StatusCode, String) {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state_with_timeout(vec![model], timeout);
    let app = larql_server::routes::single_model_router(state);
    let resp = send_chat(&app, body).await;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn chat_nanosecond_timeout_returns_504() {
    let (status, body) = post_chat_with_timeout(
        std::time::Duration::from_nanos(1),
        serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "the capital of France is"}],
            "max_tokens": 4,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert!(body.contains("timeout"), "{body}");
}

#[tokio::test]
async fn chat_zero_timeout_disables_the_deadline() {
    let (status, body) = post_chat_with_timeout(
        std::time::Duration::ZERO,
        serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "the capital of France is"}],
            "max_tokens": 2,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
