//! Coverage push for `routes/expert/single.rs`,
//! `routes/expert/cpu.rs`, `routes/expert/layer_batch.rs`,
//! `routes/expert/multi_layer_batch.rs`, and
//! `routes/expert/batch_legacy.rs` — the dispatcher branches that fire
//! before any MoE-specific compute. Specifically the rejection paths:
//!
//!   - "model is not a hybrid MoE — no expert endpoints available"
//!     (every dense model takes this path immediately)
//!   - residual-length mismatch on the request body
//!   - 404 / 400 from the model-id resolver
//!
//! The deeper MoE compute paths need a synthetic hybrid-MoE vindex
//! (router_proj + N experts + arch overriding `is_hybrid_moe()`).
//! That fixture is the same gap that keeps `walk_ffn/core.rs`
//! excluded; covered there once it lands.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

async fn post_expert(
    layer: usize,
    expert_id: usize,
    body: serde_json::Value,
) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/expert/{layer}/{expert_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    resp
}

#[tokio::test]
async fn single_expert_dense_model_returns_400_not_moe() {
    // Synthetic uses Llama arch; `is_hybrid_moe()` is false →
    // `run_expert` rejects with "model is not a hybrid MoE".
    let resp = post_expert(
        0,
        0,
        serde_json::json!({
            "residual": vec![0.0_f32; 8],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(
        body.contains("hybrid MoE") || body.contains("expert"),
        "expected MoE-specific error message; got {body}"
    );
}

#[tokio::test]
async fn single_expert_invalid_json_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/expert/0/0")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

async fn post_expert_batch(layer: usize, body: serde_json::Value) -> axum::http::Response<Body> {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/expert/{layer}/batch"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    resp
}

#[tokio::test]
async fn expert_batch_dense_model_returns_400() {
    // Same is_hybrid_moe() rejection on the batch endpoint.
    let resp = post_expert_batch(
        0,
        serde_json::json!({
            "residuals": [vec![0.0_f32; 8]],
            "expert_ids": [0],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn expert_layer_batch_dense_model_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/expert/0/layer-batch")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "residual": vec![0.0_f32; 8],
                        "expert_ids": [0],
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    assert!(
        resp.status() == StatusCode::BAD_REQUEST || resp.status() == StatusCode::NOT_FOUND,
        "dense model should reject expert/layer-batch; got {:?}",
        resp.status()
    );
}

#[tokio::test]
async fn expert_multi_layer_batch_dense_model_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/expert/batch")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "entries": [
                            {"layer": 0, "expert_id": 0, "residual": vec![0.0_f32; 8]}
                        ]
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    drop(_fixture);
    assert!(
        resp.status() == StatusCode::BAD_REQUEST || resp.status() == StatusCode::NOT_FOUND,
        "dense model should reject /v1/expert/batch; got {:?}",
        resp.status()
    );
}

// ══════════════════════════════════════════════════════════════
// Phase A expert-serving hardening (dec-readiness review):
// unresolvable experts must be a loud 400, never a 200 with a
// silently partial weighted sum; wrong-shaped tasks must be a 400
// before any kernel runs.
// ══════════════════════════════════════════════════════════════

use larql_inference::ffn::moe_remote::{
    encode_layer_batch_request, encode_multi_layer_request, encode_multi_layer_request_q8k,
    MultiLayerTask, MultiLayerTaskQ8K, LAYER_BATCH_CONTENT_TYPE, MULTI_LAYER_BATCH_CONTENT_TYPE,
    MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
};

async fn post_binary(
    state: std::sync::Arc<larql_server::state::AppState>,
    uri: &str,
    content_type: &str,
    body: Vec<u8>,
) -> axum::http::Response<Body> {
    let app = larql_server::routes::single_model_router(state);
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn body_string(resp: axum::http::Response<Body>) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The synthetic fixture is a dense Llama — no expert weights are resolvable
/// at any (layer, expert_id). Pre-fix, `/v1/experts/layer-batch` returned
/// 200 with an all-zero "weighted sum" (silently partial); now the run-count
/// check turns it into a 400 naming the layer and the missing count.
#[tokio::test]
async fn layer_batch_unresolvable_expert_returns_400_not_partial_sum() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let hidden = model.config.hidden_size;
    let state = common::state(vec![model]);
    let body = encode_layer_batch_request(0, &vec![0.5_f32; hidden], &[3], &[1.0]);
    let resp = post_binary(
        state,
        "/v1/experts/layer-batch",
        LAYER_BATCH_CONTENT_TYPE,
        body,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let text = body_string(resp).await;
    assert!(
        text.contains("could not be resolved") && text.contains("layer 0"),
        "expected loud unresolvable-expert error; got {text}"
    );
}

/// Zero-weight pairs are a legitimate skip (filtered before byte
/// resolution) — they must NOT count as unresolvable, so an all-zero-weight
/// request still succeeds even though no expert bytes exist.
#[tokio::test]
async fn layer_batch_zero_weight_experts_still_200() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let hidden = model.config.hidden_size;
    let state = common::state(vec![model]);
    let body = encode_layer_batch_request(0, &vec![0.5_f32; hidden], &[3, 7], &[0.0, 0.0]);
    let resp = post_binary(
        state,
        "/v1/experts/layer-batch",
        LAYER_BATCH_CONTENT_TYPE,
        body,
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "zero-weight experts are a legitimate skip, not an error: {}",
        body_string(resp).await
    );
}

/// Drain/heartbeat visibility (ROADMAP hardening item 13): a layer-batch
/// request must register in `requests_in_flight` while running and return
/// the counter to 0 when the handler completes, and bump `requests_total`.
#[tokio::test]
async fn layer_batch_requests_in_flight_returns_to_zero() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let hidden = model.config.hidden_size;
    let rif = model.requests_in_flight.clone();
    let total = model.requests_total.clone();
    let state = common::state(vec![model]);
    let body = encode_layer_batch_request(0, &vec![0.5_f32; hidden], &[3], &[0.0]);
    let resp = post_binary(
        state,
        "/v1/experts/layer-batch",
        LAYER_BATCH_CONTENT_TYPE,
        body,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    use std::sync::atomic::Ordering;
    assert_eq!(rif.load(Ordering::SeqCst), 0, "in-flight must drain to 0");
    assert_eq!(total.load(Ordering::SeqCst), 1, "requests_total must bump");
}

/// FIX 2: every multi-layer f32 task's residual must match the model's
/// hidden_size — a wrong-shaped activation is a 400 before any kernel runs.
#[tokio::test]
async fn multi_layer_batch_wrong_hidden_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let tasks = vec![MultiLayerTask {
        layer: 0,
        residual: vec![0.5_f32; 300], // model hidden_size is 8
        expert_ids: vec![0],
        weights: vec![1.0],
    }];
    let resp = post_binary(
        state,
        "/v1/experts/multi-layer-batch",
        MULTI_LAYER_BATCH_CONTENT_TYPE,
        encode_multi_layer_request(&tasks),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let text = body_string(resp).await;
    assert!(
        text.contains("hidden_size"),
        "expected hidden_size mismatch error; got {text}"
    );
}

/// Same loud-error contract on the multi-layer endpoint: an unresolvable
/// expert in any task is a 400 naming the layer, never a partial sum.
#[tokio::test]
async fn multi_layer_batch_unresolvable_expert_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let hidden = model.config.hidden_size;
    let state = common::state(vec![model]);
    let tasks = vec![MultiLayerTask {
        layer: 1,
        residual: vec![0.5_f32; hidden],
        expert_ids: vec![9],
        weights: vec![1.0],
    }];
    let resp = post_binary(
        state,
        "/v1/experts/multi-layer-batch",
        MULTI_LAYER_BATCH_CONTENT_TYPE,
        encode_multi_layer_request(&tasks),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let text = body_string(resp).await;
    assert!(
        text.contains("could not be resolved") && text.contains("layer 1"),
        "expected loud unresolvable-expert error; got {text}"
    );
}

/// FIX 2 (wire level): a q8k task whose `hidden` is not a multiple of 256
/// desyncs every subsequent field offset — the decoder must reject it and
/// the handler must answer 400, not decode garbage.
#[tokio::test]
async fn multi_layer_batch_q8k_misaligned_hidden_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    // Hand-build the frame: encode_multi_layer_request_q8k debug-asserts on
    // consistent shapes, so write the misaligned hidden=300 header directly.
    let mut body = Vec::new();
    body.extend_from_slice(&1u32.to_le_bytes()); // num_tasks
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&300u32.to_le_bytes()); // hidden — NOT ×256
    body.extend_from_slice(&0u32.to_le_bytes()); // num_experts
    body.extend_from_slice(&vec![0u8; 300 + 4 + 16]); // qs + d + sums
    let resp = post_binary(
        state,
        "/v1/experts/multi-layer-batch-q8k",
        MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
        body,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// FIX 2 (handler level): a well-formed q8k task (hidden = 256) whose hidden
/// doesn't match the model's hidden_size is a 400 before any kernel runs.
#[tokio::test]
async fn multi_layer_batch_q8k_wrong_hidden_returns_400() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let tasks = vec![MultiLayerTaskQ8K {
        layer: 0,
        hidden: 256, // decodes fine; model hidden_size is 8
        qs: vec![0i8; 256],
        d: vec![1.0f32],
        sums: vec![0i16; 8],
        expert_ids: vec![0],
        weights: vec![1.0],
    }];
    let resp = post_binary(
        state,
        "/v1/experts/multi-layer-batch-q8k",
        MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
        encode_multi_layer_request_q8k(&tasks),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let text = body_string(resp).await;
    assert!(
        text.contains("hidden_size"),
        "expected hidden_size mismatch error; got {text}"
    );
}

// ══════════════════════════════════════════════════════════════
// Opt-in serve_us timing trailer (DEC-1A two-scoreboard schema):
// with `x-larql-timing: 1` the multi-layer response gains an 8-byte
// magic+f32 trailer AFTER the unchanged payload; without the header
// the response is byte-identical to the pre-extension wire.
// ══════════════════════════════════════════════════════════════

use larql_inference::ffn::moe_remote::decode_multi_layer_response;
use larql_inference::ffn::remote::{
    split_timing_trailer, TIMING_HEADER, TIMING_HEADER_VALUE, TIMING_TRAILER_LEN,
};

async fn post_binary_with_headers(
    state: std::sync::Arc<larql_server::state::AppState>,
    uri: &str,
    content_type: &str,
    body: Vec<u8>,
    timing: bool,
) -> axum::http::Response<Body> {
    let app = larql_server::routes::single_model_router(state);
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, content_type);
    if timing {
        builder = builder.header(TIMING_HEADER, TIMING_HEADER_VALUE);
    }
    app.oneshot(builder.body(Body::from(body)).unwrap())
        .await
        .unwrap()
}

/// A zero-weight task is the one multi-layer request the dense synthetic
/// fixture serves with 200 (legitimate skip, no expert bytes touched) —
/// deterministic all-zero h2, so the two responses below are comparable
/// byte-for-byte.
fn zero_weight_task(hidden: usize) -> Vec<u8> {
    encode_multi_layer_request(&[MultiLayerTask {
        layer: 0,
        residual: vec![0.5_f32; hidden],
        expert_ids: vec![3],
        weights: vec![0.0],
    }])
}

#[tokio::test]
async fn multi_layer_batch_timing_header_appends_trailer_after_identical_payload() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let hidden = model.config.hidden_size;
    let state = common::state(vec![model]);

    // Without the header: pre-extension bytes.
    let resp = post_binary_with_headers(
        state.clone(),
        "/v1/experts/multi-layer-batch",
        MULTI_LAYER_BATCH_CONTENT_TYPE,
        zero_weight_task(hidden),
        false,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let plain = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let (payload, serve_us) = split_timing_trailer(&plain);
    assert_eq!(payload.len(), plain.len(), "no trailer without the header");
    assert_eq!(serve_us, None);

    // With the header: same payload + 8-byte trailer.
    let resp = post_binary_with_headers(
        state,
        "/v1/experts/multi-layer-batch",
        MULTI_LAYER_BATCH_CONTENT_TYPE,
        zero_weight_task(hidden),
        true,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let extended = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(extended.len(), plain.len() + TIMING_TRAILER_LEN);
    let (payload, serve_us) = split_timing_trailer(&extended);
    assert_eq!(
        payload,
        &plain[..],
        "payload before the trailer must be byte-identical to the plain response"
    );
    let us = serve_us.expect("trailer must carry serve_us when requested");
    assert!(us >= 0.0 && us.is_finite());

    // The production decoder parses the stripped payload (new client) AND
    // tolerates the un-stripped extended body (non-timing-aware decoder).
    let results = decode_multi_layer_response(payload).expect("stripped payload decodes");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].layer, 0);
    assert_eq!(results[0].h2.len(), hidden);
    let tolerant = decode_multi_layer_response(&extended).expect("trailer is ignored");
    assert_eq!(tolerant.len(), 1);
    assert_eq!(tolerant[0].h2, results[0].h2);
}

#[tokio::test]
async fn multi_layer_batch_ignores_non_optin_timing_header_values() {
    // `x-larql-timing: 0` (or anything but "1") must NOT append a trailer.
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let hidden = model.config.hidden_size;
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/experts/multi-layer-batch")
                .header(header::CONTENT_TYPE, MULTI_LAYER_BATCH_CONTENT_TYPE)
                .header(TIMING_HEADER, "0")
                .body(Body::from(zero_weight_task(hidden)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let (_, serve_us) = split_timing_trailer(&body);
    assert_eq!(serve_us, None);
}

#[tokio::test]
async fn multi_layer_batch_q8k_timing_header_does_not_change_error_paths() {
    // The q8k multi-layer arm can't reach 200 on the dense synthetic
    // (hidden 8 ≠ ×256), but the header must not perturb its rejection
    // path — same 400, no trailer on the JSON error body.
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let tasks = vec![MultiLayerTaskQ8K {
        layer: 0,
        hidden: 256,
        qs: vec![0i8; 256],
        d: vec![1.0f32],
        sums: vec![0i16; 8],
        expert_ids: vec![0],
        weights: vec![1.0],
    }];
    let resp = post_binary_with_headers(
        state,
        "/v1/experts/multi-layer-batch-q8k",
        MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
        encode_multi_layer_request_q8k(&tasks),
        true,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let (_, serve_us) = split_timing_trailer(&body);
    assert_eq!(serve_us, None, "error bodies never carry the trailer");
}
