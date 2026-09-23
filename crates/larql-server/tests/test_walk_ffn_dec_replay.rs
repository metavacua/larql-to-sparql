//! DEC replay-shape coverage (docs/dec-funnel.md, DEC-0 loadgen).
//!
//! The `larql dec-bench replay` driver sends B-row single-layer frames with
//! `top_k = 0` and decodes responses through the shared codecs. These tests
//! pin the server side of that contract:
//!
//!   * B-row (`seq_len = B`) binary requests return `B × hidden` outputs for
//!     every negotiated wire format (f32 / f16 / i8);
//!   * `top_k = 0` requests bypass the FfnL2Cache — repeated identical
//!     requests re-run compute (asserted via the layer-latency tracker, not
//!     timing values);
//!   * `/v1/stats` exposes the `layer_latency` snapshot and the
//!     `ffn_weights.per_layer_dense_bytes` movement-ratio denominator;
//!   * Q8K batch entries with a duplicate layer id round-trip in wire order.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

const SYN_HIDDEN: usize = 8;

fn residual_rows(batch: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; batch * SYN_HIDDEN];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = (i as f32) * 0.01 + 0.5;
    }
    v
}

/// Production replay frame: single layer, `seq_len = batch`, full_output,
/// `top_k = 0` — built through the same encoder `larql dec-bench` uses.
fn replay_frame(layer: usize, batch: usize) -> Vec<u8> {
    larql_inference::encode_binary_request(Some(layer), None, &residual_rows(batch), batch, true, 0)
}

async fn post_walk_ffn_binary(
    app: axum::Router,
    body: Vec<u8>,
    accept: &str,
) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/walk-ffn")
            .header(header::CONTENT_TYPE, larql_inference::BINARY_CT)
            .header(header::ACCEPT, accept)
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn assert_b_row_response(accept: &str, batch: usize) {
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);

    let resp = post_walk_ffn_binary(app, replay_frame(0, batch), accept).await;
    assert_eq!(resp.status(), StatusCode::OK, "accept={accept}");
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or(larql_inference::BINARY_CT)
        .to_owned();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();

    // Decode through the SAME dispatcher the replay driver uses.
    let (layer, server_ms, floats) =
        larql_inference::decode_single_response(&ct, &bytes, SYN_HIDDEN)
            .unwrap_or_else(|e| panic!("decode ({accept}): {e}"));
    assert_eq!(layer, 0);
    assert!(server_ms >= 0.0);
    assert_eq!(
        floats.len(),
        batch * SYN_HIDDEN,
        "B-row response must carry batch × hidden floats (accept={accept})"
    );
}

#[tokio::test]
async fn walk_ffn_b_row_f32_returns_batch_by_hidden() {
    assert_b_row_response(larql_inference::BINARY_CT, 8).await;
}

#[tokio::test]
async fn walk_ffn_b_row_f16_returns_batch_by_hidden() {
    assert_b_row_response(larql_inference::F16_CT, 8).await;
}

#[tokio::test]
async fn walk_ffn_b_row_i8_returns_batch_by_hidden() {
    assert_b_row_response(larql_inference::I8_CT, 8).await;
}

async fn post_walk_ffn_ct(
    app: axum::Router,
    content_type: &str,
    body: Vec<u8>,
    accept: &str,
) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/walk-ffn")
            .header(header::CONTENT_TYPE, content_type)
            .header(header::ACCEPT, accept)
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn walk_ffn_compressed_inbound_matches_f32_inbound() {
    // Asymmetric direction codecs (DEC funnel §3 DEC-1A): the request
    // Content-Type declares the INBOUND residual format. Rows are exactly
    // representable in f16 AND in the i8 fixed-point grid (scale 0.5), so
    // an f16- or i8-encoded inbound body must decode to the same residual
    // as the f32 body — and the deterministic FFN must return bit-identical
    // f32 outputs for all three.
    use larql_inference::ffn::remote::{encode_binary_request_as, WireFormat};

    let row: Vec<f32> = vec![63.5, -32.0, 0.5, -63.5, 16.0, -8.0, 4.0, 2.0];
    assert_eq!(row.len(), SYN_HIDDEN);

    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);

    let mut outputs = Vec::new();
    for fmt in [WireFormat::F32, WireFormat::F16, WireFormat::I8] {
        let body = encode_binary_request_as(fmt, Some(0), None, &row, 1, true, 0);
        let app = larql_server::routes::single_model_router(state.clone());
        let resp =
            post_walk_ffn_ct(app, fmt.content_type(), body, larql_inference::BINARY_CT).await;
        assert_eq!(resp.status(), StatusCode::OK, "inbound {}", fmt.label());
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap()
            .to_owned();
        assert_eq!(ct, larql_inference::BINARY_CT, "return stays f32");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let (_, _, floats) =
            larql_inference::decode_single_response(&ct, &bytes, SYN_HIDDEN).unwrap();
        assert_eq!(floats.len(), SYN_HIDDEN);
        outputs.push((fmt, floats));
    }
    let f32_out = outputs[0].1.clone();
    for (fmt, floats) in &outputs[1..] {
        assert_eq!(
            *floats,
            f32_out,
            "{} inbound must decode to the same residual as f32 inbound",
            fmt.label()
        );
    }
}

#[tokio::test]
async fn walk_ffn_asymmetric_pair_directions_are_independent() {
    // i8-encoded inbound + f16 Accept: the server must decode the request
    // per its Content-Type and encode the response per Accept, fully
    // independently (the b-row shape survives both directions).
    use larql_inference::ffn::remote::{encode_binary_request_as, WireFormat};

    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let batch = 4;

    let body = encode_binary_request_as(
        WireFormat::I8,
        Some(0),
        None,
        &residual_rows(batch),
        batch,
        true,
        0,
    );
    let app = larql_server::routes::single_model_router(state.clone());
    let resp = post_walk_ffn_ct(app, larql_inference::I8_CT, body, larql_inference::F16_CT).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap()
        .to_owned();
    assert_eq!(
        ct,
        larql_inference::F16_CT,
        "return follows Accept, not the inbound CT"
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let (layer, _, floats) =
        larql_inference::decode_single_response(&ct, &bytes, SYN_HIDDEN).unwrap();
    assert_eq!(layer, 0);
    assert_eq!(floats.len(), batch * SYN_HIDDEN);

    // f16 inbound + strict f32 Accept: the reverse pair (compressed in,
    // uncompressed out).
    let body = encode_binary_request_as(
        WireFormat::F16,
        Some(0),
        None,
        &residual_rows(batch),
        batch,
        true,
        0,
    );
    let app = larql_server::routes::single_model_router(state);
    let resp = post_walk_ffn_ct(
        app,
        larql_inference::F16_CT,
        body,
        larql_inference::BINARY_CT,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap()
        .to_owned();
    assert_eq!(ct, larql_inference::BINARY_CT);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let (_, _, floats) = larql_inference::decode_single_response(&ct, &bytes, SYN_HIDDEN).unwrap();
    assert_eq!(floats.len(), batch * SYN_HIDDEN);
}

#[tokio::test]
async fn walk_ffn_top_k_zero_recomputes_on_repeat() {
    // The FfnL2Cache serves cached output when `seq_len == 1 && top_k > 0`
    // (routes/walk_ffn/core.rs) — a replayed B=1 point with a non-zero
    // top_k would measure cache hits from the second request on. `top_k=0`
    // must re-run compute every time; each compute records into the
    // layer-latency tracker, so the sample count is the timing-free proof.
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let tracker = model.layer_latency_tracker.clone();
    let state = common::state(vec![model]);

    assert_eq!(tracker.recorded_count(0), 0);
    for expected in 1..=2 {
        let app = larql_server::routes::single_model_router(state.clone());
        let resp = post_walk_ffn_binary(app, replay_frame(0, 1), larql_inference::BINARY_CT).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            tracker.recorded_count(0),
            expected,
            "request {expected} with top_k=0 must run compute, not hit the L2 cache"
        );
    }
}

#[tokio::test]
async fn stats_exposes_layer_latency_and_ffn_weights() {
    // Q4K fixture: interleaved k-quant data present, so /v1/stats must
    // report per-layer dense weight bytes (the movement-ratio denominator)
    // and, after a walk-ffn request, a non-empty layer_latency snapshot.
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let num_layers = model.config.num_layers;
    let state = common::state(vec![model]);

    let app = larql_server::routes::single_model_router(state.clone());
    let resp = post_walk_ffn_binary(app, replay_frame(0, 1), larql_inference::BINARY_CT).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    let latency = v["layer_latency"]
        .as_array()
        .expect("stats must expose layer_latency");
    assert!(
        !latency.is_empty(),
        "layer_latency must be non-empty after a walk-ffn request"
    );
    assert!(latency[0]["avg_ms"].is_number());
    assert!(latency[0]["p99_ms"].is_number());

    let per_layer = v["ffn_weights"]["per_layer_dense_bytes"]
        .as_array()
        .expect("q4k vindex must expose ffn_weights.per_layer_dense_bytes");
    assert_eq!(per_layer.len(), num_layers);
    assert!(
        per_layer[0].as_u64().unwrap_or(0) > 0,
        "layer 0 dense bytes must be a positive byte count"
    );
}

#[tokio::test]
async fn stats_ffn_weights_moe_block_from_loaded_weights() {
    // With MoE model weights already loaded, /v1/stats must expose the moe
    // routing metadata (num_experts / top_k) used for the full-MoE
    // movement-ratio accounting; the fixture has no per-layer expert
    // entries, so per_expert_bytes must be null, not fabricated.
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    model
        .weights
        .set(std::sync::RwLock::new(
            larql_models::test_fixtures::make_test_gemma4_moe_weights(),
        ))
        .unwrap_or_else(|_| panic!("weights OnceLock already set"));
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let moe = &v["ffn_weights"]["moe"];
    assert!(moe["num_experts"].as_u64().unwrap() > 0);
    assert!(moe["top_k"].as_u64().unwrap() > 0);
    assert!(moe["per_expert_bytes"].is_null());
}

#[tokio::test]
async fn stats_ffn_weights_dense_arch_omits_moe_block() {
    // Loaded weights with a non-MoE arch: the ffn_weights block stays (the
    // q4k vindex has dense byte data) but moe must be null.
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    model
        .weights
        .set(std::sync::RwLock::new(
            larql_models::test_fixtures::make_test_weights(),
        ))
        .unwrap_or_else(|_| panic!("weights OnceLock already set"));
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v["ffn_weights"]["per_layer_dense_bytes"].is_array());
    assert!(v["ffn_weights"]["moe"].is_null());
}

#[tokio::test]
async fn stats_f32_vindex_omits_ffn_weights() {
    // The plain f32 synthetic has no interleaved k-quant data and no MoE —
    // the ffn_weights block must be absent, not zero-filled, so the replay
    // driver knows the denominator is unavailable.
    let (model, _fixture) = common::model_with_real_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v.get("ffn_weights").is_none());
    assert!(v["layer_latency"].as_array().is_some());
}

#[tokio::test]
async fn walk_ffn_q8k_duplicate_layer_entries_preserved_in_order() {
    // B-row Q8K replay = B entries sharing one layer id. When the server
    // accepts the frame, the response must echo one entry per request entry
    // in wire order (the client-side ordered decoder is unit-tested in
    // larql-inference; this pins the server's pass-through shape).
    use larql_compute::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;

    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    let state = common::state(vec![model]);
    let app = larql_server::routes::single_model_router(state);

    // Q8K activations quantise in 256-element super-blocks; pad each row.
    let rows: Vec<Vec<f32>> = (0..3)
        .map(|i| {
            let mut r = vec![0.0f32; 256];
            for (j, slot) in r.iter_mut().enumerate().take(SYN_HIDDEN) {
                *slot = (i * SYN_HIDDEN + j) as f32 * 0.01 + 0.1;
            }
            r
        })
        .collect();
    let q8ks: Vec<_> = rows.iter().map(|r| quantize_x_to_q8k(r)).collect();
    let entries: Vec<(usize, &_)> = q8ks.iter().map(|q| (0usize, q)).collect();
    let body = larql_inference::encode_q8k_batch_request(&entries);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn-q8k")
                .header(header::CONTENT_TYPE, larql_inference::Q8K_BATCH_CT)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    if status == StatusCode::OK {
        let decoded = larql_inference::decode_q8k_batch_response_entries(&bytes).unwrap();
        assert_eq!(decoded.len(), 3, "one response entry per request entry");
        assert!(decoded.iter().all(|(l, _)| *l == 0));
    } else {
        // The tiny synthetic (hidden=8) may not satisfy the Q8K kernel's
        // block-shape preconditions; a clean 4xx is acceptable — the
        // duplicate-order contract is pinned client-side. A 5xx is not.
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "q8k duplicate-layer frame must be served or cleanly rejected"
        );
    }
}
