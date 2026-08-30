//! HTTP-level coverage for `POST`/`DELETE /v1/runtime/model`
//! (`routes/runtime_lifecycle.rs`).
//!
//! The state-machine internals (`decide_load`/`decide_unload`,
//! the drain-timeout fail-closed path) are unit-tested in
//! `state/lifecycle.rs` and inline in `routes/runtime_lifecycle.rs`
//! itself, where the private `load_model`/`unload_model` functions
//! and a tiny injected drain timeout are reachable. What's left for
//! here is what only a real router + a real on-disk container can
//! prove: the two HTTP handlers are actually wired up, a genuine
//! `load_artifact` round trip binds a model a client can then see via
//! `GET /v1/runtime`, and a boot-time multi-model topology refuses
//! both verbs end-to-end.

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

/// `POST /v1/runtime/model` binds the fixture on disk, `GET
/// /v1/runtime` sees it, `DELETE /v1/runtime/model` unbinds it again —
/// and a second `DELETE` is idempotent.
#[tokio::test]
async fn post_then_delete_round_trips_a_real_model() {
    let (_model, fixture) = common::model_with_real_weights("unused-id");
    let state = common::state(vec![]);
    let app = common::single_model_router(state);
    let path = fixture.dir.to_string_lossy().to_string();

    let post_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/runtime/model")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({ "path": path })).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(post_resp.status(), StatusCode::OK);
    let posted = common::body_json(post_resp.into_body()).await;
    assert!(
        !posted["model"].is_null(),
        "a successful load must report the bound model: {posted}"
    );
    assert_eq!(posted["model"]["format"], "vindex2");

    let get_resp = common::get(app.clone(), "/v1/runtime").await;
    assert_eq!(get_resp.status(), StatusCode::OK);
    let snapshot = common::body_json(get_resp.into_body()).await;
    assert_eq!(
        snapshot["model"]["id"], posted["model"]["id"],
        "the bound model must show up on the next GET"
    );

    let delete_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/runtime/model")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_resp.status(), StatusCode::OK);
    let deleted = common::body_json(delete_resp.into_body()).await;
    assert!(
        deleted["model"].is_null(),
        "a successful unload must report no model bound: {deleted}"
    );

    // Idempotent: unloading again while already idle is still success.
    let second_delete = common::delete(app, "/v1/runtime/model").await;
    assert_eq!(second_delete.status(), StatusCode::OK);

    drop(fixture);
}

/// Loading the exact same path twice is idempotent success, not a
/// conflict — and loading a *different* path while one is bound is
/// refused without touching the original binding.
#[tokio::test]
async fn post_is_idempotent_for_the_same_path_and_refuses_a_different_one() {
    let (_model, fixture) = common::model_with_real_weights("unused-id");
    let state = common::state(vec![]);
    let app = common::single_model_router(state);
    let path = fixture.dir.to_string_lossy().to_string();

    let first = common::post_json(
        app.clone(),
        "/v1/runtime/model",
        serde_json::json!({ "path": path }),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    let first_body = common::body_json(first.into_body()).await;
    let bound_id = first_body["model"]["id"].clone();

    let second = common::post_json(
        app.clone(),
        "/v1/runtime/model",
        serde_json::json!({ "path": path }),
    )
    .await;
    assert_eq!(
        second.status(),
        StatusCode::OK,
        "reloading the identical path must be an idempotent success"
    );
    let second_body = common::body_json(second.into_body()).await;
    assert_eq!(second_body["model"]["id"], bound_id);

    let different = common::post_json(
        app.clone(),
        "/v1/runtime/model",
        serde_json::json!({ "path": "/some/other/path" }),
    )
    .await;
    assert_eq!(
        different.status(),
        StatusCode::CONFLICT,
        "this endpoint does not support atomic A->B replacement"
    );

    // The original binding must be untouched by the refused request.
    let get_resp = common::get(app, "/v1/runtime").await;
    let snapshot = common::body_json(get_resp.into_body()).await;
    assert_eq!(snapshot["model"]["id"], bound_id);

    drop(fixture);
}

/// A path that doesn't produce a loadable model is a 400, and leaves
/// the server idle rather than stuck mid-load.
#[tokio::test]
async fn post_with_an_unloadable_path_is_a_bad_request_and_stays_idle() {
    let state = common::state(vec![]);
    let app = common::single_model_router(state);

    let resp = common::post_json(
        app.clone(),
        "/v1/runtime/model",
        serde_json::json!({ "path": "/definitely/does/not/exist-http-lifecycle-test" }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    // Idle again, not stuck — a subsequent load attempt is still Proceed,
    // not "a load is already in progress".
    let retry = common::post_json(
        app,
        "/v1/runtime/model",
        serde_json::json!({ "path": "/still/not/real" }),
    )
    .await;
    assert_eq!(retry.status(), StatusCode::BAD_REQUEST);
}

/// A boot-time multi-model server refuses both verbs outright — the
/// 0↔1 invariant (`docs/runtime-lifecycle-design.md` §7), proven
/// against the actual `multi_model_router`.
#[tokio::test]
async fn multi_model_topology_refuses_both_verbs() {
    let model_a = common::model("model-a");
    let model_b = common::model("model-b");
    let state = common::state(vec![model_a, model_b]);
    assert_eq!(
        state.router_topology,
        larql_server::state::RouterTopology::MultiModel
    );

    // `RUNTIME_MODEL` isn't even routed on `multi_model_router` — this
    // proves the *scope* decision (single-model topology only), not
    // just the invariant check inside the handler.
    let app = common::multi_model_router(state);
    let resp = common::post_json(
        app.clone(),
        "/v1/runtime/model",
        serde_json::json!({ "path": "/a" }),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "multi_model_router must not expose a lifecycle mutation route at all"
    );

    let delete_resp = common::delete(app, "/v1/runtime/model").await;
    assert_eq!(delete_resp.status(), StatusCode::NOT_FOUND);
}

/// Belt-and-suspenders on the invariant itself, in case a future
/// change ever adds `RUNTIME_MODEL` to `multi_model_router`: even
/// reached directly, `validate_lifecycle_mutation` must still refuse.
#[tokio::test]
async fn multimodel_state_refuses_the_mutation_even_if_routed() {
    let model_a = common::model("model-a");
    let model_b = common::model("model-b");
    let state = common::state(vec![model_a, model_b]);
    assert!(state.validate_lifecycle_mutation(1).is_err());
    assert!(state.validate_lifecycle_mutation(0).is_err());
}
