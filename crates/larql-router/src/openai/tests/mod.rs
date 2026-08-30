//! Unit tests for the OpenAI grid-surface module (project
//! convention: module tests live in a `tests/` folder).

mod responses_tests;

use super::*;
fn backend(model: &str, url: &str, in_flight: u32) -> OpenAIBackend {
    OpenAIBackend {
        model_id: model.to_string(),
        listen_url: url.to_string(),
        requests_in_flight: in_flight,
    }
}

#[test]
fn select_prefers_least_loaded() {
    let backends = vec![
        backend("m", "http://a", 5),
        backend("m", "http://b", 1),
        backend("m", "http://c", 3),
    ];
    assert_eq!(
        select_backend(&backends, None).unwrap().listen_url,
        "http://b"
    );
}

#[test]
fn select_restricts_to_requested_model() {
    let backends = vec![backend("m1", "http://a", 0), backend("m2", "http://b", 9)];
    assert_eq!(
        select_backend(&backends, Some("m2")).unwrap().listen_url,
        "http://b"
    );
}

#[test]
fn select_unknown_model_errors_with_the_model_name() {
    let backends = vec![backend("m1", "http://a", 0)];
    let err = select_backend(&backends, Some("missing")).unwrap_err();
    assert_eq!(err, SelectError::UnknownModel("missing".to_string()));
    assert_eq!(err.into_response().status(), StatusCode::NOT_FOUND);
}

#[test]
fn select_empty_grid_errors_with_no_capable_server() {
    let err = select_backend(&[], Some("m")).unwrap_err();
    assert_eq!(err, SelectError::NoCapableServer);
    assert_eq!(
        err.into_response().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[test]
fn model_of_reads_the_model_field() {
    let body = Bytes::from(r#"{"model":"gemma","messages":[]}"#);
    assert_eq!(model_of(&body).as_deref(), Some("gemma"));
}

#[test]
fn model_of_absent_or_invalid_is_none() {
    assert_eq!(model_of(&Bytes::from(r#"{"messages":[]}"#)), None);
    assert_eq!(model_of(&Bytes::from("not json")), None);
    assert_eq!(model_of(&Bytes::from(r#"{"model":7}"#)), None);
}
