//! `X-Session-Id` extraction.

use super::*;
use axum::http::{HeaderMap, HeaderValue};

#[test]
fn extract_session_id_reads_the_header() {
    let mut headers = HeaderMap::new();
    headers.insert(HEADER_SESSION_ID, HeaderValue::from_static("abc-123"));
    assert_eq!(extract_session_id(&headers), Some("abc-123".to_string()));
}

#[test]
fn extract_session_id_absent_is_none() {
    assert_eq!(extract_session_id(&HeaderMap::new()), None);
}

#[test]
fn extract_session_id_non_utf8_is_none() {
    let mut headers = HeaderMap::new();
    headers.insert(
        HEADER_SESSION_ID,
        HeaderValue::from_bytes(&[0xFF, 0xFE]).expect("opaque bytes are a valid header value"),
    );
    assert_eq!(extract_session_id(&headers), None);
}
