//! HTTP wire-format helpers shared by routes that accept both binary and
//! JSON request bodies (walk-ffn, embed, expert/batch).
//!
//! The detection uses `contains` rather than `starts_with` so that
//! parameterised types (`application/json; charset=utf-8`,
//! `application/x-larql-ffn; v=2`) match. The binary content types we
//! advertise (`application/x-larql-ffn`, `application/x-larql-expert`)
//! are unique enough that no ambiguity arises.
//!
//! # Wire format negotiation (ADR-0009)
//!
//! Clients advertise their preferred dtype via the `Accept` header:
//!   - `application/x-larql-ffn`      → f32 (4 bytes/value, existing)
//!   - `application/x-larql-ffn-f16`  → f16 (2 bytes/value)
//!   - `application/x-larql-ffn-i8`   → i8 symmetric (1 byte/value + 8-byte header)
//!
//! `preferred_response_ct` selects the best format the server can satisfy.
//! The server checks `LARQL_F16_WIRE_DISABLE` before honouring f16 requests.

use axum::http::header;
use axum::http::HeaderMap;

// Content-type strings are single-sourced in the shared codec
// (`larql_inference::ffn::remote::codec`, ROADMAP hardening item 16);
// the historical `FFN_*` names are kept as re-exports.
/// f32 binary content-type (existing).
pub use larql_inference::ffn::remote::BINARY_CT as FFN_CT;
/// f16 binary content-type (ADR-0009).
pub use larql_inference::ffn::remote::F16_CT as FFN_F16_CT;
/// i8 symmetric binary content-type (ADR-0009).
pub use larql_inference::ffn::remote::I8_CT as FFN_I8_CT;

/// Select the best response content-type given the client's `Accept` header.
///
/// Priority: i8 > f16 > f32, subject to server feature flags.
/// Falls back to f32 when the client sends no `Accept` or prefers f32 only.
pub fn preferred_response_ct(accept: Option<&str>) -> &'static str {
    let Some(accept) = accept else { return FFN_CT };
    let f16_disabled = std::env::var(crate::env_flags::F16_WIRE_DISABLE).is_ok();
    let i8_enabled = std::env::var(crate::env_flags::I8_WIRE).is_ok();
    if i8_enabled && accept.contains(FFN_I8_CT) {
        return FFN_I8_CT;
    }
    if !f16_disabled && accept.contains(FFN_F16_CT) {
        return FFN_F16_CT;
    }
    FFN_CT
}

/// Detect the INBOUND wire format of a walk-ffn request from its
/// `Content-Type` header (asymmetric direction codecs, DEC funnel §3
/// DEC-1A). `None` means "not a binary walk-ffn body" — the JSON path.
///
/// Delegates to the shared codec's ordered matcher: the suffixed f16/i8
/// types are checked before f32 because `application/x-larql-ffn` is a
/// substring of both — a bare `has_content_type(…, FFN_CT)` check would
/// silently misread a compressed request body as f32.
pub fn request_wire_format(
    headers: &HeaderMap,
) -> Option<larql_inference::ffn::remote::WireFormat> {
    let ct = headers.get(header::CONTENT_TYPE)?.to_str().ok()?;
    larql_inference::ffn::remote::WireFormat::from_content_type(ct)
}

/// Returns `true` when the `Content-Type` header on `headers` contains the
/// substring `expected` (e.g. an `application/x-larql-ffn` binary type).
pub fn has_content_type(headers: &HeaderMap, expected: &str) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains(expected))
}

/// Extract the `Accept` header value as a string slice, if present.
pub fn accept_header(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::ACCEPT).and_then(|v| v.to_str().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn hm(ct: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(header::CONTENT_TYPE, HeaderValue::from_str(ct).unwrap());
        h
    }

    fn hm_accept(accept: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(header::ACCEPT, HeaderValue::from_str(accept).unwrap());
        h
    }

    #[test]
    fn matches_exact_type() {
        assert!(has_content_type(
            &hm("application/x-larql-ffn"),
            "application/x-larql-ffn"
        ));
    }

    #[test]
    fn matches_with_parameters() {
        assert!(has_content_type(
            &hm("application/json; charset=utf-8"),
            "application/json"
        ));
    }

    #[test]
    fn does_not_match_other_type() {
        assert!(!has_content_type(
            &hm("application/json"),
            "application/x-larql-ffn"
        ));
    }

    #[test]
    fn missing_header_does_not_match() {
        assert!(!has_content_type(&HeaderMap::new(), "application/json"));
    }

    #[test]
    fn no_accept_header_returns_f32() {
        assert_eq!(preferred_response_ct(None), FFN_CT);
    }

    #[test]
    fn f32_accept_returns_f32() {
        let h = hm_accept("application/x-larql-ffn");
        assert_eq!(preferred_response_ct(accept_header(&h)), FFN_CT);
    }

    #[test]
    fn accept_header_extraction() {
        let h = hm_accept("application/x-larql-ffn-f16");
        assert_eq!(accept_header(&h), Some("application/x-larql-ffn-f16"));
        assert_eq!(accept_header(&HeaderMap::new()), None);
    }

    #[test]
    fn request_wire_format_orders_suffixed_types_before_f32() {
        use larql_inference::ffn::remote::WireFormat;
        // f16/i8 CTs contain the f32 CT as a substring — the detection must
        // return the compressed format, never F32.
        assert_eq!(request_wire_format(&hm(FFN_F16_CT)), Some(WireFormat::F16));
        assert_eq!(request_wire_format(&hm(FFN_I8_CT)), Some(WireFormat::I8));
        assert_eq!(request_wire_format(&hm(FFN_CT)), Some(WireFormat::F32));
        // Parameterised type still matches.
        assert_eq!(
            request_wire_format(&hm("application/x-larql-ffn-i8; v=2")),
            Some(WireFormat::I8)
        );
        // JSON and missing header → None (the JSON path).
        assert_eq!(request_wire_format(&hm("application/json")), None);
        assert_eq!(request_wire_format(&HeaderMap::new()), None);
    }
}
