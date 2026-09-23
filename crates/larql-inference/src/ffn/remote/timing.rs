//! Opt-in response timing extension — the wire half of the DEC-1A
//! two-scoreboard schema (docs/dec-funnel.md §3, "Timing decomposition").
//!
//! # Problem
//!
//! Only the dense f32 walk-ffn response embeds a server latency field
//! (`latency_ms` at bytes 8–11 of its fixed header). The q8k walk-ffn
//! response and both multi-layer expert responses carry none — so a
//! client measuring those endpoints cannot split its round trip into
//! `serve_us` (server compute) vs `transmit_us` (network + framing).
//!
//! # Mechanism — header-gated trailer
//!
//! The client signals support with the request header
//! [`TIMING_HEADER`]` : `[`TIMING_HEADER_VALUE`]. A server that
//! understands the extension appends a fixed 8-byte trailer to the
//! response body **only when asked**:
//!
//! ```text
//! Offset (from end)  Size  Field
//! -8                 4     TIMING_TRAILER_MAGIC (u32 LE, b"LQTM")
//! -4                 4     serve_us (f32 LE, ≥ 0, finite)
//! ```
//!
//! # Why this cannot corrupt a mixed-version deployment
//!
//! * **Old client + old server** — no header sent, no trailer appended:
//!   response bytes are byte-identical to the pre-extension wire (the
//!   trailer is applied by the handler *after* the unchanged response
//!   encoders; the existing byte-identity pins keep passing untouched).
//! * **Old client + new server** — the old client never sends the
//!   header, so the new server never appends: byte-identical again.
//! * **New client + old server** — the old server ignores the unknown
//!   header and sends a plain response; [`split_timing_trailer`] fails
//!   the magic/validity check and returns the whole body with
//!   `serve_us = None`. (A payload whose final 8 bytes coincidentally
//!   look like a valid trailer has probability ≤ 2⁻³², and even then
//!   the stripped payload fails the caller's exact shape checks —
//!   entry counts and `batch × hidden` lengths no longer line up — so
//!   the failure mode is a loud decode error, never silent data
//!   corruption.)
//! * **New client + new server** — trailer present by construction;
//!   the client strips it *before* handing the payload to the normal
//!   response decoder, so every existing length guard sees exactly the
//!   payload it always saw. The affected decoders
//!   (`decode_q8k_batch_response_entries`, `decode_multi_layer_response`)
//!   also tolerate the trailer if fed the un-split body: both read
//!   exactly the declared entry count and ignore trailing bytes
//!   (pinned by tests in their modules).
//!
//! Mixed-version deployments therefore simply don't get timing — the
//! flag is per-request opt-in and degrades to `None`, never to a
//! corrupt decode.

/// Request header a client sends to opt into the response timing trailer.
pub const TIMING_HEADER: &str = "x-larql-timing";

/// The only recognised value of [`TIMING_HEADER`]. Anything else (or an
/// absent header) means "do not append the trailer".
pub const TIMING_HEADER_VALUE: &str = "1";

/// Trailer magic, little-endian `b"LQTM"` (LarQL TiMing).
pub const TIMING_TRAILER_MAGIC: u32 = u32::from_le_bytes(*b"LQTM");

/// Fixed trailer length in bytes: magic (4) + serve_us f32 (4).
pub const TIMING_TRAILER_LEN: usize = 8;

/// Does this request-header value opt into the timing trailer?
pub fn timing_requested(header_value: Option<&str>) -> bool {
    header_value.map(str::trim) == Some(TIMING_HEADER_VALUE)
}

/// Append the fixed timing trailer to an already-encoded response body.
///
/// `serve_us` is sanitised so [`split_timing_trailer`] always accepts a
/// server-produced trailer: non-finite or negative values clamp to 0.
pub fn append_timing_trailer(buf: &mut Vec<u8>, serve_us: f32) {
    let v = if serve_us.is_finite() && serve_us >= 0.0 {
        serve_us
    } else {
        0.0
    };
    buf.extend_from_slice(&TIMING_TRAILER_MAGIC.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Split a response body into `(payload, serve_us)`.
///
/// Returns `(&body[..len-8], Some(serve_us))` when the body ends in a
/// valid trailer (magic matches AND the f32 is finite and ≥ 0 — the
/// only values [`append_timing_trailer`] emits); otherwise
/// `(body, None)` unchanged. Callers that requested timing feed the
/// returned payload to the normal response decoder, whose exact length
/// checks then see the pre-extension frame.
pub fn split_timing_trailer(body: &[u8]) -> (&[u8], Option<f64>) {
    if body.len() < TIMING_TRAILER_LEN {
        return (body, None);
    }
    let split = body.len() - TIMING_TRAILER_LEN;
    let magic = u32::from_le_bytes(body[split..split + 4].try_into().unwrap());
    if magic != TIMING_TRAILER_MAGIC {
        return (body, None);
    }
    let v = f32::from_le_bytes(body[split + 4..].try_into().unwrap());
    if !v.is_finite() || v < 0.0 {
        return (body, None);
    }
    (&body[..split], Some(v as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_consts_pinned() {
        assert_eq!(TIMING_HEADER, "x-larql-timing");
        assert_eq!(TIMING_HEADER_VALUE, "1");
        assert_eq!(TIMING_TRAILER_MAGIC.to_le_bytes(), *b"LQTM");
        assert_eq!(TIMING_TRAILER_LEN, 8);
    }

    #[test]
    fn timing_requested_only_on_exact_value() {
        assert!(timing_requested(Some("1")));
        assert!(timing_requested(Some(" 1 ")), "whitespace tolerated");
        assert!(!timing_requested(Some("0")));
        assert!(!timing_requested(Some("true")));
        assert!(!timing_requested(Some("")));
        assert!(!timing_requested(None));
    }

    #[test]
    fn append_split_round_trip() {
        let mut body = vec![1u8, 2, 3, 4, 5];
        append_timing_trailer(&mut body, 1234.5);
        assert_eq!(body.len(), 5 + TIMING_TRAILER_LEN);
        let (payload, serve_us) = split_timing_trailer(&body);
        assert_eq!(payload, &[1u8, 2, 3, 4, 5]);
        assert!((serve_us.unwrap() - 1234.5).abs() < 1e-6);
    }

    #[test]
    fn append_split_round_trip_empty_payload() {
        let mut body = Vec::new();
        append_timing_trailer(&mut body, 0.0);
        let (payload, serve_us) = split_timing_trailer(&body);
        assert!(payload.is_empty());
        assert_eq!(serve_us, Some(0.0));
    }

    #[test]
    fn append_sanitises_invalid_serve_values() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -3.0] {
            let mut body = vec![9u8];
            append_timing_trailer(&mut body, bad);
            let (payload, serve_us) = split_timing_trailer(&body);
            assert_eq!(payload, &[9u8], "trailer must still split off");
            assert_eq!(serve_us, Some(0.0), "invalid values clamp to 0");
        }
    }

    #[test]
    fn split_returns_none_without_trailer() {
        // Too short.
        let (payload, serve_us) = split_timing_trailer(&[1, 2, 3]);
        assert_eq!(payload, &[1, 2, 3]);
        assert_eq!(serve_us, None);
        // Long enough but wrong magic (all zeros — the common case of a
        // plain response ending in zero floats).
        let body = vec![0u8; 16];
        let (payload, serve_us) = split_timing_trailer(&body);
        assert_eq!(payload.len(), 16);
        assert_eq!(serve_us, None);
    }

    #[test]
    fn split_rejects_magic_with_invalid_value_as_no_trailer() {
        // Right magic but a hand-crafted non-finite / negative value is
        // not something append_timing_trailer ever emits — treat as "no
        // trailer" (tightens the accidental-collision window further).
        for bad in [f32::NAN, -1.0f32] {
            let mut body = vec![7u8; 4];
            body.extend_from_slice(&TIMING_TRAILER_MAGIC.to_le_bytes());
            body.extend_from_slice(&bad.to_le_bytes());
            let (payload, serve_us) = split_timing_trailer(&body);
            assert_eq!(payload.len(), body.len());
            assert_eq!(serve_us, None);
        }
    }

    #[test]
    fn append_split_reappend_is_byte_identical() {
        // Byte-identity pin for the extended frame: split then re-append
        // must reproduce the original bytes exactly.
        let mut body: Vec<u8> = (0..37u8).collect();
        append_timing_trailer(&mut body, 987.25);
        let (payload, serve_us) = split_timing_trailer(&body);
        let mut rebuilt = payload.to_vec();
        append_timing_trailer(&mut rebuilt, serve_us.unwrap() as f32);
        assert_eq!(rebuilt, body);
    }
}
