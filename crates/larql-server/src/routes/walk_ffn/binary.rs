//! Binary + JSON codec shim for the walk-ffn wire protocol.
//!
//! The actual frame codec is single-sourced in
//! `larql_inference::ffn::remote::codec` (ROADMAP hardening item 16 —
//! same discipline as the q8k and MoE wires): encoders, decoders, length
//! guards, and the content-type/marker constants all live there. This
//! module only
//! - re-exports the response encoders the handler dispatches on, and
//! - wraps the shared request decoder into the server's
//!   [`WalkFfnRequest`] / [`ServerError`] types.
//!
//! The `#[cfg(test)]` block below is the server-side regression suite for
//! the shared implementation (truncation, alloc-bomb, and shape guards
//! must keep rejecting through this path).

use crate::error::ServerError;

use super::types::WalkFfnRequest;

pub(crate) use larql_inference::ffn::remote::{
    encode_binary_output, encode_binary_output_f16, encode_binary_output_i8,
    encode_json_full_output,
};

/// Decode a binary-format request body whose residual payload is in
/// `format` (inbound `Content-Type` dispatch — the request and response
/// directions are negotiated independently, DEC funnel §3 DEC-1A).
///
/// Thin wrapper over the shared
/// [`larql_inference::ffn::remote::decode_binary_request_as`]: maps the
/// codec's string error into [`ServerError::BadRequest`] and fills the
/// JSON-only `moe_layer` field (the binary wire has no MoE-layer flag).
pub(crate) fn decode_request(
    format: larql_inference::ffn::remote::WireFormat,
    body: &[u8],
) -> Result<WalkFfnRequest, ServerError> {
    let d = larql_inference::ffn::remote::decode_binary_request_as(format, body)
        .map_err(ServerError::BadRequest)?;
    Ok(WalkFfnRequest {
        layer: d.layer,
        layers: d.layers,
        residual: d.residual,
        seq_len: d.seq_len,
        top_k: d.top_k,
        full_output: d.full_output,
        moe_layer: false,
    })
}

/// f32-only twin of [`decode_request`] — the pre-asymmetric call shape,
/// kept for the regression suite below (production dispatches by
/// Content-Type through [`decode_request`]).
#[cfg(test)]
pub(crate) fn decode_binary_request(body: &[u8]) -> Result<WalkFfnRequest, ServerError> {
    decode_request(larql_inference::ffn::remote::WireFormat::F32, body)
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests — server-side regression suite for the shared codec (decode guards +
// every encoder variant, exercised through the shim exactly as the handler
// uses them)
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::types::{FfnEntry, FfnOutput};
    use super::*;
    use larql_inference::ffn::remote::BATCH_MARKER;

    // ── decode_binary_request ─────────────────────────────────────────────────

    fn make_single_binary(
        layer: u32,
        seq_len: u32,
        full_output: bool,
        top_k: u32,
        residual: &[f32],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&(full_output as u32).to_le_bytes());
        buf.extend_from_slice(&top_k.to_le_bytes());
        for &v in residual {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn make_batch_binary(
        layers: &[u32],
        seq_len: u32,
        full_output: bool,
        top_k: u32,
        residual: &[f32],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(layers.len() as u32).to_le_bytes());
        for &l in layers {
            buf.extend_from_slice(&l.to_le_bytes());
        }
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&(full_output as u32).to_le_bytes());
        buf.extend_from_slice(&top_k.to_le_bytes());
        for &v in residual {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn decode_single_layer_request() {
        let body = make_single_binary(5, 1, true, 8, &[1.0, 2.0, 3.0, 4.0]);
        let req = decode_binary_request(&body).unwrap();
        assert_eq!(req.layer, Some(5));
        assert!(req.layers.is_none());
        assert_eq!(req.seq_len, 1);
        assert_eq!(req.top_k, 8);
        assert!(req.full_output);
        assert!(!req.moe_layer);
        assert_eq!(req.residual, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn decode_batch_request() {
        let body = make_batch_binary(&[0, 1, 2], 1, true, 16, &[1.0; 4]);
        let req = decode_binary_request(&body).unwrap();
        assert!(req.layer.is_none());
        assert_eq!(req.layers, Some(vec![0, 1, 2]));
        assert_eq!(req.top_k, 16);
    }

    #[test]
    fn decode_features_only_binary() {
        let body = make_single_binary(0, 1, false, 8, &[1.0, 2.0, 3.0, 4.0]);
        let req = decode_binary_request(&body).unwrap();
        assert!(!req.full_output);
    }

    #[test]
    fn decode_binary_truncated_body() {
        let body = vec![0u8; 8];
        assert!(decode_binary_request(&body).is_err());
    }

    #[test]
    fn decode_binary_empty_body() {
        assert!(decode_binary_request(&[]).is_err());
    }

    #[test]
    fn decode_binary_batch_truncated_layers() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // claim 4 layers
        buf.extend_from_slice(&0u32.to_le_bytes()); // only 1
        buf.extend_from_slice(&[0u8; 4]);
        assert!(decode_binary_request(&buf).is_err());
    }

    #[test]
    fn decode_binary_batch_reject_impossible_layer_count() {
        // Alloc-bomb guard: a claimed u32::MAX layer list must be rejected
        // by the length check before any allocation happens.
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        buf.extend_from_slice(&[0u8; 12]);
        assert!(decode_binary_request(&buf).is_err());
    }

    #[test]
    fn decode_request_dispatches_f16_and_i8_inbound_bodies() {
        use larql_inference::ffn::remote::{encode_binary_request_as, WireFormat};
        // Values exactly representable in both compressed formats (f16-exact
        // fractions; i8 fixed-point with scale 0.5) so equality is exact.
        let residual = vec![63.5f32, -32.0, 0.5, -63.5];
        for fmt in [WireFormat::F16, WireFormat::I8] {
            let body = encode_binary_request_as(fmt, Some(3), None, &residual, 1, true, 0);
            let req = decode_request(fmt, &body).unwrap();
            assert_eq!(req.layer, Some(3));
            assert_eq!(req.seq_len, 1);
            assert!(req.full_output);
            assert!(!req.moe_layer);
            assert_eq!(req.residual, residual, "{fmt:?}");
        }
        // Truncated compressed bodies reject through the same shim path.
        let mut body =
            encode_binary_request_as(WireFormat::F16, Some(3), None, &residual, 1, true, 0);
        body.push(0u8); // odd f16 payload
        assert!(decode_request(WireFormat::F16, &body).is_err());
    }

    #[test]
    fn decode_binary_odd_residual_length() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.push(0u8); // 1-byte residual — not a multiple of 4
        assert!(decode_binary_request(&buf).is_err());
    }

    // ── encode_binary_output (f32) ────────────────────────────────────────────

    #[test]
    fn encode_single_entry_output() {
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 5,
                output: vec![1.0f32, -2.0, 3.5],
            }],
            seq_len: 1,
            latency_ms: 7.3,
        };
        let bytes = encode_binary_output(&out);
        assert_eq!(bytes.len(), 4 + 4 + 4 + 3 * 4);
        let layer = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let seq_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let latency = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(layer, 5);
        assert_eq!(seq_len, 1);
        assert!((latency - 7.3f32).abs() < 0.01);
        let v0 = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert!((v0 - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn encode_batch_output() {
        let out = FfnOutput {
            entries: vec![
                FfnEntry {
                    layer: 5,
                    output: vec![1.0f32, 2.0],
                },
                FfnEntry {
                    layer: 20,
                    output: vec![3.0f32, 4.0],
                },
            ],
            seq_len: 1,
            latency_ms: 15.0,
        };
        let bytes = encode_binary_output(&out);
        let marker = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num_results = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(num_results, 2);
        let latency = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert!((latency - 15.0f32).abs() < 0.01);
        let layer0 = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(layer0, 5);
        let num_floats0 = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(num_floats0, 2);
    }

    #[test]
    fn binary_roundtrip_float_preservation() {
        let original_output = vec![0.12345f32, -9.87654, 1e-7, f32::MAX / 2.0];
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 10,
                output: original_output.clone(),
            }],
            seq_len: 1,
            latency_ms: 1.0,
        };
        let bytes = encode_binary_output(&out);
        // Skip 12-byte header; decode float values.
        let decoded: Vec<f32> = bytes[12..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, original_output);
    }

    // ── encode_json_full_output ──────────────────────────────────────────────

    #[test]
    fn json_single_layer_format() {
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 7,
                output: vec![1.0f32, 2.0, 3.0],
            }],
            seq_len: 1,
            latency_ms: 4.2,
        };
        let v = encode_json_full_output(&out);
        assert!(v.get("layer").is_some());
        assert!(v.get("output").is_some());
        assert!(v.get("results").is_none());
        assert_eq!(v["layer"].as_u64(), Some(7));
    }

    #[test]
    fn json_batch_format() {
        let out = FfnOutput {
            entries: vec![
                FfnEntry {
                    layer: 0,
                    output: vec![1.0f32],
                },
                FfnEntry {
                    layer: 1,
                    output: vec![2.0f32],
                },
            ],
            seq_len: 2,
            latency_ms: 20.0,
        };
        let v = encode_json_full_output(&out);
        assert!(v.get("results").is_some());
        let results = v["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["layer"].as_u64(), Some(0));
    }

    // ── encode_binary_output_f16 / _i8 ────────────────────────────────────────

    fn make_single_output(layer: u32, vals: Vec<f32>) -> FfnOutput {
        FfnOutput {
            entries: vec![FfnEntry {
                layer: layer as usize,
                output: vals,
            }],
            seq_len: 1,
            latency_ms: 5.0,
        }
    }

    fn make_batch_output(entries: Vec<(u32, Vec<f32>)>) -> FfnOutput {
        FfnOutput {
            entries: entries
                .into_iter()
                .map(|(l, v)| FfnEntry {
                    layer: l as usize,
                    output: v,
                })
                .collect(),
            seq_len: 1,
            latency_ms: 8.0,
        }
    }

    #[test]
    fn encode_f16_single_entry_halves_payload_size() {
        let out = make_single_output(3, vec![1.0, -2.0, 3.5, 4.0]);
        let bytes = encode_binary_output_f16(&out);
        assert_eq!(bytes.len(), 4 + 4 + 4 + 4 * 2);
        let layer = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(layer, 3);
    }

    #[test]
    fn encode_f16_batch_uses_marker_header() {
        let out = make_batch_output(vec![(0, vec![1.0, 2.0]), (1, vec![3.0, 4.0])]);
        let bytes = encode_binary_output_f16(&out);
        let marker = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(num, 2);
    }

    #[test]
    fn encode_i8_single_entry_symmetric_quantisation() {
        let out = make_single_output(7, vec![1.0, -1.0, 0.5, -0.5]);
        let bytes = encode_binary_output_i8(&out);
        assert_eq!(bytes.len(), 4 + 4 + 4 + 4 + 4 + 4);
        let zero = f32::from_le_bytes(bytes[12 + 4..12 + 8].try_into().unwrap());
        assert_eq!(zero, 0.0, "symmetric quantisation: zero_point=0");
    }

    #[test]
    fn encode_i8_batch_marker_then_per_entry_quantisation() {
        let out = make_batch_output(vec![(0, vec![2.0, -2.0]), (1, vec![1.0, -1.0])]);
        let bytes = encode_binary_output_i8(&out);
        let marker = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(num, 2);
    }

    #[test]
    fn encode_i8_zero_input_uses_unit_scale() {
        let out = make_single_output(0, vec![0.0; 4]);
        let bytes = encode_binary_output_i8(&out);
        let scale = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(scale, 1.0);
        for &b in &bytes[20..24] {
            assert_eq!(b as i8, 0);
        }
    }
}
