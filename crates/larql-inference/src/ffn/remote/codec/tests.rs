//! Tests for [`super`].
//!
//! Split out of `codec.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::*;

// ── JSON serialisation ────────────────────────────────────────────────────

#[test]
fn request_serializes_with_seq_len_and_full_output() {
    let req = WalkFfnHttpRequest {
        layer: Some(3),
        layers: None,
        residual: vec![0.1, -0.2, 0.3, 0.4],
        seq_len: 2,
        full_output: true,
    };
    let v: serde_json::Value = serde_json::to_value(&req).unwrap();
    assert_eq!(v["layer"], 3);
    assert_eq!(v["seq_len"], 2);
    assert_eq!(v["full_output"], true);
    assert!(
        v.get("layers").is_none() || v["layers"].is_null(),
        "layers should not appear when None, got: {v}"
    );
    assert_eq!(v["residual"].as_array().unwrap().len(), 4);
}

#[test]
fn response_deserializes_hidden_vector() {
    let json = serde_json::json!({
        "layer": 5,
        "output": [0.1, 0.2, 0.3, 0.4, 0.5],
        "seq_len": 1,
        "latency_ms": 2.5,
    });
    let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
    assert_eq!(parsed.layer, 5);
    assert_eq!(parsed.output.len(), 5);
    assert_eq!(parsed.seq_len, 1);
}

#[test]
fn response_deserializes_multi_token_output() {
    let flat: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let json = serde_json::json!({
        "layer": 0,
        "output": flat,
        "seq_len": 3,
    });
    let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
    assert_eq!(parsed.output.len(), 12);
    assert_eq!(parsed.seq_len, 3);
}

// ── encode_binary_request ─────────────────────────────────────────────────

#[test]
fn encode_single_layer_header() {
    let residual = vec![1.0f32, 2.0, 3.0, 4.0];
    let body = encode_binary_request(Some(7), None, &residual, 1, true, 256);
    // First u32 = layer index
    let layer = u32::from_le_bytes(body[0..4].try_into().unwrap());
    assert_eq!(layer, 7);
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
    assert_eq!(seq_len, 1);
    let flags = u32::from_le_bytes(body[8..12].try_into().unwrap());
    assert_eq!(flags & 1, 1); // full_output
    let top_k = u32::from_le_bytes(body[12..16].try_into().unwrap());
    assert_eq!(top_k, 256);
    assert_eq!(body.len(), 16 + 4 * 4);
}

#[test]
fn encode_batch_header() {
    let residual = vec![0.5f32; 4];
    let body = encode_binary_request(None, Some(&[5, 20, 30]), &residual, 1, true, 512);
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    assert_eq!(marker, BATCH_MARKER);
    let num_layers = u32::from_le_bytes(body[4..8].try_into().unwrap());
    assert_eq!(num_layers, 3);
    let l0 = u32::from_le_bytes(body[8..12].try_into().unwrap());
    let l1 = u32::from_le_bytes(body[12..16].try_into().unwrap());
    let l2 = u32::from_le_bytes(body[16..20].try_into().unwrap());
    assert_eq!((l0, l1, l2), (5, 20, 30));
}

#[test]
fn encode_residual_values_preserved() {
    let residual = vec![-1.5f32, 0.0, 3.25];
    let body = encode_binary_request(Some(0), None, &residual, 1, true, 8092);
    let offset = 16; // 4 header u32s × 4 bytes
    let v0 = f32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
    let v1 = f32::from_le_bytes(body[offset + 4..offset + 8].try_into().unwrap());
    let v2 = f32::from_le_bytes(body[offset + 8..offset + 12].try_into().unwrap());
    assert_eq!(v0.to_bits(), (-1.5f32).to_bits());
    assert_eq!(v1.to_bits(), 0.0f32.to_bits());
    assert!((v2 - 3.25f32).abs() < 1e-5);
}

// ── decode_binary_single ──────────────────────────────────────────────────

fn make_single_response(layer: u32, seq_len: u32, latency: f32, output: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &v in output {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn make_batch_response(latency: f32, entries: &[(u32, &[f32])]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &(layer, floats) in entries {
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        buf.extend_from_slice(&(floats.len() as u32).to_le_bytes());
        for &v in floats {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

#[test]
fn decode_single_response_correct() {
    let output = vec![1.0f32, -2.0, 3.5];
    let body = make_single_response(5, 1, 7.3, &output);
    let (layer, floats) = decode_binary_single(&body).unwrap();
    assert_eq!(layer, 5);
    assert_eq!(floats.len(), 3);
    assert!((floats[0] - 1.0).abs() < 1e-6);
    assert!((floats[1] - (-2.0)).abs() < 1e-6);
}

#[test]
fn decode_single_response_rejects_batch_marker() {
    let body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
    let result = decode_binary_single(&body);
    assert!(result.is_err());
}

#[test]
fn decode_single_response_too_short() {
    let result = decode_binary_single(&[0u8; 8]);
    assert!(result.is_err());
}

// ── decode_binary_batch ───────────────────────────────────────────────────

#[test]
fn decode_batch_response_correct() {
    let body = make_batch_response(15.0, &[(5, &[1.0, 2.0]), (20, &[3.0, 4.0])]);
    let map = decode_binary_batch(&body).unwrap();
    assert_eq!(map.len(), 2);
    let v5 = map.get(&5).unwrap();
    assert_eq!(v5.len(), 2);
    assert!((v5[0] - 1.0).abs() < 1e-6);
    let v20 = map.get(&20).unwrap();
    assert!((v20[1] - 4.0).abs() < 1e-6);
}

#[test]
fn decode_batch_accepts_single_response() {
    // A server returning single-layer response to a same-shard batch.
    let output = vec![7.0f32, 8.0];
    let body = make_single_response(10, 1, 5.0, &output);
    let map = decode_binary_batch(&body).unwrap();
    assert_eq!(map.len(), 1);
    assert!(map.contains_key(&10));
}

#[test]
fn decode_batch_truncated_returns_error() {
    let mut body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
    body.truncate(body.len() - 4); // cut off last float
    let result = decode_binary_batch(&body);
    assert!(result.is_err());
}

#[test]
fn decode_single_rejects_partial_float_payload() {
    let mut body = make_single_response(5, 1, 7.3, &[1.0]);
    body.push(0);
    let result = decode_binary_single(&body);
    assert!(result.is_err());
}

#[test]
fn decode_batch_rejects_impossible_result_count_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&u32::MAX.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    let result = decode_binary_batch(&body);
    assert!(result.is_err());
}

#[test]
fn decode_batch_rejects_impossible_output_length_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    body.extend_from_slice(&5u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&u32::MAX.to_le_bytes());
    let result = decode_binary_batch(&body);
    assert!(result.is_err());
}

#[test]
fn decode_batch_i8_rejects_inconsistent_output_shape() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    body.extend_from_slice(&5u32.to_le_bytes());
    body.extend_from_slice(&2u32.to_le_bytes());
    body.extend_from_slice(&3u32.to_le_bytes());
    let result = decode_binary_batch_i8(&body, 2);
    assert!(result.is_err());
}

// ── decode_binary_single_f16 + decode_binary_batch_f16 ─────────────

fn make_single_response_f16(layer: u32, seq_len: u32, latency: f32, output: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut buf = Vec::new();
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &v in output {
        buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    buf
}

fn make_batch_response_f16(latency: f32, entries: &[(u32, &[f32])]) -> Vec<u8> {
    use half::f16;
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &(layer, floats) in entries {
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        buf.extend_from_slice(&(floats.len() as u32).to_le_bytes());
        for &v in floats {
            buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
    }
    buf
}

#[test]
fn decode_single_f16_round_trip_within_quant_noise() {
    let body = make_single_response_f16(7, 1, 1.0, &[0.5, -0.25, 1.5, -2.5]);
    let (layer, floats) = decode_binary_single_f16(&body).unwrap();
    assert_eq!(layer, 7);
    assert_eq!(floats.len(), 4);
    // f16 round-trip is exact for these clean fractions.
    assert!((floats[0] - 0.5).abs() < 1e-6);
    assert!((floats[3] - (-2.5)).abs() < 1e-6);
}

#[test]
fn decode_single_f16_too_short_errors() {
    assert!(decode_binary_single_f16(&[0u8; 8]).is_err());
}

#[test]
fn decode_single_f16_rejects_batch_marker() {
    let body = make_batch_response_f16(1.0, &[(0, &[1.0])]);
    assert!(decode_binary_single_f16(&body).is_err());
}

#[test]
fn decode_single_f16_rejects_odd_payload_length() {
    let mut body = make_single_response_f16(0, 1, 0.0, &[1.0]);
    body.push(0u8); // odd byte tail
    assert!(decode_binary_single_f16(&body).is_err());
}

#[test]
fn decode_batch_f16_round_trip_two_entries() {
    let body = make_batch_response_f16(2.0, &[(3, &[1.0, 2.0]), (11, &[-1.0, 0.5])]);
    let map = decode_binary_batch_f16(&body).unwrap();
    assert_eq!(map.len(), 2);
    let v3 = map.get(&3).unwrap();
    assert!((v3[0] - 1.0).abs() < 1e-6 && (v3[1] - 2.0).abs() < 1e-6);
    let v11 = map.get(&11).unwrap();
    assert!((v11[1] - 0.5).abs() < 1e-6);
}

#[test]
fn decode_batch_f16_falls_through_to_single_when_no_marker() {
    let body = make_single_response_f16(5, 1, 1.0, &[1.0, 2.0, 3.0]);
    let map = decode_binary_batch_f16(&body).unwrap();
    assert_eq!(map.len(), 1);
    assert!(map.contains_key(&5));
}

#[test]
fn decode_batch_f16_too_short_errors() {
    assert!(decode_binary_batch_f16(&[0u8; 4]).is_err());
}

#[test]
fn decode_batch_f16_rejects_impossible_result_count() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&u32::MAX.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    assert!(decode_binary_batch_f16(&body).is_err());
}

// ── decode_binary_single_i8 + decode_binary_batch_i8 ───────────────

fn make_single_response_i8(
    layer: u32,
    seq_len: u32,
    latency: f32,
    positions: &[(f32, &[i8])],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &(scale, data) in positions {
        buf.extend_from_slice(&scale.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes()); // zero_point ignored
        for &b in data {
            buf.push(b as u8);
        }
    }
    buf
}

#[allow(clippy::type_complexity)]
fn make_batch_response_i8(latency: f32, entries: &[(u32, u32, &[(f32, &[i8])])]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &(layer, seq_len, positions) in entries {
        // num_floats per the codec contract (`seq_len * hidden_size`)
        let hidden = positions.first().map(|(_, d)| d.len()).unwrap_or(0);
        let num_floats = seq_len as usize * hidden;
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&(num_floats as u32).to_le_bytes());
        for &(scale, data) in positions {
            buf.extend_from_slice(&scale.to_le_bytes());
            buf.extend_from_slice(&0.0f32.to_le_bytes());
            for &b in data {
                buf.push(b as u8);
            }
        }
    }
    buf
}

#[test]
fn decode_single_i8_round_trip_one_position() {
    let hidden = 4;
    let body = make_single_response_i8(5, 1, 1.0, &[(0.5, &[2i8, -4, 8, -8])]);
    let (layer, floats) = decode_binary_single_i8(&body, hidden).unwrap();
    assert_eq!(layer, 5);
    assert_eq!(floats, vec![1.0f32, -2.0, 4.0, -4.0]);
}

#[test]
fn decode_single_i8_round_trip_multi_position() {
    let hidden = 2;
    let body = make_single_response_i8(
        0,
        3,
        0.0,
        &[(1.0, &[10i8, 20]), (0.25, &[-4i8, 8]), (2.0, &[1i8, -1])],
    );
    let (_, floats) = decode_binary_single_i8(&body, hidden).unwrap();
    assert_eq!(floats, vec![10.0, 20.0, -1.0, 2.0, 2.0, -2.0]);
}

#[test]
fn decode_single_i8_rejects_batch_marker() {
    let body = make_batch_response_i8(1.0, &[(0, 1, &[(1.0, &[1i8])])]);
    assert!(decode_binary_single_i8(&body, 1).is_err());
}

#[test]
fn decode_single_i8_too_short_errors() {
    assert!(decode_binary_single_i8(&[0u8; 8], 4).is_err());
}

#[test]
fn decode_single_i8_zero_seq_len_treated_as_one() {
    // Codec promotes seq_len 0 → 1 to keep the per-position loop alive.
    let body = make_single_response_i8(2, 0, 0.0, &[(1.0, &[5i8])]);
    let (layer, floats) = decode_binary_single_i8(&body, 1).unwrap();
    assert_eq!(layer, 2);
    assert_eq!(floats, vec![5.0]);
}

#[test]
fn decode_single_i8_truncated_payload_errors() {
    // Position needs 8 + hidden bytes; cut the payload short.
    let mut body = make_single_response_i8(0, 1, 0.0, &[(1.0, &[1i8, 2, 3, 4])]);
    body.truncate(body.len() - 1);
    assert!(decode_binary_single_i8(&body, 4).is_err());
}

#[test]
fn decode_batch_i8_round_trip_two_layers() {
    let hidden = 2;
    let body = make_batch_response_i8(
        3.0,
        &[
            (10, 1, &[(1.0, &[7i8, -7])]),
            (20, 2, &[(0.5, &[10i8, -10]), (0.5, &[20i8, -20])]),
        ],
    );
    let map = decode_binary_batch_i8(&body, hidden).unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&10).unwrap(), &vec![7.0, -7.0]);
    assert_eq!(map.get(&20).unwrap(), &vec![5.0, -5.0, 10.0, -10.0]);
}

#[test]
fn decode_batch_i8_falls_through_to_single_when_no_marker() {
    let body = make_single_response_i8(9, 1, 0.0, &[(1.0, &[3i8, -3])]);
    let map = decode_binary_batch_i8(&body, 2).unwrap();
    assert_eq!(map.len(), 1);
    assert!(map.contains_key(&9));
}

#[test]
fn decode_batch_i8_too_short_errors() {
    assert!(decode_binary_batch_i8(&[0u8; 4], 1).is_err());
}

#[test]
fn decode_batch_i8_rejects_impossible_result_count() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&u32::MAX.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    assert!(decode_binary_batch_i8(&body, 2).is_err());
}

// ── extract_response_latency_ms ────────────────────────────────────

#[test]
fn extract_latency_returns_zero_for_short_body() {
    assert_eq!(extract_response_latency_ms(&[]), 0.0);
    assert_eq!(extract_response_latency_ms(&[0u8; 11]), 0.0);
}

#[test]
fn extract_latency_reads_offset_8_as_f32() {
    // Body: layer(4) + seq_len(4) + latency(4)=8.5
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&8.5f32.to_le_bytes());
    assert!((extract_response_latency_ms(&body) - 8.5).abs() < 1e-6);
}

#[test]
fn extract_latency_returns_zero_for_non_finite() {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&f32::NAN.to_le_bytes());
    assert_eq!(extract_response_latency_ms(&body), 0.0);
}

// ── Wire string consts ─────────────────────────────────────────────

#[test]
fn binary_content_type_consts_pin_wire_strings() {
    assert_eq!(BINARY_CT, "application/x-larql-ffn");
    assert_eq!(F16_CT, "application/x-larql-ffn-f16");
    assert_eq!(I8_CT, "application/x-larql-ffn-i8");
    assert_eq!(BATCH_MARKER, 0xFFFF_FFFFu32);
}

// ── decode_single_response (content-type dispatch) ─────────────────

#[test]
fn decode_single_response_dispatches_f32() {
    let output = vec![1.0f32, -2.0, 3.5, 0.25];
    let body = make_single_response(3, 2, 4.5, &output);
    let (layer, server_ms, floats) = decode_single_response(BINARY_CT, &body, 2).unwrap();
    assert_eq!(layer, 3);
    assert!((server_ms - 4.5).abs() < 1e-6);
    assert_eq!(floats, output);
}

#[test]
fn decode_single_response_dispatches_f16() {
    let body = make_single_response_f16(7, 2, 2.25, &[0.5, -0.25, 1.5, -2.5]);
    let (layer, server_ms, floats) = decode_single_response(F16_CT, &body, 2).unwrap();
    assert_eq!(layer, 7);
    assert!((server_ms - 2.25).abs() < 1e-6);
    assert_eq!(floats, vec![0.5, -0.25, 1.5, -2.5]);
}

#[test]
fn decode_single_response_dispatches_i8_multi_row() {
    // seq_len = 3 rows of hidden = 2: the B>1 replay shape.
    let body = make_single_response_i8(
        0,
        3,
        1.5,
        &[(1.0, &[10i8, 20]), (0.25, &[-4i8, 8]), (2.0, &[1i8, -1])],
    );
    let (layer, server_ms, floats) = decode_single_response(I8_CT, &body, 2).unwrap();
    assert_eq!(layer, 0);
    assert!((server_ms - 1.5).abs() < 1e-6);
    assert_eq!(floats, vec![10.0, 20.0, -1.0, 2.0, 2.0, -2.0]);
}

#[test]
fn decode_single_response_rejects_unknown_content_type() {
    let body = make_single_response(0, 1, 0.0, &[1.0]);
    assert!(decode_single_response("application/json", &body, 1).is_err());
}

#[test]
fn encode_binary_request_top_k_zero_pins_l2_cache_bypass() {
    // The server's FfnL2Cache only engages when seq_len==1 && top_k>0.
    // Replay frames encode top_k=0 so repeated B=1 requests measure
    // real FFN compute, not cache hits. Pin the byte position.
    let residual = vec![0.5f32; 4];
    let body = encode_binary_request(Some(3), None, &residual, 1, true, 0);
    let top_k = u32::from_le_bytes(body[12..16].try_into().unwrap());
    assert_eq!(top_k, 0);
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
    assert_eq!(seq_len, 1);
}

#[test]
fn encode_binary_request_multi_row_seq_len() {
    // B-row replay frame: residual = B × hidden floats, seq_len = B.
    let hidden = 4;
    let batch = 8;
    let rows: Vec<f32> = (0..batch * hidden).map(|i| i as f32 * 0.1).collect();
    let body = encode_binary_request(Some(0), None, &rows, batch, true, 0);
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
    assert_eq!(seq_len as usize, batch);
    assert_eq!(body.len(), 16 + batch * hidden * 4);
}

#[test]
fn binary_request_response_roundtrip() {
    // Encode a single-layer request, then simulate what the server echoes.
    let residual = vec![0.1f32, 0.2, 0.3, 0.4];
    let req = encode_binary_request(Some(5), None, &residual, 1, true, 8092);
    // Simulate server extracting the layer.
    let layer = u32::from_le_bytes(req[0..4].try_into().unwrap());
    assert_eq!(layer, 5);

    // Simulate server response.
    let output = vec![0.9f32, 0.8, 0.7, 0.6];
    let resp = make_single_response(layer, 1, 8.5, &output);
    let (resp_layer, floats) = decode_binary_single(&resp).unwrap();
    assert_eq!(resp_layer as u32, layer);
    assert_eq!(floats, output);
}

// ── decode_binary_request (server-side half) ───────────────────────

#[test]
fn decode_request_single_layer() {
    let body = encode_binary_request(Some(5), None, &[1.0, 2.0, 3.0, 4.0], 1, true, 8);
    let req = decode_binary_request(&body).unwrap();
    assert_eq!(req.layer, Some(5));
    assert!(req.layers.is_none());
    assert_eq!(req.seq_len, 1);
    assert_eq!(req.top_k, 8);
    assert!(req.full_output);
    assert_eq!(req.residual, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn decode_request_batch() {
    let body = encode_binary_request(None, Some(&[0, 1, 2]), &[1.0; 4], 1, true, 16);
    let req = decode_binary_request(&body).unwrap();
    assert!(req.layer.is_none());
    assert_eq!(req.layers, Some(vec![0, 1, 2]));
    assert_eq!(req.top_k, 16);
}

#[test]
fn decode_request_features_only_flag() {
    let body = encode_binary_request(Some(0), None, &[1.0; 4], 1, false, 8);
    let req = decode_binary_request(&body).unwrap();
    assert!(!req.full_output);
}

#[test]
fn decode_request_truncated_body_errors() {
    assert!(decode_binary_request(&[0u8; 8]).is_err());
    assert!(decode_binary_request(&[]).is_err());
}

#[test]
fn decode_request_batch_truncated_layers_errors() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes()); // claim 4 layers
    buf.extend_from_slice(&0u32.to_le_bytes()); // only 1
    buf.extend_from_slice(&[0u8; 4]);
    assert!(decode_binary_request(&buf).is_err());
}

#[test]
fn decode_request_batch_impossible_layer_count_errors() {
    // num_layers = u32::MAX must be rejected by a length guard before
    // any layer-index allocation (alloc-bomb bound, PR 104 lineage).
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    buf.extend_from_slice(&[0u8; 12]);
    assert!(decode_binary_request(&buf).is_err());
}

#[test]
fn decode_request_odd_residual_length_errors() {
    let mut buf = encode_binary_request(Some(0), None, &[1.0], 1, true, 8);
    buf.push(0u8); // 1 stray byte — residual not a multiple of 4
    assert!(decode_binary_request(&buf).is_err());
}

// ── Byte-identical wire pins (encode → decode → re-encode) ─────────
//
// Mirrors `moe_remote::multi_layer_wire`'s
// `multi_task_encode_decode_reencode_is_byte_identical`: the shared
// codec must reproduce the exact original bytes for every frame kind,
// pinning the consolidated implementation to the pre-consolidation
// wire (single + batch × f32/f16/i8 responses, single + batch
// requests).

fn reencode_request(req: &DecodedFfnRequest) -> Vec<u8> {
    encode_binary_request(
        req.layer,
        req.layers.as_deref(),
        &req.residual,
        req.seq_len,
        req.full_output,
        req.top_k,
    )
}

#[test]
fn request_encode_decode_reencode_is_byte_identical() {
    // Single-layer frame.
    let encoded = encode_binary_request(
        Some(7),
        None,
        &[1.5, -2.25, 0.0, f32::MIN_POSITIVE],
        2,
        true,
        8092,
    );
    let decoded = decode_binary_request(&encoded).unwrap();
    assert_eq!(reencode_request(&decoded), encoded);

    // Batch frame.
    let encoded =
        encode_binary_request(None, Some(&[3, 900, 41]), &[0.125, -3.5, 1e-20], 1, true, 0);
    let decoded = decode_binary_request(&encoded).unwrap();
    assert_eq!(reencode_request(&decoded), encoded);
}

// ── WireFormat (direction axis) ────────────────────────────────────

#[test]
fn wire_format_content_types_labels_and_parse() {
    assert_eq!(WireFormat::F32.content_type(), BINARY_CT);
    assert_eq!(WireFormat::F16.content_type(), F16_CT);
    assert_eq!(WireFormat::I8.content_type(), I8_CT);
    for f in [WireFormat::F32, WireFormat::F16, WireFormat::I8] {
        assert_eq!(WireFormat::parse(f.label()), Some(f));
    }
    assert_eq!(WireFormat::parse(" f16 "), Some(WireFormat::F16));
    assert_eq!(WireFormat::parse("q8k"), None);
    assert_eq!(WireFormat::default(), WireFormat::F32);
}

#[test]
fn wire_format_from_content_type_checks_suffixed_types_first() {
    // BINARY_CT is a substring of F16_CT/I8_CT — the ordered match must
    // never misread a compressed body as f32.
    assert_eq!(WireFormat::from_content_type(F16_CT), Some(WireFormat::F16));
    assert_eq!(WireFormat::from_content_type(I8_CT), Some(WireFormat::I8));
    assert_eq!(
        WireFormat::from_content_type(BINARY_CT),
        Some(WireFormat::F32)
    );
    // Parameterised types still match.
    assert_eq!(
        WireFormat::from_content_type("application/x-larql-ffn-f16; v=2"),
        Some(WireFormat::F16)
    );
    assert_eq!(WireFormat::from_content_type("application/json"), None);
}

// ── f16/i8 request encode/decode (asymmetric inbound direction) ────

#[test]
fn encode_request_as_f32_is_byte_identical_to_legacy_encoder() {
    // The symmetric wrapper and the format-parameterised encoder must
    // produce the same bytes — existing callers keep their wire.
    let residual = vec![1.5f32, -2.25, 0.0, f32::MIN_POSITIVE];
    assert_eq!(
        encode_binary_request_as(WireFormat::F32, Some(7), None, &residual, 2, true, 8092),
        encode_binary_request(Some(7), None, &residual, 2, true, 8092),
    );
    assert_eq!(
        encode_binary_request_as(WireFormat::F32, None, Some(&[1, 2]), &residual, 1, true, 0),
        encode_binary_request(None, Some(&[1, 2]), &residual, 1, true, 0),
    );
}

#[test]
fn f16_request_shares_header_layout_and_halves_payload() {
    let residual = vec![0.5f32, -0.25, 1.5, -2.5];
    let body = encode_binary_request_as(WireFormat::F16, Some(7), None, &residual, 1, true, 256);
    assert_eq!(body.len(), REQUEST_HEADER_LEN + residual.len() * 2);
    // Header bytes identical to the f32 frame's.
    let f32_body = encode_binary_request(Some(7), None, &residual, 1, true, 256);
    assert_eq!(body[..REQUEST_HEADER_LEN], f32_body[..REQUEST_HEADER_LEN]);
}

#[test]
fn request_f16_encode_decode_reencode_is_byte_identical() {
    // f16-exact values so decode → f32 → re-encode reproduces the bits.
    let residual = vec![0.5f32, -0.25, 1.5, -2.5];
    // Single-layer frame.
    let encoded =
        encode_binary_request_as(WireFormat::F16, Some(7), None, &residual, 2, true, 8092);
    let decoded = decode_binary_request_f16(&encoded).unwrap();
    assert_eq!(decoded.layer, Some(7));
    assert_eq!(decoded.seq_len, 2);
    assert_eq!(decoded.top_k, 8092);
    assert!(decoded.full_output);
    assert_eq!(decoded.residual, residual);
    let reencoded = encode_binary_request_as(
        WireFormat::F16,
        decoded.layer,
        decoded.layers.as_deref(),
        &decoded.residual,
        decoded.seq_len,
        decoded.full_output,
        decoded.top_k,
    );
    assert_eq!(reencoded, encoded);

    // Batch frame.
    let encoded = encode_binary_request_as(
        WireFormat::F16,
        None,
        Some(&[3, 900]),
        &residual,
        1,
        true,
        0,
    );
    let decoded = decode_binary_request_f16(&encoded).unwrap();
    assert_eq!(decoded.layers, Some(vec![3, 900]));
    let reencoded = encode_binary_request_as(
        WireFormat::F16,
        decoded.layer,
        decoded.layers.as_deref(),
        &decoded.residual,
        decoded.seq_len,
        decoded.full_output,
        decoded.top_k,
    );
    assert_eq!(reencoded, encoded);
}

#[test]
fn request_i8_encode_decode_reencode_is_byte_identical() {
    // Fixed-point construction (mirrors the i8 response pin): per
    // position max|v| = 127 × scale with scale a power of two, so
    // quantise(dequantise(q)) == q and the recomputed scale bit-matches.
    // Two positions of hidden = 4: scale 0.5, then scale 0.25.
    let residual = vec![63.5f32, -32.0, 0.5, -63.5, 31.75, -16.0, 0.25, 8.0];
    let encoded = encode_binary_request_as(WireFormat::I8, Some(5), None, &residual, 2, true, 0);
    assert_eq!(encoded.len(), REQUEST_HEADER_LEN + 2 * (8 + 4));
    let decoded = decode_binary_request_i8(&encoded).unwrap();
    assert_eq!(decoded.layer, Some(5));
    assert_eq!(decoded.seq_len, 2);
    assert_eq!(decoded.residual, residual);
    let reencoded = encode_binary_request_as(
        WireFormat::I8,
        decoded.layer,
        decoded.layers.as_deref(),
        &decoded.residual,
        decoded.seq_len,
        decoded.full_output,
        decoded.top_k,
    );
    assert_eq!(reencoded, encoded);

    // Batch frame, one position (seq_len 1).
    let row = vec![127.0f32, -64.0, 1.0, -127.0]; // scale 1.0
    let encoded = encode_binary_request_as(WireFormat::I8, None, Some(&[0, 9]), &row, 1, true, 0);
    let decoded = decode_binary_request_i8(&encoded).unwrap();
    assert_eq!(decoded.layers, Some(vec![0, 9]));
    assert_eq!(decoded.residual, row);
    let reencoded = encode_binary_request_as(
        WireFormat::I8,
        decoded.layer,
        decoded.layers.as_deref(),
        &decoded.residual,
        decoded.seq_len,
        decoded.full_output,
        decoded.top_k,
    );
    assert_eq!(reencoded, encoded);
}

#[test]
fn request_i8_zero_seq_len_treated_as_one() {
    // Mirrors the i8 response decoder's seq_len 0 → 1 promotion.
    let row = vec![127.0f32, -64.0];
    let encoded = encode_binary_request_as(WireFormat::I8, Some(2), None, &row, 0, true, 0);
    let decoded = decode_binary_request_i8(&encoded).unwrap();
    assert_eq!(decoded.residual, row);
}

#[test]
fn request_f16_i8_decoders_share_header_guards() {
    // Truncated header / alloc-bomb guards run before any payload work
    // in every dtype arm (shared parse_request_header).
    for decode in [
        decode_binary_request_f16 as fn(&[u8]) -> Result<DecodedFfnRequest, String>,
        decode_binary_request_i8,
    ] {
        assert!(decode(&[]).is_err());
        assert!(decode(&[0u8; 8]).is_err());
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        buf.extend_from_slice(&[0u8; 12]);
        assert!(decode(&buf).is_err(), "alloc-bomb layer count");
    }
}

#[test]
fn request_f16_rejects_odd_payload_length() {
    let mut buf = encode_binary_request_as(WireFormat::F16, Some(0), None, &[1.0], 1, true, 0);
    buf.push(0u8);
    assert!(decode_binary_request_f16(&buf).is_err());
}

#[test]
fn request_i8_rejects_bad_payload_shapes() {
    // Payload not a multiple of seq_len.
    let mut two_pos = encode_binary_request_as(
        WireFormat::I8,
        Some(0),
        None,
        &[1.0, 2.0, 3.0, 4.0],
        2,
        true,
        0,
    );
    two_pos.push(0u8);
    assert!(decode_binary_request_i8(&two_pos).is_err());
    // Per-position bytes ≤ 8 (scale/zero header only, no data).
    let mut hdr_only = Vec::new();
    hdr_only.extend_from_slice(&0u32.to_le_bytes());
    hdr_only.extend_from_slice(&1u32.to_le_bytes());
    hdr_only.extend_from_slice(&1u32.to_le_bytes());
    hdr_only.extend_from_slice(&0u32.to_le_bytes());
    hdr_only.extend_from_slice(&[0u8; 8]); // exactly one empty position
    assert!(decode_binary_request_i8(&hdr_only).is_err());
    // Empty payload decodes to an empty residual (validated downstream).
    let empty = encode_binary_request_as(WireFormat::I8, Some(0), None, &[], 1, true, 0);
    assert_eq!(
        decode_binary_request_i8(&empty).unwrap().residual,
        Vec::<f32>::new()
    );
}

#[test]
fn decode_request_as_dispatches_every_format() {
    let residual = vec![0.5f32, -0.25, 1.5, -2.5];
    for fmt in [WireFormat::F32, WireFormat::F16, WireFormat::I8] {
        let body = encode_binary_request_as(fmt, Some(3), None, &residual, 1, true, 0);
        let decoded = decode_binary_request_as(fmt, &body).unwrap();
        assert_eq!(decoded.layer, Some(3));
        assert_eq!(decoded.residual.len(), residual.len());
    }
}

// ── Asymmetric direction pairs (in/return independent — DEC-1A) ────

#[test]
fn asymmetric_pairs_round_trip_through_production_codecs() {
    // The four asymmetric combos: request encoded in `input`, decoded by
    // the server-side request decoder; response encoded in `output`,
    // decoded by the client-side content-type dispatcher. Values are
    // exactly representable in every arm (f16-exact, i8 fixed-point
    // with power-of-two scales) so equality is exact end to end.
    let hidden = 4usize;
    let residual = vec![63.5f32, -32.0, 0.5, -63.5]; // i8 scale 0.5, f16-exact
    for (input, output) in [
        (WireFormat::F16, WireFormat::I8),
        (WireFormat::I8, WireFormat::F16),
        (WireFormat::F32, WireFormat::I8),
        (WireFormat::F16, WireFormat::F32),
    ] {
        // Inbound: client encodes in `input`, server decodes by CT.
        let req_body = encode_binary_request_as(input, Some(9), None, &residual, 1, true, 0);
        let fmt = WireFormat::from_content_type(input.content_type()).unwrap();
        assert_eq!(fmt, input, "CT ↔ format mapping is bijective");
        let req = decode_binary_request_as(fmt, &req_body).unwrap();
        assert_eq!(
            req.residual,
            residual,
            "{}/{}",
            input.label(),
            output.label()
        );

        // Return: server echoes the residual through the `output`
        // response encoder; client decodes via the CT dispatcher.
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 9,
                output: req.residual.clone(),
            }],
            seq_len: 1,
            latency_ms: 2.5,
        };
        let resp_body = match output {
            WireFormat::F32 => encode_binary_output(&out),
            WireFormat::F16 => encode_binary_output_f16(&out),
            WireFormat::I8 => encode_binary_output_i8(&out),
        };
        let (layer, server_ms, floats) =
            decode_single_response(output.content_type(), &resp_body, hidden).unwrap();
        assert_eq!(layer, 9);
        assert!((server_ms - 2.5).abs() < 1e-6);
        assert_eq!(floats, residual, "{}/{}", input.label(), output.label());
    }
}

/// Rebuild an `FfnOutput` from a decoded layer→floats map, preserving
/// the original entry order (the decoders return `HashMap`s).
fn rebuild_output(
    map: &HashMap<usize, Vec<f32>>,
    layer_order: &[usize],
    seq_len: usize,
    latency_ms: f64,
) -> FfnOutput {
    FfnOutput {
        entries: layer_order
            .iter()
            .map(|&l| FfnEntry {
                layer: l,
                output: map[&l].clone(),
            })
            .collect(),
        seq_len,
        latency_ms,
    }
}

#[test]
fn response_f32_encode_decode_reencode_is_byte_identical() {
    // Single-layer frame.
    let out = FfnOutput {
        entries: vec![FfnEntry {
            layer: 5,
            output: vec![0.12345, -9.87654, 1e-7, f32::MAX / 2.0],
        }],
        seq_len: 2,
        latency_ms: 7.5,
    };
    let encoded = encode_binary_output(&out);
    let (layer, floats) = decode_binary_single(&encoded).unwrap();
    let latency = extract_response_latency_ms(&encoded);
    let rebuilt = FfnOutput {
        entries: vec![FfnEntry {
            layer,
            output: floats,
        }],
        seq_len: 2,
        latency_ms: latency,
    };
    assert_eq!(encode_binary_output(&rebuilt), encoded);

    // Batch frame.
    let out = FfnOutput {
        entries: vec![
            FfnEntry {
                layer: 3,
                output: vec![1.5, -2.25],
            },
            FfnEntry {
                layer: 29,
                output: vec![-0.0, 7.0],
            },
        ],
        seq_len: 1,
        latency_ms: 15.0,
    };
    let encoded = encode_binary_output(&out);
    let map = decode_binary_batch(&encoded).unwrap();
    let rebuilt = rebuild_output(&map, &[3, 29], 1, extract_response_latency_ms(&encoded));
    assert_eq!(encode_binary_output(&rebuilt), encoded);
}

#[test]
fn response_f16_encode_decode_reencode_is_byte_identical() {
    // Values chosen to be exactly f16-representable so decode → f32 →
    // re-encode reproduces identical half bits.
    let out = FfnOutput {
        entries: vec![FfnEntry {
            layer: 7,
            output: vec![0.5, -0.25, 1.5, -2.5],
        }],
        seq_len: 1,
        latency_ms: 2.25,
    };
    let encoded = encode_binary_output_f16(&out);
    let (layer, floats) = decode_binary_single_f16(&encoded).unwrap();
    let rebuilt = FfnOutput {
        entries: vec![FfnEntry {
            layer,
            output: floats,
        }],
        seq_len: 1,
        latency_ms: extract_response_latency_ms(&encoded),
    };
    assert_eq!(encode_binary_output_f16(&rebuilt), encoded);

    // Batch frame.
    let out = FfnOutput {
        entries: vec![
            FfnEntry {
                layer: 3,
                output: vec![1.0, 2.0],
            },
            FfnEntry {
                layer: 11,
                output: vec![-1.0, 0.5],
            },
        ],
        seq_len: 1,
        latency_ms: 8.0,
    };
    let encoded = encode_binary_output_f16(&out);
    let map = decode_binary_batch_f16(&encoded).unwrap();
    let rebuilt = rebuild_output(&map, &[3, 11], 1, extract_response_latency_ms(&encoded));
    assert_eq!(encode_binary_output_f16(&rebuilt), encoded);
}

#[test]
fn response_i8_encode_decode_reencode_is_byte_identical() {
    // Fixed-point construction: per position, max|v| = 127 × scale with
    // scale a power of two, so quantise(dequantise(q)) == q and the
    // recomputed scale bit-matches. hidden = 4.
    let out = FfnOutput {
        entries: vec![FfnEntry {
            layer: 5,
            // scale = 63.5 / 127 = 0.5 exactly; q = [127, -64, 1, -127]
            output: vec![63.5, -32.0, 0.5, -63.5],
        }],
        seq_len: 1,
        latency_ms: 3.5,
    };
    let encoded = encode_binary_output_i8(&out);
    let (layer, floats) = decode_binary_single_i8(&encoded, 4).unwrap();
    let rebuilt = FfnOutput {
        entries: vec![FfnEntry {
            layer,
            output: floats,
        }],
        seq_len: 1,
        latency_ms: extract_response_latency_ms(&encoded),
    };
    assert_eq!(encode_binary_output_i8(&rebuilt), encoded);

    // Batch frame, two layers, seq_len 2 (two quantised positions each).
    let out = FfnOutput {
        entries: vec![
            FfnEntry {
                layer: 10,
                // pos0 scale 0.25, pos1 scale 1.0
                output: vec![31.75, -16.0, 0.25, 8.0, 127.0, -64.0, 1.0, -127.0],
            },
            FfnEntry {
                layer: 20,
                // pos0 scale 2.0, pos1 scale 0.5
                output: vec![254.0, -2.0, 128.0, 4.0, 63.5, -0.5, 32.0, -63.5],
            },
        ],
        seq_len: 2,
        latency_ms: 1.0,
    };
    let encoded = encode_binary_output_i8(&out);
    let map = decode_binary_batch_i8(&encoded, 4).unwrap();
    let rebuilt = rebuild_output(&map, &[10, 20], 2, extract_response_latency_ms(&encoded));
    assert_eq!(encode_binary_output_i8(&rebuilt), encoded);
}

// ── encode_json_full_output ────────────────────────────────────────

#[test]
fn json_full_output_single_and_batch_shapes() {
    let single = FfnOutput {
        entries: vec![FfnEntry {
            layer: 7,
            output: vec![1.0, 2.0, 3.0],
        }],
        seq_len: 1,
        latency_ms: 4.24,
    };
    let v = encode_json_full_output(&single);
    assert_eq!(v["layer"].as_u64(), Some(7));
    assert!(v.get("results").is_none());
    assert_eq!(v["latency_ms"].as_f64(), Some(4.2)); // rounded to 0.1

    let batch = FfnOutput {
        entries: vec![
            FfnEntry {
                layer: 0,
                output: vec![1.0],
            },
            FfnEntry {
                layer: 1,
                output: vec![2.0],
            },
        ],
        seq_len: 2,
        latency_ms: 20.0,
    };
    let v = encode_json_full_output(&batch);
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[1]["layer"].as_u64(), Some(1));
    assert_eq!(results[1]["seq_len"].as_u64(), Some(2));
}
