//! Binary wire codec for the LARQL FFN remote protocol — the SINGLE
//! source of truth for the dense walk-ffn binary frame.
//!
//! Both halves of the wire live here (ROADMAP hardening item 16):
//! - client side: [`encode_binary_request`] + the response decoders
//!   ([`decode_binary_single`], [`decode_binary_batch`], f16/i8 variants,
//!   [`decode_single_response`]);
//! - server side: [`decode_binary_request`] + the response encoders
//!   ([`encode_binary_output`], [`encode_binary_output_f16`],
//!   [`encode_binary_output_i8`], [`encode_json_full_output`]).
//!
//! `larql-server` (`routes/walk_ffn/binary.rs`) and `larql-router` import
//! these functions/constants instead of maintaining their own copies.
//! See the `super` module doc for the full binary frame layout.

use serde::{Deserialize, Serialize};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;
use crate::collections::HashMap;

/// f32 content-type constant.
pub const BINARY_CT: &str = "application/x-larql-ffn";

/// First u32 of a binary request/response body. When equal to this marker
/// the frame is a batch (`marker + count + …`); otherwise the first u32 is
/// the layer index of a single-layer frame.
pub const BATCH_MARKER: u32 = 0xFFFF_FFFF;

/// Fixed response-header size in bytes: `layer|marker (4) + seq_len|count
/// (4) + latency_ms (4)`. Shared by the single and batch response frames
/// in every dtype arm.
pub const RESPONSE_HEADER_LEN: usize = 12;

/// Fixed single-layer request-header size in bytes: `layer (4) + seq_len
/// (4) + flags (4) + top_k (4)`. Batch requests carry `marker (4) +
/// num_layers (4) + layers (4×K)` before the trailing 12 fixed bytes.
pub const REQUEST_HEADER_LEN: usize = 16;

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("{what}: byte length overflow"))
}

fn checked_end(offset: usize, len: usize, total: usize, what: &str) -> Result<usize, String> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| format!("{what}: byte range overflow"))?;
    if end > total {
        return Err(format!(
            "{what}: truncated: need {len}, have {}",
            total.saturating_sub(offset)
        ));
    }
    Ok(end)
}

fn read_u32(body: &[u8], offset: usize, what: &str) -> Result<u32, String> {
    let end = checked_end(offset, 4, body.len(), what)?;
    Ok(u32::from_le_bytes(body[offset..end].try_into().unwrap()))
}

fn validate_batch_result_count(body: &[u8], num_results: usize, what: &str) -> Result<(), String> {
    let max_results_with_headers = body.len().saturating_sub(12) / 12;
    if num_results > max_results_with_headers {
        return Err(format!(
            "{what}: declared {num_results} results but only {max_results_with_headers} headers fit"
        ));
    }
    Ok(())
}

// ── Wire types (JSON fallback) ────────────────────────────────────────────────

#[derive(Serialize)]
#[allow(dead_code)]
pub(super) struct WalkFfnHttpRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<usize>>,
    pub residual: Vec<f32>,
    pub seq_len: usize,
    pub full_output: bool,
}

#[derive(Deserialize)]
pub(super) struct WalkFfnSingleResponse {
    #[allow(dead_code)]
    pub layer: usize,
    pub output: Vec<f32>,
    #[allow(dead_code)]
    pub seq_len: usize,
}

// ── Latency profiling result ──────────────────────────────────────────────────

/// Breakdown returned by [`super::http::RemoteWalkBackend::probe_latency`].
#[derive(Debug, Clone)]
pub struct RemoteLatencyStats {
    /// Wall-clock round-trip (client-measured), averaged over `samples` calls.
    pub total_ms: f64,
    /// FFN compute time reported by the server in the binary response header.
    pub server_ms: f64,
    /// `total_ms - server_ms`: HTTP framing + TCP + serialization overhead.
    pub overhead_ms: f64,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub samples: usize,
}

impl core::fmt::Display for RemoteLatencyStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "layers={} hidden={} samples={}\n  total    {:7.2} ms\n  server   {:7.2} ms  (FFN compute)\n  overhead {:7.2} ms  (HTTP + TCP + framing)",
            self.num_layers, self.hidden_size, self.samples,
            self.total_ms, self.server_ms, self.overhead_ms,
        )
    }
}

// ── Binary codec ──────────────────────────────────────────────────────────────

/// Append the request header (`layer|marker[+layers]`, `seq_len`, `flags`,
/// `top_k`) shared by every request dtype arm (f32/f16/i8 — ADR-0009: only
/// the residual payload encoding differs between formats).
fn push_request_header(
    buf: &mut Vec<u8>,
    layer: Option<usize>,
    layers: Option<&[usize]>,
    seq_len: usize,
    full_output: bool,
    top_k: usize,
) {
    if let Some(ls) = layers {
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(ls.len() as u32).to_le_bytes());
        for &l in ls {
            buf.extend_from_slice(&(l as u32).to_le_bytes());
        }
    } else {
        let l = layer.unwrap_or(0) as u32;
        buf.extend_from_slice(&l.to_le_bytes());
    }
    buf.extend_from_slice(&(seq_len as u32).to_le_bytes());
    buf.extend_from_slice(&(full_output as u32).to_le_bytes());
    buf.extend_from_slice(&(top_k as u32).to_le_bytes());
}

/// Encode a request as binary (f32 residual payload — the historical wire).
/// `layer` and `layers` are mutually exclusive; pass `None` for the unused one.
///
/// Thin symmetric wrapper over [`encode_binary_request_as`] with
/// [`WireFormat::F32`]; kept so no existing caller changes behaviour.
pub fn encode_binary_request(
    layer: Option<usize>,
    layers: Option<&[usize]>,
    residual: &[f32],
    seq_len: usize,
    full_output: bool,
    top_k: usize,
) -> Vec<u8> {
    encode_binary_request_as(
        WireFormat::F32,
        layer,
        layers,
        residual,
        seq_len,
        full_output,
        top_k,
    )
}

/// Encode a request with the residual payload in `format` (asymmetric
/// direction codecs, DEC funnel v0.5 §3 DEC-1A: the inbound format is
/// declared by the request `Content-Type` and is independent of the
/// `Accept`-negotiated return format).
///
/// The header bytes are identical across formats; the residual payload
/// mirrors the corresponding RESPONSE encoding exactly:
/// - f32: `f32 LE` per value;
/// - f16: `u16 LE` IEEE half per value;
/// - i8: per position `[scale f32 LE][zero_point f32 LE][data i8[hidden]]`
///   with `scale = max(|x|)/127`, symmetric (`hidden = residual.len() /
///   max(seq_len, 1)`; `residual.len()` must divide evenly).
pub fn encode_binary_request_as(
    format: WireFormat,
    layer: Option<usize>,
    layers: Option<&[usize]>,
    residual: &[f32],
    seq_len: usize,
    full_output: bool,
    top_k: usize,
) -> Vec<u8> {
    use half::f16;
    let mut buf = Vec::with_capacity(16 + residual.len() * 4);
    push_request_header(&mut buf, layer, layers, seq_len, full_output, top_k);
    match format {
        WireFormat::F32 => {
            for &v in residual {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        WireFormat::F16 => {
            for &v in residual {
                buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        WireFormat::I8 => {
            let seq = seq_len.max(1);
            debug_assert!(
                residual.is_empty() || residual.len().is_multiple_of(seq),
                "i8 request: residual length {} not a multiple of seq_len {seq}",
                residual.len()
            );
            let hidden = residual.len() / seq;
            if hidden > 0 {
                for pos in 0..seq {
                    quantise_i8_position(&residual[pos * hidden..(pos + 1) * hidden], &mut buf);
                }
            }
        }
    }
    buf
}

/// Decode a binary single-layer full_output response.
/// Returns `(layer, output_floats)`.
pub fn decode_binary_single(body: &[u8]) -> Result<(usize, Vec<f32>), String> {
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!("binary response too short: {} bytes", body.len()));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer response but got batch marker".into());
    }
    let layer = marker as usize;
    // bytes 4-7: seq_len (ignored here — caller validates against expected shape)
    // bytes 8-11: latency f32
    if !(body.len() - 12).is_multiple_of(4) {
        return Err("binary response: output byte length is not a multiple of f32".into());
    }
    let floats: Vec<f32> = body[12..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((layer, floats))
}

/// Decode a binary batch full_output response.
/// Returns a map from layer → output floats.
pub fn decode_binary_batch(body: &[u8]) -> Result<HashMap<usize, Vec<f32>>, String> {
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!(
            "binary batch response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());

    // Single-layer response — accept it as a batch of 1.
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single(body)?;
        let mut m = HashMap::default();
        m.insert(layer, floats);
        return Ok(m);
    }

    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    validate_batch_result_count(body, num_results, "binary batch")?;
    // bytes 8-11: latency f32 (skip)
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity_and_hasher(num_results, Default::default());

    for _ in 0..num_results {
        checked_end(offset, 12, body.len(), "binary batch result header")?;
        let layer = read_u32(body, offset, "binary batch layer")? as usize;
        // offset+4: seq_len (skip)
        let num_floats = read_u32(body, offset + 8, "binary batch output length")? as usize;
        offset += 12;
        let bytes_needed = checked_mul(num_floats, 4, "binary batch output")?;
        let end = checked_end(offset, bytes_needed, body.len(), "binary batch output")?;
        let floats: Vec<f32> = body[offset..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        offset = end;
        out.insert(layer, floats);
    }
    Ok(out)
}

/// f16 content-type constant (ADR-0009).
pub const F16_CT: &str = "application/x-larql-ffn-f16";

/// Decode a binary single-layer f16 response into f32 output.
pub fn decode_binary_single_f16(body: &[u8]) -> Result<(usize, Vec<f32>), String> {
    use half::f16;
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!(
            "f16 binary response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer f16 response but got batch marker".into());
    }
    let layer = marker as usize;
    if !(body.len() - 12).is_multiple_of(2) {
        return Err("f16 binary response: output byte length is not a multiple of f16".into());
    }
    let floats: Vec<f32> = body[12..]
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect();
    Ok((layer, floats))
}

/// Decode a binary batch f16 response into f32 outputs.
pub fn decode_binary_batch_f16(body: &[u8]) -> Result<HashMap<usize, Vec<f32>>, String> {
    use half::f16;
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!(
            "f16 batch response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single_f16(body)?;
        let mut m = HashMap::default();
        m.insert(layer, floats);
        return Ok(m);
    }
    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    validate_batch_result_count(body, num_results, "f16 batch")?;
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity_and_hasher(num_results, Default::default());
    for _ in 0..num_results {
        checked_end(offset, 12, body.len(), "f16 batch result header")?;
        let layer = read_u32(body, offset, "f16 batch layer")? as usize;
        let num_floats = read_u32(body, offset + 8, "f16 batch output length")? as usize;
        offset += 12;
        let bytes_needed = checked_mul(num_floats, 2, "f16 batch output")?;
        let end = checked_end(offset, bytes_needed, body.len(), "f16 batch output")?;
        let floats: Vec<f32> = body[offset..end]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        offset = end;
        out.insert(layer, floats);
    }
    Ok(out)
}

/// i8 content-type constant (ADR-0009).
pub const I8_CT: &str = "application/x-larql-ffn-i8";

/// Quantise one position into an i8 per-position block (symmetric):
/// `[scale f32 LE][zero_point f32 LE = 0][data i8[len]]` with
/// `scale = max(|x|)/127` (1.0 for an all-zero position). Shared by the
/// i8 response encoder and the i8 request encoder (same payload layout in
/// both directions — ADR-0009).
fn quantise_i8_position(vals: &[f32], buf: &mut Vec<u8>) {
    let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    buf.extend_from_slice(&scale.to_le_bytes());
    buf.extend_from_slice(&0.0f32.to_le_bytes()); // zero_point = 0
    for &v in vals {
        let q = (v / scale).clamp(-127.0, 127.0).round() as i8;
        buf.push(q as u8);
    }
}

/// Decode one position from an i8 per-position block.
/// Format: `[scale f32 LE][zero_point f32 LE (ignored)][data i8[hidden_size]]`
fn decode_i8_position(
    body: &[u8],
    offset: usize,
    hidden: usize,
) -> Result<(Vec<f32>, usize), String> {
    let needed = 8usize
        .checked_add(hidden)
        .ok_or_else(|| "i8: position byte length overflow".to_string())?;
    let end = checked_end(offset, needed, body.len(), "i8 position")?;
    let scale = f32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
    // zero_point at offset+4 is always 0.0 (symmetric), skip it
    let floats: Vec<f32> = body[offset + 8..offset + 8 + hidden]
        .iter()
        .map(|&b| (b as i8) as f32 * scale)
        .collect();
    Ok((floats, end))
}

/// Decode a binary single-layer i8 response into f32 output.
pub(crate) fn decode_binary_single_i8(
    body: &[u8],
    hidden_size: usize,
) -> Result<(usize, Vec<f32>), String> {
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!(
            "i8 binary response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer i8 response but got batch marker".into());
    }
    let layer = marker as usize;
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    let seq_len = seq_len.max(1);
    let mut offset = 12usize;
    let total_floats = checked_mul(seq_len, hidden_size, "i8 single output")?;
    let mut all_floats = Vec::with_capacity(total_floats);
    for _ in 0..seq_len {
        let (pos_floats, next_offset) = decode_i8_position(body, offset, hidden_size)?;
        all_floats.extend(pos_floats);
        offset = next_offset;
    }
    Ok((layer, all_floats))
}

/// Decode a binary batch i8 response into f32 outputs.
pub(crate) fn decode_binary_batch_i8(
    body: &[u8],
    hidden_size: usize,
) -> Result<HashMap<usize, Vec<f32>>, String> {
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!("i8 batch response too short: {} bytes", body.len()));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single_i8(body, hidden_size)?;
        let mut m = HashMap::default();
        m.insert(layer, floats);
        return Ok(m);
    }
    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    validate_batch_result_count(body, num_results, "i8 batch")?;
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity_and_hasher(num_results, Default::default());
    for _ in 0..num_results {
        checked_end(offset, 12, body.len(), "i8 batch result header")?;
        let layer = read_u32(body, offset, "i8 batch layer")? as usize;
        let seq_len = read_u32(body, offset + 4, "i8 batch sequence length")? as usize;
        let seq_len = seq_len.max(1);
        let num_floats = read_u32(body, offset + 8, "i8 batch output length")? as usize;
        let expected_floats = checked_mul(seq_len, hidden_size, "i8 batch output")?;
        if num_floats != expected_floats {
            return Err(format!(
                "i8 batch: layer {layer} declared {num_floats} floats, expected {expected_floats}"
            ));
        }
        offset += 12;
        let mut all_floats = Vec::with_capacity(num_floats);
        for _ in 0..seq_len {
            let (pos_floats, next_offset) = decode_i8_position(body, offset, hidden_size)?;
            all_floats.extend(pos_floats);
            offset = next_offset;
        }
        out.insert(layer, all_floats);
    }
    Ok(out)
}

// ── Wire format axis ─────────────────────────────────────────────────────────

/// One direction's binary dtype on the dense walk-ffn wire (ADR-0009 +
/// DEC funnel v0.5 §3 DEC-1A asymmetric direction codecs). The inbound
/// residual format is declared by the request `Content-Type`; the return
/// format by `Accept` — the two are independent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WireFormat {
    /// `application/x-larql-ffn` — f32 LE (the historical wire, default).
    #[default]
    F32,
    /// `application/x-larql-ffn-f16` — IEEE half per value.
    F16,
    /// `application/x-larql-ffn-i8` — per-position symmetric i8 blocks.
    I8,
}

impl WireFormat {
    /// The content-type string for this format.
    pub fn content_type(self) -> &'static str {
        match self {
            Self::F32 => BINARY_CT,
            Self::F16 => F16_CT,
            Self::I8 => I8_CT,
        }
    }

    /// Short label (`"f32"`, `"f16"`, `"i8"`).
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::I8 => "i8",
        }
    }

    /// Parse from a short label. Inverse of [`Self::label`].
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "i8" => Some(Self::I8),
            _ => None,
        }
    }

    /// Match a `Content-Type` header value to a wire format.
    ///
    /// ORDER MATTERS: [`BINARY_CT`] is a substring of both [`F16_CT`] and
    /// [`I8_CT`], so the suffixed types must be checked first — a plain
    /// `contains(BINARY_CT)` check would silently misread an f16/i8 body
    /// as f32. Uses `contains` so parameterised types (`…; v=2`) match.
    pub fn from_content_type(ct: &str) -> Option<Self> {
        if ct.contains(I8_CT) {
            Some(Self::I8)
        } else if ct.contains(F16_CT) {
            Some(Self::F16)
        } else if ct.contains(BINARY_CT) {
            Some(Self::F32)
        } else {
            None
        }
    }
}

/// Decode any single-layer binary walk-ffn response by content type.
///
/// Dispatches to the f32/f16/i8 decoder based on `content_type` and also
/// extracts the server-side `latency_ms` embedded at bytes 8-11.
/// Returns `(layer, server_latency_ms, output_f32)`.
pub fn decode_single_response(
    content_type: &str,
    body: &[u8],
    hidden_size: usize,
) -> Result<(usize, f64, Vec<f32>), String> {
    let server_ms = extract_response_latency_ms(body);
    let (layer, floats) = match content_type {
        I8_CT => decode_binary_single_i8(body, hidden_size)?,
        F16_CT => decode_binary_single_f16(body)?,
        BINARY_CT => decode_binary_single(body)?,
        other => {
            return Err(format!(
                "unsupported walk-ffn response content-type: {other}"
            ))
        }
    };
    Ok((layer, server_ms, floats))
}

/// Extract the `latency_ms` f32 embedded at bytes 8-11 of a binary response.
/// Returns 0.0 if the body is too short or the value is non-finite.
pub(super) fn extract_response_latency_ms(body: &[u8]) -> f64 {
    if body.len() < RESPONSE_HEADER_LEN {
        return 0.0;
    }
    // Both single-layer and batch responses have latency_ms at offset 8.
    let v = f32::from_le_bytes(body[8..12].try_into().unwrap());
    if v.is_finite() {
        v as f64
    } else {
        0.0
    }
}

// ── Server-side half: request decode + response encode ───────────────────────
//
// Moved here from `larql-server/src/routes/walk_ffn/binary.rs` (ROADMAP
// hardening item 16) so the encoder and decoder of each wire direction
// live in one module. The server wraps these in its own error/request
// types; the byte layout and every length guard are pinned by the
// round-trip + rejection tests below.

/// A decoded binary walk-ffn request — the server-side inverse of
/// [`encode_binary_request`]. Pure wire data; the server layers its
/// JSON-request semantics (e.g. `moe_layer`, serde defaults) on top.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFfnRequest {
    /// Single-layer mode. Mutually exclusive with `layers`.
    pub layer: Option<usize>,
    /// Batch mode — multiple layers in one request.
    pub layers: Option<Vec<usize>>,
    /// Row-major flat residual, `seq_len × hidden` floats.
    pub residual: Vec<f32>,
    pub seq_len: usize,
    pub top_k: usize,
    pub full_output: bool,
}

/// Parsed request header — everything before the residual payload. Shared
/// by the f32/f16/i8 request decoders (the header layout is format-invariant).
struct RequestHeader {
    layer: Option<usize>,
    layers: Option<Vec<usize>>,
    seq_len: usize,
    top_k: usize,
    full_output: bool,
    /// Offset of the first residual payload byte.
    payload_offset: usize,
}

/// Parse the shared request header (inverse of [`push_request_header`]),
/// with the same truncation and alloc-bomb guards as the original f32-only
/// decoder (PR 104 lineage).
fn parse_request_header(body: &[u8]) -> Result<RequestHeader, String> {
    if body.len() < REQUEST_HEADER_LEN {
        return Err("binary: body too short (need ≥ 16 bytes)".into());
    }

    let first = u32::from_le_bytes(body[0..4].try_into().unwrap());

    let (layer, layers, header_end) = if first == BATCH_MARKER {
        let n = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
        let layers_end = checked_mul(n, 4, "binary batch layer indices")?
            .checked_add(8)
            .ok_or_else(|| "binary batch layer indices: byte range overflow".to_string())?;
        if body.len() < layers_end {
            return Err(format!(
                "binary batch: body too short for {n} layer indices"
            ));
        }
        let layers: Vec<usize> = (0..n)
            .map(|i| u32::from_le_bytes(body[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize)
            .collect();
        (None, Some(layers), layers_end)
    } else {
        (Some(first as usize), None, 4)
    };

    if body.len() < header_end + 12 {
        return Err("binary: truncated fixed header (seq_len/flags/top_k)".into());
    }
    let seq_len = u32::from_le_bytes(body[header_end..header_end + 4].try_into().unwrap()) as usize;
    let flags = u32::from_le_bytes(body[header_end + 4..header_end + 8].try_into().unwrap());
    let top_k =
        u32::from_le_bytes(body[header_end + 8..header_end + 12].try_into().unwrap()) as usize;

    Ok(RequestHeader {
        layer,
        layers,
        seq_len,
        top_k,
        full_output: (flags & 1) != 0,
        payload_offset: header_end + 12,
    })
}

impl RequestHeader {
    fn into_request(self, residual: Vec<f32>) -> DecodedFfnRequest {
        DecodedFfnRequest {
            layer: self.layer,
            layers: self.layers,
            residual,
            seq_len: self.seq_len,
            top_k: self.top_k,
            full_output: self.full_output,
        }
    }
}

/// Decode a binary-format request body (inverse of [`encode_binary_request`]).
pub fn decode_binary_request(body: &[u8]) -> Result<DecodedFfnRequest, String> {
    let header = parse_request_header(body)?;
    let residual_bytes = &body[header.payload_offset..];
    if !residual_bytes.len().is_multiple_of(4) {
        return Err("binary: residual byte length is not a multiple of 4".into());
    }
    let residual: Vec<f32> = residual_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(header.into_request(residual))
}

/// Decode an f16-format request body (inverse of
/// [`encode_binary_request_as`] with [`WireFormat::F16`]). The residual is
/// widened to f32.
pub fn decode_binary_request_f16(body: &[u8]) -> Result<DecodedFfnRequest, String> {
    use half::f16;
    let header = parse_request_header(body)?;
    let residual_bytes = &body[header.payload_offset..];
    if !residual_bytes.len().is_multiple_of(2) {
        return Err("f16 binary request: residual byte length is not a multiple of f16".into());
    }
    let residual: Vec<f32> = residual_bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect();
    Ok(header.into_request(residual))
}

/// Decode an i8-format request body (inverse of
/// [`encode_binary_request_as`] with [`WireFormat::I8`]). The per-position
/// hidden size is self-describing: `payload_len / max(seq_len, 1) − 8`
/// (each position carries an 8-byte scale/zero header) — validated against
/// the model's hidden size downstream like every other request.
pub fn decode_binary_request_i8(body: &[u8]) -> Result<DecodedFfnRequest, String> {
    let header = parse_request_header(body)?;
    let residual_bytes = &body[header.payload_offset..];
    if residual_bytes.is_empty() {
        return Ok(header.into_request(Vec::new()));
    }
    let seq = header.seq_len.max(1);
    if !residual_bytes.len().is_multiple_of(seq) {
        return Err(format!(
            "i8 binary request: payload of {} bytes is not a multiple of seq_len {seq}",
            residual_bytes.len()
        ));
    }
    let per_pos = residual_bytes.len() / seq;
    if per_pos <= 8 {
        return Err(format!(
            "i8 binary request: {per_pos} bytes per position leaves no residual data \
             after the 8-byte scale/zero header"
        ));
    }
    let hidden = per_pos - 8;
    let mut residual = Vec::with_capacity(checked_mul(seq, hidden, "i8 request residual")?);
    let mut offset = 0usize;
    for _ in 0..seq {
        let (pos_floats, next_offset) = decode_i8_position(residual_bytes, offset, hidden)?;
        residual.extend(pos_floats);
        offset = next_offset;
    }
    Ok(header.into_request(residual))
}

/// Decode a request body whose residual payload is in `format` — the
/// server-side inbound dispatch twin of [`encode_binary_request_as`].
pub fn decode_binary_request_as(
    format: WireFormat,
    body: &[u8],
) -> Result<DecodedFfnRequest, String> {
    match format {
        WireFormat::F32 => decode_binary_request(body),
        WireFormat::F16 => decode_binary_request_f16(body),
        WireFormat::I8 => decode_binary_request_i8(body),
    }
}

/// One layer's FFN output (server-side response payload).
#[derive(Debug, Clone, PartialEq)]
pub struct FfnEntry {
    pub layer: usize,
    pub output: Vec<f32>,
}

/// Typed server response consumed by all four response encoders
/// (f32/f16/i8 binary + JSON).
#[derive(Debug, Clone, PartialEq)]
pub struct FfnOutput {
    pub entries: Vec<FfnEntry>,
    pub seq_len: usize,
    pub latency_ms: f64,
}

/// Encode an [`FfnOutput`] as the binary response format.
pub fn encode_binary_output(out: &FfnOutput) -> Vec<u8> {
    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + entry.output.len() * 4);
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for &v in &entry.output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    } else {
        let num = out.entries.len();
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + num * 12);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for &v in &entry.output {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] using f16 values for the residual/output arrays.
///
/// Wire layout: identical to the f32 format except every float in the output
/// arrays is a `u16` LE (IEEE 754 half-precision). Header fields (layer,
/// seq_len, latency_ms) remain f32/u32 LE. See ADR-0009.
pub fn encode_binary_output_f16(out: &FfnOutput) -> Vec<u8> {
    use half::f16;
    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + entry.output.len() * 2);
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for &v in &entry.output {
            buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        buf
    } else {
        let num = out.entries.len();
        let total_floats: usize = out.entries.iter().map(|e| e.output.len()).sum();
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + num * 12 + total_floats * 2);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for &v in &entry.output {
                buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] using i8 symmetric quantisation (ADR-0009).
///
/// Per position: `[scale f32 LE][zero_point f32 LE][data i8[hidden_size]]`.
/// `scale = max(|x|) / 127.0`, `zero_point = 0.0` (symmetric).
/// Header fields (layer, seq_len, latency_ms) remain f32/u32 LE.
pub fn encode_binary_output_i8(out: &FfnOutput) -> Vec<u8> {
    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let seq = out.seq_len.max(1);
        let hidden = entry.output.len() / seq;
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + seq * (8 + hidden));
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for pos in 0..seq {
            quantise_i8_position(&entry.output[pos * hidden..(pos + 1) * hidden], &mut buf);
        }
        buf
    } else {
        let num = out.entries.len();
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + num * 16);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            let seq = out.seq_len.max(1);
            let hidden = entry.output.len() / seq;
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for pos in 0..seq {
                quantise_i8_position(&entry.output[pos * hidden..(pos + 1) * hidden], &mut buf);
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] as the existing JSON response format (unchanged
/// wire contract for JSON clients).
pub fn encode_json_full_output(out: &FfnOutput) -> serde_json::Value {
    let latency_rounded = (out.latency_ms * 10.0).round() / 10.0;
    if out.entries.len() == 1 {
        let e = &out.entries[0];
        serde_json::json!({
            "layer": e.layer,
            "output": e.output,
            "seq_len": out.seq_len,
            "latency_ms": latency_rounded,
        })
    } else {
        let results: Vec<serde_json::Value> = out
            .entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "layer": e.layer,
                    "output": e.output,
                    "seq_len": out.seq_len,
                })
            })
            .collect();
        serde_json::json!({
            "results": results,
            "seq_len": out.seq_len,
            "latency_ms": latency_rounded,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
        let body =
            encode_binary_request_as(WireFormat::F16, Some(7), None, &residual, 1, true, 256);
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
        let encoded =
            encode_binary_request_as(WireFormat::I8, Some(5), None, &residual, 2, true, 0);
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
        let encoded =
            encode_binary_request_as(WireFormat::I8, None, Some(&[0, 9]), &row, 1, true, 0);
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
}
