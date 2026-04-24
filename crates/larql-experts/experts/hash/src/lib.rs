//! # Hash / encoding expert
//!
//! Reversible byte encodings (base64, hex, URL percent) and a simple non-
//! cryptographic hash (FNV-1a 32-bit). All string inputs are treated as UTF-8
//! bytes; decoders return the decoded string (caller's responsibility to
//! handle non-UTF-8 input, which is rejected via `None`).
//!
//! ## Ops
//!
//! - `base64_encode {s} → string`
//! - `base64_decode {s} → string`
//! - `hex_encode {s} → string`
//! - `hex_decode {s} → string` (accepts optional `0x` / `0X` prefix)
//! - `url_encode {s} → string` (percent-encodes everything except unreserved)
//! - `url_decode {s} → string` (`+` → space)
//! - `fnv1a_32 {s} → string` (lowercase `0x`-prefixed 8-hex-digit form)

use expert_interface::{arg_str, expert_exports, json, Value};

expert_exports!(
    id = "hash",
    tier = 1,
    description = "Encoding: base64, hex, url (percent), FNV-1a 32-bit hash",
    version = "0.2.0",
    ops = [
        ("base64_encode", ["s"]),
        ("base64_decode", ["s"]),
        ("hex_encode",    ["s"]),
        ("hex_decode",    ["s"]),
        ("url_encode",    ["s"]),
        ("url_decode",    ["s"]),
        ("fnv1a_32",      ["s"]),
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "base64_encode" => Some(json!(base64_encode(arg_str(args, "s")?.as_bytes()))),
        "base64_decode" => {
            let bytes = base64_decode(arg_str(args, "s")?)?;
            Some(json!(String::from_utf8(bytes).ok()?))
        }
        "hex_encode" => Some(json!(hex_encode(arg_str(args, "s")?.as_bytes()))),
        "hex_decode" => {
            let bytes = hex_decode(arg_str(args, "s")?)?;
            Some(json!(String::from_utf8(bytes).ok()?))
        }
        "url_encode" => Some(json!(url_encode(arg_str(args, "s")?))),
        "url_decode" => Some(json!(url_decode(arg_str(args, "s")?))),
        "fnv1a_32" => Some(json!(format!("0x{:08x}", fnv1a_32(arg_str(args, "s")?.as_bytes())))),
        _ => None,
    }
}

const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(input: &[u8]) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < input.len() {
        let b0 = input[i];
        let b1 = if i + 1 < input.len() { input[i + 1] } else { 0 };
        let b2 = if i + 2 < input.len() { input[i + 2] } else { 0 };

        out.push(B64[(b0 >> 2) as usize]);
        out.push(B64[((b0 & 0x03) << 4 | b1 >> 4) as usize]);
        out.push(if i + 1 < input.len() { B64[((b1 & 0x0f) << 2 | b2 >> 6) as usize] } else { b'=' });
        out.push(if i + 2 < input.len() { B64[(b2 & 0x3f) as usize] } else { b'=' });
        i += 3;
    }
    String::from_utf8(out).unwrap_or_default()
}

fn b64_val(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        b'=' => Some(0),
        _ => None,
    }
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let bytes: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !bytes.len().is_multiple_of(4) { return None; }
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let v0 = b64_val(bytes[i])?;
        let v1 = b64_val(bytes[i + 1])?;
        let v2 = b64_val(bytes[i + 2])?;
        let v3 = b64_val(bytes[i + 3])?;
        out.push((v0 << 2) | (v1 >> 4));
        if bytes[i + 2] != b'=' { out.push((v1 << 4) | (v2 >> 2)); }
        if bytes[i + 3] != b'=' { out.push((v2 << 6) | v3); }
        i += 4;
    }
    Some(out)
}

fn hex_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len() * 2);
    for &b in input {
        out.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        out.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    out
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim().trim_start_matches("0x").trim_start_matches("0X");
    if !s.len().is_multiple_of(2) { return None; }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        out.push((hex_nibble(bytes[i])? << 4) | hex_nibble(bytes[i + 1])?);
        i += 2;
    }
    Some(out)
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(char::from_digit((b >> 4) as u32, 16).unwrap());
            out.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
        }
    }
    out
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' { out.push(b' '); } else { out.push(bytes[i]); }
        i += 1;
    }
    String::from_utf8(out).unwrap_or_default()
}

fn fnv1a_32(input: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in input {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}
