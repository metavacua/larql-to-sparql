//! Minimal numpy `.npy` v1.0 reader for the dtypes the Apollo store uses.
//!
//! We avoid `ndarray-npy` because it depends on ndarray 0.17 while the
//! workspace pins 0.16. The format is simple enough to parse directly:
//!
//! ```text
//! 6 bytes  magic        "\x93NUMPY"
//! 2 bytes  version      0x01 0x00   (v1.0; v2.0 uses u32 header_len)
//! 2 bytes  header_len   u16 little-endian
//! N bytes  header       ASCII Python dict literal
//! remaining data        row-major contiguous, little-endian
//! ```
//!
//! Supported dtype strings (only what apollo11_store uses):
//!   - `'<f4'` → f32
//!   - `'<u4'` → u32
//!   - structured dtypes are parsed by the `apollo::store` module directly.

#[derive(Debug)]
pub struct NpyHeader {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum NpyError {
    #[error("file is not a valid .npy (bad magic)")]
    BadMagic,
    #[error("unsupported .npy version {0}.{1} (need 1.x)")]
    UnsupportedVersion(u8, u8),
    #[error("truncated .npy header")]
    TruncatedHeader,
    #[error("header is not valid UTF-8: {0}")]
    InvalidUtf8(std::str::Utf8Error),
    #[error("could not parse header field '{field}' from: {snippet}")]
    ParseField { field: &'static str, snippet: String },
    #[error("dtype mismatch: expected {expected}, got {actual}")]
    DtypeMismatch { expected: &'static str, actual: String },
    #[error("data length {got} does not match expected {expected} ({shape:?} × {stride} bytes)")]
    DataLength {
        got: usize,
        expected: usize,
        shape: Vec<usize>,
        stride: usize,
    },
    #[error("fortran-order arrays are not supported")]
    FortranOrder,
}

/// Parse the `.npy` header. Returns `(header, data_offset)` where `data_offset`
/// is the byte index at which raw array data begins.
pub fn parse_header(bytes: &[u8]) -> Result<(NpyHeader, usize), NpyError> {
    if bytes.len() < 10 {
        return Err(NpyError::TruncatedHeader);
    }
    if &bytes[..6] != b"\x93NUMPY" {
        return Err(NpyError::BadMagic);
    }
    let major = bytes[6];
    let minor = bytes[7];
    if major != 1 {
        return Err(NpyError::UnsupportedVersion(major, minor));
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_end = 10 + header_len;
    if bytes.len() < header_end {
        return Err(NpyError::TruncatedHeader);
    }
    let header_str =
        std::str::from_utf8(&bytes[10..header_end]).map_err(NpyError::InvalidUtf8)?;
    // `descr` may be either a quoted string (simple dtype like '<f4') or a
    // Python list literal (structured dtype like `[('token_id', '<u4'), ...]`).
    // Extract as raw text so both cases succeed.
    let descr = parse_field_value(header_str, "descr").ok_or_else(|| NpyError::ParseField {
        field: "descr",
        snippet: header_str.to_string(),
    })?;
    let fortran = parse_bool_field(header_str, "fortran_order").ok_or_else(|| {
        NpyError::ParseField {
            field: "fortran_order",
            snippet: header_str.to_string(),
        }
    })?;
    if fortran {
        return Err(NpyError::FortranOrder);
    }
    let shape = parse_shape(header_str).ok_or_else(|| NpyError::ParseField {
        field: "shape",
        snippet: header_str.to_string(),
    })?;
    Ok((
        NpyHeader {
            descr,
            fortran_order: fortran,
            shape,
        },
        header_end,
    ))
}

/// Read an `<f4` 1D array from .npy bytes.
pub fn read_f32_1d(bytes: &[u8]) -> Result<Vec<f32>, NpyError> {
    let (header, data_off) = parse_header(bytes)?;
    check_dtype(&header.descr, "<f4")?;
    if header.shape.len() != 1 {
        return Err(NpyError::ParseField {
            field: "shape",
            snippet: format!("expected 1D, got {:?}", header.shape),
        });
    }
    let n = header.shape[0];
    let data = &bytes[data_off..];
    let expected = n * 4;
    if data.len() != expected {
        return Err(NpyError::DataLength {
            got: data.len(),
            expected,
            shape: header.shape.clone(),
            stride: 4,
        });
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 4;
        out.push(f32::from_le_bytes([
            data[o],
            data[o + 1],
            data[o + 2],
            data[o + 3],
        ]));
    }
    Ok(out)
}

/// Read an `<f4` multi-D array as a flat Vec (row-major) plus shape.
pub fn read_f32_flat(bytes: &[u8]) -> Result<(Vec<f32>, Vec<usize>), NpyError> {
    let (header, data_off) = parse_header(bytes)?;
    check_dtype(&header.descr, "<f4")?;
    let n: usize = header.shape.iter().product();
    let data = &bytes[data_off..];
    let expected = n * 4;
    if data.len() != expected {
        return Err(NpyError::DataLength {
            got: data.len(),
            expected,
            shape: header.shape.clone(),
            stride: 4,
        });
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 4;
        out.push(f32::from_le_bytes([
            data[o],
            data[o + 1],
            data[o + 2],
            data[o + 3],
        ]));
    }
    Ok((out, header.shape))
}

/// Read an `<u4` 1D array from .npy bytes.
pub fn read_u32_1d(bytes: &[u8]) -> Result<Vec<u32>, NpyError> {
    let (header, data_off) = parse_header(bytes)?;
    check_dtype(&header.descr, "<u4")?;
    if header.shape.len() != 1 {
        return Err(NpyError::ParseField {
            field: "shape",
            snippet: format!("expected 1D, got {:?}", header.shape),
        });
    }
    let n = header.shape[0];
    let data = &bytes[data_off..];
    let expected = n * 4;
    if data.len() != expected {
        return Err(NpyError::DataLength {
            got: data.len(),
            expected,
            shape: header.shape.clone(),
            stride: 4,
        });
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 4;
        out.push(u32::from_le_bytes([
            data[o],
            data[o + 1],
            data[o + 2],
            data[o + 3],
        ]));
    }
    Ok(out)
}

// ── header parsing helpers ────────────────────────────────────────────────

fn check_dtype(got: &str, expected: &'static str) -> Result<(), NpyError> {
    if got != expected {
        Err(NpyError::DtypeMismatch {
            expected,
            actual: got.to_string(),
        })
    } else {
        Ok(())
    }
}

fn parse_string_field(header: &str, name: &str) -> Option<String> {
    let needle = format!("'{name}':");
    let start = header.find(&needle)?;
    let after = &header[start + needle.len()..];
    // Skip whitespace and opening quote (either ' or ").
    let after = after.trim_start();
    let (quote, rest) = if let Some(r) = after.strip_prefix('\'') {
        ('\'', r)
    } else if let Some(r) = after.strip_prefix('"') {
        ('"', r)
    } else {
        return None;
    };
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

/// Extract the raw text of a field value. Handles:
///   - quoted strings: `'<f4'` → `<f4`
///   - list literals: `[(...)]` → `[(...)]` (kept as-is for callers to parse)
///   - tuples: `(a, b)` → `(a, b)`
///   - bare tokens: `True` / `False` / numbers → token as-is, trimmed
fn parse_field_value(header: &str, name: &str) -> Option<String> {
    let needle = format!("'{name}':");
    let start = header.find(&needle)?;
    let rest = header[start + needle.len()..].trim_start();
    let mut chars = rest.chars();
    let first = chars.next()?;
    match first {
        '\'' | '"' => {
            // Quoted string — strip the quotes.
            let quote = first;
            let body: String = rest[1..].chars().take_while(|c| *c != quote).collect();
            Some(body)
        }
        '[' | '(' | '{' => {
            // Bracket-delimited — keep the brackets, find matching close.
            let (open, close) = match first {
                '[' => ('[', ']'),
                '(' => ('(', ')'),
                '{' => ('{', '}'),
                _ => unreachable!(),
            };
            let mut depth = 0i32;
            let mut end = 0usize;
            for (i, c) in rest.char_indices() {
                if c == open {
                    depth += 1;
                } else if c == close {
                    depth -= 1;
                    if depth == 0 {
                        end = i + c.len_utf8();
                        break;
                    }
                }
            }
            if end == 0 {
                None
            } else {
                Some(rest[..end].to_string())
            }
        }
        _ => {
            // Bare token up to comma or closing brace.
            let end = rest
                .find(|c: char| c == ',' || c == '}')
                .unwrap_or(rest.len());
            Some(rest[..end].trim().to_string())
        }
    }
}

fn parse_bool_field(header: &str, name: &str) -> Option<bool> {
    let needle = format!("'{name}':");
    let start = header.find(&needle)?;
    let after = header[start + needle.len()..].trim_start();
    if after.starts_with("True") {
        Some(true)
    } else if after.starts_with("False") {
        Some(false)
    } else {
        None
    }
}

fn parse_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape':")?;
    let after = &header[start + "'shape':".len()..];
    let open = after.find('(')?;
    let close = after.find(')')?;
    let inner = &after[open + 1..close];
    let mut out = Vec::new();
    for part in inner.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(trimmed.parse::<usize>().ok()?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal .npy v1.0 blob for an f32 1D array of given values.
    fn synth_f32_1d(values: &[f32]) -> Vec<u8> {
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
            values.len()
        );
        // Pad header to 64-byte alignment (numpy convention).
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        let header_len = padded.len();

        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header_len as u16).to_le_bytes());
        out.extend_from_slice(&padded);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn parse_1d_f32_roundtrip() {
        let vals = [1.0f32, 2.0, 3.0, -4.5, 0.125];
        let blob = synth_f32_1d(&vals);
        let parsed = read_f32_1d(&blob).expect("parse");
        assert_eq!(parsed, vals.to_vec());
    }

    #[test]
    fn parse_string_field_handles_single_quotes() {
        let hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (10,), }";
        assert_eq!(parse_string_field(hdr, "descr"), Some("<f4".into()));
    }

    #[test]
    fn parse_shape_handles_multiple_dims() {
        let hdr = "{'shape': (1, 1, 2560), 'fortran_order': False}";
        assert_eq!(parse_shape(hdr), Some(vec![1, 1, 2560]));
    }

    #[test]
    fn parse_shape_handles_trailing_comma() {
        let hdr = "{'shape': (3585, ), 'fortran_order': False}";
        assert_eq!(parse_shape(hdr), Some(vec![3585]));
    }

    #[test]
    fn dtype_mismatch_reports_what_was_found() {
        let vals = [1.0f32, 2.0];
        let blob = synth_f32_1d(&vals);
        let result = read_u32_1d(&blob);
        let err = result.unwrap_err();
        assert!(matches!(err, NpyError::DtypeMismatch { .. }));
    }
}
