//! Pure routing helpers — architecture-independent, compiles to wasm32.
//!
//! Extracted from main.rs so that wasm-pack can test these on wasm32.

/// Sentinel value that marks the start of a batch binary request header.
pub const BATCH_MARKER: u32 = 0xFFFF_FFFF;

/// A statically configured layer shard.
#[derive(Clone, Debug)]
pub struct Shard {
    pub layer_start: usize, // inclusive
    pub layer_end: usize,   // exclusive
    pub url: String,
}

impl Shard {
    pub fn owns(&self, layer: usize) -> bool {
        layer >= self.layer_start && layer < self.layer_end
    }
}

/// Parse a shard spec string into a list of [`Shard`]s.
///
/// Format: `"START-END=URL"` entries separated by commas (inclusive bounds).
/// Example: `"0-16=http://host-a:8080,17-33=http://host-b:8081"`
pub fn parse_shards(spec: &str) -> Result<Vec<Shard>, String> {
    let mut shards = Vec::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (range, url) = entry
            .split_once('=')
            .ok_or_else(|| format!("expected 'START-END=URL', got '{entry}'"))?;
        let (start_s, end_s) = range
            .split_once('-')
            .ok_or_else(|| format!("expected 'START-END', got '{range}'"))?;
        let start: usize = start_s
            .trim()
            .parse()
            .map_err(|_| format!("invalid start '{start_s}'"))?;
        let end: usize = end_s
            .trim()
            .parse()
            .map_err(|_| format!("invalid end '{end_s}'"))?;
        if end < start {
            return Err(format!("end ({end}) must be >= start ({start})"));
        }
        shards.push(Shard {
            layer_start: start,
            layer_end: end + 1,
            url: url.trim().to_string(),
        });
    }
    if shards.is_empty() {
        return Err("no shards specified".into());
    }
    Ok(shards)
}

/// Extract layer indices from a binary request body without parsing the residual.
///
/// Returns `None` if the header is malformed or truncated.
pub fn peek_binary(body: &[u8]) -> Option<Vec<usize>> {
    if body.len() < 4 {
        return None;
    }
    let first = u32::from_le_bytes(body[0..4].try_into().ok()?);
    if first == BATCH_MARKER {
        if body.len() < 8 {
            return None;
        }
        let n = u32::from_le_bytes(body[4..8].try_into().ok()?) as usize;
        let needed = 8 + n * 4;
        if body.len() < needed {
            return None;
        }
        let layers = (0..n)
            .map(|i| u32::from_le_bytes(body[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize)
            .collect();
        Some(layers)
    } else {
        Some(vec![first as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_arch = "wasm32", feature = "browser-tests"))]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    fn make_binary_single(layer: u32, residual_floats: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        buf.extend_from_slice(&1u32.to_le_bytes()); // flags (full_output)
        buf.extend_from_slice(&8092u32.to_le_bytes()); // top_k
        buf.extend(std::iter::repeat_n(0u8, residual_floats * 4));
        buf
    }

    fn make_binary_batch(layers: &[u32], residual_floats: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(layers.len() as u32).to_le_bytes());
        for &l in layers {
            buf.extend_from_slice(&l.to_le_bytes());
        }
        buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        buf.extend_from_slice(&1u32.to_le_bytes()); // flags
        buf.extend_from_slice(&8092u32.to_le_bytes()); // top_k
        buf.extend(std::iter::repeat_n(0u8, residual_floats * 4));
        buf
    }

    // ── peek_binary ───────────────────────────────────────────────────────────

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn peek_binary_single_layer() {
        let body = make_binary_single(5, 4);
        let layers = peek_binary(&body).unwrap();
        assert_eq!(layers, vec![5]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn peek_binary_batch_layers() {
        let body = make_binary_batch(&[5, 20, 30], 4);
        let layers = peek_binary(&body).unwrap();
        assert_eq!(layers, vec![5, 20, 30]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn peek_binary_empty_body_returns_none() {
        assert!(peek_binary(&[]).is_none());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn peek_binary_truncated_single_returns_value() {
        // Only 4 bytes — enough for a single-layer marker.
        let buf = 7u32.to_le_bytes();
        let layers = peek_binary(&buf).unwrap();
        assert_eq!(layers, vec![7]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn peek_binary_batch_truncated_layer_list_returns_none() {
        // Claims 10 layers but only provides 2 u32s after num_layers.
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&10u32.to_le_bytes()); // num_layers = 10
        buf.extend_from_slice(&0u32.to_le_bytes()); // layer 0
        buf.extend_from_slice(&1u32.to_le_bytes()); // layer 1 — only 2 of 10
        assert!(peek_binary(&buf).is_none());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn peek_binary_zero_batch_layers() {
        let body = make_binary_batch(&[], 4);
        let layers = peek_binary(&body).unwrap();
        assert!(layers.is_empty());
    }

    // ── parse_shards ──────────────────────────────────────────────────────────

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn parse_shards_single_entry() {
        let shards = parse_shards("0-16=http://host-a:8080").unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].layer_start, 0);
        assert_eq!(shards[0].layer_end, 17); // exclusive
        assert_eq!(shards[0].url, "http://host-a:8080");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn parse_shards_two_entries() {
        let shards = parse_shards("0-16=http://host-a:8080,17-33=http://host-b:8081").unwrap();
        assert_eq!(shards.len(), 2);
        assert!(shards[0].owns(0));
        assert!(shards[0].owns(16));
        assert!(!shards[0].owns(17));
        assert!(shards[1].owns(17));
        assert!(shards[1].owns(33));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn parse_shards_empty_string_errors() {
        assert!(parse_shards("").is_err());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn parse_shards_missing_url_errors() {
        assert!(parse_shards("0-16").is_err());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn parse_shards_end_less_than_start_errors() {
        assert!(parse_shards("16-0=http://host:8080").is_err());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn parse_shards_ignores_trailing_comma() {
        let shards = parse_shards("0-16=http://host:8080,").unwrap();
        assert_eq!(shards.len(), 1);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn shard_owns_inclusive_bounds() {
        let shards = parse_shards("0-16=http://host:8080").unwrap();
        assert!(shards[0].owns(0));
        assert!(shards[0].owns(16));
        assert!(!shards[0].owns(17));
    }
}
