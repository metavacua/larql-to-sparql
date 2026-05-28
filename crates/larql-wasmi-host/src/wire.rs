//! Binary wire-format encoding for the larql-wasm32v1-none ABI.
//!
//! Mirrors the protocol defined in `larql-wasm32v1-none-lib/src/abi.rs`.
//! The encoding is pure little-endian binary; all lengths are u32 LE.

/// Storage dtype for gate vector bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32 = 0,
    F16 = 1,
}

/// One gate layer passed to a `gate_knn` request.
pub struct LayerData<'a> {
    /// Raw gate vector bytes (F32: num_features * hidden_size * 4 bytes).
    pub bytes: &'a [u8],
    pub num_features: u32,
    pub dtype: Dtype,
}

/// Encode a `gate_knn` request.
///
/// * `hidden_size` — per-gate-vector dimension
/// * `layers` — `None` entries are transmitted as empty (has_data = 0)
/// * `query_layer` — which layer to query
/// * `query` — query vector (length should equal `hidden_size`)
/// * `k` — number of nearest neighbours to return
pub fn encode_gate_knn(
    hidden_size: u32,
    layers: &[Option<LayerData<'_>>],
    query_layer: u32,
    query: &[f32],
    k: u32,
) -> Vec<u8> {
    let cap = {
        let mut n = 1 + 4 + 4; // opcode + hidden_size + num_layers
        for layer in layers {
            n += 1; // has_data flag
            if let Some(l) = layer {
                n += 4 + 1 + l.bytes.len(); // num_features + dtype + gate data
            }
        }
        n + 4 + 4 + query.len() * 4 + 4 // query_layer + query_len + query + k
    };
    let mut buf = Vec::with_capacity(cap);
    buf.push(0x04); // opcode
    buf.extend_from_slice(&hidden_size.to_le_bytes());
    buf.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    for layer in layers {
        match layer {
            None => buf.push(0),
            Some(l) => {
                buf.push(1);
                buf.extend_from_slice(&l.num_features.to_le_bytes());
                buf.push(l.dtype as u8);
                buf.extend_from_slice(l.bytes);
            }
        }
    }
    buf.extend_from_slice(&query_layer.to_le_bytes());
    buf.extend_from_slice(&(query.len() as u32).to_le_bytes());
    for x in query {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    buf.extend_from_slice(&k.to_le_bytes());
    buf
}

/// Encode a `dot` request (opcode 0x01).
pub fn encode_dot(a: &[f32], b: &[f32]) -> Vec<u8> {
    let mut buf = vec![0x01u8];
    push_f32_vec(&mut buf, a);
    push_f32_vec(&mut buf, b);
    buf
}

/// Encode a `norm` request (opcode 0x02).
pub fn encode_norm(a: &[f32]) -> Vec<u8> {
    let mut buf = vec![0x02u8];
    push_f32_vec(&mut buf, a);
    buf
}

/// Encode a `cosine` request (opcode 0x03).
pub fn encode_cosine(a: &[f32], b: &[f32]) -> Vec<u8> {
    let mut buf = vec![0x03u8];
    push_f32_vec(&mut buf, a);
    push_f32_vec(&mut buf, b);
    buf
}

fn push_f32_vec(buf: &mut Vec<u8>, v: &[f32]) {
    buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

/// Decode a scalar `f32` response. Returns `Err` if status != 0 or bytes malformed.
pub fn decode_scalar(response: &[u8]) -> Result<f32, String> {
    if response.is_empty() {
        return Err("empty response".into());
    }
    if response[0] != 0 {
        return Err(format!(
            "guest error: {}",
            core::str::from_utf8(&response[1..]).unwrap_or("(non-utf8)")
        ));
    }
    if response.len() < 5 {
        return Err(format!("scalar response too short: {} bytes", response.len()));
    }
    Ok(f32::from_le_bytes(response[1..5].try_into().unwrap()))
}

/// One KNN result.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnResult {
    pub feature: u32,
    pub score: f32,
}

/// Decode a `gate_knn` response.
pub fn decode_gate_knn(response: &[u8]) -> Result<Vec<KnnResult>, String> {
    if response.is_empty() {
        return Err("empty response".into());
    }
    if response[0] != 0 {
        return Err(format!(
            "guest error: {}",
            core::str::from_utf8(&response[1..]).unwrap_or("(non-utf8)")
        ));
    }
    if response.len() < 5 {
        return Err(format!("knn response too short: {} bytes", response.len()));
    }
    let n = u32::from_le_bytes(response[1..5].try_into().unwrap()) as usize;
    let expected_len = 5 + n * 8;
    if response.len() < expected_len {
        return Err(format!(
            "knn response truncated: got {} bytes, need {expected_len}",
            response.len()
        ));
    }
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let off = 5 + i * 8;
        let feature = u32::from_le_bytes(response[off..off + 4].try_into().unwrap());
        let score = f32::from_le_bytes(response[off + 4..off + 8].try_into().unwrap());
        results.push(KnnResult { feature, score });
    }
    Ok(results)
}
