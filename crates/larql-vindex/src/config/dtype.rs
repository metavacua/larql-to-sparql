//! Data type conversion utilities for vindex storage.
//!
//! Supports f32 (default) and f16 (half precision) storage.
//! Half-precision conversion functions are in `larql_models::quant::half`.

use serde::{Deserialize, Serialize};

/// Storage precision for vindex binary files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum StorageDtype {
    #[default]
    F32,
    F16,
}


impl std::fmt::Display for StorageDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
        }
    }
}

/// Write `data` to `w`, encoded according to `dtype`. Returns bytes written.
///
/// Convenience wrapper around `encode_floats` for the binary writers in
/// `extract::build`, `extract::streaming`, and `format::weights::write` —
/// they all need the same f32→bytes encode + write + length-tracking
/// pattern.
pub fn write_floats(
    w: &mut impl std::io::Write,
    data: &[f32],
    dtype: StorageDtype,
) -> std::io::Result<u64> {
    let bytes = encode_floats(data, dtype);
    w.write_all(&bytes)?;
    Ok(bytes.len() as u64)
}

/// Encode f32 data as either f32 or f16 bytes.
pub fn encode_floats(data: &[f32], dtype: StorageDtype) -> Vec<u8> {
    match dtype {
        StorageDtype::F32 => {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            bytes.to_vec()
        }
        StorageDtype::F16 => larql_models::quant::half::encode_f16(data),
    }
}

/// Decode bytes back to f32, handling dtype.
pub fn decode_floats(data: &[u8], dtype: StorageDtype) -> Vec<f32> {
    match dtype {
        StorageDtype::F32 => {
            let floats: &[f32] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
            };
            floats.to_vec()
        }
        StorageDtype::F16 => larql_models::quant::half::decode_f16(data),
    }
}

/// Bytes per float for a given dtype.
pub fn bytes_per_float(dtype: StorageDtype) -> usize {
    match dtype {
        StorageDtype::F32 => 4,
        StorageDtype::F16 => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_f32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let encoded = encode_floats(&data, StorageDtype::F32);
        assert_eq!(encoded.len(), 12);
        let decoded = decode_floats(&encoded, StorageDtype::F32);
        assert_eq!(decoded, data);
    }

    #[test]
    fn encode_decode_f16() {
        let data = vec![1.0f32, 2.0, 3.0];
        let encoded = encode_floats(&data, StorageDtype::F16);
        assert_eq!(encoded.len(), 6);
        let decoded = decode_floats(&encoded, StorageDtype::F16);
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01);
        }
    }
}
