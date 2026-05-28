//! Float decoding for gate vector blobs (f32 and f16 wire formats).

use alloc::vec::Vec;

/// Storage precision of gate vectors in the index blob.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum StorageDtype {
    #[default]
    F32,
    F16,
}

/// Decode raw gate vector bytes into f32 values.
pub fn decode_floats(data: &[u8], dtype: StorageDtype) -> Vec<f32> {
    match dtype {
        StorageDtype::F32 => {
            let n = data.len() / 4;
            (0..n)
                .map(|i| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap()))
                .collect()
        }
        StorageDtype::F16 => {
            let n = data.len() / 2;
            (0..n)
                .map(|i| {
                    let bytes: [u8; 2] = data[i * 2..i * 2 + 2].try_into().unwrap();
                    half::f16::from_le_bytes(bytes).to_f32()
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_f32_roundtrip() {
        let data = alloc::vec![1.0f32, -2.5, 3.14];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let decoded = decode_floats(&bytes, StorageDtype::F32);
        assert_eq!(decoded.len(), 3);
        assert!((decoded[0] - 1.0).abs() < 1e-6);
        assert!((decoded[1] - (-2.5)).abs() < 1e-6);
    }

    #[test]
    fn decode_f16_roundtrip() {
        let orig = 1.5f32;
        let half_bytes = half::f16::from_f32(orig).to_le_bytes();
        let decoded = decode_floats(&half_bytes, StorageDtype::F16);
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - orig).abs() < 0.01);
    }
}
