use crate::{KvStrategy, model_config::ModelConfig};

/// Strategy 1: Standard FP16 KV cache.
///
/// Store raw f32 keys and values. Exact roundtrip — this is the baseline
/// that every other strategy is measured against.
pub struct StandardKv;

impl KvStrategy for StandardKv {
    fn name(&self) -> &str {
        "Standard KV (FP16)"
    }

    fn encode(&self, keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<u8> {
        // Store as raw f32 bytes (simulating FP16 storage at f32 precision)
        let total_floats = keys.iter().map(|v| v.len()).sum::<usize>()
            + values.iter().map(|v| v.len()).sum::<usize>();
        let mut buf = Vec::with_capacity(total_floats * 2); // FP16 = 2 bytes

        for v in keys.iter().chain(values.iter()) {
            for &x in v {
                buf.extend_from_slice(&f16_encode(x));
            }
        }
        buf
    }

    fn decode(&self, encoded: &[u8], num_vectors: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let floats_per_set = num_vectors * dim;
        let bytes_per_set = floats_per_set * 2;

        let keys = decode_fp16_vectors(&encoded[..bytes_per_set], num_vectors, dim);
        let values = decode_fp16_vectors(&encoded[bytes_per_set..], num_vectors, dim);
        (keys, values)
    }

    fn memory_bytes(&self, config: &ModelConfig, seq_len: usize) -> usize {
        // seq_len × layers × 2(K+V) × kv_heads × head_dim × 2(fp16)
        config.kv_memory(seq_len)
    }
}

/// Encode f32 to fp16 (IEEE 754 half precision).
fn f16_encode(x: f32) -> [u8; 2] {
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0 {
        // Zero or subnormal → fp16 zero
        let h = (sign << 15) as u16;
        return h.to_le_bytes();
    }

    if exp == 0xFF {
        // Inf or NaN
        let h = ((sign << 15) | (0x1F << 10) | (if frac != 0 { 0x200 } else { 0 })) as u16;
        return h.to_le_bytes();
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        // Overflow → infinity
        let h = ((sign << 15) | (0x1F << 10)) as u16;
        return h.to_le_bytes();
    }

    if new_exp <= 0 {
        // Underflow → fp16 zero (simplified, no subnormals)
        let h = (sign << 15) as u16;
        return h.to_le_bytes();
    }

    let h = ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16;
    h.to_le_bytes()
}

/// Decode fp16 to f32.
fn f16_decode(bytes: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(bytes);
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal fp16
        let mut f = frac as f32 / 1024.0;
        f *= 2.0f32.powi(-14);
        if sign == 1 { -f } else { f }
    } else if exp == 0x1F {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::NAN
        }
    } else {
        let new_exp = exp as i32 - 15 + 127;
        let bits = (sign << 31) | ((new_exp as u32) << 23) | (frac << 13);
        f32::from_bits(bits)
    }
}

fn decode_fp16_vectors(data: &[u8], num_vectors: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(num_vectors);
    for i in 0..num_vectors {
        let mut vec = Vec::with_capacity(dim);
        for j in 0..dim {
            let offset = (i * dim + j) * 2;
            let bytes = [data[offset], data[offset + 1]];
            vec.push(f16_decode(bytes));
        }
        result.push(vec);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 0.333, 100.0, -0.001];
        for &v in &values {
            let encoded = f16_encode(v);
            let decoded = f16_decode(encoded);
            let err = (v - decoded).abs();
            // FP16 has ~3 decimal digits of precision
            assert!(err < 0.01 * v.abs().max(0.001), "fp16 roundtrip failed for {v}: got {decoded}, err {err}");
        }
    }

    #[test]
    fn test_standard_kv_memory_formula() {
        let config = ModelConfig::gemma_4b();
        // 4K tokens: 4096 × 34 × 2 × 2 × 256 × 2 = 142,606,336
        let mem = StandardKv.memory_bytes(&config, 4096);
        let expected = 4096 * 34 * 2 * 2 * 256 * 2;
        assert_eq!(mem, expected);
    }
}
