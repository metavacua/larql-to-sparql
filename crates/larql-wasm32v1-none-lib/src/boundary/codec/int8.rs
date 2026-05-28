//! Per-vector σ-clipped int8 residual codec (`int8_clip3sigma`).

use alloc::vec::Vec;

pub const SCALE_BYTES: usize = 4;
pub const CLIP_SIGMA: f32 = 3.0;
pub const INT8_QMAX: f32 = 127.0;

pub struct Payload {
    pub scale: f32,
    pub quantized: Vec<i8>,
}

impl Payload {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SCALE_BYTES + self.quantized.len());
        out.extend_from_slice(&self.scale.to_le_bytes());
        out.extend(self.quantized.iter().map(|&v| v as u8));
        out
    }

    /// # Panics
    /// Panics if `bytes.len() < SCALE_BYTES`.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(
            bytes.len() >= SCALE_BYTES,
            "int8_clip3sigma payload is too short (got {} bytes)",
            bytes.len()
        );
        let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let quantized: Vec<i8> = bytes[SCALE_BYTES..].iter().map(|&b| b as i8).collect();
        Self { scale, quantized }
    }
}

pub fn encode(r: &[f32]) -> Payload {
    let sigma = std_dev(r);
    let clip = if sigma > 0.0 {
        CLIP_SIGMA * sigma
    } else {
        1.0
    };
    let scale = clip / INT8_QMAX;

    let quantized = r
        .iter()
        .map(|&v| {
            let clipped = v.clamp(-clip, clip);
            libm::roundf(clipped / scale).clamp(-INT8_QMAX, INT8_QMAX) as i8
        })
        .collect();

    Payload { scale, quantized }
}

pub fn decode(payload: &Payload) -> Vec<f32> {
    payload
        .quantized
        .iter()
        .map(|&q| q as f32 * payload.scale)
        .collect()
}

fn std_dev(r: &[f32]) -> f32 {
    let n = r.len() as f64;
    let mean: f64 = r.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var: f64 = r.iter().map(|&v| { let d = v as f64 - mean; d * d }).sum::<f64>() / n;
    libm::sqrt(var) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_small_vector() {
        let r: Vec<f32> = (0..256).map(|i| i as f32 * 0.5 - 64.0).collect();
        let p = encode(&r);
        let dec = decode(&p);
        let mse: f32 = r
            .iter()
            .zip(dec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / r.len() as f32;
        assert!(mse < 50.0, "MSE too high: {mse}");
    }

    #[test]
    fn outlier_is_clipped_not_nan() {
        let mut r = alloc::vec![0.5f32; 2560];
        r[100] = 150_000.0;
        r[500] = -94_208.0;
        let p = encode(&r);
        let dec = decode(&p);
        for (i, v) in dec.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {i}");
        }
    }

    #[test]
    fn payload_byte_length() {
        let r = alloc::vec![0.0f32; 2560];
        assert_eq!(encode(&r).to_bytes().len(), SCALE_BYTES + 2560);
    }

    #[test]
    fn bytes_roundtrip() {
        let r: Vec<f32> = (0..100).map(|i| i as f32 - 50.0).collect();
        let p = encode(&r);
        let bytes = p.to_bytes();
        let p2 = Payload::from_bytes(&bytes);
        let dec = decode(&p2);
        let mse: f32 = r
            .iter()
            .zip(dec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / r.len() as f32;
        assert!(mse < 50.0, "bytes-roundtrip MSE too high: {mse}");
    }

    #[test]
    fn constant_vector_does_not_panic() {
        let r = alloc::vec![5.0f32; 64];
        let p = encode(&r);
        let dec = decode(&p);
        assert_eq!(dec.len(), 64);
    }
}
