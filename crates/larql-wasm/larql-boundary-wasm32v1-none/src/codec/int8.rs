//! Per-vector σ-clipped int8 residual codec (`int8_clip3sigma`).
//!
//! # Why clipping, not absmax?
//!
//! Gemma-class residuals have absmax/σ ≈ 92×. Absmax quantisation wastes 99% of
//! the 256 int8 levels on a handful of extreme values. σ-clipping concentrates all
//! levels on the ±3σ band where the prediction-relevant geometry lives.
//!
//! # Characterised contract (Exp 43, 30 prompts, layer 33, Gemma 3 4B)
//!
//! | Metric | Value |
//! |--------|-------|
//! | Top-1 agreement (mean) | 98.7% |
//! | Top-1 agreement (min) | 93.3% |
//! | Top-5 agreement | 100% |
//! | KL divergence (mean) | 2.0 nats |
//! | Contract | D- (ArgmaxNearEquivalent) |
//!
//! # Wire format
//!
//! ```text
//! [f32 scale (4 bytes LE)] ++ [i8 × d]
//! ```
//!
//! For Gemma 3 4B (d = 2560): 4 + 2560 = **2564 bytes** vs 5120 for bf16.
//!
//! `scale = clip / INT8_QMAX`, where `clip = CLIP_SIGMA × σ(r)`.
//! The original value is recovered as `q as f32 × scale`.

/// Number of bytes in the scale header.
pub const SCALE_BYTES: usize = 4;

/// Number of σ to clip at. Values outside ±CLIP_SIGMA·σ are saturated.
pub const CLIP_SIGMA: f32 = 3.0;

/// Largest representable absolute value in the int8 payload.
pub const INT8_QMAX: f32 = 127.0;

/// A decoded payload ready for reconstruction.
pub struct Payload {
    /// Dequantisation scale in float32 units per int8 step.
    pub scale: f32,
    /// Quantised values in `[-INT8_QMAX, INT8_QMAX]`.
    pub quantized: Vec<i8>,
}

impl Payload {
    /// Serialise to the on-wire format: `[f32 scale LE] ++ [i8 × d]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SCALE_BYTES + self.quantized.len());
        out.extend_from_slice(&self.scale.to_le_bytes());
        // SAFETY: i8 and u8 are the same size; the bit-cast is well-defined.
        out.extend(self.quantized.iter().map(|&v| v as u8));
        out
    }

    /// Deserialise from the on-wire format.
    ///
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

/// Encode a `f32` residual slice with per-vector 3σ clipping + int8 quantisation.
pub fn encode(r: &[f32]) -> Payload {
    let sigma = std_dev(r);
    let clip = if sigma > 0.0 {
        CLIP_SIGMA * sigma
    } else {
        1.0 // degenerate: constant vector; any non-zero scale works
    };
    let scale = clip / INT8_QMAX;

    let quantized = r
        .iter()
        .map(|&v| {
            let clipped = v.clamp(-clip, clip);
            (clipped / scale).round().clamp(-INT8_QMAX, INT8_QMAX) as i8
        })
        .collect();

    Payload { scale, quantized }
}

/// Reconstruct a `f32` residual from an [`int8`][Payload] payload.
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
    let var: f64 = r.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    var.sqrt() as f32
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
        let mut r = vec![0.5f32; 2560];
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
        let r = vec![0.0f32; 2560];
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
        // σ = 0 edge case — degenerate vector
        let r = vec![5.0f32; 64];
        let p = encode(&r);
        let dec = decode(&p);
        assert_eq!(dec.len(), 64);
    }
}
