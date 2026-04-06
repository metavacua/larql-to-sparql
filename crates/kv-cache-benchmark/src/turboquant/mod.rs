pub mod rotation;
pub mod lloyd_max;
pub mod packing;
pub mod codebooks;

use crate::{KvStrategy, model_config::ModelConfig};

/// Strategy 2: TurboQuant (ICLR 2026).
///
/// Algorithm 1 (MSE-only, no QJL):
///   1. Normalize → unit norm, store scalar
///   2. Walsh-Hadamard rotation (spreads coordinates to Beta distribution)
///   3. Lloyd-Max scalar quantization (3 or 4 bits per coordinate)
///   4. Bit-pack indices
///   5. Decode: unpack → centroids → inverse WHT → rescale
pub struct TurboQuant {
    pub bits: u8, // 3 or 4
}

impl TurboQuant {
    pub fn new(bits: u8) -> Self {
        assert!(bits == 3 || bits == 4, "TurboQuant supports 3 or 4 bits");
        Self { bits }
    }

    /// Encode a single vector: normalize → WHT → quantize → pack.
    pub fn encode_vector(&self, x: &[f32]) -> Vec<u8> {
        let d = x.len();

        // Step 1: compute norm and normalize
        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let x_hat: Vec<f32> = if norm > 1e-12 {
            x.iter().map(|v| v / norm).collect()
        } else {
            vec![0.0; d]
        };

        // Step 2: Walsh-Hadamard transform (in-place)
        let mut y = rotation::wht(&x_hat);

        // Step 3: Lloyd-Max quantize each coordinate
        let codebook = codebooks::get_codebook(d, self.bits);
        let indices: Vec<u8> = y
            .iter()
            .map(|&val| lloyd_max::quantize_scalar(val, codebook))
            .collect();

        // Step 4: pack norm (4 bytes f32) + bit-packed indices
        let mut buf = Vec::new();
        buf.extend_from_slice(&norm.to_le_bytes());
        packing::pack_indices(&indices, self.bits, &mut buf);
        buf
    }

    /// Decode a single vector: unpack → centroids → inverse WHT → rescale.
    pub fn decode_vector(&self, encoded: &[u8], dim: usize) -> Vec<f32> {
        // Read norm
        let norm = f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);

        // Unpack indices
        let indices = packing::unpack_indices(&encoded[4..], dim, self.bits);

        // Centroid lookup
        let codebook = codebooks::get_codebook(dim, self.bits);
        let y: Vec<f32> = indices
            .iter()
            .map(|&idx| codebook.centroids[idx as usize])
            .collect();

        // Inverse WHT (WHT is self-inverse up to scaling)
        let x_hat = rotation::wht(&y);

        // Rescale
        x_hat.iter().map(|&v| v * norm).collect()
    }

    /// Bytes per encoded vector.
    fn bytes_per_vector(&self, dim: usize) -> usize {
        4 + packing::packed_size(dim, self.bits) // norm + packed indices
    }
}

impl KvStrategy for TurboQuant {
    fn name(&self) -> &str {
        match self.bits {
            3 => "TurboQuant 3-bit",
            4 => "TurboQuant 4-bit",
            _ => "TurboQuant",
        }
    }

    fn encode(&self, keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<u8> {
        let mut buf = Vec::new();
        for v in keys.iter().chain(values.iter()) {
            let enc = self.encode_vector(v);
            buf.extend_from_slice(&enc);
        }
        buf
    }

    fn decode(&self, encoded: &[u8], num_vectors: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let bytes_per = self.bytes_per_vector(dim);
        let mut keys = Vec::with_capacity(num_vectors);
        let mut values = Vec::with_capacity(num_vectors);

        for i in 0..num_vectors {
            let offset = i * bytes_per;
            keys.push(self.decode_vector(&encoded[offset..offset + bytes_per], dim));
        }
        for i in 0..num_vectors {
            let offset = (num_vectors + i) * bytes_per;
            values.push(self.decode_vector(&encoded[offset..offset + bytes_per], dim));
        }
        (keys, values)
    }

    fn memory_bytes(&self, config: &ModelConfig, seq_len: usize) -> usize {
        let num_vectors = seq_len * config.layers * config.kv_heads * 2; // K+V
        num_vectors * self.bytes_per_vector(config.kv_dim())
    }
}
