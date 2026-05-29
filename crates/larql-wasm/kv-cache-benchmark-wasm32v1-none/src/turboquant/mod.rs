//! TurboQuant — re-exported from `larql_kv::turbo_quant`.
//!
//! Algorithm modules still live here for the benchmark's KvStrategy impl;
//! the KvEngine integration lives in larql-inference.

pub mod codebooks;
pub mod lloyd_max;
pub mod packing;
pub mod rotation;

pub use larql_kv::turbo_quant::TurboQuant;

use crate::{model_config::ModelConfig, KvStrategy};

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
            buf.extend_from_slice(&self.encode_vector(v));
        }
        buf
    }

    fn decode(
        &self,
        encoded: &[u8],
        num_vectors: usize,
        dim: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
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
        let num_vectors = seq_len * config.layers * config.kv_heads * 2;
        num_vectors * self.bytes_per_vector(config.kv_dim())
    }
}
