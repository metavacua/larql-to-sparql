//! In-memory Q4_K attention projections behind [`KvIndex`].
//!
//! [`KvIndex`] is an all-defaults capability surface: implementing only
//! [`KvIndex::attn_kquant_layer_data`] routes the existing Q4K-direct
//! attention decode (`attention/decode/q4k_direct.rs`) without a vindex
//! on disk — the weights are quantised once at construction and the
//! kernels consume the packed bytes directly. Built for the TTS funnel's
//! realtime push (`docs/tts-funnel.md`), where the fp32 attention
//! projections were the largest remaining per-frame cost; general to any
//! Qwen3-shaped `ModelWeights`.
//!
//! The vindex-native version of this (step 6) carries the same packed
//! bytes pre-computed in the artefact; this type is the bridge until
//! then, and the shape both must serve is pinned by the trait.

use crate::kv_index::KvIndex;
use crate::QuantFormat;
use larql_models::ModelWeights;

use crate::cpu::ops::q4_common::quantize_q4_k;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Q4_K super-block width; projection columns must divide into it.
const Q4K_BLOCK_ELEMS: usize = 256;

struct PackedLayer {
    /// Q, K, V, O packed row-major Q4_K.
    projections: [Vec<u8>; 4],
}

/// Per-layer Q4_K-packed attention projections for one model.
pub struct PackedAttnIndex {
    layers: Vec<PackedLayer>,
}

impl PackedAttnIndex {
    /// Quantise the attention projections of every layer. Errors by
    /// message on a missing tensor or a column count that does not
    /// divide into Q4_K blocks.
    pub fn quantize_from(weights: &ModelWeights) -> Result<Self, String> {
        let arch = &*weights.arch;
        let mut layers = Vec::with_capacity(weights.num_layers);
        for layer in 0..weights.num_layers {
            let pack = |key: String| -> Result<Vec<u8>, String> {
                let tensor = weights
                    .tensors
                    .get(&key)
                    .ok_or_else(|| format!("missing attention tensor {key}"))?;
                if tensor.ncols() % Q4K_BLOCK_ELEMS != 0 {
                    return Err(format!(
                        "{key}: {} columns not a multiple of {Q4K_BLOCK_ELEMS}",
                        tensor.ncols()
                    ));
                }
                let data = tensor
                    .as_standard_layout()
                    .as_slice()
                    .expect("standard layout")
                    .to_vec();
                Ok(quantize_q4_k(&data))
            };
            layers.push(PackedLayer {
                projections: [
                    pack(arch.attn_q_key(layer))?,
                    pack(arch.attn_k_key(layer))?,
                    pack(arch.attn_v_key(layer))?,
                    pack(arch.attn_o_key(layer))?,
                ],
            });
        }
        Ok(Self { layers })
    }
}

impl KvIndex for PackedAttnIndex {
    fn attn_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        let tag = QuantFormat::Q4_K.registry_tag();
        let l = self.layers.get(layer)?;
        Some([
            (l.projections[0].as_slice(), tag),
            (l.projections[1].as_slice(), tag),
            (l.projections[2].as_slice(), tag),
            (l.projections[3].as_slice(), tag),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::test_fixtures::make_test_weights;

    #[test]
    fn refuses_non_divisible_columns() {
        // make_test_weights is 16-wide — not Q4_K-divisible.
        assert!(PackedAttnIndex::quantize_from(&make_test_weights()).is_err());
    }

    #[test]
    fn serves_packed_bytes_per_layer() {
        // A 256-wide synthetic model quantises and serves all four
        // projections with the right packed sizes.
        use larql_models::WeightArray;
        use ndarray::Array2;
        use std::collections::HashMap;

        const HIDDEN: usize = 256;
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "tinymodel",
            "hidden_size": HIDDEN,
            "num_hidden_layers": 1,
            "intermediate_size": 512,
            "head_dim": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32,
        }));
        let mat = |rows: usize| -> WeightArray {
            Array2::from_shape_fn((rows, HIDDEN), |(r, c)| ((r + c) % 13) as f32 * 0.01)
                .into_shared()
        };
        let mut tensors: HashMap<String, WeightArray> = HashMap::new();
        tensors.insert(arch.attn_q_key(0), mat(4 * 64));
        tensors.insert(arch.attn_k_key(0), mat(2 * 64));
        tensors.insert(arch.attn_v_key(0), mat(2 * 64));
        tensors.insert(arch.attn_o_key(0), mat(HIDDEN));
        let embed = mat(32);
        let weights = ModelWeights {
            tensors,
            vectors: HashMap::new(),
            raw_bytes: HashMap::new(),
            packed_mmaps: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: HashMap::new(),
            per_layer_ffn_format: HashMap::new(),
            embed: embed.clone(),
            lm_head: embed,
            position_embed: None,
            num_layers: 1,
            hidden_size: HIDDEN,
            intermediate_size: 512,
            vocab_size: 32,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 2,
            rope_base: 10000.0,
            arch,
        };

        let index = PackedAttnIndex::quantize_from(&weights).unwrap();
        let data = index.attn_kquant_layer_data(0).expect("layer 0");
        // Q4_K packs 256 elements into 144 bytes.
        assert_eq!(data[0].0.len(), 4 * 64 * 144);
        assert_eq!(data[1].0.len(), 2 * 64 * 144);
        assert_eq!(data[3].0.len(), HIDDEN * 144);
        assert!(data.iter().all(|(_, tag)| *tag == "Q4_K"));
        assert!(index.attn_kquant_layer_data(1).is_none());
    }
}
