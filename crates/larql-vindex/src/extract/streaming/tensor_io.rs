//! Safetensors-mmap helpers for the streaming extraction pipeline.
//!
//! Named `tensor_io` rather than `safetensors` to avoid shadowing the
//! external `safetensors` crate inside the parent module.
//!
//! - `MmapShard` keeps the file handle + mmap alive for the lifetime of
//!   one extraction.
//! - `GateSink` is the gate-vector writer abstraction (real file or
//!   `/dev/null` when `--drop-gate-vectors` is set).
//! - `get_tensor_f32` reads a 2D tensor by key and dequantises to f32.
//! - `normalize_key` strips a fixed prefix list from a tensor key.

use std::collections::HashMap;
use std::io::{BufWriter, Write};

use ndarray::Array2;

use crate::error::VindexError;

/// Mmap'd safetensors file — kept alive for the duration of extraction.
pub(super) struct MmapShard {
    pub(super) _file: std::fs::File,
    pub(super) mmap: memmap2::Mmap,
}

/// Sink for gate-vector bytes. With `--drop-gate-vectors` the writer
/// still walks every layer (so `layer_infos` is populated for
/// `index.json`) but redirects bytes to `/dev/null` — they're
/// recoverable from `interleaved_q4k.bin` at load time.
pub(super) enum GateSink {
    File(BufWriter<std::fs::File>),
    Discard(std::io::Sink),
}

impl Write for GateSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            GateSink::File(f) => f.write(buf),
            GateSink::Discard(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            GateSink::File(f) => f.flush(),
            GateSink::Discard(s) => s.flush(),
        }
    }
}

/// Get a 2D tensor from mmap'd safetensors, dequantizing to f32.
pub(super) fn get_tensor_f32(
    shards: &[MmapShard],
    index: &HashMap<String, (usize, String)>,
    key: &str,
) -> Result<Option<Array2<f32>>, VindexError> {
    let (shard_idx, tensor_name) = match index.get(key) {
        Some(v) => v,
        None => return Ok(None),
    };

    let st = safetensors::SafeTensors::deserialize(&shards[*shard_idx].mmap)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let view = st
        .tensor(tensor_name)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let shape = view.shape();
    if shape.len() != 2 {
        return Ok(None);
    }

    // MXFP4 (DeepSeek-V4 expert weights): the streaming extract has its own
    // f32 decoder separate from `larql-models::loading::safetensors`, so the
    // I8+F8_E8M0 pairing has to be detected here too. Without it,
    // `gate_vectors.bin` came out 0 bytes for V4 models because every layer's
    // gate fell into the catch-all `_ => return Ok(None)` below.
    //
    // Detection contract: `.weight` tensor with `I8` dtype and a `.scale`
    // companion of dtype `F8_E8M0` whose row count matches and whose column
    // count divides the unpacked-cols evenly into a sane group size
    // {16, 32, 64, 128}. Anything else falls through to the dtype match.
    if view.dtype() == safetensors::Dtype::I8 && tensor_name.ends_with(".weight") {
        let scale_name = tensor_name.replacen(".weight", ".scale", 1);
        if let Ok(scale_view) = st.tensor(&scale_name) {
            if scale_view.dtype() == safetensors::Dtype::F8_E8M0 {
                let s_shape = scale_view.shape();
                if s_shape.len() == 2 && s_shape[0] == shape[0] {
                    let cols_unpacked = shape[1] * 2;
                    if s_shape[1] > 0 && cols_unpacked % s_shape[1] == 0 {
                        let group_size = cols_unpacked / s_shape[1];
                        if [16usize, 32, 64, 128].contains(&group_size) {
                            let unpacked = larql_models::quant::mxfp4::dequantize_expert(
                                view.data(),
                                scale_view.data(),
                                shape[0],
                                s_shape[1],
                            )
                            .map_err(|e| VindexError::Parse(e.to_string()))?;
                            let arr = Array2::from_shape_vec((shape[0], cols_unpacked), unpacked)
                                .map_err(|e| VindexError::Parse(e.to_string()))?;
                            return Ok(Some(arr));
                        }
                    }
                }
            }
        }
    }

    let data = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => crate::format::quant::half::decode_f16(view.data()),
        safetensors::Dtype::BF16 => crate::format::quant::half::decode_bf16(view.data()),
        _ => return Ok(None), // skip non-float (and non-MXFP4) tensors
    };

    let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    Ok(Some(arr))
}

pub(super) fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

#[cfg(test)]
mod tests {
    use super::normalize_key;

    #[test]
    fn normalize_strips_first_matching_prefix() {
        let prefixes = ["model.", "transformer."];
        assert_eq!(
            normalize_key("model.layers.0.mlp.gate_proj.weight", &prefixes),
            "layers.0.mlp.gate_proj.weight",
        );
    }

    #[test]
    fn normalize_keeps_key_when_no_prefix_matches() {
        let prefixes = ["model.", "transformer."];
        assert_eq!(
            normalize_key("layers.0.mlp.gate_proj.weight", &prefixes),
            "layers.0.mlp.gate_proj.weight",
        );
    }

    #[test]
    fn normalize_uses_first_match_only() {
        // First matching prefix wins; the second isn't applied to the
        // residue. Pinning this matters because the safetensors loader
        // walks tensors with a fixed prefix list — re-stripping would
        // mangle keys like "model.model.embed_tokens.weight".
        let prefixes = ["model.", "model.model."];
        assert_eq!(
            normalize_key("model.model.embed_tokens.weight", &prefixes),
            "model.embed_tokens.weight",
        );
    }

    #[test]
    fn normalize_with_empty_prefix_list_is_identity() {
        assert_eq!(normalize_key("anything", &[]), "anything");
    }

    #[test]
    fn normalize_handles_empty_input() {
        let prefixes = ["model."];
        assert_eq!(normalize_key("", &prefixes), "");
    }
}
