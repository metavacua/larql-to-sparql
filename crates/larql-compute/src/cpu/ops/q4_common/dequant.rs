use super::f16_to_f32;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Dequantise a Q4_K byte stream to `n_elements` f32 values.
///
/// 256 elements per 144-byte super-block (GGUF / Ollama-canonical layout).
/// `n_elements` must be a multiple of 256 — the caller pads where required.
/// Mirrors `dequantize_row_q4_K` in llama.cpp/ggml-quants.c, kept here so
/// the CPU MoE expert path can call it without a `larql-models` dependency.
pub fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let block_size = 144;
    let super_block = 256;
    if !n_elements.is_multiple_of(super_block) {
        return Vec::new();
    }
    let n_blocks = n_elements / super_block;
    if data.len() < n_blocks * block_size {
        return Vec::new();
    }
    let mut out = vec![0.0f32; n_elements];
    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let p = &block[4..16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..4 {
            scales[j] = p[j] & 0x3F;
            mins[j] = p[j + 4] & 0x3F;
            scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
            mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
        }
        let quants = &block[16..144];
        let sb_base = sb * super_block;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = d * scales[sb_lo] as f32;
            let sc_hi = d * scales[sb_hi] as f32;
            let mn_lo = dmin * mins[sb_lo] as f32;
            let mn_hi = dmin * mins[sb_hi] as f32;
            let chunk = &quants[g * 32..(g + 1) * 32];
            let base_lo = sb_base + sb_lo * 32;
            let base_hi = sb_base + sb_hi * 32;
            for l in 0..32 {
                let byte = chunk[l];
                out[base_lo + l] = sc_lo * (byte & 0x0F) as f32 - mn_lo;
                out[base_hi + l] = sc_hi * ((byte >> 4) & 0x0F) as f32 - mn_hi;
            }
        }
    }
    out
}
