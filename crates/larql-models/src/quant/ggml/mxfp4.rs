//! MXFP4 dequant — GGUF interleaved block layout.
//!
//! Each `MXFP4_BLOCK_ELEMS = 32` element block is laid out as:
//! ```text
//! [scale: u8 (E8M0 exponent), nibbles: 16 × u8 (2 × 4-bit values each)]
//! ```
//! = `MXFP4_BLOCK_BYTES = 17` bytes per block.
//!
//! Nibble layout matches `crate::quant::mxfp4::MXFP4_TABLE`: low nibble
//! is the even-index element, high nibble is the odd-index element.
//! Scale is `2^(scale_byte - 127)` (E8M0 — exponent-only float).
//!
//! Distinct from `crate::quant::mxfp4` which handles the
//! safetensors-side packed format (separate blocks + scales tensors,
//! used by GPT-OSS). Both modules share `MXFP4_TABLE`.

use crate::detect::ModelError;
use crate::quant::ggml::{MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS};
use crate::quant::mxfp4::{e8m0_to_f32, MXFP4_TABLE};

pub fn dequantize_mxfp4(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    if n_elements % MXFP4_BLOCK_ELEMS != 0 {
        return Err(ModelError::Parse(format!(
            "MXFP4: n_elements={n_elements} not divisible by block size {MXFP4_BLOCK_ELEMS}"
        )));
    }
    let n_blocks = n_elements / MXFP4_BLOCK_ELEMS;
    let need = n_blocks.checked_mul(MXFP4_BLOCK_BYTES).ok_or_else(|| {
        ModelError::Parse(format!("MXFP4: byte count overflow ({n_blocks} blocks)"))
    })?;
    if data.len() < need {
        return Err(ModelError::Parse(format!(
            "MXFP4: data too short: {} bytes < expected {need} ({n_blocks} blocks)",
            data.len()
        )));
    }

    let mut out = Vec::with_capacity(n_elements);
    for b in 0..n_blocks {
        let block = &data[b * MXFP4_BLOCK_BYTES..(b + 1) * MXFP4_BLOCK_BYTES];
        let scale = e8m0_to_f32(block[0]);
        for i in 0..(MXFP4_BLOCK_ELEMS / 2) {
            let byte = block[1 + i];
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            out.push(MXFP4_TABLE[lo] * scale);
            out.push(MXFP4_TABLE[hi] * scale);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_mxfp4_zero_block_yields_zeros() {
        // scale=0 byte -> e8m0_to_f32(0) = 0.0; nibbles ignored.
        let mut block = vec![0u8; MXFP4_BLOCK_BYTES];
        block[0] = 0;
        for i in 1..MXFP4_BLOCK_BYTES {
            block[i] = 0xFF; // nonzero nibbles
        }
        let out = dequantize_mxfp4(&block, MXFP4_BLOCK_ELEMS).unwrap();
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dequantize_mxfp4_unit_scale_matches_table() {
        // scale byte 127 = 2^0 = 1.0 → output values are literally MXFP4_TABLE entries.
        let mut block = vec![0u8; MXFP4_BLOCK_BYTES];
        block[0] = 127;
        // pair (lo=2 →1.0, hi=3 →1.5) for first byte
        block[1] = 0x32;
        let out = dequantize_mxfp4(&block, MXFP4_BLOCK_ELEMS).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn dequantize_mxfp4_rejects_short_buffer() {
        let buf = vec![0u8; MXFP4_BLOCK_BYTES - 1];
        let err = dequantize_mxfp4(&buf, MXFP4_BLOCK_ELEMS).unwrap_err();
        assert!(matches!(err, ModelError::Parse(_)));
    }
}
