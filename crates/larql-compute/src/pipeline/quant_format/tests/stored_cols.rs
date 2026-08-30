//! `stored_cols` tests: the byte count is the width authority.
//!
//! Writers pad rows to the block boundary, so a model whose inner
//! dim is not a block multiple stores wider rows than its logical
//! width. Deriving from the logical width instead truncates the
//! superblock count and desynchronises the row stride.

use crate::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
use crate::pipeline::quant_format::*;

/// Quantise `rows` rows of `logical` values each, zero-padded to the
/// next super-block boundary — the writer's `pad_rows_to_block` shape.
fn padded_rows_q4k(rows: usize, logical: usize) -> Vec<u8> {
    let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
    let padded = logical.div_ceil(block) * block;
    let mut data = vec![0.0f32; rows * padded];
    for (i, v) in data.iter_mut().enumerate() {
        if i % padded < logical {
            *v = ((i % 31) as f32) - 15.0;
        }
    }
    quantize_q4_k(&data)
}

/// The padded-store case the derivation exists for: rows written at
/// the next block boundary (GPT-OSS's hidden 2880 → 3072 shape, here
/// at test scale) answer the STORED width, not the logical one.
#[test]
fn padded_q4k_rows_answer_the_stored_width() {
    let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
    let (rows, logical) = (4usize, block + block / 4); // 320 → stored 512
    let stored = logical.div_ceil(block) * block;
    let bytes = padded_rows_q4k(rows, logical);
    let w = QuantWeight::new(QuantFormat::Q4_K, &bytes, QuantAux::None);
    assert_eq!(w.stored_cols(rows, logical), stored);
}

#[test]
fn padded_q6k_rows_answer_the_stored_width() {
    let (block, _) = QuantFormat::Q6_K.packed_block_layout().unwrap();
    let (rows, logical) = (3usize, block / 2); // 128 → stored 256
    let stored = logical.div_ceil(block) * block;
    let mut data = vec![0.0f32; rows * stored];
    for (i, v) in data.iter_mut().enumerate() {
        if i % stored < logical {
            *v = (i % 17) as f32;
        }
    }
    let bytes = quantize_q6_k(&data);
    let w = QuantWeight::new(QuantFormat::Q6_K, &bytes, QuantAux::None);
    assert_eq!(w.stored_cols(rows, logical), stored);
}

/// Block-aligned rows are their own stored width — the derivation is
/// the identity on every model that never needed padding.
#[test]
fn aligned_rows_are_a_no_op() {
    let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
    let rows = 2usize;
    let bytes = padded_rows_q4k(rows, block);
    let w = QuantWeight::new(QuantFormat::Q4_K, &bytes, QuantAux::None);
    assert_eq!(w.stored_cols(rows, block), block);
}

/// Bytes that don't divide by the row count aren't a row store of
/// this matrix — answer the caller's fallback, never a guess.
#[test]
fn indivisible_bytes_fall_back() {
    let bytes = vec![0u8; 1001];
    let w = QuantWeight::new(QuantFormat::Q4_K, &bytes, QuantAux::None);
    assert_eq!(w.stored_cols(2, 96), 96);
}

/// Zero rows can't define a row width.
#[test]
fn zero_rows_fall_back() {
    let w = QuantWeight::new(QuantFormat::Q4_K, &[], QuantAux::None);
    assert_eq!(w.stored_cols(0, 64), 64);
}

/// A derivation NARROWER than the logical width means these bytes are
/// not a padded row store — the padding contract can only widen.
#[test]
fn narrower_than_logical_falls_back() {
    let (block, _) = QuantFormat::Q4_K.packed_block_layout().unwrap();
    let bytes = padded_rows_q4k(2, block); // stored width = 1 block
    let w = QuantWeight::new(QuantFormat::Q4_K, &bytes, QuantAux::None);
    assert_eq!(w.stored_cols(2, 3 * block), 3 * block);
}

/// Unquantised rows derive from their element size.
#[test]
fn f32_rows_derive_from_element_size() {
    let bytes = vec![0u8; 2 * 96 * 4];
    let w = QuantWeight::new(QuantFormat::F32, &bytes, QuantAux::None);
    assert_eq!(w.stored_cols(2, 80), 96);
}
