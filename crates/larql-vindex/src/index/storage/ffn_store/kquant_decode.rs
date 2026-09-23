//! Whole-component dequant for the interleaved-kquant FFN slabs —
//! the one place that knows the on-disk row-padded layout.
//!
//! The k-quant writers (`pad_rows_to_block`) pad every row to the next
//! 256-element super-block boundary, so a component slab is
//! `[rows, k_quant_padded_cols(cols)]`, not `[rows, cols]`. Both dequant
//! caches (`kquant_ffn_layer`, `kquant_ffn_layer_once`) decode through
//! this helper so the padding is stripped exactly once; decoding a
//! row-padded slab at the unpadded stride shifts every row after the
//! first whenever `cols % 256 != 0` and produces silent garbage.

use larql_models::quant::ggml::k_quant_padded_cols;

use super::FFN_DOWN;

/// Dequantise one interleaved-kquant FFN component slab into an
/// unpadded, feature-major f32 buffer of exactly
/// `intermediate * hidden` elements.
///
/// On-disk orientation: gate/up are `[intermediate, padded(hidden)]`
/// (row `feat` is that feature's weight vector); down is
/// `[hidden, padded(intermediate)]` (native `nn.Linear` orientation)
/// and is transposed here so callers get the same feature-major view
/// for all three components: `out[feat*hidden..(feat+1)*hidden]`.
///
/// Returns `None` on an unknown format tag or a dequant failure
/// (short slab), mirroring the caches' fall-back-to-other-paths
/// contract.
pub(crate) fn decode_component_feature_major(
    bytes: &[u8],
    format: &str,
    intermediate: usize,
    hidden: usize,
    component: usize,
) -> Option<Vec<f32>> {
    let (rows, cols) = if component == FFN_DOWN {
        (hidden, intermediate)
    } else {
        (intermediate, hidden)
    };
    let padded_cols = k_quant_padded_cols(cols);
    let info = crate::quant::registry::lookup(format)?;
    let decoded = (info.dequantize)(bytes, rows * padded_cols).ok()?;

    let n = intermediate * hidden;
    if component == FFN_DOWN {
        // Transpose [hidden, padded(intermediate)] → [intermediate, hidden],
        // dropping each source row's padding tail.
        let mut t = vec![0.0f32; n];
        for h in 0..rows {
            let src_row = &decoded[h * padded_cols..h * padded_cols + intermediate];
            for (i, &v) in src_row.iter().enumerate() {
                t[i * hidden + h] = v;
            }
        }
        Some(t)
    } else if padded_cols == cols {
        let mut out = decoded;
        out.truncate(n);
        Some(out)
    } else {
        // Row-padded gate/up: copy each row's logical prefix.
        let mut out = Vec::with_capacity(n);
        for r in 0..rows {
            out.extend_from_slice(&decoded[r * padded_cols..r * padded_cols + cols]);
        }
        Some(out)
    }
}

/// Test-only writer twin: quantise a logical `[rows, cols]` matrix the
/// way the production writer does (per-row zero-padding to the
/// super-block boundary — `pad_rows_to_block` — then Q4_K). Shared by
/// the decode tests here and the cache-path tests in `kquant_cache.rs`.
#[cfg(test)]
pub(crate) fn q4k_row_padded_for_tests(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(data.len(), rows * cols);
    let padded_cols = k_quant_padded_cols(cols);
    let mut padded = Vec::with_capacity(rows * padded_cols);
    for r in 0..rows {
        padded.extend_from_slice(&data[r * cols..(r + 1) * cols]);
        padded.resize((r + 1) * padded_cols, 0.0);
    }
    larql_compute::cpu::ops::q4_common::quantize_q4_k(&padded)
}

/// Q4_K round-trip error bound for small-magnitude test fixtures
/// (4-bit quant with per-32-block scales).
#[cfg(test)]
pub(crate) const Q4K_TEST_TOLERANCE: f32 = 0.05;

#[cfg(test)]
mod tests {
    //! Regression pins for the row-padded stride: every fixture uses a
    //! NON-256-aligned padded dimension, because 256-aligned shapes
    //! (`padded_cols == cols`) cannot distinguish a padding-aware
    //! decode from the pre-fix contiguous one.
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    use super::*;
    use super::{q4k_row_padded_for_tests as q4k_row_padded, Q4K_TEST_TOLERANCE as Q4K_TOLERANCE};

    #[test]
    fn gate_rows_survive_non_aligned_hidden() {
        // hidden=320 → padded 512. Row r is the constant r+1, so the
        // pre-fix contiguous decode (stride 320 over a stride-512 slab)
        // would read row-0 padding zeros into row 1 and fail loudly here.
        let (intermediate, hidden) = (3usize, 320usize);
        let data: Vec<f32> = (0..intermediate)
            .flat_map(|r| std::iter::repeat_n((r + 1) as f32 * 0.5, hidden))
            .collect();
        let bytes = q4k_row_padded(&data, intermediate, hidden);

        let out = decode_component_feature_major(&bytes, "Q4_K", intermediate, hidden, 0)
            .expect("decode succeeds");
        assert_eq!(out.len(), intermediate * hidden);
        for r in 0..intermediate {
            let expect = (r + 1) as f32 * 0.5;
            for (c, &v) in out[r * hidden..(r + 1) * hidden].iter().enumerate() {
                assert!(
                    (v - expect).abs() < Q4K_TOLERANCE,
                    "row {r} col {c}: got {v}, want ~{expect} — row stride is wrong"
                );
            }
        }
    }

    #[test]
    fn down_transpose_survives_non_aligned_intermediate() {
        // Down slab is [hidden, padded(intermediate)]: intermediate=320
        // → padded 512. Source cell [h][i] = i-dependent constant, so
        // after the feature-major transpose feature i's vector must be
        // uniformly that constant across all h.
        let (intermediate, hidden) = (320usize, 8usize);
        let data: Vec<f32> = (0..hidden)
            .flat_map(|_| (0..intermediate).map(|i| (i % 7) as f32 * 0.25))
            .collect();
        let bytes = q4k_row_padded(&data, hidden, intermediate);

        let out = decode_component_feature_major(&bytes, "Q4_K", intermediate, hidden, FFN_DOWN)
            .expect("decode succeeds");
        assert_eq!(out.len(), intermediate * hidden);
        for i in 0..intermediate {
            let expect = (i % 7) as f32 * 0.25;
            for (h, &v) in out[i * hidden..(i + 1) * hidden].iter().enumerate() {
                assert!(
                    (v - expect).abs() < Q4K_TOLERANCE,
                    "feature {i} h {h}: got {v}, want ~{expect} — transpose stride is wrong"
                );
            }
        }
    }

    #[test]
    fn aligned_shapes_keep_pre_fix_behavior() {
        // 256-aligned cols: padded_cols == cols, so the decode must be
        // byte-identical to the old contiguous path.
        let (intermediate, hidden) = (2usize, 256usize);
        let data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i % 11) as f32 * 0.1)
            .collect();
        let bytes = q4k_row_padded(&data, intermediate, hidden);

        let out = decode_component_feature_major(&bytes, "Q4_K", intermediate, hidden, 0)
            .expect("decode succeeds");
        for (i, (&got, &want)) in out.iter().zip(&data).enumerate() {
            assert!(
                (got - want).abs() < Q4K_TOLERANCE,
                "elem {i}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn unknown_format_returns_none() {
        assert!(decode_component_feature_major(&[0u8; 144], "no_such_format", 1, 256, 0).is_none());
    }

    #[test]
    fn short_slab_returns_none() {
        // One super-block of bytes can't cover [2, padded(320)] elements.
        let bytes = quantize_q4_k(&vec![0.0f32; 256]);
        assert!(decode_component_feature_major(&bytes, "Q4_K", 2, 320, 0).is_none());
    }
}
