use ndarray::Array2;

/// Dequantise a row-major Q4_K or Q6_K matrix into a dense f32 `Array2`
/// of the LOGICAL shape `[rows, cols]`.
///
/// The on-disk slab is row-padded: the k-quant writers
/// (`pad_rows_to_block`) pad every row to the next 256-element
/// super-block boundary, so the stored stride is
/// `k_quant_padded_cols(cols)`, not `cols`. This function decodes at
/// the padded stride and strips each row's padding tail — callers pass
/// logical dims and never account for padding themselves. (Decoding at
/// the unpadded stride shifts every row after the first whenever
/// `cols % 256 != 0` — e.g. GPT-OSS-20B hidden=2880.)
///
/// Unknown formats panic; callers have already dispatched on format via
/// `larql_vindex::quant::registry`.
pub(super) fn dequantize_matrix(
    bytes: &[u8],
    format: &str,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let padded_cols = larql_models::quant::ggml::k_quant_padded_cols(cols);
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format in vindex: {format}"));
    let floats = (info.dequantize)(bytes, rows * padded_cols)
        .unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let logical = if padded_cols == cols {
        let mut f = floats;
        f.truncate(rows * cols);
        f
    } else {
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            out.extend_from_slice(&floats[r * padded_cols..r * padded_cols + cols]);
        }
        out
    };
    Array2::from_shape_vec((rows, cols), logical).expect("shape mismatch dequantising Q4K matrix")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q4_K format with `rows*cols` a multiple of the 256-element super-block:
    /// `padded == n`, dequant returns exactly `n` floats, so the truncate
    /// path takes the `else { floats }` arm at L25.
    #[test]
    fn dequantize_matrix_q4k_padded_path_keeps_full_buffer() {
        let rows = 1;
        let cols = 256;
        let f32_in: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.001).collect();
        let bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&f32_in);
        let out = dequantize_matrix(&bytes, "Q4_K", rows, cols);
        assert_eq!(out.shape(), &[rows, cols]);
    }

    /// `rows*cols` not a multiple of 256 — `padded > n`, so the dequantiser
    /// returns more floats than needed and the truncate-to-`n` branch at
    /// L23 fires.
    #[test]
    fn dequantize_matrix_q4k_unpadded_path_truncates() {
        let rows = 1;
        let cols = 200; // not a multiple of 256 → padded = 256
                        // Quantiser still needs a 256-multiple input; pad with zeros.
        let mut padded = vec![0.0f32; 256];
        for (i, slot) in padded.iter_mut().take(cols).enumerate() {
            *slot = i as f32 * 0.01;
        }
        let bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&padded);
        let out = dequantize_matrix(&bytes, "Q4_K", rows, cols);
        assert_eq!(out.shape(), &[rows, cols]);
    }

    #[test]
    #[should_panic(expected = "unsupported quant format")]
    fn dequantize_matrix_panics_on_unknown_format() {
        let _ = dequantize_matrix(&[0u8; 144], "no_such_format", 1, 256);
    }

    /// Multi-row matrix with `cols % 256 != 0` — the row-padded stride
    /// case. The writer (`pad_rows_to_block`) pads each row to the next
    /// super-block boundary; row r here is the constant r+1, so a
    /// contiguous unpadded-stride decode reads row-0 padding zeros into
    /// row 1 and misses this bound by a wide margin.
    #[test]
    fn dequantize_matrix_strips_per_row_padding() {
        let (rows, cols) = (3usize, 320usize);
        let padded_cols = larql_models::quant::ggml::k_quant_padded_cols(cols); // 512
        let mut padded = Vec::with_capacity(rows * padded_cols);
        for r in 0..rows {
            padded.extend(std::iter::repeat_n((r + 1) as f32 * 0.5, cols));
            padded.resize((r + 1) * padded_cols, 0.0);
        }
        let bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&padded);

        let out = dequantize_matrix(&bytes, "Q4_K", rows, cols);
        assert_eq!(out.shape(), &[rows, cols]);
        for r in 0..rows {
            let expect = (r + 1) as f32 * 0.5;
            for c in 0..cols {
                let got = out[[r, c]];
                assert!(
                    (got - expect).abs() < 0.05,
                    "row {r} col {c}: got {got}, want ~{expect} — row stride is wrong"
                );
            }
        }
    }
}
