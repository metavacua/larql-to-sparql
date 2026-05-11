use ndarray::Array2;

/// Dequantise a row-major Q4_K or Q6_K matrix into a dense f32 `Array2`.
///
/// The on-disk layout (`rows x cols` elements) must be stored contiguously
/// row-major and padded to a multiple of 256 elements per the k-quant
/// super-block size. Unknown formats panic; callers have already dispatched on
/// format via `larql_vindex::quant::registry`.
pub(super) fn dequantize_matrix(
    bytes: &[u8],
    format: &str,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let n = rows * cols;
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded = n.div_ceil(block) * block;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format in vindex: {format}"));
    let floats =
        (info.dequantize)(bytes, padded).unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let truncated = if floats.len() > n {
        floats[..n].to_vec()
    } else {
        floats
    };
    Array2::from_shape_vec((rows, cols), truncated).expect("shape mismatch dequantising Q4K matrix")
}
