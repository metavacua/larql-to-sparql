use ndarray::Array2;

/// Dequantise a row-major Q4_K or Q6_K matrix into a dense f32 `Array2`.
///
/// **Row-stride aware.** The on-disk layout stores each row padded to the
/// next multiple of `K_QUANT_BLOCK_ELEMS` (256) — this matches the writer
/// which can only emit whole super-blocks. The function dequantises the
/// full `rows × padded_cols` byte payload and slices columns to the
/// logical `cols` requested by the caller.
///
/// For tensors whose logical `cols` IS already block-aligned (e.g. Qwen 3.6
/// hidden=4096, intermediate=27648 — both multiples of 256), `padded_cols
/// == cols` and the slice is a no-op. For Gemma 3 (hidden=1152, head_dim=256,
/// kv_dim=256) the padded width is 1280 vs the logical 1152, so each row's
/// bytes span 720 not 648 — without padding-awareness the second row onwards
/// would read at the wrong byte offset and corrupt every output column past
/// the first. That was the root cause of the long-standing Gemma 3 CPU
/// inference gibberish.
///
/// Unknown formats panic; callers have already dispatched on format via
/// `larql_vindex::quant::registry`.
pub(super) fn dequantize_matrix(
    bytes: &[u8],
    format: &str,
    rows: usize,
    cols: usize,
) -> Array2<f32> {
    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
    let padded_cols = cols.div_ceil(block) * block;
    let n_padded = rows * padded_cols;
    let info = larql_vindex::quant::registry::lookup(format)
        .unwrap_or_else(|| panic!("unsupported quant format in vindex: {format}"));
    let floats = (info.dequantize)(bytes, n_padded)
        .unwrap_or_else(|e| panic!("{format} dequant failed: {e}"));
    let full = Array2::from_shape_vec((rows, padded_cols), floats)
        .expect("shape mismatch dequantising Q4K matrix");
    if padded_cols == cols {
        full
    } else {
        full.slice(ndarray::s![.., ..cols]).to_owned()
    }
}
