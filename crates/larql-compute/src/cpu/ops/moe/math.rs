//! Numeric primitives used by the MoE forward pass.
//!
//! `pub(super)` keeps these module-private — `cpu_moe_forward` and the
//! per-expert helpers share them, nothing outside `moe/` should.

/// Dequantize a BF16 byte slice to f32.
#[inline]
pub(super) fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2)
        .map(|b| f32::from_bits((u32::from(u8::from_le_bytes([b[0]])) | (u32::from(u8::from_le_bytes([b[1]])) << 8)) << 16))
        .collect()
}

/// Extract one expert's weight slice from packed BF16 tensor and dequantize to f32.
/// Packed layout: [num_experts, out_rows, in_cols] — expert `e` starts at byte
/// `e * out_rows * in_cols * 2`.
pub(super) fn extract_expert_weights(
    packed: &[u8],
    expert_idx: usize,
    out_rows: usize,
    in_cols: usize,
) -> Vec<f32> {
    let bytes_per_expert = out_rows * in_cols * 2;
    let start = expert_idx * bytes_per_expert;
    let end = start + bytes_per_expert;
    bf16_to_f32(&packed[start..end])
}

/// RMSNorm: out[i] = x[i] / rms(x) * (w[i] + offset)
pub(super) fn rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    if w.is_empty() || x.is_empty() { return x.to_vec(); }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(w.iter()).map(|(&xi, &wi)| xi / rms * (wi + offset)).collect()
}

/// Parameter-free RMSNorm (HF `Gemma4RMSNorm(with_scale=False)`): scales
/// `x` by `1/sqrt(mean(x²) + eps)` with no learned weight. Used by the
/// Gemma 4 router, whose norm has no `.weight` tensor on disk.
pub(super) fn rms_norm_no_weight(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() { return Vec::new(); }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().map(|v| v / rms).collect()
}

/// SiLU activation: x * sigmoid(x)
#[inline]
pub(super) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU with tanh approximation (Gemma 4 expert FFN activation).
#[inline]
pub(super) fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Compute y = x @ W.T where W is [out_rows, in_cols] stored row-major.
pub(super) fn matmul_vec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_rows * in_cols);
    debug_assert_eq!(x.len(), in_cols);
    (0..out_rows).map(|row| {
        let w_row = &w[row * in_cols..(row + 1) * in_cols];
        x.iter().zip(w_row.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}

/// Softmax in-place.
pub(super) fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    if sum > 0.0 { for x in v.iter_mut() { *x /= sum; } }
}

/// Top-k indices by value (descending). Returns (indices, values).
pub(super) fn top_k(v: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(v.len());
    let mut indexed: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
    let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
    (indices, values)
}
