//! Shared Q4 utilities and pipeline state container.

use metal::ComputePipelineState;

/// Pipeline states for Q4 operations — compiled from modular shaders.
pub struct Q4Pipelines {
    pub matvec: ComputePipelineState,       // Q4 × Q8 matvec (optimised simdgroup)
    pub vecmat: ComputePipelineState,       // Q4 vector-matrix (scatter)
    pub f32_matvec: ComputePipelineState,   // Q4 × f32 matvec (transposed down)
}

/// Pre-quantize f32 vector to Q8_0 (int8 + per-block f32 scale).
/// Used to prepare input for Q4×Q8 dot product.
pub fn quantize_to_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n_blocks = x.len() / 32;
    let mut q8 = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; n_blocks];
    for (b, scale_out) in scales.iter_mut().enumerate().take(n_blocks) {
        let off = b * 32;
        let block = &x[off..off + 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        *scale_out = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for j in 0..32 {
            q8[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (q8, scales)
}
