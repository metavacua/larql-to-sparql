//! CPU-side combine step for hybrid MoE layers.
//!
//! Runs after the GPU dense-FFN has written `new_h = h_post_attn + _1(dense)`
//! and the CPU MoE block has added `moe_out` into `new_h` in place. At that
//! point `new_h - h_post_attn` equals the HF quantity `_1(dense) + _2(moe)`,
//! i.e. `h1 + h2` in the Gemma 4 decoder-layer forward.
//!
//! This module applies the outer norm (and optional `layer_scalar`) to that
//! delta, producing the final per-layer output. Extracted from the main
//! decode loop for readability — behavior is bit-identical to the inline
//! version that preceded it.
//!
//! All operations here are pure f32 arithmetic on shared-memory Metal
//! buffers; no encoder or command buffer involvement.
//!
//! Call site:
//! ```ignore
//! crate::metal::decode::moe_combine::apply_outer_combine(layer, new_h, h_post_attn, hidden);
//! ```

use crate::FullPipelineLayer;

/// Apply the outer post-FFN combine: either the HF Gemma 4 RMSNorm path
/// (`moe_combined_output_norm=true`) or the legacy `layer_scalar`-only path.
///
/// Operates in place on `new_h`. Requires that `new_h` currently holds
/// `h_post_attn + (dense_contrib + moe_contrib)`.
pub(super) fn apply_outer_combine(
    layer: &FullPipelineLayer,
    new_h: &metal::Buffer,
    h_post_attn: &metal::Buffer,
    hidden: usize,
) {
    let h_ptr = new_h.contents() as *mut f32;
    let ha_ptr = h_post_attn.contents() as *const f32;

    if layer.moe_combined_output_norm {
        // Gemma 4 hybrid MoE: apply the OUTER post_feedforward_layernorm
        // (un-suffixed, distinct from `post_ffn_norm` which is `_1` and was
        // already applied to the dense branch on the GPU in Step 7) to
        // `h1 + h2`, then add to residual.
        //
        // HF:  hidden_states = residual + post_ffn_norm( _1(dense) + _2(moe) )
        //
        // Falls back to `post_ffn_norm` when no outer norm is loaded, so
        // other architectures that collapse the two weights into one
        // (non-Gemma 4 hybrid MoE) continue to work.
        let outer_w = layer.moe_outer_post_norm.or(layer.post_ffn_norm);
        if let Some(outer_w) = outer_w {
            apply_outer_norm(h_ptr, ha_ptr, hidden, outer_w, layer.norm_offset, layer.eps, layer.layer_scalar);
        } else {
            // No outer norm weights — scale by layer_scalar only.
            apply_layer_scalar(h_ptr, ha_ptr, hidden, layer.layer_scalar);
        }
    } else {
        // Standard MoE: scale the combined delta by layer_scalar.
        apply_layer_scalar(h_ptr, ha_ptr, hidden, layer.layer_scalar);
    }
}

/// RMS-norm the `(new_h - h_post_attn)` delta with weight `outer_w`, optional
/// `layer_scalar`, then re-add to `h_post_attn`.
fn apply_outer_norm(
    h_ptr: *mut f32,
    ha_ptr: *const f32,
    hidden: usize,
    outer_w: &[f32],
    norm_offset: f32,
    eps: f32,
    layer_scalar: f32,
) {
    unsafe {
        let combined: Vec<f32> = (0..hidden)
            .map(|i| *h_ptr.add(i) - *ha_ptr.add(i))
            .collect();
        let rms = (combined.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
        let scale = if layer_scalar != 0.0 && layer_scalar != 1.0 { layer_scalar } else { 1.0 };
        for (i, (&c, &w)) in combined.iter().zip(outer_w.iter()).enumerate() {
            *h_ptr.add(i) = *ha_ptr.add(i) + scale * c / rms * (w + norm_offset);
        }
    }
}

/// In-place `new_h = h_post_attn + layer_scalar * (new_h - h_post_attn)`.
/// No-op when `layer_scalar` is 0.0 or 1.0.
fn apply_layer_scalar(h_ptr: *mut f32, ha_ptr: *const f32, hidden: usize, layer_scalar: f32) {
    if layer_scalar == 0.0 || layer_scalar == 1.0 { return; }
    unsafe {
        for i in 0..hidden {
            let pa = *ha_ptr.add(i);
            *h_ptr.add(i) = pa + layer_scalar * (*h_ptr.add(i) - pa);
        }
    }
}
