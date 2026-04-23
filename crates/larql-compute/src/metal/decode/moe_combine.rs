//! CPU-side combine step for hybrid MoE layers.
//!
//! Runs after the GPU dense-FFN has written `new_h = h_post_attn + _1(dense)`
//! and the CPU MoE block has added `moe_out` into `new_h` in place. At that
//! point `new_h - h_post_attn` equals `_1(dense) + _2(moe)` — HF's `h1 + h2`
//! in the Gemma 4 decoder-layer forward.
//!
//! Two independent HF-matching operations happen here:
//!   1. **Outer post-FFN norm** on `(h1 + h2)`, then residual add. Matches:
//!        `hidden = residual + post_feedforward_layernorm(h1 + h2)`
//!   2. **Whole-layer `layer_scalar` multiplication** on the entire output.
//!      Matches HF's final step in `Gemma4TextDecoderLayer.forward`:
//!        `hidden_states *= self.layer_scalar`
//!      NB: this multiplies `h_post_attn + ffn_delta` — not just the FFN
//!      delta — which is why folding `layer_scalar` into the outer-norm
//!      scale was wrong (prior bug: 14× mis-scaling on 26B A4B collapsed
//!      the model to degenerate token-repetition output).
//!
//! All operations here are pure f32 arithmetic on shared-memory Metal
//! buffers; no encoder or command buffer involvement.

use crate::FullPipelineLayer;

/// Apply the outer post-FFN norm (when the arch declares one) followed by
/// the whole-layer `layer_scalar` multiplication. Operates in place on
/// `new_h`. Requires that `new_h` currently holds
/// `h_post_attn + (_1(dense) + _2(moe))`.
pub(super) fn apply_outer_combine(
    layer: &FullPipelineLayer,
    new_h: &metal::Buffer,
    h_post_attn: &metal::Buffer,
    hidden: usize,
) {
    // Diagnostic bypass: leave `new_h` as `h_post_attn + _1(dense) + _2(moe)`
    // without outer norm OR layer_scalar — useful for isolating whether
    // this combine step is the broken piece.
    if std::env::var("SKIP_OUTER_NORM").is_ok() {
        return;
    }

    let h_ptr = new_h.contents() as *mut f32;
    let ha_ptr = h_post_attn.contents() as *const f32;

    // Step A — outer post-FFN norm on `(h1 + h2)`, residual-added back.
    //
    // Falls back to `post_ffn_norm` (which for Gemma 4 MoE is `_1`) when no
    // un-suffixed outer norm tensor is loaded, so older vindexes still work
    // even if incorrectly. The correct path uses `moe_outer_post_norm` which
    // the extractor now emits for hybrid-MoE architectures.
    if layer.moe_combined_output_norm {
        let outer_w = layer.moe_outer_post_norm.or(layer.post_ffn_norm);
        if let Some(outer_w) = outer_w {
            apply_outer_norm(h_ptr, ha_ptr, hidden, outer_w, layer.norm_offset, layer.eps);
        }
    }

    // Step B — whole-layer `layer_scalar` multiplication. HF's
    //   `Gemma4TextDecoderLayer.forward` ends with `hidden_states *= self.layer_scalar`
    // which scales BOTH the residual and the FFN delta. A null scalar
    // (0.0) or an identity scalar (1.0) is a no-op.
    apply_whole_layer_scalar(h_ptr, hidden, layer.layer_scalar);
}

/// Apply `new_h = h_post_attn + outer_norm(new_h - h_post_attn)` in place,
/// with `outer_norm(x) = x / rms(x) * (w + norm_offset)`.
fn apply_outer_norm(
    h_ptr: *mut f32,
    ha_ptr: *const f32,
    hidden: usize,
    outer_w: &[f32],
    norm_offset: f32,
    eps: f32,
) {
    unsafe {
        let combined: Vec<f32> = (0..hidden)
            .map(|i| *h_ptr.add(i) - *ha_ptr.add(i))
            .collect();
        let rms = (combined.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
        for (i, (&c, &w)) in combined.iter().zip(outer_w.iter()).enumerate() {
            *h_ptr.add(i) = *ha_ptr.add(i) + c / rms * (w + norm_offset);
        }
    }
}

/// In-place `new_h[i] *= layer_scalar`. Matches HF's final
/// `hidden_states *= self.layer_scalar` in `DecoderLayer.forward`.
/// No-op when `layer_scalar` is 0.0 (absent) or 1.0 (identity).
fn apply_whole_layer_scalar(h_ptr: *mut f32, hidden: usize, layer_scalar: f32) {
    if layer_scalar == 0.0 || layer_scalar == 1.0 { return; }
    unsafe {
        for i in 0..hidden {
            *h_ptr.add(i) *= layer_scalar;
        }
    }
}
