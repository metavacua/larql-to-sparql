//! CPU-side combine step for hybrid MoE layers.
//!
//! Runs after the GPU dense-FFN has written `new_h = h_post_attn + _1(dense)`
//! and the CPU MoE block has added `moe_out` into `new_h` in place. At that
//! point `new_h - h_post_attn` equals `_1(dense) + _2(moe)` — HF's `h1 + h2`
//! in the Gemma 4 decoder-layer forward.
//!
//! Two independent HF-matching operations happen here:
//!   1. **Outer post-FFN norm** on `(h1 + h2)`, then residual add. Matches:
//!      `hidden = residual + post_feedforward_layernorm(h1 + h2)`
//!   2. **Whole-layer `layer_scalar` multiplication** on the entire output.
//!      Matches HF's final step in `Gemma4TextDecoderLayer.forward`:
//!      `hidden_states *= self.layer_scalar`
//!      NB: this multiplies `h_post_attn + ffn_delta` — not just the FFN
//!      delta — which is why folding `layer_scalar` into the outer-norm
//!      scale was wrong (prior bug: 14× mis-scaling on 26B A4B collapsed
//!      the model to degenerate token-repetition output).
//!
//! All operations here are pure f32 arithmetic on shared-memory Metal
//! buffers; no encoder or command buffer involvement.

use crate::cpu::ops::outer_combine::{apply_layer_scalar_in_place, outer_post_norm_residual};
use crate::FullPipelineLayer;

/// Apply the outer post-FFN norm (when the arch declares one) followed by
/// the whole-layer `layer_scalar` multiplication. Operates in place on
/// `new_h`. Requires that `new_h` currently holds
/// `h_post_attn + (_1(dense) + _2(moe))`.
///
/// Routes through `cpu::ops::outer_combine` so the GPU MoE path and
/// the CPU MoE path (`vindex/q4k_forward.rs::run_moe_layer_cpu`) share
/// a single implementation of the math. Earlier the two backends had
/// independent transcriptions of the same formula and silently drifted
/// on Gemma 4 26B-A4B.
pub(super) fn apply_outer_combine(
    layer: &FullPipelineLayer,
    new_h: &metal::Buffer,
    h_post_attn: &metal::Buffer,
    hidden: usize,
) {
    // Diagnostic bypass: leave `new_h` as `h_post_attn + _1(dense) + _2(moe)`
    // without outer norm OR layer_scalar — useful for isolating whether
    // this combine step is the broken piece.
    if crate::options::env_flag(crate::options::ENV_SKIP_OUTER_NORM) {
        return;
    }

    // Metal buffers are shared-memory; cast to f32 slices for the
    // shared CPU helper. `hidden` is fixed by the model architecture
    // and the buffers are sized at allocation time, so the slice
    // length is correct by construction.
    let new_h_slice: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(new_h.contents() as *mut f32, hidden) };
    let h_post_attn_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(h_post_attn.contents() as *const f32, hidden) };

    // Step A — outer post-FFN norm on `(h1 + h2)`, residual-added back.
    //
    // Falls back to `post_ffn_norm` (which for Gemma 4 MoE is `_1`) when no
    // un-suffixed outer norm tensor is loaded, so older vindexes still work
    // even if incorrectly. The correct path uses `moe_outer_post_norm` which
    // the extractor now emits for hybrid-MoE architectures.
    if layer.moe_combined_output_norm {
        let outer_w = layer.moe_outer_post_norm.or(layer.post_ffn_norm);
        // Compute `h1+h2 = new_h - h_post_attn` (the delta the GPU
        // built up via dense + moe writes), pass it through the
        // shared helper, then copy the result back into `new_h`.
        let h1_plus_h2: Vec<f32> = new_h_slice
            .iter()
            .zip(h_post_attn_slice.iter())
            .map(|(&n, &ha)| n - ha)
            .collect();
        let combined = outer_post_norm_residual(
            h_post_attn_slice,
            &h1_plus_h2,
            outer_w,
            layer.norm_offset,
            layer.eps,
        );
        new_h_slice.copy_from_slice(&combined);
    }

    // Step B — whole-layer `layer_scalar` multiplication. HF's
    //   `Gemma4TextDecoderLayer.forward` ends with `hidden_states *= self.layer_scalar`
    // which scales BOTH the residual and the FFN delta. A null scalar
    // (0.0) or an identity scalar (1.0) is a no-op.
    apply_layer_scalar_in_place(new_h_slice, layer.layer_scalar);
}
