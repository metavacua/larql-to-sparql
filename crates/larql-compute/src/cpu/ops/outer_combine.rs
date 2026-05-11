//! Outer post-FFN norm + residual + whole-layer `layer_scalar` —
//! shared between the CPU MoE forward path and Metal's GPU MoE
//! dispatch so the two never silently drift in their final-step math.
//!
//! Metal's `metal/decode/moe_combine.rs::apply_outer_combine` is the
//! reference. Both backends arrive at the same point — `h_post_attn`
//! and `h1 + h2 = _1(dense) + _2(moe)` — and need to apply
//!
//!   h_out = (h_post_attn + outer_norm(h1+h2)) * layer_scalar
//!
//! where `outer_norm(x) = x / rms(x) * (w + norm_offset)`. Pulling
//! the math here means a single source of truth: when CPU output
//! disagrees with Metal output, the bug isn't in the combine step.

/// Combine the dense and MoE branches into the final residual:
///
///   h_out[i] = h_post_attn[i] + outer_norm(h1_plus_h2)[i]   if `outer_w` Some
///   h_out[i] = h_post_attn[i] + h1_plus_h2[i]               otherwise
///
/// `outer_norm(x) = x / rms(x) * (w + norm_offset)` with
/// `rms(x) = sqrt(sum(x²)/n + eps)`. f32 arithmetic to match the
/// Metal kernel exactly — using f64 here would silently put the CPU
/// path out of bit-exact agreement with the GPU path.
///
/// `outer_w == None` means the architecture either doesn't ship an
/// outer norm or the vindex didn't load one; in either case the
/// residual stream is just `h_post_attn + (h1+h2)` (matches Metal's
/// `if let Some(outer_w) = outer_w` guard which leaves new_h
/// unchanged when the weight is absent).
pub fn outer_post_norm_residual(
    h_post_attn: &[f32],
    h1_plus_h2: &[f32],
    outer_w: Option<&[f32]>,
    norm_offset: f32,
    eps: f32,
) -> Vec<f32> {
    let hidden = h_post_attn.len();
    debug_assert_eq!(h1_plus_h2.len(), hidden);
    let mut out = vec![0.0f32; hidden];
    match outer_w {
        Some(w) => {
            debug_assert_eq!(w.len(), hidden);
            // RMS computed on `h1+h2` (the Gemma 4 outer norm operates
            // on the *delta*, not on `h_post_attn + delta`).
            let rms = rms_f32(h1_plus_h2, eps);
            for i in 0..hidden {
                out[i] = h_post_attn[i] + h1_plus_h2[i] / rms * (w[i] + norm_offset);
            }
        }
        None => {
            for i in 0..hidden {
                out[i] = h_post_attn[i] + h1_plus_h2[i];
            }
        }
    }
    out
}

/// In-place whole-residual `layer_scalar` multiplication.
/// No-op when `layer_scalar` is 0.0 (absent / unloaded — multiplying
/// would zero the layer output, collapsing generation) or 1.0
/// (identity). Matches Metal's `apply_whole_layer_scalar`.
pub fn apply_layer_scalar_in_place(h_out: &mut [f32], layer_scalar: f32) {
    if layer_scalar == 0.0 || layer_scalar == 1.0 {
        return;
    }
    for v in h_out.iter_mut() {
        *v *= layer_scalar;
    }
}

/// Plain f32 RMS norm denominator: sqrt(sum(x²)/n + eps).
///
/// f32 accumulation is intentional — Metal's GPU shader accumulates
/// in f32 too, and the CPU MoE path needs to match Metal bit-for-bit
/// (within rounding) to be a credible parity reference. Using f64
/// here would put CPU ahead of Metal in precision, which made past
/// debugging confusing because "CPU is more accurate" hid which
/// branch had a real semantic bug.
#[inline]
fn rms_f32(x: &[f32], eps: f32) -> f32 {
    let n = x.len() as f32;
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    (sum_sq / n + eps).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outer_post_norm_residual_matches_handwritten_metal_logic() {
        // Reference: handwritten copy of Metal's `apply_outer_norm`
        // applied to the same inputs. Any divergence here means the
        // shared helper has drifted from Metal — the exact bug class
        // we're trying to prevent.
        let h_post_attn = vec![1.0f32, 2.0, 3.0, 4.0];
        let h1_plus_h2 = vec![0.5f32, -0.5, 1.0, -1.0];
        let outer_w = vec![1.5f32, 0.5, 2.0, 1.0];
        let eps = 1e-6f32;
        let offset = 0.0f32;

        let got = outer_post_norm_residual(&h_post_attn, &h1_plus_h2, Some(&outer_w), offset, eps);

        // Reference implementation: literal Metal apply_outer_norm.
        let n = h1_plus_h2.len() as f32;
        let sum_sq: f32 = h1_plus_h2.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n + eps).sqrt();
        let expected: Vec<f32> = h_post_attn
            .iter()
            .zip(&h1_plus_h2)
            .zip(&outer_w)
            .map(|((&ha, &c), &w)| ha + c / rms * (w + offset))
            .collect();

        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "idx {i}: got {g}, expected {e}, diff {}",
                (g - e).abs()
            );
        }
    }

    #[test]
    fn outer_post_norm_residual_skips_norm_when_weight_none() {
        // No outer norm → output is just `h_post_attn + h1_plus_h2`.
        // Mirrors Metal's `if let Some(outer_w) = outer_w` guard —
        // when the vindex didn't ship the outer norm vector, neither
        // backend should silently apply an identity-scale norm.
        let h_post_attn = vec![1.0f32, 2.0, 3.0];
        let h1_plus_h2 = vec![0.1f32, 0.2, 0.3];

        let got = outer_post_norm_residual(&h_post_attn, &h1_plus_h2, None, 0.0, 1e-6);
        assert_eq!(got, vec![1.1, 2.2, 3.3]);
    }

    #[test]
    fn norm_offset_is_added_to_each_weight() {
        // Gemma 2/3 ships RMSNorm weights as (learned - 1.0) so the
        // forward pass must add `norm_offset = 1.0` per element.
        let h_post_attn = vec![0.0f32, 0.0, 0.0, 0.0];
        let h1_plus_h2 = vec![1.0f32; 4]; // rms = 1.0 (modulo eps)
        let outer_w = vec![0.0f32; 4]; // all-zero learned weight
        let offset = 1.0f32;

        let got = outer_post_norm_residual(&h_post_attn, &h1_plus_h2, Some(&outer_w), offset, 1e-6);
        // After norm: x/rms = 1.0 (rms ≈ 1), times (0 + 1) = 1, plus
        // h_post_attn (0). So all 1.0 within eps tolerance.
        for v in &got {
            assert!((v - 1.0).abs() < 1e-3, "got {v}, expected ~1.0");
        }
    }

    #[test]
    fn apply_layer_scalar_in_place_skips_identity_and_zero() {
        let mut h = vec![1.0f32, 2.0, 3.0];
        let original = h.clone();

        apply_layer_scalar_in_place(&mut h, 1.0);
        assert_eq!(h, original, "layer_scalar=1.0 must be identity");

        apply_layer_scalar_in_place(&mut h, 0.0);
        assert_eq!(h, original, "layer_scalar=0.0 must skip (would collapse)");
    }

    #[test]
    fn apply_layer_scalar_in_place_multiplies() {
        let mut h = vec![1.0f32, 2.0, 3.0];
        apply_layer_scalar_in_place(&mut h, 2.5);
        assert_eq!(h, vec![2.5, 5.0, 7.5]);
    }
}
