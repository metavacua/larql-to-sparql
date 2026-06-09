//! DeepSeek V4 partial RoPE — rotates only the tail `n_rot` dims of each
//! head, leaving the leading `n_nope = head_dim - n_rot` dims untouched.
//!
//! Differs from [`super::rope::apply_rope_partial`]:
//! - That function rotates the FIRST `rotary_dim` dims (Qwen-style).
//! - This function rotates the LAST `n_rot` dims (DSv4-style), and
//!   supports both NORMAL (interleaved-pair) and NEOX (split-half) modes,
//!   plus an `inverse` flag that flips the rotation direction (used by
//!   the DSv4 indexer's backward pass).
//!
//! Reference: llama.cpp PR #23122 — `ggml_compute_forward_dsv4_rope_tail`
//! (ggml/src/ggml-cpu/ops.cpp:5973) and `dsv4_apply_rope_tail`
//! (src/models/deepseek4.cpp:455).

use ndarray::Array2;

/// Rotation-pair layout for the rotated tail.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DsV4RopeMode {
    /// Interleaved pairs: `(x[2i], x[2i+1])` within the tail.
    Normal,
    /// Split-half pairs: `(x[i], x[n_rot/2 + i])` within the tail.
    Neox,
}

/// Apply DSv4 tail-RoPE to Q or K.
///
/// `x`: `(seq_len, num_heads * head_dim)`. The leading `n_nope` dims of
/// each head pass through unchanged; the trailing `n_rot` dims are
/// rotated per `mode`.
///
/// `n_rot` must be even and `≤ head_dim`. `position_offset` shifts each
/// row's position (use for KV-cached decode, just like
/// [`super::rope::apply_rope_partial_at`]).
pub fn dsv4_rope_tail(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    n_rot: usize,
    rope_base: f64,
    mode: DsV4RopeMode,
    inverse: bool,
    position_offset: usize,
) -> Array2<f32> {
    assert!(n_rot <= head_dim, "n_rot ({n_rot}) > head_dim ({head_dim})");
    assert!(n_rot % 2 == 0, "n_rot ({n_rot}) must be even");
    assert!(
        x.shape() == [x.shape()[0], num_heads * head_dim],
        "x shape must be (seq_len, num_heads * head_dim)"
    );

    let seq_len = x.shape()[0];
    let mut out = x.clone();

    let n_nope = head_dim - n_rot;
    let half_rot = n_rot / 2;
    let inv_freq: Vec<f64> = (0..half_rot)
        .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / n_rot as f64))
        .collect();

    let sin_sign: f32 = if inverse { -1.0 } else { 1.0 };

    let mut trig: Vec<(f32, f32)> = Vec::with_capacity(half_rot);
    for row in 0..seq_len {
        let pos = position_offset + row;
        trig.clear();
        for i in 0..half_rot {
            let theta = pos as f64 * inv_freq[i];
            trig.push((theta.cos() as f32, (theta.sin() as f32) * sin_sign));
        }
        for h in 0..num_heads {
            let head_off = h * head_dim;
            let tail_off = head_off + n_nope;
            match mode {
                DsV4RopeMode::Normal => {
                    for i in 0..half_rot {
                        let (cos_t, sin_t) = trig[i];
                        let x0 = x[[row, tail_off + 2 * i]];
                        let x1 = x[[row, tail_off + 2 * i + 1]];
                        out[[row, tail_off + 2 * i]] = x0 * cos_t - x1 * sin_t;
                        out[[row, tail_off + 2 * i + 1]] = x0 * sin_t + x1 * cos_t;
                    }
                }
                DsV4RopeMode::Neox => {
                    for i in 0..half_rot {
                        let (cos_t, sin_t) = trig[i];
                        let x0 = x[[row, tail_off + i]];
                        let x1 = x[[row, tail_off + half_rot + i]];
                        out[[row, tail_off + i]] = x0 * cos_t - x1 * sin_t;
                        out[[row, tail_off + half_rot + i]] = x0 * sin_t + x1 * cos_t;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_qk(seq: usize, heads: usize, head_dim: usize) -> Array2<f32> {
        let n = seq * heads * head_dim;
        Array2::from_shape_vec(
            (seq, heads * head_dim),
            (0..n).map(|i| (i as f32 + 1.0) * 0.013).collect(),
        )
        .unwrap()
    }

    #[test]
    fn nope_dims_pass_through_unchanged() {
        // First n_nope dims of each head must equal input exactly.
        let heads = 2;
        let head_dim = 16;
        let n_rot = 8;
        let n_nope = head_dim - n_rot;
        let x = make_qk(3, heads, head_dim);
        let out = dsv4_rope_tail(
            &x,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Neox,
            false,
            0,
        );
        for row in 0..3 {
            for h in 0..heads {
                let off = h * head_dim;
                for i in 0..n_nope {
                    assert_eq!(out[[row, off + i]], x[[row, off + i]], "nope dim {i} h={h}");
                }
            }
        }
    }

    #[test]
    fn tail_dims_preserve_norm_per_head_neox() {
        // RoPE is a rotation — L2 norm of the rotated tail per (row,head) is preserved.
        let heads = 2;
        let head_dim = 16;
        let n_rot = 8;
        let n_nope = head_dim - n_rot;
        let x = make_qk(3, heads, head_dim);
        let out = dsv4_rope_tail(
            &x,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Neox,
            false,
            0,
        );
        for row in 0..3 {
            for h in 0..heads {
                let off = h * head_dim + n_nope;
                let orig: f32 = (0..n_rot).map(|i| x[[row, off + i]].powi(2)).sum();
                let rotd: f32 = (0..n_rot).map(|i| out[[row, off + i]].powi(2)).sum();
                assert!(
                    (orig.sqrt() - rotd.sqrt()).abs() < 1e-4,
                    "tail norm changed row={row} h={h}: {orig} → {rotd}"
                );
            }
        }
    }

    #[test]
    fn tail_dims_preserve_norm_per_head_normal() {
        let heads = 2;
        let head_dim = 16;
        let n_rot = 8;
        let n_nope = head_dim - n_rot;
        let x = make_qk(3, heads, head_dim);
        let out = dsv4_rope_tail(
            &x,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Normal,
            false,
            0,
        );
        for row in 0..3 {
            for h in 0..heads {
                let off = h * head_dim + n_nope;
                let orig: f32 = (0..n_rot).map(|i| x[[row, off + i]].powi(2)).sum();
                let rotd: f32 = (0..n_rot).map(|i| out[[row, off + i]].powi(2)).sum();
                assert!(
                    (orig.sqrt() - rotd.sqrt()).abs() < 1e-4,
                    "tail norm changed row={row} h={h}: {orig} → {rotd}"
                );
            }
        }
    }

    #[test]
    fn inverse_undoes_forward_neox() {
        // dsv4_rope_tail(inverse=true) ∘ dsv4_rope_tail(inverse=false) == identity.
        let heads = 2;
        let head_dim = 16;
        let n_rot = 8;
        let x = make_qk(4, heads, head_dim);
        let fwd = dsv4_rope_tail(
            &x,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Neox,
            false,
            0,
        );
        let back = dsv4_rope_tail(
            &fwd,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Neox,
            true,
            0,
        );
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-4, "inverse failed: {a} vs {b}");
        }
    }

    #[test]
    fn inverse_undoes_forward_normal() {
        let heads = 2;
        let head_dim = 16;
        let n_rot = 8;
        let x = make_qk(4, heads, head_dim);
        let fwd = dsv4_rope_tail(
            &x,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Normal,
            false,
            0,
        );
        let back = dsv4_rope_tail(
            &fwd,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Normal,
            true,
            0,
        );
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-4, "inverse failed: {a} vs {b}");
        }
    }

    #[test]
    fn n_rot_equals_head_dim_neox_matches_full_rope() {
        // When n_rot == head_dim, the NEOX tail-rope reduces to the standard
        // split-half RoPE applied to the whole head — same shape as
        // `apply_rope_partial(fraction=1.0)`.
        use crate::attention::rope::apply_rope_partial;
        let heads = 2;
        let head_dim = 8;
        let x = make_qk(3, heads, head_dim);
        let tail = dsv4_rope_tail(
            &x,
            heads,
            head_dim,
            head_dim, // n_rot == head_dim → n_nope == 0
            10000.0,
            DsV4RopeMode::Neox,
            false,
            0,
        );
        let full = apply_rope_partial(&x, heads, head_dim, 10000.0, 1.0);
        for (a, b) in tail.iter().zip(full.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "tail rope w/ n_rot=head_dim should equal full rope: {a} vs {b}"
            );
        }
    }

    #[test]
    fn different_positions_differ() {
        let heads = 2;
        let head_dim = 16;
        let n_rot = 8;
        let data = vec![0.5f32; 3 * heads * head_dim];
        let x = Array2::from_shape_vec((3, heads * head_dim), data).unwrap();
        let out = dsv4_rope_tail(
            &x,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Neox,
            false,
            0,
        );
        let row0: Vec<f32> = out.row(0).to_vec();
        let row1: Vec<f32> = out.row(1).to_vec();
        let differ = row0
            .iter()
            .zip(row1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "identical inputs at different positions should differ"
        );
    }

    #[test]
    fn position_offset_matches_sequential() {
        // dsv4_rope_tail(single, offset=5) row 0 == dsv4_rope_tail(seq6) row 5.
        let heads = 2;
        let head_dim = 16;
        let n_rot = 8;
        let val = 0.3f32;
        let single = Array2::from_elem((1, heads * head_dim), val);
        let seq6 = Array2::from_elem((6, heads * head_dim), val);
        let out_seq = dsv4_rope_tail(
            &seq6,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Neox,
            false,
            0,
        );
        let out_off = dsv4_rope_tail(
            &single,
            heads,
            head_dim,
            n_rot,
            10000.0,
            DsV4RopeMode::Neox,
            false,
            5,
        );
        let r5: Vec<f32> = out_seq.row(5).to_vec();
        let r0: Vec<f32> = out_off.row(0).to_vec();
        for (a, b) in r5.iter().zip(r0.iter()) {
            assert!((a - b).abs() < 1e-5, "offset=5 mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn shape_preserved() {
        let x = make_qk(5, 4, 32);
        let out = dsv4_rope_tail(&x, 4, 32, 16, 10000.0, DsV4RopeMode::Neox, false, 0);
        assert_eq!(out.shape(), x.shape());
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
