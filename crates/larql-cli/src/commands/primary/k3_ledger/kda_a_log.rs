//! What does K3's `A_log` parameterise — decay per head, or per channel?
//!
//! Pure. **This is a blocker, not a curiosity.** `A_log` sets the KDA decay
//! rate, so reading it under the wrong axis changes every recurrent state in
//! every one of the 69 KDA layers, and the result still looks like text.
//!
//! ## Two self-consistent readings, each breaking exactly one tensor
//!
//! Observed in K3 (`layers.{9,49}.self_attn`, safetensors headers):
//! `q/k/v/g_proj [12288, 7168]`, `dt_bias [12288]`, `o_norm.weight [128]`,
//! `A_log [128]`. `config.linear_attn_config` says `head_dim 128, num_heads 96`.
//!
//! `fla.ops.kda` documents `A_log` as shape `[HV]` and `dt_bias` as `[HV * K]`,
//! and its gate computes `-exp(A_log).view(H,1) * softplus(g + dt_bias.view(H,-1))`
//! — so `A_log` is indexed **per head** and broadcast across the channel axis.
//!
//! | reading | head_dim | num_heads | proj | dt_bias | A_log | o_norm |
//! |---|---|---|---|---|---|---|
//! | **A** config as written | 128 | 96 | 12288 ok | 12288 ok | **[96] != [128]** | 128 ok |
//! | **B** axes swapped | 96 | 128 | 12288 ok | 12288 ok | 128 ok | **[96] != [128]** |
//!
//! Both explain the two big tensors. Neither explains all four. **Header shapes
//! cannot settle this** — that is the finding, and why it fails closed here.
//!
//! ## The cross-check that makes it specific to K3
//!
//! `Kimi-Linear-48B-A3B-Instruct` — same architecture, released, working —
//! has `head_dim 128, num_heads 32`, `q/v_proj [4096, 2304]`, `dt_bias [4096]`,
//! `o_norm [128]`, and **`A_log [1, 1, 32, 1]`: 32 elements = `num_heads`**,
//! satisfying `fla`'s contract under reading A. K3 is the checkpoint that
//! deviates, and it also *changed the tensor's rank*, which suggests a
//! deliberate reparameterisation rather than a resize.
//!
//! ## Where the danger actually lives
//!
//! In the reference stack this fails **loudly**: `A_log.view(96, 1)` on a
//! 128-element tensor raises. The silent-corruption risk is a reimplementation
//! that broadcasts leniently — which is exactly what a LARQL adapter would do if
//! it indexed by head without asserting the length. Hence `resolve`, which
//! refuses to guess.

use serde::Serialize;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// How `A_log` maps onto the `[H, K]` decay grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ALogAxis {
    /// One entry per head, broadcast across channels. `fla`'s documented form.
    PerHead,
    /// One entry per channel within a head, shared across heads.
    PerChannel,
}

/// The outcome of trying to name the axis from shapes alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ALogVerdict {
    /// Exactly one axis matches the observed length.
    Determined(ALogAxis),
    /// `num_heads == head_dim`, so the length names both. Needs an oracle.
    AmbiguousBySymmetry,
    /// The length matches neither axis.
    Unexplained,
}

impl ALogVerdict {
    /// Fail closed: only a `Determined` verdict may be acted on.
    pub fn axis(self) -> Option<ALogAxis> {
        match self {
            Self::Determined(a) => Some(a),
            _ => None,
        }
    }
}

/// Name the axis from the observed element count, or refuse.
pub fn classify(numel: u64, num_heads: u64, head_dim: u64) -> ALogVerdict {
    if num_heads == head_dim {
        return if numel == num_heads {
            ALogVerdict::AmbiguousBySymmetry
        } else {
            ALogVerdict::Unexplained
        };
    }
    match (numel == num_heads, numel == head_dim) {
        (true, false) => ALogVerdict::Determined(ALogAxis::PerHead),
        (false, true) => ALogVerdict::Determined(ALogAxis::PerChannel),
        _ => ALogVerdict::Unexplained,
    }
}

/// The KDA decay, `-exp(A_log) * softplus(g + dt_bias)`, under a chosen axis.
///
/// `g` and `dt_bias` are row-major `[heads, channels]`. Only the `A_log`
/// indexing differs between axes — which is the whole point: it is a one-index
/// change that no shape check downstream will catch.
pub fn decay(
    axis: ALogAxis,
    a_log: &[f64],
    g: &[f64],
    dt_bias: &[f64],
    heads: usize,
    channels: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(heads * channels);
    for h in 0..heads {
        for k in 0..channels {
            let i = h * channels + k;
            let a = match axis {
                ALogAxis::PerHead => a_log[h],
                ALogAxis::PerChannel => a_log[k],
            };
            let x = g[i] + dt_bias[i];
            // softplus, in the numerically stable form.
            let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
            out.push(-a.exp() * sp);
        }
    }
    out
}

/// A fixture on which the two axes are guaranteed to disagree.
///
/// **The standing rule this encodes:** an `A_log` interpretation may only be
/// adopted after it has been shown to differ from the plausible alternative on
/// a concrete input. A square, uniform or symmetric fixture cannot show that,
/// so a passing oracle test on one proves nothing.
pub fn asymmetric_fixture(heads: usize, channels: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = heads.max(channels);
    // Strictly increasing, so no two indices share a value.
    let a_log: Vec<f64> = (0..n).map(|i| -1.0 + i as f64 * 0.25).collect();
    let g = vec![0.5; heads * channels];
    let dt_bias: Vec<f64> = (0..heads * channels).map(|i| 0.01 * i as f64).collect();
    (a_log, g, dt_bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// K3's measured geometry under reading A (the config as written).
    const K3_NUM_HEADS: u64 = 96;
    const K3_HEAD_DIM: u64 = 128;
    const K3_A_LOG_NUMEL: u64 = 128;
    /// Kimi-Linear-48B, the working sibling.
    const KL_NUM_HEADS: u64 = 32;
    const KL_HEAD_DIM: u64 = 128;
    const KL_A_LOG_NUMEL: u64 = 32;

    #[test]
    fn the_working_sibling_is_per_head_as_fla_documents() {
        assert_eq!(
            classify(KL_A_LOG_NUMEL, KL_NUM_HEADS, KL_HEAD_DIM),
            ALogVerdict::Determined(ALogAxis::PerHead)
        );
    }

    #[test]
    fn k3_classifies_as_per_channel_which_contradicts_fla() {
        // Not a green light: it means the checkpoint disagrees with the kernel
        // contract its own modeling file calls into.
        assert_eq!(
            classify(K3_A_LOG_NUMEL, K3_NUM_HEADS, K3_HEAD_DIM),
            ALogVerdict::Determined(ALogAxis::PerChannel)
        );
    }

    #[test]
    fn the_axis_swap_reading_would_make_a_log_per_head_again() {
        // Reading B: head_dim 96, num_heads 128. A_log fits, o_norm breaks.
        // Both readings survive the big tensors, which is why shapes cannot
        // decide and this module refuses to.
        assert_eq!(
            classify(K3_A_LOG_NUMEL, 128, 96),
            ALogVerdict::Determined(ALogAxis::PerHead)
        );
    }

    #[test]
    fn a_square_head_grid_is_ambiguous_and_yields_no_axis() {
        let v = classify(64, 64, 64);
        assert_eq!(v, ALogVerdict::AmbiguousBySymmetry);
        assert!(v.axis().is_none(), "must fail closed, not guess");
    }

    #[test]
    fn a_length_matching_neither_axis_is_unexplained() {
        let v = classify(77, 96, 128);
        assert_eq!(v, ALogVerdict::Unexplained);
        assert!(v.axis().is_none());
    }

    #[test]
    fn only_a_determined_verdict_hands_back_an_axis() {
        assert_eq!(
            classify(KL_A_LOG_NUMEL, KL_NUM_HEADS, KL_HEAD_DIM).axis(),
            Some(ALogAxis::PerHead)
        );
    }

    #[test]
    fn the_two_axes_disagree_on_the_asymmetric_fixture() {
        // The discriminator itself must be shown to discriminate, or an oracle
        // run that "passes" proves nothing.
        let (h, k) = (3usize, 4usize);
        let (a, g, dt) = asymmetric_fixture(h, k);
        let per_head = decay(ALogAxis::PerHead, &a, &g, &dt, h, k);
        let per_chan = decay(ALogAxis::PerChannel, &a, &g, &dt, h, k);
        assert_eq!(per_head.len(), h * k);
        let differing = per_head
            .iter()
            .zip(&per_chan)
            .filter(|(x, y)| (*x - *y).abs() > 1e-12)
            .count();
        // They coincide exactly on the diagonal (h == k) and nowhere else, so
        // the fixture must be RECTANGULAR for the diagonal to stay a minority.
        assert_eq!(
            differing,
            h * k - h.min(k),
            "expected all but the {} diagonal cells to differ",
            h.min(k)
        );
    }

    #[test]
    fn the_axes_coincide_where_the_fixture_is_diagonal() {
        // h == k is exactly the case that hides the bug: on the diagonal the
        // two readings pick the same entry. A square fixture is not a test.
        let n = 4usize;
        let (a, g, dt) = asymmetric_fixture(n, n);
        let ph = decay(ALogAxis::PerHead, &a, &g, &dt, n, n);
        let pc = decay(ALogAxis::PerChannel, &a, &g, &dt, n, n);
        for i in 0..n {
            let d = i * n + i;
            assert!((ph[d] - pc[d]).abs() < 1e-12, "diagonal must coincide");
        }
    }

    #[test]
    fn decay_is_negative_and_scales_with_a_log() {
        // Sanity on the arithmetic itself: -exp(A) * softplus(x) < 0, and a
        // larger A_log means faster decay.
        let (h, k) = (2usize, 2usize);
        let g = vec![1.0; 4];
        let dt = vec![0.0; 4];
        let small = decay(ALogAxis::PerHead, &[0.0, 0.0], &g, &dt, h, k);
        let large = decay(ALogAxis::PerHead, &[1.0, 1.0], &g, &dt, h, k);
        assert!(small.iter().all(|v| *v < 0.0));
        assert!(large[0] < small[0]);
    }

    #[test]
    fn softplus_stays_finite_on_a_large_input() {
        let out = decay(ALogAxis::PerHead, &[0.0], &[1e3], &[0.0], 1, 1);
        assert!(out[0].is_finite(), "got {}", out[0]);
    }
}
