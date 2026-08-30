//! Attention softmax, with optional attention-sink support.
//!
//! Extracted 2026-07-29 from three byte-identical copies in `gqa.rs`.
//! The trigger was adding sink support: editing the same twelve lines in
//! three places is how a kernel ends up correct in two of them.
//!
//! # Attention sinks
//!
//! A sink is a learned per-head logit that competes in the softmax and
//! then receives no output slot. The reference formulation concatenates
//! it, softmaxes, and drops the last column:
//!
//! ```text
//! probs  = softmax([l_0 … l_{n-1}, s])
//! scores = probs[..n]                    # sink column discarded
//! ```
//!
//! so the emitted weights **deliberately sum to less than one** — the
//! sink absorbs the remainder, giving each head a learned "attend to
//! nothing" option. Concatenating is wasteful in a kernel, and the
//! algebra is identical to keeping the sink out of the output buffer and
//! adding its exponential to the denominator, which is what this does.
//!
//! Architectures without sinks pass `None` and get an ordinary softmax
//! summing to one.

/// Softmax `scores` in place, optionally against an attention sink.
///
/// With `sink: None` the result sums to 1. With `sink: Some(s)` the
/// result sums to `1 - p_sink`, where `p_sink` is the share the sink
/// takes — exactly as if `s` had been appended before the softmax and
/// its column dropped afterwards.
///
/// The sink participates in the max used for numerical stability, so a
/// sink far above every real logit stays finite rather than overflowing.
/// An empty slice is a no-op.
pub fn softmax_in_place(scores: &mut [f32], sink: Option<f32>) {
    if scores.is_empty() {
        return;
    }

    let mut max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if let Some(s) = sink {
        max_val = max_val.max(s);
    }

    // f64 accumulation: a long context sums thousands of terms, and the
    // f32 rounding shows up as drift in the tail weights.
    let mut sum = 0.0f64;
    for score in scores.iter_mut() {
        let e = ((*score - max_val) as f64).exp();
        *score = e as f32;
        sum += e;
    }
    if let Some(s) = sink {
        // Contributes to the denominator only — it has no output slot.
        sum += ((s - max_val) as f64).exp();
    }

    let inv_sum = (1.0 / sum) as f32;
    for score in scores.iter_mut() {
        *score *= inv_sum;
    }
}

/// Vectorised f32 softmax for the prefill-shaped batched attention —
/// the physical-plan sibling of [`softmax_in_place`].
///
/// The f64 per-element `exp` above is the right call on the decode
/// path (one short row per step, accuracy over throughput at long
/// context), but a 343-row prefill runs ~26M of them and they were
/// measured as ~0.3 s of TTFA (`docs/tts-funnel.md` 2026-08-10). This
/// variant exponentiates in f32 via Accelerate's `vvexpf` on macOS
/// (SIMD, ~1 ulp of `expf`) with a scalar `f32::exp` fallback
/// elsewhere, and accumulates the denominator in f64 exactly like the
/// reference softmax, keeping the tail-weight drift property.
///
/// No sink support by design: the batched prefill path excludes sinks
/// (they keep the loop + [`softmax_in_place`]). Admission to the
/// speech path is gated on the step-4 token-exact dump replay, not on
/// closeness alone.
pub fn softmax_in_place_f32(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for score in scores.iter_mut() {
        *score -= max_val;
    }
    exp_f32_in_place(scores);
    let sum: f64 = scores.iter().map(|&e| e as f64).sum();
    let inv_sum = (1.0 / sum) as f32;
    for score in scores.iter_mut() {
        *score *= inv_sum;
    }
}

#[cfg(target_os = "macos")]
fn exp_f32_in_place(values: &mut [f32]) {
    // vForce (Accelerate framework, already linked for BLAS): SIMD
    // elementwise expf, documented safe in place.
    unsafe extern "C" {
        fn vvexpf(out: *mut f32, x: *const f32, n: *const i32);
    }
    let n = values.len() as i32;
    unsafe { vvexpf(values.as_mut_ptr(), values.as_ptr(), &n) };
}

#[cfg(not(target_os = "macos"))]
fn exp_f32_in_place(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = value.exp();
    }
}

#[cfg(test)]
mod tests_f32 {
    use super::*;

    #[test]
    fn f32_softmax_tracks_f64_reference() {
        // Deterministic logits with realistic attention-score spread.
        let mut scores: Vec<f32> = (0..343)
            .map(|i| ((i * 37 + 11) % 89) as f32 * 0.25 - 11.0)
            .collect();
        let mut reference = scores.clone();
        softmax_in_place(&mut reference, None);
        softmax_in_place_f32(&mut scores);

        let sum: f64 = scores.iter().map(|&v| v as f64).sum();
        assert!((sum - 1.0).abs() < 1e-5, "must normalise: sum {sum}");
        let worst = scores
            .iter()
            .zip(&reference)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-6,
            "f32 softmax drifted from the f64 reference: worst |delta| {worst:e}"
        );
    }

    #[test]
    fn f32_softmax_handles_edge_shapes() {
        softmax_in_place_f32(&mut []);
        let mut one = [3.5f32];
        softmax_in_place_f32(&mut one);
        assert_eq!(one, [1.0]);
        // Large logits stay finite through the max-subtraction.
        let mut hot = [900.0f32, 899.0, 100.0];
        softmax_in_place_f32(&mut hot);
        assert!(hot.iter().all(|v| v.is_finite()));
        assert!(hot[0] > hot[1] && hot[1] > hot[2]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: concatenate the sink, softmax the whole
    /// vector, then drop the sink column — the formulation in the
    /// published GPT-OSS model code.
    fn reference_concat_softmax_truncate(logits: &[f32], sink: f32) -> Vec<f32> {
        let mut all: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
        all.push(sink as f64);
        let m = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = all.iter().map(|v| (v - m).exp()).collect();
        let total: f64 = exps.iter().sum();
        exps[..logits.len()]
            .iter()
            .map(|e| (e / total) as f32)
            .collect()
    }

    fn sum(v: &[f32]) -> f32 {
        v.iter().sum()
    }

    // ── No sink: ordinary softmax ───────────────────────────────────────

    #[test]
    fn without_sink_sums_to_one() {
        let mut s = [1.0, 2.0, 3.0, 0.5];
        softmax_in_place(&mut s, None);
        assert!((sum(&s) - 1.0).abs() < 1e-6, "sum was {}", sum(&s));
    }

    #[test]
    fn without_sink_is_monotonic_in_the_logits() {
        let mut s = [1.0, 2.0, 3.0];
        softmax_in_place(&mut s, None);
        assert!(s[0] < s[1] && s[1] < s[2]);
    }

    #[test]
    fn equal_logits_give_a_uniform_distribution() {
        let mut s = [2.0; 4];
        softmax_in_place(&mut s, None);
        for v in s {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn empty_scores_is_a_no_op() {
        let mut s: [f32; 0] = [];
        softmax_in_place(&mut s, None);
        softmax_in_place(&mut s, Some(1.0));
    }

    #[test]
    fn single_score_without_sink_is_one() {
        let mut s = [42.0];
        softmax_in_place(&mut s, None);
        assert!((s[0] - 1.0).abs() < 1e-6);
    }

    // ── Sink: agreement with the reference formulation ──────────────────

    #[test]
    fn matches_concat_softmax_truncate() {
        let logits = [0.3f32, -1.2, 2.5, 0.0, 1.1];
        for &sink in &[-2.45f32, 0.0, 2.45, 8.19] {
            let mut got = logits;
            softmax_in_place(&mut got, Some(sink));
            let want = reference_concat_softmax_truncate(&logits, sink);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() < 1e-6,
                    "sink {sink}: index {i} got {g} want {w}"
                );
            }
        }
    }

    #[test]
    fn with_sink_sums_to_less_than_one() {
        let mut s = [1.0, 2.0, 3.0];
        softmax_in_place(&mut s, Some(2.45)); // the mean measured on gpt-oss-20b
        let total = sum(&s);
        assert!(total < 1.0, "sink must divert mass, got {total}");
        assert!(total > 0.0);
    }

    #[test]
    fn a_dominant_sink_takes_nearly_all_the_mass() {
        // The measured maximum on gpt-oss-20b is +8.19; against small
        // logits it should swamp them.
        let mut s = [0.0, 0.1, -0.3];
        softmax_in_place(&mut s, Some(8.19));
        assert!(
            sum(&s) < 0.01,
            "expected near-total diversion, got {}",
            sum(&s)
        );
    }

    #[test]
    fn a_negligible_sink_approaches_the_no_sink_result() {
        let logits = [1.0f32, 2.0, 3.0];
        let mut with = logits;
        softmax_in_place(&mut with, Some(-40.0));
        let mut without = logits;
        softmax_in_place(&mut without, None);
        for (a, b) in with.iter().zip(&without) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn sink_preserves_the_ratios_between_real_logits() {
        // The sink rescales every weight by the same factor, so relative
        // attention between real positions is unchanged.
        let logits = [0.5f32, 1.5, 2.5];
        let mut with = logits;
        softmax_in_place(&mut with, Some(1.0));
        let mut without = logits;
        softmax_in_place(&mut without, None);
        let r_with = with[2] / with[0];
        let r_without = without[2] / without[0];
        assert!((r_with - r_without).abs() < 1e-4);
    }

    // ── Numerical stability ─────────────────────────────────────────────

    #[test]
    fn large_logits_do_not_overflow() {
        let mut s = [1000.0, 1001.0, 999.0];
        softmax_in_place(&mut s, None);
        assert!(s.iter().all(|v| v.is_finite()));
        assert!((sum(&s) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_sink_far_above_every_logit_stays_finite() {
        // The sink must join the max, or `exp(sink - max)` overflows.
        let mut s = [-500.0, -501.0];
        softmax_in_place(&mut s, Some(500.0));
        assert!(s.iter().all(|v| v.is_finite()), "got {s:?}");
        assert!(sum(&s) < 1e-6);
    }

    #[test]
    fn very_negative_logits_stay_finite() {
        let mut s = [-1000.0, -1001.0];
        softmax_in_place(&mut s, None);
        assert!(s.iter().all(|v| v.is_finite()));
        assert!((sum(&s) - 1.0).abs() < 1e-6);
    }
}
