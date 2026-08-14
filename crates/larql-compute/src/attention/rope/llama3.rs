//! `llama3` RoPE scaling — wavelength-band frequency adjustment.
//!
//! Mirrors HF's `_compute_llama3_parameters` in `modeling_rope_utils.py`.
//! Unlike [`super::yarn`], this family adjusts frequencies only — there is no
//! amplitude term, so `cos`/`sin` keep unit scale.

pub use larql_models::Llama3RopeScaling;

/// Compute wavelength-adjusted `inv_freq[i]` for each rotary half-pair
/// from the standard `1 / base^(2i/d)` baseline:
///
/// - `wavelen[i] = 2π / inv_freq[i]`
/// - if `wavelen < high_freq_wavelen` (fast rotation): unchanged
/// - if `wavelen > low_freq_wavelen`  (slow rotation): divided by `factor`
/// - otherwise: smooth interpolation between the two regimes
pub fn apply_llama3_inv_freq(scaling: &Llama3RopeScaling, inv_freq: &[f64]) -> Vec<f64> {
    let low_freq_wavelen = scaling.original_max_position_embeddings / scaling.low_freq_factor;
    let high_freq_wavelen = scaling.original_max_position_embeddings / scaling.high_freq_factor;
    inv_freq
        .iter()
        .map(|&inv| {
            let wavelen = std::f64::consts::TAU / inv;
            if wavelen < high_freq_wavelen {
                inv
            } else if wavelen > low_freq_wavelen {
                inv / scaling.factor
            } else {
                let smooth = (scaling.original_max_position_embeddings / wavelen
                    - scaling.low_freq_factor)
                    / (scaling.high_freq_factor - scaling.low_freq_factor);
                (1.0 - smooth) * inv / scaling.factor + smooth * inv
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // HF's `_compute_llama3_parameters` partitions the rotary frequency band
    // into three regimes by wavelength: fast (passthrough), slow (divided by
    // factor), and a smooth interpolation between. These tests pin each
    // regime against a hand-computed value so a future refactor of the
    // formula gets caught here, not by a 0.5 % bits/char regression caught
    // hours later by `shannon verify`.

    fn llama3_default() -> Llama3RopeScaling {
        // Llama-3.2-1B canonical config.
        Llama3RopeScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        }
    }

    #[test]
    fn llama3_fast_freq_passthrough() {
        let s = llama3_default();
        // wavelen = 2π / inv → for inv = 1.0, wavelen ≈ 6.28, which is
        // well below high_freq_wavelen = 8192/4 = 2048. Fast regime →
        // passthrough unchanged.
        let out = apply_llama3_inv_freq(&s, &[1.0]);
        assert!((out[0] - 1.0).abs() < 1e-12, "fast freq must be unchanged");
    }

    #[test]
    fn llama3_slow_freq_divided_by_factor() {
        let s = llama3_default();
        // wavelen = 2π / inv → for inv = 1e-5, wavelen ≈ 628_318, well
        // above low_freq_wavelen = 8192/1 = 8192. Slow regime →
        // `inv / factor`.
        let inv = 1e-5_f64;
        let out = apply_llama3_inv_freq(&s, &[inv]);
        assert!(
            (out[0] - inv / s.factor).abs() < 1e-15,
            "slow freq must be inv/factor; got {} vs expected {}",
            out[0],
            inv / s.factor
        );
    }

    #[test]
    fn llama3_medium_freq_smooth_interpolation() {
        let s = llama3_default();
        // Pick inv so wavelen sits squarely between high_freq_wavelen
        // (2048) and low_freq_wavelen (8192). With wavelen = 4096
        // (geometric midpoint area):
        //   inv = 2π / 4096 ≈ 0.001534
        //   smooth = (8192/4096 - 1) / (4 - 1) = (2 - 1) / 3 = 0.333...
        //   expected = (1 - 1/3) * inv/32 + (1/3) * inv
        let inv = std::f64::consts::TAU / 4096.0;
        let smooth = (8192.0 / (std::f64::consts::TAU / inv) - 1.0) / (4.0 - 1.0);
        let expected = (1.0 - smooth) * inv / s.factor + smooth * inv;
        let out = apply_llama3_inv_freq(&s, &[inv]);
        assert!(
            (out[0] - expected).abs() < 1e-12,
            "medium-freq smoothing wrong: got {} vs expected {}",
            out[0],
            expected
        );
        // And: result must be bracketed by the slow-regime and fast-regime
        // values, since smoothing is a convex combination.
        assert!(
            out[0] > inv / s.factor && out[0] < inv,
            "medium-freq result must sit between slow (inv/factor) and fast (inv)"
        );
    }

    #[test]
    fn llama3_apply_preserves_length() {
        let s = llama3_default();
        let inv_freq: Vec<f64> = (0..32)
            .map(|i| 1.0 / (10000.0_f64.powf(i as f64 / 32.0)))
            .collect();
        let out = apply_llama3_inv_freq(&s, &inv_freq);
        assert_eq!(out.len(), inv_freq.len());
        assert!(out.iter().all(|v| v.is_finite()));
        // Monotonicity: scaled inv_freq is still monotonically decreasing
        // because each band's transform preserves order within and across
        // bands (slow regime divides by a constant, fast regime passes
        // through, smoothing is monotonic in inv).
        let mono = out.windows(2).all(|w| w[0] >= w[1]);
        assert!(mono, "llama3-scaled inv_freq lost monotonicity");
    }
}
