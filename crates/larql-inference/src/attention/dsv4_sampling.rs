//! DSv4 sampling primitives — temperature scaling, top-p nucleus
//! filter, and seeded probabilistic sampling.
//!
//! These compose with [`super::dsv4_topk_logits`] to give a complete
//! "logits → sampled token" pipeline:
//!
//! ```text
//! 1. Run the model forward → logits (n_tokens, n_vocab)
//! 2. Pick the last position's row             → logits[t_last]
//! 3. dsv4_apply_temperature(row, T)           → scaled logits
//! 4. dsv4_topk_logits(.., k=N) on the row     → top-N entries with softmax probs
//! 5. dsv4_topp_filter(entries, p)             → nucleus subset
//! 6. dsv4_sample_token(filtered, &mut rng)    → one u32 token id
//! ```
//!
//! Steps 1-2 live elsewhere; this module covers 3, 5, and 6. Step 4 is
//! the existing top-k primitive.
//!
//! All functions are pure (no I/O, no model state) — sampling
//! receives the RNG via `&mut impl Rng` so callers control determinism
//! by seeding (e.g. `StdRng::seed_from_u64(42)`).

use ndarray::{Array1, ArrayView1};
use rand::Rng;

use super::dsv4_topk_logits::{dsv4_topk_logits, TopKEntry};

/// Apply temperature to a single row of logits, returning a new row.
///
/// `T = 1.0` is identity. `T < 1` sharpens (top entries get even more
/// probability after softmax); `T > 1` flattens (more uniform).
///
/// `T <= 0` is invalid and panics — temperatures of exactly zero
/// correspond to argmax, which the caller should do via
/// [`super::dsv4_topk_logits::dsv4_argmax_logits`] instead of going
/// through softmax (a divide-by-zero would otherwise produce ±inf).
pub fn dsv4_apply_temperature(logits: ArrayView1<f32>, temperature: f32) -> Array1<f32> {
    assert!(
        temperature > 0.0,
        "temperature must be > 0; for greedy decode use dsv4_argmax_logits instead"
    );
    let inv = 1.0 / temperature;
    logits.iter().map(|&v| v * inv).collect()
}

/// Nucleus (top-p) filter over a top-k entry list. Keeps the smallest
/// prefix of `entries` (already sorted by descending logit/probability)
/// whose cumulative `probability` first reaches `p`. At minimum one
/// entry is always retained, even if its probability alone exceeds `p`.
///
/// `p` must be in `(0.0, 1.0]`.
///
/// Returns a `Vec<TopKEntry>` borrowing nothing — the result is owned
/// so callers can re-normalize or mutate it freely.
pub fn dsv4_topp_filter(entries: &[TopKEntry], p: f32) -> Vec<TopKEntry> {
    assert!(p > 0.0 && p <= 1.0, "p must be in (0, 1]");
    assert!(!entries.is_empty(), "entries must be non-empty");
    let mut cumulative = 0.0_f32;
    let mut out = Vec::with_capacity(entries.len());
    for &e in entries {
        out.push(e);
        cumulative += e.probability;
        if cumulative >= p {
            break;
        }
    }
    out
}

/// Sample one token from a top-k/top-p entry list by their stored
/// probabilities (renormalized to sum to 1 over the candidate set).
///
/// Uses inverse-CDF sampling: draw `u ~ Uniform(0, total_prob)`, walk
/// the entries accumulating probability mass, and return the first
/// entry whose cumulative mass exceeds `u`.
///
/// Caller passes their RNG to control reproducibility. Using
/// `rand::rngs::StdRng::seed_from_u64(seed)` gives deterministic
/// output; `rand::thread_rng()` gives non-deterministic but
/// statistically valid sampling.
pub fn dsv4_sample_token<R: Rng + ?Sized>(entries: &[TopKEntry], rng: &mut R) -> u32 {
    assert!(!entries.is_empty(), "entries must be non-empty");
    let total: f32 = entries.iter().map(|e| e.probability).sum();
    assert!(total > 0.0, "total probability must be > 0");
    let u: f32 = rng.gen_range(0.0_f32..total);
    let mut acc = 0.0_f32;
    for e in entries {
        acc += e.probability;
        if u < acc {
            return e.token_id;
        }
    }
    // Numerical drift fallback (u very close to total): return the
    // last entry. Guaranteed valid since entries is non-empty.
    entries[entries.len() - 1].token_id
}

/// One-call sampling configuration. Mirrors the standard inference
/// API knobs:
/// - `greedy`: bypass softmax and pick argmax of raw logits. When
///   true, every other field is ignored.
/// - `temperature`: must be > 0 when greedy is false. T < 1 sharpens,
///   T > 1 flattens, T = 1 is no-op scaling.
/// - `top_k`: max candidates to consider (1..=n_vocab). Smaller k →
///   tighter cutoff; k = 1 is equivalent to greedy.
/// - `top_p`: nucleus filter on the top-k subset (0.0 < p ≤ 1.0). The
///   smallest prefix whose cumulative probability reaches `top_p` is
///   kept (always ≥ 1 entry).
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub greedy: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl SamplingConfig {
    /// Greedy / deterministic config — argmax of raw logits.
    pub fn greedy() -> Self {
        Self {
            greedy: true,
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        }
    }

    /// "Default sampling" preset: T = 0.7, top_k = 40, top_p = 0.95 —
    /// the standard llama.cpp/HF chat-decoding defaults that the user
    /// of this API is most likely expecting.
    pub fn default_sampling() -> Self {
        Self {
            greedy: false,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
        }
    }
}

/// Pick the next token from a single row of model logits, applying
/// temperature → top-k → top-p → sample (or argmax if greedy).
///
/// `logits_row` is one position's logits — typically
/// `logits.row(last_position_index)` from the model forward output.
///
/// Returns one `u32` token id.
pub fn dsv4_sample_next_token<R: Rng + ?Sized>(
    logits_row: ArrayView1<f32>,
    config: SamplingConfig,
    rng: &mut R,
) -> u32 {
    if config.greedy {
        // Greedy: argmax of raw logits. No softmax needed; cheaper and
        // avoids any divide-by-temperature concern.
        return argmax_row(logits_row);
    }

    let scaled = dsv4_apply_temperature(logits_row, config.temperature);

    // dsv4_topk_logits operates on (n_tokens, n_vocab); wrap our 1D
    // row into a 1-row 2D view.
    let n_vocab = scaled.len();
    let scaled_2d =
        ndarray::Array2::<f32>::from_shape_vec((1, n_vocab), scaled.to_vec()).expect("reshape");
    let mut topk = dsv4_topk_logits(scaled_2d.view(), config.top_k.min(n_vocab));

    // Top-p over the row's top-k. `dsv4_topp_filter` requires
    // non-empty input which the top-k call guarantees (k ≥ 1).
    let row = topk.pop().expect("topk returned no rows");
    let nucleus = dsv4_topp_filter(&row, config.top_p);

    dsv4_sample_token(&nucleus, rng)
}

/// 1D argmax — first index achieving max value. Used by the greedy
/// branch of `dsv4_sample_next_token` so the row doesn't have to be
/// wrapped into a 2D array just to call the top-k argmax helper.
fn argmax_row(row: ArrayView1<f32>) -> u32 {
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i: u32 = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::super::dsv4_topk_logits::dsv4_topk_logits;
    use super::*;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Temperature 1.0 is identity.
    #[test]
    fn temperature_one_is_identity() {
        let row = Array1::<f32>::from_vec(vec![0.5, 1.0, -0.25, 3.0]);
        let scaled = dsv4_apply_temperature(row.view(), 1.0);
        for (a, b) in row.iter().zip(scaled.iter()) {
            assert_eq!(a, b);
        }
    }

    /// Temperature 2.0 halves logits; 0.5 doubles them.
    #[test]
    fn temperature_scales_logits_correctly() {
        let row = Array1::<f32>::from_vec(vec![2.0, 4.0]);
        let half = dsv4_apply_temperature(row.view(), 2.0);
        assert_eq!(half[0], 1.0);
        assert_eq!(half[1], 2.0);
        let double = dsv4_apply_temperature(row.view(), 0.5);
        assert_eq!(double[0], 4.0);
        assert_eq!(double[1], 8.0);
    }

    /// Temperature < 1 sharpens distribution; > 1 flattens.
    /// Verified by softmaxing scaled logits through dsv4_topk_logits.
    #[test]
    fn temperature_sharpens_and_flattens_distribution() {
        // 3 entries: middle one wins.
        let row = Array1::<f32>::from_vec(vec![1.0, 3.0, 2.0]);

        let sharp = dsv4_apply_temperature(row.view(), 0.1); // 10x logits
        let flat = dsv4_apply_temperature(row.view(), 10.0); // 0.1x logits

        let sharp_logits = Array2::<f32>::from_shape_vec((1, 3), sharp.to_vec()).unwrap();
        let flat_logits = Array2::<f32>::from_shape_vec((1, 3), flat.to_vec()).unwrap();
        let sharp_top = dsv4_topk_logits(sharp_logits.view(), 3);
        let flat_top = dsv4_topk_logits(flat_logits.view(), 3);

        // Sharp: top-1 should approach 1.0; flat: top-1 should be only
        // marginally above the others.
        assert!(
            sharp_top[0][0].probability > 0.99,
            "sharpened top-1 {} not dominant",
            sharp_top[0][0].probability
        );
        assert!(
            flat_top[0][0].probability < 0.5,
            "flat top-1 {} not flat enough",
            flat_top[0][0].probability
        );
    }

    #[test]
    #[should_panic(expected = "temperature must be > 0")]
    fn temperature_zero_panics() {
        let row = Array1::<f32>::from_vec(vec![1.0, 2.0]);
        let _ = dsv4_apply_temperature(row.view(), 0.0);
    }

    /// Top-p with p = 1.0 returns all entries (no filter).
    #[test]
    fn topp_p_one_returns_all() {
        let entries = vec![
            TopKEntry {
                token_id: 0,
                logit: 10.0,
                probability: 0.6,
            },
            TopKEntry {
                token_id: 1,
                logit: 9.0,
                probability: 0.3,
            },
            TopKEntry {
                token_id: 2,
                logit: 8.0,
                probability: 0.1,
            },
        ];
        let filtered = dsv4_topp_filter(&entries, 1.0);
        assert_eq!(filtered.len(), 3);
    }

    /// Top-p stops once cumulative reaches `p`.
    #[test]
    fn topp_stops_at_cumulative_p() {
        let entries = vec![
            TopKEntry {
                token_id: 0,
                logit: 10.0,
                probability: 0.6,
            },
            TopKEntry {
                token_id: 1,
                logit: 9.0,
                probability: 0.3,
            },
            TopKEntry {
                token_id: 2,
                logit: 8.0,
                probability: 0.1,
            },
        ];
        // p=0.5 → first entry (0.6 ≥ 0.5) stops there.
        assert_eq!(dsv4_topp_filter(&entries, 0.5).len(), 1);
        // p=0.7 → first two entries (cum 0.9 ≥ 0.7).
        assert_eq!(dsv4_topp_filter(&entries, 0.7).len(), 2);
        // p=0.95 → all three (cum 1.0 ≥ 0.95).
        assert_eq!(dsv4_topp_filter(&entries, 0.95).len(), 3);
    }

    /// Top-p always keeps at least one entry even if p is very small.
    #[test]
    fn topp_always_keeps_at_least_one() {
        let entries = vec![TopKEntry {
            token_id: 7,
            logit: 1.0,
            probability: 0.4,
        }];
        // Even with the single entry's prob (0.4) above any p ≤ 0.4,
        // we keep it.
        let filtered = dsv4_topp_filter(&entries, 0.01);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].token_id, 7);
    }

    #[test]
    #[should_panic(expected = "p must be in (0, 1]")]
    fn topp_p_zero_panics() {
        let entries = vec![TopKEntry {
            token_id: 0,
            logit: 1.0,
            probability: 1.0,
        }];
        let _ = dsv4_topp_filter(&entries, 0.0);
    }

    /// Seeded RNG → deterministic sample. Same seed, same token.
    #[test]
    fn sample_with_seeded_rng_is_deterministic() {
        let entries = vec![
            TopKEntry {
                token_id: 100,
                logit: 5.0,
                probability: 0.5,
            },
            TopKEntry {
                token_id: 200,
                logit: 4.5,
                probability: 0.3,
            },
            TopKEntry {
                token_id: 300,
                logit: 4.0,
                probability: 0.2,
            },
        ];
        let mut rng_a = StdRng::seed_from_u64(42);
        let mut rng_b = StdRng::seed_from_u64(42);
        let a = dsv4_sample_token(&entries, &mut rng_a);
        let b = dsv4_sample_token(&entries, &mut rng_b);
        assert_eq!(a, b, "same seed → same sample");
    }

    /// Highly-skewed distribution: 99% probability mass on one entry.
    /// Across many trials the top entry should be selected ~99% of the time.
    #[test]
    fn sample_respects_probability_distribution() {
        let entries = vec![
            TopKEntry {
                token_id: 1,
                logit: 10.0,
                probability: 0.99,
            },
            TopKEntry {
                token_id: 2,
                logit: 5.0,
                probability: 0.005,
            },
            TopKEntry {
                token_id: 3,
                logit: 5.0,
                probability: 0.005,
            },
        ];
        let mut rng = StdRng::seed_from_u64(0xc0ffee);
        let mut hits = 0_u32;
        let trials = 10_000;
        for _ in 0..trials {
            if dsv4_sample_token(&entries, &mut rng) == 1 {
                hits += 1;
            }
        }
        let ratio = hits as f32 / trials as f32;
        // 0.99 ± 0.01 is a wide enough band that random variation
        // won't tickle it but a broken distribution will.
        assert!(
            (ratio - 0.99).abs() < 0.01,
            "sampled ratio {ratio} too far from 0.99"
        );
    }

    /// Single-entry list: sample always returns that entry.
    #[test]
    fn sample_single_entry_returns_it() {
        let entries = vec![TopKEntry {
            token_id: 42,
            logit: 1.0,
            probability: 1.0,
        }];
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..16 {
            assert_eq!(dsv4_sample_token(&entries, &mut rng), 42);
        }
    }

    #[test]
    #[should_panic(expected = "entries must be non-empty")]
    fn sample_empty_panics() {
        let mut rng = StdRng::seed_from_u64(0);
        let _ = dsv4_sample_token(&[], &mut rng);
    }

    // ── Composed sample_next_token tests ──

    /// Greedy config: ignores all sampling fields, picks argmax.
    #[test]
    fn sample_next_token_greedy_picks_argmax() {
        // Token 3 has highest logit.
        let row = Array1::<f32>::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0]);
        let mut rng = StdRng::seed_from_u64(0);
        let picked = dsv4_sample_next_token(row.view(), SamplingConfig::greedy(), &mut rng);
        assert_eq!(picked, 3);
    }

    /// Greedy never consumes the RNG: rng state unchanged across calls.
    #[test]
    fn greedy_does_not_consume_rng() {
        let row = Array1::<f32>::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0]);
        let mut rng = StdRng::seed_from_u64(42);
        let pre_state: u64 = rng.gen();
        let mut rng_after = StdRng::seed_from_u64(42);
        // Burn the same value to align states, then sample greedily.
        let _ = rng_after.gen::<u64>();
        let _ = dsv4_sample_next_token(row.view(), SamplingConfig::greedy(), &mut rng_after);
        // Both RNGs should now produce the same next value — greedy
        // didn't consume any of rng_after beyond the initial burn.
        let mut rng_no_sample = StdRng::seed_from_u64(42);
        let _ = rng_no_sample.gen::<u64>();
        assert_eq!(rng_after.gen::<u64>(), rng_no_sample.gen::<u64>());
        // pre_state captured to keep the assertion symmetric with the
        // alternate-seed path — just touch it so it's not dead.
        let _ = pre_state;
    }

    /// Sampled config with top-k=1: equivalent to argmax even though
    /// the greedy bypass isn't used — the nucleus has exactly one
    /// entry, so sampling always picks it.
    #[test]
    fn sample_next_token_topk1_equals_argmax() {
        let row = Array1::<f32>::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0]);
        let config = SamplingConfig {
            greedy: false,
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..8 {
            assert_eq!(
                dsv4_sample_next_token(row.view(), config, &mut rng),
                3,
                "top_k=1 should always pick argmax"
            );
        }
    }

    /// Highly-skewed distribution + default-sampling preset: the
    /// dominant token should still come up the vast majority of the time.
    #[test]
    fn sample_next_token_default_preset_respects_distribution() {
        // Token 1 has logit far above everything else.
        let mut row_vec = vec![0.0_f32; 50];
        row_vec[1] = 50.0;
        let row = Array1::<f32>::from_vec(row_vec);
        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let mut hits = 0;
        for _ in 0..1000 {
            if dsv4_sample_next_token(row.view(), SamplingConfig::default_sampling(), &mut rng) == 1
            {
                hits += 1;
            }
        }
        // With logit gap of 50, token 1 should be selected ≥ 99% of trials.
        assert!(hits > 990, "default-sampling: token 1 only hit {hits}/1000");
    }

    /// Same seed + same logits + same config → same token (determinism).
    #[test]
    fn sample_next_token_with_same_seed_is_deterministic() {
        let row = Array1::<f32>::from_shape_fn(20, |i| (i as f32 * 0.13).sin());
        let config = SamplingConfig::default_sampling();
        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);
        let a = dsv4_sample_next_token(row.view(), config, &mut rng_a);
        let b = dsv4_sample_next_token(row.view(), config, &mut rng_b);
        assert_eq!(a, b);
    }

    /// `SamplingConfig::greedy()` constructor sets `greedy = true`.
    #[test]
    fn config_greedy_constructor() {
        let g = SamplingConfig::greedy();
        assert!(g.greedy);
        assert_eq!(g.top_k, 1);
        assert_eq!(g.top_p, 1.0);
    }

    /// `SamplingConfig::default_sampling()` matches the documented preset.
    #[test]
    fn config_default_sampling_preset() {
        let s = SamplingConfig::default_sampling();
        assert!(!s.greedy);
        assert_eq!(s.temperature, 0.7);
        assert_eq!(s.top_k, 40);
        assert_eq!(s.top_p, 0.95);
    }
}
