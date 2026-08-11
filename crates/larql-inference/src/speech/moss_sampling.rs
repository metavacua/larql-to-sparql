//! MOSS-TTS-Realtime's sampler, replicated literally.
//!
//! The reference's operating mode is sampled — greedy decoding does not
//! terminate on novel text (`docs/tts-funnel.md` step-5 gate log) — and
//! two of its choices diverge from standard implementations, so this
//! module replicates them rather than reusing the text sampler:
//!
//! - **top-p sorts ascending** and removes where the ascending cumulative
//!   softmax is `<= 1 - p`, always keeping the final (largest) element.
//!   HF's descending warper tie-breaks differently.
//! - **repetition penalty is per-codebook across frames** (each codebook
//!   penalised by its own last-`window` history), gather/scatter
//!   semantics: a duplicated id in the window is penalised once, not
//!   compounded. It is never applied to the prefill frame.
//!
//! Ground truth for the unit tests was produced by running the reference
//! implementation's `apply_top_k` / `apply_top_p` /
//! `apply_repetition_penalty` on the same inputs (fixture values inline
//! below). Sampled *tokens* are not comparable across engines (RNG
//! streams differ); the transform is what parity means here.

use rand::rngs::StdRng;
use rand::Rng;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Sampling parameters. Defaults are the reference's shipped values.
#[derive(Debug, Clone, Copy)]
pub struct MossSampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

impl Default for MossSampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 30,
            top_p: 0.6,
            repetition_penalty: 1.1,
            repetition_window: 50,
        }
    }
}

/// How the depth heads pick a codebook id.
#[derive(Debug, Clone, Copy)]
pub enum DecodeMode {
    /// Argmax — the parity mode. Bypasses every filter, like the
    /// reference's `do_sample=False` shortcut.
    Greedy,
    /// The reference's sampled mode.
    Sampled(MossSampling),
}

/// Apply temperature, top-k, top-p in the reference's order, mutating
/// `logits` in place (`-inf` marks removed ids).
pub fn filter_logits(logits: &mut [f32], sampling: &MossSampling) {
    if sampling.temperature != 1.0 {
        for value in logits.iter_mut() {
            *value /= sampling.temperature;
        }
    }
    apply_top_k(logits, sampling.top_k);
    apply_top_p_ascending(logits, sampling.top_p);
}

/// Keep the `top_k` largest logits; ids strictly below the k-th value are
/// removed (ties at the threshold survive, as in the reference).
fn apply_top_k(logits: &mut [f32], top_k: usize) {
    let top_k = top_k.max(1).min(logits.len());
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted[top_k - 1];
    for value in logits.iter_mut() {
        if *value < threshold {
            *value = f32::NEG_INFINITY;
        }
    }
}

/// The reference's ascending-sort nucleus filter: remove ids whose
/// ascending cumulative softmax mass is `<= 1 - p`; the largest id
/// always survives.
fn apply_top_p_ascending(logits: &mut [f32], top_p: f32) {
    let probs = softmax(logits);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap());
    let mut cumulative = 0.0f32;
    for (rank, &id) in order.iter().enumerate() {
        cumulative += probs[id];
        let is_largest = rank + 1 == order.len();
        if cumulative <= 1.0 - top_p && !is_largest {
            logits[id] = f32::NEG_INFINITY;
        }
    }
}

/// Penalise each id in the last `window` entries of `history` once:
/// negative logits are multiplied by `penalty`, others divided.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32, window: usize) {
    if penalty == 1.0 || history.is_empty() {
        return;
    }
    let start = history.len().saturating_sub(window);
    let mut seen = vec![false; logits.len()];
    for &id in &history[start..] {
        let id = id as usize;
        if id >= logits.len() || seen[id] {
            continue;
        }
        seen[id] = true;
        let value = logits[id];
        logits[id] = if value < 0.0 {
            value * penalty
        } else {
            value / penalty
        };
    }
}

/// Draw from the (filtered) logits' softmax.
pub fn sample_multinomial(logits: &[f32], rng: &mut StdRng) -> usize {
    let probs = softmax(logits);
    let draw: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    let mut last_viable = 0usize;
    for (id, &p) in probs.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        last_viable = id;
        cumulative += p;
        if draw < cumulative {
            return id;
        }
    }
    // Rounding left us past the total mass: the last viable id.
    last_viable
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return vec![0.0; logits.len()];
    }
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const NEG: f32 = f32::NEG_INFINITY;

    /// Fixture values produced by the reference implementation
    /// (`apply_top_k(·, 4)` then `apply_top_p(·, 0.6)` after
    /// temperature 0.8), 2026-08-09.
    #[test]
    fn matches_reference_small_case() {
        let mut logits = vec![0.1, 2.0, -1.0, 0.5, 1.5, -0.3, 0.9, 0.0];
        let sampling = MossSampling {
            temperature: 0.8,
            top_k: 4,
            top_p: 0.6,
            ..MossSampling::default()
        };
        filter_logits(&mut logits, &sampling);
        let expected = [NEG, 2.5, NEG, NEG, 1.875, NEG, NEG, NEG];
        assert_close(&logits, &expected);
    }

    #[test]
    fn matches_reference_peaked_case() {
        let mut logits = vec![10.0, 0.0, -5.0, 3.0, 2.9, 2.8, -2.0, 1.0];
        let sampling = MossSampling {
            temperature: 0.8,
            top_k: 4,
            top_p: 0.6,
            ..MossSampling::default()
        };
        filter_logits(&mut logits, &sampling);
        let expected = [12.5, NEG, NEG, NEG, NEG, NEG, NEG, NEG];
        assert_close(&logits, &expected);
    }

    #[test]
    fn matches_reference_flat_case() {
        let mut logits: Vec<f32> = (0..8).map(|i| i as f32 * 1e-3).collect();
        let sampling = MossSampling {
            temperature: 0.8,
            top_k: 4,
            top_p: 0.6,
            ..MossSampling::default()
        };
        filter_logits(&mut logits, &sampling);
        let expected = [NEG, NEG, NEG, NEG, NEG, 0.00625, 0.0075, 0.00875];
        assert_close(&logits, &expected);
    }

    #[test]
    fn matches_reference_penalty_case() {
        let mut logits = vec![1.0, -2.0, 3.0, 0.5, -0.1, 2.0, 0.0, -1.5];
        // Window 3 of [2, 5, 2, 1] → ids {5, 2, 1}.
        apply_repetition_penalty(&mut logits, &[2, 5, 2, 1], 1.1, 3);
        let expected = [1.0, -2.2, 2.7272727, 0.5, -0.1, 1.8181818, 0.0, -1.5];
        assert_close(&logits, &expected);
    }

    #[test]
    fn duplicate_history_ids_penalised_once() {
        let mut once = vec![2.0, 1.0];
        apply_repetition_penalty(&mut once, &[0], 1.1, 10);
        let mut twice = vec![2.0, 1.0];
        apply_repetition_penalty(&mut twice, &[0, 0, 0], 1.1, 10);
        assert_eq!(once, twice, "gather/scatter semantics: once per unique id");
    }

    #[test]
    fn largest_id_always_survives_top_p() {
        let mut logits = vec![100.0, 0.0, -1.0];
        let sampling = MossSampling {
            top_p: 0.999999,
            ..MossSampling::default()
        };
        filter_logits(&mut logits, &sampling);
        assert!(logits[0] > NEG);
    }

    #[test]
    fn multinomial_is_deterministic_under_seed() {
        let logits = vec![1.0, 2.0, 3.0, NEG];
        let a: Vec<usize> = {
            let mut rng = StdRng::seed_from_u64(7);
            (0..20)
                .map(|_| sample_multinomial(&logits, &mut rng))
                .collect()
        };
        let b: Vec<usize> = {
            let mut rng = StdRng::seed_from_u64(7);
            (0..20)
                .map(|_| sample_multinomial(&logits, &mut rng))
                .collect()
        };
        assert_eq!(a, b);
        assert!(a.iter().all(|&id| id != 3), "removed ids never drawn");
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let ok = if e == NEG {
                a == NEG
            } else {
                (a - e).abs() < 1e-5
            };
            assert!(ok, "index {index}: {a} vs expected {e}");
        }
    }
}
