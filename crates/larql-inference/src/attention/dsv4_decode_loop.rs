//! Multi-token decode loop orchestration.
//!
//! Repeatedly calls a user-provided forward function on the running
//! token sequence, samples the next token via
//! [`dsv4_sample_next_token`], appends it, and continues until either
//! `max_new_tokens` is reached or the EOS token is produced.
//!
//! The forward function is a closure rather than a concrete call into
//! [`super::dsv4_streaming_model_forward`] for two reasons:
//!
//! 1. **KV cache decoupling**: the eventual KV-cached forward is a
//!    different signature (it advances cache state, takes only the
//!    new tokens, etc.). Keeping the loop generic over `FnMut(&[u32])`
//!    means the same orchestration code works against today's
//!    re-prefill-everything forward AND tomorrow's cached forward.
//!
//! 2. **Testability**: a closure-based loop is trivially testable with
//!    a synthetic forward (no GGUF required). Real-GGUF runs hook in
//!    by wrapping `dsv4_streaming_model_forward` in the closure.
//!
//! Today's prefill-every-step approach is `O(n²)` in the number of
//! generated tokens (each step re-runs the full sequence through 43
//! layers). It's correctness-complete but slow; the natural follow-up
//! is layer KV cache + per-step decode that only processes the new
//! token. This module's API is stable across that future swap.

use ndarray::Array2;
use rand::Rng;

use super::dsv4_sampling::{dsv4_sample_next_token, SamplingConfig};

/// Top-level decode-loop config. `max_new_tokens` is hard-required;
/// `eos_token` is optional (None → only the max-tokens stop is used).
#[derive(Clone, Copy, Debug)]
pub struct DecodeConfig {
    /// Maximum tokens to *generate*; the returned vector has the
    /// initial prompt prefix plus up to `max_new_tokens` appended.
    pub max_new_tokens: usize,
    /// Optional end-of-sequence token. If produced, the loop stops
    /// immediately and `eos_token` itself IS included in the returned
    /// token list.
    pub eos_token: Option<u32>,
    /// Sampling strategy: greedy / temperature+top-k+top-p.
    pub sampling: SamplingConfig,
}

/// Run a multi-token decode loop.
///
/// - `initial_tokens`: the prompt (must be non-empty; first sampling
///   step needs at least one position of logits).
/// - `decode_config`: stop criteria + sampling.
/// - `rng`: source of randomness for non-greedy sampling.
/// - `forward_fn`: closure mapping the *current* full token sequence
///   to `(n_tokens, n_vocab)` logits. Errors short-circuit the loop
///   and propagate to the caller.
///
/// Returns the complete token sequence (prompt + generated). On EOS
/// or `forward_fn` error the loop ends early; the returned sequence
/// is whatever was accumulated so far.
pub fn dsv4_decode_loop<F, E, R>(
    initial_tokens: Vec<u32>,
    decode_config: DecodeConfig,
    rng: &mut R,
    mut forward_fn: F,
) -> Result<Vec<u32>, E>
where
    F: FnMut(&[u32]) -> Result<Array2<f32>, E>,
    R: Rng + ?Sized,
{
    assert!(
        !initial_tokens.is_empty(),
        "initial_tokens must be non-empty"
    );
    let mut tokens = initial_tokens;
    for _ in 0..decode_config.max_new_tokens {
        let logits = forward_fn(&tokens)?;
        // Take the last position's row — that's the next-token prediction.
        let last_row_idx = logits.shape()[0] - 1;
        let last_row = logits.row(last_row_idx);
        let next = dsv4_sample_next_token(last_row, decode_config.sampling, rng);
        tokens.push(next);
        if Some(next) == decode_config.eos_token {
            break;
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::convert::Infallible;

    /// Build a synthetic forward closure that returns a fixed-output
    /// logits row for every position — always picks `predicted` as
    /// the argmax. Useful for testing the loop's stop conditions
    /// without an actual model.
    fn fake_forward_predicting(
        predicted: u32,
        n_vocab: usize,
    ) -> impl FnMut(&[u32]) -> Result<Array2<f32>, Infallible> {
        move |tokens: &[u32]| {
            let n_tokens = tokens.len();
            let mut out = Array2::<f32>::zeros((n_tokens, n_vocab));
            for t in 0..n_tokens {
                out[[t, predicted as usize]] = 100.0; // dominate softmax
            }
            Ok(out)
        }
    }

    /// Greedy loop with synthetic forward that always predicts token 7
    /// → returned sequence ends with `max_new_tokens` copies of 7.
    #[test]
    fn greedy_loop_produces_max_new_tokens() {
        let initial = vec![1u32, 2, 3];
        let cfg = DecodeConfig {
            max_new_tokens: 5,
            eos_token: None,
            sampling: SamplingConfig::greedy(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let out = dsv4_decode_loop(
            initial.clone(),
            cfg,
            &mut rng,
            fake_forward_predicting(7, 32),
        )
        .unwrap();
        assert_eq!(out.len(), initial.len() + 5);
        assert_eq!(&out[..3], &initial[..]);
        for &t in &out[3..] {
            assert_eq!(t, 7);
        }
    }

    /// EOS stops early. Synthetic forward predicts EOS=11 → loop ends
    /// after the first generated token.
    #[test]
    fn eos_stops_loop_early() {
        let initial = vec![1u32, 2];
        let cfg = DecodeConfig {
            max_new_tokens: 100,
            eos_token: Some(11),
            sampling: SamplingConfig::greedy(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let out = dsv4_decode_loop(
            initial.clone(),
            cfg,
            &mut rng,
            fake_forward_predicting(11, 32),
        )
        .unwrap();
        // Prompt (2) + exactly 1 generated EOS.
        assert_eq!(out.len(), 3);
        assert_eq!(out[2], 11);
    }

    /// Different EOS than what's predicted → loop runs to max_new_tokens.
    #[test]
    fn non_eos_predicted_runs_to_max() {
        let cfg = DecodeConfig {
            max_new_tokens: 4,
            eos_token: Some(99), // not what forward predicts
            sampling: SamplingConfig::greedy(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let out = dsv4_decode_loop(vec![0], cfg, &mut rng, fake_forward_predicting(5, 16)).unwrap();
        assert_eq!(out.len(), 5); // 1 prompt + 4 generated
        for &t in &out[1..] {
            assert_eq!(t, 5);
        }
    }

    /// `max_new_tokens = 0`: loop generates nothing, returns prompt.
    #[test]
    fn max_new_tokens_zero_returns_prompt() {
        let initial = vec![1u32, 2, 3];
        let cfg = DecodeConfig {
            max_new_tokens: 0,
            eos_token: None,
            sampling: SamplingConfig::greedy(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let out = dsv4_decode_loop(
            initial.clone(),
            cfg,
            &mut rng,
            fake_forward_predicting(7, 32),
        )
        .unwrap();
        assert_eq!(out, initial);
    }

    /// Forward error propagates out, partial sequence isn't returned.
    /// Variant: forward fails on the 3rd call.
    #[test]
    fn forward_error_propagates() {
        #[derive(Debug, PartialEq)]
        struct FakeError;

        let mut call_count = 0;
        let forward = |tokens: &[u32]| -> Result<Array2<f32>, FakeError> {
            call_count += 1;
            if call_count >= 3 {
                return Err(FakeError);
            }
            let mut out = Array2::<f32>::zeros((tokens.len(), 16));
            for t in 0..tokens.len() {
                out[[t, 5]] = 100.0;
            }
            Ok(out)
        };
        let cfg = DecodeConfig {
            max_new_tokens: 10,
            eos_token: None,
            sampling: SamplingConfig::greedy(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let err = dsv4_decode_loop(vec![1], cfg, &mut rng, forward).unwrap_err();
        assert_eq!(err, FakeError);
    }

    /// Forward fn receives the *growing* token sequence each call:
    /// call 1 sees N tokens, call 2 sees N+1, etc.
    #[test]
    fn forward_sees_growing_sequence() {
        let mut seen_lengths: Vec<usize> = Vec::new();
        // Use a Vec inside RefCell to avoid moving sealed.
        let forward = |tokens: &[u32]| -> Result<Array2<f32>, Infallible> {
            seen_lengths.push(tokens.len());
            let mut out = Array2::<f32>::zeros((tokens.len(), 8));
            for t in 0..tokens.len() {
                out[[t, 3]] = 100.0;
            }
            Ok(out)
        };
        let cfg = DecodeConfig {
            max_new_tokens: 4,
            eos_token: None,
            sampling: SamplingConfig::greedy(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let _ = dsv4_decode_loop(vec![1, 2], cfg, &mut rng, forward).unwrap();
        // Initial length 2; grows by 1 per iteration.
        assert_eq!(seen_lengths, vec![2, 3, 4, 5]);
    }

    /// Same seed → same generated sequence (determinism).
    #[test]
    fn same_seed_produces_same_sequence() {
        // Build a forward that gives an interesting distribution:
        // every position predicts a uniform-ish 32-vocab distribution.
        let forward_fn = |tokens: &[u32]| -> Result<Array2<f32>, Infallible> {
            let mut out = Array2::<f32>::zeros((tokens.len(), 32));
            for t in 0..tokens.len() {
                for v in 0..32 {
                    out[[t, v]] = ((v + t * 7) as f32 * 0.13).sin();
                }
            }
            Ok(out)
        };
        let cfg = DecodeConfig {
            max_new_tokens: 6,
            eos_token: None,
            sampling: SamplingConfig::default_sampling(),
        };
        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);
        let a = dsv4_decode_loop(vec![0], cfg, &mut rng_a, forward_fn).unwrap();
        let b = dsv4_decode_loop(vec![0], cfg, &mut rng_b, forward_fn).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "initial_tokens must be non-empty")]
    fn empty_initial_tokens_panics() {
        let cfg = DecodeConfig {
            max_new_tokens: 5,
            eos_token: None,
            sampling: SamplingConfig::greedy(),
        };
        let mut rng = StdRng::seed_from_u64(0);
        let _ = dsv4_decode_loop::<_, Infallible, _>(
            vec![],
            cfg,
            &mut rng,
            fake_forward_predicting(0, 16),
        );
    }
}
