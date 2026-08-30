//! Generation above the [`LogitsSession`] seam.
//!
//! The existing sampler and EOS machinery
//! ([`Sampler`](crate::layer_graph::generate::sampling::Sampler),
//! [`EosConfig`](crate::layer_graph::generate::eos::EosConfig)) drive
//! any [`LogitsSession`] — they never learn what a layer is. Token ids
//! go in and come out as ids: a tokenizer is part of the fixture on
//! the V3 path (only one side of a parity comparison may choose it),
//! so detokenisation composes *outside* this driver.

use crate::error::InferenceError;
use crate::layer_graph::generate::eos::EosConfig;
use crate::layer_graph::generate::sampling::{Sampler, SamplingConfig};

use super::session::LogitsSession;

/// A logits mask: receives the generated-so-far ids and the mutable
/// logits, carrying any grammar state in its closure — the V2
/// constrained driver's contract.
pub type LogitsMask<'a> = &'a mut dyn FnMut(&[u32], &mut Vec<f32>);

/// Built outside the generic driver so the refusal exists (and is
/// counted) once, not once per session instantiation.
fn sampler_exhausted_error() -> InferenceError {
    InferenceError::Parse("sampler produced no token — logits were empty or non-finite".to_string())
}

/// What one generation run produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionGeneration {
    /// Prompt length consumed by prefill, in tokens.
    pub prompt_len: usize,
    /// Generated token ids, in emission order. A stop token is never
    /// included — generation ends *before* emitting it.
    pub tokens: Vec<u32>,
}

/// Generate up to `max_new_tokens` ids from `session`, sampling with
/// `sampling` and stopping on `eos`. `on_token` fires once per emitted
/// id, in order — the streaming surface at this seam.
///
/// EOS is judged on token ids only (`EosConfig::is_eos` with no
/// decoded text): stop *strings* need a tokenizer, which lives above
/// this driver.
pub fn generate_session<S: LogitsSession + ?Sized>(
    session: &mut S,
    prompt: &[u32],
    max_new_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    on_token: impl FnMut(u32),
) -> Result<SessionGeneration, InferenceError> {
    let logits = session.prefill(prompt)?;
    continue_session(session, logits, max_new_tokens, sampling, eos, on_token)
}

/// Continue generation from logits already in hand — the resume-aware
/// driver (VI3-SERVE-1). The caller has typically batch-prefilled the
/// session's continuation state
/// ([`Vindex3Runtime::prefill_into`](super::Vindex3Runtime::prefill_into))
/// and holds the prefill's last-position logits; this drives sampling
/// and stepping from there. [`generate_session`] is prefill followed
/// by exactly this function, so the two paths cannot drift.
///
/// The result's `prompt_len` is the session's position at entry — the
/// positions the continuation stands on, however they were consumed.
pub fn continue_session<S: LogitsSession + ?Sized>(
    session: &mut S,
    logits: Vec<f32>,
    max_new_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    on_token: impl FnMut(u32),
) -> Result<SessionGeneration, InferenceError> {
    drive_session(
        session,
        logits,
        max_new_tokens,
        sampling,
        eos,
        None,
        on_token,
    )
}

/// [`continue_session`] with a logits mask applied before every sample
/// (N0.6 — constrained decoding on the V3 runtime). `mask_fn` receives
/// the generated-so-far ids and the mutable logits, exactly the V2
/// constrained driver's contract, and carries any grammar state in its
/// closure. Both public drivers are one loop ([`drive_session`]), so
/// the constrained and free paths cannot drift.
///
/// Mask-exhaustion semantics mirror the V2 constrained driver: every
/// candidate masked out before the FIRST emission is an error (the
/// grammar admits nothing — a broken constraint); exhaustion after at
/// least one emission is a natural stop (the grammar completed).
pub fn continue_session_masked<S: LogitsSession + ?Sized>(
    session: &mut S,
    logits: Vec<f32>,
    max_new_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    mask_fn: LogitsMask<'_>,
    on_token: impl FnMut(u32),
) -> Result<SessionGeneration, InferenceError> {
    drive_session(
        session,
        logits,
        max_new_tokens,
        sampling,
        eos,
        Some(mask_fn),
        on_token,
    )
}

/// The one generation loop behind both public drivers. `mask` decides
/// the sampler-exhaustion policy: with no mask, exhaustion is always
/// an error (finite logits should always yield a token — anything else
/// is broken state); with a mask, exhaustion after the first emission
/// is the grammar completing.
fn drive_session<S: LogitsSession + ?Sized>(
    session: &mut S,
    mut logits: Vec<f32>,
    max_new_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    mut mask: Option<LogitsMask<'_>>,
    mut on_token: impl FnMut(u32),
) -> Result<SessionGeneration, InferenceError> {
    let prompt_len = session.position();
    let mut sampler = Sampler::new(sampling);
    let mut tokens = Vec::new();
    while tokens.len() < max_new_tokens {
        if let Some(mask_fn) = mask.as_deref_mut() {
            mask_fn(&tokens, &mut logits);
        }
        let Some(id) = sampler.sample_with_history(&logits, &tokens) else {
            if mask.is_some() && !tokens.is_empty() {
                // The mask admitted nothing after emission began: the
                // grammar completed — a natural stop, like EOS.
                break;
            }
            return Err(sampler_exhausted_error());
        };
        if eos.is_eos(id, "") {
            break;
        }
        tokens.push(id);
        on_token(id);
        if tokens.len() == max_new_tokens {
            break;
        }
        logits = session.step(id)?;
    }
    Ok(SessionGeneration { prompt_len, tokens })
}
