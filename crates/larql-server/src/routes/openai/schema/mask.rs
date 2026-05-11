//! Token-level mask adapter — wraps the schema-typed [`Fsm`] into the
//! `FnMut(&[u32], &mut Vec<f32>)` shape that
//! `larql_inference::generate_constrained` expects.
//!
//! ## Strategy
//!
//! At each generation step:
//!
//! 1. Replay the prompt + previously-generated tokens through a private
//!    FSM. (Cached across steps using a "starts_with previous" check —
//!    in the steady state, only the newest token is replayed.)
//! 2. For every candidate token id in the vocab, snapshot the FSM,
//!    simulate stepping its surface chars; if the simulation rejects,
//!    set the candidate's logit to `-inf`.
//! 3. Allow EOS token ids only when [`Fsm::is_complete`].
//!
//! ## Cost
//!
//! `O(vocab × avg_token_len)` per generation step. For Gemma 3 4B
//! (~256K vocab), this adds ~5–15 ms per step on a modest CPU. The FSM
//! `clone()` is cheap (the stack is typically <8 frames deep).
//!
//! Future optimisations:
//! - Per-state mask cache keyed by FSM "profile" (frame stack hash).
//! - Trie-of-allowed-prefixes representation to skip mass-rejected
//!   tokens whose first char is already invalid.

use std::collections::HashSet;
use std::sync::Arc;

use super::fsm::{Fsm, StepResult};

/// Build the `mask_fn` adapter expected by
/// [`larql_inference::layer_graph::generate_constrained`].
///
/// `prompt_text` is the JSON the FSM should consider already produced
/// before any tokens were generated — for `response_format: json_object`
/// the server prefills `{` so the model is biased into JSON, and the
/// mask FSM starts with that `{` already consumed.
///
/// `eos_token_ids` are the model's natural EOS markers; they're masked
/// out while the FSM is incomplete.
pub fn build_mask(
    tokenizer: Arc<larql_inference::tokenizers::Tokenizer>,
    fsm_template: Fsm,
    prompt_text: String,
    eos_token_ids: HashSet<u32>,
) -> impl FnMut(&[u32], &mut Vec<f32>) {
    // Surface form for every vocab id — built lazily on first call.
    let mut surfaces: Option<Vec<Option<String>>> = None;
    // Last replay-state cache so steady-state tokens don't replay the
    // entire history.
    let mut last_replay: Option<(Vec<u32>, Fsm)> = None;

    move |generated: &[u32], logits: &mut Vec<f32>| {
        let surface_table: &Vec<Option<String>> = surfaces.get_or_insert_with(|| {
            let n = logits.len();
            (0..n)
                .map(|i| larql_inference::decode_token(&tokenizer, i as u32))
                .collect()
        });

        // Replay prompt + generated through a FSM. Reuse cached state
        // when the new `generated` extends the previous one.
        let fsm: Fsm = match last_replay.as_ref() {
            Some((prev, fsm)) if generated.starts_with(prev) => {
                let mut fsm = fsm.clone();
                let mut ok = true;
                for &id in &generated[prev.len()..] {
                    if let Some(Some(s)) = surface_table.get(id as usize) {
                        if fsm.step_str(s) == StepResult::Reject {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    fsm
                } else {
                    fresh_fsm(&fsm_template, &prompt_text, surface_table, generated)
                }
            }
            _ => fresh_fsm(&fsm_template, &prompt_text, surface_table, generated),
        };
        last_replay = Some((generated.to_vec(), fsm.clone()));

        // Iterate the vocab and mask out candidates the FSM rejects.
        for (id, score) in logits.iter_mut().enumerate() {
            if eos_token_ids.contains(&(id as u32)) {
                if !fsm.is_complete() {
                    *score = f32::NEG_INFINITY;
                }
                continue;
            }
            let surface = match surface_table.get(id) {
                Some(Some(s)) => s,
                _ => {
                    *score = f32::NEG_INFINITY;
                    continue;
                }
            };
            let mut probe = fsm.clone();
            if probe.step_str(surface) == StepResult::Reject {
                *score = f32::NEG_INFINITY;
            }
        }
    }
}

fn fresh_fsm(template: &Fsm, prompt: &str, surfaces: &[Option<String>], generated: &[u32]) -> Fsm {
    let mut fsm = template.clone();
    let _ = fsm.step_str(prompt);
    for &id in generated {
        if let Some(Some(s)) = surfaces.get(id as usize) {
            if fsm.step_str(s) == StepResult::Reject {
                break;
            }
        }
    }
    fsm
}
