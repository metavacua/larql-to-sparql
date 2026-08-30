//! Phase 1b of the V3 compose install: residual capture on the **clean
//! base** program.
//!
//! V2's contract, ported literally (`insert/capture.rs`): the canonical
//! prompt's residual is captured with NO overlay observed — prior
//! installs' slots must not fire during capture, or they contaminate
//! the new fact's residual with earlier targets in ways the refine
//! pass cannot cleanly undo. (The KNN arm is the opposite by the same
//! V2 contract: its capture runs over the patched forward.)
//!
//! Decoys are the same two sets V2 builds — the fixed canonical prompt
//! list plus up to ten template-matched decoys scanned from the
//! tokenizer vocabulary — captured at the install layer on the base and
//! cached per layer on the session, so subsequent INSERTs at the same
//! layer reuse them for free.

use crate::error::LqlError;
use crate::executor::vindex3::{capture_layer_residual, V3Runtime};
use crate::executor::Session;
use larql_vindex::tokenizers::Tokenizer;

use super::super::capture::CANONICAL_DECOY_PROMPTS;

/// How many template-matched decoys the vocab scan contributes and how
/// far it scans — V2's literals (`capture.rs`).
const TEMPLATE_DECOY_COUNT: usize = 10;
const TEMPLATE_DECOY_VOCAB_SCAN: usize = 5000;

/// Decoys for one install layer, when the session cache does not hold
/// that layer yet. The caller commits them to
/// `session.decoy_residual_cache` once its immutable borrow ends.
pub(super) fn capture_layer_decoys(
    session: &Session,
    runtime: &V3Runtime,
    tokenizer: &Tokenizer,
    vocab_size: usize,
    entity: &str,
    relation: &str,
    layer: usize,
) -> Result<Option<Vec<larql_vindex::ndarray::Array1<f32>>>, LqlError> {
    if session.decoy_residual_cache.contains_key(&layer) {
        return Ok(None);
    }

    let mut decoy_prompts: Vec<String> = CANONICAL_DECOY_PROMPTS
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Template-matched decoys: same relation template, entities sampled
    // from single vocab tokens that decode to alphabetic 3+-char words
    // different from the entity being inserted.
    let mut template_decoys_added = 0;
    for tid in 0..vocab_size.min(TEMPLATE_DECOY_VOCAB_SCAN) as u32 {
        if template_decoys_added >= TEMPLATE_DECOY_COUNT {
            break;
        }
        let decoded = tokenizer.decode(&[tid], true).unwrap_or_default();
        let word = decoded.trim();
        if word.len() >= 3
            && word.chars().all(|c| c.is_alphabetic())
            && !word.eq_ignore_ascii_case(entity)
        {
            decoy_prompts.push(crate::executor::tuning::canonical_prompt(relation, word));
            template_decoys_added += 1;
        }
    }

    let mut captured = Vec::with_capacity(decoy_prompts.len());
    for decoy_prompt in &decoy_prompts {
        // Base program, same tap as the canonical capture — decoy
        // residuals must match the baseline the gates will be judged
        // against.
        let residual = capture_layer_residual(
            runtime,
            tokenizer,
            decoy_prompt.as_str(),
            layer,
            None,
            session.v3_bos_token(),
        )?;
        captured.push(larql_vindex::ndarray::Array1::from_vec(residual));
    }
    Ok(Some(captured))
}
