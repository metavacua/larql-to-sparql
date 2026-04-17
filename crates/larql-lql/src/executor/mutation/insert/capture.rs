//! Phase 1b of `INSERT INTO EDGES` (Compose mode): forward-pass the
//! canonical prompt through the base vindex to capture per-layer
//! residuals, plus opportunistically capture decoy residuals for any
//! install layer not already in `session.decoy_residual_cache`.
//!
//! Both sets feed Phase 2's refine pass (cliff-breaker stack). Decoys
//! are captured here rather than at install time because the model
//! work is already loaded and the decoy set is layer-keyed — once
//! cached, subsequent INSERTs at the same layer reuse it for free.

use crate::error::LqlError;
use crate::executor::Session;

use super::plan::InstallPlan;

/// Output of `capture_install_residuals`. Caller commits the pending
/// decoys to `session.decoy_residual_cache` after the immutable borrow
/// of `self` ends.
pub(super) struct CapturedResiduals {
    /// Per-layer captured residual at the install layers. Empty when
    /// `plan.use_constellation` is false (browse-only vindex).
    pub per_layer: Vec<(usize, Vec<f32>)>,
    /// Decoys captured this call for install layers that weren't
    /// already in the session cache. Caller merges into
    /// `session.decoy_residual_cache` once the immutable borrow ends.
    pub pending_decoys: Vec<(usize, Vec<larql_vindex::ndarray::Array1<f32>>)>,
}

impl Session {
    /// Capture the canonical-prompt residual at each install layer plus
    /// decoy residuals (canonical + template-matched) for any layer not
    /// already cached. Returns an empty `per_layer` when the vindex has
    /// no model weights — Phase 2 then falls back to the entity
    /// embedding direction for the gate.
    pub(super) fn capture_install_residuals(
        &self,
        entity: &str,
        relation: &str,
        plan: &InstallPlan,
    ) -> Result<CapturedResiduals, LqlError> {
        if !plan.use_constellation {
            return Ok(CapturedResiduals {
                per_layer: Vec::new(),
                pending_decoys: Vec::new(),
            });
        }

        let (path, config, patched) = self.require_vindex()?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        // The install captures the model's residual by forward-passing
        // a synthesised canonical question for the fact, then uses the
        // unit-normalised result as the gate direction. Template:
        //
        //     "The {relation} of {entity} is"
        //
        // For canonical relations ("capital", "author", "language",
        // "currency"), this matches what the user will later INFER on —
        // so the captured residual at L26 has near-unit cosine with the
        // inference residual, the slot fires strongly, and the install
        // lifts the answer (validated end-to-end by `refine_demo` on
        // 10 capital-of facts, matching the Python reference in
        // `experiments/14_vindex_compilation`).
        //
        // For non-canonical relations (e.g. "ocean-rank"), the template
        // produces a prompt that doesn't match inference — the install
        // remains invisible rather than hijacking, because the captured
        // residual has small cosine with any real inference residual
        // and the slot doesn't fire. This is a known limitation: the
        // LQL INSERT surface supports canonical-form relations only.
        // Non-canonical facts can be installed via the Python pipeline
        // in `experiments/14_vindex_compilation` for now.
        let rel_words = relation.replace(['-', '_'], " ");
        let prompt = format!("The {rel_words} of {entity} is");

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load weights", e))?;

        let encoding = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Capture through the BASE index (no patch overlay), with
        // UNLIMITED top_k to match what INFER does at query time.
        // Two coupled choices:
        //
        // 1. BASE index (not `patched`): prior INSERTs' slots
        //    shouldn't fire during this capture — they would
        //    contaminate the new fact's residual with earlier
        //    targets, and the refine pass can't undo that cleanly.
        //    Matches Python exp 14 Phase 2: capture all on clean
        //    model, then install.
        //
        // 2. UNLIMITED top_k: the INFER path in `query.rs` uses
        //    `new_unlimited_with_trace`, so the L26 residual at
        //    inference time is built from a full-power baseline (all
        //    16384 features fire). If we captured at top_k=8092 — a
        //    half-power baseline — the captured residual would differ
        //    from the inference residual in magnitude even when the
        //    direction matches. We'd engineer gates against half-power
        //    residuals and fire them against full-power ones,
        //    producing the "cosines look fine, activations have a
        //    25-unit gap" silent-drift class of bug noted in
        //    `experiments/15_v11_model/RESULTS.md §20.3`.
        let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(
            &weights,
            patched.base(),
        );
        let _result = larql_inference::predict_with_ffn(
            &weights, &tokenizer, &token_ids, 1, &walk_ffn,
        );

        let per_layer: Vec<(usize, Vec<f32>)> = walk_ffn
            .take_residuals()
            .into_iter()
            .filter(|(layer, _)| plan.layers.contains(layer))
            .collect();

        // Capture decoy residuals for any install layer that isn't
        // already cached on the session. Two sets:
        //
        // 1. CANONICAL decoys — generic prompts ("Once upon a time",
        //    etc.) that suppress bleed onto unrelated text.
        //
        // 2. TEMPLATE-MATCHED decoys — same relation template ("The
        //    {relation} of {X} is") with different entities sampled
        //    from high-frequency vocabulary. These suppress bleed
        //    onto prompts that share the template structure but
        //    differ in entity — the single-fact bleed that generic
        //    decoys can't reach because "The capital of France is"
        //    has near-unit cosine with "The capital of Atlantis is"
        //    at L26 while "Once upon a time" has near-zero cosine
        //    with both.
        //
        //    The entities are sampled from the tokenizer vocab
        //    (single tokens that decode to alphabetic strings of 3+
        //    chars) so this is fully generic — no domain-specific
        //    entity list.
        let mut pending_decoys: Vec<(usize, Vec<larql_vindex::ndarray::Array1<f32>>)> = Vec::new();
        for &layer in &plan.layers {
            if self.decoy_residual_cache.contains_key(&layer) {
                continue;
            }
            // Build the full decoy prompt list: canonical + template-matched.
            let mut decoy_prompts: Vec<String> = CANONICAL_DECOY_PROMPTS
                .iter()
                .map(|s| s.to_string())
                .collect();

            // Generate template-matched decoys by substituting the
            // entity with diverse vocab tokens.
            let template_decoy_count = 10;
            let mut template_decoys_added = 0;
            for tid in 0..config.vocab_size.min(5000) as u32 {
                if template_decoys_added >= template_decoy_count {
                    break;
                }
                let decoded = tokenizer.decode(&[tid], true).unwrap_or_default();
                let word = decoded.trim();
                // Pick single-token words that are alphabetic, 3+ chars,
                // and different from the entity being inserted.
                if word.len() >= 3
                    && word.chars().all(|c| c.is_alphabetic())
                    && !word.eq_ignore_ascii_case(entity)
                {
                    let decoy = format!("The {rel_words} of {word} is");
                    decoy_prompts.push(decoy);
                    template_decoys_added += 1;
                }
            }

            let mut captured = Vec::with_capacity(decoy_prompts.len());
            for decoy_prompt in &decoy_prompts {
                let enc = tokenizer
                    .encode(decoy_prompt.as_str(), true)
                    .map_err(|e| LqlError::exec("tokenize decoy", e))?;
                let ids: Vec<u32> = enc.get_ids().to_vec();
                // Also unlimited top_k here so decoy residuals match
                // the full-power baseline INFER will produce.
                let ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(
                    &weights,
                    patched.base(),
                );
                let _ = larql_inference::predict_with_ffn(
                    &weights, &tokenizer, &ids, 1, &ffn,
                );
                let r = ffn.take_residuals().into_iter().find(|(l, _)| *l == layer);
                if let Some((_, vec)) = r {
                    captured.push(larql_vindex::ndarray::Array1::from_vec(vec));
                }
            }
            pending_decoys.push((layer, captured));
        }

        Ok(CapturedResiduals {
            per_layer,
            pending_decoys,
        })
    }
}

/// Canonical decoy prompt set used by Phase 1b alongside the
/// template-matched decoys generated from the tokenizer vocab.
///
/// Same set as `experiments/14_vindex_compilation/experiment_vindex_compilation.py`.
/// These prompts span literary, philosophical, poetic, and common
/// completion templates — the canonical bleed targets for a
/// fact-install slot operating at `gate_scale=30`. Capturing residuals
/// at the install layer through the clean base index and
/// orthogonalising the installed gate against those residuals
/// prevents the slot from firing on unrelated prompts.
///
/// Hardcoded so every session gets the same defense without user
/// configuration. A future refinement could move this to
/// `EXTRACT ... WITH DECOYS` or `INSERT ... WITH DECOYS`, but v0
/// ships this fixed list that covers the validated reference cases.
pub(super) const CANONICAL_DECOY_PROMPTS: &[&str] = &[
    "Once upon a time",
    "The quick brown fox",
    "To be or not to be",
    "Water is a",
    "A long time ago",
    "In the beginning",
    "The weather today is",
    "She opened the door and",
    "He looked at the sky",
    "The children played in the",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_decoys_have_unique_3word_prefixes() {
        let prefixes: std::collections::HashSet<String> = CANONICAL_DECOY_PROMPTS
            .iter()
            .map(|p| p.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
            .collect();
        assert_eq!(
            prefixes.len(),
            CANONICAL_DECOY_PROMPTS.len(),
            "decoy prompts should have unique 3-word prefixes"
        );
    }
}
