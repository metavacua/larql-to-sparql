use serde_json::json;

use super::super::address::attention_argmax;
use super::super::program::{fields, strata, Predicate};
use super::context::{FitContext, PromptCapture};
use super::localize::LocalizeResult;

/// A candidate guard predicate with coverage statistics.
pub struct GuardCandidate {
    pub predicate: Predicate,
    pub complexity: usize,
    /// Fraction of fragile positions the predicate covers (should be ~1.0).
    pub fragile_coverage: f64,
    /// Fraction of safe positions the predicate fires on (should be ~0.0).
    pub safe_false_positive: f64,
    pub label: String,
}

// ────────────────────────────────────────────────────────────────────────────
// Per-position feature extraction
// ────────────────────────────────────────────────────────────────────────────

struct PosFeatures {
    stratum: String,
    position: usize,
    attends_bos: bool,
    attends_prev: bool,
}

fn extract_features(capture: &PromptCapture, pos: usize, head_idx: usize) -> PosFeatures {
    let attn_row = capture
        .attention_rows
        .get(pos)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let attn_argmax = attention_argmax(attn_row, pos);
    let _ = head_idx; // attention_rows are already for the target head
    PosFeatures {
        stratum: capture.stratum.clone(),
        position: pos,
        attends_bos: attn_argmax == 0,
        attends_prev: pos > 0 && attn_argmax + 1 == pos,
    }
}

fn predicate_fires(pred: &Predicate, feat: &PosFeatures) -> bool {
    use super::super::program::PositionContext;
    let ctx = PositionContext {
        stratum: feat.stratum.clone(),
        position: feat.position,
        token_id: 0,
        prev_token_id: None,
        attends_bos: feat.attends_bos,
        attends_prev: feat.attends_prev,
        original_code: 0,
        current_code: 0,
    };
    pred.eval(&ctx)
}

// ────────────────────────────────────────────────────────────────────────────
// Atomic predicate builders
// ────────────────────────────────────────────────────────────────────────────

fn eq_stratum(s: &str) -> Predicate {
    Predicate::Eq(vec![json!(fields::STRATUM), json!(s)])
}

fn eq_bool(field: &str, val: bool) -> Predicate {
    Predicate::Eq(vec![json!(field), json!(val)])
}

fn attends_bos_pred() -> Predicate {
    eq_bool(fields::ATTENDS_BOS, true)
}
fn attends_prev_pred() -> Predicate {
    eq_bool(fields::ATTENDS_PREV, true)
}
fn natural_prose_pred() -> Predicate {
    eq_stratum(strata::NATURAL_PROSE)
}

fn attends_bos_or_prev() -> Predicate {
    Predicate::Or(vec![attends_bos_pred(), attends_prev_pred()])
}

// ────────────────────────────────────────────────────────────────────────────
// Candidate generation and evaluation
// ────────────────────────────────────────────────────────────────────────────

fn evaluate_candidate(
    pred: &Predicate,
    fragile_features: &[PosFeatures],
    safe_features: &[PosFeatures],
) -> (f64, f64) {
    // coverage: fraction of fragile positions where predicate fires (we WANT these preserved)
    let coverage = if fragile_features.is_empty() {
        0.0
    } else {
        fragile_features
            .iter()
            .filter(|f| predicate_fires(pred, f))
            .count() as f64
            / fragile_features.len() as f64
    };
    // false_positive: fraction of safe positions where predicate fires (we DON'T want those preserved)
    let fp = if safe_features.is_empty() {
        0.0
    } else {
        safe_features
            .iter()
            .filter(|f| predicate_fires(pred, f))
            .count() as f64
            / safe_features.len() as f64
    };
    (coverage, fp)
}

fn make_candidate(
    pred: Predicate,
    label: &str,
    fragile: &[PosFeatures],
    safe: &[PosFeatures],
) -> GuardCandidate {
    let complexity = pred.complexity();
    let (cov, fp) = evaluate_candidate(&pred, fragile, safe);
    GuardCandidate {
        predicate: pred,
        complexity,
        fragile_coverage: cov,
        safe_false_positive: fp,
        label: label.to_string(),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Main guard synthesis entry point
// ────────────────────────────────────────────────────────────────────────────

/// Search the DSL predicate space for a guard that:
/// - covers all fragile positions (coverage ~ 1.0)
/// - has low false-positive rate on safe positions
///
/// Priority: atomics → conjunctions → depth-2 compounds.
pub fn synthesize_guard(
    fit: &FitContext,
    localize: &LocalizeResult,
    merged_codes: &[usize],
    _target: usize,
) -> Option<GuardCandidate> {
    let target_group = fit.group;

    // Collect fragile features: positions of the fragile code in the worst prompt
    // that, when kept oracle, clear the strict-gate failure.
    let worst_capture = fit
        .captures
        .iter()
        .find(|c| c.id == localize.worst_prompt_id)?;

    let fragile_positions = &localize.fragile_positions;
    let fragile_features: Vec<PosFeatures> = fragile_positions
        .iter()
        .map(|&pos| extract_features(worst_capture, pos, fit.head.head))
        .collect();

    // Safe features: all positions across all eval prompts where the fragile
    // code appears but the position is NOT in the fragile set (or is in a
    // different prompt), meaning they can safely be merged to target.
    let mut safe_features: Vec<PosFeatures> = Vec::new();
    for capture in &fit.captures {
        for (pos, codes) in capture.oracle_codes.iter().enumerate() {
            let orig = codes[target_group];
            if !merged_codes.contains(&orig) {
                continue;
            }
            // Skip the fragile positions themselves.
            let is_fragile =
                capture.id == localize.worst_prompt_id && localize.fragile_positions.contains(&pos);
            if is_fragile {
                continue;
            }
            safe_features.push(extract_features(capture, pos, fit.head.head));
        }
    }

    eprintln!(
        "  guard synthesis: {} fragile features, {} safe features",
        fragile_features.len(),
        safe_features.len()
    );

    // Generate candidates, depth 1 then depth 2.
    let mut candidates: Vec<GuardCandidate> = vec![
        make_candidate(
            natural_prose_pred(),
            "natural_prose",
            &fragile_features,
            &safe_features,
        ),
        make_candidate(
            attends_bos_pred(),
            "attends_bos",
            &fragile_features,
            &safe_features,
        ),
        make_candidate(
            attends_prev_pred(),
            "attends_prev",
            &fragile_features,
            &safe_features,
        ),
        make_candidate(
            attends_bos_or_prev(),
            "attends_bos_or_prev",
            &fragile_features,
            &safe_features,
        ),
        make_candidate(
            Predicate::And(vec![natural_prose_pred(), attends_bos_pred()]),
            "natural_prose&&attends_bos",
            &fragile_features,
            &safe_features,
        ),
        make_candidate(
            Predicate::And(vec![natural_prose_pred(), attends_prev_pred()]),
            "natural_prose&&attends_prev",
            &fragile_features,
            &safe_features,
        ),
        make_candidate(
            Predicate::And(vec![natural_prose_pred(), attends_bos_or_prev()]),
            "natural_prose&&(attends_bos||attends_prev)",
            &fragile_features,
            &safe_features,
        ),
    ];

    for c in &candidates {
        eprintln!(
            "    {} coverage={:.3} fp={:.3} complexity={}",
            c.label, c.fragile_coverage, c.safe_false_positive, c.complexity
        );
    }

    // Select: highest coverage, then lowest false-positive, then lowest complexity.
    candidates.sort_by(|a, b| {
        b.fragile_coverage
            .partial_cmp(&a.fragile_coverage)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.safe_false_positive
                    .partial_cmp(&b.safe_false_positive)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.complexity.cmp(&b.complexity))
    });

    let best = candidates.into_iter().find(|c| c.fragile_coverage >= 0.8)?;
    eprintln!("  selected guard: {}", best.label);
    Some(best)
}
