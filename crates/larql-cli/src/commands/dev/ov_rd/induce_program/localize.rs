// Diagnostic-capture structs accumulate per-prompt fields (clears_failure,
// metrics, worst_prompt_kl, …) for a future dump/serializer; suppress until
// the viewer is wired.
#![allow(dead_code)]

use larql_vindex::VectorIndex;

use super::super::metrics::{argmax, kl_logp, log_softmax, mean, percentile};
use super::super::oracle_pq_forward::{final_logits, forward_q4k_predicted_address_mode_d_head};
use super::super::program::BehaviorMetrics;
use super::context::{FitContext, PromptCapture};

// ────────────────────────────────────────────────────────────────────────────
// Result types
// ────────────────────────────────────────────────────────────────────────────

pub struct LeaveCodeResult {
    pub left_oracle: usize,
    pub metrics: BehaviorMetrics,
    pub clears_failure: bool,
}

pub struct LeavePositionResult {
    pub position: usize,
    pub original_code: usize,
    pub prompt_kl: f64,
    pub clears_failure: bool,
}

pub struct LocalizeResult {
    /// Codes that, when kept oracle while the rest are merged, clear the failure.
    pub fragile_codes: Vec<usize>,
    /// The worst prompt ID from the full-split evaluation of the failing program.
    pub worst_prompt_id: String,
    pub worst_prompt_kl: f64,
    /// Within the worst prompt: positions of the fragile code that, when
    /// individually kept oracle, minimally clear the strict-gate failure.
    pub fragile_positions: Vec<usize>,
    pub fragile_code: usize,
    /// Leave-code-oracle results (all merged codes).
    pub leave_code_results: Vec<LeaveCodeResult>,
    /// Leave-position-oracle results (worst prompt only, fragile code positions).
    pub leave_position_results: Vec<LeavePositionResult>,
}

// ────────────────────────────────────────────────────────────────────────────
// Core: evaluate a program with one code kept oracle on the full split
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate a merge map on the full eval split, with `leave_oracle_code`
/// excluded (it stays at its oracle code). All other `merged_codes` → target.
pub fn eval_leave_code_oracle(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    fit: &FitContext,
    merged_codes: &[usize],
    target: usize,
    leave_oracle_code: Option<usize>,
) -> Result<BehaviorMetrics, Box<dyn std::error::Error>> {
    let target_group = fit.group;
    let mut kls = Vec::with_capacity(fit.captures.len());
    let mut top1_hits = 0usize;
    let mut top5_hits = 0usize;

    for capture in &fit.captures {
        let remapped: Vec<Vec<usize>> = capture
            .oracle_codes
            .iter()
            .map(|codes| {
                let mut c = codes.clone();
                let original = codes[target_group];
                if merged_codes.contains(&original) {
                    let leave_it = leave_oracle_code.map(|l| l == original).unwrap_or(false);
                    if !leave_it {
                        c[target_group] = target;
                    }
                }
                c
            })
            .collect();

        let h = forward_q4k_predicted_address_mode_d_head(
            weights,
            &capture.token_ids,
            index,
            fit.head,
            &fit.mode_d_table,
            &remapped,
            &capture.stratum,
        )?;
        let logits = final_logits(weights, &h);
        let logp = log_softmax(&logits);
        let top1 = argmax(&logits);
        let top5 = super::super::metrics::top_k_indices(&logits, 5);

        kls.push(kl_logp(&capture.baseline_logp, &logp));
        if capture.baseline_top1 == top1 {
            top1_hits += 1;
        }
        if top5.contains(&capture.baseline_top1) {
            top5_hits += 1;
        }
    }

    let n = fit.captures.len().max(1) as f64;
    Ok(BehaviorMetrics {
        mean_kl: mean(&kls),
        p95_kl: percentile(kls.clone(), 0.95),
        max_kl: kls.iter().cloned().fold(0.0_f64, f64::max),
        top1: top1_hits as f64 / n,
        top5: top5_hits as f64 / n,
    })
}

/// Evaluate a single prompt with one specific (code, position) kept oracle.
/// All other merged_code positions → target.
pub fn eval_leave_position_oracle(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    fit: &FitContext,
    capture: &PromptCapture,
    merged_codes: &[usize],
    target: usize,
    fragile_code: usize,
    leave_oracle_position: Option<usize>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let target_group = fit.group;

    let remapped: Vec<Vec<usize>> = capture
        .oracle_codes
        .iter()
        .enumerate()
        .map(|(pos, codes)| {
            let mut c = codes.clone();
            let original = codes[target_group];
            if merged_codes.contains(&original) {
                let is_fragile_at_this_pos = original == fragile_code
                    && leave_oracle_position.map(|lp| lp == pos).unwrap_or(false);
                if !is_fragile_at_this_pos {
                    c[target_group] = target;
                }
            }
            c
        })
        .collect();

    let h = forward_q4k_predicted_address_mode_d_head(
        weights,
        &capture.token_ids,
        index,
        fit.head,
        &fit.mode_d_table,
        &remapped,
        &capture.stratum,
    )?;
    let logits = final_logits(weights, &h);
    let logp = log_softmax(&logits);
    Ok(kl_logp(&capture.baseline_logp, &logp))
}

// ────────────────────────────────────────────────────────────────────────────
// Main localization entry point
// ────────────────────────────────────────────────────────────────────────────

pub fn localize_failure(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    fit: &FitContext,
    merged_codes: &[usize],
    target: usize,
    _failing_max_kl: f64,
) -> Result<LocalizeResult, Box<dyn std::error::Error>> {
    // Step 1: find worst prompt from the full-merge evaluation.
    let _full_metrics = eval_leave_code_oracle(weights, index, fit, merged_codes, target, None)?;
    let (worst_idx, worst_kl) = fit
        .captures
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let remapped: Vec<Vec<usize>> = c
                .oracle_codes
                .iter()
                .map(|codes| {
                    let mut r = codes.clone();
                    if merged_codes.contains(&codes[fit.group]) {
                        r[fit.group] = target;
                    }
                    r
                })
                .collect();
            let kl = forward_q4k_predicted_address_mode_d_head(
                weights,
                &c.token_ids,
                index,
                fit.head,
                &fit.mode_d_table,
                &remapped,
                &c.stratum,
            )
            .ok()
            .map(|h| kl_logp(&c.baseline_logp, &log_softmax(&final_logits(weights, &h))))
            .unwrap_or(0.0);
            (i, kl)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0));

    let worst_prompt = &fit.captures[worst_idx];
    let worst_prompt_id = worst_prompt.id.clone();

    eprintln!("  worst prompt: {} KL={:.6}", worst_prompt_id, worst_kl);

    // Step 2: leave-code-oracle for each merged code.
    let strict_max = 0.03_f64;
    let mut leave_code_results = Vec::new();
    let mut fragile_codes = Vec::new();

    for &code in merged_codes {
        eprint!("  leave-oracle code {code}: ");
        let m = eval_leave_code_oracle(weights, index, fit, merged_codes, target, Some(code))?;
        let clears = m.max_kl <= strict_max && m.p95_kl <= strict_max && m.top1 >= 0.99;
        eprintln!(
            "mean={:.6} max={:.6} [{}]",
            m.mean_kl,
            m.max_kl,
            if clears { "clears" } else { "still_fails" }
        );
        if clears {
            fragile_codes.push(code);
        }
        leave_code_results.push(LeaveCodeResult {
            left_oracle: code,
            metrics: m,
            clears_failure: clears,
        });
    }

    // Step 3: leave-position-oracle on worst prompt for fragile codes.
    // Use the first fragile code identified (most common case: one fragile code).
    let fragile_code = fragile_codes.first().copied().unwrap_or(merged_codes[0]);
    let fragile_pos_in_worst: Vec<usize> = worst_prompt
        .oracle_codes
        .iter()
        .enumerate()
        .filter(|(_, codes)| codes[fit.group] == fragile_code)
        .map(|(pos, _)| pos)
        .collect();

    eprintln!(
        "  fragile code={fragile_code}, {} positions in '{}'",
        fragile_pos_in_worst.len(),
        worst_prompt_id
    );

    let mut leave_position_results = Vec::new();
    let mut minimal_fragile_positions = Vec::new();

    for &pos in &fragile_pos_in_worst {
        let kl = eval_leave_position_oracle(
            weights,
            index,
            fit,
            worst_prompt,
            merged_codes,
            target,
            fragile_code,
            Some(pos),
        )?;
        let clears = kl <= strict_max;
        eprintln!(
            "    leave-oracle pos {pos}: KL={kl:.6} [{}]",
            if clears { "clears" } else { "fails" }
        );
        if clears {
            minimal_fragile_positions.push(pos);
        }
        leave_position_results.push(LeavePositionResult {
            position: pos,
            original_code: fragile_code,
            prompt_kl: kl,
            clears_failure: clears,
        });
    }

    eprintln!(
        "  minimal fragile positions: {:?}",
        minimal_fragile_positions
    );

    Ok(LocalizeResult {
        fragile_codes,
        worst_prompt_id,
        worst_prompt_kl: worst_kl,
        fragile_positions: minimal_fragile_positions,
        fragile_code,
        leave_code_results,
        leave_position_results,
    })
}
