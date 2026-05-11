// SynthesisStep variants (Fail, NeedsOracle.suggested_program) are
// constructed for a diagnostic trace dump; suppress until consumed.
#![allow(dead_code)]

use std::path::PathBuf;

use clap::Args;

use super::program_cache::ProgramCache;

#[derive(Args)]
pub(super) struct SynthesizeProgramArgs {
    /// Program cache JSON built by build-program-cache.
    #[arg(long)]
    cache: PathBuf,

    /// Output directory for the synthesized program.
    #[arg(long)]
    out: PathBuf,
}

pub(super) fn run_synthesize_program(
    args: SynthesizeProgramArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    let cache: ProgramCache = serde_json::from_str(&std::fs::read_to_string(&args.cache)?)?;

    eprintln!(
        "Synthesizing program for L{}H{} group {} (fingerprint {})",
        cache.head_layer, cache.head_idx, cache.group, cache.codebook_fingerprint
    );
    eprintln!(
        "Oracle Mode D: mean {:.6}  p95 {:.6?}  max {:.6}  gate: {}",
        cache.oracle.mean_kl, cache.oracle.p95_kl, cache.oracle.max_kl, cache.oracle.gate
    );

    let synthesis = synthesize(&cache);
    print_synthesis_trace(&synthesis);
    emit_artifacts(&synthesis, &cache, &args.out)?;
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Synthesis result
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
enum SynthesisStep {
    Pairwise {
        source: usize,
        target: usize,
        gate: String,
    },
    SetMerge {
        sources: Vec<usize>,
        target: usize,
        gate: String,
    },
    GuardFound {
        name: String,
        gate: String,
        mean_kl: f64,
        max_kl: f64,
    },
    NeedsOracle {
        reason: String,
        suggested_program: String,
    },
    Fail {
        reason: String,
    },
}

#[derive(Debug)]
struct SynthesisResult {
    steps: Vec<SynthesisStep>,
    outcome: SynthesisOutcome,
}

#[derive(Debug)]
enum SynthesisOutcome {
    /// Program found entirely from cache — no new forwards needed.
    CacheHit { program_name: String },
    /// Candidate program proposed; needs one full-forward verification.
    NeedsVerification { reason: String },
    /// Cache insufficient; synthesis blocked.
    Blocked { reason: String },
}

// ────────────────────────────────────────────────────────────────────────────
// Synthesis algorithm (zero forwards)
// ────────────────────────────────────────────────────────────────────────────

fn synthesize(cache: &ProgramCache) -> SynthesisResult {
    let mut steps = Vec::new();

    // Phase 1: collect strict-pass pairwise edges.
    let pairs = cache.strict_pass_pairwise();
    if pairs.is_empty() {
        return SynthesisResult {
            steps,
            outcome: SynthesisOutcome::Blocked { reason: "no strict-pass pairwise edges in cache; run eval-program on pairwise variants first".to_string() },
        };
    }
    for &(src, tgt) in &pairs {
        let key = format!("{src}->{tgt}");
        if let Some(c) = cache.substitutions.get(&key) {
            steps.push(SynthesisStep::Pairwise {
                source: src,
                target: tgt,
                gate: c.gate.clone(),
            });
        }
    }

    // Phase 2: find dominant sink and compatible merge clique.
    let sink = match cache.dominant_sink() {
        Some(s) => s,
        None => {
            return SynthesisResult {
                steps,
                outcome: SynthesisOutcome::Blocked {
                    reason: "cannot determine dominant sink from pairwise edges".to_string(),
                },
            }
        }
    };
    let merge_candidates = cache.merge_candidates_for(sink);
    eprintln!("  dominant sink={sink}  merge candidates={merge_candidates:?}");

    // Build the candidate clique: the sink itself + everything that merges into it.
    let mut clique = merge_candidates.clone();
    clique.push(sink);
    clique.sort_unstable();
    clique.dedup();

    // Phase 3: check if a set-merge of this clique is in cache.
    let set_merge_key = super::program_cache::set_merge_key(&clique, sink);
    match cache.get_set_merge(&clique, sink) {
        Some(c) if c.passes_strict() => {
            steps.push(SynthesisStep::SetMerge {
                sources: clique.clone(),
                target: sink,
                gate: c.gate.clone(),
            });
            // Strict pass without guard — return this directly.
            let program_name = c.source.clone();
            return SynthesisResult {
                steps,
                outcome: SynthesisOutcome::CacheHit { program_name },
            };
        }
        Some(c) if c.passes_smoke() => {
            steps.push(SynthesisStep::SetMerge {
                sources: clique.clone(),
                target: sink,
                gate: c.gate.clone(),
            });
            // Smoke pass — need a guard. Try to find one in guarded_programs.
        }
        Some(c) => {
            steps.push(SynthesisStep::SetMerge {
                sources: clique.clone(),
                target: sink,
                gate: c.gate.clone(),
            });
            // Outright fail — may still need guard.
        }
        None => {
            steps.push(SynthesisStep::NeedsOracle {
                reason: format!("set merge {set_merge_key} not in cache"),
                suggested_program: format!(
                    "larql dev ov-rd eval-program --program variants/class_collapse_like.json ..."
                ),
            });
        }
    }

    // Phase 4: check if any guarded program passes strict gate.
    let guarded = cache.strict_pass_guarded();
    if guarded.is_empty() {
        return SynthesisResult {
            steps,
            outcome: SynthesisOutcome::NeedsVerification {
                reason: format!(
                    "set merge of clique {clique:?}→{sink} does not pass strict gate; \
                     no cached guarded programs pass — need guard synthesis (full forward)"
                ),
            },
        };
    }

    // Found a guarded program that passes.
    let (best_name, best) = guarded
        .into_iter()
        .min_by(|(_, a), (_, b)| {
            a.mean_kl
                .partial_cmp(&b.mean_kl)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    steps.push(SynthesisStep::GuardFound {
        name: best_name.to_string(),
        gate: best.gate.clone(),
        mean_kl: best.mean_kl,
        max_kl: best.max_kl,
    });

    SynthesisResult {
        steps,
        outcome: SynthesisOutcome::CacheHit {
            program_name: best_name.to_string(),
        },
    }
}

fn print_synthesis_trace(result: &SynthesisResult) {
    eprintln!("\n=== Synthesis trace ===");
    for step in &result.steps {
        match step {
            SynthesisStep::Pairwise {
                source,
                target,
                gate,
            } => eprintln!("  pairwise {source}->{target}: {gate}"),
            SynthesisStep::SetMerge {
                sources,
                target,
                gate,
            } => eprintln!("  set-merge {sources:?}->{target}: {gate}"),
            SynthesisStep::GuardFound {
                name,
                gate,
                mean_kl,
                max_kl,
            } => eprintln!("  guard found: {name}  [{gate}]  mean={mean_kl:.6}  max={max_kl:.6}"),
            SynthesisStep::NeedsOracle { reason, .. } => eprintln!("  → NEEDS ORACLE: {reason}"),
            SynthesisStep::Fail { reason } => eprintln!("  → FAIL: {reason}"),
        }
    }
    match &result.outcome {
        SynthesisOutcome::CacheHit { program_name } => {
            eprintln!("\n✓ Cache hit: program '{program_name}' passes strict gate")
        }
        SynthesisOutcome::NeedsVerification { reason } => {
            eprintln!("\n→ Needs verification: {reason}")
        }
        SynthesisOutcome::Blocked { reason } => eprintln!("\n✗ Blocked: {reason}"),
    }
}

fn emit_artifacts(
    result: &SynthesisResult,
    cache: &ProgramCache,
    out: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let report = serde_json::json!({
        "head": {"layer": cache.head_layer, "head": cache.head_idx},
        "group": cache.group,
        "outcome": match &result.outcome {
            SynthesisOutcome::CacheHit { program_name } => serde_json::json!({
                "type": "cache_hit",
                "program_name": program_name,
            }),
            SynthesisOutcome::NeedsVerification { reason } => serde_json::json!({
                "type": "needs_verification",
                "reason": reason,
            }),
            SynthesisOutcome::Blocked { reason } => serde_json::json!({
                "type": "blocked",
                "reason": reason,
            }),
        },
    });
    let out_path = out.join("synthesis_result.json");
    serde_json::to_writer_pretty(std::fs::File::create(&out_path)?, &report)?;
    eprintln!("Wrote {}", out_path.display());
    Ok(())
}
