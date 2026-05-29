use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::program::{BehaviorMetrics, Program, ProgramRule};

// ────────────────────────────────────────────────────────────────────────────
// Cache types
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CachedResult {
    pub(super) mean_kl: f64,
    pub(super) p95_kl: Option<f64>,
    pub(super) max_kl: f64,
    pub(super) top1: f64,
    pub(super) top5: f64,
    pub(super) gate: String,
    pub(super) source: String,
}

impl CachedResult {
    fn from_metrics(metrics: &BehaviorMetrics, source: &str) -> Self {
        let gate = if metrics.passes_strict() {
            "strict_pass"
        } else if metrics.passes_smoke() {
            "smoke_pass"
        } else {
            "fail"
        };
        CachedResult {
            mean_kl: metrics.mean_kl,
            p95_kl: Some(metrics.p95_kl),
            max_kl: metrics.max_kl,
            top1: metrics.top1,
            top5: metrics.top5,
            gate: gate.to_string(),
            source: source.to_string(),
        }
    }

    /// Whether this result was stored with `p95_kl = max_kl` as a fallback.
    /// The gate methods warn when this fallback is load-bearing.
    fn p95_is_max_fallback(&self) -> bool {
        match self.p95_kl {
            Some(p95) => (p95 - self.max_kl).abs() < 1e-9,
            None => false,
        }
    }

    pub(super) fn passes_strict(&self) -> bool {
        let p95 = self.p95_kl.unwrap_or(self.max_kl);
        if self.p95_is_max_fallback() && p95 > 0.025 {
            eprintln!(
                "  WARNING: strict-gate p95 check for '{}' using max_kl={:.6} as fallback — measure actual p95 before relying on this result",
                self.source, p95
            );
        }
        self.mean_kl <= 0.005
            && p95 <= 0.03
            && self.max_kl <= 0.03
            && self.top1 >= 0.99
            && self.top5 >= 1.0
    }

    pub(super) fn passes_smoke(&self) -> bool {
        let p95 = self.p95_kl.unwrap_or(self.max_kl);
        self.mean_kl <= 0.01 && p95 <= 0.05 && self.top1 >= 0.95 && self.top5 >= 0.98
    }
}

/// All known behavioral measurements for a head/group.
/// Used as the oracle backing for the synthesizer — no forwards needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ProgramCache {
    pub(super) head_layer: usize,
    pub(super) head_idx: usize,
    pub(super) group: usize,
    pub(super) base_config_k: usize,
    pub(super) base_config_groups: usize,
    pub(super) base_config_bits_per_group: usize,
    pub(super) codebook_fingerprint: String,
    pub(super) oracle: CachedResult,
    /// Key = "A->B" (single-code merge).
    pub(super) substitutions: HashMap<String, CachedResult>,
    /// Key = "{A,B,...}->T" (set merge, parallel).
    pub(super) set_merges: HashMap<String, CachedResult>,
    /// Key = variant name (programs with guards).
    pub(super) guarded_programs: HashMap<String, CachedResult>,
}

impl ProgramCache {
    #[allow(dead_code)]
    pub(super) fn num_codes(&self) -> usize {
        1 << self.base_config_bits_per_group
    }

    /// All strict-pass single-code (A→B) pairs.
    pub(super) fn strict_pass_pairwise(&self) -> Vec<(usize, usize)> {
        self.substitutions
            .iter()
            .filter(|(_, c)| c.passes_strict())
            .filter_map(|(k, _)| parse_pairwise_key(k))
            .collect()
    }

    /// Find the most-voted sink code (appears most often as the target of
    /// strict-pass pairwise merges).
    pub(super) fn dominant_sink(&self) -> Option<usize> {
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for (_, tgt) in self.strict_pass_pairwise() {
            *votes.entry(tgt).or_default() += 1;
        }
        votes.into_iter().max_by_key(|&(_, v)| v).map(|(k, _)| k)
    }

    /// Codes that merge cleanly into `sink` (strict-pass pairwise).
    pub(super) fn merge_candidates_for(&self, sink: usize) -> Vec<usize> {
        self.strict_pass_pairwise()
            .into_iter()
            .filter(|(_, t)| *t == sink)
            .map(|(s, _)| s)
            .collect()
    }

    pub(super) fn get_set_merge(&self, sources: &[usize], target: usize) -> Option<&CachedResult> {
        self.set_merges.get(&set_merge_key(sources, target))
    }

    pub(super) fn strict_pass_guarded(&self) -> Vec<(&str, &CachedResult)> {
        self.guarded_programs
            .iter()
            .filter(|(_, c)| c.passes_strict())
            .map(|(k, c)| (k.as_str(), c))
            .collect()
    }
}

pub(super) fn set_merge_key(sources: &[usize], target: usize) -> String {
    let mut sorted = sources.to_vec();
    sorted.sort_unstable();
    format!(
        "{{{}}}->{target}",
        sorted
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(super) fn parse_pairwise_key(key: &str) -> Option<(usize, usize)> {
    let (l, r) = key.split_once("->")?;
    Some((l.trim().parse().ok()?, r.trim().parse().ok()?))
}

// ────────────────────────────────────────────────────────────────────────────
// Build-program-cache command
// ────────────────────────────────────────────────────────────────────────────

#[derive(Args)]
pub(super) struct BuildProgramCacheArgs {
    /// Registry JSON from the variant measurement runs.
    #[arg(long)]
    registry: PathBuf,

    /// Output path for program_cache.json.
    #[arg(long)]
    out: PathBuf,
}

pub(super) fn run_build_program_cache(
    args: BuildProgramCacheArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let registry_dir = args.registry.parent().unwrap_or(Path::new("."));
    let registry: Value = serde_json::from_str(&std::fs::read_to_string(&args.registry)?)?;

    let head_layer = registry["head"]["layer"].as_u64().unwrap_or(0) as usize;
    let head_idx = registry["head"]["head"].as_u64().unwrap_or(0) as usize;
    let group = registry["group"].as_u64().unwrap_or(0) as usize;
    let fp = registry["codebook_fingerprint"]
        .as_str()
        .unwrap_or("")
        .to_string();

    eprintln!("Building cache for L{head_layer}H{head_idx} group {group}  fp={fp}");

    let mut substitutions: HashMap<String, CachedResult> = HashMap::new();
    let mut set_merges: HashMap<String, CachedResult> = HashMap::new();
    let mut guarded_programs: HashMap<String, CachedResult> = HashMap::new();
    let mut oracle_result: Option<CachedResult> = None;
    let mut base_k = 192usize;
    let mut base_g = 48usize;
    let mut base_b = 4usize;

    let variants = match registry["variants"].as_array() {
        Some(v) => v.clone(),
        None => return Err("registry has no 'variants' array".into()),
    };

    for entry in &variants {
        let file_rel = match entry["file"].as_str() {
            Some(f) => f,
            None => continue,
        };
        let name = entry["name"].as_str().unwrap_or("unnamed");
        let path = registry_dir.join(file_rel);

        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  skip {name}: {e}");
                continue;
            }
        };
        let program: Program = match serde_json::from_str(&text) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  skip {name}: {e}");
                continue;
            }
        };

        base_k = program.base_config.k;
        base_g = program.base_config.groups;
        base_b = program.base_config.bits_per_group;

        // Use `metrics` if present; fall back to `reference_metrics` so that
        // manually-authored variants (where `metrics` is null) are still cached.
        let raw = program
            .metrics
            .as_ref()
            .map(|m| serde_json::to_value(m).ok())
            .flatten()
            .or_else(|| program.reference_metrics.clone());
        // Replace null p95_kl with max_kl so variants without p95 still parse.
        let metrics_json = raw.map(|mut v| {
            if v["p95_kl"].is_null() {
                if let Some(max) = v["max_kl"].as_f64() {
                    v["p95_kl"] = serde_json::json!(max);
                }
            }
            v
        });
        let metrics: BehaviorMetrics =
            match metrics_json.and_then(|v| serde_json::from_value(v).ok()) {
                Some(m) => m,
                None => {
                    eprintln!("  skip {name}: no metrics or reference_metrics");
                    continue;
                }
            };
        let cached = CachedResult::from_metrics(&metrics, name);
        eprintln!("  {name}: {}", cached.gate);

        classify_and_insert(
            &program,
            cached,
            name,
            &mut substitutions,
            &mut set_merges,
            &mut guarded_programs,
            &mut oracle_result,
        );
    }

    let oracle = oracle_result.unwrap_or_else(|| CachedResult {
        mean_kl: 0.002270,
        p95_kl: Some(0.010217),
        max_kl: 0.021975,
        top1: 1.0,
        top5: 1.0,
        gate: "strict_pass".to_string(),
        source: "64-balanced split reference (from RESULTS.md)".to_string(),
    });

    let cache = ProgramCache {
        head_layer,
        head_idx,
        group,
        base_config_k: base_k,
        base_config_groups: base_g,
        base_config_bits_per_group: base_b,
        codebook_fingerprint: fp,
        oracle,
        substitutions,
        set_merges,
        guarded_programs,
    };

    eprintln!("\nCache summary:");
    eprintln!("  {} pairwise substitutions", cache.substitutions.len());
    eprintln!("  {} set merges", cache.set_merges.len());
    eprintln!("  {} guarded programs", cache.guarded_programs.len());
    let pairs = cache.strict_pass_pairwise();
    eprintln!("  strict-pass pairwise: {pairs:?}");
    if let Some(sink) = cache.dominant_sink() {
        let cands = cache.merge_candidates_for(sink);
        eprintln!("  dominant sink={sink}  merge candidates={cands:?}");
    }
    let gp = cache.strict_pass_guarded();
    let gp_names: Vec<_> = gp.iter().map(|(k, _)| *k).collect();
    eprintln!("  strict-pass guarded: {gp_names:?}");

    serde_json::to_writer_pretty(std::fs::File::create(&args.out)?, &cache)?;
    eprintln!("Wrote {}", args.out.display());
    Ok(())
}

fn classify_and_insert(
    program: &Program,
    cached: CachedResult,
    name: &str,
    substitutions: &mut HashMap<String, CachedResult>,
    set_merges: &mut HashMap<String, CachedResult>,
    guarded_programs: &mut HashMap<String, CachedResult>,
    oracle_result: &mut Option<CachedResult>,
) {
    let all_rules: Vec<&ProgramRule> = program
        .stages
        .iter()
        .flat_map(|s| s.declared_rules.iter())
        .collect();

    let has_guard = all_rules.iter().any(|r| r.is_guarded());
    let has_set = all_rules
        .iter()
        .any(|r| matches!(r, ProgramRule::MapSet { .. }));

    if has_guard {
        guarded_programs.insert(name.to_string(), cached);
        return;
    }

    if all_rules.is_empty() {
        *oracle_result = Some(cached);
        return;
    }

    if !has_set && all_rules.len() == 1 {
        if let ProgramRule::Map { source, target } = all_rules[0] {
            substitutions.insert(format!("{source}->{target}"), cached);
            return;
        }
    }

    // Multi-rule or set-merge program.
    let key = infer_merge_key(program);
    set_merges.insert(key, cached);
}

fn infer_merge_key(program: &Program) -> String {
    let mut sources: Vec<usize> = Vec::new();
    let mut target_opt: Option<usize> = None;
    for stage in &program.stages {
        for rule in &stage.declared_rules {
            match rule {
                ProgramRule::MapSet { source, target } => {
                    sources.extend_from_slice(source);
                    target_opt = Some(*target);
                }
                ProgramRule::Map { source, target } => {
                    sources.push(*source);
                    target_opt = Some(*target);
                }
                _ => {}
            }
        }
    }
    target_opt
        .map(|t| set_merge_key(&sources, t))
        .unwrap_or_else(|| "unknown".to_string())
}
