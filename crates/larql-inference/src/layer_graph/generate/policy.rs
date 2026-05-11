//! Tokenizer-level generation policy shared by generation frontends.

use std::collections::HashSet;

use larql_compute::prelude::*;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use super::eos::EosConfig;
use super::lm_head::{lm_head_topk_with_policy, LmHeadPolicy};

const SUPPRESSED_TOKEN_CANDIDATE_TOPK: usize = 256;
const DEBUG_SUPPRESS_PROBE_IDS: &[u32] = &[5, 31, 4, 168, 184];
const ENV_DEBUG_TOKEN_IDS: &str = "LARQL_DEBUG_TOKEN_IDS";
const ENV_DEBUG_TOPK: &str = "LARQL_DEBUG_TOPK";
const ENV_METAL_COMPARE_CPU: &str = "LARQL_METAL_COMPARE_CPU";
const ENV_PROFILE_DECODE: &str = "LARQL_PROFILE_DECODE";
const ENV_PROFILE_SPLIT: &str = "LARQL_PROFILE_SPLIT";

#[derive(Clone, Debug)]
pub(crate) struct TokenSelectionPolicy {
    pub debug_token_ids: bool,
    pub debug_topk: bool,
    pub suppress_candidate_topk: usize,
    pub lm_head: LmHeadPolicy,
}

impl Default for TokenSelectionPolicy {
    fn default() -> Self {
        Self {
            debug_token_ids: false,
            debug_topk: false,
            suppress_candidate_topk: SUPPRESSED_TOKEN_CANDIDATE_TOPK,
            lm_head: LmHeadPolicy::default(),
        }
    }
}

impl TokenSelectionPolicy {
    pub(crate) fn from_env() -> Self {
        Self {
            debug_token_ids: std::env::var(ENV_DEBUG_TOKEN_IDS).is_ok(),
            debug_topk: std::env::var(ENV_DEBUG_TOPK).is_ok(),
            suppress_candidate_topk: SUPPRESSED_TOKEN_CANDIDATE_TOPK,
            lm_head: LmHeadPolicy::from_env(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GenerationRuntimeConfig {
    pub compare_cpu: bool,
    pub profile_decode: bool,
    pub profile_split: bool,
    pub lm_head: LmHeadPolicy,
}

impl GenerationRuntimeConfig {
    pub(crate) fn from_env() -> Self {
        Self {
            compare_cpu: std::env::var(ENV_METAL_COMPARE_CPU).is_ok(),
            profile_decode: std::env::var(ENV_PROFILE_DECODE).is_ok(),
            profile_split: std::env::var(ENV_PROFILE_SPLIT).is_ok(),
            lm_head: LmHeadPolicy::from_env(),
        }
    }
}

/// IDs of tokens that should never be picked during text generation.
///
/// Built from the tokenizer's `added_tokens` table (everything marked
/// `special: true`) minus any IDs in the EOS set. Vocab-resident structural
/// markers like `<unusedN>` and `[multimodal]` are also suppressed.
pub(crate) fn build_special_suppress_set_with_policy(
    tokenizer: &tokenizers::Tokenizer,
    eos: &EosConfig,
    policy: &TokenSelectionPolicy,
) -> HashSet<u32> {
    let mut out = HashSet::new();
    for (&id, added) in tokenizer.get_added_tokens_decoder().iter() {
        if added.special && !eos.eos_token_ids.contains(&id) {
            out.insert(id);
        }
    }

    let vocab = tokenizer.get_vocab(true);
    let mut structural_count = 0;
    for (tok, &id) in vocab.iter() {
        if eos.eos_token_ids.contains(&id) || out.contains(&id) {
            continue;
        }
        if is_structural_marker(tok) {
            out.insert(id);
            structural_count += 1;
        }
    }

    if policy.debug_token_ids {
        eprintln!(
            "[suppress] {} ids ({} from added_tokens.special, {} from structural-marker scan)",
            out.len(),
            out.len() - structural_count,
            structural_count,
        );
        let mut sorted: Vec<u32> = out.iter().copied().collect();
        sorted.sort_unstable();
        let sample: Vec<String> = sorted
            .iter()
            .take(20)
            .map(|id| {
                let raw = tokenizer.id_to_token(*id).unwrap_or_default();
                format!("{id}={raw:?}")
            })
            .collect();
        eprintln!("[suppress] first 20: {}", sample.join(", "));
        for &probe in DEBUG_SUPPRESS_PROBE_IDS {
            let raw = tokenizer.id_to_token(probe).unwrap_or_default();
            let in_set = out.contains(&probe);
            let in_vocab = vocab.contains_key(&raw);
            eprintln!(
                "[suppress] probe id={probe} raw={raw:?} in_set={in_set} in_vocab={in_vocab}"
            );
        }
    }
    out
}

fn is_structural_marker(tok: &str) -> bool {
    if tok.is_empty() {
        return false;
    }
    let trimmed = tok.trim();
    if trimmed.len() < 2 {
        return false;
    }
    let bytes = trimmed.as_bytes();
    let first = bytes[0];
    let last = bytes[bytes.len() - 1];
    let bracketed = (first == b'<' && last == b'>') || (first == b'[' && last == b']');
    if !bracketed {
        return false;
    }
    let body = &trimmed[1..trimmed.len() - 1];
    !body.is_empty() && !body.chars().any(char::is_whitespace)
}

/// Pick the top-1 vocabulary id from logits, skipping any id in `suppress`.
pub(crate) fn pick_next_filtered_with_policy(
    index: &VectorIndex,
    weights: &ModelWeights,
    h: &ndarray::Array1<f32>,
    backend: &dyn ComputeBackend,
    suppress: &HashSet<u32>,
    tokenizer: &tokenizers::Tokenizer,
    policy: &TokenSelectionPolicy,
) -> u32 {
    if suppress.is_empty() && !policy.debug_topk {
        return lm_head_topk_with_policy(index, weights, h, 1, backend, &policy.lm_head)
            .into_iter()
            .next()
            .map(|(id, _)| id)
            .unwrap_or(0);
    }

    let candidates = lm_head_topk_with_policy(
        index,
        weights,
        h,
        policy.suppress_candidate_topk,
        backend,
        &policy.lm_head,
    );
    if policy.debug_topk {
        let summary: Vec<String> = candidates
            .iter()
            .take(8)
            .map(|(id, score)| {
                let raw = tokenizer.id_to_token(*id).unwrap_or_default();
                let mark = if suppress.contains(id) { "x" } else { " " };
                format!("{mark}id={id:6} {score:+.4e} {raw:?}")
            })
            .collect();
        let max_abs = candidates.iter().fold(0.0f32, |a, &(_, s)| a.max(s.abs()));
        let nan_count = candidates.iter().filter(|(_, s)| s.is_nan()).count();
        let zero_count = candidates.iter().filter(|(_, s)| *s == 0.0).count();
        let suppressed_in_top16 = candidates
            .iter()
            .take(16)
            .filter(|(id, _)| suppress.contains(id))
            .count();
        eprintln!(
            "    top8: {}\n    (max|score|={max_abs:.6e}  zeros={zero_count}/{}  nans={nan_count}  suppressed_top16={suppressed_in_top16}/16)",
            summary.join("  |  "),
            candidates.len()
        );
    }
    candidates
        .iter()
        .find(|(id, _)| !suppress.contains(id))
        .or_else(|| candidates.first())
        .map(|(id, _)| *id)
        .unwrap_or(0)
}
