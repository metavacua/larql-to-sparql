//! `infer_patched` — the single forward-pass entry point shared by the LQL
//! `INFER` executor (`larql-lql/src/executor/query/infer.rs`) and the Python
//! binding (`larql-python/src/vindex.rs`).
//!
//! Both surfaces must produce byte-identical top-k predictions for any
//! `(weights, gate_index, knn_store, prompt)` — see ADR 0001. This function
//! owns the three parameters that are easy to drift between callers:
//!
//!   1. `top_k_features` on the walk FFN — always unlimited, because a
//!      bounded cap misroutes post-INSERT on Gemma (a strong `×30` gate slot
//!      dominates a half-weakened baseline).
//!   2. The KNN cosine threshold — `KNN_COSINE_THRESHOLD = 0.75`.
//!   3. Layer iteration order — the first stored layer (lowest index) whose
//!      top-1 cosine exceeds the threshold wins.
//!
//! Callers pass a `&dyn GateIndex` + `Option<&KnnStore>`. `PatchedVindex`
//! bundles both; `PyVindex` keeps them as separate fields. Both pass through
//! here.

use larql_vindex::{GateIndex, KnnStore, PatchedVindex, VectorIndex, WalkHit};
use tokenizers::Tokenizer;

use crate::model::ModelWeights;
use crate::vindex::WalkFfn;
use crate::vindex::{predict_kquant_with_ffn, predict_kquant_with_ffn_early_exit};

use super::predict::{predict_with_ffn, predict_with_ffn_early_exit};
use super::PredictResult;

/// Cosine threshold for the L0 KnnStore override. A stored key whose top-1
/// cosine against the captured residual exceeds this value replaces the
/// walk FFN's top-1 prediction.
pub const KNN_COSINE_THRESHOLD: f32 = 0.75;

/// Metadata for a KNN override, if one fired.
#[derive(Clone, Debug)]
pub struct KnnOverride {
    pub token: String,
    pub cosine: f32,
    pub layer: usize,
}

/// Which KnnStore router the forward pass uses. `Legacy` is the default and is
/// byte-identical to the original top-1 + fixed-`KNN_COSINE_THRESHOLD` gate
/// (ADR 0001). `Verified` (FR1) and `TwoTier` (FR2) are opt-in — selected per
/// statement by the LQL `ROUTE` clause, or globally via the `LARQL_KNN_*` env
/// vars through [`KnnRouteMode::from_env`].
#[derive(Clone, Debug, PartialEq, Default)]
pub enum KnnRouteMode {
    /// Top-1 cosine > threshold wins at the first stored layer (legacy).
    #[default]
    Legacy,
    /// FR1: top-`k` candidates + entity-in-prompt verify + abstain,
    /// resolved-layer-first. See [`apply_knn_override_verified`].
    Verified { k: usize, threshold: f32 },
    /// FR2: `Verified` tier 1 then a top-1 activation alias fallback.
    /// See [`apply_knn_override_two_tier`].
    TwoTier { k: usize, threshold: f32 },
}

impl KnnRouteMode {
    /// Resolve the mode from the `LARQL_KNN_*` env vars — the opt-in default
    /// used by callers (Python, EXPLAIN, install) that don't carry an explicit
    /// LQL `ROUTE` clause. `LARQL_KNN_VERIFY` → `Verified`; adding
    /// `LARQL_KNN_FALLBACK` → `TwoTier`; neither → `Legacy`. Knobs: `LARQL_KNN_TOPK`,
    /// `LARQL_KNN_MIN_COS`.
    pub fn from_env() -> Self {
        match knn_verify_config() {
            None => Self::Legacy,
            Some(cfg) if cfg.fallback => Self::TwoTier {
                k: cfg.k_candidates,
                threshold: cfg.threshold,
            },
            Some(cfg) => Self::Verified {
                k: cfg.k_candidates,
                threshold: cfg.threshold,
            },
        }
    }
}

/// Result of the shared INFER pipeline.
pub struct InferPatchedResult {
    /// Top-k predictions. When `knn_override` is `Some`, position 0 holds the
    /// stored target token with probability `1.0` and positions `1..k` hold
    /// the walk FFN's own top-`(k-1)`. When `None`, this is the walk FFN's
    /// raw top-k.
    pub predictions: Vec<(String, f64)>,
    /// Walk FFN's raw top-1 before the KnnStore post-logits override is
    /// applied. This lets display layers show what the model path produced
    /// before an unmaterialized retrieval sidecar changed the answer.
    pub model_top1: Option<(String, f64)>,
    /// Metadata on the KNN override for callers that want to surface it
    /// (e.g. the LQL display layer prints `"KNN override, cos=X, L{layer}"`).
    pub knn_override: Option<KnnOverride>,
    /// Per-layer residuals captured at the last-token position during the
    /// walk FFN pass. LQL uses these to build its inference trace.
    pub residuals: Vec<(usize, Vec<f32>)>,
    /// Wall-clock milliseconds for the walk FFN pass itself.
    pub walk_ms: f64,
}

/// Run a full forward pass with the walk FFN, consult the KnnStore for a
/// possible top-1 override, and return the top-k predictions.
///
/// This is the **only** implementation of the INFER pipeline. `exec_infer`
/// (LQL) and `PyVindex::infer` (Python) both delegate here. Per ADR 0001 any
/// new forward-pass surface MUST call this function rather than assembling a
/// local pipeline.
pub fn infer_patched(
    weights: &ModelWeights,
    tokenizer: &Tokenizer,
    gate_index: &dyn GateIndex,
    knn_store: Option<&KnnStore>,
    token_ids: &[u32],
    top_k: usize,
    route_mode: &KnnRouteMode,
) -> InferPatchedResult {
    let walk_ffn = WalkFfn::new_unlimited_with_trace(weights, gate_index);

    let start = std::time::Instant::now();
    let PredictResult {
        predictions: raw, ..
    } = predict_with_ffn(weights, tokenizer, token_ids, top_k, &walk_ffn);
    let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

    let residuals = walk_ffn.take_residuals();
    let model_top1 = raw.first().cloned();
    let (predictions, knn_override) = route_knn_override(
        raw, &residuals, knn_store, top_k, route_mode, tokenizer, token_ids,
    );

    InferPatchedResult {
        predictions,
        model_top1,
        knn_override,
        residuals,
        walk_ms,
    }
}

/// **Early-exit** variant of `infer_patched` (FR retrieval-augmented early
/// exit). Runs the walk forward only as far as the highest stored KnnStore
/// layer L\*, checks the FR1 verified router there, and — if a verified hit
/// fires — returns the stored target immediately, **skipping layers L\*+1..end
/// and the lm_head**. On a miss it transparently completes the full forward, so
/// the result is identical to `infer_patched` in `Verified` mode (the verified
/// router checks every stored layer ≤ L\*, which are all computed by L\*).
///
/// The returned `bool` is `true` when the early exit fired. Parity is structural
/// (residuals ≤ L\* are independent of the tail) and proven bit-exact in
/// `examples/fr_early_exit_parity.rs`; this is the production-path wiring whose
/// tok/s win is measured in `examples/fr_early_exit_bench.rs`.
#[allow(clippy::too_many_arguments)]
pub fn infer_patched_early_exit(
    weights: &ModelWeights,
    tokenizer: &Tokenizer,
    gate_index: &dyn GateIndex,
    knn_store: Option<&KnnStore>,
    token_ids: &[u32],
    top_k: usize,
    k_candidates: usize,
    threshold: f32,
) -> (InferPatchedResult, bool) {
    let walk_ffn = WalkFfn::new_unlimited_with_trace(weights, gate_index);

    // Check at the highest stored layer — by then every stored-layer residual
    // (all ≤ L*) has been captured, so the verified route sees the same set it
    // would post-hoc. No store / no valid layer → never exits (full forward).
    let stop = knn_store
        .map(|s| s.layers())
        .and_then(|ls| ls.into_iter().filter(|l| *l < weights.num_layers).max())
        .unwrap_or_else(|| weights.num_layers.saturating_sub(1));

    let prompt = tokenizer.decode(token_ids, true).unwrap_or_default();
    let prompt_lc = prompt.to_lowercase();

    let mut fired: Option<KnnOverride> = None;
    let start = std::time::Instant::now();
    let (predictions, exited);
    {
        let mut on_stop = || -> Option<Vec<(String, f64)>> {
            let store = knn_store?;
            let residuals = walk_ffn.peek_residuals();
            let ovr = verified_route(store, &residuals, &prompt_lc, k_candidates, threshold)?;
            let preds = assemble_predictions(Vec::new(), &Some(ovr.clone()), top_k);
            fired = Some(ovr);
            Some(preds)
        };
        (predictions, exited) = predict_with_ffn_early_exit(
            weights,
            tokenizer,
            token_ids,
            top_k,
            &walk_ffn,
            stop,
            &mut on_stop,
        );
    }
    let walk_ms = start.elapsed().as_secs_f64() * 1000.0;
    let residuals = walk_ffn.take_residuals();
    // On an exit the model's own lm_head never ran, so there is no model_top1.
    let model_top1 = if exited {
        None
    } else {
        predictions.first().cloned()
    };

    (
        InferPatchedResult {
            predictions,
            model_top1,
            knn_override: fired,
            residuals,
            walk_ms,
        },
        exited,
    )
}

#[allow(clippy::too_many_arguments)]
/// Q4K variant of `infer_patched`. Identical contract but routes the forward
/// pass through `predict_kquant_with_ffn`, which dequantises one layer at a time
/// from the vindex instead of reading pre-loaded f32 tensors.
pub fn infer_patched_q4k(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    gate_index: &dyn GateIndex,
    knn_store: Option<&KnnStore>,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    route_mode: &KnnRouteMode,
) -> InferPatchedResult {
    // SAFETY: WalkFfn reads only `weights.arch` and `weights.vectors` (neither
    // of which is mutated by `predict_kquant_with_ffn`). The q4k forward pass
    // mutates only `weights.tensors` (inserting/removing per-layer attn matrices).
    // These are non-overlapping HashMap fields — the aliased read is sound.
    let weights_ref: &ModelWeights = unsafe { &*(weights as *const ModelWeights) };
    let walk_ffn = WalkFfn::new_unlimited_with_trace(weights_ref, gate_index);

    let start = std::time::Instant::now();
    let PredictResult {
        predictions: raw, ..
    } = predict_kquant_with_ffn(weights, tokenizer, token_ids, top_k, index, &walk_ffn);
    let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

    let residuals = walk_ffn.take_residuals();
    let model_top1 = raw.first().cloned();
    let (predictions, knn_override) = route_knn_override(
        raw, &residuals, knn_store, top_k, route_mode, tokenizer, token_ids,
    );

    InferPatchedResult {
        predictions,
        model_top1,
        knn_override,
        residuals,
        walk_ms,
    }
}

/// Q4K early-exit — the Q4_K twin of [`infer_patched_early_exit`]. Same
/// short-circuit contract (stop at the highest stored layer L\*, emit the
/// verified target, skip the tail + lm_head; on a miss complete the full
/// forward), routed through the per-layer-dequant q4k forward.
#[allow(clippy::too_many_arguments)]
pub fn infer_patched_q4k_early_exit(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    gate_index: &dyn GateIndex,
    knn_store: Option<&KnnStore>,
    token_ids: &[u32],
    top_k: usize,
    index: &VectorIndex,
    k_candidates: usize,
    threshold: f32,
) -> (InferPatchedResult, bool) {
    // SAFETY: identical aliasing argument to `infer_patched_q4k` — WalkFfn reads
    // only `weights.arch`/`weights.vectors`, the q4k forward mutates only
    // `weights.tensors`.
    let weights_ref: &ModelWeights = unsafe { &*(weights as *const ModelWeights) };
    let walk_ffn = WalkFfn::new_unlimited_with_trace(weights_ref, gate_index);

    let stop = knn_store
        .map(|s| s.layers())
        .and_then(|ls| ls.into_iter().filter(|l| *l < weights.num_layers).max())
        .unwrap_or_else(|| weights.num_layers.saturating_sub(1));
    let prompt = tokenizer.decode(token_ids, true).unwrap_or_default();
    let prompt_lc = prompt.to_lowercase();

    let mut fired: Option<KnnOverride> = None;
    let start = std::time::Instant::now();
    let (predictions, exited);
    {
        let mut on_stop = || -> Option<Vec<(String, f64)>> {
            let store = knn_store?;
            let residuals = walk_ffn.peek_residuals();
            let ovr = verified_route(store, &residuals, &prompt_lc, k_candidates, threshold)?;
            let preds = assemble_predictions(Vec::new(), &Some(ovr.clone()), top_k);
            fired = Some(ovr);
            Some(preds)
        };
        (predictions, exited) = predict_kquant_with_ffn_early_exit(
            weights,
            tokenizer,
            token_ids,
            top_k,
            index,
            &walk_ffn,
            stop,
            &mut on_stop,
        );
    }
    let walk_ms = start.elapsed().as_secs_f64() * 1000.0;
    let residuals = walk_ffn.take_residuals();
    let model_top1 = if exited {
        None
    } else {
        predictions.first().cloned()
    };

    (
        InferPatchedResult {
            predictions,
            model_top1,
            knn_override: fired,
            residuals,
            walk_ms,
        },
        exited,
    )
}

/// Pure function: given raw walk predictions, per-layer residuals, and an
/// optional KnnStore, return `(predictions, knn_override)`.
///
/// Split out of `infer_patched` to be unit-testable without a real forward
/// pass. The behaviour is the contract that ADR 0001's byte-identical claim
/// rests on: the first stored layer (lowest index) whose top-1 cosine against
/// the captured residual exceeds `KNN_COSINE_THRESHOLD` replaces position 0
/// of the top-k with the stored target token at probability `1.0`; positions
/// `1..top_k` are the walk FFN's own top-`(top_k - 1)`.
pub fn apply_knn_override(
    raw: Vec<(String, f64)>,
    residuals: &[(usize, Vec<f32>)],
    knn_store: Option<&KnnStore>,
    top_k: usize,
) -> (Vec<(String, f64)>, Option<KnnOverride>) {
    let knn_override = knn_store.and_then(|store| {
        if store.is_empty() {
            return None;
        }
        let layers = store.layers();
        for (layer, residual) in residuals {
            if !layers.contains(layer) {
                continue;
            }
            if let Some((entry, cosine)) = store.query_top1(*layer, residual) {
                if cosine > KNN_COSINE_THRESHOLD {
                    return Some(KnnOverride {
                        token: entry.target_token.clone(),
                        cosine,
                        layer: *layer,
                    });
                }
            }
        }
        None
    });

    let predictions = assemble_predictions(raw, &knn_override, top_k);
    (predictions, knn_override)
}

/// Default number of activation candidates the verified router considers per
/// stored layer (`LARQL_KNN_TOPK` overrides). FR1 measured top-5 recall ~0.95
/// where top-1 was 0.89, so 5 candidates is the verify pool.
pub const KNN_VERIFY_TOPK: usize = 5;

/// FR1 build — **top-k + verify + abstain** override. Opt-in (the default path
/// is `apply_knn_override`); enabled via `LARQL_KNN_VERIFY` in the forward
/// entry points, so default behaviour is byte-identical (the parity spine).
///
/// The FR1 measurement ([`docs/diagnoses/fr1-topk-fuzzy-router.md`]) showed the
/// top-1 + fixed-0.75 gate is non-discriminative: near-rank-1 residuals clear
/// 0.75 on ~every query (gate fired 150/150) and inject a confident-wrong fact
/// 11% of the time at the resolved layer (84% at an early phrasing-trap layer).
/// This path fixes both failure modes:
///
///   1. **Resolved-layer-first.** The entity key sharpens with depth (FR1/FR3:
///      early layers are phrasing-traps, the entity resolves in later layers —
///      the *specific* resolved layer is model-dependent, e.g. ~L24-L26 on
///      Gemma-3-4B), so iterate whatever layers the store holds highest-first
///      rather than lowest-first. No layer index is hardcoded — the store's
///      layers come from wherever `INSERT … MODE KNN` installed for this model.
///   2. **Verify, don't trust cosine.** Among the top-`k_candidates`, override
///      only with a fact whose stored `entity` the prompt actually names. A
///      cross-entity collision (the confident-wrong case) is rejected; a correct
///      entity sitting at rank 2-5 (top-5 recall ~0.95) is still found.
///   3. **Abstain.** If no candidate verifies, return raw with no override —
///      the model answers rather than a wrong fact being injected.
///
/// `threshold` is a permissive floor (cosine is non-discriminative; the verify
/// is the real gate). Alias resolution where the prompt does *not* name the
/// canonical entity is FR2's two-tier job, not this verifier.
pub fn apply_knn_override_verified(
    raw: Vec<(String, f64)>,
    residuals: &[(usize, Vec<f32>)],
    knn_store: Option<&KnnStore>,
    top_k: usize,
    prompt: &str,
    k_candidates: usize,
    threshold: f32,
) -> (Vec<(String, f64)>, Option<KnnOverride>) {
    let prompt_lc = prompt.to_lowercase();
    let knn_override = knn_store
        .and_then(|store| verified_route(store, residuals, &prompt_lc, k_candidates, threshold));
    let predictions = assemble_predictions(raw, &knn_override, top_k);
    (predictions, knn_override)
}

/// FR2 build — **two-tier router**: symbolic-primary (the FR1 verify, i.e. the
/// prompt names the routed entity) → **activation-fuzzy fallback** when no
/// candidate's entity is named. Opt-in (`LARQL_KNN_VERIFY` + `LARQL_KNN_FALLBACK`
/// in the forward entry points); default behaviour is byte-identical.
///
/// FR2's measurement ([`docs/diagnoses/fr2-two-tier-router.md`]) showed exact
/// entity-string routing resolves 0/10 historical aliases (the canonical name
/// is absent — "Persia" ≠ "Iran") while the activation key recovers them. So:
///
///   1. **Tier 1 (verify).** Exactly `verified_route` — if the prompt names a
///      top-`k` candidate's entity, override with it (precision-1.0 path, the
///      confident-wrong fix from FR1).
///   2. **Tier 2 (fallback).** If tier 1 abstains (no entity named — the alias /
///      paraphrase case), take the **top-1 activation candidate** at the
///      resolved layer above `threshold`. This recovers aliases exact-string
///      can't, at the honest cost FR2/E16 flagged: the fallback is a fuzzy
///      ~0.7-0.9 route with NO entity-name guard, so on an OPEN query about a
///      non-stored entity it confident-wrongs exactly like the legacy gate
///      (the gain benchmark measured 0/20 distractor-safe vs verified's 20/20 —
///      `docs/diagnoses/fr-routing-gain.md`). **Use this only for queries known
///      to be aliases of stored entities; `Verified` is the safe open default.**
pub fn apply_knn_override_two_tier(
    raw: Vec<(String, f64)>,
    residuals: &[(usize, Vec<f32>)],
    knn_store: Option<&KnnStore>,
    top_k: usize,
    prompt: &str,
    k_candidates: usize,
    threshold: f32,
) -> (Vec<(String, f64)>, Option<KnnOverride>) {
    let prompt_lc = prompt.to_lowercase();
    let knn_override = knn_store.and_then(|store| {
        verified_route(store, residuals, &prompt_lc, k_candidates, threshold)
            .or_else(|| fallback_route(store, residuals, threshold))
    });
    let predictions = assemble_predictions(raw, &knn_override, top_k);
    (predictions, knn_override)
}

/// Stored layers present in `residuals`, **highest-first** (resolved-layer-first
/// — the entity key sharpens with depth; the resolved layer is model-dependent,
/// never hardcoded). Shared by both router tiers.
fn stored_layers_high_first<'a>(
    store: &KnnStore,
    residuals: &'a [(usize, Vec<f32>)],
) -> Vec<&'a (usize, Vec<f32>)> {
    let layers = store.layers();
    let mut stored: Vec<&(usize, Vec<f32>)> = residuals
        .iter()
        .filter(|(l, _)| layers.contains(l))
        .collect();
    stored.sort_by_key(|(l, _)| std::cmp::Reverse(*l));
    stored
}

/// Tier 1 — verified route: the first top-`k` candidate (resolved-layer-first,
/// cosine > `threshold`) whose stored `entity` the lowercased prompt names.
/// `None` = abstain.
fn verified_route(
    store: &KnnStore,
    residuals: &[(usize, Vec<f32>)],
    prompt_lc: &str,
    k_candidates: usize,
    threshold: f32,
) -> Option<KnnOverride> {
    if store.is_empty() {
        return None;
    }
    for (layer, residual) in stored_layers_high_first(store, residuals) {
        for (entry, cosine) in store.query_knn(*layer, residual, k_candidates) {
            if cosine <= threshold {
                break; // query_knn is descending — nothing further passes
            }
            if !entry.entity.is_empty() && prompt_lc.contains(&entry.entity.to_lowercase()) {
                return Some(KnnOverride {
                    token: entry.target_token.clone(),
                    cosine,
                    layer: *layer,
                });
            }
        }
    }
    None
}

/// Tier 2 — fuzzy fallback: top-1 activation candidate at the resolved layer
/// above `threshold`, no string verification (the alias case has nothing to
/// verify against). Lower-confidence than tier 1; `None` = abstain.
fn fallback_route(
    store: &KnnStore,
    residuals: &[(usize, Vec<f32>)],
    threshold: f32,
) -> Option<KnnOverride> {
    if store.is_empty() {
        return None;
    }
    for (layer, residual) in stored_layers_high_first(store, residuals) {
        if let Some((entry, cosine)) = store.query_top1(*layer, residual) {
            if cosine > threshold {
                return Some(KnnOverride {
                    token: entry.target_token.clone(),
                    cosine,
                    layer: *layer,
                });
            }
        }
    }
    None
}

/// Place a fired override at position 0 (probability `1.0`) ahead of the walk
/// FFN's own top-`(top_k - 1)`; pass `raw` through unchanged when no override
/// fired or `top_k == 0`. Shared by both override paths so they assemble the
/// result identically.
fn assemble_predictions(
    raw: Vec<(String, f64)>,
    knn_override: &Option<KnnOverride>,
    top_k: usize,
) -> Vec<(String, f64)> {
    match knn_override {
        Some(ovr) if top_k > 0 => {
            let mut out = Vec::with_capacity(top_k);
            out.push((ovr.token.clone(), 1.0));
            for pair in raw.into_iter().take(top_k.saturating_sub(1)) {
                out.push(pair);
            }
            out
        }
        _ => raw,
    }
}

/// Resolved opt-in router config from the environment.
struct KnnRouteConfig {
    /// Top-k candidates the verifier considers (`LARQL_KNN_TOPK`).
    k_candidates: usize,
    /// Cosine floor (`LARQL_KNN_MIN_COS`).
    threshold: f32,
    /// FR2 alias fallback enabled (`LARQL_KNN_FALLBACK`).
    fallback: bool,
}

/// `Some(cfg)` when `LARQL_KNN_VERIFY` is set (FR1, plus FR2 if
/// `LARQL_KNN_FALLBACK`), else `None` (legacy top-1 + fixed-gate, byte-identical).
fn knn_verify_config() -> Option<KnnRouteConfig> {
    std::env::var_os("LARQL_KNN_VERIFY")?;
    let k_candidates = std::env::var("LARQL_KNN_TOPK")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&k| k > 0)
        .unwrap_or(KNN_VERIFY_TOPK);
    let threshold = std::env::var("LARQL_KNN_MIN_COS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(KNN_COSINE_THRESHOLD);
    let fallback = std::env::var_os("LARQL_KNN_FALLBACK").is_some();
    Some(KnnRouteConfig {
        k_candidates,
        threshold,
        fallback,
    })
}

/// Dispatch to the legacy override (default), the FR1 verified router, or the
/// FR2 two-tier router per `mode`. Decodes the prompt for the verifier only when
/// an opt-in path is enabled.
fn route_knn_override(
    raw: Vec<(String, f64)>,
    residuals: &[(usize, Vec<f32>)],
    knn_store: Option<&KnnStore>,
    top_k: usize,
    mode: &KnnRouteMode,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
) -> (Vec<(String, f64)>, Option<KnnOverride>) {
    match mode {
        KnnRouteMode::Legacy => apply_knn_override(raw, residuals, knn_store, top_k),
        KnnRouteMode::Verified { k, threshold } => {
            let prompt = tokenizer.decode(token_ids, true).unwrap_or_default();
            apply_knn_override_verified(raw, residuals, knn_store, top_k, &prompt, *k, *threshold)
        }
        KnnRouteMode::TwoTier { k, threshold } => {
            let prompt = tokenizer.decode(token_ids, true).unwrap_or_default();
            apply_knn_override_two_tier(raw, residuals, knn_store, top_k, &prompt, *k, *threshold)
        }
    }
}

/// Rebuild a per-layer walk trace from captured residuals — shared between
/// the LQL `INFER` / `EXPLAIN INFER` display paths and the HTTP `/explain`
/// route. Each layer's residual is re-queried against the patched vindex's
/// gate KNN for the top-20 hits, then paired with `FeatureMeta` for display.
///
/// Kept here so that any surface using `infer_patched` can reconstruct the
/// same trace view without duplicating the loop or re-consuming WalkFfn's
/// internal `take_trace` (which drains residuals and so can't coexist with
/// the KNN-override residual capture above).
pub fn walk_trace_from_residuals(
    residuals: &[(usize, Vec<f32>)],
    patched: &PatchedVindex,
) -> Vec<(usize, Vec<WalkHit>)> {
    let mut out = Vec::with_capacity(residuals.len());
    for (layer, residual) in residuals {
        let r = ndarray::Array1::from_vec(residual.clone());
        let hits = patched.gate_knn(*layer, &r, 20);
        let walk_hits: Vec<WalkHit> = hits
            .into_iter()
            .filter_map(|(feature, gate_score)| {
                let meta = patched.feature_meta(*layer, feature)?;
                Some(WalkHit {
                    layer: *layer,
                    feature,
                    gate_score,
                    cosine_score: 0.0,
                    meta,
                })
            })
            .collect();
        out.push((*layer, walk_hits));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store_with_key(layer: usize, key: Vec<f32>, target: &str) -> KnnStore {
        let mut store = KnnStore::default();
        store.add(
            layer,
            key,
            0,
            target.to_string(),
            "Atlantis".to_string(),
            "capital".to_string(),
            1.0,
        );
        store
    }

    fn raw(tokens: &[&str]) -> Vec<(String, f64)> {
        tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), 1.0 - 0.1 * i as f64))
            .collect()
    }

    #[test]
    fn no_store_passes_through_raw_topk() {
        let raw = raw(&["a", "b", "c"]);
        let residuals: Vec<(usize, Vec<f32>)> = vec![(5, vec![1.0, 0.0, 0.0])];

        let (predictions, override_) = apply_knn_override(raw.clone(), &residuals, None, 3);

        assert!(override_.is_none());
        assert_eq!(predictions, raw);
    }

    #[test]
    fn empty_store_passes_through() {
        let raw = raw(&["a", "b", "c"]);
        let residuals = vec![(5, vec![1.0, 0.0, 0.0])];
        let store = KnnStore::default();

        let (predictions, override_) = apply_knn_override(raw.clone(), &residuals, Some(&store), 3);

        assert!(override_.is_none());
        assert_eq!(predictions, raw);
    }

    #[test]
    fn matching_key_overrides_position_zero() {
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon");

        let (predictions, override_) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 3);

        let ovr = override_.expect("key exactly matches residual — override must fire");
        assert_eq!(ovr.token, "Poseidon");
        assert_eq!(ovr.layer, 5);
        assert!(
            ovr.cosine > 0.99,
            "cosine of identical vectors must be ~1.0"
        );

        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0], ("Poseidon".to_string(), 1.0));
        assert_eq!(predictions[1].0, "a");
        assert_eq!(predictions[2].0, "b");
    }

    #[test]
    fn mismatched_key_below_threshold_passes_through() {
        // Orthogonal vectors → cos = 0, well below 0.75 threshold.
        let residuals = vec![(5, vec![1.0, 0.0, 0.0])];
        let store = make_store_with_key(5, vec![0.0, 1.0, 0.0], "Poseidon");

        let (predictions, override_) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 3);

        assert!(
            override_.is_none(),
            "orthogonal residual must not trigger override"
        );
        assert_eq!(predictions[0].0, "a");
    }

    #[test]
    fn override_only_fires_on_stored_layers() {
        // Residual matches a key, but at a layer not present in the store.
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(7, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon");

        let (predictions, override_) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 3);

        assert!(
            override_.is_none(),
            "residual layer not in store — no override"
        );
        assert_eq!(predictions[0].0, "a");
    }

    #[test]
    fn first_matching_layer_wins() {
        // Two stored layers both match; the earliest one (by iteration order
        // of the residuals slice) must take precedence.
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone()), (7, key.clone())];
        let mut store = make_store_with_key(5, key.clone(), "First");
        store.add(
            7,
            key,
            1,
            "Second".to_string(),
            "Atlantis".to_string(),
            "capital".to_string(),
            1.0,
        );

        let (predictions, override_) = apply_knn_override(raw(&["a"]), &residuals, Some(&store), 5);

        let ovr = override_.unwrap();
        assert_eq!(ovr.token, "First");
        assert_eq!(ovr.layer, 5);
        assert_eq!(predictions[0].0, "First");
    }

    #[test]
    fn top_k_one_returns_only_override() {
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon");

        let (predictions, _) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 1);

        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0], ("Poseidon".to_string(), 1.0));
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon");

        let (predictions, override_) =
            apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 0);

        // Override metadata still fires (the match is real) but predictions
        // collapses to raw (which is then truncated by the caller if needed).
        assert!(override_.is_some());
        assert_eq!(predictions.len(), 3);
    }

    // ── apply_knn_override_verified (FR1 build: top-k + verify + abstain) ──

    #[test]
    fn verified_entity_in_prompt_overrides() {
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon"); // entity "Atlantis"
        let (pred, ovr) = apply_knn_override_verified(
            raw(&["a", "b", "c"]),
            &residuals,
            Some(&store),
            3,
            "The capital of Atlantis is",
            5,
            KNN_COSINE_THRESHOLD,
        );
        let o = ovr.expect("entity named in prompt + cosine match → override fires");
        assert_eq!(o.token, "Poseidon");
        assert_eq!(pred[0], ("Poseidon".to_string(), 1.0));
    }

    #[test]
    fn verified_entity_not_in_prompt_abstains() {
        // The headline confident-wrong fix: residual matches the key exactly
        // (cos = 1.0, would fire the legacy 0.75 gate) but the prompt does NOT
        // name the stored entity → abstain rather than inject a wrong fact.
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon"); // entity "Atlantis"
        let (pred, ovr) = apply_knn_override_verified(
            raw(&["a", "b", "c"]),
            &residuals,
            Some(&store),
            3,
            "The capital of France is",
            5,
            KNN_COSINE_THRESHOLD,
        );
        assert!(
            ovr.is_none(),
            "cos=1.0 but entity absent from prompt → abstain"
        );
        assert_eq!(pred[0].0, "a");
    }

    #[test]
    fn verified_picks_correct_candidate_from_topk() {
        // Top-1 (by cosine) is the wrong entity for this prompt; the right
        // entity sits at rank 2 and IS named — verify rescues it (the top-5
        // recall the legacy top-1 path throws away).
        let mut store = make_store_with_key(5, vec![1.0, 0.0, 0.0], "Poseidon"); // Atlantis
        store.add(
            5,
            vec![0.8, 0.6, 0.0],
            1,
            "Lemuria".into(),
            "Zog".into(),
            "capital".into(),
            1.0,
        );
        let residuals = vec![(5, vec![1.0, 0.0, 0.0])]; // nearest to Atlantis
        let (pred, ovr) = apply_knn_override_verified(
            raw(&["a"]),
            &residuals,
            Some(&store),
            3,
            "The capital of Zog is",
            5,
            KNN_COSINE_THRESHOLD,
        );
        let o = ovr.expect("rank-2 entity named in prompt → override with it");
        assert_eq!(o.token, "Lemuria");
        assert_eq!(pred[0].0, "Lemuria");
    }

    #[test]
    fn verified_prefers_higher_resolved_layer() {
        // Both stored layers match and are named; resolved-layer-first picks the
        // HIGHER layer (contrast `first_matching_layer_wins`, the legacy path).
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone()), (7, key.clone())];
        let mut store = make_store_with_key(5, key.clone(), "Low"); // entity Atlantis
        store.add(
            7,
            key,
            1,
            "High".into(),
            "Atlantis".into(),
            "capital".into(),
            1.0,
        );
        let (pred, ovr) = apply_knn_override_verified(
            raw(&["a"]),
            &residuals,
            Some(&store),
            3,
            "The capital of Atlantis is",
            5,
            KNN_COSINE_THRESHOLD,
        );
        let o = ovr.expect("named + match → override");
        assert_eq!(
            o.layer, 7,
            "highest stored layer wins (resolved-layer-first)"
        );
        assert_eq!(o.token, "High");
        assert_eq!(pred[0].0, "High");
    }

    #[test]
    fn verified_below_threshold_abstains() {
        // Entity is named, but the residual is orthogonal to the key (cos 0) →
        // below the floor → abstain.
        let residuals = vec![(5, vec![1.0, 0.0, 0.0])];
        let store = make_store_with_key(5, vec![0.0, 1.0, 0.0], "Poseidon"); // Atlantis
        let (pred, ovr) = apply_knn_override_verified(
            raw(&["a", "b"]),
            &residuals,
            Some(&store),
            3,
            "The capital of Atlantis is",
            5,
            KNN_COSINE_THRESHOLD,
        );
        assert!(
            ovr.is_none(),
            "cosine below floor → abstain even if entity named"
        );
        assert_eq!(pred[0].0, "a");
    }

    // ── apply_knn_override_two_tier (FR2 build: symbolic → alias fallback) ──

    #[test]
    fn two_tier_verify_tier_fires_when_named() {
        // Entity named → tier 1 (verify) fires, same as the FR1 path.
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let store = make_store_with_key(5, key, "Poseidon"); // entity Atlantis
        let (pred, ovr) = apply_knn_override_two_tier(
            raw(&["a"]),
            &residuals,
            Some(&store),
            3,
            "The capital of Atlantis is",
            5,
            KNN_COSINE_THRESHOLD,
        );
        assert_eq!(ovr.expect("named → tier-1 fires").token, "Poseidon");
        assert_eq!(pred[0].0, "Poseidon");
    }

    #[test]
    fn two_tier_fallback_recovers_alias() {
        // The FR2 win: residual matches the Iran key but the prompt says
        // "Persia" (Iran not named) → tier 1 abstains, tier 2 fallback recovers.
        let key = vec![1.0, 0.0, 0.0];
        let residuals = vec![(5, key.clone())];
        let mut store = KnnStore::default();
        store.add(
            5,
            key,
            0,
            "Tehran".into(),
            "Iran".into(),
            "capital".into(),
            1.0,
        );
        let (pred, ovr) = apply_knn_override_two_tier(
            raw(&["a", "b"]),
            &residuals,
            Some(&store),
            3,
            "The capital of Persia is",
            5,
            KNN_COSINE_THRESHOLD,
        );
        let o = ovr.expect("alias: tier-1 abstains, tier-2 fallback recovers Iran");
        assert_eq!(o.token, "Tehran");
        assert_eq!(pred[0].0, "Tehran");
    }

    #[test]
    fn two_tier_fallback_below_threshold_abstains() {
        // Entity not named AND cosine below floor → both tiers abstain.
        let residuals = vec![(5, vec![1.0, 0.0, 0.0])];
        let mut store = KnnStore::default();
        store.add(
            5,
            vec![0.0, 1.0, 0.0],
            0,
            "Tehran".into(),
            "Iran".into(),
            "capital".into(),
            1.0,
        );
        let (pred, ovr) = apply_knn_override_two_tier(
            raw(&["a", "b"]),
            &residuals,
            Some(&store),
            3,
            "The capital of Persia is",
            5,
            KNN_COSINE_THRESHOLD,
        );
        assert!(ovr.is_none(), "no name + below floor → abstain");
        assert_eq!(pred[0].0, "a");
    }

    #[test]
    fn two_tier_prefers_verified_over_fallback() {
        // Top-1 by cosine is Zog (not named); rank-2 Atlantis IS named. Tier 1
        // (verify) must win with Atlantis, not tier-2's top-1 Zog.
        let mut store = KnnStore::default();
        store.add(
            5,
            vec![1.0, 0.0, 0.0],
            0,
            "Lemuria".into(),
            "Zog".into(),
            "capital".into(),
            1.0,
        );
        store.add(
            5,
            vec![0.9, 0.4359, 0.0],
            1,
            "Poseidon".into(),
            "Atlantis".into(),
            "capital".into(),
            1.0,
        );
        let residuals = vec![(5, vec![1.0, 0.0, 0.0])];
        let (pred, ovr) = apply_knn_override_two_tier(
            raw(&["a"]),
            &residuals,
            Some(&store),
            3,
            "The capital of Atlantis is",
            5,
            KNN_COSINE_THRESHOLD,
        );
        assert_eq!(
            ovr.expect("override fires").token,
            "Poseidon",
            "tier-1 verify (Atlantis named) beats tier-2 top-1 (Zog)"
        );
        assert_eq!(pred[0].0, "Poseidon");
    }

    // ── KnnRouteMode::from_env (LARQL_KNN_* → mode) ────────────────────
    //
    // Env is process-global; no other test in this crate reads the
    // `LARQL_KNN_*` vars (the forward entry points take an explicit
    // `&KnnRouteMode`), so this single test owns them. It sets, asserts,
    // and clears each var in sequence, leaving the environment clean.

    #[test]
    fn from_env_maps_vars_to_modes() {
        use std::env::{remove_var, set_var};

        let clear = || {
            remove_var("LARQL_KNN_VERIFY");
            remove_var("LARQL_KNN_FALLBACK");
            remove_var("LARQL_KNN_TOPK");
            remove_var("LARQL_KNN_MIN_COS");
        };

        // Default (nothing set) → Legacy, byte-identical to the old gate.
        clear();
        assert_eq!(KnnRouteMode::from_env(), KnnRouteMode::Legacy);

        // LARQL_KNN_VERIFY alone → Verified with the default top-k + floor.
        clear();
        set_var("LARQL_KNN_VERIFY", "1");
        assert_eq!(
            KnnRouteMode::from_env(),
            KnnRouteMode::Verified {
                k: KNN_VERIFY_TOPK,
                threshold: KNN_COSINE_THRESHOLD,
            }
        );

        // Adding LARQL_KNN_FALLBACK promotes Verified → TwoTier; TOPK /
        // MIN_COS override the knobs.
        clear();
        set_var("LARQL_KNN_VERIFY", "1");
        set_var("LARQL_KNN_FALLBACK", "1");
        set_var("LARQL_KNN_TOPK", "9");
        set_var("LARQL_KNN_MIN_COS", "0.5");
        assert_eq!(
            KnnRouteMode::from_env(),
            KnnRouteMode::TwoTier {
                k: 9,
                threshold: 0.5,
            }
        );

        // A zero / unparseable TOPK is ignored, falling back to the default.
        clear();
        set_var("LARQL_KNN_VERIFY", "1");
        set_var("LARQL_KNN_TOPK", "0");
        assert_eq!(
            KnnRouteMode::from_env(),
            KnnRouteMode::Verified {
                k: KNN_VERIFY_TOPK,
                threshold: KNN_COSINE_THRESHOLD,
            }
        );

        // FALLBACK without VERIFY does nothing (VERIFY is the gate).
        clear();
        set_var("LARQL_KNN_FALLBACK", "1");
        assert_eq!(KnnRouteMode::from_env(), KnnRouteMode::Legacy);

        clear();
    }

    // ── infer_patched (full forward pass) ──────────────────────────────

    #[test]
    fn infer_patched_returns_top_k_predictions_and_residuals() {
        use crate::test_utils::TestFixtures;
        let fx = TestFixtures::build();
        let tokens = vec![0u32, 1, 2];
        let result = infer_patched(
            &fx.weights,
            &fx.tokenizer,
            &fx.index,
            None,
            &tokens,
            5,
            &KnnRouteMode::Legacy,
        );
        assert!(result.predictions.len() <= 5);
        // Walk pass populates residuals at every layer.
        assert!(!result.residuals.is_empty());
        assert!(result.knn_override.is_none());
        assert_eq!(result.model_top1, result.predictions.first().cloned());
        assert!(result.walk_ms >= 0.0);
    }

    #[test]
    fn walk_trace_from_residuals_returns_per_layer_walk_hits() {
        use crate::test_utils::TestFixtures;
        let fx = TestFixtures::build();
        let patched = larql_vindex::PatchedVindex::new(fx.index);
        let residuals = vec![
            (0usize, vec![0.1f32; fx.weights.hidden_size]),
            (1usize, vec![0.2f32; fx.weights.hidden_size]),
        ];
        let trace = walk_trace_from_residuals(&residuals, &patched);
        // One entry per residual.
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].0, 0);
        assert_eq!(trace[1].0, 1);
        // Synthetic vindex returns no FeatureMeta, so walk_hits is empty
        // — but the per-layer entry must still be present.
    }

    #[test]
    fn walk_trace_from_residuals_empty_input_returns_empty() {
        use crate::test_utils::TestFixtures;
        let fx = TestFixtures::build();
        let patched = larql_vindex::PatchedVindex::new(fx.index);
        let trace = walk_trace_from_residuals(&[], &patched);
        assert!(trace.is_empty());
    }

    #[test]
    fn infer_patched_q4k_returns_predictions_via_quantised_path() {
        // Exercises `infer_patched_q4k` end-to-end — same contract as
        // `infer_patched` but routes through the Q4K dequant forward
        // path. Uses the Q4KTestFixtures so the vindex has Q4K bytes
        // for attention + FFN.
        use crate::test_utils::Q4KTestFixtures;
        let mut fx = Q4KTestFixtures::build();
        let tokens = vec![0u32, 1, 2];
        let result = infer_patched_q4k(
            &mut fx.weights,
            &fx.tokenizer,
            &fx.index,
            None,
            &tokens,
            5,
            &fx.index,
            &KnnRouteMode::Legacy,
        );
        assert!(result.predictions.len() <= 5);
        assert!(result.knn_override.is_none());
        assert_eq!(result.model_top1, result.predictions.first().cloned());
        assert!(result.walk_ms >= 0.0);
    }

    #[test]
    fn infer_patched_with_knn_store_override_routes_through() {
        use crate::test_utils::TestFixtures;
        let fx = TestFixtures::build();
        let tokens = vec![0u32, 1];
        // First, run without override to capture the residuals — then plant
        // a key matching the L0 residual exactly so the override fires on
        // the rerun.
        let baseline = infer_patched(
            &fx.weights,
            &fx.tokenizer,
            &fx.index,
            None,
            &tokens,
            3,
            &KnnRouteMode::Legacy,
        );
        let (l0_layer, l0_residual) = baseline
            .residuals
            .first()
            .expect("at least one residual captured");
        let store = make_store_with_key(*l0_layer, l0_residual.clone(), "PLANTED");
        let result = infer_patched(
            &fx.weights,
            &fx.tokenizer,
            &fx.index,
            Some(&store),
            &tokens,
            3,
            &KnnRouteMode::Legacy,
        );
        let ovr = result
            .knn_override
            .expect("planted key matching residual must fire override");
        assert_eq!(ovr.token, "PLANTED");
        assert_eq!(result.predictions[0].0, "PLANTED");
        assert!((result.predictions[0].1 - 1.0).abs() < 1e-6);
        // model_top1 reflects the unoverridden walk pass.
        assert_eq!(result.model_top1, baseline.predictions.first().cloned());
    }
}
