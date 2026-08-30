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
    // `WalkFfn` serves every layer locally and leaves `forward_moe_full_layer`
    // at the trait default (`Ok(None)`), so it has no way to refuse. Asserting
    // that here keeps the impossibility auditable: if a refusing backend is
    // ever threaded through this path it fails loudly instead of answering
    // with a token the route declined to compute.
    let PredictResult {
        predictions: raw, ..
    } = predict_kquant_with_ffn(weights, tokenizer, token_ids, top_k, index, &walk_ffn)
        .expect("WalkFfn cannot refuse a layer; a refusal here needs a real error channel");
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
        // See `infer_patched_q4k`: `WalkFfn` leaves `forward_moe_full_layer` at
        // the trait default, so it cannot refuse. Loud if that ever changes.
        (predictions, exited) = predict_kquant_with_ffn_early_exit(
            weights,
            tokenizer,
            token_ids,
            top_k,
            index,
            &walk_ffn,
            stop,
            &mut on_stop,
        )
        .expect("WalkFfn cannot refuse a layer; a refusal here needs a real error channel");
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

/// Env var: top-k candidates the KNN verifier considers (defaults to
/// [`KNN_VERIFY_TOPK`]).
const ENV_KNN_TOPK: &str = "LARQL_KNN_TOPK";
/// Env var: cosine floor for KNN verification (defaults to
/// [`KNN_COSINE_THRESHOLD`]).
const ENV_KNN_MIN_COS: &str = "LARQL_KNN_MIN_COS";

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
    let k_candidates = std::env::var(ENV_KNN_TOPK)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&k| k > 0)
        .unwrap_or(KNN_VERIFY_TOPK);
    let threshold = std::env::var(ENV_KNN_MIN_COS)
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
///
/// NOTE (2026-07-30 review, item 17): this is deliberately a POST-HOC KNN
/// view — "what does the patched index associate with these residuals" —
/// so its hits are built via `WalkHit::from_gate` with the execution
/// fields `None`. For the features a walk actually executed, use
/// `WalkFfn::take_trace` / `take_runtime_trace`, which emit from the
/// executed path.
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
                Some(WalkHit::from_gate(*layer, feature, gate_score, meta))
            })
            .collect();
        out.push((*layer, walk_hits));
    }
    out
}

#[cfg(test)]
mod tests;
