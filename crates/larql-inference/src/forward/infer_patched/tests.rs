//! Tests for [`super`].
//!
//! Split out of `infer_patched.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

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

    let (predictions, _) = apply_knn_override(raw(&["a", "b", "c"]), &residuals, Some(&store), 1);

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
