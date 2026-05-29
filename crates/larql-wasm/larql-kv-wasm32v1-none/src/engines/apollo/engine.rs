//! ApolloEngine — retrieval-augmented generation via vec_inject.
//!
//! At prefill: routes the prompt through the RoutingIndex, retrieves the
//! most relevant VecInjectEntry records, computes a combined injection delta
//! (scaled token embeddings), then runs the forward pass on the context
//! (window_tokens ++ query_tokens) with the delta injected at `crystal_layer`.
//!
//! At decode: extends the context by one token per step and re-runs the
//! forward pass with the same injection delta. Generation is O(N) per step —
//! there is no K/V cache; accuracy comes from the injection residual.
//!
//! Memory: ~2.8 MB for 176 windows × 3,585 entries on the Apollo 11 corpus,
//! vs ~25.8 GB Standard KV at 370K tokens (~20,000× compression).
//!
//! Simplifications vs the full Python pipeline:
//! - Injection is at the last token position only (Python does per-entry
//!   `position_in_window`).
//! - Routing uses tf-idf-lite on raw token IDs (no stemming/stopwords).
//! - Boundary-residual replay not yet wired (`prefill_to_layer` is a TODO).

use ndarray::{s, Array1, Array2};
use thiserror::Error;

use super::entry::{InjectionConfig, VecInjectEntry};
use super::routing::{RoutingIndex, RoutingQuery};
use super::store::ApolloStore;
use crate::{EngineInfo, KvEngine};
use larql_inference::forward::{embed_tokens_pub, forward_from_layer, forward_raw_logits};
use larql_inference::model::ModelWeights;

/// (context_tokens, injection_delta, boundary_residual, crystal_layer)
type InjectionPrep = (Vec<u32>, ndarray::Array1<f32>, Option<Vec<f32>>, usize);

// ─── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ApolloError {
    #[error("store not loaded")]
    StoreNotLoaded,
    #[error("routing index not built — call build_routing_index() first")]
    RoutingNotBuilt,
    #[error("invalid window id: {0}")]
    InvalidWindowId(u16),
    #[error("forward pass failed")]
    Forward,
    #[error("no windows matched query (routing returned empty)")]
    NoMatch,
}

// ─── Trace types ─────────────────────────────────────────────────────────────

/// Summary of a single query answered by the engine.
#[derive(Debug, Clone)]
pub struct QueryTrace {
    pub routed_windows: Vec<u16>,
    pub injected_entries: Vec<VecInjectEntry>,
    pub context_tokens: usize,
    pub top1_token_id: u32,
    pub top1_logit: f32,
}

// ─── Engine struct ────────────────────────────────────────────────────────────

pub struct ApolloEngine {
    pub store: Option<ApolloStore>,
    pub routing: RoutingIndex,
    pub config: InjectionConfig,
    /// State maintained between prefill and decode steps.
    context_tokens: Vec<u32>,
    injection_delta: Option<Array1<f32>>,
    /// Boundary residual for the routed window (output of layer `crystal_layer - 1`).
    /// When `Some`, `prefill` and `decode_step` use `forward_from_layer` instead of
    /// running all 34 layers — ~8.5× faster on Gemma 3 4B (crystal_layer=30 → 4 layers).
    boundary_residual: Option<Vec<f32>>,
    crystal_layer: usize,
}

impl ApolloEngine {
    pub fn new(config: InjectionConfig) -> Self {
        Self {
            store: None,
            routing: RoutingIndex::new(),
            config,
            context_tokens: Vec::new(),
            injection_delta: None,
            boundary_residual: None,
            crystal_layer: 0,
        }
    }

    pub fn with_store(mut self, store: ApolloStore) -> Self {
        self.store = Some(store);
        self
    }

    pub fn build_routing_index(&mut self) -> Result<(), ApolloError> {
        let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
        self.routing = RoutingIndex::from_store(store);
        Ok(())
    }

    pub fn config(&self) -> &InjectionConfig {
        &self.config
    }
    pub fn has_store(&self) -> bool {
        self.store.is_some()
    }
    pub fn store(&self) -> Option<&ApolloStore> {
        self.store.as_ref()
    }
    pub fn routing(&self) -> &RoutingIndex {
        &self.routing
    }

    /// Return the top-k entries most relevant to `query_token_ids`,
    /// scoped to `candidate_windows`. Uses seed + proximity + fact-group +
    /// backfill ranking.
    pub fn retrieve_entries(
        &self,
        query_token_ids: &[u32],
        candidate_windows: &[u16],
    ) -> Result<Vec<VecInjectEntry>, ApolloError> {
        const PROXIMITY_RADIUS: u16 = 10;
        let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
        if query_token_ids.is_empty() {
            return Ok(vec![]);
        }
        let qset: std::collections::HashSet<u32> = query_token_ids.iter().copied().collect();
        let wset: std::collections::HashSet<u16> = candidate_windows.iter().copied().collect();
        let in_candidate = |e: &VecInjectEntry| wset.is_empty() || wset.contains(&e.window_id);
        let entry_key =
            |e: &VecInjectEntry| (e.window_id, e.position_in_window, e.token_id, e.fact_id);

        let seeds: Vec<&VecInjectEntry> = store
            .entries
            .iter()
            .filter(|e| in_candidate(e) && qset.contains(&e.token_id))
            .collect();

        if seeds.is_empty() {
            let mut scored: Vec<(VecInjectEntry, f32)> = store
                .entries
                .iter()
                .filter(|e| in_candidate(e))
                .map(|e| (*e, e.coefficient))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(self.config.top_k);
            return Ok(scored.into_iter().map(|(e, _)| e).collect());
        }

        let seed_facts: std::collections::HashSet<u16> = seeds.iter().map(|e| e.fact_id).collect();
        let seed_positions: std::collections::HashSet<(u16, u16)> = seeds
            .iter()
            .map(|e| (e.window_id, e.position_in_window))
            .collect();

        let mut scored: Vec<(VecInjectEntry, f32)> = Vec::new();
        let mut seen: std::collections::HashSet<(u16, u16, u32, u16)> =
            std::collections::HashSet::new();

        for e in &seeds {
            scored.push((**e, e.coefficient));
            seen.insert(entry_key(e));
        }
        for e in store.entries.iter().filter(|e| in_candidate(e)) {
            let k = entry_key(e);
            if seen.contains(&k) {
                continue;
            }
            let near = seed_positions.iter().any(|(w, p)| {
                *w == e.window_id
                    && (e.position_in_window as i32 - *p as i32).abs() <= PROXIMITY_RADIUS as i32
            });
            if near {
                scored.push((*e, e.coefficient * 1.3));
                seen.insert(k);
            }
        }
        for e in store
            .entries
            .iter()
            .filter(|e| in_candidate(e) && seed_facts.contains(&e.fact_id))
        {
            let k = entry_key(e);
            if !seen.contains(&k) {
                scored.push((*e, e.coefficient * 1.3));
                seen.insert(k);
            }
        }
        if scored.len() < self.config.top_k {
            let mut pool: Vec<&VecInjectEntry> = store
                .entries
                .iter()
                .filter(|e| in_candidate(e) && !seen.contains(&entry_key(e)))
                .collect();
            pool.sort_by(|a, b| {
                b.coefficient
                    .partial_cmp(&a.coefficient)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for e in pool.into_iter().take(self.config.top_k - scored.len()) {
                scored.push((*e, e.coefficient * 0.8));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k);
        Ok(scored.into_iter().map(|(e, _)| e).collect())
    }

    /// Build the injection delta, context, and optional boundary residual
    /// for a set of query tokens.
    /// Returns `(context_tokens, injection_delta, boundary_residual, crystal_layer)`.
    fn prepare_injection(
        &self,
        weights: &ModelWeights,
        query_ids: &[u32],
    ) -> Option<InjectionPrep> {
        let store = self.store.as_ref()?;
        let q = RoutingQuery {
            token_ids: query_ids.to_vec(),
        };
        let routed = self.routing.resolve(&q, 3);
        let top_window = *routed.first()?;

        let entries = self.retrieve_entries(query_ids, &[top_window]).ok()?;
        let window_tokens = store.window_tokens.get(top_window as usize)?;

        // Context = window_tokens ++ query_tokens (drop leading BOS if present).
        let mut context: Vec<u32> = window_tokens.clone();
        let skip = if !query_ids.is_empty() && query_ids[0] == 2 {
            1
        } else {
            0
        };
        context.extend_from_slice(&query_ids[skip..]);

        // Injection delta: sum of answer-side entry embeddings.
        let hidden = weights.hidden_size;
        let mut delta = vec![0.0f32; hidden];
        let qset: std::collections::HashSet<u32> = query_ids.iter().copied().collect();
        for e in &entries {
            if qset.contains(&e.token_id) {
                continue;
            }
            let emb = embed_tokens_pub(weights, &[e.token_id]);
            let scale = e.coefficient * self.config.inject_coefficient;
            for (i, v) in emb.row(0).iter().enumerate() {
                delta[i] += v * scale;
            }
        }

        // Boundary residual: if the store has one for this window, the compressed
        // path can skip layers 0..crystal_layer entirely.
        let boundary = store.boundaries.get(top_window as usize).cloned();
        let crystal = store.manifest.crystal_layer;

        Some((context, Array1::from(delta), boundary, crystal))
    }

    /// One-shot query: route → retrieve → inject → forward. Uses the compressed
    /// path (boundary + 4 layers) when the store has boundary residuals.
    pub fn query_greedy(&self, weights: &ModelWeights, query_ids: &[u32]) -> Option<QueryTrace> {
        let (context, delta, boundary, crystal) = self.prepare_injection(weights, query_ids)?;
        let perturb = Some((self.config.injection_layer, delta.view()));
        let raw = if let Some(ref bnd) = boundary {
            // Compressed: skip layers 0..crystal, run only crystal..34 (~4 layers)
            forward_from_layer(weights, query_ids, bnd, crystal, perturb)
        } else {
            forward_raw_logits(weights, &context, perturb)
        };
        let (top1_id, top1_logit) = raw
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i as u32, v))?;
        let q = RoutingQuery {
            token_ids: query_ids.to_vec(),
        };
        let routed = self.routing.resolve(&q, 3);
        let entries = self
            .retrieve_entries(query_ids, routed.get(..1).unwrap_or(&[]))
            .unwrap_or_default();
        Some(QueryTrace {
            routed_windows: routed,
            injected_entries: entries,
            context_tokens: context.len(),
            top1_token_id: top1_id,
            top1_logit,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

// ─── KvEngine impl ────────────────────────────────────────────────────────────

impl KvEngine for ApolloEngine {
    fn name(&self) -> &str {
        "apollo"
    }

    fn info(&self) -> EngineInfo {
        let windows = self.store.as_ref().map_or(0, |s| s.window_tokens.len());
        let entries = self.store.as_ref().map_or(0, |s| s.entries.len());
        let store_kb = self.store.as_ref().map_or(0, |s| s.total_bytes()) / 1024;
        let crystal = self.store.as_ref().map_or(0, |s| s.manifest.crystal_layer);
        let has_boundaries = self
            .store
            .as_ref()
            .is_some_and(|s| !s.boundaries.is_empty());
        let path = if has_boundaries {
            format!("compressed(layer={crystal})")
        } else {
            "uncompressed".into()
        };
        EngineInfo {
            name: "apollo".into(),
            description: format!(
                "retrieval+injection [{path}]: {windows} windows, {entries} entries, {store_kb}KB",
            ),
            backend: "cpu".into(),
            config: format!(
                "inject_layer={}, coef={}, top_k={}",
                self.config.injection_layer, self.config.inject_coefficient, self.config.top_k,
            ),
        }
    }

    /// Prefill routes token_ids, retrieves entries, builds the injection delta,
    /// and runs the forward pass.
    ///
    /// **Compressed path** (when store has boundary residuals): runs only
    /// `crystal_layer..num_layers` (~4 layers for Gemma 3 4B), ~8.5× faster.
    ///
    /// **Uncompressed path** (no boundaries): full forward over window+query tokens.
    fn prefill(&mut self, weights: &ModelWeights, token_ids: &[u32]) -> Option<Array2<f32>> {
        if self.routing.is_empty() {
            let store = self.store.as_ref()?;
            self.routing = RoutingIndex::from_store(store);
        }

        let (context, delta, boundary, crystal) = self.prepare_injection(weights, token_ids)?;
        let perturb = Some((self.config.injection_layer, delta.view()));

        let raw = if let Some(ref bnd) = boundary {
            // Compressed: boundary residual acts as position-0; skip layers 0..crystal.
            forward_from_layer(weights, token_ids, bnd, crystal, perturb)
        } else {
            forward_raw_logits(weights, &context, perturb)
        };

        // Cache decode state.
        self.context_tokens = if boundary.is_some() {
            token_ids.to_vec() // compressed: just the query
        } else {
            context
        };
        self.injection_delta = Some(delta);
        self.boundary_residual = boundary;
        self.crystal_layer = crystal;

        let last = raw.h_pre_norm.shape()[0] - 1;
        Some(raw.h_pre_norm.slice(s![last..=last, ..]).to_owned())
    }

    /// Extend by one token. Uses the boundary compressed path when available
    /// (4 layers), otherwise full 34-layer re-forward.
    fn decode_step(&mut self, weights: &ModelWeights, token_id: u32) -> Option<Array2<f32>> {
        self.context_tokens.push(token_id);
        let delta = self.injection_delta.as_ref()?;
        let perturb = Some((self.config.injection_layer, delta.view()));

        let raw = if let Some(ref bnd) = self.boundary_residual {
            // Compressed: re-run only crystal_layer..num_layers over growing query.
            forward_from_layer(
                weights,
                &self.context_tokens,
                bnd,
                self.crystal_layer,
                perturb,
            )
        } else {
            forward_raw_logits(weights, &self.context_tokens, perturb)
        };

        let last = raw.h_pre_norm.shape()[0] - 1;
        Some(raw.h_pre_norm.slice(s![last..=last, ..]).to_owned())
    }

    fn memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.total_bytes())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::apollo::store::{ArchConfig, StoreManifest};

    /// Build a minimal in-memory ApolloStore with synthetic data.
    fn mk_store(windows: usize, window_size: usize, hidden: usize) -> ApolloStore {
        let window_tokens: Vec<Vec<u32>> = (0..windows)
            .map(|w| {
                (0..window_size)
                    .map(|i| (w * window_size + i) as u32)
                    .collect()
            })
            .collect();
        let boundaries: Vec<Vec<f32>> =
            (0..windows).map(|w| vec![w as f32 * 0.1; hidden]).collect();
        let entries = vec![
            VecInjectEntry {
                token_id: 42,
                coefficient: 5.0,
                window_id: 0,
                position_in_window: 10,
                fact_id: 1,
            },
            VecInjectEntry {
                token_id: 43,
                coefficient: 3.0,
                window_id: 0,
                position_in_window: 11,
                fact_id: 1,
            },
            VecInjectEntry {
                token_id: 99,
                coefficient: 4.0,
                window_id: 1,
                position_in_window: 5,
                fact_id: 2,
            },
        ];
        ApolloStore {
            manifest: StoreManifest {
                version: 1,
                num_entries: entries.len(),
                num_windows: windows,
                num_tokens: windows * window_size,
                entries_per_window: 1,
                crystal_layer: 30,
                window_size,
                arch_config: ArchConfig::default(),
                has_residuals: true,
            },
            boundaries,
            boundary_residual: None,
            window_tokens,
            entries,
        }
    }

    fn mk_engine_with_store(windows: usize) -> ApolloEngine {
        let store = mk_store(windows, 8, 16);
        let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
        engine.build_routing_index().expect("index build failed");
        engine
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_engine_has_no_store() {
        let engine = ApolloEngine::new(InjectionConfig::default());
        assert!(!engine.has_store());
        assert!(engine.routing().is_empty());
    }

    #[test]
    fn with_store_attaches_store() {
        let store = mk_store(2, 8, 16);
        let engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
        assert!(engine.has_store());
    }

    #[test]
    fn build_routing_index_populates_index() {
        let store = mk_store(3, 8, 16);
        let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
        engine.build_routing_index().unwrap();
        assert!(!engine.routing().is_empty());
    }

    // ── EngineInfo ────────────────────────────────────────────────────────────

    #[test]
    fn info_no_store_shows_zero_windows() {
        let engine = ApolloEngine::new(InjectionConfig::default());
        let info = engine.info();
        assert_eq!(info.name, "apollo");
        assert!(info.description.contains("0 windows"));
        assert!(info.config.contains("inject_layer=30"));
    }

    #[test]
    fn info_with_store_shows_window_count() {
        let engine = mk_engine_with_store(3);
        let info = engine.info();
        assert!(
            info.description.contains("3 windows"),
            "got: {}",
            info.description
        );
        assert!(
            info.description.contains("3 entries"),
            "got: {}",
            info.description
        );
    }

    #[test]
    fn info_shows_compressed_path_when_boundaries_present() {
        let engine = mk_engine_with_store(2);
        let info = engine.info();
        assert!(
            info.description.contains("compressed(layer=30)"),
            "got: {}",
            info.description
        );
    }

    #[test]
    fn info_shows_uncompressed_path_when_no_boundaries() {
        let store = mk_store(2, 8, 16);
        // Remove boundaries
        let mut store = store;
        store.boundaries.clear();
        let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
        engine.build_routing_index().unwrap();
        assert!(engine.info().description.contains("uncompressed"));
    }

    // ── retrieve_entries ─────────────────────────────────────────────────────

    #[test]
    fn retrieve_returns_err_when_no_store() {
        let engine = ApolloEngine::new(InjectionConfig::default());
        assert!(engine.retrieve_entries(&[1], &[0]).is_err());
    }

    #[test]
    fn retrieve_empty_query_returns_empty() {
        let engine = mk_engine_with_store(2);
        let entries = engine.retrieve_entries(&[], &[0]).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn retrieve_seed_token_matched() {
        let engine = mk_engine_with_store(2);
        // token_id=42 is in window 0 with coefficient 5.0
        let entries = engine.retrieve_entries(&[42], &[0]).unwrap();
        assert!(!entries.is_empty(), "expected at least one entry");
        assert!(
            entries.iter().any(|e| e.token_id == 42),
            "seed token not in results"
        );
    }

    #[test]
    fn retrieve_proximity_neighbour_included() {
        // token 43 is at position 11 — adjacent to token 42 at position 10.
        // Querying [42] should include 43 via proximity (radius=10).
        let engine = mk_engine_with_store(2);
        let entries = engine.retrieve_entries(&[42], &[0]).unwrap();
        assert!(
            entries.iter().any(|e| e.token_id == 43),
            "adjacent entry (pos=11) not promoted via proximity"
        );
    }

    #[test]
    fn retrieve_scoped_to_candidate_windows() {
        // token 99 is only in window 1; asking for window 0 should not return it.
        let engine = mk_engine_with_store(2);
        let entries = engine.retrieve_entries(&[1], &[0]).unwrap();
        assert!(
            !entries.iter().any(|e| e.token_id == 99),
            "entry from window 1 leaked into window 0 result"
        );
    }

    #[test]
    fn retrieve_backfills_to_top_k() {
        // Query with no matching seeds → backfill to top_k by coefficient.
        let engine = mk_engine_with_store(2);
        let cfg = engine.config();
        let entries = engine.retrieve_entries(&[9999], &[0]).unwrap();
        // Should get up to top_k entries even with no seed match.
        assert!(entries.len() <= cfg.top_k);
    }

    // ── memory_bytes ─────────────────────────────────────────────────────────

    #[test]
    fn memory_bytes_zero_without_store() {
        let engine = ApolloEngine::new(InjectionConfig::default());
        assert_eq!(engine.memory_bytes(), 0);
    }

    #[test]
    fn memory_bytes_nonzero_with_store() {
        let engine = mk_engine_with_store(3);
        assert!(engine.memory_bytes() > 0);
    }

    // ── store() getter ───────────────────────────────────────────────────────

    #[test]
    fn store_getter_none_until_attached() {
        let engine = ApolloEngine::new(InjectionConfig::default());
        assert!(engine.store().is_none());
        let engine = engine.with_store(mk_store(2, 4, 8));
        assert!(engine.store().is_some());
    }

    // ── KvEngine name() ──────────────────────────────────────────────────────

    #[test]
    fn name_returns_apollo() {
        let engine = ApolloEngine::new(InjectionConfig::default());
        assert_eq!(engine.name(), "apollo");
    }

    // ── KvEngine prefill / decode_step (compressed path) ────────────────────
    //
    // These exercise prepare_injection + the compressed forward through
    // synthetic test_utils weights. The synthetic model has 2 layers and
    // hidden=16, so we build a store with hidden=16 and inject at layer 1.

    fn mk_apollo_for_synthetic_weights(weights: &larql_inference::ModelWeights) -> ApolloEngine {
        // Synthetic test weights have vocab=32; clamp every token_id used in
        // the store to be < 32 so embed_tokens_pub doesn't panic.
        let store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
        let cfg = InjectionConfig {
            injection_layer: 1, // 2-layer model: inject before final layer
            inject_coefficient: 2.0,
            top_k: 4,
        };
        let mut engine = ApolloEngine::new(cfg).with_store(store);
        engine.build_routing_index().unwrap();
        engine
    }

    /// Variant of `mk_store` whose token IDs (window tokens + entries) are all
    /// strictly less than `vocab` — required when the engine forwards through
    /// real synthetic weights that embed those tokens.
    fn mk_store_in_vocab(
        windows: usize,
        window_size: usize,
        hidden: usize,
        vocab: usize,
    ) -> ApolloStore {
        let v = vocab.max(2) as u32;
        let window_tokens: Vec<Vec<u32>> = (0..windows)
            .map(|w| {
                (0..window_size)
                    .map(|i| ((w * window_size + i) as u32) % v)
                    .collect()
            })
            .collect();
        let boundaries: Vec<Vec<f32>> =
            (0..windows).map(|w| vec![w as f32 * 0.1; hidden]).collect();
        let entries = vec![
            VecInjectEntry {
                token_id: 0 % v,
                coefficient: 5.0,
                window_id: 0,
                position_in_window: 0,
                fact_id: 1,
            },
            VecInjectEntry {
                token_id: 1 % v,
                coefficient: 3.0,
                window_id: 0,
                position_in_window: 1,
                fact_id: 1,
            },
            VecInjectEntry {
                token_id: 2 % v,
                coefficient: 4.0,
                window_id: 1,
                position_in_window: 0,
                fact_id: 2,
            },
        ];
        ApolloStore {
            manifest: StoreManifest {
                version: 1,
                num_entries: entries.len(),
                num_windows: windows,
                num_tokens: windows * window_size,
                entries_per_window: 1,
                crystal_layer: 1,
                window_size,
                arch_config: ArchConfig::default(),
                has_residuals: true,
            },
            boundaries,
            boundary_residual: None,
            window_tokens,
            entries,
        }
    }

    #[test]
    fn prefill_compressed_returns_hidden_state() {
        let weights = larql_inference::test_utils::make_test_weights();
        let mut engine = mk_apollo_for_synthetic_weights(&weights);
        // Use one of the window tokens so routing succeeds.
        let h = engine.prefill(&weights, &[0u32, 1u32]).expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_step_after_compressed_prefill_grows_context() {
        let weights = larql_inference::test_utils::make_test_weights();
        let mut engine = mk_apollo_for_synthetic_weights(&weights);
        engine.prefill(&weights, &[0u32]).expect("prefill");
        let h = engine.decode_step(&weights, 1).expect("decode_step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn prefill_uncompressed_path_when_no_boundaries() {
        let weights = larql_inference::test_utils::make_test_weights();
        let mut store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
        store.boundaries.clear();
        let mut engine = ApolloEngine::new(InjectionConfig {
            injection_layer: 1,
            inject_coefficient: 1.0,
            top_k: 4,
        })
        .with_store(store);
        engine.build_routing_index().unwrap();
        // Token 0 is in window 0 → routing finds it; uncompressed full forward runs.
        let h = engine
            .prefill(&weights, &[0u32, 1u32])
            .expect("prefill uncompressed");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn prefill_returns_none_without_routing_or_store() {
        let weights = larql_inference::test_utils::make_test_weights();
        // No store → routing won't initialize from store.
        let mut engine = ApolloEngine::new(InjectionConfig::default());
        assert!(engine.prefill(&weights, &[0u32]).is_none());
    }

    // ── query_greedy ─────────────────────────────────────────────────────────

    #[test]
    fn query_greedy_returns_trace_with_top1_token() {
        let weights = larql_inference::test_utils::make_test_weights();
        let engine = mk_apollo_for_synthetic_weights(&weights);
        let trace = engine.query_greedy(&weights, &[0u32, 1u32]).expect("trace");
        assert!(!trace.routed_windows.is_empty());
        assert!(trace.context_tokens > 0);
        // top1 logit is finite.
        assert!(trace.top1_logit.is_finite());
    }

    #[test]
    fn query_greedy_returns_none_without_routing() {
        let weights = larql_inference::test_utils::make_test_weights();
        let engine = ApolloEngine::new(InjectionConfig::default());
        assert!(engine.query_greedy(&weights, &[0u32]).is_none());
    }
}
