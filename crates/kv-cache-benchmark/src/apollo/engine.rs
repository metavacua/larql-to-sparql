//! Top-level Apollo engine — combines routing, replay, and vec_inject.
//!
//! Current scope: MVP end-to-end pipeline. Produces logits/tokens for a
//! query by routing to the best window, retrieving relevant entries, and
//! running one forward pass over (window tokens + query tokens) with the
//! retrieved entries injected as a combined residual-stream perturbation
//! at `injection_layer`. This matches the essential retrieval + amplify
//! mechanism from Apollo v12 at a coarser granularity than the Python
//! demo's per-position injection.
//!
//! Known simplifications vs the full Apollo pipeline:
//! - Injection happens at the *last* token position only (via
//!   `forward_raw_logits`'s `perturb` parameter). Python injects at each
//!   entry's `position_in_window`.
//! - No boundary-residual replay — we forward the window's *tokens*
//!   directly, so this path doesn't exercise the 2.8 MB compression claim.
//!   That requires a `prefill_to_layer(initial_residual=...)` primitive
//!   that larql-inference doesn't expose yet.
//! - Routing uses a simple tf-idf-lite over token IDs, no stemming or
//!   stopword filtering.
//!
//! These are individually small follow-ups. The current implementation
//! validates the retrieval+injection loop end-to-end.

use thiserror::Error;

use super::entry::{InjectionConfig, VecInjectEntry};
use super::routing::{RoutingIndex, RoutingQuery};
use super::store::ApolloStore;

#[derive(Debug, Error)]
pub enum ApolloError {
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("store not loaded")]
    StoreNotLoaded,
    #[error("routing index not built — call build_routing_index() first")]
    RoutingNotBuilt,
    #[error("invalid window id: {0}")]
    InvalidWindowId(u16),
    #[error("tokenizer error: {0}")]
    Tokenize(String),
    #[error("forward pass failed")]
    Forward,
    #[error("no windows matched query (routing returned empty)")]
    NoMatch,
}

/// Summary of a single query answered by the engine — useful for diagnostics.
#[derive(Debug, Clone)]
pub struct QueryTrace {
    /// Window IDs the router picked, in ranked order.
    pub routed_windows: Vec<u16>,
    /// Entries the engine chose to inject (top-k by score).
    pub injected_entries: Vec<VecInjectEntry>,
    /// Number of context tokens sent through the forward pass
    /// (window tokens + query tokens).
    pub context_tokens: usize,
    /// Greedy top-1 token ID predicted by the forward pass.
    pub top1_token_id: u32,
    /// Top logit value.
    pub top1_logit: f32,
}

/// Summary of an iterative decode — N forward passes under injection.
#[derive(Debug, Clone)]
pub struct GenerationTrace {
    pub routed_windows: Vec<u16>,
    pub injected_entries: Vec<VecInjectEntry>,
    /// Tokens generated, in order. Does NOT include the query.
    pub generated_token_ids: Vec<u32>,
    /// Per-step top-1 logit (useful for seeing when the model loses
    /// confidence). Same length as `generated_token_ids`.
    pub per_step_logits: Vec<f32>,
    /// Number of tokens in the initial prompt
    /// (boundary virtual + query). Each subsequent step appends 1.
    pub initial_context_tokens: usize,
    /// Whether decode stopped early on EOS.
    pub stopped_on_eos: bool,
}

pub struct ApolloEngine {
    store: Option<ApolloStore>,
    routing: RoutingIndex,
    config: InjectionConfig,
}

impl ApolloEngine {
    pub fn new(config: InjectionConfig) -> Self {
        Self {
            store: None,
            routing: RoutingIndex::new(),
            config,
        }
    }

    pub fn with_store(mut self, store: ApolloStore) -> Self {
        self.store = Some(store);
        self
    }

    /// Build the routing index from the loaded store. Call once after
    /// `with_store`; subsequent queries hit the index.
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

    /// Return the top-k entries most relevant to the query tokens, scoped
    /// to the given set of candidate windows.
    ///
    /// Empirically, Apollo's store has one entry per `fact_id`, so
    /// fact-group promotion is a no-op. We instead use a **positional-
    /// proximity + high-coefficient backfill** heuristic:
    ///
    /// 1. **Seed**: entries whose `token_id ∈ query` AND `window ∈ candidate`.
    ///    These matched verbatim.
    /// 2. **Neighbour promotion**: for each seed, include entries in the
    ///    same window whose `position_in_window` is within
    ///    `PROXIMITY_RADIUS` of the seed. Answer tokens tend to live near
    ///    the question tokens in the source text.
    /// 3. **Fact-group promotion**: if multiple entries share a `fact_id`
    ///    with any seed, include them all (no-op on current Apollo store,
    ///    but correct for stores with multi-token facts).
    /// 4. **Backfill**: if still below `top_k`, add highest-coefficient
    ///    entries from candidate windows — the store's own notion of
    ///    "important tokens" for this context.
    /// 5. **Score + rank**: seed = `coef`, neighbour = `coef × 1.3`,
    ///    backfill = `coef × 0.8`. Sort desc, truncate to `top_k`.
    pub fn retrieve_entries(
        &self,
        query_token_ids: &[u32],
        candidate_windows: &[u16],
    ) -> Result<Vec<VecInjectEntry>, ApolloError> {
        /// Positional proximity radius within a window.
        const PROXIMITY_RADIUS: u16 = 10;

        let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
        if query_token_ids.is_empty() {
            return Ok(vec![]);
        }
        let qset: std::collections::HashSet<u32> = query_token_ids.iter().copied().collect();
        let wset: std::collections::HashSet<u16> = candidate_windows.iter().copied().collect();
        let in_candidate = |e: &VecInjectEntry| wset.is_empty() || wset.contains(&e.window_id);

        // Step 1: seed entries.
        let seeds: Vec<&VecInjectEntry> = store
            .entries
            .iter()
            .filter(|e| in_candidate(e) && qset.contains(&e.token_id))
            .collect();

        if seeds.is_empty() {
            // No query overlap — return top-k by raw coefficient in candidates.
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

        // Step 2 + 3: build scored set.
        //   key: (window_id, position_in_window, token_id, fact_id) uniqueness.
        //   We track which entries we've already scored so we don't double-count.
        let seed_facts: std::collections::HashSet<u16> =
            seeds.iter().map(|e| e.fact_id).collect();
        let seed_positions: std::collections::HashSet<(u16, u16)> = seeds
            .iter()
            .map(|e| (e.window_id, e.position_in_window))
            .collect();

        // Entry-uniqueness key used to dedupe across the three passes.
        let entry_key = |e: &VecInjectEntry| {
            (e.window_id, e.position_in_window, e.token_id, e.fact_id)
        };

        let mut scored: Vec<(VecInjectEntry, f32)> = Vec::new();
        let mut seen: std::collections::HashSet<(u16, u16, u32, u16)> =
            std::collections::HashSet::new();

        // Seed entries at base coefficient.
        for e in &seeds {
            scored.push((**e, e.coefficient));
            seen.insert(entry_key(e));
        }

        // Neighbours within PROXIMITY_RADIUS of any seed, in the same window.
        for e in store.entries.iter().filter(|e| in_candidate(e)) {
            let k = entry_key(e);
            if seen.contains(&k) {
                continue;
            }
            let near_seed = seed_positions.iter().any(|(w, p)| {
                *w == e.window_id
                    && (e.position_in_window as i32 - *p as i32).abs()
                        <= PROXIMITY_RADIUS as i32
            });
            if near_seed {
                scored.push((*e, e.coefficient * 1.3));
                seen.insert(k);
            }
        }

        // Fact-group promotion (no-op on current Apollo store but correct
        // for multi-token-fact stores).
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

        // Step 4: backfill to top_k with highest-coefficient entries.
        if scored.len() < self.config.top_k {
            let mut pool: Vec<&VecInjectEntry> = store
                .entries
                .iter()
                .filter(|e| in_candidate(e) && !seen.contains(&entry_key(e)))
                .collect();
            pool.sort_by(|a, b| {
                b.coefficient.partial_cmp(&a.coefficient).unwrap_or(std::cmp::Ordering::Equal)
            });
            let need = self.config.top_k - scored.len();
            for e in pool.into_iter().take(need) {
                scored.push((*e, e.coefficient * 0.8));
                seen.insert(entry_key(e));
            }
        }

        // Step 5: rank and truncate.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k);
        Ok(scored.into_iter().map(|(e, _)| e).collect())
    }
}

// ── Forward-pass integration (feature-gated since it pulls in larql-inference) ──

#[cfg(feature = "real-model")]
mod forward_integration {
    use super::*;
    use larql_inference::forward::{
        embed_tokens_pub, forward_raw_logits, forward_raw_logits_with_prefix,
    };
    use larql_inference::model::ModelWeights;
    use tokenizers::Tokenizer;

    impl ApolloEngine {
        /// Run the full query pipeline and return a trace of what happened.
        ///
        /// 1. Tokenize the query.
        /// 2. Route: top-k windows by tf-idf-lite on query token IDs.
        /// 3. Retrieve: top-k entries from those windows whose token_id
        ///    overlaps the query.
        /// 4. Forward pass over `(selected_window_tokens ++ query_tokens)`.
        ///    At `injection_layer`, add the combined scaled embedding of
        ///    retrieved entries to the last position's residual.
        /// 5. Greedy top-1 decode.
        pub fn query_greedy(
            &self,
            weights: &ModelWeights,
            tokenizer: &Tokenizer,
            query_str: &str,
            top_windows_k: usize,
        ) -> Result<QueryTrace, ApolloError> {
            let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
            if self.routing.is_empty() {
                return Err(ApolloError::RoutingNotBuilt);
            }

            // Tokenize query.
            let enc = tokenizer
                .encode(query_str, true)
                .map_err(|e| ApolloError::Tokenize(e.to_string()))?;
            let query_ids: Vec<u32> = enc.get_ids().to_vec();
            if query_ids.is_empty() {
                return Err(ApolloError::Tokenize("empty query".into()));
            }

            // Route to best windows.
            let q = RoutingQuery {
                token_ids: query_ids.clone(),
            };
            let routed = self.routing.resolve(&q, top_windows_k);
            if routed.is_empty() {
                return Err(ApolloError::NoMatch);
            }

            // Retrieve entries scoped to the TOP routed window only — pulling
            // from all routed windows injects noise from near-miss contexts
            // (e.g. a "\n" or " the" with high coefficient in a window that
            // the router ranked low but still returned). Restricting to
            // routed[0] keeps the injection focused on the best window's
            // answer tokens.
            let retrieval_scope = &routed[..1];
            let entries = self.retrieve_entries(&query_ids, retrieval_scope)?;

            // Build the context: tokens of the top window followed by the
            // query tokens. Skip the query's BOS if the window's first token
            // is already a special token (pragmatic MVP choice).
            let top_window = routed[0];
            let window_tokens = store
                .window_tokens
                .get(top_window as usize)
                .ok_or(ApolloError::InvalidWindowId(top_window))?;
            let mut context: Vec<u32> = Vec::with_capacity(window_tokens.len() + query_ids.len());
            context.extend_from_slice(window_tokens);
            // Drop BOS from query if present (duplicate with window's leading
            // special). Heuristic: BOS for Gemma is token 2.
            let bos_id = tokenizer.token_to_id("<bos>").unwrap_or(u32::MAX);
            let skip = if !query_ids.is_empty() && query_ids[0] == bos_id {
                1
            } else {
                0
            };
            context.extend_from_slice(&query_ids[skip..]);

            // Build combined injection delta. CRITICAL: entries whose token_id
            // matches a query token are the "question side" — injecting their
            // embedding just echoes the query back. Those are kept in the
            // retrieval set for context, but EXCLUDED from the injection delta.
            // The delta is the sum of "answer side" entries only.
            let hidden = weights.hidden_size;
            let mut delta = vec![0.0f32; hidden];
            let global_coef = self.config.inject_coefficient;
            let qset: std::collections::HashSet<u32> = query_ids.iter().copied().collect();
            let mut answer_entries = 0usize;
            for e in &entries {
                if qset.contains(&e.token_id) {
                    continue; // question-side echo — skip
                }
                let emb = embed_tokens_pub(weights, &[e.token_id]);
                let scale = e.coefficient * global_coef;
                for (i, v) in emb.row(0).iter().enumerate() {
                    delta[i] += v * scale;
                }
                answer_entries += 1;
            }

            // Forward with perturbation — only if we actually have answer-side
            // entries to inject. If all retrieved entries were question-side
            // echoes, we forward with no perturbation rather than a zero delta.
            let delta_arr = ndarray::Array1::from(delta);
            let perturb = if answer_entries == 0 {
                None
            } else {
                Some((self.config.injection_layer, delta_arr.view()))
            };
            let raw = forward_raw_logits(weights, &context, perturb);

            let (top1_idx, top1_logit) = raw
                .logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i as u32, *v))
                .ok_or(ApolloError::Forward)?;

            Ok(QueryTrace {
                routed_windows: routed,
                injected_entries: entries,
                context_tokens: context.len(),
                top1_token_id: top1_idx,
                top1_logit,
            })
        }

        /// Compressed-path query: instead of forwarding the full window tokens
        /// as context, feed just [boundary_as_virtual_pos_0, query_tokens]
        /// through the model. The 10 KB boundary vector is the compressed
        /// state that replaces 512 tokens × hidden-size × 2 bytes of KV.
        ///
        /// This is the path that actually exercises Apollo's ~20,000× store
        /// compression claim at inference time — the previous `query_greedy`
        /// reconstructs context from the tokens themselves, which works but
        /// doesn't test the boundary-compression mechanism.
        ///
        /// Correctness trade-off: single-vector boundary at crystal_layer is
        /// lossy vs joint forward (variant ii, F1' measured cos ≈ 0.965);
        /// `vec_inject` amplification at L30 is what recovers task accuracy.
        /// Expect weaker prediction quality vs the non-compressed path on
        /// the same query — this is measuring the compressed path, not
        /// trying to beat it.
        pub fn query_greedy_compressed(
            &self,
            weights: &ModelWeights,
            tokenizer: &Tokenizer,
            query_str: &str,
            top_windows_k: usize,
        ) -> Result<QueryTrace, ApolloError> {
            let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
            if self.routing.is_empty() {
                return Err(ApolloError::RoutingNotBuilt);
            }

            let enc = tokenizer
                .encode(query_str, true)
                .map_err(|e| ApolloError::Tokenize(e.to_string()))?;
            let query_ids: Vec<u32> = enc.get_ids().to_vec();
            if query_ids.is_empty() {
                return Err(ApolloError::Tokenize("empty query".into()));
            }

            let q = RoutingQuery {
                token_ids: query_ids.clone(),
            };
            let routed = self.routing.resolve(&q, top_windows_k);
            if routed.is_empty() {
                return Err(ApolloError::NoMatch);
            }

            let top_window = routed[0];
            let boundary = store
                .boundaries
                .get(top_window as usize)
                .ok_or(ApolloError::InvalidWindowId(top_window))?;
            // Sanity-check boundary dim matches the model's hidden size.
            if boundary.len() != weights.hidden_size {
                return Err(ApolloError::Forward);
            }

            // Retrieve entries scoped to the top window (same ranking as the
            // non-compressed path).
            let entries = self.retrieve_entries(&query_ids, &[top_window])?;

            // Build the injection delta from answer-side entries only (same
            // mechanism as the uncompressed path).
            let hidden = weights.hidden_size;
            let mut delta = vec![0.0f32; hidden];
            let global_coef = self.config.inject_coefficient;
            let qset: std::collections::HashSet<u32> =
                query_ids.iter().copied().collect();
            let mut answer_entries = 0usize;
            for e in &entries {
                if qset.contains(&e.token_id) {
                    continue;
                }
                let emb = embed_tokens_pub(weights, &[e.token_id]);
                let scale = e.coefficient * global_coef;
                for (i, v) in emb.row(0).iter().enumerate() {
                    delta[i] += v * scale;
                }
                answer_entries += 1;
            }

            let delta_arr = ndarray::Array1::from(delta);
            let perturb = if answer_entries == 0 {
                None
            } else {
                Some((self.config.injection_layer, delta_arr.view()))
            };

            // Forward with boundary residual as prefix. Context = 1 virtual
            // boundary + query tokens, typically ~20 tokens total — vs ~520
            // tokens in the uncompressed path.
            let raw = forward_raw_logits_with_prefix(
                weights,
                &query_ids,
                Some(boundary.as_slice()),
                perturb,
            );

            let (top1_idx, top1_logit) = raw
                .logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i as u32, *v))
                .ok_or(ApolloError::Forward)?;

            Ok(QueryTrace {
                routed_windows: routed,
                injected_entries: entries,
                // +1 for the virtual boundary row prepended by the prefix path
                context_tokens: query_ids.len() + 1,
                top1_token_id: top1_idx,
                top1_logit,
            })
        }

        /// Iterative greedy decode over the compressed path.
        ///
        /// Generates up to `max_new_tokens` tokens by repeatedly running
        /// `forward_raw_logits_with_prefix` with the stored boundary
        /// residual as virtual position-0 and the growing context
        /// (query + already-generated tokens). The same injection delta
        /// is applied at every step, so the amplification is continuous.
        ///
        /// Stops early on EOS. Routing + retrieval happen once up front;
        /// each decode step re-runs the full forward (no KV-cache reuse
        /// yet — that's a follow-up optimisation).
        ///
        /// Cost: `max_new_tokens × forward_cost(query + n_tokens)`.
        /// For Gemma 3 4B CPU this is ~1.5 s × N, so ≤ 30 tokens feasible
        /// for demos, not production.
        pub fn query_generate_compressed(
            &self,
            weights: &ModelWeights,
            tokenizer: &Tokenizer,
            query_str: &str,
            max_new_tokens: usize,
            top_windows_k: usize,
        ) -> Result<GenerationTrace, ApolloError> {
            let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
            if self.routing.is_empty() {
                return Err(ApolloError::RoutingNotBuilt);
            }

            let enc = tokenizer
                .encode(query_str, true)
                .map_err(|e| ApolloError::Tokenize(e.to_string()))?;
            let query_ids: Vec<u32> = enc.get_ids().to_vec();
            if query_ids.is_empty() {
                return Err(ApolloError::Tokenize("empty query".into()));
            }

            let q = RoutingQuery {
                token_ids: query_ids.clone(),
            };
            let routed = self.routing.resolve(&q, top_windows_k);
            if routed.is_empty() {
                return Err(ApolloError::NoMatch);
            }

            let top_window = routed[0];
            let boundary = store
                .boundaries
                .get(top_window as usize)
                .ok_or(ApolloError::InvalidWindowId(top_window))?;
            if boundary.len() != weights.hidden_size {
                return Err(ApolloError::Forward);
            }

            let entries = self.retrieve_entries(&query_ids, &[top_window])?;

            // Build injection delta once (same for every decode step).
            let hidden = weights.hidden_size;
            let mut delta = vec![0.0f32; hidden];
            let global_coef = self.config.inject_coefficient;
            let qset: std::collections::HashSet<u32> =
                query_ids.iter().copied().collect();
            let mut answer_entries = 0usize;
            for e in &entries {
                if qset.contains(&e.token_id) {
                    continue;
                }
                let emb = embed_tokens_pub(weights, &[e.token_id]);
                let scale = e.coefficient * global_coef;
                for (i, v) in emb.row(0).iter().enumerate() {
                    delta[i] += v * scale;
                }
                answer_entries += 1;
            }
            let delta_arr = ndarray::Array1::from(delta);
            let perturb_spec: Option<(usize, ndarray::Array1<f32>)> = if answer_entries == 0 {
                None
            } else {
                Some((self.config.injection_layer, delta_arr))
            };

            let eos_id = tokenizer.token_to_id("<eos>").unwrap_or(u32::MAX);
            // Gemma 3 also emits <end_of_turn> as a stop signal — treat it
            // the same way so generation halts naturally.
            let turn_eos_id = tokenizer.token_to_id("<end_of_turn>").unwrap_or(u32::MAX);

            let mut context = query_ids.clone();
            let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
            let mut logits_trace: Vec<f32> = Vec::with_capacity(max_new_tokens);
            let mut stopped_on_eos = false;

            for step in 0..max_new_tokens {
                // Inject only on the first step. Static injection at every
                // step locks the model into the amplified tokens (observed:
                // " John" → "oyle" → "oyle" → "oyle"...). Apollo's demo
                // treats injection as a one-shot correction — once the fact
                // is in the residual stream the model should continue
                // organically from the boundary + emitted tokens.
                let perturb_view = if step == 0 {
                    perturb_spec
                        .as_ref()
                        .map(|(layer, arr)| (*layer, arr.view()))
                } else {
                    None
                };
                let raw = forward_raw_logits_with_prefix(
                    weights,
                    &context,
                    Some(boundary.as_slice()),
                    perturb_view,
                );

                let (top1_idx, top1_logit) = raw
                    .logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, v)| (i as u32, *v))
                    .ok_or(ApolloError::Forward)?;

                logits_trace.push(top1_logit);
                if top1_idx == eos_id || top1_idx == turn_eos_id {
                    stopped_on_eos = true;
                    break;
                }
                generated.push(top1_idx);
                context.push(top1_idx);
            }

            Ok(GenerationTrace {
                routed_windows: routed,
                injected_entries: entries,
                generated_token_ids: generated,
                per_step_logits: logits_trace,
                initial_context_tokens: query_ids.len() + 1,
                stopped_on_eos,
            })
        }

        /// Iterative greedy decode over the uncompressed path — forwards
        /// the full window tokens + query (+ growing generation) with
        /// one-shot injection at step 0. This is the "Markov reconstruction"
        /// equivalent: the model has the actual window text to ground on,
        /// and injection acts as a retrieval hint rather than load-bearing.
        ///
        /// Trade-off vs `query_generate_compressed`: bigger context per
        /// step (~520 tokens vs ~9) so slower, but the model has more
        /// content to produce grounded continuation text.
        pub fn query_generate_uncompressed(
            &self,
            weights: &ModelWeights,
            tokenizer: &Tokenizer,
            query_str: &str,
            max_new_tokens: usize,
            top_windows_k: usize,
        ) -> Result<GenerationTrace, ApolloError> {
            let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
            if self.routing.is_empty() {
                return Err(ApolloError::RoutingNotBuilt);
            }

            let enc = tokenizer
                .encode(query_str, true)
                .map_err(|e| ApolloError::Tokenize(e.to_string()))?;
            let query_ids: Vec<u32> = enc.get_ids().to_vec();
            if query_ids.is_empty() {
                return Err(ApolloError::Tokenize("empty query".into()));
            }

            let q = RoutingQuery {
                token_ids: query_ids.clone(),
            };
            let routed = self.routing.resolve(&q, top_windows_k);
            if routed.is_empty() {
                return Err(ApolloError::NoMatch);
            }

            let top_window = routed[0];
            let window_tokens = store
                .window_tokens
                .get(top_window as usize)
                .ok_or(ApolloError::InvalidWindowId(top_window))?;

            let entries = self.retrieve_entries(&query_ids, &[top_window])?;

            // Build injection delta from answer-side entries (same as
            // compressed path).
            let hidden = weights.hidden_size;
            let mut delta = vec![0.0f32; hidden];
            let global_coef = self.config.inject_coefficient;
            let qset: std::collections::HashSet<u32> =
                query_ids.iter().copied().collect();
            let mut answer_entries = 0usize;
            for e in &entries {
                if qset.contains(&e.token_id) {
                    continue;
                }
                let emb = embed_tokens_pub(weights, &[e.token_id]);
                let scale = e.coefficient * global_coef;
                for (i, v) in emb.row(0).iter().enumerate() {
                    delta[i] += v * scale;
                }
                answer_entries += 1;
            }
            let delta_arr = ndarray::Array1::from(delta);
            let perturb_spec: Option<(usize, ndarray::Array1<f32>)> = if answer_entries == 0 {
                None
            } else {
                Some((self.config.injection_layer, delta_arr))
            };

            let eos_id = tokenizer.token_to_id("<eos>").unwrap_or(u32::MAX);
            let turn_eos_id = tokenizer.token_to_id("<end_of_turn>").unwrap_or(u32::MAX);

            // Build initial context: window tokens + query (drop query BOS).
            let bos_id = tokenizer.token_to_id("<bos>").unwrap_or(u32::MAX);
            let skip = if !query_ids.is_empty() && query_ids[0] == bos_id {
                1
            } else {
                0
            };
            let initial_ctx_len = window_tokens.len() + (query_ids.len() - skip);
            let mut context: Vec<u32> =
                Vec::with_capacity(initial_ctx_len + max_new_tokens);
            context.extend_from_slice(window_tokens);
            context.extend_from_slice(&query_ids[skip..]);

            let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
            let mut logits_trace: Vec<f32> = Vec::with_capacity(max_new_tokens);
            let mut stopped_on_eos = false;

            for step in 0..max_new_tokens {
                let perturb_view = if step == 0 {
                    perturb_spec
                        .as_ref()
                        .map(|(layer, arr)| (*layer, arr.view()))
                } else {
                    None
                };
                let raw = forward_raw_logits(weights, &context, perturb_view);

                let (top1_idx, top1_logit) = raw
                    .logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, v)| (i as u32, *v))
                    .ok_or(ApolloError::Forward)?;

                logits_trace.push(top1_logit);
                if top1_idx == eos_id || top1_idx == turn_eos_id {
                    stopped_on_eos = true;
                    break;
                }
                generated.push(top1_idx);
                context.push(top1_idx);
            }

            Ok(GenerationTrace {
                routed_windows: routed,
                injected_entries: entries,
                generated_token_ids: generated,
                per_step_logits: logits_trace,
                initial_context_tokens: initial_ctx_len,
                stopped_on_eos,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_engine_has_no_store() {
        let eng = ApolloEngine::new(InjectionConfig::default());
        assert!(!eng.has_store());
        assert_eq!(eng.config().injection_layer, 30);
    }

    #[test]
    fn retrieve_requires_store() {
        let eng = ApolloEngine::new(InjectionConfig::default());
        let err = eng.retrieve_entries(&[1, 2, 3], &[]).unwrap_err();
        assert!(matches!(err, ApolloError::StoreNotLoaded));
    }

    #[test]
    fn build_routing_requires_store() {
        let mut eng = ApolloEngine::new(InjectionConfig::default());
        let err = eng.build_routing_index().unwrap_err();
        assert!(matches!(err, ApolloError::StoreNotLoaded));
    }
}
