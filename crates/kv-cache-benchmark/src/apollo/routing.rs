//! Keyword-driven routing index.
//!
//! Given a query string, returns a ranked list of window IDs likely to
//! contain relevant facts. Apollo's routing is tf-idf over tokenized
//! keywords; ~120 KB on disk for the Apollo 11 corpus.
//!
//! **Status**: scaffold. `resolve` is unimplemented.

//! Token-ID routing index.
//!
//! Given a set of *query token IDs*, ranks windows by how many of those IDs
//! appear in the window's archived tokens. This is the simplest possible
//! routing: count-based overlap in token-ID space, no stemming or idf. It's
//! the MVP that replaces Python's tf-idf layer for the initial Rust port.
//!
//! Production version would: tokenize with stemming, filter stopwords, apply
//! per-term idf weighting, and consider fact_id grouping. That's follow-up
//! work. Reference: `chuk-mlx/.../research/_stopwords.py`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::store::ApolloStore;

/// Inverted index: token_id → list of (window_id, term_frequency) pairs.
/// term_frequency = number of occurrences of that token in that window.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RoutingIndex {
    pub index: HashMap<u32, Vec<(u16, u32)>>,
    /// Total number of windows indexed.
    pub num_windows: usize,
}

/// A parsed query ready for routing.
pub struct RoutingQuery {
    pub token_ids: Vec<u32>,
}

impl RoutingIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build an inverted index from the store's `window_tokens`.
    /// O(total_tokens); ~90K entries on Apollo 11.
    pub fn from_store(store: &ApolloStore) -> Self {
        let mut index: HashMap<u32, HashMap<u16, u32>> = HashMap::new();
        for (window_id, tokens) in store.window_tokens.iter().enumerate() {
            let wid = window_id as u16;
            for &tok in tokens {
                *index.entry(tok).or_default().entry(wid).or_insert(0) += 1;
            }
        }
        let compacted: HashMap<u32, Vec<(u16, u32)>> = index
            .into_iter()
            .map(|(tok, wf)| (tok, wf.into_iter().collect()))
            .collect();
        Self {
            index: compacted,
            num_windows: store.window_tokens.len(),
        }
    }

    /// Return the top-k window IDs most relevant to the query, ranked by
    /// sum of (term_frequency × log(N / df + 1)) — simple tf-idf lite.
    pub fn resolve(&self, query: &RoutingQuery, top_k: usize) -> Vec<u16> {
        if self.num_windows == 0 || query.token_ids.is_empty() {
            return vec![];
        }
        let n = self.num_windows as f64;
        let mut scores: HashMap<u16, f64> = HashMap::new();
        for &tok in &query.token_ids {
            let Some(postings) = self.index.get(&tok) else {
                continue;
            };
            let df = postings.len() as f64;
            // Skip terms that appear in every window — no discrimination value.
            if df >= n {
                continue;
            }
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(wid, tf) in postings {
                *scores.entry(wid).or_insert(0.0) += (tf as f64) * idf;
            }
        }
        let mut ranked: Vec<(u16, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.into_iter().take(top_k).map(|(w, _)| w).collect()
    }

    /// Total bytes used by the serialized index.
    pub fn total_bytes(&self) -> usize {
        self.index
            .iter()
            .map(|(_k, v)| 4 + v.len() * std::mem::size_of::<(u16, u32)>())
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apollo::store::{ArchConfig, StoreManifest};

    fn mk_store(per_window_tokens: Vec<Vec<u32>>) -> ApolloStore {
        ApolloStore {
            manifest: StoreManifest {
                version: 1,
                num_entries: 0,
                num_windows: per_window_tokens.len(),
                num_tokens: per_window_tokens.iter().map(|w| w.len()).sum(),
                entries_per_window: 0,
                crystal_layer: 0,
                window_size: 0,
                arch_config: ArchConfig::default(),
                has_residuals: false,
            },
            boundaries: vec![],
            boundary_residual: None,
            window_tokens: per_window_tokens,
            entries: vec![],
        }
    }

    #[test]
    fn empty_index_is_zero_bytes() {
        let r = RoutingIndex::new();
        assert!(r.is_empty());
        assert_eq!(r.total_bytes(), 0);
    }

    #[test]
    fn resolve_ranks_matching_windows_first() {
        // window 0 contains token 42 twice, window 1 contains it once, window
        // 2 doesn't. Query on 42 should rank 0 > 1 > (2 dropped).
        let store = mk_store(vec![
            vec![1, 42, 3, 42, 5],
            vec![42, 7, 8],
            vec![9, 10, 11],
        ]);
        let idx = RoutingIndex::from_store(&store);
        let q = RoutingQuery {
            token_ids: vec![42],
        };
        let res = idx.resolve(&q, 3);
        assert_eq!(res, vec![0, 1]);
    }

    #[test]
    fn resolve_ignores_ubiquitous_terms() {
        // Token 99 appears in every window — df == N, so it's skipped.
        // Token 7 only in window 1, so query {99, 7} should pick window 1.
        let store = mk_store(vec![
            vec![99, 1, 2],
            vec![99, 7, 3],
            vec![99, 4, 5],
        ]);
        let idx = RoutingIndex::from_store(&store);
        let q = RoutingQuery {
            token_ids: vec![99, 7],
        };
        let res = idx.resolve(&q, 2);
        assert_eq!(res[0], 1);
    }

    #[test]
    fn resolve_empty_query_returns_nothing() {
        let store = mk_store(vec![vec![1, 2, 3]]);
        let idx = RoutingIndex::from_store(&store);
        let q = RoutingQuery { token_ids: vec![] };
        assert!(idx.resolve(&q, 5).is_empty());
    }
}
