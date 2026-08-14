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

/// Window ids are `u16` throughout the index, the store entries, and the
/// engine's routing surface, so at most `u16::MAX + 1` windows (ids
/// `0..=65535`) are addressable. A larger store would silently truncate
/// ids at index build (`window_id as u16`), aliasing distinct windows.
pub const MAX_ROUTABLE_WINDOWS: usize = (u16::MAX as usize) + 1;

/// Construction-time failure building a [`RoutingIndex`].
#[derive(Debug, thiserror::Error)]
pub enum RoutingBuildError {
    #[error(
        "store has {count} windows but window ids are u16 — at most {max} windows are routable"
    )]
    WindowCountExceedsU16 { count: usize, max: usize },
}

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
    ///
    /// Errors when the store holds more windows than a `u16` window id can
    /// address — the truncation would otherwise be silent.
    pub fn from_store(store: &ApolloStore) -> Result<Self, RoutingBuildError> {
        let count = store.window_tokens.len();
        if count > MAX_ROUTABLE_WINDOWS {
            return Err(RoutingBuildError::WindowCountExceedsU16 {
                count,
                max: MAX_ROUTABLE_WINDOWS,
            });
        }
        let mut index: HashMap<u32, HashMap<u16, u32>> = HashMap::new();
        for (window_id, tokens) in store.window_tokens.iter().enumerate() {
            // Safe cast: count <= MAX_ROUTABLE_WINDOWS checked above.
            let wid = window_id as u16;
            for &tok in tokens {
                *index.entry(tok).or_default().entry(wid).or_insert(0) += 1;
            }
        }
        let compacted: HashMap<u32, Vec<(u16, u32)>> = index
            .into_iter()
            .map(|(tok, wf)| (tok, wf.into_iter().collect()))
            .collect();
        Ok(Self {
            index: compacted,
            num_windows: count,
        })
    }

    /// Return the top-k window IDs most relevant to the query, ranked by
    /// sum of tf × idf with the BM25-style idf
    /// `ln((N - df + 0.5) / (df + 0.5) + 1)`.
    ///
    /// Determinism invariant: scores accumulate in a `HashMap`, so equal
    /// scores would surface in hash order without a secondary key. Ties
    /// break on window_id ascending, making the routed window (and hence
    /// the injected context) stable across calls and processes.
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
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.into_iter().take(top_k).map(|(w, _)| w).collect()
    }

    /// Total bytes used by the serialized index.
    pub fn total_bytes(&self) -> usize {
        self.index
            .values()
            .map(|v| 4 + v.len() * std::mem::size_of::<(u16, u32)>())
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::apollo::store::{ArchConfig, StoreManifest};

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
        let store = mk_store(vec![vec![1, 42, 3, 42, 5], vec![42, 7, 8], vec![9, 10, 11]]);
        let idx = RoutingIndex::from_store(&store).unwrap();
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
        let store = mk_store(vec![vec![99, 1, 2], vec![99, 7, 3], vec![99, 4, 5]]);
        let idx = RoutingIndex::from_store(&store).unwrap();
        let q = RoutingQuery {
            token_ids: vec![99, 7],
        };
        let res = idx.resolve(&q, 2);
        assert_eq!(res[0], 1);
    }

    #[test]
    fn resolve_empty_query_returns_nothing() {
        let store = mk_store(vec![vec![1, 2, 3]]);
        let idx = RoutingIndex::from_store(&store).unwrap();
        let q = RoutingQuery { token_ids: vec![] };
        assert!(idx.resolve(&q, 5).is_empty());
    }

    #[test]
    fn resolve_breaks_score_ties_by_window_id_ascending() {
        // Windows 0 and 1 have identical content (token 7, tf=1 each), so
        // their tf-idf scores tie exactly. Window 2 exists only to keep
        // df < N so token 7 isn't dropped as ubiquitous. Without the
        // secondary sort key the tie resolves in the HashMap's per-instance
        // hash order, which flips between calls; repeated resolution must
        // return the same ranking every time, lowest window id first.
        const TIE_TRIALS: usize = 50;
        let store = mk_store(vec![vec![7], vec![7], vec![9]]);
        let idx = RoutingIndex::from_store(&store).unwrap();
        let q = RoutingQuery { token_ids: vec![7] };
        for trial in 0..TIE_TRIALS {
            assert_eq!(
                idx.resolve(&q, 2),
                vec![0, 1],
                "tie-break unstable at trial {trial}"
            );
        }
    }

    #[test]
    fn from_store_rejects_window_count_beyond_u16_range() {
        // MAX_ROUTABLE_WINDOWS windows (ids 0..=65535) are fine; one more
        // would truncate `window_id as u16` silently, so it must error.
        let ok_store = mk_store(vec![vec![]; MAX_ROUTABLE_WINDOWS]);
        assert!(RoutingIndex::from_store(&ok_store).is_ok());

        let over_store = mk_store(vec![vec![]; MAX_ROUTABLE_WINDOWS + 1]);
        match RoutingIndex::from_store(&over_store) {
            Err(RoutingBuildError::WindowCountExceedsU16 { count, max }) => {
                assert_eq!(count, MAX_ROUTABLE_WINDOWS + 1);
                assert_eq!(max, MAX_ROUTABLE_WINDOWS);
            }
            Ok(_) => panic!("expected WindowCountExceedsU16, got Ok"),
        }
    }
}
