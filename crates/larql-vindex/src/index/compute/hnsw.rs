//! HNSW (Hierarchical Navigable Small World) index for gate vector search.
//!
//! Replaces brute-force gate KNN (O(N) comparisons per query) with
//! approximate nearest neighbor search via graph traversal (O(log N)).
//!
//! Uses random projection to reduce dimensionality during graph construction
//! and search traversal. Final candidates are scored with exact dot products
//! by the caller. This makes the build practical at dim=2560.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Max-heap element (best score first).
#[derive(Clone, Copy)]
struct MaxScored {
    score: f32,
    id: u32,
}
impl PartialEq for MaxScored {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id
    }
}
impl Eq for MaxScored {}
impl PartialOrd for MaxScored {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MaxScored {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score.partial_cmp(&o.score).unwrap_or(Ordering::Equal)
    }
}

/// Min-heap element (worst score first — for eviction).
#[derive(Clone, Copy)]
struct MinScored {
    score: f32,
    id: u32,
}
impl PartialEq for MinScored {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id
    }
}
impl Eq for MinScored {}
impl PartialOrd for MinScored {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinScored {
    fn cmp(&self, o: &Self) -> Ordering {
        o.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
    }
}

/// Projected dimension for graph construction.
/// Full-dim dot products are only done for final candidate scoring.
const PROJ_DIM: usize = 64;

/// HNSW index for a single layer's gate vectors.
///
/// The graph is built and traversed using random-projected vectors (dim=64).
/// This makes build O(N log N) at dim=64 instead of dim=2560 — ~40x faster.
/// Search returns candidate IDs; the caller does exact scoring on the originals.
pub struct HnswLayer {
    num_vectors: usize,
    m: usize,
    m_max0: usize,
    max_level: usize,
    entry_point: usize,
    node_levels: Vec<u8>,
    level0: Vec<u32>,
    upper: Vec<Vec<u32>>,
    /// Random projection matrix: [dim, PROJ_DIM] for query projection.
    proj_matrix: Array2<f32>,
    /// Projected vectors: [num_vectors, PROJ_DIM] for fast graph traversal.
    projected: Array2<f32>,
}

impl HnswLayer {
    /// Build an HNSW index from gate vectors.
    ///
    /// `vectors`: [num_vectors, dim] matrix (used for random projection).
    /// `m`: max connections per node (8-16 typical for 10K vectors).
    /// `ef_construction`: beam width during build (32-100 typical).
    pub fn build(vectors: &ArrayView2<f32>, m: usize, ef_construction: usize) -> Self {
        let n = vectors.shape()[0];
        let dim = vectors.shape()[1];
        let m_max0 = m * 2;
        let ml = 1.0 / (m as f64).ln();

        if n == 0 {
            return Self {
                num_vectors: 0,
                m,
                m_max0,
                max_level: 0,
                entry_point: 0,
                node_levels: vec![],
                level0: vec![],
                upper: vec![],
                proj_matrix: Array2::zeros((0, PROJ_DIM)),
                projected: Array2::zeros((0, PROJ_DIM)),
            };
        }

        // Random projection: dim -> PROJ_DIM
        let proj_matrix = Self::random_projection_matrix(dim, PROJ_DIM);
        let cpu = larql_compute::CpuBackend;
        use larql_compute::MatMul;
        let projected = cpu.matmul(vectors.view(), proj_matrix.view());

        // Assign random levels
        let mut node_levels = vec![0u8; n];
        let mut max_level = 0usize;
        let mut rng = 42u64;
        for nl in node_levels.iter_mut().take(n) {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng >> 33) as f64 / (1u64 << 31) as f64;
            let level = ((-u.max(1e-12).ln() * ml).floor() as usize).min(12);
            *nl = level as u8;
            if level > max_level {
                max_level = level;
            }
        }

        let level0 = vec![u32::MAX; n * m_max0];
        let upper: Vec<Vec<u32>> = (0..max_level).map(|_| vec![u32::MAX; n * m]).collect();

        let entry_point = node_levels
            .iter()
            .enumerate()
            .max_by_key(|(_, &l)| l)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut index = Self {
            num_vectors: n,
            m,
            m_max0,
            max_level,
            entry_point,
            node_levels,
            level0,
            upper,
            proj_matrix,
            projected,
        };

        // Build graph using projected vectors (dim=64, fast).
        // Clone projected to avoid borrow conflict with mutable index methods.
        let proj = index.projected.clone();
        let proj_view = proj.view();
        for id in 0..n {
            if id == entry_point && id == 0 {
                continue;
            }
            let q = proj_view.row(id);
            let node_level = index.node_levels[id] as usize;

            let mut ep = index.entry_point;
            for lev in (node_level.saturating_add(1)..=index.max_level).rev() {
                ep = index.greedy_closest(&proj_view, &q, ep, lev);
            }

            for lev in (0..=node_level.min(index.max_level)).rev() {
                let max_conn = if lev == 0 { m_max0 } else { m };
                let candidates = index.search_level(&proj_view, &q, ep, ef_construction, lev);

                let selected: Vec<u32> = candidates.iter().take(max_conn).map(|s| s.id).collect();

                index.set_neighbors(id, lev, &selected);

                for &nb in &selected {
                    index.add_connection(nb as usize, lev, id as u32, max_conn, &proj_view);
                }

                if let Some(closest) = selected.first() {
                    ep = *closest as usize;
                }
            }

            if node_level > index.node_levels[index.entry_point] as usize {
                index.entry_point = id;
            }
        }

        index
    }

    /// Search for top-K nearest neighbors.
    ///
    /// Uses projected vectors for graph traversal, then scores final candidates
    /// with exact full-dimensional dot products against `vectors`.
    ///
    /// Returns (feature_index, exact_score) sorted by score descending.
    pub fn search(
        &self,
        vectors: &ArrayView2<f32>,
        query: &Array1<f32>,
        top_k: usize,
        ef_search: usize,
    ) -> Vec<(usize, f32)> {
        if self.num_vectors == 0 {
            return vec![];
        }

        let ef = ef_search.max(top_k);

        // Project query to low-dim (PROJ_DIM) for fast graph traversal
        let proj_view = self.projected.view();
        let cpu = larql_compute::CpuBackend;
        use larql_compute::MatMul;
        let x = query
            .view()
            .into_shape_with_order((1, query.len()))
            .unwrap();
        let proj_2d = cpu.matmul(x, self.proj_matrix.view());
        let proj_query = Array1::from_vec(proj_2d.into_raw_vec_and_offset().0);

        // Upper levels: greedy descent using projected vectors (dim=64, fast)
        let mut ep = self.entry_point;
        for lev in (1..=self.max_level).rev() {
            ep = self.greedy_closest(&proj_view, &proj_query.view(), ep, lev);
        }

        // Level 0: beam search using projected vectors (ef comparisons at dim=64)
        let candidates = self.search_level(&proj_view, &proj_query.view(), ep, ef, 0);

        // Re-score final candidates with exact full-dim dot products
        let mut results: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|s| {
                let exact_score = Self::dot(&vectors.row(s.id as usize), &query.view());
                (s.id as usize, exact_score)
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Generate a random projection matrix [dim, proj_dim].
    /// Uses the same deterministic RNG for reproducibility.
    fn random_projection_matrix(dim: usize, proj_dim: usize) -> Array2<f32> {
        let scale = 1.0 / (proj_dim as f32).sqrt();
        let mut rng = 123456789u64;
        Array2::from_shape_fn((dim, proj_dim), |_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0;
            u * scale
        })
    }

    // ── Internals ──

    #[inline(always)]
    fn dot(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
        larql_compute::dot(a, b)
    }

    fn greedy_closest(
        &self,
        vectors: &ArrayView2<f32>,
        query: &ArrayView1<f32>,
        mut ep: usize,
        level: usize,
    ) -> usize {
        let mut best = Self::dot(&vectors.row(ep), query);
        loop {
            let mut changed = false;
            for &nb in self.neighbors(ep, level) {
                if nb == u32::MAX {
                    break;
                }
                let s = Self::dot(&vectors.row(nb as usize), query);
                if s > best {
                    best = s;
                    ep = nb as usize;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        ep
    }

    fn search_level(
        &self,
        vectors: &ArrayView2<f32>,
        query: &ArrayView1<f32>,
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<MaxScored> {
        let mut visited = vec![false; self.num_vectors];
        visited[entry] = true;

        let entry_score = Self::dot(&vectors.row(entry), query);

        let mut candidates: BinaryHeap<MaxScored> = BinaryHeap::new();
        candidates.push(MaxScored {
            score: entry_score,
            id: entry as u32,
        });

        let mut results: BinaryHeap<MinScored> = BinaryHeap::new();
        results.push(MinScored {
            score: entry_score,
            id: entry as u32,
        });

        while let Some(current) = candidates.pop() {
            let worst = results.peek().map(|s| s.score).unwrap_or(f32::NEG_INFINITY);
            if current.score < worst && results.len() >= ef {
                break;
            }

            for &nb in self.neighbors(current.id as usize, level) {
                if nb == u32::MAX {
                    break;
                }
                let nid = nb as usize;
                if nid >= self.num_vectors || visited[nid] {
                    continue;
                }
                visited[nid] = true;

                let score = Self::dot(&vectors.row(nid), query);
                let worst = results.peek().map(|s| s.score).unwrap_or(f32::NEG_INFINITY);

                if score > worst || results.len() < ef {
                    candidates.push(MaxScored { score, id: nb });
                    results.push(MinScored { score, id: nb });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<MaxScored> = results
            .into_iter()
            .map(|s| MaxScored {
                score: s.score,
                id: s.id,
            })
            .collect();
        out.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        out
    }

    fn neighbors(&self, node: usize, level: usize) -> &[u32] {
        if level == 0 {
            let s = node * self.m_max0;
            &self.level0[s..s + self.m_max0]
        } else if level <= self.upper.len() {
            let s = node * self.m;
            let arr = &self.upper[level - 1];
            if s + self.m <= arr.len() {
                &arr[s..s + self.m]
            } else {
                &[]
            }
        } else {
            &[]
        }
    }

    fn set_neighbors(&mut self, node: usize, level: usize, nbs: &[u32]) {
        if level == 0 {
            let s = node * self.m_max0;
            for (i, &n) in nbs.iter().take(self.m_max0).enumerate() {
                self.level0[s + i] = n;
            }
        } else if level <= self.upper.len() {
            let s = node * self.m;
            let arr = &mut self.upper[level - 1];
            for (i, &n) in nbs.iter().take(self.m).enumerate() {
                arr[s + i] = n;
            }
        }
    }

    fn add_connection(
        &mut self,
        node: usize,
        level: usize,
        new_nb: u32,
        max_conn: usize,
        vectors: &ArrayView2<f32>,
    ) {
        let (arr, start, cap) = if level == 0 {
            (
                &mut self.level0 as &mut Vec<u32>,
                node * self.m_max0,
                self.m_max0.min(max_conn),
            )
        } else if level <= self.upper.len() {
            (
                &mut self.upper[level - 1] as &mut Vec<u32>,
                node * self.m,
                self.m.min(max_conn),
            )
        } else {
            return;
        };

        if start + cap > arr.len() {
            return;
        }
        let slot = &mut arr[start..start + cap];

        for s in slot.iter_mut().take(cap) {
            if *s == u32::MAX {
                *s = new_nb;
                return;
            }
            if *s == new_nb {
                return;
            }
        }

        // Evict worst neighbor if new one is better
        let node_vec = vectors.row(node);
        let new_score = Self::dot(&node_vec, &vectors.row(new_nb as usize));
        let mut worst_i = 0;
        let mut worst_s = f32::MAX;
        for (i, &nb) in slot.iter().enumerate().take(cap) {
            let s = Self::dot(&node_vec, &vectors.row(nb as usize));
            if s < worst_s {
                worst_s = s;
                worst_i = i;
            }
        }
        if new_score > worst_s {
            slot[worst_i] = new_nb;
        }
    }

    pub fn len(&self) -> usize {
        self.num_vectors
    }
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // Dataset geometry: small enough to brute-force, large enough that
    // the graph has multiple levels and neighbor eviction happens.
    const DIM: usize = 24;
    const NUM_VECTORS: usize = 200;
    /// Size at which the level-0 graph is still fully connected (see
    /// `level0_graph_fully_connected_at_small_n` — beyond ~64 vectors
    /// the naive neighbor eviction starts orphaning nodes).
    const SMALL_N: usize = 50;
    /// Size for the clustered recall benchmark (measured recall 0.97).
    const RECALL_N: usize = 100;
    const M: usize = 8;
    const EF_CONSTRUCTION: usize = 64;
    const EF_SEARCH: usize = 100;
    const TOP_K: usize = 10;
    /// Minimum acceptable recall@10 vs brute force on clustered data at
    /// RECALL_N. Measured behavior is 0.97 (deterministic build +
    /// dataset); the floor leaves headroom for benign re-tuning while
    /// still catching a broken graph.
    ///
    /// Deliberately NOT asserted on uniform-random data at
    /// NUM_VECTORS=200: the naive add_connection eviction fragments the
    /// level-0 graph there (only ~33/200 nodes reachable from the entry
    /// point, recall ~0.16 even with ef=n). That is a defect of the
    /// current construction, documented by
    /// `level0_graph_fully_connected_at_small_n`'s size bound rather
    /// than papered over with a loose floor.
    const RECALL_FLOOR: f64 = 0.8;
    const NUM_RECALL_QUERIES: usize = 30;

    /// Test-local RNG for dataset generation (xorshift64*; distinct
    /// from the production LCGs on purpose).
    const DATASET_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    // The production level-assignment LCG (lines ~122-126). Pinned
    // here so a constant change breaks a test instead of silently
    // reshaping every previously built graph.
    const LEVEL_LCG_SEED: u64 = 42;
    const LEVEL_LCG_MUL: u64 = 6364136223846793005;
    const LEVEL_LCG_ADD: u64 = 1442695040888963407;
    const LEVEL_CAP: usize = 12;

    fn next_dataset_val(state: &mut u64) -> f32 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        // Top 24 bits, mapped to [-1, 1)
        (bits >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    /// `n` unit-norm random vectors, deterministic across runs.
    fn unit_vectors(n: usize, dim: usize) -> Array2<f32> {
        let mut state = DATASET_SEED;
        let mut m = Array2::from_shape_fn((n, dim), |_| next_dataset_val(&mut state));
        for mut row in m.rows_mut() {
            let norm = row.dot(&row).sqrt().max(1e-12);
            row.mapv_inplace(|v| v / norm);
        }
        m
    }

    /// Exact top-k feature ids by dot product, descending.
    fn brute_force_top_k(vectors: &Array2<f32>, query: &Array1<f32>, k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = vectors
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| (i, row.dot(&query.view())))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn empty_index_reports_empty_and_searches_to_nothing() {
        let vectors = Array2::<f32>::zeros((0, DIM));
        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        let query = Array1::from_vec(vec![1.0; DIM]);
        assert!(index
            .search(&vectors.view(), &query, TOP_K, EF_SEARCH)
            .is_empty());
    }

    #[test]
    fn single_element_index_returns_that_element() {
        let vectors = unit_vectors(1, DIM);
        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
        let query = vectors.row(0).to_owned();
        let results = index.search(&vectors.view(), &query, TOP_K, EF_SEARCH);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        // Unit vector against itself: exact score is 1.
        assert!((results[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_larger_than_n_returns_all_elements() {
        const SMALL_N: usize = 3;
        let vectors = unit_vectors(SMALL_N, DIM);
        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);
        let query = vectors.row(0).to_owned();
        let results = index.search(&vectors.view(), &query, TOP_K, EF_SEARCH);
        assert_eq!(results.len(), SMALL_N);
        let mut ids: Vec<usize> = results.iter().map(|r| r.0).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn search_returns_inserted_item_and_sorts_descending() {
        let vectors = unit_vectors(SMALL_N, DIM);
        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);
        assert_eq!(index.len(), SMALL_N);

        const PROBE_ID: usize = 17;
        let query = vectors.row(PROBE_ID).to_owned();
        let results = index.search(&vectors.view(), &query, TOP_K, EF_SEARCH);
        assert!(!results.is_empty() && results.len() <= TOP_K);
        assert!(
            results.iter().any(|&(id, _)| id == PROBE_ID),
            "query identical to vector {PROBE_ID} must surface it: {results:?}"
        );
        for pair in results.windows(2) {
            assert!(pair[0].1 >= pair[1].1, "scores not descending: {results:?}");
        }
        // Scores are exact full-dim dot products, so self-match is ~1.
        assert!((results[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn planted_near_neighbour_is_found() {
        // Plant a near-duplicate of vector 0 at a known slot and check
        // both base and plant surface for a query at the base.
        const BASE_ID: usize = 0;
        const PLANTED_ID: usize = 40;
        const PLANT_NOISE: f32 = 0.05;

        let mut vectors = unit_vectors(SMALL_N, DIM);
        let planted: Vec<f32> = {
            let base = vectors.row(BASE_ID).to_owned();
            let mut state = DATASET_SEED ^ 0xABCD;
            let noisy: Vec<f32> = base
                .iter()
                .map(|v| v + PLANT_NOISE * next_dataset_val(&mut state))
                .collect();
            let norm = noisy.iter().map(|v| v * v).sum::<f32>().sqrt();
            noisy.into_iter().map(|v| v / norm).collect()
        };
        for (j, v) in planted.iter().enumerate() {
            vectors[[PLANTED_ID, j]] = *v;
        }

        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);
        let query = vectors.row(BASE_ID).to_owned();
        let results = index.search(&vectors.view(), &query, TOP_K, EF_SEARCH);
        let ids: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(ids.contains(&BASE_ID), "base vector missing: {ids:?}");
        assert!(
            ids.contains(&PLANTED_ID),
            "planted neighbour missing: {ids:?}"
        );
    }

    #[test]
    fn recall_at_10_clears_floor_on_clustered_dataset() {
        let vectors = clustered_vectors(RECALL_N, DIM);
        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);

        let mut hits = 0usize;
        for q in 0..NUM_RECALL_QUERIES {
            let query = vectors.row(q).to_owned();
            let expected = brute_force_top_k(&vectors, &query, TOP_K);
            let got: Vec<usize> = index
                .search(&vectors.view(), &query, TOP_K, EF_SEARCH)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits += expected.iter().filter(|id| got.contains(id)).count();
        }
        let recall = hits as f64 / (NUM_RECALL_QUERIES * TOP_K) as f64;
        assert!(
            recall >= RECALL_FLOOR,
            "recall@{TOP_K} = {recall:.3} below floor {RECALL_FLOOR}"
        );
    }

    #[test]
    fn duplicate_vectors_are_handled_without_panic() {
        const NUM_DUPES: usize = 5;
        let mut vectors = unit_vectors(50, DIM);
        let first = vectors.row(0).to_owned();
        for dup in 1..NUM_DUPES {
            for j in 0..DIM {
                vectors[[dup, j]] = first[j];
            }
        }
        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);
        let results = index.search(&vectors.view(), &first, NUM_DUPES, EF_SEARCH);
        assert!(!results.is_empty());
        // All returned copies of the duplicate score identically (~1).
        for &(id, score) in &results {
            if id < NUM_DUPES {
                assert!((score - 1.0).abs() < 1e-5, "dup {id} scored {score}");
            }
        }
    }

    #[test]
    fn ef_search_smaller_than_top_k_is_clamped() {
        const TINY_EF: usize = 1;
        let vectors = unit_vectors(SMALL_N, DIM);
        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);
        let query = vectors.row(3).to_owned();
        let results = index.search(&vectors.view(), &query, TOP_K, TINY_EF);
        // ef = max(ef_search, top_k), so we still get a full top-k.
        assert_eq!(results.len(), TOP_K);
    }

    #[test]
    fn build_is_deterministic_across_runs() {
        let vectors = unit_vectors(NUM_VECTORS, DIM);
        let a = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);
        let b = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);
        assert_eq!(a.node_levels, b.node_levels);
        assert_eq!(a.entry_point, b.entry_point);
        assert_eq!(a.max_level, b.max_level);
        assert_eq!(a.level0, b.level0);
        assert_eq!(a.upper, b.upper);
        assert_eq!(a.projected, b.projected);
        assert_eq!(a.proj_matrix, b.proj_matrix);
    }

    /// Pin the level-assignment LCG. The constants at build() are the
    /// wire-format of every graph ever built with this code: replicate
    /// the sequence independently and require identical levels, so a
    /// "harmless" constant tweak fails here instead of silently
    /// reshaping the graph.
    #[test]
    fn level_assignment_rng_matches_pinned_lcg_sequence() {
        let vectors = unit_vectors(NUM_VECTORS, DIM);
        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);

        let ml = 1.0 / (M as f64).ln();
        let mut rng = LEVEL_LCG_SEED;
        let mut expected = Vec::with_capacity(NUM_VECTORS);
        for _ in 0..NUM_VECTORS {
            rng = rng.wrapping_mul(LEVEL_LCG_MUL).wrapping_add(LEVEL_LCG_ADD);
            let u = (rng >> 33) as f64 / (1u64 << 31) as f64;
            let level = ((-u.max(1e-12).ln() * ml).floor() as usize).min(LEVEL_CAP);
            expected.push(level as u8);
        }
        assert_eq!(index.node_levels, expected);
        // Entry point is (a) node with the maximal level.
        let max = *expected.iter().max().unwrap();
        assert_eq!(index.node_levels[index.entry_point], max);
    }

    /// Pin the connectivity regime: at SMALL_N every node is reachable
    /// from the entry point over level-0 adjacency. This is the bound
    /// that makes the exact-retrieval tests above sound. It does NOT
    /// hold at NUM_VECTORS=200 on uniform data (only ~33/200 reachable
    /// — naive eviction in add_connection orphans nodes as n grows);
    /// if construction is ever fixed, extend this test upward.
    #[test]
    fn level0_graph_fully_connected_at_small_n() {
        let vectors = unit_vectors(SMALL_N, DIM);
        let index = HnswLayer::build(&vectors.view(), M, EF_CONSTRUCTION);

        let mut seen = [false; SMALL_N];
        let mut stack = vec![index.entry_point];
        seen[index.entry_point] = true;
        let mut reachable = 1;
        while let Some(u) = stack.pop() {
            for &nb in index.neighbors(u, 0) {
                if nb == u32::MAX {
                    break;
                }
                if !seen[nb as usize] {
                    seen[nb as usize] = true;
                    reachable += 1;
                    stack.push(nb as usize);
                }
            }
        }
        assert_eq!(
            reachable, SMALL_N,
            "level-0 graph must be fully connected at n={SMALL_N}"
        );
    }

    /// Clustered dataset: NUM_CLUSTERS well-separated directions, each
    /// with members = normalize(center + noise).
    fn clustered_vectors(n: usize, dim: usize) -> Array2<f32> {
        const NUM_CLUSTERS: usize = 20;
        const CLUSTER_NOISE: f32 = 0.15;
        let centers = unit_vectors(NUM_CLUSTERS, dim);
        let mut state = DATASET_SEED ^ 0x5A5A;
        let mut m = Array2::zeros((n, dim));
        for i in 0..n {
            let c = i % NUM_CLUSTERS;
            let mut row: Vec<f32> = centers
                .row(c)
                .iter()
                .map(|v| v + CLUSTER_NOISE * next_dataset_val(&mut state))
                .collect();
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            row.iter_mut().for_each(|v| *v /= norm);
            for (j, v) in row.into_iter().enumerate() {
                m[[i, j]] = v;
            }
        }
        m
    }

    #[test]
    fn projection_matrix_is_deterministic_and_scaled() {
        const TEST_DIM: usize = 40;
        let a = HnswLayer::random_projection_matrix(TEST_DIM, PROJ_DIM);
        let b = HnswLayer::random_projection_matrix(TEST_DIM, PROJ_DIM);
        assert_eq!(a.shape(), &[TEST_DIM, PROJ_DIM]);
        assert_eq!(a, b);
        // Values are bounded by the 1/sqrt(proj_dim) scale.
        let bound = 1.0 / (PROJ_DIM as f32).sqrt() + 1e-6;
        assert!(a.iter().all(|v| v.abs() <= bound));
        // And are not degenerate (all zero).
        assert!(a.iter().any(|v| v.abs() > 0.0));
    }
}
