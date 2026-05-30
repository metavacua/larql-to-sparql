//! Relation discovery via clustering.
//!
//! Clusters normalized direction vectors (target embeddings from down projections)
//! to discover natural relation types in the model's knowledge.
//!
//! Modules:
//! - `kmeans`: K-means clustering with BLAS acceleration
//! - `labeling`: Auto-generate labels from cluster members + curated vocabulary
//! - `categories`: Curated category vocabulary and stop words

pub mod categories;
pub mod kmeans;
pub mod labeling;
pub mod pair_matching;
pub mod probe;

use ndarray::Array1;
use serde::{Deserialize, Serialize};

// Re-export the main entry points
pub use kmeans::kmeans;
pub use labeling::{auto_label_clusters, auto_label_clusters_from_embeddings};
pub use pair_matching::{
    label_clusters_from_outputs, label_clusters_from_pairs, load_reference_databases,
};

/// Result of clustering: centres + assignments + auto-generated labels.
#[derive(Serialize, Deserialize, Clone)]
pub struct ClusterResult {
    pub k: usize,
    pub centres: Vec<Vec<f32>>,
    pub labels: Vec<String>,
    pub counts: Vec<usize>,
    pub top_tokens: Vec<Vec<String>>,
}

/// Classify a direction vector against stored cluster centres.
/// Returns (cluster_index, cosine_similarity).
pub fn classify_direction(direction: &Array1<f32>, centres: &[Vec<f32>]) -> (usize, f32) {
    let mut best_c = 0;
    let mut best_sim = f32::NEG_INFINITY;

    let d_norm = larql_compute::norm(&direction.view());
    if d_norm < 1e-8 {
        return (0, 0.0);
    }

    for (c, centre) in centres.iter().enumerate() {
        let centre_arr = Array1::from_vec(centre.clone());
        let c_norm = larql_compute::norm(&centre_arr.view());
        if c_norm < 1e-8 {
            continue;
        }
        let sim = larql_compute::dot(&direction.view(), &centre_arr.view()) / (d_norm * c_norm);
        if sim > best_sim {
            best_sim = sim;
            best_c = c;
        }
    }

    (best_c, best_sim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_direction_picks_highest_cosine() {
        // 3 centres pointing along the three axes; query along the
        // y-axis must pick centre 1 with cosine = 1.0.
        let centres = vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let q = Array1::from_vec(vec![0.0_f32, 1.0, 0.0]);
        let (idx, sim) = classify_direction(&q, &centres);
        assert_eq!(idx, 1);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn classify_direction_handles_off_axis_query() {
        // Query at 45° in xy-plane: closer to centre 0 than centre 1
        // by an infinitesimal margin (numerical equality), but the
        // first-seen-wins tie-break in `>` (not `>=`) keeps centre 0.
        let centres = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        let q = Array1::from_vec(vec![1.0_f32, 1.0]);
        let (idx, sim) = classify_direction(&q, &centres);
        // Both have cos = 1/sqrt(2); strict `>` keeps the first one.
        assert_eq!(idx, 0);
        assert!((sim - (1.0_f32 / 2.0_f32.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn classify_direction_zero_query_returns_default() {
        // d_norm < 1e-8 → early return (0, 0.0) regardless of centres.
        let centres = vec![vec![1.0_f32; 4], vec![2.0; 4]];
        let q = Array1::<f32>::zeros(4);
        let (idx, sim) = classify_direction(&q, &centres);
        assert_eq!(idx, 0);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn classify_direction_skips_zero_centres() {
        // Centre 0 has zero norm and must be skipped; centre 1 wins
        // even though it would have lost on a literal first-pass index.
        let centres = vec![vec![0.0_f32; 3], vec![1.0, 0.0, 0.0]];
        let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0]);
        let (idx, sim) = classify_direction(&q, &centres);
        assert_eq!(idx, 1);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn classify_direction_empty_centres_keeps_default() {
        let q = Array1::from_vec(vec![1.0_f32, 0.0]);
        let (idx, sim) = classify_direction(&q, &[]);
        // No centres → loop never runs; defaults survive.
        assert_eq!(idx, 0);
        assert_eq!(sim, f32::NEG_INFINITY);
    }

    #[test]
    fn classify_direction_negative_correlation_picks_least_negative() {
        // All centres anti-aligned with query: best is the *least* negative.
        let centres = vec![
            vec![-1.0_f32, 0.0], // cos = -1
            vec![-0.5, -0.866],  // cos ≈ -0.5
        ];
        let q = Array1::from_vec(vec![1.0_f32, 0.0]);
        let (idx, sim) = classify_direction(&q, &centres);
        assert_eq!(idx, 1, "least-negative correlation wins");
        assert!(sim < 0.0);
    }

    #[test]
    fn cluster_result_is_serde_round_trip() {
        let r = ClusterResult {
            k: 2,
            centres: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            labels: vec!["x".into(), "y".into()],
            counts: vec![10, 5],
            top_tokens: vec![vec!["a".into(), "b".into()], vec!["c".into()]],
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: ClusterResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.k, 2);
        assert_eq!(back.centres, r.centres);
        assert_eq!(back.labels, r.labels);
        assert_eq!(back.counts, r.counts);
        assert_eq!(back.top_tokens, r.top_tokens);
    }
}
