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
pub use pair_matching::{label_clusters_from_pairs, label_clusters_from_outputs, load_reference_databases};

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
pub fn classify_direction(
    direction: &Array1<f32>,
    centres: &[Vec<f32>],
) -> (usize, f32) {
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
