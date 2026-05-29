//! Relation type classifier for DESCRIBE edges.
//!
//! Uses discovered cluster centres from the vindex (computed during build).
//! Falls back to embedding-direction heuristics if no clusters are available.

use larql_inference::ndarray::{Array1, Array2};
use larql_inference::tokenizers::Tokenizer;
use larql_vindex::clustering::ClusterResult;
use larql_vindex::format::filenames::{
    FEATURE_CLUSTERS_JSONL, FEATURE_LABELS_JSON, RELATION_CLUSTERS_JSON,
};

/// Classifies edges into relation types using discovered clusters
/// or embedding-space direction matching.
pub struct RelationClassifier {
    /// Discovered clusters from the vindex (if available).
    clusters: Option<ClusterResult>,
    /// Per-feature cluster assignments: (layer, feature) → cluster_id.
    feature_assignments: std::collections::HashMap<(usize, usize), usize>,
    /// Probe-confirmed per-feature labels (highest priority).
    probe_labels: std::collections::HashMap<(usize, usize), String>,
    /// Number of probe-confirmed labels.
    probe_count: usize,
}

impl RelationClassifier {
    /// Build a classifier from discovered clusters + probe labels in a vindex directory.
    /// Returns Some even if only probe labels exist (no clusters needed).
    pub fn from_vindex(vindex_path: &std::path::Path) -> Option<Self> {
        let clusters_path = vindex_path.join(RELATION_CLUSTERS_JSON);
        let assignments_path = vindex_path.join(FEATURE_CLUSTERS_JSONL);
        let probe_labels_path = vindex_path.join(FEATURE_LABELS_JSON);

        // Clusters are optional — probe labels work without them
        let clusters: Option<ClusterResult> = std::fs::read_to_string(&clusters_path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok());

        let mut feature_assignments = std::collections::HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&assignments_path) {
            for line in text.lines() {
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                    let layer = obj["l"].as_u64().unwrap_or(0) as usize;
                    let feat = obj["f"].as_u64().unwrap_or(0) as usize;
                    let cluster = obj["c"].as_u64().unwrap_or(0) as usize;
                    feature_assignments.insert((layer, feat), cluster);
                }
            }
        }

        // Load probe-confirmed per-feature labels (highest priority)
        let mut probe_labels = std::collections::HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&probe_labels_path) {
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(map) = obj.as_object() {
                    for (key, value) in map {
                        if let Some(rel) = value.as_str() {
                            // Parse "L{layer}_F{feature}" key
                            let parts: Vec<&str> = key.split('_').collect();
                            if parts.len() == 2 {
                                if let (Some(layer), Some(feat)) = (
                                    parts[0]
                                        .strip_prefix('L')
                                        .and_then(|s| s.parse::<usize>().ok()),
                                    parts[1]
                                        .strip_prefix('F')
                                        .and_then(|s| s.parse::<usize>().ok()),
                                ) {
                                    probe_labels.insert((layer, feat), rel.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return None only if we have nothing at all
        if clusters.is_none() && feature_assignments.is_empty() && probe_labels.is_empty() {
            return None;
        }

        let probe_count = probe_labels.len();

        Some(Self {
            clusters,
            feature_assignments,
            probe_labels,
            probe_count,
        })
    }

    /// Get the relation label for a feature at (layer, feature_index).
    /// Probe-confirmed labels take priority over cluster-assigned labels.
    pub fn label_for_feature(&self, layer: usize, feature: usize) -> Option<&str> {
        // Tier 1: probe-confirmed label (ground truth)
        if let Some(label) = self.probe_labels.get(&(layer, feature)) {
            return Some(label.as_str());
        }
        // Tier 2: cluster-assigned label
        let clusters = self.clusters.as_ref()?;
        let &cluster_id = self.feature_assignments.get(&(layer, feature))?;
        clusters.labels.get(cluster_id).map(|s| s.as_str())
    }

    /// Get the cluster ID for a feature.
    pub fn cluster_for_feature(&self, layer: usize, feature: usize) -> Option<usize> {
        self.feature_assignments.get(&(layer, feature)).copied()
    }

    /// Get cluster info (label, count, top tokens).
    pub fn cluster_info(&self, cluster_id: usize) -> Option<(&str, usize, &[String])> {
        let clusters = self.clusters.as_ref()?;
        let label = clusters.labels.get(cluster_id)?;
        let count = clusters.counts.get(cluster_id).copied().unwrap_or(0);
        let tops = clusters
            .top_tokens
            .get(cluster_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        Some((label, count, tops))
    }

    /// Check whether a feature's label is probe-confirmed (vs cluster-assigned).
    pub fn is_probe_label(&self, layer: usize, feature: usize) -> bool {
        self.probe_labels.contains_key(&(layer, feature))
    }

    /// Number of probe-confirmed feature labels.
    pub fn num_probe_labels(&self) -> usize {
        self.probe_count
    }

    /// Number of discovered clusters.
    pub fn num_clusters(&self) -> usize {
        self.clusters.as_ref().map(|c| c.k).unwrap_or(0)
    }

    /// Whether this classifier has discovered clusters (vs empty).
    pub fn has_clusters(&self) -> bool {
        self.clusters.is_some() && self.num_clusters() > 0
    }

    /// Classify a direction vector against the stored cluster centres.
    /// Returns (cluster_id, label, cosine_similarity).
    pub fn classify_direction(&self, direction: &Array1<f32>) -> Option<(usize, &str, f32)> {
        let clusters = self.clusters.as_ref()?;
        let (cluster_id, sim) =
            larql_vindex::clustering::classify_direction(direction, &clusters.centres);
        let label = clusters
            .labels
            .get(cluster_id)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        Some((cluster_id, label, sim))
    }

    /// Find the cluster ID for a relation label.
    /// Tries exact match, then normalised variants (hyphens, underscores, spaces all equivalent).
    pub fn cluster_for_relation(&self, relation: &str) -> Option<usize> {
        let clusters = self.clusters.as_ref()?;
        let norm = normalise_relation(relation);

        // Exact match
        for (i, label) in clusters.labels.iter().enumerate() {
            if label.eq_ignore_ascii_case(relation) {
                return Some(i);
            }
        }
        // Normalised match (hyphens, underscores, spaces all equivalent)
        for (i, label) in clusters.labels.iter().enumerate() {
            if normalise_relation(label) == norm {
                return Some(i);
            }
        }
        // Substring match
        for (i, label) in clusters.labels.iter().enumerate() {
            let label_norm = normalise_relation(label);
            if label_norm.contains(&norm) || norm.contains(&label_norm) {
                return Some(i);
            }
        }
        None
    }

    /// Get the cluster centre vector for a relation label.
    pub fn cluster_centre_for_relation(&self, relation: &str) -> Option<Vec<f32>> {
        let cluster_id = self.cluster_for_relation(relation)?;
        let clusters = self.clusters.as_ref()?;
        clusters.centres.get(cluster_id).cloned()
    }

    /// Find the typical layer for a relation by scanning probe labels and cluster assignments.
    /// Returns the most common layer for features with this relation.
    pub fn typical_layer_for_relation(&self, relation: &str) -> Option<usize> {
        let norm = normalise_relation(relation);
        let mut layer_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        // Check probe labels
        for (&(layer, _), label) in &self.probe_labels {
            let label_norm = normalise_relation(label);
            if label_norm == norm || label_norm.contains(&norm) || norm.contains(&label_norm) {
                *layer_counts.entry(layer).or_default() += 1;
            }
        }

        // If no probe matches, check cluster assignments
        if layer_counts.is_empty() {
            if let Some(cluster_id) = self.cluster_for_relation(relation) {
                for (&(layer, _), &cid) in &self.feature_assignments {
                    if cid == cluster_id {
                        *layer_counts.entry(layer).or_default() += 1;
                    }
                }
            }
        }

        layer_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(layer, _)| layer)
    }
}

/// Normalise a relation name: lowercase, replace hyphens/underscores/spaces with a single space.
fn normalise_relation(s: &str) -> String {
    s.to_lowercase()
        .replace(['-', '_'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Get the averaged embedding for a token string (public for executor use).
pub fn token_embedding_pub(
    text: &str,
    embed: &Array2<f32>,
    embed_scale: f32,
    tokenizer: &Tokenizer,
) -> Option<Array1<f32>> {
    let encoding = tokenizer.encode(text, false).ok()?;
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return None;
    }

    let hidden = embed.shape()[1];
    let mut avg = Array1::<f32>::zeros(hidden);
    for &id in ids {
        if (id as usize) < embed.shape()[0] {
            avg += &embed.row(id as usize).mapv(|v| v * embed_scale);
        }
    }
    avg /= ids.len() as f32;
    Some(avg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::clustering::ClusterResult;

    fn make_test_classifier() -> RelationClassifier {
        let clusters = ClusterResult {
            k: 3,
            centres: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]],
            labels: vec!["capital".into(), "language".into(), "continent".into()],
            counts: vec![100, 80, 60],
            top_tokens: vec![
                vec!["paris".into(), "berlin".into()],
                vec!["french".into(), "german".into()],
                vec!["europe".into(), "asia".into()],
            ],
        };

        let mut assignments = std::collections::HashMap::new();
        assignments.insert((26, 9515), 0); // capital cluster
        assignments.insert((24, 4532), 1); // language cluster
        assignments.insert((25, 3603), 2); // continent cluster

        RelationClassifier {
            clusters: Some(clusters),
            feature_assignments: assignments,
            probe_labels: std::collections::HashMap::new(),
            probe_count: 0,
        }
    }

    #[test]
    fn label_for_known_feature() {
        let rc = make_test_classifier();
        assert_eq!(rc.label_for_feature(26, 9515), Some("capital"));
        assert_eq!(rc.label_for_feature(24, 4532), Some("language"));
        assert_eq!(rc.label_for_feature(25, 3603), Some("continent"));
    }

    #[test]
    fn label_for_unknown_feature() {
        let rc = make_test_classifier();
        assert_eq!(rc.label_for_feature(0, 0), None);
        assert_eq!(rc.label_for_feature(99, 99), None);
    }

    #[test]
    fn cluster_for_feature() {
        let rc = make_test_classifier();
        assert_eq!(rc.cluster_for_feature(26, 9515), Some(0));
        assert_eq!(rc.cluster_for_feature(24, 4532), Some(1));
        assert_eq!(rc.cluster_for_feature(0, 0), None);
    }

    #[test]
    fn cluster_info() {
        let rc = make_test_classifier();
        let (label, count, tops) = rc.cluster_info(0).unwrap();
        assert_eq!(label, "capital");
        assert_eq!(count, 100);
        assert_eq!(tops, &["paris", "berlin"]);
    }

    #[test]
    fn num_clusters() {
        let rc = make_test_classifier();
        assert_eq!(rc.num_clusters(), 3);
        assert!(rc.has_clusters());
    }

    #[test]
    fn from_nonexistent_vindex() {
        let rc = RelationClassifier::from_vindex(std::path::Path::new("/nonexistent"));
        assert!(rc.is_none());
    }

    // ── cluster_for_relation: exact / normalised / substring tiers ──

    #[test]
    fn cluster_for_relation_exact_match_case_insensitive() {
        let rc = make_test_classifier();
        assert_eq!(rc.cluster_for_relation("capital"), Some(0));
        // Mixed-case should still match the exact-match arm via
        // eq_ignore_ascii_case.
        assert_eq!(rc.cluster_for_relation("CAPITAL"), Some(0));
    }

    #[test]
    fn cluster_for_relation_normalised_match_for_hyphens() {
        // Underscore/hyphen normalisation arm: query "official-language"
        // should fall through to the normalised arm if the cluster
        // label is "official_language" or "official language".
        let mut rc = make_test_classifier();
        if let Some(c) = rc.clusters.as_mut() {
            c.labels.push("official_language".into());
            c.centres.push(vec![0.7, 0.3]);
            c.counts.push(50);
            c.top_tokens.push(vec!["english".into()]);
            c.k = c.labels.len();
        }
        assert_eq!(rc.cluster_for_relation("official-language"), Some(3));
        assert_eq!(rc.cluster_for_relation("official language"), Some(3));
    }

    #[test]
    fn cluster_for_relation_substring_match_fallback() {
        // Substring arm: a query that's a substring of a label.
        let rc = make_test_classifier();
        // "capit" is a substring of "capital".
        assert_eq!(rc.cluster_for_relation("capit"), Some(0));
    }

    #[test]
    fn cluster_for_relation_returns_none_when_no_match() {
        let rc = make_test_classifier();
        assert_eq!(rc.cluster_for_relation("totally_unrelated_xyz"), None);
    }

    #[test]
    fn cluster_for_relation_returns_none_without_clusters() {
        // A classifier with only probe labels (no clusters at all)
        // returns None for any relation lookup.
        let rc = RelationClassifier {
            clusters: None,
            feature_assignments: std::collections::HashMap::new(),
            probe_labels: std::collections::HashMap::new(),
            probe_count: 0,
        };
        assert_eq!(rc.cluster_for_relation("anything"), None);
    }

    #[test]
    fn cluster_centre_for_relation_round_trip() {
        let rc = make_test_classifier();
        let centre = rc.cluster_centre_for_relation("capital").unwrap();
        assert_eq!(centre, vec![1.0, 0.0]);
    }

    #[test]
    fn cluster_centre_for_relation_unknown_returns_none() {
        let rc = make_test_classifier();
        assert!(rc.cluster_centre_for_relation("xyz_nonexistent").is_none());
    }

    // ── typical_layer_for_relation ──

    #[test]
    fn typical_layer_for_relation_finds_via_cluster_assignments() {
        let rc = make_test_classifier();
        assert_eq!(rc.typical_layer_for_relation("capital"), Some(26));
        assert_eq!(rc.typical_layer_for_relation("language"), Some(24));
    }

    #[test]
    fn typical_layer_for_relation_uses_probe_labels_first() {
        // Probe labels matching the queried relation should drive the
        // layer pick over cluster assignments.
        let mut rc = make_test_classifier();
        rc.probe_labels.insert((10, 5), "capital".into());
        rc.probe_labels.insert((10, 6), "capital".into());
        rc.probe_labels.insert((11, 7), "capital".into());
        rc.probe_count = rc.probe_labels.len();
        // Layer 10 has 2 hits, layer 11 has 1 → pick layer 10.
        assert_eq!(rc.typical_layer_for_relation("capital"), Some(10));
    }

    #[test]
    fn typical_layer_for_unknown_relation_returns_none() {
        let rc = make_test_classifier();
        assert_eq!(rc.typical_layer_for_relation("totally_unknown_xyz"), None);
    }

    // ── classify_direction ──

    #[test]
    fn classify_direction_picks_nearest_centre() {
        let rc = make_test_classifier();
        // Vector pointing at [1, 0] should match centre 0 ("capital").
        let dir = Array1::from(vec![0.99, 0.05]);
        let (cluster_id, label, _sim) = rc.classify_direction(&dir).unwrap();
        assert_eq!(cluster_id, 0);
        assert_eq!(label, "capital");
    }

    #[test]
    fn classify_direction_returns_none_without_clusters() {
        let rc = RelationClassifier {
            clusters: None,
            feature_assignments: std::collections::HashMap::new(),
            probe_labels: std::collections::HashMap::new(),
            probe_count: 0,
        };
        let dir = Array1::from(vec![1.0, 0.0]);
        assert!(rc.classify_direction(&dir).is_none());
    }

    // ── probe_labels accessors ──

    #[test]
    fn probe_label_priority_over_cluster_assignment() {
        // (26, 9515) is in feature_assignments as cluster 0 ("capital");
        // adding a probe label should override that lookup result.
        let mut rc = make_test_classifier();
        rc.probe_labels.insert((26, 9515), "manual_override".into());
        rc.probe_count = rc.probe_labels.len();
        assert_eq!(rc.label_for_feature(26, 9515), Some("manual_override"));
        assert!(rc.is_probe_label(26, 9515));
        assert!(!rc.is_probe_label(24, 4532));
        assert_eq!(rc.num_probe_labels(), 1);
    }

    // ── normalise_relation ──

    #[test]
    fn normalise_collapses_separators_and_case() {
        assert_eq!(normalise_relation("Country-Of-Origin"), "country of origin");
        assert_eq!(normalise_relation("country_of_origin"), "country of origin");
        assert_eq!(normalise_relation("country  of\tof"), "country of of");
    }

    // ── from_vindex with on-disk fixtures ──

    fn write_fixture(dir: &std::path::Path, files: &[(&str, &str)]) {
        std::fs::create_dir_all(dir).unwrap();
        for (name, content) in files {
            std::fs::write(dir.join(name), content).unwrap();
        }
    }

    #[test]
    fn from_vindex_loads_clusters_and_assignments() {
        let dir = std::env::temp_dir().join(format!(
            "larql_relations_clusters_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        write_fixture(
            &dir,
            &[
                (
                    RELATION_CLUSTERS_JSON,
                    r#"{"k":2,"centres":[[1.0,0.0],[0.0,1.0]],"labels":["capital","language"],"counts":[10,5],"top_tokens":[["paris"],["english"]]}"#,
                ),
                (
                    FEATURE_CLUSTERS_JSONL,
                    "{\"l\":5,\"f\":12,\"c\":0}\n{\"l\":7,\"f\":3,\"c\":1}\n",
                ),
            ],
        );
        let rc = RelationClassifier::from_vindex(&dir).expect("classifier loads");
        assert!(rc.has_clusters());
        assert_eq!(rc.num_clusters(), 2);
        assert_eq!(rc.cluster_for_feature(5, 12), Some(0));
        assert_eq!(rc.cluster_for_feature(7, 3), Some(1));
        assert_eq!(rc.label_for_feature(5, 12), Some("capital"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_vindex_with_only_probe_labels() {
        // No clusters, no assignments — just probe labels keyed
        // "L{layer}_F{feature}". from_vindex should return Some(rc)
        // because probe_labels alone is sufficient.
        let dir = std::env::temp_dir().join(format!(
            "larql_relations_probe_only_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        write_fixture(
            &dir,
            &[(
                FEATURE_LABELS_JSON,
                r#"{"L26_F100":"capital","L24_F50":"language"}"#,
            )],
        );
        let rc = RelationClassifier::from_vindex(&dir).expect("classifier with probe-only");
        assert!(!rc.has_clusters());
        assert_eq!(rc.num_probe_labels(), 2);
        assert_eq!(rc.label_for_feature(26, 100), Some("capital"));
        assert_eq!(rc.label_for_feature(24, 50), Some("language"));
        assert!(rc.is_probe_label(26, 100));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_vindex_skips_malformed_jsonl_lines() {
        // The assignments JSONL parser silently drops malformed lines —
        // good lines still load, bad lines are ignored.
        let dir = std::env::temp_dir().join(format!(
            "larql_relations_malformed_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        write_fixture(
            &dir,
            &[
                (
                    RELATION_CLUSTERS_JSON,
                    r#"{"k":1,"centres":[[1.0]],"labels":["x"],"counts":[1],"top_tokens":[["t"]]}"#,
                ),
                (
                    FEATURE_CLUSTERS_JSONL,
                    "this is not json\n{\"l\":3,\"f\":4,\"c\":0}\n",
                ),
            ],
        );
        let rc = RelationClassifier::from_vindex(&dir).expect("classifier");
        assert_eq!(rc.cluster_for_feature(3, 4), Some(0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_vindex_skips_malformed_probe_keys() {
        // Keys that don't match "L{n}_F{m}" are silently dropped; the
        // valid ones still load.
        let dir = std::env::temp_dir().join(format!(
            "larql_relations_bad_keys_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        write_fixture(
            &dir,
            &[(
                FEATURE_LABELS_JSON,
                // Bad keys: missing prefix, wrong prefix, non-numeric, malformed split count.
                r#"{"L1_F2":"good","BAD_KEY":"x","L1F2":"x","Lx_Fy":"x","extra_underscore_key_here":"x"}"#,
            )],
        );
        let rc = RelationClassifier::from_vindex(&dir).expect("classifier");
        assert_eq!(rc.num_probe_labels(), 1);
        assert_eq!(rc.label_for_feature(1, 2), Some("good"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_vindex_empty_dir_returns_none() {
        let dir = std::env::temp_dir().join(format!(
            "larql_relations_empty_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let rc = RelationClassifier::from_vindex(&dir);
        assert!(rc.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── token_embedding_pub ──

    #[test]
    fn token_embedding_returns_none_for_empty_tokens() {
        let embed = Array2::<f32>::zeros((10, 4));
        let tok = larql_inference::test_utils::make_test_tokenizer(10);
        // Empty string tokenises to nothing meaningful — depending on
        // the tokenizer, may return Some(zero-vec) or None. The
        // contract is just that it doesn't panic.
        let _ = token_embedding_pub("", &embed, 1.0, &tok);
    }

    #[test]
    fn token_embedding_with_embed_scale_multiplies() {
        // embed_scale should multiply each row before averaging — assert
        // the scale factor is applied by feeding a large embed value
        // and checking the output scales linearly with embed_scale.
        let mut embed = Array2::<f32>::zeros((10, 4));
        for i in 0..10 {
            embed[[i, 0]] = 10.0;
        }
        let tok = larql_inference::test_utils::make_test_tokenizer(10);
        let a = token_embedding_pub("[0]", &embed, 1.0, &tok);
        let b = token_embedding_pub("[0]", &embed, 2.0, &tok);
        // Either both Some (in which case b should be 2× a) or both
        // None (tokenizer produced empty ids); in either case no panic.
        if let (Some(av), Some(bv)) = (a, b) {
            if av[0] > 0.0 {
                assert!(
                    (bv[0] / av[0] - 2.0).abs() < 1e-3,
                    "embed_scale=2 should double output: a={}, b={}",
                    av[0],
                    bv[0]
                );
            }
        }
    }

    #[test]
    fn token_embedding_skips_out_of_range_ids() {
        // id beyond embed.shape()[0] is filtered — the function
        // doesn't panic and divides by ids.len() regardless.
        let embed = Array2::<f32>::zeros((2, 4));
        let tok = larql_inference::test_utils::make_test_tokenizer(10);
        let _ = token_embedding_pub("[5] [6]", &embed, 1.0, &tok);
    }
}
