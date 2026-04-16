//! Cluster-labeling algorithms — match (input, output) or (output-only)
//! token pairs against reference databases (`super::database`) and
//! pick the winning relation per cluster.

use std::collections::HashMap;

use super::database::RelationDatabase;

/// Label clusters by matching (input, output) token pairs against reference databases.
///
/// For each cluster:
/// 1. Collect all (input_token, output_token) string pairs from its features
/// 2. Look up each pair in the reference databases
/// 3. Count matches per relation type
/// 4. The relation type with the most matches = the cluster label
///
/// Returns labels for all k clusters. Unlabeled clusters get None.
pub fn label_clusters_from_pairs(
    assignments: &[usize],
    input_tokens: &[String],
    output_tokens: &[String],
    k: usize,
    databases: &[&RelationDatabase],
) -> Vec<Option<String>> {
    // Group (input, output) pairs by cluster
    let mut cluster_pairs: Vec<Vec<(&str, &str)>> = vec![Vec::new(); k];

    for (i, &cluster) in assignments.iter().enumerate() {
        if cluster < k && i < input_tokens.len() && i < output_tokens.len() {
            let inp = input_tokens[i].as_str();
            let out = output_tokens[i].as_str();
            if !inp.is_empty() && !out.is_empty() {
                cluster_pairs[cluster].push((inp, out));
            }
        }
    }

    // For each cluster, find the best matching relation
    let mut labels = vec![None; k];

    for c in 0..k {
        let pairs = &cluster_pairs[c];
        if pairs.is_empty() {
            continue;
        }

        // Count matches per relation type across all databases
        let mut relation_counts: HashMap<String, usize> = HashMap::new();

        for &(inp, out) in pairs {
            for db in databases {
                for rel in db.lookup(inp, out) {
                    *relation_counts.entry(rel.to_string()).or_default() += 1;
                }
            }
        }

        // Find the best relation (most matches)
        if let Some((best_rel, best_count)) = relation_counts
            .iter()
            .max_by_key(|(_, &count)| count)
        {
            // Require at least 2 matches or 10% of the cluster's pairs
            let threshold = 2.max(pairs.len() / 10);
            if *best_count >= threshold {
                labels[c] = Some(best_rel.clone());
            }
        }
    }

    labels
}

/// Label clusters by matching OUTPUT tokens only against reference database objects.
///
/// For each cluster, collect the output tokens from its features. Check which
/// relation has the most matching objects. This doesn't need the gate input token.
///
/// The caller should pass the appropriate databases for the layer range:
/// - L14-27 features: Wikidata only
/// - L0-13 features: WordNet only
pub fn label_clusters_from_outputs(
    assignments: &[usize],
    output_tokens: &[String],
    k: usize,
    databases: &[&RelationDatabase],
) -> Vec<Option<String>> {
    // Build inverted index: object_lower → relation_names
    let mut object_to_relations: HashMap<String, Vec<String>> = HashMap::new();
    for db in databases {
        for (rel_name, pairs) in db.relations_iter() {
            for (_, obj) in pairs {
                object_to_relations
                    .entry(obj.clone())
                    .or_default()
                    .push(rel_name.to_string());
            }
        }
    }

    // Group output tokens by cluster
    let mut cluster_outputs: Vec<Vec<&str>> = vec![Vec::new(); k];
    for (i, &cluster) in assignments.iter().enumerate() {
        if cluster < k && i < output_tokens.len() {
            let out = output_tokens[i].trim();
            if !out.is_empty() {
                cluster_outputs[cluster].push(out);
            }
        }
    }

    // For each cluster, count which relations match its output tokens
    let mut labels = vec![None; k];

    for c in 0..k {
        let outputs = &cluster_outputs[c];
        if outputs.is_empty() {
            continue;
        }

        let mut relation_counts: HashMap<String, usize> = HashMap::new();

        for &out in outputs {
            let key = out.to_lowercase();
            if let Some(rels) = object_to_relations.get(&key) {
                for rel in rels {
                    *relation_counts.entry(rel.clone()).or_default() += 1;
                }
            }
        }

        // Find top two relations by match count
        let mut sorted_rels: Vec<(&String, &usize)> = relation_counts.iter().collect();
        sorted_rels.sort_by(|a, b| b.1.cmp(a.1));

        if let Some(&(best_rel, &best_count)) = sorted_rels.first() {
            let second_count = sorted_rels.get(1).map(|&(_, &c)| c).unwrap_or(0);

            // Require: at least 2 matches AND clear separation from runner-up.
            // The best relation must have 2x the matches of the second-best.
            // This prevents "occupation" from winning when it ties with "language".
            if best_count >= 2 && (second_count == 0 || best_count >= second_count * 2) {
                labels[c] = Some(best_rel.clone());
            }
        }
    }

    labels
}
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::database::RelationDatabase;


    #[test]
    fn test_lookup() {
        let mut db = RelationDatabase::default();
        db.relations.insert(
            "capital".to_string(),
            vec![
                ("france".to_string(), "paris".to_string()),
                ("germany".to_string(), "berlin".to_string()),
            ],
        );
        db.build_index();

        assert_eq!(db.lookup("France", "Paris"), vec!["capital"]);
        assert_eq!(db.lookup("Germany", "Berlin"), vec!["capital"]);
        assert!(db.lookup("France", "Berlin").is_empty());
    }

    #[test]
    fn test_label_clusters() {
        let mut db = RelationDatabase::default();
        db.relations.insert(
            "capital".to_string(),
            vec![
                ("france".to_string(), "paris".to_string()),
                ("germany".to_string(), "berlin".to_string()),
                ("japan".to_string(), "tokyo".to_string()),
            ],
        );
        db.build_index();

        let assignments = vec![0, 0, 0, 1, 1];
        let inputs = vec![
            "France".into(), "Germany".into(), "Japan".into(),
            "dog".into(), "cat".into(),
        ];
        let outputs = vec![
            "Paris".into(), "Berlin".into(), "Tokyo".into(),
            "bark".into(), "meow".into(),
        ];

        let labels = label_clusters_from_pairs(
            &assignments, &inputs, &outputs, 2, &[&db],
        );

        assert_eq!(labels[0], Some("capital".to_string()));
        assert_eq!(labels[1], None); // no matches
    }

    #[test]
    fn test_multiple_databases() {
        let mut db1 = RelationDatabase::default();
        db1.relations.insert(
            "capital".to_string(),
            vec![("france".to_string(), "paris".to_string())],
        );
        db1.build_index();

        let mut db2 = RelationDatabase::default();
        db2.relations.insert(
            "synonym".to_string(),
            vec![("big".to_string(), "large".to_string())],
        );
        db2.build_index();

        let assignments = vec![0, 1];
        let inputs = vec!["France".into(), "big".into()];
        let outputs = vec!["Paris".into(), "large".into()];

        let labels = label_clusters_from_pairs(
            &assignments, &inputs, &outputs, 2, &[&db1, &db2],
        );

        // Both should fail threshold (only 1 match each, need 2)
        // But the algorithm requires max(2, len/10)
        assert_eq!(labels[0], None); // 1 match < threshold of 2
        assert_eq!(labels[1], None);
    }

    #[test]
    fn test_threshold_met() {
        let mut db = RelationDatabase::default();
        db.relations.insert(
            "capital".to_string(),
            vec![
                ("france".to_string(), "paris".to_string()),
                ("germany".to_string(), "berlin".to_string()),
                ("japan".to_string(), "tokyo".to_string()),
                ("italy".to_string(), "rome".to_string()),
                ("spain".to_string(), "madrid".to_string()),
            ],
        );
        db.build_index();

        // All 5 in one cluster — should hit threshold
        let assignments = vec![0, 0, 0, 0, 0];
        let inputs: Vec<String> = vec!["France", "Germany", "Japan", "Italy", "Spain"]
            .into_iter().map(Into::into).collect();
        let outputs: Vec<String> = vec!["Paris", "Berlin", "Tokyo", "Rome", "Madrid"]
            .into_iter().map(Into::into).collect();

        let labels = label_clusters_from_pairs(
            &assignments, &inputs, &outputs, 1, &[&db],
        );

        assert_eq!(labels[0], Some("capital".to_string()));
    }

    #[test]
    fn test_case_insensitive_lookup() {
        let mut db = RelationDatabase::default();
        db.relations.insert(
            "capital".to_string(),
            vec![("france".to_string(), "paris".to_string())],
        );
        db.build_index();

        // Case shouldn't matter
        assert_eq!(db.lookup("FRANCE", "PARIS"), vec!["capital"]);
        assert_eq!(db.lookup("france", "paris"), vec!["capital"]);
        assert_eq!(db.lookup("France", "Paris"), vec!["capital"]);
    }

    #[test]
    fn test_empty_database() {
        let db = RelationDatabase::default();
        assert!(db.lookup("anything", "anything").is_empty());
        assert_eq!(db.num_relations(), 0);
        assert_eq!(db.num_pairs(), 0);
    }

    #[test]
    fn test_empty_cluster_pairs() {
        let db = RelationDatabase::default();
        let labels = label_clusters_from_pairs(
            &[], &[], &[], 3, &[&db],
        );
        assert_eq!(labels.len(), 3);
        assert!(labels.iter().all(|l| l.is_none()));
    }

    #[test]
    fn test_add_relation() {
        let mut db = RelationDatabase::default();
        db.add_relation("capital", vec![
            ("france".into(), "paris".into()),
            ("germany".into(), "berlin".into()),
        ]);
        assert_eq!(db.num_relations(), 1);
        assert_eq!(db.num_pairs(), 2);
        assert_eq!(db.lookup("France", "Paris"), vec!["capital"]);
    }

    #[test]
    fn test_multiple_relations_same_pair() {
        let mut db = RelationDatabase::default();
        db.add_relation("capital", vec![
            ("france".into(), "paris".into()),
        ]);
        db.add_relation("largest_city", vec![
            ("france".into(), "paris".into()),
        ]);
        let rels = db.lookup("France", "Paris");
        assert!(rels.contains(&"capital"));
        assert!(rels.contains(&"largest_city"));
    }

    #[test]
    fn test_realistic_wikidata_pairs() {
        // Simulate realistic Wikidata data
        let mut db = RelationDatabase::default();
        db.add_relation("capital", vec![
            ("france".into(), "paris".into()),
            ("germany".into(), "berlin".into()),
            ("japan".into(), "tokyo".into()),
            ("kenya".into(), "nairobi".into()),
            ("people's republic of china".into(), "beijing".into()),
        ]);
        db.add_relation("official language", vec![
            ("france".into(), "french".into()),
            ("germany".into(), "german".into()),
            ("japan".into(), "japanese".into()),
            ("kenya".into(), "swahili".into()),
        ]);
        db.add_relation("continent", vec![
            ("france".into(), "europe".into()),
            ("japan".into(), "asia".into()),
            ("kenya".into(), "africa".into()),
        ]);

        // Cluster 0: capital-type features
        // Cluster 1: language-type features
        // Cluster 2: continent-type features
        let assignments = vec![
            0, 0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2,
        ];
        let inputs: Vec<String> = vec![
            "France", "Germany", "Japan", "Kenya", "People's Republic of China",
            "France", "Germany", "Japan", "Kenya",
            "France", "Japan", "Kenya",
        ].into_iter().map(Into::into).collect();
        let outputs: Vec<String> = vec![
            "Paris", "Berlin", "Tokyo", "Nairobi", "Beijing",
            "French", "German", "Japanese", "Swahili",
            "Europe", "Asia", "Africa",
        ].into_iter().map(Into::into).collect();

        let labels = label_clusters_from_pairs(
            &assignments, &inputs, &outputs, 3, &[&db],
        );

        assert_eq!(labels[0], Some("capital".to_string()));
        assert_eq!(labels[1], Some("official language".to_string()));
        assert_eq!(labels[2], Some("continent".to_string()));
    }

    #[test]
    fn test_wordnet_synonym_matching() {
        let mut db = RelationDatabase::default();
        db.add_relation("synonym", vec![
            ("big".into(), "large".into()),
            ("fast".into(), "quick".into()),
            ("happy".into(), "glad".into()),
            ("small".into(), "tiny".into()),
            ("hot".into(), "warm".into()),
        ]);

        let assignments = vec![0, 0, 0, 0, 0];
        let inputs: Vec<String> = vec!["big", "fast", "happy", "small", "hot"]
            .into_iter().map(Into::into).collect();
        let outputs: Vec<String> = vec!["large", "quick", "glad", "tiny", "warm"]
            .into_iter().map(Into::into).collect();

        let labels = label_clusters_from_pairs(
            &assignments, &inputs, &outputs, 1, &[&db],
        );

        assert_eq!(labels[0], Some("synonym".to_string()));
    }

    #[test]
    fn test_mixed_databases() {
        // Wikidata
        let mut wikidata = RelationDatabase::default();
        wikidata.add_relation("capital", vec![
            ("france".into(), "paris".into()),
            ("germany".into(), "berlin".into()),
            ("japan".into(), "tokyo".into()),
        ]);

        // WordNet
        let mut wordnet = RelationDatabase::default();
        wordnet.add_relation("synonym", vec![
            ("big".into(), "large".into()),
            ("fast".into(), "quick".into()),
            ("happy".into(), "glad".into()),
        ]);

        // Two clusters: one from Wikidata, one from WordNet
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let inputs: Vec<String> = vec![
            "France", "Germany", "Japan",
            "big", "fast", "happy",
        ].into_iter().map(Into::into).collect();
        let outputs: Vec<String> = vec![
            "Paris", "Berlin", "Tokyo",
            "large", "quick", "glad",
        ].into_iter().map(Into::into).collect();

        let labels = label_clusters_from_pairs(
            &assignments, &inputs, &outputs, 2, &[&wikidata, &wordnet],
        );

        assert_eq!(labels[0], Some("capital".to_string()));
        assert_eq!(labels[1], Some("synonym".to_string()));
    }

    #[test]
    fn test_partial_matches() {
        // Cluster has 10 features, only 3 match Wikidata
        let mut db = RelationDatabase::default();
        db.add_relation("capital", vec![
            ("france".into(), "paris".into()),
            ("germany".into(), "berlin".into()),
            ("japan".into(), "tokyo".into()),
        ]);

        let assignments = vec![0; 10];
        let inputs: Vec<String> = vec![
            "France", "Germany", "Japan",  // 3 matches
            "dog", "cat", "house", "tree", "book", "water", "fire",  // 7 non-matches
        ].into_iter().map(Into::into).collect();
        let outputs: Vec<String> = vec![
            "Paris", "Berlin", "Tokyo",
            "bark", "meow", "roof", "leaf", "page", "ocean", "flame",
        ].into_iter().map(Into::into).collect();

        let labels = label_clusters_from_pairs(
            &assignments, &inputs, &outputs, 1, &[&db],
        );

        // 3 matches >= threshold (max(2, 10/10)=2), so should label
        assert_eq!(labels[0], Some("capital".to_string()));
    }
}
