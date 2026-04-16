//! Clustering and relation discovery demo.
//!
//! Demonstrates the clustering pipeline: k-means, labeling, pair matching.
//! Uses synthetic data to show how the pipeline works end-to-end.
//!
//! Run: cargo run -p larql-inference --example clustering_demo

use larql_vindex::clustering::{
    kmeans,
    pair_matching::{label_clusters_from_pairs, RelationDatabase},
};
use larql_vindex::clustering::labeling::detect_entity_pattern;

fn main() {
    println!("=== Clustering & Relation Discovery Demo ===\n");

    // ── K-means clustering ──
    section("K-means Clustering");

    // Create synthetic 2D data with 3 clear clusters
    let data = ndarray::Array2::from_shape_vec(
        (9, 2),
        vec![
            // Cluster 0: rightward
            1.0, 0.1, 0.9, 0.2, 0.95, 0.05,
            // Cluster 1: upward
            0.1, 1.0, 0.2, 0.9, 0.05, 0.95,
            // Cluster 2: diagonal
            0.7, 0.7, 0.6, 0.8, 0.8, 0.6,
        ],
    )
    .unwrap();

    let (centres, assignments, distances) = kmeans(&data, 3, 100);

    println!("  Input: 9 points in 2D, k=3");
    println!("  Centres: {:?}", centres.shape());
    println!("  Assignments: {:?}", assignments);
    println!(
        "  Max distance: {:.4}",
        distances.iter().cloned().fold(0.0f32, f32::max)
    );

    // Verify clusters are correct
    assert_eq!(assignments[0], assignments[1]);
    assert_eq!(assignments[1], assignments[2]);
    assert_eq!(assignments[3], assignments[4]);
    assert_ne!(assignments[0], assignments[3]);
    println!("  Cluster integrity: OK\n");

    // ── Entity pattern detection ──
    section("Entity Pattern Detection");

    let patterns = vec![
        (
            vec!["australia", "italy", "germany", "france", "japan"],
            "country",
        ),
        (
            vec!["english", "french", "german", "spanish", "italian"],
            "language",
        ),
        (
            vec!["january", "february", "march", "october", "november"],
            "month",
        ),
        (
            vec!["one", "two", "three", "four", "five"],
            "number",
        ),
        (
            vec!["ing", "tion", "ness", "ment"],
            "morphological",
        ),
        (
            vec!["Paris", "music", "running", "table"],
            "(none)",
        ),
    ];

    for (members, expected) in &patterns {
        let members: Vec<String> = members.iter().map(|s| s.to_string()).collect();
        let result = detect_entity_pattern(&members)
            .unwrap_or_else(|| "(none)".into());
        let status = if result == *expected { "OK" } else { "MISMATCH" };
        println!(
            "  {:40} → {:<15} {}",
            format!("[{}]", members.join(", ")),
            result,
            status,
        );
    }

    // ── Pair-based matching ──
    section("Pair-Based Relation Matching");

    // Create a reference database
    let mut db = RelationDatabase::default();

    // Add some Wikidata-style relations
    db.add_relation("capital", vec![
        ("france".into(), "paris".into()),
        ("germany".into(), "berlin".into()),
        ("japan".into(), "tokyo".into()),
        ("italy".into(), "rome".into()),
        ("spain".into(), "madrid".into()),
    ]);
    db.add_relation("language", vec![
        ("france".into(), "french".into()),
        ("germany".into(), "german".into()),
        ("japan".into(), "japanese".into()),
        ("italy".into(), "italian".into()),
        ("spain".into(), "spanish".into()),
    ]);
    db.add_relation("synonym", vec![
        ("big".into(), "large".into()),
        ("fast".into(), "quick".into()),
        ("happy".into(), "glad".into()),
        ("small".into(), "tiny".into()),
    ]);

    println!("  Database: {} relations, {} pairs",
        db.num_relations(), db.num_pairs());

    // Simulate cluster features with (input, output) pairs
    // Cluster 0: capital features, Cluster 1: language features, Cluster 2: synonyms
    let assignments = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2];
    let inputs: Vec<String> = vec![
        "France", "Germany", "Japan", "Italy", "Spain",
        "France", "Germany", "Japan", "Italy", "Spain",
        "big", "fast", "happy", "small",
    ].into_iter().map(Into::into).collect();
    let outputs: Vec<String> = vec![
        "Paris", "Berlin", "Tokyo", "Rome", "Madrid",
        "French", "German", "Japanese", "Italian", "Spanish",
        "large", "quick", "glad", "tiny",
    ].into_iter().map(Into::into).collect();

    let labels = label_clusters_from_pairs(
        &assignments, &inputs, &outputs, 3, &[&db],
    );

    println!("\n  Cluster labeling results:");
    for (i, label) in labels.iter().enumerate() {
        let label_str = label.as_deref().unwrap_or("(unlabeled)");
        let members: Vec<&str> = assignments.iter().enumerate()
            .filter(|(_, &c)| c == i)
            .take(3)
            .map(|(j, _)| outputs[j].as_str())
            .collect();
        println!("    Cluster {}: {} → [{}]", i, label_str, members.join(", "));
    }

    assert_eq!(labels[0], Some("capital".to_string()));
    assert_eq!(labels[1], Some("language".to_string()));
    assert_eq!(labels[2], Some("synonym".to_string()));
    println!("  All labels correct!\n");

    // ── Lookup examples ──
    section("Reference Database Lookup");

    let lookups = vec![
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("France", "French"),
        ("big", "large"),
        ("France", "Berlin"),  // wrong pair
        ("dog", "cat"),        // not in database
    ];

    for (subject, object) in lookups {
        let rels = db.lookup(subject, object);
        if rels.is_empty() {
            println!("  {} → {} : (no match)", subject, object);
        } else {
            println!("  {} → {} : {}", subject, object, rels.join(", "));
        }
    }

    println!("\n=== Done ===");
}

fn section(name: &str) {
    println!("── {} ──\n", name);
}
