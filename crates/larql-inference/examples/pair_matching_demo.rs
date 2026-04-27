//! Pair-based relation matching demo.
//!
//! Demonstrates matching (input, output) token pairs against
//! Wikidata triples and WordNet relations to label clusters.
//!
//! Run: cargo run -p larql-inference --example pair_matching_demo

use larql_vindex::clustering::pair_matching::{
    label_clusters_from_pairs, load_reference_databases, RelationDatabase,
};

fn main() {
    println!("=== Pair-Based Relation Matching Demo ===\n");

    // ── Load reference databases ──
    section("Reference Databases");

    let ref_dbs = load_reference_databases();
    let mut dbs: Vec<&RelationDatabase> = Vec::new();
    if let Some(ref wk) = ref_dbs.wikidata {
        println!("  Wikidata: {} relations, {} pairs", wk.num_relations(), wk.num_pairs());
        dbs.push(wk);
    }
    if let Some(ref wn) = ref_dbs.wordnet {
        println!("  WordNet: {} relations, {} pairs", wn.num_relations(), wn.num_pairs());
        dbs.push(wn);
    }
    if dbs.is_empty() {
        println!("  No reference databases found in data/");
        println!("  Run: python3 scripts/assemble_triples.py");
        println!("  Run: python3 scripts/fetch_wordnet_relations.py");
        println!("\n  Falling back to built-in test data...\n");
        run_with_builtin();
        return;
    }

    // ── Test individual lookups ──
    section("Individual Lookups");

    let test_pairs = vec![
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("France", "French"),
        ("Kenya", "Nairobi"),
        ("Kenya", "Africa"),
        ("big", "large"),
        ("happy", "glad"),
        ("dog", "animal"),
        ("read", "reading"),
        ("France", "Berlin"),  // wrong
        ("xyz", "abc"),        // not in any DB
    ];

    for (subject, object) in &test_pairs {
        let mut matches = Vec::new();
        for db in &dbs {
            matches.extend(db.lookup(subject, object));
        }
        if matches.is_empty() {
            println!("  {:15} → {:15} (no match)", subject, object);
        } else {
            println!("  {:15} → {:15} → {}", subject, object, matches.join(", "));
        }
    }

    // ── Test cluster labeling with realistic pairs ──
    section("Cluster Labeling");

    // Simulate 3 clusters with realistic (input, output) pairs
    // Cluster 0: capital-like (country → city)
    // Cluster 1: language-like (country → language)
    // Cluster 2: random/unknown
    let assignments = vec![
        0, 0, 0, 0, 0,    // cluster 0
        1, 1, 1, 1, 1,    // cluster 1
        2, 2, 2, 2, 2,    // cluster 2
    ];

    let inputs: Vec<String> = vec![
        // Cluster 0: countries
        "France", "Germany", "Japan", "Kenya", "Brazil",
        // Cluster 1: countries
        "France", "Germany", "Japan", "Kenya", "Brazil",
        // Cluster 2: random
        "table", "running", "blue", "quickly", "seven",
    ].into_iter().map(Into::into).collect();

    let outputs: Vec<String> = vec![
        // Cluster 0: capitals
        "Paris", "Berlin", "Tokyo", "Nairobi", "Brasília",
        // Cluster 1: languages
        "French", "German", "Japanese", "Swahili", "Portuguese",
        // Cluster 2: random
        "chair", "jogging", "red", "slowly", "eight",
    ].into_iter().map(Into::into).collect();

    let labels = label_clusters_from_pairs(
        &assignments, &inputs, &outputs, 3, &dbs,
    );

    println!("  Results:");
    for (i, label) in labels.iter().enumerate() {
        let label_str = label.as_deref().unwrap_or("(unlabeled)");
        let sample_pairs: Vec<String> = (0..3)
            .map(|j| {
                let idx = i * 5 + j;
                format!("{}→{}", inputs[idx], outputs[idx])
            })
            .collect();
        println!("    Cluster {}: {:<25} [{}]", i, label_str, sample_pairs.join(", "));
    }

    // ── Show what the Wikidata DB contains ──
    section("Wikidata Coverage");

    // Try to find the Wikidata DB specifically
    for db in &dbs {
        if db.num_relations() > 20 {
            // This is likely Wikidata (more relations than WordNet)
            println!("  Top Wikidata relations by pair count:");
            // We can't iterate relations directly, but we can test known ones
            let test_rels = vec![
                ("France", "Paris", "capital?"),
                ("France", "French", "language?"),
                ("Kenya", "Africa", "continent?"),
                ("Kenya", "Nairobi", "capital?"),
                ("Michelle Bachelet", "Chile", "citizenship?"),
            ];
            for (s, o, expected) in test_rels {
                let rels = db.lookup(s, o);
                if rels.is_empty() {
                    println!("    {:20} → {:20} (no match) {}", s, o, expected);
                } else {
                    println!("    {:20} → {:20} → {} {}", s, o, rels.join(", "), expected);
                }
            }
        }
    }

    println!("\n=== Done ===");
}

fn run_with_builtin() {
    section("Built-in Test Data");

    let mut db = RelationDatabase::default();
    db.add_relation("capital", vec![
        ("france".into(), "paris".into()),
        ("germany".into(), "berlin".into()),
        ("japan".into(), "tokyo".into()),
        ("italy".into(), "rome".into()),
        ("spain".into(), "madrid".into()),
        ("kenya".into(), "nairobi".into()),
    ]);
    db.add_relation("official language", vec![
        ("france".into(), "french".into()),
        ("germany".into(), "german".into()),
        ("japan".into(), "japanese".into()),
        ("spain".into(), "spanish".into()),
        ("kenya".into(), "swahili".into()),
    ]);
    db.add_relation("continent", vec![
        ("france".into(), "europe".into()),
        ("japan".into(), "asia".into()),
        ("kenya".into(), "africa".into()),
        ("brazil".into(), "south america".into()),
    ]);
    db.add_relation("synonym", vec![
        ("big".into(), "large".into()),
        ("fast".into(), "quick".into()),
        ("happy".into(), "glad".into()),
        ("small".into(), "tiny".into()),
    ]);

    // Test lookups
    println!("  Lookups:");
    let tests = vec![
        ("France", "Paris"), ("France", "French"), ("Kenya", "Africa"),
        ("big", "large"), ("France", "Berlin"),
    ];
    for (s, o) in tests {
        let rels = db.lookup(s, o);
        if rels.is_empty() {
            println!("    {} → {} : (no match)", s, o);
        } else {
            println!("    {} → {} : {}", s, o, rels.join(", "));
        }
    }

    // Test cluster labeling
    let assignments = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2];
    let inputs: Vec<String> = vec![
        "France", "Germany", "Japan", "Italy", "Spain",
        "France", "Germany", "Japan", "Spain", "Kenya",
        "big", "fast", "happy", "small",
    ].into_iter().map(Into::into).collect();
    let outputs: Vec<String> = vec![
        "Paris", "Berlin", "Tokyo", "Rome", "Madrid",
        "French", "German", "Japanese", "Spanish", "Swahili",
        "large", "quick", "glad", "tiny",
    ].into_iter().map(Into::into).collect();

    let labels = label_clusters_from_pairs(
        &assignments, &inputs, &outputs, 3, &[&db],
    );

    println!("\n  Cluster labels:");
    let cluster_names = ["capitals", "languages", "synonyms"];
    for (i, label) in labels.iter().enumerate() {
        let label_str = label.as_deref().unwrap_or("(unlabeled)");
        let expected = cluster_names.get(i).unwrap_or(&"?");
        let status = if label.is_some() { "OK" } else { "MISS" };
        println!("    Cluster {} ({}): {:<25} {}", i, expected, label_str, status);
    }

    println!("\n=== Done ===");
}

fn section(name: &str) {
    println!("── {} ──\n", name);
}
