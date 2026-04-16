//! Reference databases for pair-based relation labeling.
//!
//! Holds the `RelationDatabase` data type (a name → (subject, object)
//! pair set), the loaders for Wikidata + WordNet, and the bundled
//! `ReferenceDatabases` struct returned by `load_reference_databases`.
//!
//! Consumed by `super::labeling`.

use std::collections::HashMap;
use std::path::Path;

/// A reference database of (subject, object) pairs per relation type.
#[derive(Default)]
pub struct RelationDatabase {
    /// relation_name → set of (subject_lower, object_lower) pairs.
    /// `pub(super)` so the test module in `super::labeling` can drive
    /// it directly without going through `add_relation` for every case.
    pub(super) relations: HashMap<String, Vec<(String, String)>>,
    /// Inverted index: (subject_lower, object_lower) → relation_names
    pair_index: HashMap<(String, String), Vec<String>>,
}

impl RelationDatabase {
    /// Add a relation with its (subject, object) pairs.
    pub fn add_relation(&mut self, name: &str, pairs: Vec<(String, String)>) {
        self.relations.insert(name.to_string(), pairs);
        self.rebuild_index();
    }

    fn rebuild_index(&mut self) {
        self.pair_index.clear();
        for (rel_name, pairs) in &self.relations {
            for (s, o) in pairs {
                self.pair_index
                    .entry((s.clone(), o.clone()))
                    .or_default()
                    .push(rel_name.clone());
            }
        }
    }

    /// Load from Wikidata triples JSON file.
    pub fn load_wikidata(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let data: serde_json::Value = serde_json::from_str(&text).ok()?;
        let obj = data.as_object()?;

        let mut db = Self::default();

        for (label, value) in obj {
            if let Some(pairs) = value.get("pairs").and_then(|v| v.as_array()) {
                let mut rel_pairs = Vec::new();
                for pair in pairs {
                    if let Some(arr) = pair.as_array() {
                        if arr.len() >= 2 {
                            let s = arr[0].as_str().unwrap_or("").to_lowercase();
                            let o = arr[1].as_str().unwrap_or("").to_lowercase();
                            if !s.is_empty() && !o.is_empty() {
                                rel_pairs.push((s, o));
                            }
                        }
                    }
                }
                db.relations.insert(label.clone(), rel_pairs);
            }
        }

        db.build_index();
        Some(db)
    }

    /// Load from WordNet relations JSON file.
    pub fn load_wordnet(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let data: serde_json::Value = serde_json::from_str(&text).ok()?;
        let obj = data.as_object()?;

        let mut db = Self::default();

        for (label, value) in obj {
            if let Some(pairs) = value.get("pairs").and_then(|v| v.as_array()) {
                let mut rel_pairs = Vec::new();
                for pair in pairs {
                    if let Some(arr) = pair.as_array() {
                        if arr.len() >= 2 {
                            let s = arr[0].as_str().unwrap_or("").to_lowercase();
                            let o = arr[1].as_str().unwrap_or("").to_lowercase();
                            if !s.is_empty() && !o.is_empty() {
                                rel_pairs.push((s, o));
                            }
                        }
                    }
                }
                db.relations.insert(label.clone(), rel_pairs);
            }
        }

        db.build_index();
        Some(db)
    }

    pub(super) fn build_index(&mut self) {
        self.pair_index.clear();
        for (rel_name, pairs) in &self.relations {
            for (s, o) in pairs {
                self.pair_index
                    .entry((s.clone(), o.clone()))
                    .or_default()
                    .push(rel_name.clone());
            }
        }
    }

    /// Look up which relations contain this (subject, object) pair.
    pub fn lookup(&self, subject: &str, object: &str) -> Vec<&str> {
        let key = (subject.to_lowercase(), object.to_lowercase());
        self.pair_index
            .get(&key)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Number of relation types loaded.
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }

    /// Total number of pairs across all relations.
    pub fn num_pairs(&self) -> usize {
        self.relations.values().map(|v| v.len()).sum()
    }

    /// Iterate all relations and their (subject, object) pairs.
    /// Used by `super::labeling` to build inverted indexes for
    /// output-only matching.
    pub fn relations_iter(&self) -> impl Iterator<Item = (&str, &[(String, String)])> {
        self.relations.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }
}
/// Loaded reference databases, separated by layer range.
pub struct ReferenceDatabases {
    /// Wikidata — for L14-27 factual relations.
    pub wikidata: Option<RelationDatabase>,
    /// WordNet — for L0-13 linguistic relations.
    pub wordnet: Option<RelationDatabase>,
}

/// Load all available reference databases from the data directory.
pub fn load_reference_databases() -> ReferenceDatabases {
    let mut result = ReferenceDatabases {
        wikidata: None,
        wordnet: None,
    };

    for base in &["data", "../data", "../../data"] {
        let base = Path::new(base);

        if result.wikidata.is_none() {
            let wikidata_path = base.join("wikidata_triples.json");
            if wikidata_path.exists() {
                if let Some(db) = RelationDatabase::load_wikidata(&wikidata_path) {
                    eprintln!(
                        "  Loaded Wikidata: {} relations, {} pairs",
                        db.num_relations(),
                        db.num_pairs()
                    );
                    result.wikidata = Some(db);
                }
            }
        }

        if result.wordnet.is_none() {
            let wordnet_path = base.join("wordnet_relations.json");
            if wordnet_path.exists() {
                if let Some(db) = RelationDatabase::load_wordnet(&wordnet_path) {
                    eprintln!(
                        "  Loaded WordNet: {} relations, {} pairs",
                        db.num_relations(),
                        db.num_pairs()
                    );
                    result.wordnet = Some(db);
                }
            }
        }

        if result.wikidata.is_some() && result.wordnet.is_some() {
            break;
        }
    }

    result
}

