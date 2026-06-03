//! Pair-based relation labeling.
//!
//! For each cluster, collect (gate_input_token, output_token) pairs,
//! then match against Wikidata triples and WordNet relations.
//! The relation type with the most matching pairs wins.
//!
//! - `database`: `RelationDatabase`, `ReferenceDatabases`,
//!               loaders for Wikidata and WordNet.
//! - `labeling`: `label_clusters_from_pairs`, `label_clusters_from_outputs`.

pub mod database;
pub mod labeling;

pub use database::{load_reference_databases, ReferenceDatabases, RelationDatabase};
pub use labeling::{label_clusters_from_outputs, label_clusters_from_pairs};
