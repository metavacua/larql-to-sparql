//! Tunables shared by extraction paths.

/// Feature batch size for down-token projection during extraction.
pub const FEATURE_PROJECTION_BATCH: usize = 1024;

/// Feature batch size for gate-token projection during extraction.
pub const GATE_TOP_TOKEN_BATCH: usize = 1024;

/// Upper bound for relation clusters in the automatic labeling pass.
pub const MAX_RELATION_CLUSTERS: usize = 512;

/// K-means iterations for relation-cluster labeling.
pub const RELATION_KMEANS_ITERS: usize = 50;

/// Token ids below this are treated as tokenizer control/special ids in
/// extraction-time relation labeling.
pub const FIRST_CONTENT_TOKEN_ID: usize = 3;

/// Default top-K down tokens stored when rebuilding metadata from a
/// partially extracted vindex.
pub const DEFAULT_DOWN_TOP_K: usize = 10;
