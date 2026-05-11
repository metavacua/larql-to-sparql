//! HNSW build-time tuning constants.
//!
//! Centralised here so layer- and expert-grain HNSW indexes share a single
//! source of truth. The values are empirically tuned for Gemma-class gate
//! matrices (~10K features per layer, ~700 features per expert); revisit
//! if the typical shape shifts by an order of magnitude.

/// HNSW parameters used at build time.
///
/// `m` is the max number of neighbours per node in the graph; bigger
/// graphs give better recall at the cost of more memory and slower
/// build. `ef_construction` is the beam width during construction;
/// higher = better recall, slower build, no runtime cost.
#[derive(Debug, Clone, Copy)]
pub struct HnswBuildConfig {
    pub m: usize,
    pub ef_construction: usize,
}

impl HnswBuildConfig {
    /// Whole-layer index — tuned for ~10K-feature dense gate matrices.
    pub const LAYER: Self = Self {
        m: 8,
        ef_construction: 32,
    };

    /// Per-(layer, expert) index — smaller because each expert covers
    /// only ~700 features. The layer-level constants are overkill at
    /// this size and the smaller graph builds ~3× faster with
    /// comparable recall.
    pub const EXPERT: Self = Self {
        m: 6,
        ef_construction: 16,
    };
}

// Compile-time invariants. These are `const` assertions (not runtime
// `assert!`s) because the values are constants — clippy correctly
// notes runtime asserts on constants get optimised out, so the only
// honest way to enforce them is at compile time.

// Layer- and per-expert tunings must differ. If they collapse, either
// the size-class assumption changed or the constants got copy-pasted.
const _: () = assert!(HnswBuildConfig::LAYER.m > HnswBuildConfig::EXPERT.m);
const _: () =
    assert!(HnswBuildConfig::LAYER.ef_construction > HnswBuildConfig::EXPERT.ef_construction);

// HNSW recall degrades badly when ef_construction <= m. Pin the
// invariant on both sides so a future tweak can't violate it silently.
const _: () = assert!(HnswBuildConfig::LAYER.ef_construction > HnswBuildConfig::LAYER.m);
const _: () = assert!(HnswBuildConfig::EXPERT.ef_construction > HnswBuildConfig::EXPERT.m);
