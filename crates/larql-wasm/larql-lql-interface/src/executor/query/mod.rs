//! Query executor: WALK, INFER, SELECT, DESCRIBE, EXPLAIN.
//!
//! Each verb lives in its own file. Shared helpers (layer-band
//! resolution) live here because both DESCRIBE and EXPLAIN INFER
//! consume them.

mod describe;
mod explain;
mod infer;
mod infer_trace;
mod select;
mod walk;

/// Resolve the layer-band boundaries from the vindex config, with a
/// family-based default and a final whole-range fallback.
pub(super) fn resolve_bands(config: &larql_vindex::VindexConfig) -> larql_vindex::LayerBands {
    let last = config.num_layers.saturating_sub(1);
    config
        .layer_bands
        .clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        })
}
