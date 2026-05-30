//! Shared helpers for FFN band names and layer filtering.
//!
//! Three routes (describe, explain, stream) independently replicated the same
//! "syntax/knowledge/output/all" match arm and the same layer-bands fallback
//! chain. This module centralises both.

use larql_vindex::LayerBands;

use crate::state::LoadedModel;

pub const BAND_SYNTAX: &str = "syntax";
pub const BAND_KNOWLEDGE: &str = "knowledge";
pub const BAND_OUTPUT: &str = "output";
pub const BAND_ALL: &str = "all";

/// Inference mode passed as `?mode=` or in a JSON body.
pub const INFER_MODE_WALK: &str = "walk";
pub const INFER_MODE_DENSE: &str = "dense";
pub const INFER_MODE_COMPARE: &str = "compare";

/// Insert-result mode field values.
pub const INSERT_MODE_CONSTELLATION: &str = "constellation";
pub const INSERT_MODE_EMBEDDING: &str = "embedding";

/// Source label applied to probe-confirmed relation edges.
/// Used in JSON responses (describe, walk) and gRPC edge structs.
pub const PROBE_RELATION_SOURCE: &str = "probe";

/// Status string returned by the health endpoint and gRPC HealthResponse.
pub const HEALTH_STATUS_OK: &str = "ok";

/// Resolve the layer-bands for a model, falling back to family-derived bands
/// and then to a flat range covering all layers.
pub fn get_layer_bands(model: &LoadedModel) -> LayerBands {
    let last = model.config.num_layers.saturating_sub(1);
    model
        .config
        .layer_bands
        .clone()
        .or_else(|| LayerBands::for_family(&model.config.family, model.config.num_layers))
        .unwrap_or(LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        })
}

/// Filter a layer list to only those that fall within the named band.
/// `BAND_ALL` (or any unrecognised string) returns all layers unchanged.
pub fn filter_layers_by_band(all_layers: Vec<usize>, band: &str, bands: &LayerBands) -> Vec<usize> {
    match band {
        BAND_SYNTAX => all_layers
            .into_iter()
            .filter(|l| *l >= bands.syntax.0 && *l <= bands.syntax.1)
            .collect(),
        BAND_KNOWLEDGE => all_layers
            .into_iter()
            .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
            .collect(),
        BAND_OUTPUT => all_layers
            .into_iter()
            .filter(|l| *l >= bands.output.0 && *l <= bands.output.1)
            .collect(),
        _ => all_layers,
    }
}
