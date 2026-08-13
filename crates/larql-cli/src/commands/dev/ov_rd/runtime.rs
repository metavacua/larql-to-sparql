use larql_inference::ModelWeights;
use larql_vindex::VectorIndex;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

pub(super) fn insert_q4k_layer_tensors(
    weights: &mut ModelWeights,
    index: &VectorIndex,
    layer: usize,
) -> Result<Vec<String>, Box<dyn core::error::Error>> {
    // `insert_q4k_layer_tensors_resident` already returns a plain
    // `String` error -- `String` implements `Into<Box<dyn Error>>` on
    // its own, so the old `std::io::Error::new(..., err)` wrapper was
    // unnecessary (and `std::io` has no core/alloc equivalent at all).
    larql_inference::vindex::insert_q4k_layer_tensors_resident(weights, index, layer)
        .map_err(Into::into)
}

pub(super) fn remove_layer_tensors(weights: &mut ModelWeights, keys: Vec<String>) {
    larql_inference::vindex::remove_layer_tensors_resident(weights, keys);
}
