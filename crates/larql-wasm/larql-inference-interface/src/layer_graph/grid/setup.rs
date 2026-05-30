use crate::ffn::moe_remote::RemoteMoeError;
use crate::layer_graph::pipeline_layer::{
    build_pipeline_layers, kv_cache_shapes_for_arch, patch_pipeline_layers_for_remote_ffn,
    patch_pipeline_layers_for_remote_moe, DEFAULT_GPU_KV_CACHE_MAX_SEQ,
};
use larql_compute::{prelude::ComputeBackend, FullPipelineLayer};
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

#[derive(Clone, Copy, Debug)]
pub(super) enum RemotePatch {
    Moe,
    Ffn,
}

pub(super) struct GridPipelineSetup<'a> {
    pub layers: Vec<FullPipelineLayer<'a>>,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
}

pub(super) fn build_grid_pipeline_setup<'a>(
    weights: &'a ModelWeights,
    index: &'a VectorIndex,
    patch: RemotePatch,
) -> Result<GridPipelineSetup<'a>, RemoteMoeError> {
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let q4_ffn = gate_index
        .interleaved_q4k_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .ok_or_else(|| {
            RemoteMoeError::BadResponse("no interleaved Q4 FFN mmap in vindex".into())
        })?;
    let ffn_format = if gate_index.interleaved_q4k_mmap_ref().is_some() {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = ffn_format
        .packed_matrix_bytes(intermediate, hidden)
        .ok_or_else(|| RemoteMoeError::BadResponse("unsupported interleaved FFN format".into()))?;

    let mut layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn,
        q4_ffn_per_matrix,
        ffn_format,
    );
    match patch {
        RemotePatch::Moe => patch_pipeline_layers_for_remote_moe(&mut layers, weights),
        RemotePatch::Ffn => patch_pipeline_layers_for_remote_ffn(&mut layers),
    }

    Ok(GridPipelineSetup {
        layers,
        hidden,
        intermediate,
        num_layers,
    })
}

pub(super) fn reset_and_preallocate_grid_kv(weights: &ModelWeights, backend: &dyn ComputeBackend) {
    backend.reset_kv_cache();
    let kv_shapes = kv_cache_shapes_for_arch(weights);
    backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);
}
