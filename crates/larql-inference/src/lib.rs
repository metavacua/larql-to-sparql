extern crate blas_src;

pub mod attention;
pub mod capture;
pub mod error;
pub mod ffn;
pub mod forward;
pub mod graph_ffn;
pub mod layer_graph;
pub mod model;
pub mod prompt;
pub mod residual;
pub mod tokenizer;
pub mod trace;
pub mod trie;
pub mod vindex;
pub mod walker;
pub mod experts;

// Re-export dependencies for downstream crates.
pub use larql_models;
pub use larql_vindex;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Backend re-exports (from larql-compute).
pub use larql_compute::{ComputeBackend, MatMulOp, default_backend, cpu_backend, dot_proj_gpu, matmul_gpu};
pub use larql_compute::CpuBackend;
pub use larql_compute::cpu::ops::moe::{run_single_expert, run_single_expert_with_norm, cpu_moe_forward};
pub use larql_compute::MoeLayerWeights;
pub use larql_compute::Activation as ComputeActivation;

/// Map a model's activation function to the compute-layer `Activation` enum.
pub fn activation_from_arch(arch: &dyn larql_models::ModelArchitecture) -> larql_compute::Activation {
    match arch.activation() {
        larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
        _ => larql_compute::Activation::Silu,
    }
}
#[cfg(feature = "metal")]
pub use larql_compute::MetalBackend;

// Re-export essentials at crate root.
pub use capture::{
    CaptureCallbacks, CaptureConfig, InferenceModel, TopKEntry, VectorFileHeader, VectorRecord,
};
pub use error::InferenceError;
pub use ffn::{
    FfnBackend, LayerFfnRouter, RemoteFfnConfig, RemoteFfnError, RemoteWalkBackend,
    RemoteLatencyStats, SparseFfn, WeightFfn,
    MoeRouterWeights, RemoteMoeBackend, RemoteMoeError, ShardConfig,
};
pub use attention::AttentionWeights;
pub use forward::{
    calibrate_scalar_gains, capture_decoy_residuals, capture_ffn_activation_matrix,
    capture_residuals, estimate_ffn_covariance, forward_to_layer, logit_lens_top1, predict,
    predict_from_hidden, predict_from_hidden_with_ffn, predict_with_ffn,
    predict_with_ffn_attention, predict_with_ffn_trace, predict_with_router,
    predict_with_strategy, trace_forward, trace_forward_full, trace_forward_with_ffn,
    LayerAttentionCapture, LayerMode, PredictResult, PredictResultWithAttention,
    PredictResultWithResiduals, TraceResult,
    capture_spec_residuals, SpecCapture,
    run_memit, run_memit_with_target_opt, MemitFact, MemitResult, MemitFactResult,
    TargetDelta, TargetDeltaOpts,
    apply_knn_override, infer_patched, infer_patched_q4k, walk_trace_from_residuals, InferPatchedResult,
    KnnOverride, KNN_COSINE_THRESHOLD,
    forward_raw_logits, RawForward, hidden_to_raw_logits,
    generate_cached_constrained,
};
pub use graph_ffn::{GateIndex, IndexBuildCallbacks, SilentIndexCallbacks};
pub use trace::{
    trace_residuals, trace as trace_decomposed, AnswerWaypoint, LayerSummary,
    ResidualTrace, TraceNode, TracePositions, TraceStore, TraceWriter,
    BoundaryStore, BoundaryWriter,
    ContextStore, ContextWriter, ContextTier,
};
pub use layer_graph::{
    // Production
    LayerGraph, LayerOutput, DenseLayerGraph, WalkLayerGraph, PipelinedLayerGraph,
    CachedLayerGraph, PerLayerGraph,
    predict_with_graph, predict_with_graph_vindex_logits, predict_pipeline,
    predict_split_pass, predict_split_cached, predict_honest, generate, GenerateResult, AttentionCache,
    hybrid::predict_hybrid,
    trace_with_graph, build_adaptive_graph,
    // Analysis/validation
    TemplatePattern, TemplateUniverse, GuidedWalkLayerGraph,
    detect_template,
    // Expert grid generation
    grid::{generate_with_remote_moe, GridGenerateResult},
};
pub use vindex::{WalkFfn, WalkFfnConfig, FfnL1Cache, predict_q4k};
pub use model::{load_model_dir, resolve_model_path, ModelWeights};
pub use tokenizer::{decode_token, decode_token_raw, encode_prompt, load_tokenizer};

// Walker re-exports.
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};
