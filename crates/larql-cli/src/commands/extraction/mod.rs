pub mod attn_bottleneck_cmd;
pub mod build_cmd;
pub mod convert_cmd;
pub mod hf_cmd;
pub mod verify_cmd;
pub mod attention_capture_cmd;
pub mod attention_walk_cmd;
pub mod bfs_cmd;
pub mod circuit_discover_cmd;
pub mod extract_index_cmd;
pub mod extract_routes_cmd;
#[allow(deprecated)]
pub mod ffn_bench_cmd;
pub mod ffn_bottleneck_cmd;
pub mod ffn_overlap_cmd;
#[allow(deprecated)]
pub mod ffn_throughput_cmd;
// pub mod graph_walk_cmd;  // Removed: uses deprecated FeatureListFfn
pub mod index_gates_cmd;
pub mod kg_bench_cmd;
pub mod ov_gate_cmd;
#[allow(deprecated)]
pub mod predict_cmd;
pub mod qk_modes_cmd;
pub mod qk_rank_cmd;
pub mod qk_templates_cmd;
pub mod residuals_cmd;
pub mod vector_extract_cmd;
pub mod walk_cmd;
pub mod bottleneck_test_cmd;
pub mod embedding_jump_cmd;
pub mod fingerprint_extract_cmd;
pub mod projection_test_cmd;
pub mod trajectory_trace_cmd;
// pub mod vindex_bench_cmd;  // Removed: uses deprecated DownClusteredFfn
pub mod weight_walk_cmd;
