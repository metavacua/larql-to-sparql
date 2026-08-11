// Per-module `forbid(unsafe_code)` (pattern 18) -- see
// commands/dev/ov_rd/mod.rs for the rationale. Unmarked modules call
// `ndarray::s![...]`, CI-confirmed via workflow run 31464601274.
#[forbid(unsafe_code)]
pub mod attention_capture_cmd;
#[forbid(unsafe_code)]
pub mod attention_walk_cmd;
#[forbid(unsafe_code)]
pub mod attn_bottleneck_cmd;
#[forbid(unsafe_code)]
pub mod bfs_cmd;
#[forbid(unsafe_code)]
pub mod bottleneck_test_cmd;
#[forbid(unsafe_code)]
pub mod build_cmd;
pub mod circuit_discover_cmd;
#[forbid(unsafe_code)]
pub mod compile_cmd;
#[forbid(unsafe_code)]
pub mod convert_cmd;
#[forbid(unsafe_code)]
pub mod embedding_jump_cmd;
#[forbid(unsafe_code)]
pub mod extract_index_cmd;
#[forbid(unsafe_code)]
pub mod ffn_bottleneck_cmd;
#[forbid(unsafe_code)]
pub mod ffn_latency_cmd;
#[forbid(unsafe_code)]
pub mod ffn_overlap_cmd;
pub mod fingerprint_extract_cmd;
#[forbid(unsafe_code)]
pub mod hf_cmd;
#[forbid(unsafe_code)]
pub mod index_gates_cmd;
#[forbid(unsafe_code)]
pub mod kg_bench_cmd;
pub mod ov_gate_cmd;
#[forbid(unsafe_code)]
pub mod predict_cmd;
#[forbid(unsafe_code)]
pub mod projection_test_cmd;
pub mod qk_modes_cmd;
pub mod qk_rank_cmd;
#[forbid(unsafe_code)]
pub mod qk_templates_cmd;
#[forbid(unsafe_code)]
pub mod residuals_cmd;
pub mod trajectory_trace_cmd;
#[forbid(unsafe_code)]
pub mod vector_extract_cmd;
#[forbid(unsafe_code)]
pub mod verify_cmd;
pub mod walk_cmd;
// pub mod vindex_bench_cmd;  // Removed: uses deprecated DownClusteredFfn
#[forbid(unsafe_code)]
pub mod weight_walk_cmd;
