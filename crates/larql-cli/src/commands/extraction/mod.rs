// Per-module `forbid(unsafe_code)` (pattern 18) -- see
// commands/dev/ov_rd/mod.rs for the rationale. Unmarked modules call
// `ndarray::s![...]`, CI-confirmed via workflow run 31464601274.
//
// Per-module `#[cfg(not(target_arch = "wasm32"))]` (wasm32v1-none gating,
// round 1) -- every module in this file turned out WHOLESALE_NATIVE
// (clap::Args CLI surface, std::fs/Instant/tokenizers/InferenceModel
// taken directly, etc.), so every `pub mod` line below is native-gated.
// `compile_cmd` is gated as a whole rather than split, which also
// excludes its two factually-portable private children
// (`compile_cmd::detect`, `compile_cmd::edge`) from the wasm32 build --
// they have no portable-side caller today, so this is not a loss.
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod attention_capture_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod attention_walk_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod attn_bottleneck_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod bfs_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod bottleneck_test_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod build_cmd;
#[cfg(not(target_arch = "wasm32"))]
pub mod circuit_discover_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod compile_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod convert_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod embedding_jump_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod extract_index_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod ffn_bottleneck_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod ffn_latency_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod ffn_overlap_cmd;
#[cfg(not(target_arch = "wasm32"))]
pub mod fingerprint_extract_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod hf_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod index_gates_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod kg_bench_cmd;
#[cfg(not(target_arch = "wasm32"))]
pub mod ov_gate_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod predict_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod projection_test_cmd;
#[cfg(not(target_arch = "wasm32"))]
pub mod qk_modes_cmd;
#[cfg(not(target_arch = "wasm32"))]
pub mod qk_rank_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod qk_templates_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod residuals_cmd;
#[cfg(not(target_arch = "wasm32"))]
pub mod trajectory_trace_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod vector_extract_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod verify_cmd;
#[cfg(not(target_arch = "wasm32"))]
pub mod walk_cmd;
// pub mod vindex_bench_cmd;  // Removed: uses deprecated DownClusteredFfn
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod weight_walk_cmd;
