// Per-module `forbid(unsafe_code)` (pattern 18) -- see
// commands/dev/ov_rd/mod.rs for the rationale. Unmarked modules
// (circuit_discover_cmd, fingerprint_extract_cmd, ov_gate_cmd,
// qk_modes_cmd, qk_rank_cmd) call `ndarray::s![...]`, which is a real,
// structural E0453 conflict -- confirmed via a fresh, live experiment
// (run 31604654148, 2026-08-12, dispatched larql-cli.yml directly
// rather than trusting the original claim's narrative: 38 E0453 errors
// across exactly these 22 files workspace-wide, "allow(unsafe_code)
// incompatible with previous forbid", all attributed to
// "this error originates in the macro `ndarray::s`"). walk_cmd and
// trajectory_trace_cmd were PREVIOUSLY, INCORRECTLY grouped under this
// same comment despite grep showing neither references `ndarray::s!`
// at all -- the same fresh run confirms it precisely: their unsafe
// blocks (walk_cmd.rs:10, trajectory_trace_cmd.rs:707) produce a plain
// "error: usage of an `unsafe` block", no E0453, no macro involved --
// forbid working exactly as intended, not blocked by anything. Both
// now correctly carry `#[forbid(unsafe_code)]` below.
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
#[forbid(unsafe_code)]
pub mod trajectory_trace_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod vector_extract_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod verify_cmd;
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod walk_cmd;
// pub mod vindex_bench_cmd;  // Removed: uses deprecated DownClusteredFfn
#[cfg(not(target_arch = "wasm32"))]
#[forbid(unsafe_code)]
pub mod weight_walk_cmd;
