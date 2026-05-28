//! larql-boundary — confidence-gated BOUNDARY ref codec.
//!
//! Transforms transformer final-layer residuals into compact, contract-bearing
//! protocol objects. Compressed when the boundary is confident; exact when fragile.
//!
//! ```text
//! KV cache for the present.
//! Residual boundaries for memory.
//! ```
//!
//! # Architecture
//!
//! ```text
//! Phase 1 — codec      residual bytes  ↔  f32 slices
//! Phase 2 — metadata   per-boundary confidence fields from logit slices
//! Phase 3 — gate       per-boundary decision: compress / bf16 / cold-replay
//! ```
//!
//! **Model-agnostic.** This crate takes raw `f32` slices only — no model weights,
//! no inference backend, no MLX. The caller (`larql-inference`) runs the forward
//! pass and provides logit slices.
//!
//! # Quick start
//!
//! ```rust
//! use larql_boundary::{codec, gate, metadata};
//! use larql_boundary::gate::{BoundaryDecision, BoundaryGateConfig};
//!
//! // ── Phase 1: compress a residual ──────────────────────────────
//! // int8_clip3σ: 2564 bytes for d=2560 vs 5120 for bf16 (2× compression)
//! let residual = vec![0.1f32; 2560];
//! let payload = codec::int8::encode(&residual);
//! let decoded  = codec::int8::decode(&payload);
//! assert_eq!(decoded.len(), residual.len());
//!
//! // ── Phase 2: compute confidence metadata from logits ──────────
//! // Caller provides: lm_head(final_norm(raw_residual)) and
//! //                  lm_head(final_norm(decoded_compressed_residual))
//! let raw_logits = vec![0.0f32; 262_145]; // Gemma 3 4B vocab size
//! let hat_logits = raw_logits.clone();
//! let mut meta = metadata::compute(&raw_logits, Some(&hat_logits));
//!
//! // ── Phase 3: gate decision ─────────────────────────────────────
//! // Exp 44 calibrated: min_log_prob_margin = 2.16 for Gemma 3 4B.
//! // Default config has calibration_mode = true → always bf16 until calibrated.
//! let config = BoundaryGateConfig {
//!     calibration_mode: false,     // flip after running calibrate.py
//!     min_log_prob_margin: 2.16,   // Exp 44 Track A (log-prob margin units)
//!     min_top1_prob: 0.5,
//!     ..Default::default()
//! };
//! let decision = gate::apply(&mut meta, &config);
//! match decision {
//!     BoundaryDecision::CompressedOk { .. } => { /* emit int8 frame   */ }
//!     BoundaryDecision::UseBf16             => { /* emit bf16 frame   */ }
//!     _                                     => { /* cold replay / reject */ }
//! }
//! ```
//!
//! # Accuracy contract
//!
//! The accuracy contract is **top-1 token preservation**, not residual MSE.
//!
//! Characterised by Exp 43 (30 prompts, layer 33, Gemma 3 4B):
//!
//! ```text
//! int8_clip3σ:  top-1 = 98.7% mean (93.3% min)
//!               top-5 = 100%
//!               KL    = ~2.0 nats
//!               Contract: D- (ArgmaxNearEquivalentHighMargin)
//! ```

pub use larql_wasm32v1_none_lib::boundary::codec;
pub use larql_wasm32v1_none_lib::boundary::frame;
pub use larql_wasm32v1_none_lib::boundary::gate;
pub use larql_wasm32v1_none_lib::boundary::metadata;

pub use larql_wasm32v1_none_lib::boundary::frame::{
    BoundaryAgreement, BoundaryCompression, BoundaryContract, BoundaryFrame, FallbackPolicy,
};
pub use larql_wasm32v1_none_lib::boundary::gate::{BoundaryDecision, BoundaryGateConfig};
pub use larql_wasm32v1_none_lib::boundary::metadata::BoundaryMetadata;
