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
//! Residual MSE is dominated by the outlier saturation and does not predict
//! downstream quality.
//!
//! Characterised by Exp 43 (30 prompts, layer 33, Gemma 3 4B):
//!
//! ```text
//! int8_clip3σ:  top-1 = 98.7% mean (93.3% min)
//!               top-5 = 100%
//!               KL    = ~2.0 nats
//!               Contract: D- (ArgmaxNearEquivalentHighMargin)
//! ```
//!
//! D-@high guarantees the first ~5 continuation tokens at 4.8% early-div
//! (95% CI 1.6%–10.7%, n=62). Total divergence over 20 tokens is ~20%
//! regardless of threshold — cascade compounds after the first wrong token.
//! Use for boundary-to-fresh-decode, not for long uninterrupted continuation.
//!
//! # Performance (M3 Max)
//!
//! ```text
//! bf16 encode  d=2560:    1.2 µs   (memory-bound, bit manipulation)
//! bf16 decode  d=2560:    0.27 µs
//! int8 encode  d=2560:    4.6 µs   (σ + clamp + quantize)
//! int8 decode  d=2560:    0.23 µs
//! metadata::compute:      517 µs   (log_softmax over 262K vocab — bottleneck)
//! ```
//!
//! `metadata::compute` at 517 µs is 0.005% of the ~10 s boundary-arrival
//! budget at 50 tok/s with 512-token chunks. Never on the critical path.
//!
//! See `benches/codec.rs` for full benchmark suite.

pub mod codec;
pub mod frame;
pub mod gate;
pub mod metadata;

pub use frame::{
    BoundaryAgreement, BoundaryCompression, BoundaryContract, BoundaryFrame, FallbackPolicy,
};
pub use gate::{BoundaryDecision, BoundaryGateConfig};
pub use metadata::BoundaryMetadata;
