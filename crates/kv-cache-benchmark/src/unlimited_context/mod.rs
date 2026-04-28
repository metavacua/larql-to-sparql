//! Tier 2 — Unlimited Context Engine (Rust port of Python/MLX `UnlimitedContextEngine`).
// SPDX-License-Identifier: Apache-2.0

//!
//! Three-tier storage with sparse K,V checkpoints and model-forward replay:
//!
//! ```text
//! ┌──────────────────────┬─────────────────────┬──────────────────┐
//! │   Boundary (WARM)    │   Active window KV   │ Token archive    │
//! │   1 K,V per layer    │   grows as window    │ ~4 B / token     │
//! │   per closed window  │   is extended        │ (cold tier)      │
//! └──────────────────────┴─────────────────────┴──────────────────┘
//! ```
//!
//! - Each window is `window_size` tokens (default 512). As the window fills,
//!   the engine extends an in-memory K,V cache via `rs_extend_from_checkpoint`.
//! - When the window closes: (a) the last-position K,V per layer is saved to
//!   `CheckpointStore`, (b) the window's token IDs are appended to
//!   `TokenArchive`, (c) the full window K,V is evicted.
//! - To query any past window, call `replay_window(id)` — it reconstructs the
//!   window's K,V by running a model-forward pass over the archived tokens
//!   starting from the prior window's boundary checkpoint.
//!
//! ## Correctness claim (what's bit-exact, what isn't)
//!
//! - **Within-window bit-exact**: `rs_extend_from_checkpoint(tokens, prior, abs_start)`
//!   produces the same `h_new` and K,V for `tokens` as the same call with
//!   identical inputs. The forward pass is deterministic up to numerical
//!   precision (bf16/f32 arithmetic).
//! - **Against joint prefill**: replay(window_N, N>0) differs from joint
//!   `prefill([w_0, ..., w_N])` at the window-N positions because the 1-token
//!   prior checkpoint compresses `|w_{N-1}|` positions of K,V to 1. This is
//!   the same lossiness variant (ii) per-layer boundary gives, measured at
//!   cos ≈ 0.965 in `experiments/20_free_monoids_poincare/f1prime_*.py`.
//!
//! **Memory** on Gemma 3 4B (34 layers, 4 KV heads, head_dim=256, bf16):
//! 1 checkpoint = 34 × 2 × (4 × 256) × 2 B ≈ 139 KB. Python docs call this
//! ~174 KB accounting for some overhead. Matches either way.

mod checkpoint_store;
mod engine;
mod extend;
mod token_archive;

pub use checkpoint_store::CheckpointStore;
pub use engine::{EngineStats, UnlimitedContextEngine};
pub use extend::{empty_prior, rs_extend_from_checkpoint, ExtendOutput};
pub use token_archive::TokenArchive;

/// Test-only re-export so integration tests can construct an empty prior
/// without importing the inner module path.
#[doc(hidden)]
pub use extend::empty_prior as __empty_prior_for_test;
