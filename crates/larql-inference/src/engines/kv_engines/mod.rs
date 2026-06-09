//! KV-cache engine implementations.
//!
//! Each engine implements [`crate::engines::KvEngine`] — a common interface
//! for prefill + autoregressive decode that manages inference state differently:
//!
//! ## Engine ladder (Gemma 3 4B @ 370K tokens)
//!
//! | Engine | Speed (tok/s) | Memory | Compression | Accuracy |
//! |---|---|---|---|---|
//! | [`markov_residual`] | ~95 (Metal Q4K) | ~171 MB | ~287× | exact (KL=0.0) |
//! | [`unlimited_context`] | ~94 (Metal Q4K) | ~193 MB | ~254× | exact within window |
//! | [`turbo_quant`] | ~95 (Metal Q4K) | ~12.7 GB | ~4× | cos≈0.991 |
//! | [`apollo`] | ~8× faster with boundaries | ~11 MB | ~4,414× | task accuracy |
//!
//! ## Selecting an engine
//!
//! ```text
//! larql bench gemma3-4b-q4k --engine markov-rs:window=512
//! larql bench gemma3-4b-q4k --engine unlimited-context:window=256
//! larql bench gemma3-4b-q4k --engine turbo-quant:bits=3
//! larql bench gemma3-4b-q4k --engine apollo:layer=25,coef=8.0
//! ```
//!
//! See [`crate::engines::EngineKind::from_name`] for the full parameter syntax.
//!
//! ## Architecture notes
//!
//! - **Metal Q4K path** (`prefill_q4k` / `decode_step_q4k`): all four engines
//!   use the Metal `decode_token` full pipeline when a Q4K VectorIndex and a
//!   Metal backend are available. This gives 93-95 tok/s — matching or exceeding
//!   the standard larql-metal path (76 tok/s) because the engine bench uses
//!   faster Metal lm_head KNN rather than a full vocab matmul.
//!
//! - **CPU fallback**: when Metal is unavailable, engines fall back to a CPU
//!   path using dequantised attention tensors (lazily inserted into
//!   `weights.tensors`) and `WalkFfn` for Q4K FFN.
//!
//! - **Apollo compressed path**: when the store has boundary residuals captured
//!   at `crystal_layer` (default 30), `forward_from_layer` runs only
//!   `crystal_layer..num_layers` layers (~4 instead of 34), ~8.5× faster per step.

pub mod apollo;
pub mod markov_residual;
pub mod turbo_quant;
pub mod unlimited_context;
