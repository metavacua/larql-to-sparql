//! OpenAI-compatible HTTP endpoints (N0).
//!
//! Slice 1 (`models`, `embeddings`, `completions`) and slice 2
//! (`chat::completions`) ship the OpenAI request/response shapes so
//! existing `openai` Python and JS SDKs work unmodified — point
//! `base_url` at the larql server and the SDK calls just work. Slice
//! 3 adds SSE streaming on completions + chat completions.
//!
//! Module layout:
//!
//! ```text
//! routes/openai/
//! ├── mod.rs           — re-exports + module declarations
//! ├── util.rs          — shared helpers (StopSpec, id-suffix, unix_now,
//! │                      stop-string trimming, SSE error chunk)
//! ├── embeddings.rs    — POST /v1/embeddings (mean-pooled lookup)
//! ├── completions.rs   — POST /v1/completions (legacy text completions
//! │                      + slice 3 SSE streaming)
//! └── chat.rs          — POST /v1/chat/completions (chat-template
//!                        rendering + slice 3 SSE streaming)
//! ```
//!
//! Roadmap entries: ROADMAP.md → N0.1, N0.2, N0.4, N0.5 (live);
//! N0.3, N0.6, N0.2-fast, N0-router (queued).

pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod schema;
pub mod util;

pub use error::OpenAIError;

// Re-export the handler functions so the route table in `routes/mod.rs`
// can reach them as `openai::chat::handle_chat_completions`, etc. The
// indirection isn't strictly necessary, but it (a) documents the public
// surface of this folder and (b) makes it clear which functions are
// intended as HTTP handlers vs internal helpers.
pub use chat::handle_chat_completions;
pub use completions::handle_completions;
pub use embeddings::handle_embeddings;
