//! `POST /v1/responses` — the OpenAI Responses API.
//!
//! The Responses API is OpenAI's successor to chat completions:
//! polymorphic `input` items instead of `messages`, server-side
//! conversation state via `store` / `previous_response_id`, an
//! `output[]` item list instead of `choices`, and a typed-event SSE
//! stream. This module serves the text + function-calling subset over
//! both runtimes (V2 `ModelWeights` and VINDEX3 containers).
//!
//! Module layout:
//!
//! ```text
//! routes/openai/responses/
//! ├── mod.rs       — module declarations + handler re-exports
//! ├── types.rs     — request/response wire types + shared constants
//! ├── input.rs     — `input` items → chat messages
//! ├── tools.rs     — Responses↔chat tool/format shape adapters
//! ├── engine.rs    — V2/V3 generation dispatch (callback-driven)
//! ├── handler.rs   — validation, planning, buffered serving
//! ├── stream.rs    — typed-event SSE serving
//! ├── retrieve.rs  — GET / DELETE /v1/responses/{id}
//! └── tests/       — unit tests (module tests folder)
//! ```
//!
//! Deliberate scope bounds, each rejected with a clear 400 rather than
//! silently dropped: `background: true`; non-text content parts
//! (images/audio); hosted tools; tools / structured output on V3
//! runtimes (no constrained-mask hook there yet).

pub mod engine;
pub mod events;
pub mod handler;
pub mod input;
pub mod retrieve;
pub mod stream;
pub mod tools;
pub mod types;

pub use handler::handle_responses;
pub use retrieve::{handle_delete_response, handle_get_response};

#[cfg(test)]
mod tests;
