//! Speech-model generation drivers — engine-shaped composition of the
//! substrate primitives for models whose output domain is not text.
//!
//! Nothing here touches a tokenizer: these drivers produce model-domain
//! ids natively, which is the output-adapter invariant of
//! `docs/tts-funnel.md` §3 step 4 — core generation produces ids/state,
//! and interpretation (text detokenisation, codec decode) belongs to
//! whatever consumes them.

// Pull `larql_models::speech::*` (native-only upstream) and/or
// `tokenizers::Tokenizer` directly in real (non-test) code.
#[cfg(not(target_arch = "wasm32"))]
pub mod moss_prompt;
#[cfg(not(target_arch = "wasm32"))]
pub mod moss_realtime;
pub mod moss_sampling;
#[cfg(not(target_arch = "wasm32"))]
pub mod moss_session;
pub mod stream_timing;
// Not #[cfg(test)]-gated despite the name -- always-compiled fixture code.
#[cfg(not(target_arch = "wasm32"))]
pub mod test_support;
