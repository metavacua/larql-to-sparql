//! Unit tests for the responses module (project convention: module
//! tests live in a `tests/` folder, not inline beside the source).
//! Handler/stream behaviour is covered end-to-end by the crate
//! integration tests (`tests/test_openai_responses_coverage.rs`).

mod events_tests;
mod handler_tests;
mod input_tests;
mod tools_tests;
mod types_tests;
