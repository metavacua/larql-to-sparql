pub mod attention_walker;
pub mod utils;
pub mod vector_extractor;
pub mod weight_walker;

// Test-only fixture builder. Always compiled (so integration tests in
// `tests/` can reach it) but not re-exported from the crate root.
#[doc(hidden)]
pub mod test_fixture;
