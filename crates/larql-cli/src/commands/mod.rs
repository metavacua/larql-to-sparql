// `forbid(unsafe_code)` is applied per-submodule rather than crate-wide
// (see main.rs) because `dev`, `extraction`, and `primary` each contain
// a mix of forbid-safe files and files whose `ndarray::s!` calls trip
// E0453 -- forbid cannot be locally overridden even through a macro
// expansion, so it has to stop one level above each such file rather
// than being declared here and inherited. `diagnostics`/`query` contain
// no such files (pattern 18 in the gating plan doc).
pub mod dev;
#[forbid(unsafe_code)]
pub mod diagnostics;
pub mod extraction;
pub mod primary;
#[forbid(unsafe_code)]
pub mod query;
