pub mod bfs;
pub mod chain;
// reqwest/futures-core need std (their own crates don't declare
// #![no_std]), so on wasm32 the http_provider module must be excluded
// entirely even when the "http" feature is on -- the feature and the
// dependency's availability are separate axes here (dep:reqwest is
// target-gated in Cargo.toml, but Cargo features themselves aren't
// target-conditional, so "http" stays enabled on the default wasm32
// leg with the dependency simply absent).
#[cfg(all(feature = "http", not(target_arch = "wasm32")))]
pub mod http_provider;
pub mod mock_provider;
pub mod provider;
pub mod templates;
