#![cfg_attr(target_arch = "wasm32", no_std)]
#[macro_use]
extern crate alloc;

// Command execution (clap parsing + vindex/inference/IO) is native.
// formatting (empty) and utils (pure base64/string helpers) stay portable.
#[cfg(not(target_arch = "wasm32"))]
pub mod commands;
pub mod formatting;
pub mod utils;

/// WASM entry point: CLI dispatch for a host caller via bridge.
/// Phase B will route through larql-bridge traits.
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn larql_cli_dispatch(_cmd_ptr: u32, _cmd_len: u32) -> u32 {
    0
}
