// Compile smoke test for larql-router-protocol on wasm32.
// The entire public API is gated behind cfg(not(target_arch = "wasm32"))
// because it depends on tonic (gRPC), which requires a native async runtime.
// The absence of passing test cells in wasm-test is the forensic signal for
// the gRPC porting backlog (e.g., replacing tonic with a WASM-compatible
// transport layer).

use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_node_experimental);

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn smoke() {
    // Crate loads. All public API is cfg(not(wasm32)); no native tests.
}
