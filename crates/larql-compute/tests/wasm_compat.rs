// Compile smoke test for larql-compute on wasm32.
// All substantive tests require BLAS (blas_src / OpenBLAS) or the heavy_tests
// feature, neither of which is available in wasm32.  The absence of test
// cells in wasm-test is the forensic signal for the BLAS porting backlog.

use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_node_experimental);

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn smoke() {
    // Crate loads. Native tests run in tests/test_correctness.rs etc.
}
