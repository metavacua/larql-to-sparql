// larql-router tests live in src/grid.rs and src/rebalancer.rs (inline).
// They are dual-annotated with wasm_bindgen_test directly in those files.
// This file satisfies wasm-pack's requirement for at least one tests/ file.

use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_node_experimental);
