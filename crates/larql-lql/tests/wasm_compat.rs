// larql-lql lexer tests live in src/lexer.rs (inline, pub(crate) access).
// They are dual-annotated with wasm_bindgen_test directly in that file.
// This file satisfies wasm-pack's requirement for at least one test file
// in tests/ and provides a smoke check of the public parse() API.

use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_node_experimental);

use larql_lql::{parse, Statement};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_walk() {
    let stmt = parse(r#"WALK "hello" TOP 3;"#).unwrap();
    assert!(matches!(stmt, Statement::Walk { .. }));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn parse_stats() {
    let stmt = parse("STATS;").unwrap();
    assert!(matches!(stmt, Statement::Stats { .. }));
}
