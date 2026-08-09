//! Print what the parser actually builds, for statements whose AST has been
//! argued about.
//!
//! This exists because claims about the AST were repeatedly made by reading
//! `ast.rs` and the executor's parameter names — including "the FORMAT clause
//! is parsed and discarded", inferred from an underscore-prefixed binding.
//! That is inference, not observation. Running the parser and printing the
//! structure is observation.
//!
//! It asserts almost nothing on purpose: the output is the evidence, and what
//! it means is decided by a reader, not by this file. The one thing it does
//! assert is that the statements parse at all, so a silent parse failure
//! cannot masquerade as an empty result.
//!
//! Run: `cargo test -p larql-lql --test ast_dump -- --nocapture`

const STATEMENTS: &[&str] = &[
    // The three COMPILE INTO MODEL forms the evidence job runs.
    r#"COMPILE CURRENT INTO MODEL "out" FORMAT safetensors;"#,
    r#"COMPILE CURRENT INTO MODEL "out" FORMAT gguf;"#,
    r#"COMPILE CURRENT INTO MODEL "out";"#,
    // The sibling that is reported to work, for comparison.
    r#"COMPILE CURRENT INTO VINDEX "out";"#,
    // A trailing slash on the destination — the corpus uses one, the evidence
    // job does not, and nothing has established whether it matters.
    r#"COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors;"#,
];

#[test]
fn dump_compile_asts() {
    println!("\n===== PARSED AST =====");
    for src in STATEMENTS {
        match larql_lql::parse(src) {
            Ok(stmt) => println!("{src}\n    {stmt:?}\n"),
            Err(e) => println!("{src}\n    PARSE ERROR: {e}\n"),
        }
    }

    // The only assertion: these must parse. An empty or error-filled dump
    // would otherwise read as "the AST has no such field".
    for src in STATEMENTS {
        assert!(
            larql_lql::parse(src).is_ok(),
            "statement did not parse, so the dump above says nothing about its AST: {src}"
        );
    }
}
