//! Every cell the strategy matrix ships must parse.
//!
//! A malformed probe is a harness defect, not a finding about larql. Six cells
//! in `commands.jsonl` sent `BEGIN PATCH;`, but `parse_begin` calls
//! `expect_string()`, so the path is mandatory — those cells never opened a
//! patch session. Worse, a cell is a BATCH: the parse error was the first
//! error in it and masked every later statement, so the corpus defect
//! presented as a product defect.
//!
//! This checks the corpus with LQL's own lexer and parser — `split_statements`
//! feeding `parser::parse`, the same pair `run_batch` uses — rather than with
//! a pattern that approximates the grammar. An approximation drifts from the
//! real thing and starts inventing defects, which is the failure mode the
//! whole harness exists to avoid. It also misses real ones: the pattern this
//! replaced knew only about `BEGIN PATCH` and would have passed the corpus
//! while `MERGE;` was still malformed.
//!
//! It asserts nothing about what a cell DOES. Whether a well-formed statement
//! succeeds, errors, or declines is the run's business and belongs in the
//! captures, not here.

use std::path::PathBuf;

/// `scripts/lql_matrix/` relative to this crate.
fn corpus(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/lql_matrix")
        .join(name)
}

/// Statements the parser is expected to reject: the corpus deliberately
/// carries negative cells, and a corpus test that demanded everything parse
/// would delete exactly the coverage that proves larql rejects bad input.
/// Keyed by cell id, so a cell going from "rejected" to "accepted" still
/// shows up as a failure here.
fn is_negative_cell(id: &str) -> bool {
    id.starts_with("neg.") || id.starts_with("error.")
}

#[test]
fn every_shipped_lql_cell_parses() {
    let path = corpus("commands.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut failures: Vec<String> = Vec::new();
    let mut cells = 0usize;
    let mut statements = 0usize;

    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("{}:{}: not JSON: {e}", path.display(), lineno + 1));
        let id = row["id"].as_str().unwrap_or_else(|| {
            panic!("{}:{}: cell has no string `id`", path.display(), lineno + 1)
        });
        let lql = row["lql"].as_str().unwrap_or_else(|| {
            panic!(
                "{}:{}: cell {id:?} has no string `lql`",
                path.display(),
                lineno + 1
            )
        });
        cells += 1;

        // The same decomposition run_batch performs, including its comment
        // stripping — testing a different split would test a different thing
        // than the one that runs.
        for stmt_text in larql_lql::split_statements(lql) {
            let trimmed: String = stmt_text
                .lines()
                .filter(|l| !l.trim().starts_with("--"))
                .collect::<Vec<_>>()
                .join("\n");
            let trimmed = trimmed.trim();
            if trimmed.is_empty() {
                continue;
            }
            statements += 1;
            match larql_lql::parse(trimmed) {
                Ok(_) => {}
                Err(e) => {
                    if !is_negative_cell(id) {
                        failures.push(format!(
                            "{}:{}: cell {id:?}\n    statement: {trimmed}\n    {e}",
                            path.display(),
                            lineno + 1
                        ));
                    }
                }
            }
        }
    }

    assert!(cells > 0, "corpus is empty — the path is probably wrong");
    assert!(
        failures.is_empty(),
        "{} of {cells} cells ({statements} statements) contain unparseable LQL:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// The specific regression: a bare `BEGIN PATCH` is rejected by the real
/// grammar, so if one ever returns to the corpus the test above catches it.
/// This pins the reason, so a future reader sees why rather than only that.
#[test]
fn bare_begin_patch_is_rejected_by_the_grammar() {
    assert!(
        larql_lql::parse("BEGIN PATCH;").is_err(),
        "the grammar accepted a pathless BEGIN PATCH — parse_begin no longer \
         requires a path, and the corpus rule this test guards is obsolete"
    );
    assert!(
        larql_lql::parse("BEGIN PATCH \"/tmp/x.vlp\";").is_ok(),
        "the grammar rejected a named BEGIN PATCH"
    );
}
