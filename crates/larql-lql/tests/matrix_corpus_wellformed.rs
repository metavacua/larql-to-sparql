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
fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/lql_matrix")
}

/// Every shipped LQL corpus, DISCOVERED rather than named.
///
/// The first version of this test hardcoded `commands.jsonl`, so when the
/// statements-as-a-list migration landed it silently skipped
/// `commands-model.jsonl` — which then failed on a runner, in the one job the
/// local suite could not have caught. A test that lists its own inputs by hand
/// only covers the inputs its author remembered.
fn every_lql_corpus() -> Vec<PathBuf> {
    let dir = corpus_dir();
    let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            // `commands*.jsonl` are the LQL corpora. Task 6/7 add cli-*.jsonl,
            // which carry `argv` rather than `lql` and are not parsed here.
            n.starts_with("commands") && n.ends_with(".jsonl")
        })
        .collect();
    found.sort();
    assert!(
        found.len() >= 2,
        "expected at least commands.jsonl and commands-model.jsonl in {}, found {:?}",
        dir.display(),
        found
    );
    found
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
    let mut failures: Vec<String> = Vec::new();
    let mut cells = 0usize;
    let mut statements = 0usize;

    for path in every_lql_corpus() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

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
            let entries: Vec<String> = row["lql"]
                .as_array()
                .unwrap_or_else(|| {
                    panic!(
                        "{}:{}: cell {id:?} has no `lql` ARRAY. Statements are \
                     authored data now — a string here is an unmigrated cell.",
                        path.display(),
                        lineno + 1
                    )
                })
                .iter()
                .map(|v| v.as_str().expect("statement is not a string").to_string())
                .collect();
            cells += 1;

            for stmt in &entries {
                statements += 1;
                if let Err(e) = larql_lql::parse(stmt) {
                    if !is_negative_cell(id) {
                        failures.push(format!(
                            "{}:{}: cell {id:?}\n    statement: {stmt}\n    {e}",
                            path.display(),
                            lineno + 1
                        ));
                    }
                }
            }

            // The one-shot `lql` driver hands the SPACE-JOINED cell to `run_batch`,
            // which splits it again with this same function. If that split ever
            // disagreed with the authored list, the three LQL drivers would stop
            // exercising identical statement sequences and the design's
            // driver-parity property would be silently gone. Pin it with the real
            // splitter rather than trusting that the migration was faithful.
            let resplit: Vec<String> = larql_lql::split_statements(&entries.join(" "))
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            assert_eq!(
                resplit,
                entries,
                "{}:{}: cell {id:?} does not survive join-then-split — the one-shot \
             driver would run a different statement sequence than the REPL drivers",
                path.display(),
                lineno + 1
            );
        }
    }

    assert!(cells > 0, "corpora are empty — the path is probably wrong");
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
