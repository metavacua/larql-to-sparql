//! V3-LQL-3D gates: `COMPACT INTO VINDEX` — semantics-preserving
//! physical reorganisation, PROVEN by the logical DIFF:
//!
//! ```text
//! DIFF input output          → no semantic differences
//! INFER / WALK before==after → runtime oracle agrees
//! report                     → names every dropped file
//! ```
//!
//! …and COMPACT is not a second compiler: overlay state refuses with
//! direction (COMPILE first).

use std::path::Path;

use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

fn spaced_tokenizer_json(vocab: usize) -> String {
    let mut entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    entries.extend((0..vocab).map(|i| format!("\" [{i}]\":{i}")));
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":null,\"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn v3_container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "compact-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        spaced_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    container
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

/// Normalised to rank + prob + id — the spaced vocab maps two
/// surfaces onto one id and WordLevel decode picks one untracked.
fn infer(session: &mut Session) -> Vec<String> {
    run(session, "INFER \"[3]\" TOP 5;")
        .into_iter()
        .filter(|l| !l.trim_end().ends_with("ms"))
        .map(|l| {
            let trimmed = l.trim_start();
            match (trimmed.split_once(". "), l.rfind('(')) {
                (Some((rank, _)), Some(paren)) if rank.chars().all(|c| c.is_ascii_digit()) => {
                    format!("{rank}. {}", &l[paren..])
                }
                _ => l,
            }
        })
        .collect()
}

#[test]
fn compact_preserves_semantics_and_names_what_it_drops() {
    let container = v3_container();
    std::fs::write(container.path().join("segments/orphan.bin"), b"junk").unwrap();
    std::fs::write(container.path().join("scratch.txt"), b"junk").unwrap();
    let a = lql_path(container.path());
    let out_dir = tempfile::tempdir().unwrap();
    let out = lql_path(out_dir.path().join("compacted.v3"));

    let mut session = Session::new();
    run(&mut session, &format!("USE \"{a}\";"));
    let before = infer(&mut session);

    let report = run(&mut session, &format!("COMPACT INTO VINDEX \"{out}\";")).join("\n");
    assert!(report.contains("2 unreferenced files dropped"), "{report}");
    assert!(report.contains("- scratch.txt"), "{report}");
    assert!(report.contains("- segments/orphan.bin"), "{report}");

    // DIFF is the proof instrument: no semantic differences, and the
    // segments themselves are byte-identical (this policy reorganises
    // the DIRECTORY — the physical change is the dropped files the
    // report names).
    let logical = run(&mut session, &format!("DIFF \"{a}\" \"{out}\";")).join("\n");
    assert!(logical.contains("no semantic differences"), "{logical}");
    let physical = run(&mut session, &format!("DIFF \"{a}\" \"{out}\" PHYSICAL;")).join("\n");
    assert!(physical.contains("identical segments"), "{physical}");

    // Runtime oracle: the compacted container executes identically.
    let mut clean = Session::new();
    run(&mut clean, &format!("USE \"{out}\";"));
    assert_eq!(infer(&mut clean), before);
}

/// The lifecycle composes: mutate → COMPILE (materialise meaning) →
/// COMPACT (reorganise storage) — each step proven by DIFF.
#[test]
fn compile_then_compact_composes_with_diff_as_the_proof() {
    let container = v3_container();
    let a = lql_path(container.path());
    let dirs = tempfile::tempdir().unwrap();
    let compiled = lql_path(dirs.path().join("compiled.v3"));
    let compacted = lql_path(dirs.path().join("compacted.v3"));

    let mut session = Session::new();
    run(&mut session, &format!("USE \"{a}\";"));
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "d", "[6]") AT LAYER 1 ALPHA 5.0 MODE COMPOSE;"#,
    );

    // COMPACT refuses while the overlay holds the edit.
    let stmt = format!("COMPACT INTO VINDEX \"{compacted}\";");
    let err = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("overlay state must refuse");
    assert!(
        err.to_string()
            .contains("COMPILE CURRENT INTO VINDEX first"),
        "{err}"
    );

    // COMPILE materialises the meaning; COMPACT then reorganises it.
    run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{compiled}\";"),
    );
    let mut compiled_session = Session::new();
    run(&mut compiled_session, &format!("USE \"{compiled}\";"));
    run(
        &mut compiled_session,
        &format!("COMPACT INTO VINDEX \"{compacted}\";"),
    );

    // The chain end-to-end: compacted == the session's composed state.
    let end = run(
        &mut compiled_session,
        &format!("DIFF \"{compiled}\" \"{compacted}\";"),
    )
    .join("\n");
    assert!(end.contains("no semantic differences"), "{end}");
    let vs_base = run(
        &mut compiled_session,
        &format!("DIFF \"{a}\" \"{compacted}\";"),
    )
    .join("\n");
    assert!(
        vs_base.contains("gate_row, up_row, down_col changed"),
        "{vs_base}"
    );
}
