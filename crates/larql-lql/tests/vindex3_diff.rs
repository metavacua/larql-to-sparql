//! V3-LQL-3D gates: logical DIFF at the statement surface — the
//! COMPILE oracle in reverse:
//!
//! ```text
//! A = pristine       B = CURRENT (A + session overlay)
//! C = COMPILE(B)
//!
//! DIFF "A" CURRENT  ==  DIFF "A" "C"      (meaning, not storage)
//! DIFF "C" CURRENT  →  no semantic differences
//! DIFF "A" "C" PHYSICAL  →  the rewrite is visible down here
//! ```

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
        "diff-fixture",
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

/// The header names the sides; everything after it is the report.
fn report_body(lines: Vec<String>) -> Vec<String> {
    lines.into_iter().skip(1).collect()
}

#[test]
fn diff_sees_meaning_not_storage() {
    let container = v3_container();
    let a = lql_path(container.path());
    let out_dir = tempfile::tempdir().unwrap();
    let c_path = out_dir.path().join("compiled.v3");
    let c = lql_path(&c_path);

    // B: the session — A plus a KNN edge and a compose install.
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{a}\";"));
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#,
    );
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "d", "[6]") AT LAYER 1 ALPHA 5.0 MODE COMPOSE;"#,
    );

    // Self-diff first: CURRENT is equivalent to itself.
    let self_diff = run(&mut session, "DIFF CURRENT CURRENT;").join("\n");
    assert!(self_diff.contains("no semantic differences"), "{self_diff}");

    let ab = report_body(run(&mut session, &format!("DIFF \"{a}\" CURRENT;")));
    let rendered = ab.join("\n");
    assert!(rendered.contains("KNOWLEDGE"), "{rendered}");
    assert!(rendered.contains("+ a —[b]→ [5]"), "{rendered}");
    assert!(rendered.contains("FEATURES"), "{rendered}");
    assert!(
        rendered.contains("gate_row, up_row, down_col changed"),
        "{rendered}"
    );
    assert!(rendered.contains("SUMMARY"), "{rendered}");
    assert!(rendered.contains("knowledge +1/−0"), "{rendered}");

    // C: the clean bake of the same state.
    run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{c}\";"),
    );

    // THE gate: the overlay and its bake diff identically against A…
    let ac = report_body(run(&mut session, &format!("DIFF \"{a}\" \"{c}\";")));
    assert_eq!(ab, ac, "the diff must see meaning, not storage");

    // …and diff each other as equivalent, while the physical layer
    // reports the rewrite.
    let bc = run(&mut session, &format!("DIFF \"{c}\" CURRENT;")).join("\n");
    assert!(bc.contains("no semantic differences"), "{bc}");
    let phys = run(&mut session, &format!("DIFF \"{a}\" \"{c}\" PHYSICAL;")).join("\n");
    assert!(phys.contains("segment"), "{phys}");
    assert!(phys.contains("→"), "{phys}");

    // A against itself is physically identical.
    let phys_same = run(&mut session, &format!("DIFF \"{a}\" \"{a}\" PHYSICAL;")).join("\n");
    assert!(phys_same.contains("identical segments"), "{phys_same}");
}

/// Filters shape the report without changing what it sees.
#[test]
fn diff_filters_apply_at_render_time() {
    let container = v3_container();
    let a = lql_path(container.path());
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{a}\";"));
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "d", "[6]") AT LAYER 1 ALPHA 5.0 MODE COMPOSE;"#,
    );

    // The slot lives at layer 1 — a LAYER 0 filter hides it, the
    // summary still counts it.
    let filtered = run(&mut session, &format!("DIFF \"{a}\" CURRENT LAYER 0;")).join("\n");
    assert!(!filtered.contains("gate_row"), "{filtered}");
    assert!(filtered.contains("feature slots changed: 1"), "{filtered}");

    // INTO PATCH is a later rung on V3.
    let stmt = format!("DIFF \"{a}\" CURRENT INTO PATCH \"x.vlp\";");
    let err = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("INTO PATCH refuses on V3");
    assert!(
        err.to_string()
            .contains("not supported on a VINDEX3 container"),
        "{err}"
    );
}

/// Direction and filters: removed edges render with −, RELATION
/// filters the knowledge section, LIMIT truncates the slot listing,
/// and diffing genuinely different models reports metadata.
#[test]
fn diff_renders_direction_metadata_and_limits() {
    let container = v3_container();
    let a = lql_path(container.path());
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{a}\";"));
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#,
    );
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "d", "[6]") AT LAYER 1 ALPHA 5.0 MODE COMPOSE;"#,
    );
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("e", "d", "[7]") AT LAYER 1 ALPHA 5.0 MODE COMPOSE;"#,
    );

    // Reversed sides: CURRENT's knowledge is REMOVED relative to A.
    let reversed = run(&mut session, &format!("DIFF CURRENT \"{a}\";")).join("\n");
    assert!(reversed.contains("- a —[b]→ [5]"), "{reversed}");

    // RELATION filters the knowledge section.
    let filtered = run(
        &mut session,
        &format!("DIFF \"{a}\" CURRENT RELATION \"nope\";"),
    )
    .join("\n");
    assert!(!filtered.contains("—[b]→"), "{filtered}");

    // LIMIT truncates the slot listing but the summary stays whole.
    let limited = run(&mut session, &format!("DIFF \"{a}\" CURRENT LIMIT 1;")).join("\n");
    assert!(limited.contains("… 1 more (LIMIT 1)"), "{limited}");
    assert!(limited.contains("feature slots changed: 2"), "{limited}");

    // A genuinely different model reports metadata, not a slot table.
    let other_checkpoint = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    encode_fixture_container(
        larql_vindex::format::vindex3::fixtures::dense_f32_model,
        other_checkpoint.path(),
        other.path(),
        "other-model",
    );
    std::fs::write(
        other.path().join("tokenizer.json"),
        spaced_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    let meta = run(
        &mut session,
        &format!("DIFF \"{a}\" \"{}\";", lql_path(other.path())),
    )
    .join("\n");
    assert!(meta.contains("METADATA"), "{meta}");
    assert!(meta.contains("model: diff-fixture → other-model"), "{meta}");
}
