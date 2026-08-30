//! V3-LQL-3D gates: `COMPILE CURRENT INTO VINDEX` on a VINDEX3
//! binding — the bake oracle:
//!
//! ```text
//! overlaid original  ==  compiled clean container
//! ```
//!
//! across the meaningful surfaces (INFER TOP, GENERATE, TRACE, WALK,
//! SELECT, DESCRIBE's L0 knowledge), with the stronger check that the
//! compiled container binds with a ZERO-override overlay — behaviour
//! comes from the stored bytes, proving the compile is a genuine bake
//! and not "copy base + patch file". A negative control pins that the
//! compiled container actually differs from the pristine base.
//!
//! Annotation `c_score` is excluded by design: V3 annotations are
//! DERIVED from weights (`embed · feature_down`), so a compiled
//! container re-derives them from the baked columns; identity and
//! top-token agreement are asserted, the overlay's stored score
//! display is not. The same authority rule is why tombstones and
//! meta-only relabels REFUSE to compile (gated below) — they have no
//! physical form in a clean container.

use std::path::Path;

use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

/// Spaced-form vocabulary (see vindex3_mutation.rs): the V2 target
/// contract encodes `" {target}"`, which must be in-vocab or the
/// compose payload degrades to the UNK embedding.
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
        "compile-fixture",
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

fn bound_session(container: &Path) -> Session {
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{}\";", lql_path(container)));
    session
}

/// INFER output normalised to ranks + probs + ids (decode surfaces are
/// ambiguous under the spaced vocab — tokenizer cosmetics, not program
/// output).
fn infer_lines(session: &mut Session, prompt: &str) -> Vec<String> {
    run(session, &format!("INFER \"{prompt}\" TOP 5;"))
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

fn generate_ids(session: &mut Session, prompt: &str) -> String {
    run(session, &format!("INFER \"{prompt}\" GENERATE 8;"))
        .iter()
        .find_map(|l| l.trim_start().strip_prefix("ids:"))
        .expect("ids line")
        .trim()
        .to_string()
}

fn traced_next(session: &mut Session, prompt: &str) -> String {
    run(session, &format!("TRACE \"{prompt}\";"))
        .iter()
        .rev()
        .find(|l| l.starts_with("next token"))
        .expect("next-token line")
        .split_whitespace()
        .nth(2)
        .unwrap()
        .to_string()
}

fn walk_hits(session: &mut Session, prompt: &str) -> Vec<String> {
    run(session, &format!("WALK \"{prompt}\" TOP 5;"))
        .iter()
        .filter(|l| l.trim_start().starts_with('L'))
        .map(|l| {
            // `  L 0: F14  gate=…` — keep layer + feature, drop the
            // display tail.
            l.split_whitespace().take(3).collect::<Vec<_>>().join(" ")
        })
        .collect()
}

/// The bake oracle: overlaid original == compiled clean container.
#[test]
fn compile_bakes_the_overlay_into_a_clean_equivalent_container() {
    let container = v3_container();
    let out_dir = tempfile::tempdir().unwrap();
    let out = out_dir.path().join("compiled.v3");
    const PROMPT: &str = "The d of c is";

    let mut session = bound_session(container.path());
    let pristine_infer = infer_lines(&mut session, PROMPT);

    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#,
    );
    run(
        &mut session,
        // ALPHA 5000: the pristine-vs-compiled negative control below
        // observes the payload through INFER's 2-decimal display; on
        // the random-head LCG fixture a small alpha's change is below
        // the display quantum and platform rounding decides whether a
        // digit flips (the same hazard the mutation suite hit on
        // Windows CI).
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "d", "[6]") AT LAYER 1 ALPHA 5000.0 MODE COMPOSE;"#,
    );

    let overlaid_infer = infer_lines(&mut session, PROMPT);
    let overlaid_gen = generate_ids(&mut session, PROMPT);
    let overlaid_trace = traced_next(&mut session, PROMPT);
    let overlaid_walk = walk_hits(&mut session, "[3]");
    let overlaid_describe = run(&mut session, r#"DESCRIBE "a";"#).join("\n");
    assert!(overlaid_describe.contains("→ [5]"), "{overlaid_describe}");

    let compiled = run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&out)),
    )
    .join("\n");
    assert!(compiled.contains("clean container"), "{compiled}");
    assert!(compiled.contains("segments rewritten"), "{compiled}");

    // ── The compiled container, bound fresh ──
    let mut clean = bound_session(&out);

    // Zero active overlay: nothing recorded, nothing pending.
    let patches = run(&mut clean, "SHOW PATCHES;").join("\n");
    assert!(patches.contains("no patches applied"), "{patches}");

    // Execution surfaces: exact.
    assert_eq!(
        infer_lines(&mut clean, PROMPT),
        overlaid_infer,
        "INFER must match the overlaid original"
    );
    assert_eq!(
        generate_ids(&mut clean, PROMPT),
        overlaid_gen,
        "GENERATE must match"
    );
    assert_eq!(
        traced_next(&mut clean, PROMPT),
        overlaid_trace,
        "TRACE must match"
    );

    // Browse surfaces: the walk over baked gate rows, the derived
    // annotation of the composed slot, and the L0 knowledge.
    assert_eq!(
        walk_hits(&mut clean, "[3]"),
        overlaid_walk,
        "WALK must match"
    );
    let rows = run(
        &mut clean,
        "SELECT * FROM FEATURES WHERE layer = 1 LIMIT 300;",
    )
    .join("\n");
    assert!(
        rows.contains("[6]"),
        "the baked slot's derived annotation must promote the target: {rows}"
    );
    let describe = run(&mut clean, r#"DESCRIBE "a";"#).join("\n");
    assert!(describe.contains("→ [5]"), "{describe}");

    // Negative control: the compiled container is NOT the pristine
    // base wearing a new name.
    assert_ne!(
        infer_lines(&mut clean, PROMPT),
        pristine_infer,
        "compiled behaviour must differ from the pristine base"
    );
}

/// The authority refusals: overlay state without a physical form in a
/// clean container refuses to compile — never a silent drop.
#[test]
fn compile_refuses_unbakeable_annotation_state() {
    let container = v3_container();
    let out_dir = tempfile::tempdir().unwrap();

    // Tombstone.
    let mut session = bound_session(container.path());
    run(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    );
    let stmt = format!(
        "COMPILE CURRENT INTO VINDEX \"{}\";",
        lql_path(out_dir.path().join("t.v3"))
    );
    let err = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("a tombstone cannot bake");
    assert!(err.to_string().contains("tombstone at (0,0)"), "{err}");

    // Meta-only relabel.
    let mut session = bound_session(container.path());
    run(
        &mut session,
        r#"UPDATE EDGES SET target = "[9]" WHERE layer = 0 AND feature = 1;"#,
    );
    let err = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("a meta-only relabel cannot bake");
    assert!(
        err.to_string().contains("meta-only override at (0,1)"),
        "{err}"
    );

    // INTO MODEL is a later rung.
    let mut session = bound_session(container.path());
    let stmt = format!(
        "COMPILE CURRENT INTO MODEL \"{}\";",
        lql_path(out_dir.path().join("m"))
    );
    let err = session
        .execute(&parse(&stmt).unwrap())
        .expect_err("INTO MODEL refuses on V3");
    assert!(
        err.to_string()
            .contains("not supported on a VINDEX3 container"),
        "{err}"
    );
}

/// A compile with no overlay state is a faithful copy: every segment
/// hard-linked/copied, and the result binds and executes identically.
#[test]
fn compile_of_a_pristine_session_is_a_faithful_copy() {
    let container = v3_container();
    let out_dir = tempfile::tempdir().unwrap();
    let out = out_dir.path().join("copy.v3");

    let mut session = bound_session(container.path());
    let pristine = infer_lines(&mut session, "[3]");
    let compiled = run(
        &mut session,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", lql_path(&out)),
    )
    .join("\n");
    assert!(compiled.contains("0 tensors baked"), "{compiled}");

    let mut clean = bound_session(&out);
    assert_eq!(infer_lines(&mut clean, "[3]"), pristine);
}
