//! LQL over a VINDEX3 container — the control plane end to end.
//!
//! Encodes the miniature judged-semantics checkpoint into a real
//! VINDEX3 container (with a tokenizer), then drives an actual LQL
//! session through every statement the V3 binding serves — plus the
//! refusals, because on a V3 binding a refusal is a capability
//! statement, not a failure:
//!
//! ```text
//! USE          bind once, on the container's own generation marker
//! STATS        the container's own authority
//! SHOW LAYERS  per-layer facts off the executable plan
//! INFER        top-k from batch-prefill logits
//! INFER … GENERATE   greedy continuation through the runtime seam
//! EXPLAIN INFER      the executable plan, statically
//! TRACE        observe the canonical executor while it runs
//! WALK / SELECT / DESCRIBE   browse via semantic roles (V3-LQL-3A)
//! ```
//!
//! Run: cargo run -p larql-demos --example vindex3_demo
//!
//! Pass a container path to drive a REAL VINDEX3 container instead of
//! the self-encoded miniature:
//!
//! ```sh
//! cargo run -p larql-demos --example vindex3_demo -- path/to/model.vindex3
//! ```
//!
//! On a real container the demo runs the static authority surface
//! (USE / STATS / SHOW LAYERS / EXPLAIN INFER) plus whatever the
//! container's capabilities allow: a container without tokenizer.json
//! binds and explains, and the text verbs refuse naming the missing
//! capability — that refusal is the capability model working, not a
//! failure.

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

fn run(session: &mut Session, stmt: &str) {
    println!("larql> {stmt}");
    match parse(stmt) {
        Ok(parsed) => match session.execute(&parsed) {
            Ok(lines) => {
                for line in lines {
                    println!("{line}");
                }
            }
            Err(e) => println!("Error: {e}"),
        },
        Err(e) => println!("Parse error: {e}"),
    }
    println!();
}

fn main() {
    println!("=== LQL x VINDEX3 Demo ===\n");

    // Real-container mode: `-- <path>` binds that container and runs
    // the static authority surface against it.
    if let Some(container) = std::env::args().nth(1) {
        let mut session = Session::new();
        run(&mut session, &format!("USE \"{container}\";"));
        run(&mut session, "STATS;");
        run(&mut session, "SHOW LAYERS;");
        run(&mut session, r#"EXPLAIN INFER "x";"#);
        // Text inference/browse need the tokenizer capability; on a
        // container without one this prints the capability refusal.
        run(&mut session, r#"INFER "x" TOP 3;"#);
        println!("=== Done (real container) ===");
        return;
    }

    // A real container: the miniature Glimmer anatomy (sliding+full
    // attention split, four-norm placement) encoded into VINDEX3, plus
    // a tokenizer so the text statements work. This is the same
    // fixture the executor's parity gates certify.
    let checkpoint = tempfile::tempdir().expect("tempdir");
    let container = tempfile::tempdir().expect("tempdir");
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "demo-glimmer",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .expect("write tokenizer");

    let mut session = Session::new();

    // ── Bind once; everything after consumes declared facts ──
    // (backslashes doubled so the LQL lexer's escape pass keeps
    // Windows paths intact)
    let container_path = container.path().display().to_string().replace('\\', "\\\\");
    run(&mut session, &format!("USE \"{container_path}\";"));

    // ── The container's own authority ──
    run(&mut session, "STATS;");
    run(&mut session, "SHOW LAYERS;");

    // ── Inference through the proven runtime seam ──
    run(&mut session, r#"INFER "[3]" TOP 5;"#);
    run(&mut session, r#"INFER "[3]" GENERATE 16;"#);

    // ── Explain the program that will run; observe it running ──
    run(&mut session, r#"EXPLAIN INFER "[3]";"#);
    run(&mut session, r#"TRACE "[3]";"#);

    // ── Browse: the model as a database, via semantic roles ──
    run(&mut session, r#"WALK "[3]" TOP 3;"#);
    run(&mut session, r#"EXPLAIN WALK "[3]";"#);
    run(
        &mut session,
        r#"SELECT * FROM FEATURES WHERE layer = 0 LIMIT 5;"#,
    );
    run(&mut session, r#"SELECT * FROM EDGES LIMIT 5;"#);
    run(&mut session, r#"SELECT * FROM ENTITIES LIMIT 5;"#);
    run(
        &mut session,
        r#"SELECT * FROM EDGES NEAREST TO "[3]" AT LAYER 0 LIMIT 3;"#,
    );
    run(&mut session, r#"SHOW FEATURES 0 LIMIT 5;"#);
    run(&mut session, r#"SHOW ENTITIES LIMIT 5;"#);
    run(&mut session, r#"SHOW RELATIONS;"#);
    run(&mut session, r#"DESCRIBE "[3]";"#);

    // ── Mutation (V3-LQL-3B): the default KNN insert executes and is
    // immediately observable — the container stays untouched on disk ──
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]");"#,
    );
    run(&mut session, r#"DESCRIBE "a";"#);
    run(&mut session, r#"INFER "The b of a is" TOP 3;"#);
    run(&mut session, "SHOW PATCHES;");

    // ── Feature-slot mutation: overlay meta overrides + tombstones,
    // V2's contract — the container stays untouched on disk ──
    run(
        &mut session,
        r#"UPDATE EDGES SET target = "[9]" WHERE layer = 1 AND feature = 1;"#,
    );
    run(
        &mut session,
        "DELETE FROM EDGES WHERE layer = 0 AND feature = 0;",
    );
    run(
        &mut session,
        r#"SELECT * FROM FEATURES WHERE layer = 1 LIMIT 5;"#,
    );

    // ── Compose install: the FFN slot lands in the overlay and
    // execution observes it through the operand-source seam ──
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("x", "y", "[7]") MODE COMPOSE;"#,
    );
    run(&mut session, r#"INFER "The y of x is" TOP 3;"#);

    // ── Refusals are capability statements (lifecycle is next) ──
    run(&mut session, "COMPACT MINOR;");

    println!("=== Done ===");
}
