//! Migration rung M2: `EXTRACT ... FORMAT VINDEX3` produces a container.
//!
//! The oracle is the same one the whole V3 arc has used — a round trip
//! through the statement surface, compared against the path it must be
//! equivalent to:
//!
//! ```text
//! EXTRACT FORMAT VINDEX3  → container detects as V3
//!                         → session is bound, no USE needed
//!                         → INFER through the auto-bind
//!                              == INFER through a fresh USE of the same
//!                                 container, line for line
//! ```
//!
//! The auto-bind equality is the point: `EXTRACT` must not grow a second
//! binding path that drifts from `USE`'s.

use std::path::Path;

use larql_inference::test_utils::synthetic_tokenizer_json;
use larql_lql::{parse, Session};
use larql_vindex::format::generation::{detect_generation, ContainerGeneration};
use larql_vindex::format::vindex3::fixtures::{miniature_glimmer, G_VOCAB};

/// Windows temp paths contain backslashes, which the LQL lexer's escape
/// pass would consume; doubling them leaves the path untouched on every
/// platform.
fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

fn try_run(session: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).map_err(|e| format!("parse: {e}"))?;
    session
        .execute(&parsed)
        .map_err(|e| format!("execute: {e}"))
}

/// An HF-checkpoint-shaped directory: config.json + safetensors, plus
/// the tokenizer the capability snapshot must carry into the container.
fn checkpoint_with_tokenizer() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    std::fs::write(
        dir.path().join("tokenizer.json"),
        synthetic_tokenizer_json(G_VOCAB),
    )
    .unwrap();
    dir
}

/// INFER's prediction rows, without the elapsed-time line: the claim
/// under test is what the program predicts, not how long it took.
fn predictions(lines: Vec<String>) -> Vec<String> {
    lines.into_iter().filter(|l| l.contains("[id ")).collect()
}

fn extract_v3(session: &mut Session, checkpoint: &Path, out: &Path) -> Vec<String> {
    run(
        session,
        &format!(
            "EXTRACT MODEL \"{}\" INTO \"{}\" FORMAT VINDEX3;",
            lql_path(checkpoint),
            lql_path(out)
        ),
    )
}

/// THE gate: the statement produces a V3 container, carries the
/// checkpoint's tokenizer into it, leaves the session bound, and that
/// auto-bind behaves identically to a fresh `USE` of the same container.
#[test]
fn extract_format_vindex3_encodes_binds_and_matches_use() {
    let checkpoint = checkpoint_with_tokenizer();
    let out = tempfile::tempdir().unwrap();
    let out_dir = out.path().join("container");

    let mut session = Session::new();
    let report = extract_v3(&mut session, checkpoint.path(), &out_dir).join("\n");

    // The container is VINDEX3 by its own marker — not by filename.
    assert_eq!(
        detect_generation(&out_dir).unwrap(),
        ContainerGeneration::V3,
        "{report}"
    );
    // The capability snapshot ran: the tokenizer is beside the segments,
    // so the container binds servable rather than token-id-only.
    assert!(report.contains("Capabilities: tokenizer.json"), "{report}");
    assert!(out_dir.join("tokenizer.json").exists());
    assert!(report.contains("VINDEX3"), "{report}");

    // The session is already bound — a statement runs with no USE.
    let auto_bound = predictions(run(&mut session, r#"INFER "[3]" TOP 3;"#));

    // …and that binding is the one USE produces, row for row.
    let mut fresh = Session::new();
    run(&mut fresh, &format!("USE \"{}\";", lql_path(&out_dir)));
    let via_use = predictions(run(&mut fresh, r#"INFER "[3]" TOP 3;"#));
    assert!(!auto_bound.is_empty(), "INFER must return prediction rows");
    assert_eq!(
        auto_bound, via_use,
        "EXTRACT's auto-bind must be USE's binding, not a second path"
    );
}

/// A checkpoint with no tokenizer still encodes — absence narrows
/// capability, it is not an error — and the report says so rather than
/// leaving the caller to discover it at INFER time.
#[test]
fn extract_format_vindex3_reports_a_tokenizerless_checkpoint() {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let out = tempfile::tempdir().unwrap();
    let out_dir = out.path().join("container");

    let mut session = Session::new();
    let report = extract_v3(&mut session, checkpoint.path(), &out_dir).join("\n");
    assert_eq!(
        detect_generation(&out_dir).unwrap(),
        ContainerGeneration::V3
    );
    assert!(report.contains("token-id capability only"), "{report}");
    assert!(!out_dir.join("tokenizer.json").exists());
}

/// A source the V3 encoder cannot consume refuses BY NAME, naming the
/// escape hatch — never a silent downgrade to a V2 extraction.
#[test]
fn extract_format_vindex3_refuses_a_gguf_source_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let gguf = dir.path().join("model.gguf");
    std::fs::write(&gguf, b"GGUF").unwrap();
    let out = tempfile::tempdir().unwrap();

    let mut session = Session::new();
    let err = try_run(
        &mut session,
        &format!(
            "EXTRACT MODEL \"{}\" INTO \"{}\" FORMAT VINDEX3;",
            lql_path(&gguf),
            lql_path(out.path().join("container"))
        ),
    )
    .expect_err("GGUF is not an HF checkpoint");
    assert!(err.contains("GGUF"), "{err}");
    assert!(err.contains("FORMAT VINDEX2"), "{err}");
}

/// A directory that is not an HF checkpoint refuses through the
/// statement surface, naming what the encoder consumes — the encode
/// pipeline's refusals reach the caller rather than being flattened
/// into "extraction failed".
///
/// (The plan gate's itemised blocking findings are gated where they are
/// rendered, in larql-vindex's `encode::tests`; the drafter-shaped
/// inventory fixture that produces an inadmissible plan is internal to
/// that crate.)
#[test]
fn extract_format_vindex3_refuses_a_non_checkpoint_directory() {
    let not_a_checkpoint = tempfile::tempdir().unwrap();
    std::fs::write(not_a_checkpoint.path().join("readme.txt"), b"nothing").unwrap();
    let out = tempfile::tempdir().unwrap();

    let mut session = Session::new();
    let err = try_run(
        &mut session,
        &format!(
            "EXTRACT MODEL \"{}\" INTO \"{}\" FORMAT VINDEX3;",
            lql_path(not_a_checkpoint.path()),
            lql_path(out.path().join("container"))
        ),
    )
    .expect_err("a directory with no config.json is not a checkpoint");
    assert!(err.contains("config.json + safetensors"), "{err}");
}

// ── M3 consumer readiness: no V3 artifact silently disappears ──

/// `SHOW MODELS` lists containers of BOTH generations.
///
/// It previously listed only directories whose `index.json` parsed as a
/// VINDEX2 config, so a V3 container in the working directory vanished
/// from the listing entirely — the user saw nothing beside a model they
/// had just extracted. Invisibility is not an allowed consumer state:
/// a container is understood, or explicitly accounted for.
#[test]
fn show_models_lists_both_generations_and_never_hides_one() {
    let dir = tempfile::tempdir().unwrap();

    // A real V3 container, produced the way M2 produces them.
    let checkpoint = checkpoint_with_tokenizer();
    let mut session = Session::new();
    extract_v3(
        &mut session,
        checkpoint.path(),
        &dir.path().join("v3-model"),
    );

    // A V2 container beside it (index.json is the listing's whole input).
    std::fs::create_dir(dir.path().join("v2-model")).unwrap();
    std::fs::write(
        dir.path().join("v2-model").join("index.json"),
        r#"{"version":2,"model":"legacy","num_layers":34}"#,
    )
    .unwrap();

    // And a directory holding an index.json this binary cannot identify.
    std::fs::create_dir(dir.path().join("mystery")).unwrap();
    std::fs::write(dir.path().join("mystery").join("index.json"), "{}").unwrap();

    let listing = larql_lql::Session::show_models_in(dir.path())
        .expect("listing")
        .join("\n");

    assert!(
        listing.contains("v3-model"),
        "V3 must be listed:\n{listing}"
    );
    assert!(
        listing.contains("v2-model"),
        "V2 must be listed:\n{listing}"
    );
    assert!(
        listing.contains("mystery"),
        "an unidentifiable container is accounted for, not dropped:\n{listing}"
    );
    // The generation is named, so V2 is not the implicit normal case.
    assert!(listing.contains("v3"), "{listing}");
    assert!(listing.contains("v2"), "{listing}");
    assert!(listing.contains("unreadable"), "{listing}");
}
