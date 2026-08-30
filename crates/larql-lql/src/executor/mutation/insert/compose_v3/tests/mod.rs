//! REFINE-IDEMPOTENCE — the invariant that fixed V2's compound-drift
//! bug, frozen for the V3 port: the batch refine reads the session's
//! RAW captured residuals, never refined state. Re-refining from the
//! same raws is a fixed point, and later installs rebuild earlier
//! slots from the ORIGINAL captures.

use crate::executor::{Backend, Session};
use crate::parse;
use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, miniature_glimmer};

/// A word-level tokenizer WITH a whitespace pre-tokenizer, so the
/// canonical prompts of distinct facts tokenize to distinct sequences.
/// (The plain synthetic tokenizer has no pre-tokenizer: every prose
/// prompt collapses to one `[UNK]`, giving every fact an IDENTICAL
/// capture — refine then rightly annihilates them all, which tests the
/// fixture, not the invariant.)
fn word_tokenizer_json() -> String {
    let vocab = r#""[UNK]":0,"The":1,"of":2,"is":3,"a":4,"b":5,"c":6,"[5]":7,"[6]":8"#;
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"Whitespace\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{vocab}}},\"unk_token\":\"[UNK]\"}}}}"
    )
}

fn v3_session() -> (tempfile::TempDir, Session) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "refine-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        word_tokenizer_json(),
    )
    .unwrap();
    let mut session = Session::new();
    let use_stmt = format!(
        "USE \"{}\";",
        container.path().display().to_string().replace('\\', "\\\\")
    );
    session.execute(&parse(&use_stmt).unwrap()).unwrap();
    (container, session)
}

fn run(session: &mut Session, stmt: &str) {
    session
        .execute(&parse(stmt).unwrap())
        .unwrap_or_else(|e| panic!("{stmt}: {e}"));
}

fn overlay_gate(session: &Session, layer: usize, feature: usize) -> Vec<f32> {
    let Backend::Vindex3 { overlay, .. } = &session.backend else {
        panic!("V3 session");
    };
    overlay
        .gate_override_at(layer, feature)
        .expect("composed slot has a gate")
        .to_vec()
}

fn composed_slots(session: &Session) -> Vec<(usize, usize)> {
    let mut slots: Vec<(usize, usize)> = session.raw_install_residuals.keys().copied().collect();
    slots.sort_unstable();
    slots
}

/// Re-refining from the same raw captures is a fixed point: the
/// orchestration feeds `refine_gates` the raw snapshot, so a second
/// pass reproduces the stored gates bit for bit.
#[test]
fn re_refining_from_the_same_raws_is_a_fixed_point() {
    let (_c, mut session) = v3_session();
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#,
    );
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#,
    );
    let slots = composed_slots(&session);
    assert_eq!(slots.len(), 2, "two installs at the layer");
    let before: Vec<Vec<f32>> = slots
        .iter()
        .map(|&(l, f)| overlay_gate(&session, l, f))
        .collect();

    // A second refine over the same session state must change nothing.
    // (g_ref/u_ref only rescale the unit direction; reusing the norms
    // implied by the stored vectors keeps this exact.)
    let g_ref = norm(&before[0]) / crate::executor::tuning::GATE_SCALE;
    let up0 = {
        let Backend::Vindex3 { overlay, .. } = &session.backend else {
            panic!("V3 session")
        };
        overlay
            .up_override_at(slots[0].0, slots[0].1)
            .unwrap()
            .to_vec()
    };
    let u_ref = norm(&up0);
    session.refine_layer_from_raw_v3(1, g_ref, u_ref);

    for (i, &(l, f)) in slots.iter().enumerate() {
        assert_eq!(
            overlay_gate(&session, l, f),
            before[i],
            "refine from the same raws must be a fixed point at ({l},{f})"
        );
    }
}

/// Later installs rebuild earlier slots from the ORIGINAL raw
/// captures: after B lands, A's stored gate equals a fresh
/// `refine_gates` run over the raw snapshot — not a re-projection of
/// A's already-refined gate.
///
/// The decoy cache is pre-seeded SMALL: the miniature's hidden size is
/// 12, and the production decoy set (~20 vectors) spans the whole
/// space — Gram-Schmidt then annihilates every gate and the surviving
/// direction is rounding junk, which is a fixture artifact, not the
/// invariant under test. Three suppressors keep the space
/// well-conditioned so directions are meaningful.
#[test]
fn later_installs_refine_from_original_captures() {
    let (_c, mut session) = v3_session();
    let seeded: Vec<larql_vindex::ndarray::Array1<f32>> = (0..2)
        .map(|i| {
            let mut v = vec![0.0f32; 12];
            v[i] = 1.0;
            larql_vindex::ndarray::Array1::from_vec(v)
        })
        .collect();
    session.decoy_residual_cache.insert(1, seeded);
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "[5]") AT LAYER 1 MODE COMPOSE;"#,
    );
    let raw_a = session.raw_install_residuals.clone();
    run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("c", "b", "[6]") AT LAYER 1 MODE COMPOSE;"#,
    );

    // The raw snapshot still holds A's ORIGINAL capture, untouched.
    for (key, original) in &raw_a {
        assert_eq!(
            session.raw_install_residuals.get(key).unwrap(),
            original,
            "raw captures are immutable inputs"
        );
    }

    // A's stored gate direction is the pure recomputation from raws +
    // decoys — proving refine consumed raw captures.
    let slots = composed_slots(&session);
    let inputs: Vec<larql_vindex::RefineInput> = slots
        .iter()
        .map(|&(l, f)| larql_vindex::RefineInput {
            layer: l,
            feature: f,
            gate: session.raw_install_residuals[&(l, f)].clone(),
        })
        .collect();
    let decoys = session
        .decoy_residual_cache
        .get(&1)
        .cloned()
        .unwrap_or_default();
    let expected = larql_vindex::refine_gates(&inputs, &decoys);
    for refined in expected.gates {
        let stored = overlay_gate(&session, refined.layer, refined.feature);
        let stored_dir = unit(&stored);
        let expected_dir = unit(refined.gate.as_slice().unwrap());
        let cos: f32 = stored_dir
            .iter()
            .zip(&expected_dir)
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            cos > 0.999_999,
            "stored gate at ({},{}) must be the raw recomputation (cos {cos}, \
             stored norm {}, expected norm {}, retained {})",
            refined.layer,
            refined.feature,
            norm(&stored),
            norm(refined.gate.as_slice().unwrap()),
            refined.retained_norm
        );
    }
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = norm(v).max(1e-12);
    v.iter().map(|x| x / n).collect()
}
