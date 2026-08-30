//! The `--profile` ledger renders what it was given: per-stage ms, the
//! byte floor per class, the sampling distortion line, and a loud
//! overflow warning.

use std::collections::BTreeMap;

use larql_compute_metal::lowering::profile::{Stage, StageProfile};

use super::super::profile::{StageBytes, StageLedger};

fn token(ms: &[(Stage, f64)], span_ms: f64, overflowed: usize) -> StageProfile {
    StageProfile {
        stage_ns: ms.iter().map(|(s, v)| (*s, (v * 1e6) as u64)).collect(),
        stage_runs: ms.iter().map(|(s, _)| (*s, 1u32)).collect(),
        span_ns: (span_ms * 1e6) as u64,
        overflowed,
    }
}

#[test]
fn empty_ledger_says_so() {
    let ledger = StageLedger::default();
    assert_eq!(
        ledger.render(),
        vec!["profile: no tokens recorded".to_string()]
    );
}

#[test]
fn render_prices_byte_classes_and_averages_over_tokens() {
    let mut ledger = StageLedger {
        bytes: StageBytes {
            attn_proj: 367_000_000, // exactly 1.00 ms at the 367 GB/s ceiling
            attn_out: 0,
            dense_ffn: 0,
            experts: 734_000_000, // 2.00 ms floor
            head: 0,
        },
        ..Default::default()
    };
    // Two tokens: attn.proj 2 ms, experts 4 ms each; span 7 ms; GPU 7.5.
    let t = token(&[(Stage::AttnProj, 2.0), (Stage::Experts, 4.0)], 7.0, 0);
    ledger.record(&t, 7.5);
    ledger.record(&t, 7.5);
    let lines = ledger.render();
    let text = lines.join("\n");
    assert!(text.contains("over 2 token(s)"), "{text}");
    // attn.proj: 2.000 ms, 367 MB, 184 GB/s (2× the floor), floor 1.00.
    let proj = lines
        .iter()
        .find(|l| l.contains("attn.proj"))
        .expect("attn.proj row");
    assert!(proj.contains("2.000"), "{proj}");
    assert!(proj.contains("367.0"), "{proj}");
    assert!(proj.contains("184"), "{proj}");
    assert!(proj.contains("1.00"), "{proj}");
    // experts: 4 ms over a 2 ms floor.
    let experts = lines
        .iter()
        .find(|l| l.contains("ffn.experts"))
        .expect("experts row");
    assert!(
        experts.contains("4.000") && experts.contains("2.00"),
        "{experts}"
    );
    // A stage with no byte class prints dashes.
    let mut no_bytes = StageLedger::default();
    no_bytes.record(&token(&[(Stage::AttnNorm, 0.5)], 0.5, 0), 0.5);
    let norm = no_bytes
        .render()
        .into_iter()
        .find(|l| l.contains("attn.norm"))
        .expect("norm row");
    assert!(norm.contains(" - "), "{norm}");
    // Totals: attributed 6 ms, floor 3 ms, residual 3 ms; gaps 1 ms.
    assert!(text.contains("attributed       6.000"), "{text}");
    assert!(text.contains("residual over byte floor: 3.000"), "{text}");
    assert!(text.contains("gaps between stages 1.000"), "{text}");
    assert!(text.contains("command-buffer GPU span 7.500"), "{text}");
    assert!(!text.contains("WARNING"), "{text}");
}

#[test]
fn render_warns_on_overflow() {
    let mut ledger = StageLedger::default();
    ledger.record(&token(&[(Stage::Head, 1.0)], 1.0, 3), 1.0);
    let text = ledger.render().join("\n");
    assert!(text.contains("WARNING: 3 stage run(s)"), "{text}");
}

#[test]
fn stage_bytes_total_and_class_mapping() {
    let b = StageBytes {
        attn_proj: 1,
        attn_out: 2,
        dense_ffn: 4,
        experts: 8,
        head: 16,
    };
    assert_eq!(b.total(), 31);
    let _ = BTreeMap::<Stage, u64>::new();
}
